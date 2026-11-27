"""With third-party research discontinued (owner 2026-11-27) the analyses read our own posts, and nothing analyses an
empty table as if it said something: recipes load own posts with their first-party outcome, gaps say why they filed
nothing, a question with no posts ends unanswerable without running tests, and `mw verify` flags any analysis
recorded over zero posts."""

from __future__ import annotations

import math

from psycopg.types.json import Jsonb

from manyworlds import policy as policy_mod, verify
from manyworlds.brain import inquiry, recipes, research_questions as rq
from manyworlds.research import annotation as ann


def _off(policy):
    """The test policy with third-party research off, as shipped."""
    intel = policy.body["intelligence"]
    return policy_mod.from_dict({**policy.body, "intelligence": {
        **intel, "external": {**intel["external"], "enabled": False},
        "questions": {**intel["questions"], "gaps": {**(intel["questions"].get("gaps") or {}), "enabled": True}}}})


def _own_post(conn, policy, creator, *, coded, normalized=None):
    post = conn.execute("INSERT INTO posts (creator_id, format, status, published_at) VALUES (%s, 'photo', "
                        "'published', now()) RETURNING id", (creator,)).fetchone()["id"]
    asset = conn.execute("INSERT INTO assets (creator_id, kind, uri, sha256, bytes) VALUES (%s, 'image', %s, 'h', 1) "
                         "RETURNING id", (creator, f"s3://b/{post}.jpg")).fetchone()["id"]
    conn.execute("INSERT INTO reference_annotations (source, own_post_id, own_asset_id, schema, observed) "
                 "VALUES ('own', %s, %s, %s, %s)",
                 (post, asset, ann.ANNOTATION_SCHEMA,
                  Jsonb({"coded": coded, "vocab_versions": ann.vocab_versions(policy)})))
    if normalized is not None:
        unit = conn.execute("INSERT INTO content_units (source, level, ref_table, ref_id, platform) VALUES "
                            "('own', 'post', 'posts', %s, 'own') RETURNING id", (str(post),)).fetchone()["id"]
        conn.execute("INSERT INTO outcomes (unit_id, metric, horizon, raw, normalized) VALUES (%s, 'engagement', "
                     "'d7', 10, %s)", (unit, normalized))
    return str(post)


def test_recipes_read_our_own_posts_and_their_first_party_outcome_while_external_is_off(conn, policy):
    off = _off(policy)
    assert recipes.source(off) == "own" and recipes.source(policy) == "external"
    me = conn.execute("INSERT INTO creators (handle, publish_mode) VALUES ('fp1', 'instagram') RETURNING id"
                      ).fetchone()["id"]
    with_y = _own_post(conn, off, me, coded={"candid_level": "candid"}, normalized=0.4)
    _own_post(conn, off, me, coded={"candid_level": "posed"})                  # not published long enough: no outcome
    rows = {r.post_id: r for r in recipes.load(conn, off)}
    assert len(rows) == 2
    assert math.isclose(rows[with_y].y, 0.4) and rows[with_y].context["platform"] == "own"
    assert sum(r.y is None for r in rows.values()) == 1                     # excluded and counted, never zero
    assert recipes.load(conn, policy) == []                                 # the external tables stay empty


def test_gaps_say_why_they_filed_nothing(conn, policy):
    off = _off(policy)
    out = rq.file_gaps(conn, off)
    assert out["filed"] == [] and out["posts"] == 0 and out["source"] == "own"
    assert "own posts with an outcome" in out["skipped"]


def test_a_question_with_no_posts_is_unanswerable_and_runs_no_analysis(conn, policy):
    conn.execute("DELETE FROM research_questions")
    off = _off(policy)
    qid = rq.ask(conn, off, "showrunner", "Does candid beat posed?",
                 spec={"tests": [{"kind": "main", "attrs": ["candid_level=candid"]}],
                       "explore": [{"method": "contrast_sources"}]})
    rq.run_due(conn, off, None)
    row = conn.execute("SELECT status, result FROM research_questions WHERE id = %s", (qid,)).fetchone()
    assert row["status"] == "unanswerable" and row["result"]["data"]["no_data"] == "no own posts match"
    assert conn.execute("SELECT count(*) n FROM analysis_runs WHERE question_id = %s", (qid,)).fetchone()["n"] == 0
    assert verify.run(conn, off)["checks"]["empty_analyses"]["count"] == 0


def test_verify_flags_an_analysis_recorded_over_zero_posts(conn, policy):
    inquiry.record(conn, policy, requester="showrunner", method="prevalence", spec={}, result={}, data={"posts": 0})
    inquiry.record(conn, policy, requester="showrunner", method="prevalence", spec={}, result={}, data={"posts": 12})
    inquiry.record(conn, policy, requester="chief_of_staff", method="spine_sync", spec={}, result={},
                   data={"posts": 0})                                       # bookkeeping, not an analysis
    out = verify.run(conn, policy)
    assert out["checks"]["empty_analyses"]["count"] == 1 and "empty_analyses" in out["failing"]
