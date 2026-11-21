"""The evaluator scores ideas before they are made and images after, independently of whoever
planned them, and stores every score with what it was scored under."""

from __future__ import annotations

import dataclasses

import pytest

from manyworlds import queue
from manyworlds.brain import evaluation, ideas, runtime
from manyworlds.brain.agents import evaluator

from brain_helpers import ScoringLLM, creator, make_ctx, planned_post


def _ctx(conn, policy, llm):
    job = queue.enqueue(conn, "brain.evaluator", {})
    ctx = make_ctx(conn, policy, job)
    ctx.llm = llm
    return ctx


def test_from_post_is_idempotent(conn):
    post = planned_post(conn, creator(conn))
    a, b = ideas.from_post(conn, post), ideas.from_post(conn, post)
    row = conn.execute("SELECT concept, format, attributes, post_id FROM ideas WHERE id = %s", (a,)).fetchone()
    assert a == b and row["post_id"] == post and row["format"] == "photo"
    assert "Venice" in row["concept"] and "best pasta" in row["concept"] and row["attributes"]["pillar"] == "food"


def test_an_idea_is_scored_and_stored_with_its_provenance(conn, policy):
    idea = ideas.from_post(conn, planned_post(conn, creator(conn)))
    eid = evaluator.evaluate_idea(_ctx(conn, policy, ScoringLLM(7, conn=conn)), idea)
    e = conn.execute("SELECT * FROM evaluations WHERE id = %s", (eid,)).fetchone()
    assert (e["item_kind"], e["idea_id"], e["rubric"], e["rubric_version"]) == ("idea", idea, "idea", 1)
    assert e["evaluator_model"] == "stub-evaluator" and e["prompt_version"].startswith("eval-v1-") and e["format"] == "photo"
    assert e["samples"] == policy.get("brain", "evaluation", "samples") and e["scores"]["novelty"] == 7
    assert evaluation.latest(conn, idea_id=idea)["id"] == eid


def test_the_evaluator_never_sees_the_showrunners_reasoning(conn, policy):
    idea = ideas.from_post(conn, planned_post(conn, creator(conn)))
    conn.execute("UPDATE ideas SET body = body || '{\"reasoning\": \"SECRET PLAN\"}' WHERE id = %s", (idea,))
    llm = ScoringLLM(conn=conn)
    evaluator.evaluate_idea(_ctx(conn, policy, llm), idea)
    assert llm.calls and not any("SECRET PLAN" in str(c["content"]) for c in llm.calls)


def test_a_rerun_does_not_store_a_second_evaluation(conn, policy):
    idea = ideas.from_post(conn, planned_post(conn, creator(conn)))
    ctx = _ctx(conn, policy, ScoringLLM(conn=conn))
    assert evaluator.evaluate_idea(ctx, idea) == evaluator.evaluate_idea(ctx, idea)
    # (about 1 in 10 ideas also gets a strong-model check row; that one is also written once)
    assert conn.execute("SELECT role, count(*) n FROM evaluations GROUP BY role HAVING count(*) > 1").fetchall() == []


def test_checked_items_get_a_strong_model_row_too(conn, policy):
    ev = {**policy.body["brain"]["evaluation"], "check_rate": 1.0}
    pol = dataclasses.replace(policy, body={**policy.body, "brain": {**policy.body["brain"], "evaluation": ev}})
    idea = ideas.from_post(conn, planned_post(conn, creator(conn)))
    evaluator.evaluate_idea(_ctx(conn, pol, ScoringLLM(conn=conn)), idea)
    models = sorted(r["evaluator_model"] for r in conn.execute("SELECT evaluator_model FROM evaluations").fetchall())
    assert models == ["stub-evaluator", "stub-evaluator_check"]


def test_an_image_is_scored_on_the_execution_rubric(conn, policy, store, tmp_path):
    cid = creator(conn)
    post = planned_post(conn, cid)
    p = tmp_path / "img.png"
    p.write_bytes(b"\x89PNG fake")
    job = queue.enqueue(conn, "image.zimage", {}, creator_id=cid, post_id=post)
    asset = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'image', %s, 'x', 1) "
                         "RETURNING id", (job, cid, store.put_file(p, "a/img.png").uri)).fetchone()["id"]
    llm = ScoringLLM(8, conn=conn)
    eid = evaluator.evaluate_asset(_ctx(conn, policy, llm), asset, store=store)
    e = conn.execute("SELECT item_kind, asset_id, rubric, scores FROM evaluations WHERE id = %s", (eid,)).fetchone()
    assert (e["item_kind"], e["asset_id"], e["rubric"], e["scores"]["believability"]) == ("asset", asset, "execution", 8)
    assert any(b.get("type") == "image" for b in llm.calls[0]["content"] if isinstance(b, dict))
    assert "best pasta" in str(llm.calls[0]["content"])                     # the caption is judged too


def test_an_asset_that_cannot_be_fetched_fails_without_a_row(conn, policy, store):
    cid = creator(conn)
    job = queue.enqueue(conn, "image.zimage", {}, creator_id=cid)
    asset = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) "
                         "VALUES (%s, %s, 'image', 'file:///nowhere/x.png', 'x', 1) RETURNING id", (job, cid)).fetchone()["id"]
    with pytest.raises(Exception):
        evaluator.evaluate_asset(_ctx(conn, policy, ScoringLLM(conn=conn)), asset, store=store)
    assert conn.execute("SELECT count(*) n FROM evaluations").fetchone()["n"] == 0


def test_the_evaluator_runs_as_a_brain_job(conn, policy):
    idea = ideas.from_post(conn, planned_post(conn, creator(conn)))
    job = queue.enqueue(conn, "brain.evaluator", {"idea_id": str(idea)})
    out = runtime.tick(conn, policy, ScoringLLM(conn=conn), worker_id="w1")
    assert out[0]["status"] == "done"
    assert conn.execute("SELECT count(*) n FROM evaluations WHERE idea_id = %s AND role = 'evaluator'", (idea,)).fetchone()["n"] == 1
    assert conn.execute("SELECT agent FROM cost_entries LIMIT 1").fetchone()["agent"] == "evaluator"


# --- final review fixes -------------------------------------------------------------------------

def _asset(conn, store, tmp_path, cid, refs=2):
    from psycopg.types.json import Jsonb
    ref_uris = []
    for i in range(refs):
        r = tmp_path / f"ref{i}.png"
        r.write_bytes(b"\x89PNG ref")
        ref_uris.append({"uri": store.put_file(r, f"r/{cid}/{i}.png").uri})
    conn.execute("UPDATE creators SET anchors = %s WHERE id = %s", (Jsonb({"reference_images": ref_uris}), cid))
    p = tmp_path / "img.png"
    p.write_bytes(b"\x89PNG fake")
    job = queue.enqueue(conn, "image.zimage", {}, creator_id=cid, post_id=planned_post(conn, cid))
    return conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'image', %s, 'x', 1) "
                        "RETURNING id", (job, cid, store.put_file(p, "a/img.png").uri)).fetchone()["id"]


def test_identity_is_judged_against_the_creators_reference_photos(conn, policy, store, tmp_path):
    cid = creator(conn)
    asset = _asset(conn, store, tmp_path, cid, refs=2)
    llm = ScoringLLM(8, conn=conn)
    eid = evaluator.evaluate_asset(_ctx(conn, policy, llm), asset, store=store)
    content = llm.calls[0]["content"]
    assert sum(1 for b in content if isinstance(b, dict) and b.get("type") == "image") == 3   # 2 references + the image
    assert any("REFERENCE" in b.get("text", "") for b in content if isinstance(b, dict))
    assert conn.execute("SELECT scores FROM evaluations WHERE id = %s", (eid,)).fetchone()["scores"]["identity"] == 8


def test_without_reference_photos_identity_is_null_not_invented(conn, policy, store, tmp_path):
    cid = creator(conn)
    asset = _asset(conn, store, tmp_path, cid, refs=0)
    eid = evaluator.evaluate_asset(_ctx(conn, policy, ScoringLLM(8, conn=conn)), asset, store=store)
    scores = conn.execute("SELECT scores FROM evaluations WHERE id = %s", (eid,)).fetchone()["scores"]
    assert scores["identity"] is None and scores["believability"] == 8


def test_the_evaluator_sees_the_creators_audience_and_pillars_from_the_profile(conn, policy):
    from psycopg.types.json import Jsonb
    cid = creator(conn)
    body = {"persona": {"vibe": "dry and warm", "caption_voice": "lowercase, deadpan"},
            "audience": {"what_they_get": "calm for night-shift nurses"},
            "content": {"pillars": [{"name": "slow mornings", "about": "the hour before the city wakes"}]}}
    conn.execute("INSERT INTO creator_profiles (creator_id, version, body, author) VALUES (%s, 3, %s, 'brain')",
                 (cid, Jsonb(body)))
    conn.execute("UPDATE creators SET profile_version = 3 WHERE id = %s", (cid,))
    idea = ideas.from_post(conn, planned_post(conn, cid))
    llm = ScoringLLM(conn=conn)
    eid = evaluator.evaluate_idea(_ctx(conn, policy, llm), idea)
    seen = str(llm.calls[0]["content"])
    assert "night-shift nurses" in seen and "slow mornings" in seen and "deadpan" in seen
    assert conn.execute("SELECT profile_version FROM evaluations WHERE id = %s", (eid,)).fetchone()["profile_version"] == 3


def test_latest_returns_the_volume_evaluator_not_the_check(conn, policy):
    ev = {**policy.body["brain"]["evaluation"], "check_rate": 1.0}
    pol = dataclasses.replace(policy, body={**policy.body, "brain": {**policy.body["brain"], "evaluation": ev}})
    idea = ideas.from_post(conn, planned_post(conn, creator(conn)))
    evaluator.evaluate_idea(_ctx(conn, pol, ScoringLLM(conn=conn)), idea)
    roles = {r["role"] for r in conn.execute("SELECT role FROM evaluations").fetchall()}
    assert roles == {"evaluator", "evaluator_check"}
    for _ in range(5):
        assert evaluation.latest(conn, idea_id=idea)["role"] == "evaluator"
    assert evaluation.latest(conn, idea_id=idea, role="evaluator_check")["role"] == "evaluator_check"


def test_the_prompt_version_changes_when_the_prompt_or_effort_does(policy):
    a = evaluation.prompt_version(policy)
    ev = {**policy.body["brain"]["evaluation"], "effort": "high"}
    pol = dataclasses.replace(policy, body={**policy.body, "brain": {**policy.body["brain"], "evaluation": ev}})
    assert a.startswith(policy.get("brain", "evaluation", "prompt_version")) and evaluation.prompt_version(pol) != a


def test_stored_evaluations_carry_the_computed_prompt_version(conn, policy):
    idea = ideas.from_post(conn, planned_post(conn, creator(conn)))
    eid = evaluator.evaluate_idea(_ctx(conn, policy, ScoringLLM(conn=conn)), idea)
    assert conn.execute("SELECT prompt_version FROM evaluations WHERE id = %s", (eid,)).fetchone()["prompt_version"] == \
        evaluation.prompt_version(policy)


# --- live-run fixes (2026-11-14) -------------------------------------------------------------------

def _check_everything(policy):
    ev = {**policy.body["brain"]["evaluation"], "check_rate": 1.0}
    return dataclasses.replace(policy, body={**policy.body, "brain": {**policy.body["brain"], "evaluation": ev}})


def test_a_refused_check_never_blocks_the_idea(conn, policy):
    from manyworlds.llm import Refused

    class RefusingCheck(ScoringLLM):
        def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
            if role == "evaluator_check":
                raise Refused("declined (reasoning_extraction)")
            return super().structured(role, system=system, content=content, schema=schema)
    pol = _check_everything(policy)
    idea = ideas.from_post(conn, planned_post(conn, creator(conn)))
    evaluator.evaluate_idea(_ctx(conn, pol, RefusingCheck(conn=conn)), idea)
    rows = conn.execute("SELECT role, rationale FROM evaluations WHERE idea_id = %s", (idea,)).fetchall()
    assert [r["role"] for r in rows] == ["evaluator"]


def test_the_check_scores_once_and_the_rubric_prefix_is_cached(conn, policy):
    pol = _check_everything(policy)
    idea = ideas.from_post(conn, planned_post(conn, creator(conn)))
    llm = ScoringLLM(conn=conn)
    evaluator.evaluate_idea(_ctx(conn, pol, llm), idea)
    check = conn.execute("SELECT samples FROM evaluations WHERE role = 'evaluator_check'").fetchone()
    assert check["samples"] == pol.get("brain", "evaluation", "check_samples") == 1
    first = llm.calls[0]["content"]
    assert first[0].get("cache_control") == {"type": "ephemeral"} and "RUBRIC" in first[0]["text"]


def test_an_idea_is_scored_once_unless_it_sits_near_a_floor_that_drops_ideas(conn, policy):
    # 2026-11-21: three samples per idea tripled the largest Brain bill; only scores that decide a drop need more.
    floor = policy.get("brain", "qa", "min_scores")["naturalness"]
    extra = policy.get("brain", "evaluation", "resample", "extra")
    c = creator(conn)
    far = evaluator.evaluate_idea(_ctx(conn, policy, ScoringLLM(floor + 3, conn=conn)),
                                  ideas.from_post(conn, planned_post(conn, c)))
    near = evaluator.evaluate_idea(_ctx(conn, policy, ScoringLLM(floor, conn=conn)),
                                   ideas.from_post(conn, planned_post(conn, c)))
    got = {str(r["id"]): r["samples"] for r in conn.execute("SELECT id, samples FROM evaluations WHERE role = 'evaluator'")}
    assert got[str(far)] == policy.get("brain", "evaluation", "samples") == 1
    assert got[str(near)] == 1 + extra
