"""Launching creators with no bespoke code (Brain M8): an approved launch walks the creator lab's steps by
itself (bible, reference pack, anchors, LoRA, voice, profile, dossier) and ends with a creator ready to plan in the sandbox and
an account request for the owner. A step that keeps failing stops the launch and tells the owner."""

from __future__ import annotations

import pytest
from psycopg.types.json import Jsonb

from manyworlds import policy as policy_mod
from manyworlds.brain import launches

from brain_helpers import StubLLM
from test_review import BIBLE
from voice_helpers import cast_meta, finished_cast


@pytest.fixture
def policy(policy):
    """These tests walk the launch path without casting (bible first); the casting step (brain.casting.enabled, on
    in the shipped policy) has its own tests in test_talent_agency.py."""
    return policy_mod.from_dict({**policy.body, "brain": {**policy.body["brain"],
                                                         "casting": {**policy.body["brain"]["casting"],
                                                                     "enabled": False}}})


def _thesis(conn, summary="cosy-chaos home cook in her 30s"):
    return conn.execute("INSERT INTO theses (summary, status, body) VALUES (%s, 'proposed', %s) RETURNING id",
                        (summary, Jsonb({"summary": summary}))).fetchone()["id"]


def _approve_launch(conn, thesis_id, cohort="c1"):
    return conn.execute("INSERT INTO approvals (kind, subject, reason, requested_by, status, decided_by) "
                        "VALUES ('launch', %s, 'test', 'strategist', 'approved', 'owner') RETURNING id",
                        (Jsonb({"thesis_id": str(thesis_id), "cohort": cohort}),)).fetchone()["id"]


def _jobs(conn, kind=None):
    q = "SELECT id, kind, payload, status FROM jobs" + (" WHERE kind = %s" if kind else "") + " ORDER BY created_at"
    return conn.execute(q, (kind,) if kind else ()).fetchall()


def _finish(conn, kind_prefix):
    conn.execute("UPDATE jobs SET status = 'done' WHERE kind LIKE %s AND status = 'queued'", (kind_prefix + "%",))


class FakeLab:
    """Stands in for the creator lab's Claude steps (bible, anchors, profile)."""

    def __init__(self, conn, refs=8):
        self.conn, self.refs = conn, refs

    def bible(self, conn, policy, llm, thesis):
        cid = conn.execute("INSERT INTO creators (handle, status, bible, anchors) VALUES ('newcook', 'draft', %s, %s) "
                           "RETURNING id", (Jsonb({**BIBLE, "display_name": "Nia", "one_line": "cooks badly, happily",
                                  "reference_prompts": [f"reference scene {i}" for i in range(10)]}),
                            Jsonb({"voice": {"voice": "af_heart"}}))
                           ).fetchone()["id"]
        conn.execute("UPDATE theses SET creator_id = %s, status = 'launched' WHERE id = %s", (cid, thesis["id"]))
        return cid

    def anchors(self, conn, policy, llm, creator_id):
        conn.execute("UPDATE creators SET anchors = anchors || %s WHERE id = %s",
                     (Jsonb({"reference_images": [{"uri": f"s3://b/r{i}"} for i in range(self.refs)]}), creator_id))

    def profile(self, conn, policy, llm, creator_id):
        conn.execute("UPDATE creators SET profile_version = 1 WHERE id = %s", (creator_id,))


def _run_lab(conn, policy, monkeypatch, refs=8):
    """The brain worker runs whatever creator_lab jobs are queued."""
    from manyworlds.brain.agents import creator_lab
    from brain_helpers import make_ctx
    monkeypatch.setattr(creator_lab, "LAB", FakeLab(conn, refs))
    for j in _jobs(conn, "brain.creator_lab"):
        if j["status"] == "queued":
            ctx = make_ctx(conn, policy, j["id"])
            ctx.llm = StubLLM(conn=conn)
            creator_lab.CreatorLab().run(ctx)
            conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (j["id"],))


def test_an_approved_launch_becomes_a_ready_sandbox_creator_with_an_account_request(conn, policy, monkeypatch):
    t = _thesis(conn)
    _approve_launch(conn, t)
    launches.advance(conn, policy)                                     # approval -> launch -> bible step
    lid = conn.execute("SELECT id, step FROM launches").fetchone()
    assert lid["step"] == "bible"
    _run_lab(conn, policy, monkeypatch)
    launches.advance(conn, policy)                                     # creator exists -> reference pack + voice
    assert {j["kind"] for j in _jobs(conn)} >= {"image.zimage", "image.voice"}
    launches.advance(conn, policy)
    assert conn.execute("SELECT step FROM launches").fetchone()["step"] == "references"   # waits for the renders
    _finish(conn, "image.")
    launches.advance(conn, policy)                                     # -> anchors (Claude picks references)
    _run_lab(conn, policy, monkeypatch)
    launches.advance(conn, policy)                                     # -> LoRA training queued
    assert _jobs(conn, "train.lora")
    _finish(conn, "train.")
    launches.advance(conn, policy)                                     # -> voice: cast automatically
    (cast,) = _jobs(conn, "voice.cast")
    assert conn.execute("SELECT step FROM launches").fetchone()["step"] == "voice"
    engine = cast["payload"]["engine"]
    finished_cast(conn, conn.execute("SELECT creator_id FROM launches").fetchone()["creator_id"],
                  [cast_meta(0, age=29.0, engine=engine)], job_id=cast["id"])
    launches.advance(conn, policy)                                     # her voice is stored -> profile
    assert conn.execute("SELECT c.anchors->'voice'->>'engine' AS e FROM launches l JOIN creators c "
                        "ON c.id = l.creator_id").fetchone()["e"] == engine
    _run_lab(conn, policy, monkeypatch)
    launches.advance(conn, policy)                                     # -> dossier, by the talent manager
    tm = _jobs(conn, "brain.talent_manager")
    assert [j["payload"]["mode"] for j in tm] == ["onboard"] and conn.execute(
        "SELECT step FROM launches").fetchone()["step"] == "dossier"
    _finish(conn, "brain.talent_manager")
    launches.advance(conn, policy)                                     # -> style, by the style director
    assert _jobs(conn, "brain.style_director") and conn.execute("SELECT step FROM launches").fetchone()["step"] == "style"
    _finish(conn, "brain.style_director")
    out = launches.advance(conn, policy)
    row = conn.execute("SELECT l.step, l.status, c.status cs, c.publish_mode FROM launches l "
                       "JOIN creators c ON c.id = l.creator_id").fetchone()
    assert (row["step"], row["status"], row["cs"], row["publish_mode"]) == ("ready", "done", "active", "simulated")
    acc = conn.execute("SELECT platform, status FROM accounts").fetchall()
    assert [(a["platform"], a["status"]) for a in acc] == [("instagram", "requested")]
    ev = conn.execute("SELECT type, routed_to FROM events WHERE type = 'account_needed'").fetchone()
    assert ev["routed_to"] == "owner" and out["ready"] == 1


def test_one_approval_launches_once(conn, policy):
    _approve_launch(conn, _thesis(conn))
    launches.advance(conn, policy)
    launches.advance(conn, policy)
    assert conn.execute("SELECT count(*) n FROM launches").fetchone()["n"] == 1
    assert len(_jobs(conn, "brain.creator_lab")) == 1


def test_a_launch_whose_step_keeps_failing_stops_and_tells_the_owner(conn, policy, monkeypatch):
    _approve_launch(conn, _thesis(conn))
    launches.advance(conn, policy)
    _run_lab(conn, policy, monkeypatch)
    launches.advance(conn, policy)
    limit = int(policy.get("brain", "launches", "max_step_retries"))
    for _ in range(limit + 1):
        conn.execute("UPDATE jobs SET status = 'failed' WHERE kind LIKE 'image.%' AND status = 'queued'")
        launches.advance(conn, policy)
    row = conn.execute("SELECT status, step FROM launches").fetchone()
    assert row["status"] == "failed" and row["step"] == "references"
    assert conn.execute("SELECT routed_to FROM events WHERE type = 'launch_failed'").fetchone()["routed_to"] == "owner"


def test_a_launch_that_cannot_train_stops_with_the_reason(conn, policy, monkeypatch):
    _approve_launch(conn, _thesis(conn))
    launches.advance(conn, policy)
    _run_lab(conn, policy, monkeypatch)
    launches.advance(conn, policy)
    _finish(conn, "image.")
    launches.advance(conn, policy)
    _run_lab(conn, policy, monkeypatch, refs=2)                         # too few usable reference images
    out = launches.advance(conn, policy)                                # the producer loop does not crash
    row = conn.execute("SELECT status, error FROM launches").fetchone()
    assert out["failed"] == 1 and row["status"] == "failed" and "reference images" in row["error"]


def _launch_to_creator(conn, policy, monkeypatch, body):
    t = conn.execute("INSERT INTO theses (summary, status, body) VALUES ('s', 'proposed', %s) RETURNING id",
                     (Jsonb(body),)).fetchone()["id"]
    _approve_launch(conn, t)
    launches.advance(conn, policy)
    _run_lab(conn, policy, monkeypatch)
    launches.advance(conn, policy)                                     # the bible step finds the creator
    return conn.execute("SELECT market_id, cluster_id FROM creators WHERE handle = 'newcook'").fetchone()


def test_a_launched_creator_joins_the_market_its_thesis_names(conn, policy, monkeypatch):
    row = _launch_to_creator(conn, policy, monkeypatch, {"summary": "s", "market": "slow living"})
    m = conn.execute("SELECT id FROM markets WHERE name = 'slow living'").fetchone()
    assert m and row["market_id"] == m["id"] and row["cluster_id"] is None


def test_a_cohort_thesis_places_its_creator_in_its_market_and_cluster(conn, policy, monkeypatch):
    m = conn.execute("INSERT INTO markets (name) VALUES ('night routine') RETURNING id").fetchone()["id"]
    c = conn.execute("INSERT INTO clusters (market_id, name) VALUES (%s, 'quiet nights') RETURNING id", (m,)).fetchone()["id"]
    row = _launch_to_creator(conn, policy, monkeypatch, {"summary": "s", "market_id": str(m), "cluster_id": str(c)})
    assert (row["market_id"], row["cluster_id"]) == (m, c)
    assert conn.execute("SELECT count(*) n FROM markets").fetchone()["n"] == 1   # no market made


def test_a_thesis_without_a_market_leaves_the_creator_unplaced(conn, policy, monkeypatch):
    row = _launch_to_creator(conn, policy, monkeypatch, {"summary": "s", "market_id": "not-a-uuid"})
    assert row["market_id"] is None and row["cluster_id"] is None
    assert conn.execute("SELECT count(*) n FROM markets").fetchone()["n"] == 0


def _to_voice_step(conn, policy, monkeypatch):
    _approve_launch(conn, _thesis(conn))
    launches.advance(conn, policy)
    _run_lab(conn, policy, monkeypatch)
    launches.advance(conn, policy)
    _finish(conn, "image.")
    launches.advance(conn, policy)
    _run_lab(conn, policy, monkeypatch)
    launches.advance(conn, policy)
    _finish(conn, "train.")
    launches.advance(conn, policy)


def test_a_cast_where_no_candidate_qualifies_is_cast_again_with_new_candidates(conn, policy, monkeypatch):
    _to_voice_step(conn, policy, monkeypatch)
    cid = conn.execute("SELECT creator_id FROM launches").fetchone()["creator_id"]
    (first,) = _jobs(conn, "voice.cast")
    finished_cast(conn, cid, [cast_meta(0, gender="male")], job_id=first["id"])     # the wrong gender: no voice
    launches.advance(conn, policy)
    casts = _jobs(conn, "voice.cast")
    assert len(casts) == 2 and casts[1]["payload"]["seed"] != first["payload"]["seed"]
    row = conn.execute("SELECT step, status, retries FROM launches").fetchone()
    assert (row["step"], row["status"], row["retries"]) == ("voice", "running", 1)


def test_casting_at_launch_can_be_switched_off(conn, policy, monkeypatch):
    import dataclasses
    off = dataclasses.replace(policy, body={**policy.body, "voice": {
        **policy.body["voice"], "casting": {**policy.body["voice"]["casting"], "at_launch": False}}})
    _to_voice_step(conn, off, monkeypatch)
    assert conn.execute("SELECT step FROM launches").fetchone()["step"] == "profile" and not _jobs(conn, "voice.cast")
