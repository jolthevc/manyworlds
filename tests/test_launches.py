"""Launching creators with no bespoke code (Brain M8): an approved launch walks the creator lab's steps by
itself (bible, reference pack, anchors, LoRA, profile) and ends with a creator ready to plan in the sandbox and
an account request for the owner. A step that keeps failing stops the launch and tells the owner."""

from __future__ import annotations

from psycopg.types.json import Jsonb

from manyworlds.brain import launches

from brain_helpers import StubLLM
from test_review import BIBLE


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
                                  "reference_prompts": ["at the stove", "at the market"]}),
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
    launches.advance(conn, policy)                                     # -> profile
    _run_lab(conn, policy, monkeypatch)
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
