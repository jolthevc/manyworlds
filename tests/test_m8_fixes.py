"""Fixes from the M8 review (2026-11-20): Brain steps can't take production down, budgets pause launches rather
than kill them, creators waiting for an account stop rendering, test clones don't take launch slots, launches
can't touch an existing creator, and approved stage moves are applied."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds.brain import launches, portfolio, showrunner_loop

from brain_helpers import creator
from test_launches import _approve_launch, _run_lab, _thesis


def test_a_failing_brain_step_does_not_stop_production(conn, policy, monkeypatch):
    from manyworlds.brain import events
    from manyworlds import cli

    def boom(*a, **k):
        raise RuntimeError("detector bug")
    monkeypatch.setattr(events, "tick", boom)
    out = cli.brain_steps(conn, policy)
    assert out["errors"] and "detector bug" in out["errors"][0]["error"]
    ev = conn.execute("SELECT type, routed_to FROM events WHERE type = 'brain_step_failed'").fetchone()
    assert ev["routed_to"] == "owner"


def test_an_over_budget_lab_step_waits_instead_of_failing_the_launch(conn, policy):
    _approve_launch(conn, _thesis(conn))
    launches.advance(conn, policy)
    job = conn.execute("SELECT id FROM jobs WHERE kind = 'brain.creator_lab'").fetchone()["id"]
    conn.execute("UPDATE jobs SET status = 'failed', error = 'agent creator_lab is over its daily LLM budget of $5.00' "
                 "WHERE id = %s", (job,))
    for _ in range(5):
        launches.advance(conn, policy)
    row = conn.execute("SELECT status, retries, jobs FROM launches").fetchone()
    assert row["status"] == "running" and row["retries"] == 0 and row["jobs"].get("wait_until")
    assert conn.execute("SELECT count(*) n FROM jobs WHERE kind = 'brain.creator_lab'").fetchone()["n"] == 1


def test_a_creator_with_a_backlog_of_unpublished_posts_gets_no_new_batch(conn, policy):
    cid = creator(conn)
    conn.execute("UPDATE creators SET status = 'active', profile_version = 1, anchors = %s WHERE id = %s",
                 (Jsonb({"reference_images": [{"uri": "s3://b/r"}]}), cid))
    cap = int(policy.get("production", "max_ready_unpublished"))
    for _ in range(cap):
        conn.execute("INSERT INTO posts (creator_id, format, status, scheduled_for) VALUES (%s, 'photo', 'ready', now())",
                     (cid,))
    assert showrunner_loop.ensure_batches(conn, policy, now=datetime(2026, 11, 19, 8, tzinfo=timezone.utc)) == []


def test_sandbox_clones_and_failed_launches_do_not_take_launch_slots(conn, policy):
    clone = creator(conn)
    conn.execute("UPDATE creators SET status = 'draft', anchors = %s WHERE id = %s",
                 (Jsonb({"simulation": {"scenario": "null"}}), clone))
    snap = portfolio.snapshot(conn, policy)
    assert str(clone) not in [c["creator_id"] for c in snap["creators"]]


def test_an_approval_for_a_used_thesis_never_relaunches_a_creator(conn, policy):
    maren = creator(conn, publish_mode="instagram")
    t = _thesis(conn)
    conn.execute("UPDATE theses SET creator_id = %s, status = 'launched' WHERE id = %s", (maren, t))
    _approve_launch(conn, t)
    out = launches.advance(conn, policy)
    assert out["started"] == 0 and conn.execute("SELECT count(*) n FROM launches").fetchone()["n"] == 0
    assert conn.execute("SELECT publish_mode FROM creators WHERE id = %s", (maren,)).fetchone()["publish_mode"] == "instagram"


def test_a_rejected_launch_returns_its_thesis(conn, policy):
    t = _thesis(conn)
    conn.execute("UPDATE theses SET status = 'chosen' WHERE id = %s", (t,))
    conn.execute("INSERT INTO approvals (kind, subject, reason, requested_by, status) VALUES ('launch', %s, 'r', "
                 "'strategist', 'rejected')", (Jsonb({"thesis_id": str(t)}),))
    launches.advance(conn, policy)
    assert conn.execute("SELECT status FROM theses WHERE id = %s", (t,)).fetchone()["status"] == "proposed"


def test_an_approved_retirement_is_applied(conn, policy):
    from manyworlds.brain import ops
    cid = creator(conn, publish_mode="instagram")
    a = conn.execute("INSERT INTO approvals (kind, subject, reason, requested_by) VALUES ('stage_move', %s, 'flat', "
                     "'strategist') RETURNING id", (Jsonb({"creator_id": str(cid), "to": "retired", "reason": "flat"}),)
                     ).fetchone()["id"]
    ops.decide(conn, a, approved=True, by="owner")
    assert conn.execute("SELECT status FROM creators WHERE id = %s", (cid,)).fetchone()["status"] == "retired"
    assert conn.execute("SELECT stage FROM creator_states WHERE creator_id = %s", (cid,)).fetchone()["stage"] == "retired"


def test_coming_back_from_paused_reactivates_the_creator(conn, policy):
    cid = creator(conn, publish_mode="instagram")
    conn.execute("UPDATE creators SET status = 'paused' WHERE id = %s", (cid,))
    portfolio.apply_stage_move(conn, cid, from_stage="paused", to="iterating", reason="new plan", downward=set())
    assert conn.execute("SELECT status FROM creators WHERE id = %s", (cid,)).fetchone()["status"] == "active"


def test_a_reference_step_succeeds_when_enough_renders_did(conn, policy, monkeypatch):
    _approve_launch(conn, _thesis(conn))
    launches.advance(conn, policy)
    _run_lab(conn, policy, monkeypatch)
    launches.advance(conn, policy)
    refs = conn.execute("SELECT id FROM jobs WHERE kind = 'image.zimage' ORDER BY created_at").fetchall()
    conn.execute("UPDATE jobs SET status = 'done' WHERE kind LIKE 'image.%'")
    conn.execute("UPDATE jobs SET status = 'failed' WHERE id = %s", (refs[0]["id"],))   # one render failed for good
    launches.advance(conn, policy)
    assert conn.execute("SELECT step FROM launches").fetchone()["step"] == "anchors"


def test_a_launch_stuck_waiting_for_too_long_tells_the_owner(conn, policy):
    _approve_launch(conn, _thesis(conn))
    launches.advance(conn, policy)
    hours = float(policy.get("brain", "launches", "step_timeout_hours"))
    conn.execute("UPDATE launches SET updated_at = now() - make_interval(hours => %s)", (int(hours) + 1,))
    launches.advance(conn, policy)
    assert conn.execute("SELECT status FROM launches").fetchone()["status"] == "failed"
