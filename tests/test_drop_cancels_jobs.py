"""A post production drops takes its still-queued jobs with it (queue.cancel): live 2026-11-24 a dropped post left six
comfy.wan_animate2 jobs queued that would have started a GPU machine. Running jobs are never touched (a worker holds
their lease), other posts' jobs stay, and a piece shared by idempotency key (body.shared_key) stays while a sibling
post that shares it is still alive."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds import production, queue

from brain_helpers import creator


def _post(conn, cid, *, status="producing", fmt="photo", body=None):
    return conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, %s, %s, %s) RETURNING id",
                        (cid, fmt, status, Jsonb(body or {}))).fetchone()["id"]


def _job(conn, pid, key, *, status="queued"):
    jid = queue.enqueue(conn, "comfy.wan_animate2", {"x": key}, idempotency_key=key, post_id=pid)
    if status != "queued":
        conn.execute("UPDATE jobs SET status = %s WHERE id = %s", (status, jid))
    return jid


def _status(conn, jid):
    row = conn.execute("SELECT status, error FROM jobs WHERE id = %s", (jid,)).fetchone()
    return row["status"], row["error"] or ""


def test_a_dropped_post_cancels_its_queued_jobs_not_its_running_ones(conn):
    cid = creator(conn)
    pid, other = _post(conn, cid), _post(conn, cid)
    queued = [_job(conn, pid, f"post:{pid}:motion:{i}") for i in range(6)]
    running = _job(conn, pid, f"post:{pid}:motion:run", status="running")
    theirs = _job(conn, other, f"post:{other}:motion:0")
    production._set(conn, pid, "dropped", {"drop_reason": "review"}, qa_status="failed")
    for jid in queued:
        status, error = _status(conn, jid)
        assert status == "dead" and error.startswith(queue.CANCELLED) and "dropped" in error
    assert _status(conn, running)[0] == "running"
    assert _status(conn, theirs)[0] == "queued"


def test_a_post_whose_production_keeps_failing_cancels_its_queued_jobs(conn, policy):
    cid = creator(conn)
    limit = int(policy.get("production", "max_advance_errors"))
    pid = _post(conn, cid, body={"advance_errors": limit - 1})
    jid = _job(conn, pid, f"post:{pid}:talk:0")
    row = conn.execute("SELECT * FROM posts WHERE id = %s", (pid,)).fetchone()
    assert production._advance_failed(conn, policy, row, RuntimeError("boom")) is True
    assert _status(conn, jid)[0] == "dead"


def test_a_post_that_expires_unproduced_cancels_its_queued_jobs(conn, policy):
    cid = creator(conn)
    pid = _post(conn, cid, status="planned", fmt="not_a_made_format")
    conn.execute("UPDATE posts SET scheduled_for = %s WHERE id = %s",
                 (datetime.now(timezone.utc) - timedelta(days=1), pid))
    jid = _job(conn, pid, f"post:{pid}:voice:0:0")
    assert production.advance(conn, policy, now=datetime.now(timezone.utc))["expired"] == 1
    assert _status(conn, jid)[0] == "dead"


def test_a_shared_piece_stays_while_a_sibling_that_shares_it_lives(conn):
    cid = creator(conn)
    a = _post(conn, cid, body={"shared_key": "cmp-1"})
    b = _post(conn, cid, body={"shared_key": "cmp-1"})
    shared = _job(conn, a, "shared:cmp-1:0:0")          # made once for both; its post_id is the first post's
    own = _job(conn, a, f"post:{a}:talk:0")
    production._set(conn, a, "dropped", {"shared_key": "cmp-1"}, qa_status="failed")
    assert _status(conn, shared)[0] == "queued" and _status(conn, own)[0] == "dead"
    production._set(conn, b, "dropped", {"shared_key": "cmp-1"}, qa_status="failed")
    assert _status(conn, shared)[0] == "dead"           # the last sharer gone: nobody needs it
