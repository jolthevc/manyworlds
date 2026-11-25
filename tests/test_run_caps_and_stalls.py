"""Backstops that hold whatever a driver process does (review 2026-11-25): a test run's spend cap is enforced by the
autoscaler too, and posts left producing with nothing to run are dropped."""

from __future__ import annotations

from datetime import timedelta

from psycopg.types.json import Jsonb

from manyworlds import breadth, ledger, queue
from manyworlds.deploy import autoscale
from manyworlds.production.common import reap_stalled

from test_autoscale import T0, providers, registry  # noqa: F401 - registry: autouse image registry stub
from test_production import creator


def test_a_run_at_its_cap_gets_no_machine_and_its_queued_jobs_are_cancelled_without_its_driver(conn, policy):
    batch = "test-breadth-capped"
    breadth.register_cap(conn, batch, 1.0)
    done = queue.enqueue(conn, "train.lora", {}, batch=batch)
    ledger.record(conn, category="llm", provider="x", quantity=1, unit="token", usd=1.5, job_id=done)
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (done,))
    waiting = queue.enqueue(conn, "train.lora", {}, batch=batch)
    ps = providers()
    actions = autoscale.tick(conn, policy, ps, now=T0)
    assert not [m for p in ps.values() for m in p.launched]
    assert any(a.get("action") == "run_capped" and a["batch"] == batch for a in actions)
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (waiting,)).fetchone()["status"] == "failed"


def test_a_run_under_its_cap_still_launches(conn, policy):
    breadth.register_cap(conn, "test-breadth-ok", 5.0)
    queue.enqueue(conn, "train.lora", {}, batch="test-breadth-ok")
    ps = providers()
    autoscale.tick(conn, policy, ps, now=T0)
    assert [m for p in ps.values() for m in p.launched]


def test_a_post_producing_with_nothing_to_run_is_dropped_and_the_owner_told(conn, policy):
    cid = creator(conn)
    stale = conn.execute("INSERT INTO posts (creator_id, format, status, body, updated_at) VALUES (%s, 'photo', "
                         "'producing', %s, now() - interval '3 days') RETURNING id", (cid, Jsonb({}))).fetchone()["id"]
    busy = conn.execute("INSERT INTO posts (creator_id, format, status, body, updated_at) VALUES (%s, 'photo', "
                        "'producing', %s, now() - interval '3 days') RETURNING id", (cid, Jsonb({}))).fetchone()["id"]
    queue.enqueue(conn, "image.zimage", {}, post_id=busy, creator_id=cid)        # still has work: kept
    gone = reap_stalled(conn, policy)
    assert gone == [str(stale)]
    row = conn.execute("SELECT status, body FROM posts WHERE id = %s", (stale,)).fetchone()
    assert row["status"] == "dropped" and row["body"]["drop_kind"] == "failed"
    assert conn.execute("SELECT count(*) n FROM events WHERE type = 'posts_stalled'").fetchone()["n"] == 1
    assert conn.execute("SELECT status FROM posts WHERE id = %s", (busy,)).fetchone()["status"] == "producing"


def test_the_weights_volume_pool_is_scored_with_its_plain_twin_and_stays_first(conn, policy):
    # review 2026-11-25: scored alone, early volume failures left it "exhausted" behind the plain pool for good
    from datetime import datetime, timezone
    from manyworlds.deploy import scoring
    now = datetime.now(timezone.utc)
    for name, pool, outputs in (("mw-a", "runpod_secure_l40s", 30), ("mw-b", "runpod_secure_l40s_vol", 0)):
        conn.execute("INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd, "
                     "started_at, stopped_at, billed_usd, family) VALUES (%s, 'runpod', %s, 1, '{talk.infinitetalk}', "
                     "%s, 3, %s, %s, 2, 'talk')", (name, pool, now, now - timedelta(hours=2), now - timedelta(hours=1)))
    rows = scoring.score_pools(conn, policy, "talk", ["runpod_secure_l40s_vol", "runpod_secure_l40s"], now)
    assert rows[0]["pool"] == "runpod_secure_l40s_vol" and rows[0]["status"] == rows[1]["status"]
