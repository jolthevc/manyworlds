"""Paid GPU time goes to real posts, or the machine stops (production.pull_forward)."""

from __future__ import annotations

from datetime import date, timedelta

from psycopg.types.json import Jsonb

from manyworlds import policy as policy_mod, production, queue
from manyworlds.deploy.enforce import tick
from manyworlds.production.pull_forward import demand, pull_forward
from test_enforce import FakeAPI, register
from test_production import NOW, PlanStub, creator, post
from manyworlds.brain import planner


def far_post(conn, policy, cid, days=3):
    """A planned photo post scheduled `days` after NOW: beyond lead_hours (48), within the horizon (120)."""
    planner.plan_period(conn, policy, PlanStub([post(0)]), cid, start=date(2026, 11, 13) + timedelta(days=days))
    pid = conn.execute("SELECT id FROM posts WHERE creator_id = %s ORDER BY created_at DESC", (cid,)).fetchone()["id"]
    conn.execute("UPDATE posts SET scheduled_for = %s WHERE id = %s", (NOW + timedelta(days=days), pid))
    return pid


def warm(conn, name="mw-image-4090-w", started=None, **kw):
    started = started or NOW - timedelta(minutes=30)
    register(conn, name, hours=4, max_usd=5.0, started=started, **kw)
    conn.execute("UPDATE gpu_groups SET last_busy_at = %s WHERE name = %s", (NOW - timedelta(minutes=5), name))
    job = queue.enqueue(conn, "image.zimage", {})
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job,))
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, outcome, gpu_group, started_at, finished_at) "
                 "VALUES (%s, 'w', 'salad_batch_4090', 'done', %s, %s, %s)",
                 (job, name, NOW - timedelta(minutes=10), NOW - timedelta(minutes=5)))


def status(conn, pid):
    return conn.execute("SELECT status, body FROM posts WHERE id = %s", (pid,)).fetchone()


def test_a_warm_idle_machine_starts_the_next_planned_post_early(conn, policy):
    cid = creator(conn)
    pid = far_post(conn, policy, cid)
    warm(conn)
    out = production.advance(conn, policy, now=NOW)
    assert out["started"] == 0 and out["pulled_forward"] == 1
    row = status(conn, pid)
    assert row["status"] == "producing" and row["body"]["pulled_forward"]["machine"] == "mw-image-4090-w"
    assert conn.execute("SELECT count(*) n FROM jobs WHERE post_id = %s AND status = 'queued'", (pid,)).fetchone()["n"] == 1


def test_nothing_is_pulled_without_a_warm_machine_or_while_it_is_busy(conn, policy):
    cid = creator(conn)
    pid = far_post(conn, policy, cid)
    assert production.advance(conn, policy, now=NOW).get("pulled_forward") is None     # no machine: wait
    warm(conn)
    for _ in range(3):
        queue.enqueue(conn, "image.zimage", {})                                            # today's work first
    assert pull_forward(conn, policy, now=NOW, start=production.start)["why"] == "machine busy"
    assert status(conn, pid)["status"] == "planned"


def test_a_booting_machine_or_one_near_its_deadline_is_not_fed(conn, policy):
    cid = creator(conn)
    far_post(conn, policy, cid)
    register(conn, "mw-image-4090-b", hours=4, max_usd=5.0, started=NOW - timedelta(minutes=5))   # no attempt yet
    assert pull_forward(conn, policy, now=NOW, start=production.start)["why"] == "no warm machine"
    warm(conn, "mw-image-4090-d", started=NOW - timedelta(hours=3, minutes=50))                 # 10 min left
    assert pull_forward(conn, policy, now=NOW, start=production.start)["why"] == "no warm machine"


def test_beyond_the_horizon_test_posts_and_paused_operations_wait(conn, policy, raw_policy):
    cid = creator(conn)
    far = far_post(conn, policy, cid, days=7)                     # beyond horizon_hours
    test = far_post(conn, policy, cid, days=3)
    conn.execute("UPDATE posts SET body = body || %s WHERE id = %s", (Jsonb({"breadth_run": "r1"}), test))
    warm(conn)
    assert pull_forward(conn, policy, now=NOW, start=production.start)["pulled"] == 0
    assert status(conn, far)["status"] == "planned" and status(conn, test)["status"] == "planned"
    paused = policy_mod.from_dict({**policy.body, "operations": {**policy.body["operations"], "mode": "paused"}})
    assert pull_forward(conn, paused, now=NOW, start=production.start)["why"] == "operations paused"


def test_over_the_daily_budget_share_nothing_is_pulled(conn, policy):
    cid = creator(conn)
    far_post(conn, policy, cid)
    warm(conn)
    cap = float(policy.get("gpu_runs", "daily_gpu_usd"))
    conn.execute("INSERT INTO cost_entries (category, provider, quantity, unit, usd) "
                 "VALUES ('gpu_billed', 'salad', 1, 'hour', %s)", (cap,))
    assert pull_forward(conn, policy, now=NOW, start=production.start)["why"] == "daily GPU budget"


def test_with_no_real_work_coming_an_idle_machine_stops_after_the_short_allowance(conn, policy):
    warm(conn, started=NOW - timedelta(minutes=30))
    assert demand(conn, policy, now=NOW) is False
    api = FakeAPI(["mw-image-4090-w"])
    actions = tick(conn, policy, api, now=NOW)       # idle 5 min (> no-demand 2 min, < salad's 20)
    assert api.stopped == ["mw-image-4090-w"] and actions[0]["reason"] == "idle"


def test_with_a_post_producing_the_usual_idle_allowance_holds(conn, policy):
    cid = creator(conn)
    pid = far_post(conn, policy, cid)
    conn.execute("UPDATE posts SET status = 'producing' WHERE id = %s", (pid,))
    warm(conn, started=NOW - timedelta(minutes=30))
    assert demand(conn, policy, now=NOW) is True
    api = FakeAPI(["mw-image-4090-w"])
    tick(conn, policy, api, now=NOW)
    assert api.stopped == []
