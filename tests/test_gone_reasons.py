"""A machine missing from its provider's list was not stopped by us: its worker destroyed it (idle, retired after
machine faults, past its deadline) or the provider took it. 'gone' alone hid which (2026-11-22..24: $3.81 of the
$6.76 billed on machines that finished nothing was 'gone'); the enforcer now says which one it was."""

from __future__ import annotations

from datetime import timedelta

from manyworlds import queue
from manyworlds.deploy.enforce import tick

from test_autoscale import registry  # noqa: F401 - fixture: a registry and published images
from test_enforce import T0, FakeAPI, register


def _attempt(conn, group, *, start, end=None, outcome=None, kind="image.zimage"):
    job = queue.enqueue(conn, kind, {})
    conn.execute("UPDATE jobs SET status = 'failed' WHERE id = %s", (job,))
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, started_at, finished_at, outcome, gpu_group) "
                 "VALUES (%s, 'w1', 'salad_batch_4090', %s, %s, %s, %s)",
                 (job, T0 + timedelta(minutes=start), None if end is None else T0 + timedelta(minutes=end),
                  outcome, group))
    return job


def _gone(conn, policy, name, minutes):
    api = FakeAPI([])
    actions = tick(conn, policy, api, now=T0 + timedelta(minutes=minutes))
    row = conn.execute("SELECT stop_reason FROM gpu_groups WHERE name = %s", (name,)).fetchone()
    assert [a["action"] for a in actions if a["group"] == name] == ["marked gone"]
    return row["stop_reason"]


def test_a_worker_that_idled_out_is_gone_idle(conn, policy):
    register(conn, "mw-a", hours=12, max_usd=100)
    _attempt(conn, "mw-a", start=2, end=10, outcome="done")
    idle = float(policy.get("gpu_runs", "idle_stop_minutes"))   # salad: the default window
    assert _gone(conn, policy, "mw-a", 10 + idle + 1) == "gone:idle"


def test_a_machine_that_vanished_mid_job_says_so(conn, policy):
    register(conn, "mw-b", hours=12, max_usd=100)
    _attempt(conn, "mw-b", start=2, end=10, outcome="done")
    _attempt(conn, "mw-b", start=10)                             # running when the machine disappeared
    assert _gone(conn, policy, "mw-b", 13) == "gone:mid_job"


def test_a_machine_retired_after_machine_faults_says_so(conn, policy):
    # The worker moves its deadline to now and destroys itself after N machine faults in a row (CUDA unknown error,
    # disk quota exceeded): seven such machines on 2026-11-22..24.
    register(conn, "mw-c", hours=12, max_usd=100)
    n = int(policy.get("queue", "worker_max_consecutive_faults"))
    for i in range(n):
        _attempt(conn, "mw-c", start=2 + i * 0.1, end=2.05 + i * 0.1, outcome="worker_fault")
    conn.execute("UPDATE gpu_groups SET deadline = %s WHERE name = 'mw-c'", (T0 + timedelta(minutes=3),))
    assert _gone(conn, policy, "mw-c", 4) == "gone:retired"


def test_past_deadline_and_before_any_job(conn, policy):
    register(conn, "mw-d", hours=1)
    _attempt(conn, "mw-d", start=2, end=50, outcome="done")
    assert _gone(conn, policy, "mw-d", 61) == "gone:deadline"
    register(conn, "mw-e", hours=12, max_usd=100)
    queue.enqueue(conn, "image.zimage", {})
    assert _gone(conn, policy, "mw-e", 30) == "gone:before_first_job"


def test_gone_soon_after_a_job_is_unexplained(conn, policy):
    register(conn, "mw-f", hours=12, max_usd=100)
    _attempt(conn, "mw-f", start=2, end=10, outcome="done")
    assert _gone(conn, policy, "mw-f", 11) == "gone"


def test_a_worker_that_idled_out_after_work_triggers_no_relaunch_cooldown(conn, policy, registry):
    """Its worker exits once the queue is empty; work that arrives later is new work, not a crash loop. A machine
    that idled out having finished nothing (every job failed) still waits out the cooldown."""
    from test_autoscale import T0 as A0, providers
    from manyworlds.deploy import autoscale
    queue.enqueue(conn, "train.lora", {})
    ps = providers()
    autoscale.tick(conn, policy, ps, now=A0)
    name = conn.execute("SELECT name FROM gpu_groups").fetchone()["name"]
    job = conn.execute("SELECT id FROM jobs").fetchone()["id"]
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job,))
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, started_at, finished_at, outcome, gpu_group) "
                 "VALUES (%s, 'w1', 'p', %s, %s, 'done', %s)", (job, A0, A0 + timedelta(minutes=5), name))
    ps["vast"].live.clear()                                       # the worker destroyed its machine when idle
    later = A0 + timedelta(minutes=30)
    autoscale.tick(conn, policy, ps, now=later)
    assert conn.execute("SELECT stop_reason FROM gpu_groups WHERE name = %s", (name,)).fetchone()["stop_reason"] \
        == "gone:idle"
    queue.enqueue(conn, "train.lora", {})
    actions = autoscale.tick(conn, policy, ps, now=later + timedelta(minutes=1))
    assert not any(a.get("action") == "cooldown" for a in actions) and len(ps["vast"].launched) == 2
