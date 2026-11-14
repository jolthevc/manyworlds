"""The enforcer is the only thing standing between an idle GPU and a 20-hour bill."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from manyworlds import ledger, queue
from manyworlds.deploy.enforce import tick

T0 = datetime(2026, 11, 12, 6, 0, tzinfo=timezone.utc)


class FakeAPI:
    def __init__(self, names):
        self.live = set(names)
        self.stopped = []

    def list_names(self):
        return sorted(self.live)

    def stop_delete(self, name):
        self.stopped.append(name)
        self.live.discard(name)


def register(conn, name, *, kinds=("image.zimage",), price=0.16, hours=2, max_usd=1.0, started=T0):
    conn.execute(
        """INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, replicas, job_kinds, deadline, max_usd,
                                   started_at, billed_until, last_busy_at)
           VALUES (%s, 'salad', 'salad_batch_4090', %s, 1, %s, %s, %s, %s, %s, %s)""",
        (name, price, list(kinds), started + timedelta(hours=hours), max_usd, started, started, started))


def test_meters_billed_time_and_keeps_busy_groups(conn, policy):
    register(conn, "mw-image-4090-a")
    queue.enqueue(conn, "image.zimage", {})
    api = FakeAPI(["mw-image-4090-a"])
    actions = tick(conn, policy, api, now=T0 + timedelta(minutes=30))
    assert actions[0]["action"] == "ok" and api.stopped == []
    billed = conn.execute("SELECT sum(usd) u, sum(quantity) q FROM cost_entries WHERE category = 'gpu_billed'").fetchone()
    assert abs(float(billed["u"]) - 0.08) < 1e-9 and float(billed["q"]) == 1800
    tick(conn, policy, api, now=T0 + timedelta(minutes=40))   # incremental, not re-billed
    assert abs(ledger.bill_total(conn) - 0.16 / 6 * 4) < 1e-9


def test_stops_at_deadline_even_when_busy(conn, policy):
    register(conn, "mw-image-4090-a", hours=1)
    queue.enqueue(conn, "image.zimage", {})
    api = FakeAPI(["mw-image-4090-a"])
    actions = tick(conn, policy, api, now=T0 + timedelta(minutes=61))
    assert api.stopped == ["mw-image-4090-a"] and actions[0]["reason"] == "deadline"


def test_stops_idle_group_after_boot_grace_but_not_during_it(conn, policy):
    register(conn, "mw-image-4090-a", hours=12, max_usd=100)
    api = FakeAPI(["mw-image-4090-a"])
    tick(conn, policy, api, now=T0 + timedelta(minutes=40))    # still booting: no jobs, but in grace
    assert api.stopped == []
    actions = tick(conn, policy, api, now=T0 + timedelta(minutes=70))
    assert api.stopped == ["mw-image-4090-a"] and actions[0]["reason"] == "idle"


def test_yesterdays_failure_mode_is_caught(conn, policy):
    """Work finished at 08:19, nobody stopped the group: it must stop within the idle window."""
    register(conn, "mw-image-4090-r2", hours=12, max_usd=100)
    job = queue.enqueue(conn, "image.zimage", {})
    api = FakeAPI(["mw-image-4090-r2"])
    tick(conn, policy, api, now=T0 + timedelta(minutes=50))     # busy
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job,))
    tick(conn, policy, api, now=T0 + timedelta(minutes=60))     # idle 10 min: keep
    assert api.stopped == []
    tick(conn, policy, api, now=T0 + timedelta(minutes=72))     # idle 22 min: stop
    assert api.stopped == ["mw-image-4090-r2"]


def test_spend_cap_and_unregistered_groups(conn, policy):
    register(conn, "mw-image-4090-a", hours=12, max_usd=0.05)
    queue.enqueue(conn, "image.zimage", {})
    api = FakeAPI(["mw-image-4090-a", "mw-image-3090-rogue", "someone-elses-group"])
    actions = tick(conn, policy, api, now=T0 + timedelta(minutes=30))   # $0.08 billed > $0.05 cap
    reasons = {a["group"]: a.get("reason") for a in actions}
    assert reasons["mw-image-4090-a"] == "max_usd"
    assert reasons["mw-image-3090-rogue"] == "unregistered"
    assert "someone-elses-group" not in api.stopped               # never touches non-Manyworlds groups


def test_dry_run_changes_nothing(conn, policy):
    register(conn, "mw-image-4090-a", hours=1)
    api = FakeAPI(["mw-image-4090-a"])
    tick(conn, policy, api, now=T0 + timedelta(hours=3), dry_run=True)
    assert api.stopped == [] and ledger.bill_total(conn) == 0


def _vast(conn, name):
    register(conn, name, hours=12, max_usd=100)
    conn.execute("UPDATE gpu_groups SET provider = 'vast', pool = 'vast_3090' WHERE name = %s", (name,))


def _worked(conn, name, kind="image.zimage", at=T0):
    """The machine ran a job that is now finished."""
    job = queue.enqueue(conn, kind, {})
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job,))
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, outcome, gpu_group, started_at) "
                 "VALUES (%s, 'w', 'vast_3090', 'done', %s, %s)", (job, name, at))


def test_a_machine_that_finished_its_work_stops_without_waiting_out_boot_grace(conn, policy):
    from manyworlds.deploy.enforce import idle_stop_minutes
    _vast(conn, "mw-image-3090-vast-a")
    _worked(conn, "mw-image-3090-vast-a", at=T0 + timedelta(minutes=5))
    api = FakeAPI(["mw-image-3090-vast-a"])
    idle = idle_stop_minutes(policy, "vast")
    assert idle < float(policy.get("gpu_runs", "boot_grace_minutes"))
    conn.execute("UPDATE gpu_groups SET last_busy_at = %s", (T0 + timedelta(minutes=10),))  # job ended at 10 min
    tick(conn, policy, api, now=T0 + timedelta(minutes=10 + idle - 1))     # within the idle window
    assert api.stopped == []
    actions = tick(conn, policy, api, now=T0 + timedelta(minutes=10 + idle + 1))
    assert api.stopped == ["mw-image-3090-vast-a"] and actions[0]["reason"] == "idle"


def test_a_machine_still_booting_keeps_its_grace(conn, policy):
    _vast(conn, "mw-image-3090-vast-b")                                    # no job started yet
    api = FakeAPI(["mw-image-3090-vast-b"])
    tick(conn, policy, api, now=T0 + timedelta(minutes=30))
    assert api.stopped == []


def test_slow_starting_providers_keep_the_longer_idle_window(conn, policy):
    from manyworlds.deploy.enforce import idle_stop_minutes
    assert idle_stop_minutes(policy, "salad") == float(policy.get("gpu_runs", "idle_stop_minutes"))
    assert idle_stop_minutes(policy, "vast") < idle_stop_minutes(policy, "salad")


def test_a_machine_whose_worker_went_silent_is_stopped_so_work_can_move(conn, policy):
    # 2026-11-14: a Vast worker hung loading the model; its job stayed 'running' with an
    # expired lease, so the machine looked busy and would have billed until its deadline.
    register(conn, "mw-image-4090-a", hours=12, max_usd=100)
    job = queue.enqueue(conn, "image.zimage", {})
    conn.execute("UPDATE jobs SET status = 'running', worker_id = 'w1', updated_at = %s, lease_until = %s WHERE id = %s",
                 (T0 + timedelta(minutes=8), T0 + timedelta(minutes=10), job))
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, started_at, gpu_group) "
                 "VALUES (%s, 'w1', 'salad_batch_4090', %s, 'mw-image-4090-a')", (job, T0 + timedelta(minutes=5)))
    api = FakeAPI(["mw-image-4090-a"])
    stuck = float(policy.get("gpu_runs", "stuck_minutes"))
    tick(conn, policy, api, now=T0 + timedelta(minutes=8 + stuck - 1))
    assert api.stopped == []                                  # heartbeat recent enough
    actions = tick(conn, policy, api, now=T0 + timedelta(minutes=8 + stuck + 1))
    assert api.stopped == ["mw-image-4090-a"] and actions[0]["reason"] == "stuck"
