"""The enforcer is the only thing standing between an idle GPU and a 20-hour bill."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from manyworlds import ledger, policy as policy_mod, queue
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


def _usual(policy):
    """The usual idle window: real work could still come (the no-demand stop is tested in test_pull_forward.py)."""
    return policy_mod.from_dict({**policy.body, "gpu_runs": {
        k: v for k, v in policy.body["gpu_runs"].items() if k != "idle_stop_minutes_no_demand"}})


def test_yesterdays_failure_mode_is_caught(conn, policy):
    """Work finished at 08:19, nobody stopped the group: it must stop within the idle window."""
    policy = _usual(policy)
    register(conn, "mw-image-4090-r2", hours=12, max_usd=100)
    job = queue.enqueue(conn, "image.zimage", {})
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, started_at, gpu_group) "
                 "VALUES (%s, 'w1', 'salad_batch_4090', %s, 'mw-image-4090-r2')", (job, T0 + timedelta(minutes=45)))
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
    policy = _usual(policy)
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


def test_a_machine_alive_but_never_finishing_its_job_is_stopped(conn, policy):
    # 2026-11-14: a Vast worker heartbeated for 25 min while its model download stalled.
    register(conn, "mw-image-4090-a", hours=12, max_usd=100)
    conn.execute("UPDATE gpu_groups SET family = 'image' WHERE name = 'mw-image-4090-a'")
    job = queue.enqueue(conn, "image.zimage", {})
    limit = float(policy.get("gpu_runs", "max_attempt_minutes_by_family")["default"])
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, started_at, gpu_group) "
                 "VALUES (%s, 'w1', 'salad_batch_4090', %s, 'mw-image-4090-a')", (job, T0))
    api = FakeAPI(["mw-image-4090-a"])
    for m in (limit - 1, limit + 1):    # heartbeating the whole time
        conn.execute("UPDATE jobs SET status = 'running', worker_id = 'w1', updated_at = %s WHERE id = %s",
                     (T0 + timedelta(minutes=m), job))
        actions = tick(conn, policy, api, now=T0 + timedelta(minutes=m))
    assert api.stopped == ["mw-image-4090-a"] and actions[0]["reason"] == "attempt_timeout"


def test_a_machine_that_never_starts_while_work_waits_is_stopped_after_boot_grace(conn, policy):
    # 2026-11-14: a RunPod pod sat rented for 100 min without its container starting. Jobs were
    # queued, so it counted as busy (never idle) and had no attempts (never stuck).
    register(conn, "mw-image-4090-a", hours=12, max_usd=100)
    queue.enqueue(conn, "image.zimage", {})
    api = FakeAPI(["mw-image-4090-a"])
    grace = float(policy.get("gpu_runs", "boot_grace_minutes"))
    tick(conn, policy, api, now=T0 + timedelta(minutes=grace - 1))
    assert api.stopped == []
    actions = tick(conn, policy, api, now=T0 + timedelta(minutes=grace + 1))
    assert api.stopped == ["mw-image-4090-a"] and actions[0]["reason"] == "never_started"


def test_fast_booting_providers_get_a_shorter_boot_grace(conn, policy):
    # Measured 2026-11-14: RunPod machines that worked started their first job within 8 minutes;
    # the one that took longer never started and billed 100 minutes.
    from manyworlds.deploy.enforce import boot_grace_minutes
    grace = boot_grace_minutes(policy, "runpod")
    assert grace < boot_grace_minutes(policy, "salad") == float(policy.get("gpu_runs", "boot_grace_minutes"))
    register(conn, "mw-image-3090-runpod-a", hours=12, max_usd=100)
    conn.execute("UPDATE gpu_groups SET provider = 'runpod' WHERE name = 'mw-image-3090-runpod-a'")
    queue.enqueue(conn, "image.zimage", {})
    api = FakeAPI(["mw-image-3090-runpod-a"])
    tick(conn, policy, api, now=T0 + timedelta(minutes=grace - 1))
    assert api.stopped == []
    actions = tick(conn, policy, api, now=T0 + timedelta(minutes=grace + 1))
    assert api.stopped == ["mw-image-3090-runpod-a"] and actions[0]["reason"] == "never_started"


def test_the_machine_report_separates_machines_that_made_nothing(conn, policy):
    from manyworlds.deploy import report
    now = datetime.now(timezone.utc)
    for name, minutes in (("mw-image-3090-vast-w", 30), ("mw-image-3090-vast-x", 20)):
        register(conn, name, started=now - timedelta(minutes=minutes))
        conn.execute("UPDATE gpu_groups SET stopped_at = %s, stop_reason = %s, billed_usd = %s WHERE name = %s",
                     (now, "idle" if name.endswith("w") else "never_started", minutes / 100, name))
    _worked(conn, "mw-image-3090-vast-w", at=now - timedelta(minutes=25))
    conn.execute("UPDATE attempts SET finished_at = started_at + interval '15 minutes'")
    rows = report.machines(conn, hours=1, now=now)
    s = report.summary(rows)
    assert s["machines"] == 2 and s["no_output_usd"] == 0.2 and s["no_output_by_stop_reason"] == {"never_started": 0.2}
    assert s["busy_share_of_working_machines"] == 0.5            # 15 of 30 billed minutes


def test_a_machine_keeps_the_attempt_limit_it_was_launched_with(conn, policy):
    # 2026-11-20: the GitHub backstop runs main's policy, which had no limit for a new family (talk), so it
    # stopped a talk machine at the 20-minute default while it downloaded 94 GB of weights. Like its deadline
    # and spend cap, a machine's attempt limit is recorded when it is registered, and every enforcer uses it.
    register(conn, "mw-talk-4090-a", kinds=("talk.infinitetalk",), hours=12, max_usd=100)
    conn.execute("UPDATE gpu_groups SET family = 'talk', max_attempt_minutes = 90 WHERE name = 'mw-talk-4090-a'")
    job = queue.enqueue(conn, "talk.infinitetalk", {})
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, started_at, gpu_group) "
                 "VALUES (%s, 'w1', 'salad_batch_4090', %s, 'mw-talk-4090-a')", (job, T0))
    old = policy.body["gpu_runs"]
    import dataclasses
    stale = dataclasses.replace(policy, body={**policy.body, "gpu_runs": {
        **old, "max_attempt_minutes_by_family": {"default": 20}}})   # an enforcer that never heard of talk
    api = FakeAPI(["mw-talk-4090-a"])
    for m in (30, 89, 91):
        conn.execute("UPDATE jobs SET status = 'running', worker_id = 'w1', updated_at = %s WHERE id = %s",
                     (T0 + timedelta(minutes=m), job))
        actions = tick(conn, stale, api, now=T0 + timedelta(minutes=m))
        if m < 90:
            assert api.stopped == [], m
    assert api.stopped == ["mw-talk-4090-a"] and actions[0]["reason"] == "attempt_timeout"


def test_the_launcher_records_the_familys_attempt_limit(conn, policy):
    from manyworlds.deploy.autoscale import attempt_limit_minutes
    assert attempt_limit_minutes(policy, "talk") == policy.get("gpu_runs", "max_attempt_minutes_by_family")["talk"]
    assert attempt_limit_minutes(policy, "nothing") == policy.get("gpu_runs", "max_attempt_minutes_by_family")["default"]
