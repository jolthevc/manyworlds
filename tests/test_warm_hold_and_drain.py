"""The GPU factory keeps a warm machine for work due soon, drains instead of killing at the daily cap, and puts heavy
families on the weights volume (docs/research/2026-11-25-gpu-factory-audit.md, items d, e and 6)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds import ledger, policy as policy_mod, queue
from manyworlds.deploy import autoscale, enforce, report
from manyworlds.worker.runner import Worker

from test_autoscale import FakeProvider, providers, registry  # noqa: F401 - registry: autouse image registry stub
from test_enforce import FakeAPI
from test_production import creator

NOW = datetime.now(timezone.utc)
VOL, VOL2, L40S = "runpod_secure_l40s_vol", "runpod_secure_l40s_vol2", "runpod_secure_l40s"


def machine(conn, name="mw-talk-l40s-runpod-w1", *, kinds=("talk.infinitetalk",), pool=L40S, family="talk",
            started=None, last_job_min=3, in_flight=False, max_usd=3.0, billed=0.2, hours=2):
    """A running RunPod machine that finished a job `last_job_min` minutes ago (and has one open, if in_flight)."""
    started = started or NOW - timedelta(minutes=30)
    conn.execute(
        """INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, replicas, job_kinds, deadline, max_usd, started_at,
                                   billed_until, last_busy_at, family, billed_usd, image)
           VALUES (%s, 'runpod', %s, 1.09, 1, %s, %s, %s, %s, %s, %s, %s, %s, 'mw-worker-talkcomfy:v24')""",
        (name, pool, list(kinds), started + timedelta(hours=hours), max_usd, started, NOW, NOW, family, billed))
    job = queue.enqueue(conn, kinds[0], {}, batch="test-breadth-r1")
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job,))
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, outcome, gpu_group, started_at, finished_at) "
                 "VALUES (%s, 'w', %s, 'done', %s, %s, %s)",
                 (job, pool, name, NOW - timedelta(minutes=last_job_min + 4), NOW - timedelta(minutes=last_job_min)))
    if in_flight:
        job = queue.enqueue(conn, kinds[0], {}, batch="test-breadth-r1")
        conn.execute("UPDATE jobs SET status = 'running', worker_id = 'w' WHERE id = %s", (job,))
        conn.execute("INSERT INTO attempts (job_id, worker_id, pool, gpu_group, started_at) VALUES (%s, 'w', %s, %s, %s)",
                     (job, pool, name, NOW - timedelta(minutes=2)))
    return name


def producing_test_post(conn, fmt="talking_video"):
    cid = creator(conn)
    return conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, %s, 'producing', %s) "
                        "RETURNING id", (cid, fmt, Jsonb({"breadth_run": "r1"}))).fetchone()["id"]


def hold_of(conn, name):
    return conn.execute("SELECT keep_warm_until FROM gpu_groups WHERE name = %s", (name,)).fetchone()["keep_warm_until"]


def hold_minutes(policy, family):
    return policy.get("autoscale", "warm_hold", "max_hold_minutes")[family]


# --- warm hold ---------------------------------------------------------------------------------------------------------

def test_a_machine_is_held_while_a_test_post_between_stages_will_need_its_family(conn, policy):
    name = machine(conn, last_job_min=3)
    assert autoscale.warm_holds(conn, policy, NOW) == [] and hold_of(conn, name) is None   # nothing due: no hold
    producing_test_post(conn, "talking_video")
    acts = autoscale.warm_holds(conn, policy, NOW)
    assert acts and acts[0]["why"] == "producing"
    last = NOW - timedelta(minutes=3)
    assert abs((hold_of(conn, name) - (last + timedelta(minutes=hold_minutes(policy, "talk")))).total_seconds()) < 1
    # a photo post never needs a talk machine
    conn.execute("UPDATE posts SET format = 'photo'")
    autoscale.warm_holds(conn, policy, NOW)
    assert hold_of(conn, name) is None


def test_a_queued_job_it_may_claim_holds_it_but_one_pinned_elsewhere_does_not(conn, policy):
    name = machine(conn)
    queue.enqueue(conn, "talk.infinitetalk", {}, batch="test-x", pools=["vast_4090"])
    autoscale.warm_holds(conn, policy, NOW)
    assert hold_of(conn, name) is None
    queue.enqueue(conn, "talk.infinitetalk", {}, batch="test-x", pools=[L40S])
    assert autoscale.warm_holds(conn, policy, NOW)[0]["why"] == "queued_jobs"
    # while paused, production work is not served: it holds nothing
    paused = policy_mod.from_dict({**policy.body, "operations": {**policy.body["operations"], "mode": "paused"}})
    conn.execute("DELETE FROM jobs WHERE status = 'queued'")
    queue.enqueue(conn, "talk.infinitetalk", {}, batch="post-abc")
    autoscale.warm_holds(conn, paused, NOW)
    assert hold_of(conn, name) is None
    autoscale.warm_holds(conn, policy, NOW)                                          # live: it would claim it
    assert hold_of(conn, name) is not None


def test_the_hold_is_capped_by_the_spend_cap_and_deadline_and_off_while_draining_or_disabled(conn, policy):
    producing_test_post(conn)
    rich = machine(conn, "mw-a", last_job_min=1)
    poor = machine(conn, "mw-b", last_job_min=1, max_usd=1.0, billed=0.98)       # ~1 min of money left
    late = machine(conn, "mw-c", last_job_min=1, started=NOW - timedelta(hours=2) + timedelta(minutes=4))
    drained = machine(conn, "mw-d", last_job_min=1)
    conn.execute("UPDATE gpu_groups SET draining_since = %s WHERE name = 'mw-d'", (NOW,))
    autoscale.warm_holds(conn, policy, NOW)
    assert hold_of(conn, rich) > NOW + timedelta(minutes=10)
    assert NOW < hold_of(conn, poor) < NOW + timedelta(minutes=2)
    assert hold_of(conn, late) == NOW + timedelta(minutes=4)                         # its deadline
    assert hold_of(conn, drained) is None
    off = policy_mod.from_dict({**policy.body, "autoscale": {
        **policy.body["autoscale"], "warm_hold": {**policy.body["autoscale"]["warm_hold"], "enabled": False}}})
    autoscale.warm_holds(conn, off, NOW)
    assert hold_of(conn, rich) is None


def test_no_hold_once_the_days_gpu_budget_is_reached(conn, policy):
    producing_test_post(conn)
    name = machine(conn)
    ledger.record(conn, category="gpu_billed", provider="x", quantity=1, unit="instance_second", usd=50,
                  policy_version=policy.version)
    autoscale.warm_holds(conn, policy, NOW)
    assert hold_of(conn, name) is None


def test_the_enforcer_and_the_worker_keep_a_held_idle_machine_then_stop_it(conn, policy, store, database_url):
    name = machine(conn, last_job_min=8)
    conn.execute("UPDATE gpu_groups SET last_busy_at = %s, keep_warm_until = %s WHERE name = %s",
                 (NOW - timedelta(minutes=8), NOW + timedelta(minutes=5), name))
    api = FakeAPI([name])
    acts = enforce.tick(conn, policy, api, now=NOW)
    assert api.stopped == [] and acts[0]["action"] == "ok"
    w = Worker(handlers={}, store=store, policy=policy, pool=L40S, database_url=database_url, group=name,
               self_destroy=lambda: "runpod")
    w._idle_since = time.monotonic() - 8 * 60
    assert w.self_stop_reason(now=NOW) is None                                       # held past its 5 idle minutes
    assert w.self_stop_reason(now=NOW + timedelta(minutes=6)) == "idle"              # the hold has passed
    enforce.tick(conn, policy, api, now=NOW + timedelta(minutes=6))
    assert api.stopped == [name]


# --- the daily cap drains --------------------------------------------------------------------------------------------

def over_cap(conn, policy, extra=0.0):
    cap = policy.get("gpu_runs", "daily_gpu_usd")
    ledger.record(conn, category="gpu_billed", provider="x", quantity=1, unit="instance_second", usd=cap + extra,
                  policy_version=policy.version)


def test_at_the_daily_cap_a_machine_mid_job_drains_and_an_idle_one_stops(conn, policy):
    busy = machine(conn, "mw-busy", in_flight=True)
    idle = machine(conn, "mw-idle")
    over_cap(conn, policy)
    api = FakeAPI([busy, idle])
    acts = {a["group"]: a for a in enforce.tick(conn, policy, api, now=NOW)}
    assert acts[idle]["action"] == "stopped" and acts[idle]["reason"] == "daily_gpu_budget"
    assert acts[busy]["action"] == "draining" and api.stopped == [idle]
    assert conn.execute("SELECT draining_since FROM gpu_groups WHERE name = %s", (busy,)).fetchone()["draining_since"]
    # its job ends: stopped on the next tick
    conn.execute("UPDATE attempts SET finished_at = %s, outcome = 'done' WHERE gpu_group = %s AND finished_at IS NULL",
                 (NOW + timedelta(minutes=1), busy))
    enforce.tick(conn, policy, api, now=NOW + timedelta(minutes=2))
    assert api.stopped == [idle, busy]


def test_a_draining_machine_is_stopped_at_the_grace_ceiling(conn, policy):
    busy = machine(conn, "mw-busy", in_flight=True)
    over_cap(conn, policy)
    api = FakeAPI([busy])
    enforce.tick(conn, policy, api, now=NOW)
    grace = policy.get("gpu_runs", "daily_drain_grace_minutes")
    enforce.tick(conn, policy, api, now=NOW + timedelta(minutes=grace - 1))
    assert api.stopped == []
    acts = enforce.tick(conn, policy, api, now=NOW + timedelta(minutes=grace + 1))
    assert api.stopped == [busy] and acts[0]["reason"] == "daily_gpu_budget"


def test_a_draining_machine_is_stopped_once_spend_passes_the_overshoot(conn, policy):
    busy = machine(conn, "mw-busy", in_flight=True)
    over_cap(conn, policy, extra=policy.get("gpu_runs", "daily_drain_max_overshoot_usd") + 0.01)
    api = FakeAPI([busy])
    enforce.tick(conn, policy, api, now=NOW)
    assert api.stopped == [busy]


def test_without_a_grace_the_cap_stops_at_once(conn, policy):
    pol = policy_mod.from_dict({**policy.body, "gpu_runs": {
        k: v for k, v in policy.body["gpu_runs"].items() if k != "daily_drain_grace_minutes"}})
    busy = machine(conn, "mw-busy", in_flight=True)
    over_cap(conn, pol)
    api = FakeAPI([busy])
    enforce.tick(conn, pol, api, now=NOW)
    assert api.stopped == [busy]


def test_a_draining_worker_claims_nothing_and_stops_its_machine(conn, policy, store, database_url):
    name = machine(conn, "mw-fake-drain", kinds=("fake.image",), pool="local", family="fake")
    queue.enqueue(conn, "fake.image", {"n": 1}, batch="test-x")
    conn.execute("UPDATE gpu_groups SET draining_since = %s WHERE name = %s", (NOW, name))
    from manyworlds.worker.runner import handlers_for
    w = Worker(handlers=handlers_for(["fake.image"]), store=store, policy=policy, pool="local",
               database_url=database_url, group=name, self_destroy=lambda: "runpod")
    assert w.run(idle_exit_seconds=0.2, poll_seconds=0.05) == 0
    w.stop()                                                                         # ends its watchdog thread
    assert conn.execute("SELECT count(*) n FROM jobs WHERE status = 'queued'").fetchone()["n"] == 1
    assert w.self_stop_reason(now=NOW) == "draining"
    w._busy = True
    assert w.self_stop_reason(now=NOW) is None                                       # mid-job: it finishes first
    # a new UTC day under the cap: the flag is cleared and the worker claims again
    enforce.tick(conn, policy, FakeAPI([name]), now=NOW)
    assert not w.draining()


# --- weights volumes -------------------------------------------------------------------------------------------------

def reading(conn, pool, free_gb, name="mw-reader"):
    conn.execute("INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd, family, "
                 "started_at, stopped_at, stop_reason, volume_free_gb, volume_low, volume_checked_at) "
                 "VALUES (%s, 'runpod', %s, 1.09, '{talk.infinitetalk}', %s, 1, 'talk', %s, %s, 'idle', %s, false, %s)",
                 (name, pool, NOW, NOW - timedelta(hours=1), NOW - timedelta(minutes=30), free_gb,
                  NOW - timedelta(minutes=50)))


def test_tts_runs_on_the_volume_first_with_the_plain_pool_behind_it(policy):
    from manyworlds import breadth
    pol = breadth.test_policy(policy)
    assert breadth._cfg(pol)["pools_by_family"]["tts"][:2] == [VOL, L40S]
    for kind in ("voice.speak.indextts", "voice.speak.qwen3", "voice.cast"):
        assert policy.get("job_kinds", kind, "pools")[0] == VOL


def test_a_volume_not_created_is_skipped_and_never_launched(conn, policy):
    assert autoscale.volume_skip(conn, policy, VOL2, ["talk"]) == "volume_not_created"
    assert autoscale.volume_skip(conn, policy, L40S, ["talk"]) is None
    queue.enqueue(conn, "talk.infinitetalk", {}, batch="bench-talk", pools=[VOL2, L40S])
    ps = providers()
    acts = autoscale.scale_up(conn, policy, ps, now=NOW)
    assert [a["pool"] for a in acts if a.get("action") == "launched"] == [L40S]


def test_tts_goes_to_the_plain_pool_while_the_volume_lacks_room_for_its_weights(conn, policy):
    floor = policy.get("gpu_pools", VOL, "network_volume", "min_free_gb")
    reading(conn, VOL, floor + 5)                                 # above the floor, not enough for tts's weights
    assert autoscale.volume_skip(conn, policy, VOL, ["tts"]).startswith("volume_headroom")
    queue.enqueue(conn, "voice.speak.indextts", {}, batch="bench-tts", pools=[VOL, L40S])
    acts = autoscale.scale_up(conn, policy, providers(), now=NOW)
    assert [a["pool"] for a in acts if a.get("action") == "launched"] == [L40S]
    # once tts has finished a job there its weights are cached: the volume is fine for it again
    conn.execute("INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd, family) "
                 "VALUES ('mw-tts-old', 'runpod', %s, 1.09, '{voice.cast}', %s, 1, 'tts')", (VOL, NOW))
    job = queue.enqueue(conn, "voice.cast", {}, batch="old")
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, gpu_group, outcome) VALUES (%s, 'w', %s, "
                 "'mw-tts-old', 'done')", (job, VOL))
    assert autoscale.volume_skip(conn, policy, VOL, ["tts"]) is None


def test_two_volumes_are_tried_holding_the_weights_first_then_by_free_space(conn, policy):
    body = dict(policy.body)
    pools = dict(body["gpu_pools"])
    pools[VOL2] = {**pools[VOL2], "network_volume": {**pools[VOL2]["network_volume"], "id": "vol2id",
                                                     "data_center": "EU-RO-1"}}
    pol = policy_mod.from_dict({**body, "gpu_pools": pools})
    order = [VOL, VOL2, L40S]
    assert autoscale.volume_order(conn, pol, order, ["talk"]) == order          # no readings: as given
    reading(conn, VOL, 20, "mw-r1")
    reading(conn, VOL2, 90, "mw-r2")
    assert autoscale.volume_order(conn, pol, order, ["talk"]) == [VOL2, VOL, L40S]   # more room first
    job = queue.enqueue(conn, "talk.infinitetalk", {}, batch="old")
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, gpu_group, outcome) VALUES (%s, 'w', %s, 'mw-r1', "
                 "'done')", (job, VOL))
    assert autoscale.volume_order(conn, pol, order, ["talk"]) == [VOL, VOL2, L40S]   # its weights are there


def test_machines_report_shows_each_machines_worker_image(conn, policy):
    machine(conn, "mw-tagged")
    rows = report.machines(conn, hours=2, now=NOW)
    assert [r["image"] for r in rows if r["name"] == "mw-tagged"] == ["mw-worker-talkcomfy:v24"]
