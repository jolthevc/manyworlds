"""A broken weights-volume pool falls back to the family's plain pools instead of stopping the whole family
(deploy/autoscale.scale_up and the no-output guard)."""

from __future__ import annotations

from datetime import timedelta

from manyworlds import queue
from manyworlds.deploy import autoscale

from test_autoscale import T0, providers, registry  # noqa: F401 - registry: autouse image registry stub
from test_volume import VOL_POOL, _machine


def _failed_machine(conn, policy, name, pool, minutes_ago):
    img = policy.get("worker_images", "talk")
    _machine(conn, name, pool=pool, stopped_at=T0 - timedelta(minutes=minutes_ago), stop_reason="attempt_timeout",
             image=f"{img['image']}:{img['tag']}")
    job = queue.enqueue(conn, "fake.image", {"n": 1}, batch=f"old-{name}")
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, gpu_group, outcome) VALUES (%s, 'w', %s, %s, "
                 "'worker_fault')", (job, pool, name))


def test_a_broken_volume_pool_falls_back_to_the_plain_pool_without_stopping_the_family(conn, policy):
    # 2026-11-24: three volume machines died of "Disk quota exceeded"; the guard then stopped every talk machine.
    n = policy.get("autoscale", "no_output_guard", "machines")
    for i in range(n):
        _failed_machine(conn, policy, f"mw-talk-l40s-runpod-f{i}", VOL_POOL, minutes_ago=30 + i)
    pins = [VOL_POOL, "runpod_secure_l40s"]
    queue.enqueue(conn, "talk.infinitetalk", {}, batch="bench-talk", pools=pins)
    ps = providers()
    actions = autoscale.scale_up(conn, policy, ps, now=T0 + timedelta(hours=1))
    assert [a for a in actions if a.get("action") == "volume_pools_skipped"]
    launched = [a for a in actions if a.get("action") == "launched"]
    assert launched and launched[0]["pool"] == "runpod_secure_l40s"


def test_the_family_still_stops_when_its_plain_pools_fail_too(conn, policy):
    n = policy.get("autoscale", "no_output_guard", "machines")
    for i in range(n):
        _failed_machine(conn, policy, f"mw-talk-l40s-runpod-f{i}", VOL_POOL, minutes_ago=30 + i)
        _failed_machine(conn, policy, f"mw-talk-l40s-runpod-p{i}", "runpod_secure_l40s", minutes_ago=60 + i)
    queue.enqueue(conn, "talk.infinitetalk", {}, batch="bench-talk", pools=[VOL_POOL, "runpod_secure_l40s"])
    ps = providers()
    actions = autoscale.scale_up(conn, policy, ps, now=T0 + timedelta(hours=1))
    assert not ps["runpod"].launched
    assert [a for a in actions if a.get("reason") == "no_output_streak"]
