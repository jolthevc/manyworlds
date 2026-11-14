"""Machines follow the backlog: one for a small batch, more for a big one, ramping one per
tick and never past the policy's ceiling."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from manyworlds import queue
from manyworlds.deploy import autoscale

from test_autoscale import providers

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setenv("MW_REGISTRY", "docker.io/example")
    monkeypatch.setattr(autoscale, "image_published", lambda image: True)


def launched(actions):
    return [a for a in actions if a["action"] == "launched"]


def enqueue(conn, n, kind="image.zimage"):
    for i in range(n):
        queue.enqueue(conn, kind, {"i": i})


def test_a_small_batch_gets_one_machine(conn, policy):
    enqueue(conn, 5)
    ps = providers()
    for _ in range(3):
        autoscale.scale_up(conn, policy, ps, now=NOW)
    assert len(ps["salad"].launched) + len(ps["vast"].launched) == 1


def test_a_big_backlog_ramps_up_one_machine_per_tick_to_the_ceiling(conn, policy):
    enqueue(conn, 2000)
    ps = providers()
    counts = []
    for _ in range(10):
        counts.append(len(launched(autoscale.scale_up(conn, policy, ps, now=NOW))))
    ceiling = policy.get("autoscale", "max_groups_per_kind")["image"]
    assert counts[:ceiling] == [1] * ceiling and sum(counts) == ceiling


def test_wanted_machines_use_measured_job_time(conn, policy):
    drain = policy.get("autoscale", "scale", "drain_target_minutes")["default"] * 60
    job = queue.enqueue(conn, "image.zimage", {})
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, outcome, gpu_seconds, started_at) "
                 "VALUES (%s, 'w', 'vast_3090', 'done', 50, now())", (job,))
    # 50 s per job: a drain target's worth of work is drain/50 jobs per machine.
    per_machine = drain // 50
    assert autoscale.machines_wanted(conn, policy, "image.zimage", per_machine) == 1
    assert autoscale.machines_wanted(conn, policy, "image.zimage", per_machine + 1) == 2


def test_control_host_work_never_rents_a_gpu(conn, policy):
    import dataclasses
    body = {**policy.body, "job_kinds": {**policy.body["job_kinds"], "ctl.thing": {"pools": ["local"]}}}
    policy = dataclasses.replace(policy, body=body)
    enqueue(conn, 50, kind="ctl.thing")
    ps = providers()
    actions = autoscale.scale_up(conn, policy, ps, now=NOW)
    assert not launched(actions) and not any(a["action"] == "no_capacity" for a in actions)
