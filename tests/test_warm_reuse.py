"""Warm reuse (2026-11-25): pinned test work goes to a running machine with its kind on the same card before a new one
is rented (deploy/autoscale.route_to_warm). Benchmarks and production work keep their pins."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from manyworlds import queue
from manyworlds.deploy import autoscale

from test_autoscale import providers, registry  # noqa: F401 - registry: autouse image registry stub

NOW = datetime.now(timezone.utc)
VOL = "runpod_secure_l40s_vol"
L40S = "runpod_secure_l40s"


def _warm(conn, name, pool, *, kinds=("talk.infinitetalk",), deadline_min=120, billed=0.2, max_usd=3.0):
    conn.execute("INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd, family, "
                 "started_at, billed_until, last_busy_at, billed_usd) "
                 "VALUES (%s, 'runpod', %s, 1.09, %s, %s, %s, 'talk', %s, %s, %s, %s)",
                 (name, pool, list(kinds), NOW + timedelta(minutes=deadline_min), max_usd,
                  NOW - timedelta(minutes=30), NOW, NOW, billed))


def _pools(conn, job):
    return conn.execute("SELECT pools FROM jobs WHERE id = %s", (job,)).fetchone()["pools"]


def test_pinned_test_work_goes_to_a_warm_machine_with_the_same_card(conn, policy):
    _warm(conn, "mw-talk-l40s-runpod-warm", L40S)
    job = queue.enqueue(conn, "talk.infinitetalk", {}, batch="test-breadth-b", pools=[VOL])
    ps = providers()
    actions = autoscale.scale_up(conn, policy, ps, now=NOW)
    routed = [a for a in actions if a.get("action") == "routed_to_warm"]
    assert routed and routed[0]["machine"] == "mw-talk-l40s-runpod-warm" and routed[0]["jobs"] == 1
    assert not ps["runpod"].launched
    assert _pools(conn, job) == [VOL, L40S]               # the pin stays first: a preference, not a lock
    claimed = queue.claim(conn, ["talk.infinitetalk"], "w", 60, pool=L40S)
    assert claimed is not None and claimed.id == job
    # next tick: served by the warm machine, nothing rented
    assert not [a for a in autoscale.scale_up(conn, policy, ps, now=NOW) if a.get("action") == "launched"]


def test_a_benchmark_keeps_its_pin_and_gets_its_own_machine(conn, policy):
    _warm(conn, "mw-talk-l40s-runpod-warm", L40S)
    job = queue.enqueue(conn, "talk.infinitetalk", {}, batch="bench-talk.infinitetalk-x", pools=[VOL])
    ps = providers()
    actions = autoscale.scale_up(conn, policy, ps, now=NOW)
    assert not [a for a in actions if a.get("action") == "routed_to_warm"]
    assert [a["pool"] for a in actions if a.get("action") == "launched"] == [VOL]
    assert _pools(conn, job) == [VOL]


def test_no_warm_reuse_on_another_card_near_its_deadline_or_cap(conn, policy):
    _warm(conn, "mw-talk-4090-runpod-a", "runpod_comm_4090")                       # another card
    _warm(conn, "mw-talk-l40s-runpod-b", L40S, deadline_min=5)                     # about to stop
    _warm(conn, "mw-talk-l40s-runpod-c", L40S, billed=2.9)                         # nearly at its cap
    job = queue.enqueue(conn, "talk.infinitetalk", {}, batch="test-breadth-b", pools=[VOL])
    actions = autoscale.scale_up(conn, policy, providers(), now=NOW)
    assert not [a for a in actions if a.get("action") == "routed_to_warm"]
    assert _pools(conn, job) == [VOL]


def test_a_machine_without_the_kind_is_not_warm_for_it(conn, policy):
    _warm(conn, "mw-voice-l40s-runpod-v", L40S, kinds=("voice.speak",))
    job = queue.enqueue(conn, "talk.infinitetalk", {}, batch="test-breadth-b", pools=[VOL])
    actions = autoscale.scale_up(conn, policy, providers(), now=NOW)
    assert not [a for a in actions if a.get("action") == "routed_to_warm"]
    assert _pools(conn, job) == [VOL]


def test_production_work_is_never_routed_to_a_machine_that_may_take_only_tests(conn, policy):
    # a worker built while operations were paused claims only test batches: production work stays on its own pins
    _warm(conn, "mw-talk-l40s-runpod-warm", L40S)
    job = queue.enqueue(conn, "talk.infinitetalk", {}, batch="post-abc", pools=[VOL])
    actions = autoscale.scale_up(conn, policy, providers(), now=NOW)
    assert not [a for a in actions if a.get("action") == "routed_to_warm"]
    assert _pools(conn, job) == [VOL]
