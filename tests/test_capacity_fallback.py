"""Work no pool has capacity for (2026-11-25): reported to the owner, and after a while widened to the kind's other
pools on the same card (deploy/autoscale.capacity_fallback). Benchmarks keep their pins; another card is never chosen."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from manyworlds import queue
from manyworlds.deploy import autoscale

from test_autoscale import FakeProvider, providers, registry  # noqa: F401 - registry: autouse image registry stub

NOW = datetime.now(timezone.utc)
VOL = "runpod_secure_l40s_vol"
L40S = "runpod_secure_l40s"


def _pools(conn, job):
    return conn.execute("SELECT pools FROM jobs WHERE id = %s", (job,)).fetchone()["pools"]


def _aged(conn, job, minutes):
    conn.execute("UPDATE jobs SET created_at = %s WHERE id = %s", (NOW - timedelta(minutes=minutes), job))


def test_no_capacity_is_reported_and_then_widened_to_the_same_card(conn, policy):
    job = queue.enqueue(conn, "talk.infinitetalk", {}, batch="test-breadth-b", pools=[VOL])
    ps = providers(runpod=FakeProvider("runpod", capacity=False))
    _aged(conn, job, 5)                       # fresh: no report yet
    autoscale.scale_up(conn, policy, ps, now=NOW)
    assert conn.execute("SELECT count(*) n FROM events WHERE type = 'no_capacity'").fetchone()["n"] == 0
    _aged(conn, job, 20)                      # reported, not yet widened
    autoscale.scale_up(conn, policy, ps, now=NOW)
    ev = conn.execute("SELECT subject, payload FROM events WHERE type = 'no_capacity'").fetchall()
    assert len(ev) == 1 and ev[0]["subject"]["kind"] == "talk.infinitetalk" and ev[0]["payload"]["waiting"] == 1
    assert _pools(conn, job) == [VOL]
    _aged(conn, job, 45)                      # widened to the kind's other L40S pools; told once a day only
    actions = autoscale.scale_up(conn, policy, ps, now=NOW)
    fb = [a for a in actions if a.get("action") == "capacity_fallback"]
    assert fb and fb[0]["added"] == [L40S]
    assert _pools(conn, job) == [VOL, L40S]
    assert conn.execute("SELECT count(*) n FROM events WHERE type = 'no_capacity'").fetchone()["n"] == 1


def test_a_pin_on_a_card_with_no_twin_is_reported_but_never_moved_to_another_card(conn, policy):
    # 2026-11-25: talk jobs pinned to vast_5090 waited an hour for an offer; a 4090 or an L40S is not the same test
    job = queue.enqueue(conn, "talk.infinitetalk", {}, batch="test-breadth-b", pools=["vast_5090"])
    _aged(conn, job, 90)
    actions = autoscale.scale_up(conn, policy, providers(vast=FakeProvider("vast", capacity=False)), now=NOW)
    assert [a for a in actions if a.get("action") == "no_capacity"]
    assert not [a for a in actions if a.get("action") == "capacity_fallback"]
    assert _pools(conn, job) == ["vast_5090"]
    assert conn.execute("SELECT count(*) n FROM events WHERE type = 'no_capacity'").fetchone()["n"] == 1


def test_a_benchmark_without_capacity_is_reported_not_widened(conn, policy):
    job = queue.enqueue(conn, "talk.infinitetalk", {}, batch="bench-talk.infinitetalk-x", pools=[VOL])
    _aged(conn, job, 90)
    autoscale.scale_up(conn, policy, providers(runpod=FakeProvider("runpod", capacity=False)), now=NOW)
    assert _pools(conn, job) == [VOL]
    assert conn.execute("SELECT count(*) n FROM events WHERE type = 'no_capacity'").fetchone()["n"] == 1
