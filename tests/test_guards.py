"""Guards that must hold whatever else runs: a worker that can't reach the queue gives up its machine, raw
research items expire, and no machine starts once the day's whole bill reaches budgets.daily_usd."""

from __future__ import annotations

import pytest

from manyworlds import db, ledger, queue
from manyworlds.deploy import autoscale
from manyworlds.worker import runner

from test_autoscale import T0, providers, registry  # noqa: F401 - registry: autouse image registry stub


def test_a_worker_that_cannot_reach_the_queue_destroys_its_machine(policy, store, monkeypatch):
    monkeypatch.setenv("MW_GROUP", "mw-image-3090-runpod-x")
    tries, destroyed = [], []

    def unreachable(url=None, timeout=None):
        tries.append(timeout)
        raise OSError("connection timed out")
    with pytest.raises(SystemExit) as e:
        runner.start(["fake.image"], pool="runpod_comm_3090", policy=policy, store=store, connect=unreachable,
                     self_destroy=lambda: destroyed.append(1) or "runpod")
    assert e.value.code == 2 and destroyed == [1]
    assert len(tries) == policy.get("queue", "startup_connect_attempts")
    assert set(tries) == {policy.get("queue", "connect_timeout_seconds")}


def test_a_worker_that_reaches_the_queue_runs(conn, database_url, policy, store):
    job = queue.enqueue(conn, "fake.image", {"prompt": "x", "n": 1})
    w = runner.start(["fake.image"], pool="local", policy=policy, store=store,
                     connect=lambda url=None, timeout=None: db.connect(database_url, timeout=timeout))
    assert w.run(max_jobs=1) == 1
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job,)).fetchone()["status"] == "done"


def test_a_gpu_pool_worker_refuses_to_run_outside_a_registered_machine(policy, store, monkeypatch):
    monkeypatch.delenv("MW_GROUP", raising=False)
    with pytest.raises(SystemExit, match="MW_GROUP"):
        runner.start(["fake.image"], pool="runpod_comm_3090", policy=policy, store=store)


def test_expired_research_items_are_purged_by_the_always_on_loop(conn):
    conn.execute("INSERT INTO research_items (platform, external_id, purge_after) VALUES "
                 "('instagram', 'old', now() - interval '1 day'), ('instagram', 'new', now() + interval '29 days')")
    assert autoscale.housekeeping(conn) == [{"action": "purged_research_items", "items": 1}]
    assert [r["external_id"] for r in conn.execute("SELECT external_id FROM research_items")] == ["new"]
    assert autoscale.housekeeping(conn) == []


def test_no_machine_starts_once_the_days_whole_bill_reaches_the_budget(conn, policy):
    ledger.record(conn, category="llm", provider="claude-opus-5-5", quantity=1, unit="token",
                  usd=float(policy.get("budgets", "daily_usd")))
    queue.enqueue(conn, "image.zimage", {}, batch="bench-x")
    ps = providers()
    actions = autoscale.tick(conn, policy, ps, now=T0)
    assert not [p for p in ps.values() if p.launched]
    assert any(a.get("reason") == "daily_total_budget" for a in actions)


def test_gpu_attribution_entries_are_not_counted_twice(conn, policy):
    ledger.record(conn, category="gpu", provider="runpod_3090", quantity=1, unit="gpu_second", usd=1000)
    assert ledger.bill_today(conn) == 0
