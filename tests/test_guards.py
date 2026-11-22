"""Guards that must hold whatever else runs: a worker that can't reach the queue gives up its machine, raw
research items expire, and no machine starts once the day's whole bill reaches budgets.daily_usd."""

from __future__ import annotations

from datetime import timedelta

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


def _stopped_machine(conn, policy, name, *, outcome, image=None, minutes_ago=30):
    img = policy.get("worker_images", "talk")
    conn.execute(
        "INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd, started_at, "
        "stopped_at, stop_reason, family, image) VALUES (%s, 'runpod', 'runpod_secure_a40', 0.6, "
        "'{talk.infinitetalk}', %s, 1, %s, %s, 'idle', 'talk', %s)",
        (name, T0, T0 - timedelta(minutes=minutes_ago + 30), T0 - timedelta(minutes=minutes_ago),
         image or f"{img['image']}:{img['tag']}"))
    job = queue.enqueue(conn, "fake.image", {"n": 1}, batch=f"old-{name}")
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, gpu_group, outcome) VALUES (%s, 'w', "
                 "'runpod_secure_a40', %s, %s)", (job, name, outcome))


def test_a_family_whose_last_machines_all_failed_stops_renting_and_tells_the_owner(conn, policy):
    n = policy.get("autoscale", "no_output_guard", "machines")
    for i in range(n):
        _stopped_machine(conn, policy, f"mw-talk-a40-runpod-{i}", outcome="failed", minutes_ago=30 + i)
    queue.enqueue(conn, "talk.infinitetalk", {}, batch="bench-talk", pools=["runpod_secure_a40"])
    ps = providers()
    actions = autoscale.tick(conn, policy, ps, now=T0)
    assert not ps["runpod"].launched
    held = [a for a in actions if a.get("reason") == "no_output_streak"]
    assert held and len(held[0]["machines"]) == n
    from manyworlds import notify
    assert notify.from_autoscaler(conn, policy, actions, T0) == 1


def test_one_finished_job_or_a_new_worker_image_rearms_the_family(conn, policy):
    n = policy.get("autoscale", "no_output_guard", "machines")
    for i in range(n - 1):
        _stopped_machine(conn, policy, f"mw-talk-a40-runpod-{i}", outcome="failed", minutes_ago=30 + i)
    _stopped_machine(conn, policy, "mw-talk-a40-runpod-ok", outcome="done", minutes_ago=60)
    assert autoscale.no_output_streak(conn, policy, "talk", T0) is None
    conn.execute("TRUNCATE attempts, gpu_groups CASCADE")
    for i in range(n):
        _stopped_machine(conn, policy, f"mw-talk-old-{i}", outcome="failed", image="mw-worker-talk:v1",
                         minutes_ago=30 + i)
    assert autoscale.no_output_streak(conn, policy, "talk", T0) is None      # a new image gets its own try


def test_fixed_monthly_costs_accrue_to_the_ledger_once_a_day(conn, policy):
    items = policy.get("fixed_costs")
    assert items and autoscale.housekeeping(conn, policy) == []
    autoscale.housekeeping(conn, policy)                                   # same day: not twice
    rows = conn.execute("SELECT provider, usd FROM cost_entries WHERE category = 'fixed'").fetchall()
    assert len(rows) == len(items)
    assert float(rows[0]["usd"]) == pytest.approx(items[0]["usd_per_month"] / 30, abs=1e-6)
    assert ledger.bill_today(conn) >= float(rows[0]["usd"])                # on the bill and under the daily caps


def test_failures_on_one_card_do_not_block_a_different_one(conn, policy):
    # 2026-11-22: three L40S talk machines ran out of memory; an 80 GB H100 is a different question.
    n = policy.get("autoscale", "no_output_guard", "machines")
    for i in range(n):
        _stopped_machine(conn, policy, f"mw-talk-a40-runpod-{i}", outcome="failed", minutes_ago=30 + i)
    assert autoscale.no_output_streak(conn, policy, "talk", T0, ["runpod_secure_a40"])
    assert autoscale.no_output_streak(conn, policy, "talk", T0, ["runpod_secure_h100"]) is None
