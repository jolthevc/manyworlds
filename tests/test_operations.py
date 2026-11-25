"""The master switch: paused (the default in policy) runs nothing on its own; only tests and benchmarks run."""

from __future__ import annotations

import dataclasses

from manyworlds import operations, queue
from manyworlds.brain import runtime
from manyworlds.deploy import autoscale

from brain_helpers import StubLLM
from test_autoscale import T0, providers


def _paused(policy):
    return dataclasses.replace(policy, body={**policy.body, "operations": {**policy.body["operations"], "mode": "paused"}})


def test_the_shipped_policy_is_paused(raw_policy):
    assert operations.mode(raw_policy) == "paused"


def test_paused_brain_runs_only_test_jobs(conn, policy):
    pol = _paused(policy)
    real = queue.enqueue(conn, "brain.echo", {"say": "hi"})
    test = queue.enqueue(conn, "brain.echo", {"say": "hi", "test": True})
    out = {r["job"]: r["status"] for r in runtime.tick(conn, pol, StubLLM(conn=conn), worker_id="w1")}
    assert out == {test: "done"}
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (real,)).fetchone()["status"] == "queued"


def test_paused_autoscaler_starts_machines_only_for_test_batches(conn, policy, monkeypatch):
    monkeypatch.setenv("MW_REGISTRY", "docker.io/example")
    monkeypatch.setattr(autoscale, "image_published", lambda image: True)
    pol = _paused(policy)
    queue.enqueue(conn, "image.zimage", {}, batch="post-123")              # production work waits
    ps = providers()
    actions = autoscale.tick(conn, pol, ps, now=T0)
    assert not [a for a in actions if a.get("action") == "launched"]
    queue.enqueue(conn, "image.zimage", {}, batch="bench-image.zimage-x")  # a benchmark may run
    actions = autoscale.tick(conn, pol, ps, now=T0)
    assert [a for a in actions if a.get("action") == "launched"]


def test_paused_producer_does_nothing(conn, policy, monkeypatch):
    from manyworlds import cli, production
    called = []
    monkeypatch.setattr(production, "advance", lambda *a, **k: called.append("advance") or {})
    quiet = _paused(policy)
    quiet = dataclasses.replace(quiet, body={**quiet.body, "operations": {**quiet.body["operations"], "while_paused": []}})
    out = cli.produce_tick(conn, quiet)
    assert out == {"operations": "paused"} and called == []


def test_paused_producer_queues_only_the_research_the_owner_allows(conn, policy, monkeypatch):
    # owner 2026-11-25: research keeps learning during the build (operations.while_paused: research_monitor)
    from manyworlds import cli, production
    called = []
    monkeypatch.setattr(production, "advance", lambda *a, **k: called.append("advance") or {})
    out = cli.produce_tick(conn, _paused(policy))
    assert set(out["research"]) == {"monitor", "desk"} and called == []
    modes = sorted(r["payload"]["mode"] for r in conn.execute("SELECT payload FROM jobs"))
    assert modes == ["desk", "monitor"]
    cli.produce_tick(conn, _paused(policy))
    assert conn.execute("SELECT count(*) n FROM jobs").fetchone()["n"] == 2          # once a day


def test_paused_producer_still_stores_finished_casts(conn, policy, monkeypatch):
    # 2026-11-25: Maren's recast finished but sat unjudged while paused, so she kept her old voice.
    from manyworlds import cli, production
    from manyworlds.creator import voice_lab
    called = []
    monkeypatch.setattr(production, "advance", lambda *a, **k: called.append("advance") or {})
    monkeypatch.setattr(voice_lab, "settle_casts", lambda conn, pol: {"judged": 1, "picked": 1})
    out = cli.produce_tick(conn, _paused(policy))
    assert out["casts"] == {"judged": 1, "picked": 1} and called == []
