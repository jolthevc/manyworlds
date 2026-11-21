"""The master switch: paused (the default in policy) runs nothing on its own; only tests and benchmarks run."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

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
    out = cli.produce_tick(conn, _paused(policy))
    assert out == {"operations": "paused"} and called == []
