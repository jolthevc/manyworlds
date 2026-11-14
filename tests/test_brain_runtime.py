"""Agents run as brain.<agent> jobs on the queue: each run is traced, idempotent and budgeted,
and one broken job never stops the others."""

from __future__ import annotations

from manyworlds import ledger, queue
from manyworlds.brain import agent, runtime

from brain_helpers import StubLLM, make_ctx


def test_an_agent_job_runs_and_records_its_decision(conn, policy):
    job = queue.enqueue(conn, "brain.echo", {"say": "hi"})
    out = runtime.tick(conn, policy, StubLLM(conn=conn), worker_id="w1")
    assert out == [{"job": job, "agent": "echo", "status": "done"}]
    d = conn.execute("SELECT agent, job_id, step, output FROM decisions WHERE job_id = %s", (job,)).fetchone()
    assert d["agent"] == "echo" and d["output"]["say"] == "hi"
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job,)).fetchone()["status"] == "done"


def test_a_rerun_after_a_crash_does_not_duplicate_the_decision(conn, policy):
    job = queue.enqueue(conn, "brain.echo", {"say": "hi"})
    ctx = make_ctx(conn, policy, job)
    first = agent.record_decision(ctx, "echo", step="echo", inputs={}, output={}, rationale="r")
    again = agent.record_decision(ctx, "echo", step="echo", inputs={}, output={}, rationale="r")
    assert first == again
    assert conn.execute("SELECT count(*) n FROM decisions WHERE job_id = %s", (job,)).fetchone()["n"] == 1


def test_an_unknown_agent_fails_once_and_others_still_run(conn, policy):
    bad = queue.enqueue(conn, "brain.nobody", {})
    good = queue.enqueue(conn, "brain.echo", {})
    out = {r["job"]: r for r in runtime.tick(conn, policy, StubLLM(conn=conn), worker_id="w1")}
    assert out[bad]["status"] == "failed" and "no agent named 'nobody'" in out[bad]["error"]
    assert out[good]["status"] == "done"
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (bad,)).fetchone()["status"] == "failed"


def test_an_agent_over_budget_fails_without_retry(conn, policy):
    ledger.record(conn, category="llm", provider="m", quantity=1, unit="token", usd=10_000, agent="echo")
    job = queue.enqueue(conn, "brain.echo", {})
    out = runtime.tick(conn, policy, StubLLM(conn=conn), worker_id="w1")
    assert out[0]["status"] == "failed" and "budget" in out[0]["error"]
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job,)).fetchone()["status"] == "failed"


def test_an_agent_that_crashes_is_retried(conn, policy, monkeypatch):
    job = queue.enqueue(conn, "brain.echo", {})

    def boom(ctx):
        raise RuntimeError("transient")
    monkeypatch.setattr(agent.get("echo"), "run", boom)
    out = runtime.tick(conn, policy, StubLLM(conn=conn), worker_id="w1")
    assert out[0]["status"] == "retry"
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job,)).fetchone()["status"] == "queued"


def test_llm_calls_inside_an_agent_are_charged_to_it(conn, policy):
    job = queue.enqueue(conn, "brain.echo", {})
    runtime.tick(conn, policy, StubLLM(cost=0.01, conn=conn), worker_id="w1")
    row = conn.execute("SELECT agent, job_id, usd FROM cost_entries").fetchone()
    assert row["agent"] == "echo" and row["job_id"] == job and float(row["usd"]) == 0.01


def test_commissioned_work_is_linked_and_not_duplicated(conn, policy):
    job = queue.enqueue(conn, "brain.echo", {})
    ctx = make_ctx(conn, policy, job)
    a = agent.commission(ctx, "echo", {"q": 1}, reason="why", key="k")
    b = agent.commission(ctx, "echo", {"q": 1}, reason="why", key="k")
    row = conn.execute("SELECT kind, parent_job_id, payload FROM jobs WHERE id = %s", (a,)).fetchone()
    assert a == b and row["kind"] == "brain.echo" and row["parent_job_id"] == job
    assert row["payload"]["requested_by"] == "echo" and row["payload"]["reason"] == "why"


def test_other_job_kinds_are_left_alone(conn, policy):
    other = queue.enqueue(conn, "fake.image", {})
    assert runtime.tick(conn, policy, StubLLM(conn=conn), worker_id="w1") == []
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (other,)).fetchone()["status"] == "queued"
