"""Agents run as brain.<agent> jobs on the queue: each run is traced, idempotent and budgeted,
and one broken job never stops the others."""

from __future__ import annotations

from manyworlds import ledger, queue
from manyworlds.brain import agent, runtime

from manyworlds.brain.agents.echo import _Ack as Ack

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


# --- final review fixes -------------------------------------------------------------------------

def _short_lease(policy, seconds):
    import dataclasses
    brain = {**policy.body["brain"], "runtime": {**policy.body["brain"]["runtime"], "lease_seconds": seconds}}
    return dataclasses.replace(policy, body={**policy.body, "brain": brain})


def test_a_long_agent_keeps_its_lease_so_no_one_else_runs_it(conn, policy, monkeypatch):
    import time
    pol = _short_lease(policy, 1)
    job = queue.enqueue(conn, "brain.echo", {})
    stolen = []

    def slow(ctx):
        time.sleep(2.5)                                    # longer than the 1 s lease
        stolen.append(queue.claim(conn, ["brain.echo"], "intruder", 1, pool="local"))
        return {}
    monkeypatch.setattr(agent.get("echo"), "run", slow)
    out = runtime.tick(conn, pol, StubLLM(conn=conn), worker_id="w1")
    assert stolen == [None] and out[0]["status"] == "done"


def test_a_job_lost_to_another_worker_is_reported_lost_not_done(conn, policy, monkeypatch):
    job = queue.enqueue(conn, "brain.echo", {})

    def hijacked(ctx):
        conn.execute("UPDATE jobs SET worker_id = 'other' WHERE id = %s", (job,))
        return {}
    monkeypatch.setattr(agent.get("echo"), "run", hijacked)
    assert runtime.tick(conn, policy, StubLLM(conn=conn), worker_id="w1")[0]["status"] == "lost"


def test_a_rerun_reuses_the_recorded_step_instead_of_asking_the_llm_again(conn, policy):
    job = queue.enqueue(conn, "brain.echo", {"say": "hi"})
    llm = StubLLM(conn=conn)
    ctx = make_ctx(conn, policy, job)
    ctx.llm = llm
    first = agent.get("echo").run(ctx)
    again = agent.get("echo").run(ctx)                     # the same job run again after a crash
    assert first == again and len(llm.calls) == 1


def test_remember_returns_the_stored_output_on_rerun(conn, policy):
    job = queue.enqueue(conn, "brain.echo", {})
    ctx = make_ctx(conn, policy, job)
    calls = []

    def compute():
        calls.append(1)
        return {"kind": "k", "inputs": {}, "output": {"n": len(calls)}, "rationale": "r"}
    assert agent.remember(ctx, "s", compute) == {"n": 1}
    assert agent.remember(ctx, "s", compute) == {"n": 1} and len(calls) == 1


def test_an_agent_that_keeps_calling_the_llm_is_stopped_at_its_budget(conn, policy, monkeypatch):
    from manyworlds.brain import budget
    cap = budget.daily_cap(policy, "echo")

    def loop(ctx):
        for _ in range(100):
            ctx.llm.structured("brain", system="s", content="c", schema=Ack)
        return {}
    monkeypatch.setattr(agent.get("echo"), "run", loop)
    queue.enqueue(conn, "brain.echo", {})
    out = runtime.tick(conn, policy, StubLLM(cost=cap / 4, conn=conn), worker_id="w1")
    assert out[0]["status"] == "failed" and "budget" in out[0]["error"]
    assert budget.spent_today(conn, "echo") <= cap + 1e-9


# --- tech debt closed before publishing (2026-11-20) --------------------------------------------

def test_a_key_error_inside_an_agent_is_retried_not_dropped(conn, policy, monkeypatch):
    job = queue.enqueue(conn, "brain.echo", {})

    def bug(ctx):
        return {}["missing"]
    monkeypatch.setattr(agent.get("echo"), "run", bug)
    out = runtime.tick(conn, policy, StubLLM(conn=conn), worker_id="w1")
    assert out[0]["status"] == "retry"
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job,)).fetchone()["status"] == "queued"


def test_a_crashing_job_does_not_hold_up_the_rest_of_the_tick(conn, policy, monkeypatch):
    bad = queue.enqueue(conn, "brain.echo", {"crash": True}, priority=5)
    good = [queue.enqueue(conn, "brain.echo", {"say": i}) for i in range(3)]
    real = agent.get("echo").run

    def maybe_crash(ctx):
        if (ctx.job.payload or {}).get("crash"):
            raise RuntimeError("transient")
        return real(ctx)
    monkeypatch.setattr(agent.get("echo"), "run", maybe_crash)
    out = {r["job"]: r["status"] for r in runtime.tick(conn, policy, StubLLM(conn=conn), worker_id="w1")}
    assert out[bad] == "retry" and all(out[g] == "done" for g in good)


def test_the_brain_worker_reconnects_after_a_dropped_connection(conn, policy, monkeypatch):
    import psycopg
    connects, ticks = [], []

    def connect():
        connects.append(1)
        return conn

    def fake_tick(c, pol, llm, *, worker_id, now=None):
        ticks.append(1)
        if len(ticks) == 1:
            raise psycopg.OperationalError("server closed the connection unexpectedly")
        return []
    monkeypatch.setattr(runtime, "tick", fake_tick)
    runtime.serve(connect, lambda: policy, lambda c, p: StubLLM(conn=c), worker_id="w1", loops=3, sleep=lambda s: None)
    assert len(ticks) == 3 and len(connects) == 2        # one reconnect, and the loop kept going


def test_the_brain_worker_rebuilds_its_claude_client_when_policy_changes(conn, policy, monkeypatch):
    import dataclasses
    versions = iter([policy, policy, dataclasses.replace(policy, version="changed")])
    built = []
    monkeypatch.setattr(runtime, "tick", lambda c, p, llm, *, worker_id, now=None: [])
    runtime.serve(lambda: conn, lambda: next(versions), lambda c, p: built.append(p.version) or StubLLM(conn=c),
                  worker_id="w1", loops=3, sleep=lambda s: None)
    assert built == [policy.version, "changed"]


def test_the_brain_worker_runs_again_at_once_while_work_is_waiting(conn, policy, monkeypatch):
    # At scale a fixed 30 s sleep after every full tick caps one worker at a few jobs a minute.
    full = int(policy.get("brain", "runtime", "max_jobs_per_tick"))
    ticks = iter([[{}] * full, [{}] * full, [{}]])                     # two full ticks, then a short one
    monkeypatch.setattr(runtime, "tick", lambda *a, **k: next(ticks))
    slept = []
    runtime.serve(lambda: conn, lambda: policy, lambda c, p: StubLLM(conn=c), worker_id="w1", loops=4,
                  sleep=slept.append)
    assert len(slept) == 1                                             # only after the tick that drained the queue
