"""Operators can see what the Brain did and why: a job's decisions, the work it commissioned,
and what each agent spent; and decide pending approvals."""

from __future__ import annotations

from manyworlds import queue
from manyworlds.brain import agent, ops, overrides, runtime

from brain_helpers import StubLLM, make_ctx


def test_trace_shows_decisions_children_and_cost(conn, policy):
    parent = queue.enqueue(conn, "brain.echo", {"say": "parent"})
    ctx = make_ctx(conn, policy, parent)
    child = agent.commission(ctx, "echo", {"say": "child"}, reason="follow up", key="c")
    conn.execute("UPDATE jobs SET status = 'queued', worker_id = NULL WHERE id = %s", (parent,))
    runtime.tick(conn, policy, StubLLM(cost=0.01, conn=conn), worker_id="w1")
    t = ops.trace(conn, parent)
    assert t["job"]["id"] == parent
    assert [d["output"]["say"] for d in t["decisions"]] == ["parent"]
    assert [c["job"]["id"] for c in t["children"]] == [child]
    assert t["children"][0]["decisions"][0]["output"]["say"] == "child"
    assert t["llm_usd_by_agent"] == {"echo": 0.02}          # parent + child, both echo


def test_pending_approvals_are_listed_and_decided(conn, policy):
    r = overrides.propose(conn, policy, "brain.exploration_share", 0.9, author="strategist", reason="explore more")
    assert [a["id"] for a in ops.approvals(conn)] == [r["approval_id"]]
    ops.decide(conn, r["approval_id"], approved=False, by="owner")
    assert ops.approvals(conn) == []
