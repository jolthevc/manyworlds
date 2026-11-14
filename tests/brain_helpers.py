"""Shared helpers for the Brain's tests."""

from __future__ import annotations

import contextlib
import uuid

from manyworlds import ledger, queue


def creator(conn, *, publish_mode: str = "simulated"):
    return conn.execute("INSERT INTO creators (handle, publish_mode) VALUES (%s, %s) RETURNING id",
                        (f"c{uuid.uuid4().hex[:8]}", publish_mode)).fetchone()["id"]


def make_post(conn, creator_id=None):
    cid = creator_id or creator(conn)
    return conn.execute("INSERT INTO posts (creator_id, format) VALUES (%s, 'photo') RETURNING id",
                        (cid,)).fetchone()["id"]


class StubLLM:
    """Stands in for Claude in Brain tests: returns an empty instance of the asked schema and
    charges `cost` per call to the agent and job of the active tag, like Claude does."""

    def __init__(self, cost: float = 0.0, conn=None):
        self.cost, self.conn, self.calls, self._tag = cost, conn, [], {}

    def model_for(self, role):
        return f"stub-{role}"

    @contextlib.contextmanager
    def tagged(self, agent, job_id, before_call=None):
        before, self._tag = self._tag, {"agent": agent, "job_id": job_id, "before_call": before_call}
        try:
            yield
        finally:
            self._tag = before

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        if self._tag.get("before_call"):
            self._tag["before_call"]()
        self.calls.append({"role": role, "content": content, "agent": self._tag.get("agent")})
        if self.conn is not None:
            ledger.record(self.conn, category="llm", provider=self.model_for(role), quantity=1, unit="token",
                          usd=self.cost, agent=self._tag.get("agent"), job_id=self._tag.get("job_id"))
        return schema.model_construct()


def make_ctx(conn, policy, job_id):
    """Claim this job as worker 'test' and build the context an agent would get."""
    from datetime import datetime, timezone

    from manyworlds.brain.agent import AgentContext
    conn.execute("UPDATE jobs SET status = 'running', worker_id = 'test', attempts = attempts + 1 WHERE id = %s",
                 (job_id,))
    row = conn.execute("SELECT id, kind, payload, attempts, max_attempts, creator_id, batch FROM jobs WHERE id = %s",
                       (job_id,)).fetchone()
    return AgentContext(conn=conn, policy=policy, llm=StubLLM(conn=conn), job=queue.Job(**row),
                        now=datetime.now(timezone.utc))
