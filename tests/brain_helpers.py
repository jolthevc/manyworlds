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


class ScoringLLM(StubLLM):
    """Scores every rubric dimension `value` (any rubric schema); records role and content."""

    def __init__(self, value: int = 6, conn=None):
        super().__init__(conn=conn)
        self.value = value

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema)
        self.calls[-1]["system"] = system
        dims = {k: {"score": self.value, "why": "w"} for k in schema.model_fields if k != "overall_note"}
        return schema.model_validate({**dims, "overall_note": "n"})


def planned_post(conn, creator_id, **body):
    from psycopg.types.json import Jsonb
    b = {"scenes": ["dinner on a terrace in Venice, golden hour"], "script": "", "pillar": "food", "tone": "warm",
         "series": "", **body}
    return conn.execute("INSERT INTO posts (creator_id, format, status, caption, body) "
                        "VALUES (%s, 'photo', 'planned', 'first night, best pasta of my life', %s) RETURNING id",
                        (creator_id, Jsonb(b))).fetchone()["id"]


def sandbox_history(conn, policy, n: int, *, scenario: str = "personality_wins", seed: int = 0, cid=None):
    """A sandbox creator with n simulated publications: posts with scored ideas and scenario outcomes."""
    import random
    from datetime import datetime, timedelta, timezone

    from psycopg.types.json import Jsonb

    from manyworlds.brain import simulation
    cid = cid or creator(conn, publish_mode="simulated")
    rng = random.Random(f"history:{seed}")
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
    for i in range(n):
        fmt = rng.choice(["photo", "carousel", "story", "story_clip"])
        scores = {d: rng.randint(2, 9) for d in ("personality", "aspiration", "novelty", "naturalness")}
        when = t0 + timedelta(hours=8 * i)
        post = conn.execute("INSERT INTO posts (creator_id, format, status, caption, body, scheduled_for, published_at) "
                            "VALUES (%s, %s, 'simulated', 'SECRET CAPTION TEXT', '{}', %s, %s) RETURNING id",
                            (cid, fmt, when, when)).fetchone()["id"]
        idea = conn.execute("INSERT INTO ideas (creator_id, concept, format, status, post_id, attributes) "
                            "VALUES (%s, 'x', %s, 'scheduled', %s, %s) RETURNING id",
                            (cid, fmt, post, Jsonb({"mood": rng.choice(["warm", "dry"])}))).fetchone()["id"]
        conn.execute("INSERT INTO evaluations (item_kind, idea_id, creator_id, rubric, rubric_version, evaluator_model, "
                     "prompt_version, scores, samples, role) VALUES ('idea', %s, %s, 'idea', 1, 'm', 'p', %s, 3, 'evaluator')",
                     (idea, cid, Jsonb(scores)))
        conn.execute("INSERT INTO publications (post_id, platform, simulated, published_at) VALUES (%s, 'simulated', true, %s)",
                     (post, when))
        simulation.simulate_post(conn, policy, post, scenario=scenario, seed=seed)
    return cid
