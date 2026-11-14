"""The agent contract: what every Brain agent is given, how it records decisions and how it
asks other agents for work.

An agent is a module with a `name` and `run(ctx) -> dict`, registered on import of
`manyworlds.brain.agents`. It runs as a `brain.<name>` job (manyworlds.brain.runtime), reads
shared state from Postgres, writes its outputs to tables, and records each decision with the
policy and override version it ran under. State lives in the database, never in chat history.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from psycopg.types.json import Jsonb

from .. import queue
from ..llm import StructuredLLM
from ..policy import Policy


@dataclass
class AgentContext:
    conn: Any
    policy: Policy          # the effective policy: file plus runtime overrides (brain.overrides)
    llm: StructuredLLM
    job: queue.Job
    now: datetime

    @property
    def agent(self) -> str:
        return self.job.kind.removeprefix("brain.")


class Agent(Protocol):
    name: str

    def run(self, ctx: AgentContext) -> dict: ...


REGISTRY: dict[str, Agent] = {}


def register(a: Agent) -> Agent:
    REGISTRY[a.name] = a
    return a


def get(name: str) -> Agent:
    from . import agents  # noqa: F401 - registers the agents
    if name not in REGISTRY:
        raise KeyError(f"no agent named {name!r}")
    return REGISTRY[name]


def override_version(policy: Policy) -> int:
    """The runtime-override version a policy carries ('<file hash>+o<n>'), 0 without overrides."""
    m = re.search(r"\+o(\d+)$", policy.version or "")
    return int(m.group(1)) if m else 0


def record_decision(ctx: AgentContext, kind: str, *, step: str, inputs: dict, output: dict, rationale: str,
                    creator_id=None, model: str | None = None) -> uuid.UUID:
    """One decision per job step: a rerun of the same job finds the one it already wrote."""
    row = ctx.conn.execute(
        """INSERT INTO decisions (kind, creator_id, inputs, output, model, policy_version, agent, job_id, step,
                                  rationale, override_version)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (job_id, step) WHERE job_id IS NOT NULL DO NOTHING RETURNING id""",
        (kind, creator_id, Jsonb(inputs), Jsonb(output), model, ctx.policy.version, ctx.agent, ctx.job.id, step,
         rationale, override_version(ctx.policy))).fetchone()
    if row:
        return row["id"]
    return ctx.conn.execute("SELECT id FROM decisions WHERE job_id = %s AND step = %s",
                            (ctx.job.id, step)).fetchone()["id"]


def remember(ctx: AgentContext, step: str, compute) -> dict:
    """Run one step of an agent once per job. If this job already recorded `step` (a rerun after a
    crash), return what it decided then, without asking the LLM again, so what the agent acts on
    is always what the decision log says. `compute()` returns {kind, inputs, output, rationale}
    and optionally creator_id and model."""
    row = ctx.conn.execute("SELECT output FROM decisions WHERE job_id = %s AND step = %s",
                           (ctx.job.id, step)).fetchone()
    if row:
        return row["output"]
    d = compute()
    record_decision(ctx, d["kind"], step=step, inputs=d.get("inputs", {}), output=d["output"],
                    rationale=d.get("rationale", ""), creator_id=d.get("creator_id"), model=d.get("model"))
    return ctx.conn.execute("SELECT output FROM decisions WHERE job_id = %s AND step = %s",
                            (ctx.job.id, step)).fetchone()["output"]


def commission(ctx: AgentContext, target: str, payload: dict, *, reason: str, key: str) -> uuid.UUID:
    """Ask another agent for work. The same request from the same job is enqueued once."""
    return queue.enqueue(ctx.conn, f"brain.{target}", {**payload, "reason": reason, "requested_by": ctx.agent},
                         creator_id=ctx.job.creator_id, parent_job_id=ctx.job.id,
                         idempotency_key=f"{ctx.job.id}:{key}")
