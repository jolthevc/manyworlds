"""Runs Brain agents: claims `brain.*` jobs on pool `local` and runs each one's agent.

Per job: check the agent's daily LLM budget, build its context, charge its LLM calls to it,
run it, complete the job. Non-retryable failures (unknown agent, over budget, bad input) fail
the job once; anything else goes back to the queue for another attempt. One job's failure
never stops the others in the tick.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .. import queue
from ..policy import Policy
from ..worker.runner import NonRetryable
from . import agent as agent_mod
from . import budget, overrides

log = logging.getLogger("manyworlds.brain")
POOL = "local"


def _waiting_kinds(conn) -> list[str]:
    return [r["kind"] for r in conn.execute(
        """SELECT DISTINCT kind FROM jobs WHERE kind LIKE 'brain.%%' AND attempts < max_attempts
             AND (status = 'queued' OR (status = 'running' AND lease_until < now()))""").fetchall()]


def _context_policy(conn, policy: Policy) -> Policy:
    return overrides.effective_policy(conn, policy)


def run_job(conn, policy: Policy, llm, job: queue.Job, *, worker_id: str, now: datetime) -> dict:
    name = job.kind.removeprefix("brain.")
    try:
        a = agent_mod.get(name)
        effective = _context_policy(conn, policy)
        budget.check(conn, effective, name)
        ctx = agent_mod.AgentContext(conn=conn, policy=effective, llm=llm, job=job, now=now)
        with llm.tagged(name, job.id):
            result = a.run(ctx)
        queue.complete(conn, job.id, worker_id, result or {})
        return {"job": job.id, "agent": name, "status": "done"}
    except (KeyError, NonRetryable) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
        queue.fail(conn, job.id, worker_id, msg, retryable=False)
        return {"job": job.id, "agent": name, "status": "failed", "error": msg}
    except Exception as e:  # noqa: BLE001 - one agent's crash must not stop the tick
        log.exception("agent %s failed on job %s", name, job.id)
        status = queue.fail(conn, job.id, worker_id, f"{type(e).__name__}: {e}", retryable=True)
        return {"job": job.id, "agent": name, "status": "retry" if status == "queued" else "failed",
                "error": f"{type(e).__name__}: {e}"}


def tick(conn, policy: Policy, llm, *, worker_id: str, now: datetime | None = None) -> list[dict]:
    cfg = policy.get("brain", "runtime", default={}) or {}
    now = now or datetime.now(timezone.utc)
    out: list[dict] = []
    ran: set = set()
    for _ in range(int(cfg.get("max_jobs_per_tick", 10))):
        kinds = _waiting_kinds(conn)
        job = queue.claim(conn, kinds, worker_id, int(cfg.get("lease_seconds", 600)), pool=POOL) if kinds else None
        if job is None:
            break
        if job.id in ran:   # a job retried in this tick waits for the next one, not a tight loop
            queue.unclaim(conn, job.id, worker_id)
            break
        ran.add(job.id)
        out.append(run_job(conn, policy, llm, job, worker_id=worker_id, now=now))
    return out
