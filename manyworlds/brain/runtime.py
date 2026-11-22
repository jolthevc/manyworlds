"""Runs Brain agents: claims `brain.*` jobs on pool `local` and runs each one's agent.

Per job: check the agent's daily LLM budget (again before every LLM call), build its context,
charge its LLM calls to it, keep its lease alive from a heartbeat thread on a second connection,
run it, complete the job. A job taken over by another worker meanwhile is reported "lost".
Non-retryable failures (unknown agent, over budget, bad input) fail the job once; anything else goes back to the queue for another attempt. One job's failure
never stops the others in the tick.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from datetime import datetime, timezone

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row

from .. import operations, queue
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


@contextlib.contextmanager
def _heartbeat(conn, job_id, worker_id: str, lease_s: int):
    """Extend the job's lease every third of it while the agent runs. Its own connection: the
    agent's connection is busy inside long LLM calls and queries."""
    stop = threading.Event()

    def beat():
        try:
            side = psycopg.connect(make_conninfo(conn.info.dsn, password=conn.info.password),
                                   autocommit=True, row_factory=dict_row)
        except Exception:  # noqa: BLE001 - no heartbeat is a lost lease later, not a crash now
            log.exception("brain heartbeat could not connect")
            return
        with side:
            while not stop.wait(max(lease_s / 3, 0.2)):
                try:
                    if not queue.heartbeat(side, job_id, worker_id, lease_s):
                        return
                except Exception:  # noqa: BLE001
                    log.warning("brain heartbeat failed for job %s", job_id)
    t = threading.Thread(target=beat, daemon=True, name=f"brain-heartbeat-{job_id}")
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=5)


def _context_policy(conn, policy: Policy) -> Policy:
    return overrides.effective_policy(conn, policy)


def run_job(conn, policy: Policy, llm, job: queue.Job, *, worker_id: str, now: datetime, lease_s: int = 600) -> dict:
    name = job.kind.removeprefix("brain.")
    try:
        a = agent_mod.get(name)
        effective = _context_policy(conn, policy)
        tag = budget.tag_for(conn, name, job.creator_id)   # sandbox runs spend their own budget, never real agents'
        budget.check(conn, effective, tag)
        ctx = agent_mod.AgentContext(conn=conn, policy=effective, llm=llm, job=job, now=now)
        with _heartbeat(conn, job.id, worker_id, lease_s), \
                llm.tagged(tag, job.id, before_call=lambda: budget.check(conn, effective, tag)):
            result = a.run(ctx)
        if not queue.complete(conn, job.id, worker_id, result or {}):
            log.warning("brain job %s was taken over before %s finished", job.id, name)
            return {"job": job.id, "agent": name, "status": "lost"}
        return {"job": job.id, "agent": name, "status": "done"}
    except (agent_mod.UnknownAgent, NonRetryable) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
        queue.fail(conn, job.id, worker_id, msg, retryable=False)
        return {"job": job.id, "agent": name, "status": "failed", "error": msg}
    except Exception as e:  # noqa: BLE001 - one agent's crash must not stop the tick
        log.exception("agent %s failed on job %s", name, job.id)
        status = queue.fail(conn, job.id, worker_id, f"{type(e).__name__}: {e}", retryable=True)
        return {"job": job.id, "agent": name, "status": "retry" if status == "queued" else "failed",
                "error": f"{type(e).__name__}: {e}"}


def serve(connect, load_policy, make_llm, *, worker_id: str, loops: int | None = None, sleep=None,
          on_output=None) -> None:
    """The brain worker loop. A dropped database connection is reconnected instead of ending the process
    (publishing and metrics jobs must not stall behind a container restart and a lapsed lease), and the
    Claude client is rebuilt whenever the policy changes, so model roles and prices are never stale."""
    import time

    import psycopg

    sleep = sleep or time.sleep
    conn, pol = connect(), load_policy()
    llm, n = make_llm(conn, pol), 0
    while loops is None or n < loops:
        n += 1
        out: list = []
        try:
            out = tick(conn, pol, llm, worker_id=worker_id)
            if out and on_output:
                on_output(out)
        except (psycopg.OperationalError, psycopg.InterfaceError):
            log.exception("brain worker lost its database connection; reconnecting")
            try:
                conn = connect()
                llm = make_llm(conn, pol)
            except Exception:  # noqa: BLE001 - try again next loop
                log.exception("reconnect failed")
        if loops is not None and n >= loops:
            break
        # A tick that ran its maximum leaves work waiting: run the next at once. Sleeping 30 s after every 10 jobs
        # caps one worker at ~3-4 jobs a minute with LLM calls of 10-30 s, about what 1,000 creators need.
        full = bool(out) and len(out) >= int(pol.get("brain", "runtime", "max_jobs_per_tick", default=10))
        if not full:
            sleep(float(pol.get("brain", "runtime", "loop_seconds", default=30)))
        new = load_policy()
        if new.version != pol.version:
            pol, llm = new, make_llm(conn, new)


def tick(conn, policy: Policy, llm, *, worker_id: str, now: datetime | None = None) -> list[dict]:
    cfg = policy.get("brain", "runtime", default={}) or {}
    now = now or datetime.now(timezone.utc)
    out: list[dict] = []
    ran: set = set()
    for _ in range(int(cfg.get("max_jobs_per_tick", 10))):
        kinds = _waiting_kinds(conn)
        # A job retried in this tick waits for the next one; the others still run (it used to end the tick).
        # Paused operations (the default): only jobs marked as tests run (manyworlds.operations).
        job = queue.claim(conn, kinds, worker_id, int(cfg.get("lease_seconds", 600)), pool=POOL,
                          exclude=list(ran), only_test=not operations.is_live(policy)) if kinds else None
        if job is None:
            break
        ran.add(job.id)
        out.append(run_job(conn, policy, llm, job, worker_id=worker_id, now=now,
                           lease_s=int(cfg.get("lease_seconds", 600))))
    return out
