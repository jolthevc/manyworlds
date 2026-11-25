"""Postgres job queue built for interruptible GPUs.

Workers claim with `FOR UPDATE SKIP LOCKED`, so many workers never take the same job.
A claim takes a lease; the worker heartbeats to extend it. If a SaladCloud or Vast.ai
node is preempted, its heartbeats stop, the lease expires and the next `claim` picks
the job up again. Jobs that run out of attempts are marked `dead` by `reap` and
surface as exceptions instead of retrying forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from psycopg.types.json import Jsonb


@dataclass(frozen=True)
class Job:
    id: Any
    kind: str
    payload: dict[str, Any]
    attempts: int
    max_attempts: int
    creator_id: Any
    batch: str | None


def enqueue(
    conn,
    kind: str,
    payload: dict[str, Any],
    *,
    priority: int = 0,
    max_attempts: int = 3,
    idempotency_key: str | None = None,
    creator_id=None,
    parent_job_id=None,
    batch: str | None = None,
    pools: list[str] | None = None,
    post_id=None,
):
    """Add a job. With an idempotency key, re-enqueueing returns the existing job's id.
    `pools` pins the job to those GPU pools (e.g. benchmarking one provider).
    Research media never travels in a production job (manyworlds.research.references.check_job)."""
    from .research.references import check_job
    check_job(kind, payload)
    row = conn.execute(
        """
        INSERT INTO jobs (kind, payload, priority, max_attempts, idempotency_key,
                          creator_id, parent_job_id, batch, pools, post_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (idempotency_key) DO NOTHING
        RETURNING id
        """,
        (kind, Jsonb(payload), priority, max_attempts, idempotency_key, creator_id, parent_job_id, batch, pools, post_id),
    ).fetchone()
    if row:
        return row["id"]
    return conn.execute("SELECT id FROM jobs WHERE idempotency_key = %s", (idempotency_key,)).fetchone()["id"]


def claim(conn, kinds: list[str], worker_id: str, lease_seconds: int, pool: str | None = None,
          exclude: list | None = None, only_test: bool = False, batches: list[str] | None = None,
          prefer: list[str] | None = None, need_links: bool = False) -> Job | None:
    """Atomically take the next runnable job of the given kinds, or None. Jobs pinned to
    other pools are left for workers in those pools. `batches` (SQL LIKE patterns) limits it to those
    batches: while operations are paused, only tests and benchmarks run (manyworlds.operations). A job retried with
    a backoff (`fail(backoff_seconds=...)`) waits until its run_after. Within a
    priority, kinds in `prefer` (the model the worker has loaded) come first, so a machine serving several
    families runs one model's work back to back instead of reloading 16-30 GB at every switch. With `need_links` (a
    worker holding no storage key, manyworlds.deploy.presign) only jobs whose presigned links outlive an attempt."""
    row = conn.execute(
        """
        UPDATE jobs SET status = 'running',
                        attempts = attempts + 1,
                        worker_id = %(worker)s,
                        lease_until = now() + make_interval(secs => %(lease)s),
                        updated_at = now()
        WHERE id = (
            SELECT id FROM jobs
            WHERE kind = ANY(%(kinds)s)
              AND attempts < max_attempts
              AND (pools IS NULL OR %(pool)s::text = ANY(pools))
              AND (status = 'queued' OR (status = 'running' AND lease_until < now()))
              AND (run_after IS NULL OR run_after <= now())
              AND NOT (id = ANY(%(exclude)s::uuid[]))
              AND (NOT %(only_test)s OR payload->>'test' = 'true')
              AND (%(batches)s::text[] IS NULL OR batch LIKE ANY(%(batches)s::text[]))
              AND (NOT %(need_links)s OR coalesce((payload->'_storage'->>'claim_by')::float8, 0)
                                         > extract(epoch FROM now()))
            ORDER BY priority DESC, (kind = ANY(%(prefer)s::text[])) DESC, created_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING id, kind, payload, attempts, max_attempts, creator_id, batch
        """,
        {"worker": worker_id, "lease": lease_seconds, "kinds": kinds, "pool": pool, "exclude": list(exclude or []),
         "only_test": only_test, "batches": batches, "prefer": list(prefer or []), "need_links": need_links},
    ).fetchone()
    return Job(**row) if row else None


def claim_job(conn, job_id, worker_id: str, lease_seconds: int) -> Job | None:
    """Claim one specific runnable job (e.g. a sandbox run draining only its own creator's jobs)."""
    row = conn.execute(
        """UPDATE jobs SET status = 'running', attempts = attempts + 1, worker_id = %(worker)s,
                          lease_until = now() + make_interval(secs => %(lease)s), updated_at = now()
           WHERE id = %(id)s AND attempts < max_attempts
             AND (status = 'queued' OR (status = 'running' AND lease_until < now()))
           RETURNING id, kind, payload, attempts, max_attempts, creator_id, batch""",
        {"worker": worker_id, "lease": lease_seconds, "id": job_id}).fetchone()
    return Job(**row) if row else None


def heartbeat(conn, job_id, worker_id: str, lease_seconds: int) -> bool:
    """Extend the lease. False means the job was lost to another worker: stop work on it."""
    row = conn.execute(
        """
        UPDATE jobs SET lease_until = now() + make_interval(secs => %s), updated_at = now()
        WHERE id = %s AND worker_id = %s AND status = 'running'
        RETURNING id
        """,
        (lease_seconds, job_id, worker_id),
    ).fetchone()
    return row is not None


def complete(conn, job_id, worker_id: str, result: dict[str, Any]) -> bool:
    row = conn.execute(
        """
        UPDATE jobs SET status = 'done', result = %s, error = NULL, lease_until = NULL, updated_at = now()
        WHERE id = %s AND worker_id = %s AND status = 'running'
        RETURNING id
        """,
        (Jsonb(result), job_id, worker_id),
    ).fetchone()
    return row is not None


def fail(conn, job_id, worker_id: str, error: str, *, retryable: bool = True,
         backoff_seconds: float = 0) -> str | None:
    """Record a failure. Retryable jobs with attempts left go back to the queue; with `backoff_seconds`
    (policy queue.retry_backoff_seconds) not before attempts x backoff_seconds from now, so a failure that
    is about to clear (a download, a host blip) is not retried at once."""
    row = conn.execute(
        """
        UPDATE jobs SET status = CASE WHEN %(retry)s AND attempts < max_attempts
                                      THEN 'queued' ELSE 'failed' END,
                        run_after = CASE WHEN %(retry)s AND attempts < max_attempts AND %(backoff)s > 0
                                         THEN now() + make_interval(secs => attempts * %(backoff)s)
                                         ELSE NULL END,
                        error = %(error)s, lease_until = NULL, updated_at = now()
        WHERE id = %(id)s AND worker_id = %(worker)s AND status = 'running'
        RETURNING status
        """,
        {"retry": retryable, "error": error[:4000], "id": job_id, "worker": worker_id,
         "backoff": float(backoff_seconds)},
    ).fetchone()
    return row["status"] if row else None


def defer(conn, job_id, worker_id: str, *, seconds: float, result: dict[str, Any] | None = None) -> str | None:
    """Give a job back to wait for something it asked for (a batched LLM answer, manyworlds.llm_batch) or for its next
    step (an agent working in steps): queued again with no attempt charged, not before `seconds` from now unless woken
    earlier (run_after set to now). `result`, when given, records the progress so far on the job."""
    row = conn.execute(
        """UPDATE jobs SET status = 'queued', attempts = greatest(attempts - 1, 0), worker_id = NULL,
                           lease_until = NULL, run_after = now() + make_interval(secs => %s), updated_at = now(),
                           result = coalesce(%s::jsonb, result)
           WHERE id = %s AND worker_id = %s AND status = 'running' RETURNING status""",
        (float(seconds), Jsonb(result) if result is not None else None, job_id, worker_id)).fetchone()
    return row["status"] if row else None


def release(conn, job_id, worker_id: str, error: str, *, max_faults: int = 6,
            machine: str | None = None) -> str | None:
    """Give a job back without charging it an attempt: the *worker* failed (GPU out of
    memory, broken CUDA context), not the job. A job that has faulted `max_faults` different
    machines is probably the cause itself, so it fails instead of bouncing forever.

    `machine` is the GPU group (else the worker id). Repeat faults on one machine don't count:
    on 2026-11-14 a RunPod host with a broken driver restarted its container and faulted the
    same job six times in 12 seconds, failing a job that never got a healthy GPU.
    Returns the job's new status."""
    row = conn.execute(
        """
        WITH seen AS (
            SELECT EXISTS (SELECT 1 FROM attempts WHERE job_id = %(id)s AND outcome = 'worker_fault'
                           AND coalesce(gpu_group, worker_id) = %(machine)s) AS again)
        UPDATE jobs SET worker_faults = worker_faults + CASE WHEN seen.again THEN 0 ELSE 1 END,
                        status = CASE WHEN worker_faults + CASE WHEN seen.again THEN 0 ELSE 1 END >= %(max)s
                                      THEN 'failed' ELSE 'queued' END,
                        attempts = greatest(attempts - 1, 0), worker_id = NULL,
                        lease_until = NULL, error = %(error)s, updated_at = now()
        FROM seen
        WHERE id = %(id)s AND worker_id = %(worker)s AND status = 'running'
        RETURNING status
        """,
        {"max": max_faults, "error": error[:4000], "id": job_id, "worker": worker_id,
         "machine": machine or worker_id},
    ).fetchone()
    return row["status"] if row else None


def reap(conn) -> int:
    """Mark jobs dead whose lease expired with no attempts left (e.g. repeated preemption)."""
    rows = conn.execute(
        """
        UPDATE jobs SET status = 'dead', error = coalesce(error, 'lease expired on final attempt'),
                        updated_at = now()
        WHERE status = 'running' AND lease_until < now() AND attempts >= max_attempts
        RETURNING id
        """
    ).fetchall()
    # An open attempt is lost once its worker no longer holds the job.
    conn.execute(
        """
        UPDATE attempts a SET outcome = 'lost', finished_at = now()
        FROM jobs j
        WHERE a.job_id = j.id AND a.finished_at IS NULL
          AND NOT (j.status = 'running' AND j.worker_id = a.worker_id)
        """
    )
    return len(rows)


# A cancelled job is `dead` with this prefix on its error: the queue's terminal states are done, failed and dead
# (the jobs status CHECK), and the failure reports (chief of staff, summary) leave cancelled jobs out.
CANCELLED = "cancelled: "


def cancel(conn, job_ids, reason: str) -> list[Any]:
    """End jobs nobody needs any more, but only those still waiting (`queued`): a running job holds a lease and a
    worker is spending on it, so it is never touched here. Returns the ids actually cancelled."""
    ids = list(job_ids)
    if not ids:
        return []
    rows = conn.execute(
        "UPDATE jobs SET status = 'dead', error = %s, lease_until = NULL, updated_at = now() "
        "WHERE id = ANY(%s) AND status = 'queued' RETURNING id", (f"{CANCELLED}{reason}"[:500], ids)).fetchall()
    return [r["id"] for r in rows]


def counts(conn, *, batch: str | None = None) -> list[dict[str, Any]]:
    return conn.execute(
        """
        SELECT kind, status, count(*) AS n FROM jobs
        WHERE %(batch)s::text IS NULL OR batch = %(batch)s
        GROUP BY kind, status ORDER BY kind, status
        """,
        {"batch": batch},
    ).fetchall()
