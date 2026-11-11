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
):
    """Add a job. With an idempotency key, re-enqueueing returns the existing job's id."""
    row = conn.execute(
        """
        INSERT INTO jobs (kind, payload, priority, max_attempts, idempotency_key,
                          creator_id, parent_job_id, batch)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (idempotency_key) DO NOTHING
        RETURNING id
        """,
        (kind, Jsonb(payload), priority, max_attempts, idempotency_key, creator_id, parent_job_id, batch),
    ).fetchone()
    if row:
        return row["id"]
    return conn.execute("SELECT id FROM jobs WHERE idempotency_key = %s", (idempotency_key,)).fetchone()["id"]


def claim(conn, kinds: list[str], worker_id: str, lease_seconds: int) -> Job | None:
    """Atomically take the next runnable job of the given kinds, or None."""
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
              AND (status = 'queued' OR (status = 'running' AND lease_until < now()))
            ORDER BY priority DESC, created_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING id, kind, payload, attempts, max_attempts, creator_id, batch
        """,
        {"worker": worker_id, "lease": lease_seconds, "kinds": kinds},
    ).fetchone()
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


def fail(conn, job_id, worker_id: str, error: str, *, retryable: bool = True) -> str | None:
    """Record a failure. Retryable jobs with attempts left go back to the queue."""
    row = conn.execute(
        """
        UPDATE jobs SET status = CASE WHEN %(retry)s AND attempts < max_attempts
                                      THEN 'queued' ELSE 'failed' END,
                        error = %(error)s, lease_until = NULL, updated_at = now()
        WHERE id = %(id)s AND worker_id = %(worker)s AND status = 'running'
        RETURNING status
        """,
        {"retry": retryable, "error": error[:4000], "id": job_id, "worker": worker_id},
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


def counts(conn, *, batch: str | None = None) -> list[dict[str, Any]]:
    return conn.execute(
        """
        SELECT kind, status, count(*) AS n FROM jobs
        WHERE %(batch)s::text IS NULL OR batch = %(batch)s
        GROUP BY kind, status ORDER BY kind, status
        """,
        {"batch": batch},
    ).fetchall()
