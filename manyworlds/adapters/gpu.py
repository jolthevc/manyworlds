"""Adapters that hand work to the GPU worker fleet through the job queue.

The orchestrator never talks to a GPU directly: it enqueues a job, and whichever
worker (SaladCloud, Vast.ai, RunPod or local) has that job kind picks it up.
"""

from __future__ import annotations

import time
from dataclasses import asdict

from .. import queue
from .base import Generated, ImageRequest, SpeechRequest, VideoRequest


class JobTimeout(RuntimeError):
    pass


class QueueBackedAdapter:
    def __init__(self, conn, job_kind: str, *, max_attempts: int = 3, priority: int = 0,
                 poll_seconds: float = 1.0, timeout_seconds: float = 1800):
        self.conn = conn
        self.job_kind = job_kind
        self.max_attempts = max_attempts
        self.priority = priority
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds

    def submit(self, payload: dict, *, creator_id=None, batch=None, idempotency_key=None):
        return queue.enqueue(
            self.conn, self.job_kind, payload, priority=self.priority, max_attempts=self.max_attempts,
            creator_id=creator_id, batch=batch, idempotency_key=idempotency_key,
        )

    def wait(self, job_id) -> Generated:
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            job = self.conn.execute("SELECT status, result, error FROM jobs WHERE id = %s", (job_id,)).fetchone()
            if job["status"] == "done":
                assets = self.conn.execute(
                    "SELECT id FROM assets WHERE job_id = %s ORDER BY created_at", (job_id,)
                ).fetchall()
                usd = self.conn.execute(
                    "SELECT coalesce(sum(usd), 0) AS usd FROM cost_entries WHERE job_id = %s", (job_id,)
                ).fetchone()["usd"]
                return Generated(asset_ids=[a["id"] for a in assets], job_id=job_id,
                                 usd=float(usd), meta=job["result"] or {})
            if job["status"] in ("failed", "dead"):
                raise RuntimeError(f"job {job_id} {job['status']}: {job['error']}")
            time.sleep(self.poll_seconds)
        raise JobTimeout(f"job {job_id} not finished after {self.timeout_seconds}s")


class QueueImageGen(QueueBackedAdapter):
    def generate(self, req: ImageRequest, *, creator_id=None) -> Generated:
        return self.wait(self.submit(asdict(req), creator_id=creator_id))


class QueueVideoGen(QueueBackedAdapter):
    def generate(self, req: VideoRequest, *, creator_id=None) -> Generated:
        return self.wait(self.submit(asdict(req), creator_id=creator_id))


class QueueTTS(QueueBackedAdapter):
    def speak(self, req: SpeechRequest, *, creator_id=None) -> Generated:
        return self.wait(self.submit(asdict(req), creator_id=creator_id))
