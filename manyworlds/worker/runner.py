"""Generic GPU worker loop.

One container image per job family runs this loop on SaladCloud, Vast.ai or RunPod.
It claims a job, runs the handler, uploads outputs, writes assets and GPU cost to the
ledger, and completes the job. A heartbeat thread keeps the lease alive; if the node is
preempted the lease lapses and another worker reruns the job, so handlers must be
idempotent (same payload in, equivalent outputs out, nothing external half-written).
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from psycopg.types.json import Jsonb

from .. import db, ledger, queue
from ..policy import Policy
from ..storage import ObjectStore

log = logging.getLogger("manyworlds.worker")


@dataclass
class OutputFile:
    path: Path
    kind: str                      # image | video | audio
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class HandlerResult:
    outputs: list[OutputFile]
    recipe: dict[str, Any]         # model, prompt, params: stored on every asset
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobContext:
    job: queue.Job
    workdir: Path
    store: ObjectStore

    def fetch(self, uri: str, name: str) -> Path:
        """Download an input (reference image, LoRA) into the job's working directory."""
        return self.store.get_file(uri, self.workdir / "inputs" / name)


class Handler(Protocol):
    kind: str

    def __call__(self, payload: dict[str, Any], ctx: JobContext) -> HandlerResult: ...


class NonRetryable(Exception):
    """Raise from a handler when retrying cannot help (bad payload, policy violation)."""


class WorkerUnhealthy(RuntimeError):
    """The worker hit too many consecutive machine faults and is exiting so the platform
    can restart it (often on a different node)."""


# Errors that mean the machine is broken, not the job: the job goes back to the queue
# without losing an attempt, and the worker counts toward its circuit breaker.
WORKER_FAULT_MARKERS = ("CUDA error", "out of memory", "CUDA out of memory", "cuDNN error",
                        "CUBLAS_STATUS", "NCCL error", "device-side assert", "no CUDA GPUs")


def is_worker_fault(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(m.lower() in text.lower() for m in WORKER_FAULT_MARKERS) or type(exc).__name__ in (
        "OutOfMemoryError", "AcceleratorError")


class Worker:
    def __init__(
        self,
        *,
        handlers: dict[str, Handler],
        store: ObjectStore,
        policy: Policy,
        pool: str,
        database_url: str | None = None,
        worker_id: str | None = None,
        connect: Callable = db.connect,
    ):
        self.handlers = handlers
        self.store = store
        self.policy = policy
        self.pool = pool
        self.pool_info = policy.pool(pool)
        self.usd_per_second = policy.usd_per_gpu_second(pool)
        self.lease = int(policy.get("queue", "lease_seconds", default=120))
        self.max_consecutive_faults = int(policy.get("queue", "worker_max_consecutive_faults", default=3))
        self.consecutive_faults = 0
        self.database_url = database_url
        self.worker_id = worker_id or f"{pool}:{socket.gethostname()}:{uuid.uuid4().hex[:6]}"
        self._connect = connect
        self.conn = connect(database_url)
        self._stop = threading.Event()

    def stop(self, *_):
        log.info("stop requested; finishing current job")
        self._stop.set()

    def run(self, *, max_jobs: int | None = None, idle_exit_seconds: float | None = None,
            poll_seconds: float = 1.0) -> int:
        """Process jobs until stopped. Returns the number of jobs handled."""
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self.stop)
        handled, idle_since = 0, time.monotonic()
        while not self._stop.is_set() and (max_jobs is None or handled < max_jobs):
            job = queue.claim(self.conn, list(self.handlers), self.worker_id, self.lease)
            if job is None:
                if idle_exit_seconds is not None and time.monotonic() - idle_since > idle_exit_seconds:
                    break
                time.sleep(poll_seconds)
                continue
            self._run_one(job)
            handled += 1
            idle_since = time.monotonic()
            if self.consecutive_faults >= self.max_consecutive_faults:
                raise WorkerUnhealthy(
                    f"{self.consecutive_faults} consecutive machine faults on {self.worker_id}; exiting")
        return handled

    def _heartbeat(self, job_id, done: threading.Event, lost: threading.Event):
        conn = self._connect(self.database_url)
        try:
            while not done.wait(self.lease / 3):
                if not queue.heartbeat(conn, job_id, self.worker_id, self.lease):
                    lost.set()
                    return
        finally:
            conn.close()

    def _run_one(self, job: queue.Job) -> None:
        attempt_id = self.conn.execute(
            "INSERT INTO attempts (job_id, worker_id, pool, gpu_type) VALUES (%s, %s, %s, %s) RETURNING id",
            (job.id, self.worker_id, self.pool, self.pool_info.get("gpu")),
        ).fetchone()["id"]
        done, lost = threading.Event(), threading.Event()
        hb = threading.Thread(target=self._heartbeat, args=(job.id, done, lost), daemon=True)
        hb.start()
        t0 = time.monotonic()
        outcome, error = "failed", None
        try:
            with tempfile.TemporaryDirectory(prefix="mw-") as tmp:
                ctx = JobContext(job=job, workdir=Path(tmp), store=self.store)
                result = self.handlers[job.kind](job.payload, ctx)
                if lost.is_set():
                    raise RuntimeError("lease lost to another worker; discarding outputs")
                asset_ids = self._store_outputs(job, result)
            # Charge wall time including upload: the rented GPU is billed while we wait.
            gpu_seconds = time.monotonic() - t0
            self._charge(job, attempt_id, gpu_seconds)
            queue.complete(self.conn, job.id, self.worker_id,
                           {"asset_ids": [str(a) for a in asset_ids], "gpu_seconds": round(gpu_seconds, 3),
                            "pool": self.pool, **result.meta})
            outcome = "done"
            self.consecutive_faults = 0
        except NonRetryable as e:
            error = f"{type(e).__name__}: {e}"
            self._charge(job, attempt_id, time.monotonic() - t0)
            queue.fail(self.conn, job.id, self.worker_id, error, retryable=False)
        except Exception as e:  # noqa: BLE001 - any handler error is recorded and retried
            error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"
            self._charge(job, attempt_id, time.monotonic() - t0)
            if is_worker_fault(e):
                outcome = "worker_fault"
                self.consecutive_faults += 1
                log.error("machine fault on job %s (%d in a row): %s", job.id, self.consecutive_faults, e)
                queue.release(self.conn, job.id, self.worker_id, error,
                              max_faults=int(self.policy.get("queue", "job_max_worker_faults", default=6)))
            else:
                log.warning("job %s failed: %s", job.id, e)
                queue.fail(self.conn, job.id, self.worker_id, error, retryable=True)
        finally:
            done.set()
            hb.join(timeout=5)
            self.conn.execute(
                "UPDATE attempts SET finished_at = now(), gpu_seconds = %s, outcome = %s, error = %s WHERE id = %s",
                (round(time.monotonic() - t0, 3), outcome, error, attempt_id),
            )

    def _store_outputs(self, job: queue.Job, result: HandlerResult) -> list:
        ids = []
        creator = str(job.creator_id) if job.creator_id else "unassigned"
        for i, out in enumerate(result.outputs):
            key = f"assets/{creator}/{job.id}/{i:02d}{out.path.suffix}"
            stored = self.store.put_file(out.path, key)
            # Same job + same key = same asset: a rerun after preemption overwrites, never duplicates.
            row = self.conn.execute(
                """
                INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta, recipe)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (job_id, uri) DO UPDATE SET sha256 = EXCLUDED.sha256, bytes = EXCLUDED.bytes,
                    meta = EXCLUDED.meta, recipe = EXCLUDED.recipe, created_at = now()
                RETURNING id
                """,
                (job.id, job.creator_id, out.kind, stored.uri, stored.sha256, stored.bytes,
                 Jsonb(out.meta), Jsonb(result.recipe)),
            ).fetchone()
            ids.append(row["id"])
        return ids

    def _charge(self, job: queue.Job, attempt_id, gpu_seconds: float) -> None:
        seconds = round(gpu_seconds, 3)  # price the recorded quantity so the ledger reconciles
        ledger.record(
            self.conn, category="gpu", provider=self.pool, quantity=seconds,
            unit="gpu_second", usd=seconds * self.usd_per_second, job_id=job.id,
            attempt_id=attempt_id, creator_id=job.creator_id, policy_version=self.policy.version,
        )


def handlers_for(kinds: list[str]) -> dict[str, Handler]:
    """Import only the handlers this container needs (GPU libraries are heavy)."""
    found: dict[str, Handler] = {}
    for kind in kinds:
        family = kind.split(".")[0]
        if family == "fake":
            from .handlers import fake as mod
        elif family == "image":
            from .handlers import image_zimage as mod
        elif family == "video":
            from .handlers import video_wan as mod
        elif family == "tts":
            from .handlers import tts_kokoro as mod
        elif family == "train":
            from .handlers import train_lora as mod
        else:
            raise ValueError(f"no handler family for {kind!r}")
        found[kind] = mod.HANDLERS[kind]
    return found


def main_from_env() -> None:
    from .. import policy as policy_mod
    from .. import storage

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    kinds = [k.strip() for k in os.environ["MW_WORKER_KINDS"].split(",") if k.strip()]
    worker = Worker(handlers=handlers_for(kinds), store=storage.from_url(), policy=policy_mod.load(),
                    pool=os.environ.get("MW_POOL", "local"))
    log.info("worker %s serving %s", worker.worker_id, kinds)
    worker.run()
