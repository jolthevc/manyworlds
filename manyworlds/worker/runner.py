"""Generic GPU worker loop.

One container image per job family runs this loop on SaladCloud, Vast.ai or RunPod.
It claims a job, runs the handler, uploads outputs, writes assets and GPU cost to the
ledger, and completes the job. A heartbeat thread keeps the lease alive; if the node is
preempted the lease lapses and another worker reruns the job, so handlers must be
idempotent (same payload in, equivalent outputs out, nothing external half-written).
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import socket
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Protocol

from psycopg.types.json import Jsonb

from .. import db, ledger, logs, operations, queue
from ..policy import Policy
from ..deploy import presign
from ..storage import MissingLink, ObjectStore, UrlStore

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
    charges: list[dict] = field(default_factory=list)   # hosted API spend, recorded whatever the outcome

    def charge(self, provider: str, usd: float, quantity: float = 1, unit: str = "request") -> None:
        """A hosted API billed us (a video render, a voice line): into the ledger as `hosted_api`, success or not."""
        self.charges.append({"provider": provider, "usd": float(usd), "quantity": quantity, "unit": unit})

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
                        "CUBLAS_STATUS", "NCCL error", "device-side assert", "no CUDA GPUs",
                        "CUDA unknown error", "CUDA driver", "NVML", "no NVIDIA driver",
                        "CUDA-capable device", "Disk quota exceeded", "No space left on device")

# Errors that mean the code (or the image) is wrong, not the machine or luck: a retry fails the same way on any
# machine, so the job fails at once instead of paying for two more attempts (2026-11-22: 24 package/signature
# errors and 49 InfiniteTalk failures of ~5 GPU-minutes each were retried).
CODE_FAULT_TYPES = ("TypeError", "ImportError", "ModuleNotFoundError", "AttributeError", "NameError", "SyntaxError")
CODE_FAULT_MARKERS = ("Failed to find C compiler", "takes no keyword arguments", "unexpected keyword argument",
                      "ComfyUI exited at start", "No module named")


def is_worker_fault(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(m.lower() in text.lower() for m in WORKER_FAULT_MARKERS) or type(exc).__name__ in (
        "OutOfMemoryError", "AcceleratorError")


def is_code_fault(exc: BaseException) -> bool:
    """A machine's fault wins (ComfyUI exiting at start because the card is broken is retried elsewhere), and an
    attribute of None is a transient empty answer, not wrong code."""
    text = f"{type(exc).__name__}: {exc}"
    if is_worker_fault(exc) or (type(exc).__name__ == "AttributeError" and "'NoneType'" in str(exc)):
        return False
    return type(exc).__name__ in CODE_FAULT_TYPES or any(m.lower() in text.lower() for m in CODE_FAULT_MARKERS)


def family_of(policy: Policy, kind: str) -> str:
    """A kind's worker family: job_kinds.<kind>.family, else its prefix."""
    return str((policy.get("job_kinds", kind, default={}) or {}).get("family") or kind.split(".")[0])


def gpu_memory_gb() -> float | None:
    """Total memory of the first GPU in GB (nvidia-smi), or None without one: kinds whose policy floor
    (job_kinds.<kind>.min_vram_gb) the card can't meet are not served (2026-11-22: 320 Z-Image attempts died of
    out-of-memory on consumer cards with less usable memory than advertised)."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return round(float(out[0]) / 1024, 1) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def kinds_that_fit(kinds: list[str], policy: Policy, memory_gb: float | None) -> list[str]:
    if memory_gb is None:
        return kinds
    return [k for k in kinds
            if float((policy.get("job_kinds", k, default={}) or {}).get("min_vram_gb") or 0) <= memory_gb]


NO_GPU_POOLS = ("local", "hosted")


def check_registered(pool: str) -> None:
    """A worker for a GPU pool runs only on a machine the autoscaler registered (it always sets MW_GROUP).
    2026-11-20: a diagnostic container on the owner's laptop, started with a real pool name, claimed a talk
    benchmark job and began downloading model weights there. Pools on no GPU (`local`: fakes, brain; `hosted`:
    hosted APIs on the control plane) are exempt."""
    if pool not in NO_GPU_POOLS and not os.environ.get("MW_GROUP"):
        raise SystemExit(f"refusing to work pool {pool!r} outside a registered machine (MW_GROUP is not set)")


def pool_info(policy: Policy, pool: str) -> dict:
    """The pool's definition: from the policy baked into this image, or, for a pool added after the image
    was built, from MW_POOL_INFO, which the autoscaler sets on every machine. 2026-11-20: a talk worker on a
    new RunPod pool crashed at start ("unknown GPU pool") until the never-started check stopped its machine."""
    try:
        return policy.pool(pool)
    except KeyError:
        raw = os.environ.get("MW_POOL_INFO")
        if not raw:
            raise
        import json
        return json.loads(raw)


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
        group: str | None = None,
        self_destroy: Callable[[], str | None] | None = None,
        need_links: bool | None = None,
    ):
        self.handlers = handlers
        self.store = store
        self.policy = policy
        self.pool = pool
        self.pool_info = pool_info(policy, pool)
        self.usd_per_second = float(self.pool_info["usd_per_hour"]) / 3600.0
        self.lease = int(policy.get("queue", "lease_seconds", default=120))
        self.max_consecutive_faults = int(policy.get("queue", "worker_max_consecutive_faults", default=3))
        self.consecutive_faults = 0
        self.database_url = database_url
        self.worker_id = worker_id or f"{pool}:{socket.gethostname()}:{uuid.uuid4().hex[:6]}"
        self._connect = connect
        self.conn = connect(database_url)
        self._stop = threading.Event()
        # Self-stop: a worker on a rented machine destroys that machine when it has been
        # idle too long, is past its deadline, or the autoscaler already marked it stopped.
        # One of three independent layers; see manyworlds.deploy.autoscale.
        self.group = group if group is not None else os.environ.get("MW_GROUP")
        self._self_destroy = self_destroy
        # No storage key on this machine (security.worker_storage: presigned): only jobs carrying fresh links.
        self.need_links = presign.needs_links() if need_links is None else need_links
        self.link_timeout = float(policy.get("security", "presigned", "http_timeout_seconds", default=600))
        self._busy = False
        self._idle_since = time.monotonic()
        self.last_kind: str | None = None   # model affinity: claim the loaded model's kinds first, free on a switch
        self.ran: list[str] = []            # kinds run since the last switch: every engine they loaded is freed

    def _switch_to(self, kind: str) -> None:
        """Before a job of another family than the last (a merged machine: voice, motion, lip sync), the last
        family's engine gives its GPU memory back (handler module `release()`, e.g. ComfyUI /free)."""
        last, self.last_kind = self.last_kind, kind
        if last is None or family_of(self.policy, last) == family_of(self.policy, kind):
            if kind not in self.ran:
                self.ran.append(kind)
            return
        # Every engine the last family loaded (Z-Image stays loaded through an image.narrated job), once each.
        ran, self.ran = self.ran, [kind]
        done = set()
        for k in ran:
            release = RELEASE.get(k)
            if release is None or release in done:
                continue
            done.add(release)
            try:
                release()
                log.info("released %s's models before %s", k, kind)
            except Exception as e:  # noqa: BLE001 - freeing is best effort; the next load may still fit
                log.warning("could not release %s: %s", k, e)

    def stop(self, *_):
        log.info("stop requested; finishing current job")
        self._stop.set()

    def self_stop_reason(self, conn=None, now=None) -> str | None:
        """Why this worker's machine should go away now, or None."""
        from datetime import datetime, timezone

        if not self.group:
            return None
        now = now or datetime.now(timezone.utc)
        g = (conn or self.conn).execute("SELECT deadline, stopped_at FROM gpu_groups WHERE name = %s",
                              (self.group,)).fetchone()
        if g is None:
            return None
        if g["stopped_at"] is not None:
            return "stopped_by_autoscaler"
        if now >= g["deadline"]:
            return "deadline"
        from ..deploy.enforce import idle_stop_minutes

        idle_min = idle_stop_minutes(self.policy, self.pool_info.get("provider"))
        if not self._busy and time.monotonic() - self._idle_since > idle_min * 60:
            return "idle"
        return None

    def _retire_machine(self) -> None:
        """A machine whose GPU keeps faulting is broken: give it up rather than restart on it.
        Its deadline moves to now (so every enforcement layer stops it), then it destroys itself."""
        if not self.group:
            return
        from ..deploy.providers import self_destroy
        try:
            self.conn.execute("UPDATE gpu_groups SET deadline = least(deadline, now()) WHERE name = %s",
                              (self.group,))
            log.warning("retiring faulty machine %s: %s", self.group, (self._self_destroy or self_destroy)())
        except Exception as e:  # noqa: BLE001 - the autoscaler still stops it by deadline
            log.warning("could not retire %s: %s", self.group, e)

    def _watchdog(self):
        from ..deploy.providers import self_destroy

        destroy = self._self_destroy or self_destroy
        every = float(self.policy.get("autoscale", "tick_seconds", default=60))
        conn = self._connect(self.database_url)
        try:
            while not self._stop.wait(every):
                try:
                    reason = self.self_stop_reason(conn)
                    if reason:
                        provider = destroy()
                        if provider is None:
                            log.info("self-stop (%s) not possible on this provider; autoscaler will stop it", reason)
                            return
                        log.warning("destroyed own %s machine: %s", provider, reason)
                        self._stop.set()
                        return
                except Exception as e:  # noqa: BLE001 - try again next round
                    log.warning("watchdog check failed: %s", e)
        finally:
            conn.close()

    def run(self, *, max_jobs: int | None = None, idle_exit_seconds: float | None = None,
            poll_seconds: float = 1.0) -> int:
        """Process jobs until stopped. Returns the number of jobs handled."""
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self.stop)
        if self.group and self.policy.get("autoscale", "worker_self_stop", default=True):
            threading.Thread(target=self._watchdog, daemon=True, name="self-stop").start()
        handled, idle_since = 0, time.monotonic()
        while not self._stop.is_set() and (max_jobs is None or handled < max_jobs):
            prefer = [k for k in self.handlers if self.last_kind and
                      family_of(self.policy, k) == family_of(self.policy, self.last_kind)]
            job = queue.claim(self.conn, list(self.handlers), self.worker_id, self.lease, self.pool,
                              batches=operations.test_batch_patterns(self.policy), prefer=prefer,
                              need_links=self.need_links)
            if job is None:
                if idle_exit_seconds is not None and time.monotonic() - idle_since > idle_exit_seconds:
                    break
                time.sleep(poll_seconds)
                continue
            self._busy = True
            try:
                with logs.context(job_id=job.id, kind=job.kind, creator_id=job.creator_id):
                    self._switch_to(job.kind)
                    self._run_one(job)
            finally:
                self._busy = False
                self._idle_since = time.monotonic()
            handled += 1
            idle_since = time.monotonic()
            if self.consecutive_faults >= self.max_consecutive_faults:
                self._retire_machine()
                raise WorkerUnhealthy(
                    f"{self.consecutive_faults} consecutive machine faults on {self.worker_id}; exiting")
        return handled

    def _ensure_conn(self) -> None:
        """Reconnect if the main connection was dropped while a job ran."""
        try:
            self.conn.execute("SELECT 1")
        except Exception:  # noqa: BLE001 - any failure here means the connection is unusable
            log.warning("database connection dropped during the job; reconnecting")
            try:
                self.conn.close()
            except Exception:  # noqa: BLE001
                pass
            self.conn = self._connect(self.database_url)

    def _heartbeat(self, job_id, done: threading.Event, lost: threading.Event):
        """Extend the lease every third of it. A dropped database connection (Neon restarts its
        compute; poolers drop connections) must not silence a healthy worker: on 2026-11-14 two
        machines lost their heartbeat this way mid-copy and were stopped as stuck. So any error
        reconnects and the loop goes on; only the queue saying the job is gone ends it."""
        conn = None
        try:
            while not done.wait(self.lease / 3):
                try:
                    if conn is None:
                        conn = self._connect(self.database_url)
                    if not queue.heartbeat(conn, job_id, self.worker_id, self.lease):
                        lost.set()
                        return
                except Exception as e:  # noqa: BLE001 - reconnect next round
                    log.warning("heartbeat failed (%s); reconnecting", str(e)[:200])
                    try:
                        if conn is not None:
                            conn.close()
                    except Exception:  # noqa: BLE001
                        pass
                    conn = None
        finally:
            if conn is not None:
                conn.close()

    def _run_one(self, job: queue.Job) -> None:
        attempt_id = self.conn.execute(
            "INSERT INTO attempts (job_id, worker_id, pool, gpu_type, gpu_group) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (job.id, self.worker_id, self.pool, self.pool_info.get("gpu"), self.group),
        ).fetchone()["id"]
        done, lost = threading.Event(), threading.Event()
        hb = threading.Thread(target=self._heartbeat, args=(job.id, done, lost), daemon=True)
        hb.start()
        t0 = time.monotonic()
        outcome, error = "failed", None
        charges: list[dict] = []
        log_ctx = contextlib.ExitStack()   # every line about this attempt carries its id (manyworlds.logs)
        log_ctx.enter_context(logs.context(attempt_id=attempt_id))
        try:
            with tempfile.TemporaryDirectory(prefix="mw-") as tmp:
                job, store = self._job_store(job)
                ctx = JobContext(job=job, workdir=Path(tmp), store=store, charges=charges)
                result = self.handlers[job.kind](job.payload, ctx)
                # A long job (hours of LoRA training) leaves this connection idle; poolers and
                # Neon drop idle connections, and losing it here would throw away the result.
                self._ensure_conn()
                if lost.is_set():
                    raise RuntimeError("lease lost to another worker; discarding outputs")
                asset_ids = self._store_outputs(job, result, store)
            # Charge wall time including upload: the rented GPU is billed while we wait.
            gpu_seconds = time.monotonic() - t0
            self._charge(job, attempt_id, gpu_seconds)
            if queue.complete(self.conn, job.id, self.worker_id,
                              {"asset_ids": [str(a) for a in asset_ids], "gpu_seconds": round(gpu_seconds, 3),
                               "pool": self.pool, **result.meta}):
                outcome = "done"
                self.consecutive_faults = 0
            else:
                # The lease went to another worker (or the job was reaped) while this one finished: the job is not
                # ours to finish. The cost stays charged; the attempt is lost, not done.
                outcome, error = "lost", "lease lost before completion; result not recorded"
                log.warning("job %s: lease lost before completion; result discarded", job.id)
        except (NonRetryable, MissingLink) as e:   # links that miss an object fail the same on any machine
            error = f"{type(e).__name__}: {e}"
            self._ensure_conn()   # the connection may have died with the job: charge and fail on a live one
            self._charge(job, attempt_id, time.monotonic() - t0)
            queue.fail(self.conn, job.id, self.worker_id, error, retryable=False)
        except Exception as e:  # noqa: BLE001 - any handler error is recorded and retried
            error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"
            self._ensure_conn()   # the connection may have died with the job: charge and fail on a live one
            self._charge(job, attempt_id, time.monotonic() - t0)
            if is_code_fault(e) and not is_worker_fault(e):
                log.error("code fault on job %s (fails the same on any machine): %s", job.id, e)
                queue.fail(self.conn, job.id, self.worker_id, f"code fault: {error}", retryable=False)
            elif is_worker_fault(e):
                outcome = "worker_fault"
                self.consecutive_faults += 1
                log.error("machine fault on job %s (%d in a row): %s", job.id, self.consecutive_faults, e)
                queue.release(self.conn, job.id, self.worker_id, error,
                              max_faults=int(self.policy.get("queue", "job_max_worker_faults", default=6)),
                              machine=self.group)
            else:
                log.warning("job %s failed: %s", job.id, e)
                queue.fail(self.conn, job.id, self.worker_id, error, retryable=True,
                           backoff_seconds=float(self.policy.get("queue", "retry_backoff_seconds", default=0)))
        finally:
            done.set()
            hb.join(timeout=5)
            self._ensure_conn()
            for c in charges:
                ledger.record(self.conn, category="hosted_api", provider=c["provider"], quantity=c["quantity"],
                              unit=c["unit"], usd=c["usd"], job_id=job.id, attempt_id=attempt_id,
                              creator_id=job.creator_id, policy_version=self.policy.version)
            self.conn.execute(
                "UPDATE attempts SET finished_at = now(), gpu_seconds = %s, outcome = %s, error = %s WHERE id = %s",
                (round(time.monotonic() - t0, 3), outcome, error, attempt_id),
            )
            log_ctx.close()

    def _job_store(self, job: queue.Job) -> tuple[queue.Job, ObjectStore]:
        """The job as its handler sees it and the store it uses: a job carrying presigned links (payload._storage,
        manyworlds.deploy.presign) reads and writes through them only, and its handler never sees the links."""
        links = job.payload.get(presign.FIELD) if isinstance(job.payload, dict) else None
        if not links:
            if self.need_links:
                raise MissingLink("this machine holds no storage key and the job carries no links")
            return job, self.store
        payload = {k: v for k, v in job.payload.items() if k != presign.FIELD}
        return replace(job, payload=payload), UrlStore(links, timeout=self.link_timeout)

    def _store_outputs(self, job: queue.Job, result: HandlerResult, store: ObjectStore | None = None) -> list:
        ids = []
        creator = str(job.creator_id) if job.creator_id else "unassigned"
        for i, out in enumerate(result.outputs):
            key = f"assets/{creator}/{job.id}/{i:02d}{out.path.suffix}"
            stored = (store or self.store).put_file(out.path, key)
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


def kinds_from_env(policy: Policy, env=None) -> list[str]:
    """The kinds this worker serves: MW_WORKER_KINDS (the autoscaler sets it on every machine, a merged family's
    machine included: every kind of every member family), else every kind of the families in MW_WORKER_FAMILIES
    (comma-separated, read from this image's policy), so one machine can serve several families without a rebuild."""
    env = os.environ if env is None else env
    kinds = [k.strip() for k in (env.get("MW_WORKER_KINDS") or "").split(",") if k.strip()]
    if kinds:
        return kinds
    families = {f.strip() for f in (env.get("MW_WORKER_FAMILIES") or "").split(",") if f.strip()}
    kinds = [k for k in (policy.get("job_kinds", default={}) or {}) if policy.family(k) in families]
    if not kinds:
        raise SystemExit("no kinds to serve: set MW_WORKER_KINDS or MW_WORKER_FAMILIES (or pass --kinds)")
    return kinds


RELEASE: dict[str, Callable[[], None]] = {}   # kind -> its module's release() (frees the GPU between families)


def handlers_for(kinds: list[str]) -> dict[str, Handler]:
    """Import only the handlers this container needs (GPU libraries are heavy)."""
    found: dict[str, Handler] = {}
    for kind in kinds:
        family = kind.split(".")[0]
        if family == "fake":
            from .handlers import fake as mod
        elif kind == "image.narrated":   # CPU edit that rides along on image machines
            from .handlers import edit_narrated as mod
        elif kind == "image.voice":      # Kokoro voice line, also on image machines
            from .handlers import tts_kokoro as mod
        elif kind == "image.qwen_edit":  # scene photos from references (creator.sets): image worker image, family edit
            from .handlers import image_qwen_edit as mod
        elif family == "image":
            from .handlers import image_zimage as mod
        elif family == "train":
            from .handlers import train_lora as mod
        elif kind == "talk.longcat":     # one continuous take: LongCat-Video-Avatar 1.6 on its own repo (talk image)
            from .handlers import talk_longcat as mod
        elif family == "talk":           # the image chooses the engine: ComfyUI + WanVideoWrapper, or the research script
            if os.environ.get("MW_TALK_ENGINE") == "comfy":
                from .handlers import talk_comfy as mod
            else:
                from .handlers import talk_infinitetalk as mod
        elif kind == "comfy.wan_animate2":   # dance (policy dance, off): Wan-Animate-2 on the ComfyUI image, family dance
            from .handlers import dance_animate as mod
        elif kind == "comfy.graph":     # any named ComfyUI workflow (worker/graphs, policy comfy_graphs)
            from .handlers import comfy_graph as mod
        elif family == "comfy":          # ComfyUI + WanVideoWrapper graphs on the talk worker image (story clips)
            from .handlers import comfy_video as mod
        elif family == "voice":          # designed voices (VoxCPM2), also on the talk worker image
            from .handlers import tts_voxcpm as mod
        elif family == "sound":          # generated effects and music beds (policy sound, off)
            from .handlers import sound_gen as mod
        elif kind in ("music.convert_voice", "music.train_voice"):   # one voice per artist (policy music.voice, off)
            from .handlers import music_voice as mod
        elif family == "music":          # original songs (policy music, off): ACE-Step 1.5 on the ComfyUI image
            from .handlers import music_song as mod
        elif kind == "hosted.avatar":    # hosted talking avatar (policy hosted_avatar, off), on the control plane
            from .handlers import hosted_avatar as mod
        elif family == "hosted":         # hosted APIs, on the control plane: no GPU machine
            from .handlers import hosted_video as mod
        elif family == "finish":         # the finish's and the takes' join encodes, on the finisher: no GPU
            from .handlers import finish_video as mod
        else:
            raise ValueError(f"no handler family for {kind!r}")
        found[kind] = mod.HANDLERS[kind]
        if callable(getattr(mod, "release", None)):
            RELEASE[kind] = mod.release
    return found


def start(kinds: list[str], *, pool: str, policy: Policy, store: ObjectStore, connect: Callable = db.connect,
          self_destroy: Callable[[], str | None] | None = None) -> Worker:
    """The worker `mw worker` runs (every rented machine starts this way). It refuses to run a GPU pool outside a
    registered machine, retries the database a few times, and if the queue stays unreachable destroys its own
    machine: one that can't reach the queue can never do work, so it must not be paid for."""
    check_registered(pool)
    vol = None
    if pool not in NO_GPU_POOLS:
        from . import volume
        vol = volume.check()   # a weights volume: stray paths deleted and free space measured before any download
        memory = gpu_memory_gb()
        fit = kinds_that_fit(kinds, policy, memory)
        if not fit:   # nothing this card can run: give the machine up rather than pay for it idling
            if self_destroy is None:
                from ..deploy.providers import self_destroy
            log.error("GPU has %s GB, below every kind's floor %s; destroyed own machine: %s", memory, kinds,
                      self_destroy())
            raise SystemExit(3)
        if fit != kinds:
            log.warning("GPU has %s GB: not serving %s (below their min_vram_gb)", memory,
                        sorted(set(kinds) - set(fit)))
        kinds = fit
    handlers = handlers_for(kinds)
    timeout = float(policy.get("queue", "connect_timeout_seconds", default=20))
    worker = None
    for attempt in range(int(policy.get("queue", "startup_connect_attempts", default=3))):
        try:
            worker = Worker(handlers=handlers, store=store, policy=policy, pool=pool,
                            connect=lambda url=None: connect(url, timeout=timeout), self_destroy=self_destroy)
            break
        except Exception as e:  # noqa: BLE001 - some rented hosts block outbound database traffic
            log.warning("database unreachable (attempt %d): %s", attempt + 1, type(e).__name__)
    if worker is None:
        if self_destroy is None:
            from ..deploy.providers import self_destroy
        log.error("database unreachable from this host; destroyed own machine: %s", self_destroy())
        raise SystemExit(2)
    if vol is not None:
        from . import volume
        volume.record(worker.conn, worker.group, vol)
        if vol["low"]:   # its first download would die of "Disk quota exceeded": give the machine up now
            log.error("weights volume %s has %.1f GB free, below the floor of %s GB; giving up this machine",
                      vol["path"], vol["free_gb"], vol["min_free_gb"])
            worker._retire_machine()
            raise SystemExit(4)
    return worker
