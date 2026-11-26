"""A shared weights volume, checked by each worker at boot (before any download).

On a pool with a network volume (policy gpu_pools.<pool>.network_volume) the autoscaler tells the machine where the
volume is mounted, which stray paths under it to delete (`prune`), its size and the free-space floor
(`min_free_gb`). At boot the worker deletes the listed paths, then measures free space. Below the floor the machine
cannot download a model without dying of "Disk quota exceeded" (2026-11-24: ~47 GB of LongCat weights in
/models/longcat filled the 150 GB volume and three machines died at their first download), so it reports the
measurement on its own machine's row (gpu_groups.volume_*) and gives the machine up; the autoscaler then stops
routing work to that volume's pools until a later check (policy network_volume.recheck_hours).

Environment (set by deploy/autoscale.launch_one from the policy, so changing the policy needs no image rebuild):
  MW_VOLUME_PATH         the mount (e.g. /models); unset: no volume, nothing checked
  MW_VOLUME_PRUNE        comma-separated paths relative to the mount, deleted at boot
  MW_VOLUME_SIZE_GB      the volume's provisioned size; free space is the lesser of the filesystem's own figure and
                         size minus what the files use (a network filesystem can report the cluster's free space,
                         not the volume's quota)
  MW_VOLUME_MIN_FREE_GB  the floor
  MW_VOLUME_PRUNE_IDLE_MINUTES  a listed path written to within this many minutes is in use by another machine and
                         is left alone this boot (0 or unset: no such check)

Shared by every worker that writes to the volume (owner 2026-11-26: no corrupted caches, no worker touching another
worker's active model files):
  exclusive(lock)        an O_EXCL lock file on the volume (flock is unreliable on network filesystems), refreshed
                         while held so a long download is never mistaken for a dead holder
  room(path, need)       before a download: the bytes still to fetch plus the floor must fit, else VolumeFull (a
                         machine fault: the job is released, never half-written)
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import threading
import time
from pathlib import Path

log = logging.getLogger("manyworlds.worker.volume")

GB = 1024 ** 3


def _within(mount: Path, entry: str) -> Path | None:
    """The path of a prune entry, only when it lies strictly under the mount (never the mount itself or outside it)."""
    entry = entry.strip()
    if not entry or os.path.isabs(entry) or ".." in Path(entry).parts:
        return None
    target = (mount / entry).resolve()
    root = mount.resolve()
    return target if root in target.parents else None


def _bytes_under(path: Path) -> int:
    total = 0
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    for d, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(d, f)).st_size
            except OSError:
                pass
    return total


def _last_write(path: Path) -> float:
    """The newest modification time of anything under `path` (itself included)."""
    newest = path.lstat().st_mtime
    if path.is_dir() and not path.is_symlink():
        for d, dirs, files in os.walk(path):
            for n in dirs + files:
                try:
                    newest = max(newest, os.lstat(os.path.join(d, n)).st_mtime)
                except OSError:
                    pass
    return newest


def prune(mount: str | os.PathLike, entries: list[str], protect: list[str] | None = None,
          idle_minutes: float = 0) -> list[dict]:
    """Delete each listed path under the mount. Entries that would leave the mount, or that hold (or are) a protected
    path (the weights cache the pool's machines share: MW_MODEL_DIR, HF_HOME), are refused and logged. A path written
    to within `idle_minutes` is in use by a machine still running and is kept for a later boot."""
    root = Path(mount)
    kept = [Path(p).resolve() for p in (protect or []) if p]
    out: list[dict] = []
    for entry in entries:
        if not entry.strip():
            continue
        target = _within(root, entry)
        if target is not None and any(k == target or target in k.parents for k in kept):
            target = None
        if target is None:
            log.error("volume prune: refusing %r (not a stray path strictly under %s)", entry, root)
            out.append({"path": entry, "refused": True})
            continue
        if not (target.exists() or target.is_symlink()):
            continue
        if idle_minutes and time.time() - _last_write(target) < float(idle_minutes) * 60:
            log.warning("volume prune: %s was written to in the last %s minutes; kept for now", target, idle_minutes)
            out.append({"path": str(target), "active": True})
            continue
        size = _bytes_under(target)
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
        log.warning("volume prune: deleted %s (%.1f GB)", target, size / GB)
        out.append({"path": str(target), "gb": round(size / GB, 1)})
    return out


def free_gb(mount: str | os.PathLike, size_gb: float | None = None) -> float:
    free = shutil.disk_usage(mount).free / GB
    if size_gb:
        free = min(free, float(size_gb) - _bytes_under(Path(mount)) / GB)
    return round(free, 1)


def check(env=None) -> dict | None:
    """Prune, then measure. None on a machine without a volume."""
    env = os.environ if env is None else env
    mount = env.get("MW_VOLUME_PATH")
    if not mount or not os.path.isdir(mount):
        return None
    pruned = prune(mount, [e for e in (env.get("MW_VOLUME_PRUNE") or "").split(",") if e.strip()],
                   protect=[env.get("MW_MODEL_DIR"), env.get("HF_HOME")],
                   idle_minutes=float(env.get("MW_VOLUME_PRUNE_IDLE_MINUTES") or 0))
    size = float(env["MW_VOLUME_SIZE_GB"]) if env.get("MW_VOLUME_SIZE_GB") else None
    free = free_gb(mount, size)
    floor = float(env["MW_VOLUME_MIN_FREE_GB"]) if env.get("MW_VOLUME_MIN_FREE_GB") else None
    low = floor is not None and free < floor
    (log.error if low else log.info)("volume %s: %.1f GB free (floor %s GB)", mount, free, floor)
    return {"path": mount, "pruned": pruned, "free_gb": free, "min_free_gb": floor, "low": low,
            "inventory": inventory(mount)}


def inventory(mount: str | os.PathLike, depth: int = 2, min_gb: float = 0.1) -> dict:
    """{path under the mount: GB} for folders up to `depth` levels down holding at least `min_gb`: what the volume
    held when this machine booted, so successive machines show which weights persisted."""
    root = Path(mount)
    out: dict[str, float] = {}

    def walk(p: Path, level: int):
        try:
            children = [c for c in p.iterdir() if c.is_dir() and not c.is_symlink() and not c.name.startswith(".")]
        except OSError:
            return
        for c in children:
            gb = _bytes_under(c) / GB
            if gb >= min_gb:
                out[str(c.relative_to(root))] = round(gb, 2)
                if level < depth:
                    walk(c, level + 1)
    walk(root, 1)
    return out


def record(conn, group: str | None, result: dict) -> None:
    """The measurement on this machine's own row (the worker role may write only these columns and its deadline)."""
    if not group:
        return
    from psycopg.types.json import Jsonb
    conn.execute("UPDATE gpu_groups SET volume_free_gb = %s, volume_low = %s, volume_checked_at = now(), "
                 "volume_inventory = %s WHERE name = %s",
                 (result["free_gb"], result["low"], Jsonb(result.get("inventory") or {}), group))


class VolumeFull(OSError):
    """Not enough room for a download. Its message carries "No space left on device", so the runner treats it as the
    machine's fault (worker.runner.WORKER_FAULT_MARKERS): the job is released for another machine, not failed."""


def room(path: str | os.PathLike, need_bytes: float, env=None) -> float:
    """Raise VolumeFull unless `need_bytes` plus the volume's floor fit in its free space, measured as the boot check
    measures it when `path` is on the volume, else on `path`'s own filesystem. Returns the free GB measured."""
    env = os.environ if env is None else env
    mount = env.get("MW_VOLUME_PATH")
    target = Path(path).resolve()
    on_volume = bool(mount) and (target == Path(mount).resolve() or Path(mount).resolve() in target.parents)
    if on_volume:
        size = float(env["MW_VOLUME_SIZE_GB"]) if env.get("MW_VOLUME_SIZE_GB") else None
        free = free_gb(mount, size)
        floor = float(env.get("MW_VOLUME_MIN_FREE_GB") or 0)
    else:
        probe = target
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        free, floor = round(shutil.disk_usage(probe).free / GB, 1), 0.0
    need = float(need_bytes) / GB
    if need + floor > free:
        raise VolumeFull(f"No space left on device (checked before downloading): {need:.1f} GB to fetch plus the "
                         f"{floor:.0f} GB floor, {free:.1f} GB free at {mount if on_volume else path}")
    return free


@contextlib.contextmanager
def exclusive(lock: str | os.PathLike, *, wait_s: float = 3600, stale_s: float = 7200, poll_s: float = 10,
              what: str = "a download"):
    """Hold `lock` (a file created with O_EXCL) for the block. The holder touches it every stale_s/4, so only a lock
    left by a dead machine goes stale; a waiter removes a stale lock and takes it. A volume too full to write even the
    lock file raises at once and never leaves a lock behind."""
    lock = Path(lock)
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_s
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
            except OSError:
                lock.unlink(missing_ok=True)
                raise
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale_s:   # its holder died
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                raise RuntimeError(f"another machine has held {lock.name} ({what}) for too long") from None
            time.sleep(poll_s)
    stop = threading.Event()

    def beat():
        while not stop.wait(max(stale_s / 4, 1)):
            try:
                os.utime(lock)
            except OSError:
                pass
    threading.Thread(target=beat, daemon=True).start()
    try:
        yield
    finally:
        stop.set()
        lock.unlink(missing_ok=True)


# Lock waits come from policy (gpu_pools.<pool>.network_volume.locks) through the environment the autoscaler sets;
# these defaults apply only where no volume is mounted (a container disk has no other machine downloading).
LOCK_DEFAULTS = {"MW_VOLUME_LOCK_WAIT_S": 3600.0, "MW_VOLUME_LOCK_STALE_S": 7200.0, "MW_HF_LOCK_WAIT_S": 1800.0}


def _env_seconds(name: str, env=None) -> float:
    env = os.environ if env is None else env
    try:
        return float(env.get(name) or LOCK_DEFAULTS[name])
    except ValueError:
        return LOCK_DEFAULTS[name]


def lock_waits(env=None) -> tuple[float, float]:
    """(wait_s, stale_s) for the weights download lock."""
    return _env_seconds("MW_VOLUME_LOCK_WAIT_S", env), _env_seconds("MW_VOLUME_LOCK_STALE_S", env)


def hf_lock_wait(env=None) -> float:
    """How long a Hugging Face load waits for another machine's download of the same repo."""
    return _env_seconds("MW_HF_LOCK_WAIT_S", env)
