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
"""

from __future__ import annotations

import logging
import os
import shutil
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


def prune(mount: str | os.PathLike, entries: list[str], protect: list[str] | None = None) -> list[dict]:
    """Delete each listed path under the mount. Entries that would leave the mount, or that hold (or are) a protected
    path (the weights cache the pool's machines share: MW_MODEL_DIR, HF_HOME), are refused and logged."""
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
                   protect=[env.get("MW_MODEL_DIR"), env.get("HF_HOME")])
    size = float(env["MW_VOLUME_SIZE_GB"]) if env.get("MW_VOLUME_SIZE_GB") else None
    free = free_gb(mount, size)
    floor = float(env["MW_VOLUME_MIN_FREE_GB"]) if env.get("MW_VOLUME_MIN_FREE_GB") else None
    low = floor is not None and free < floor
    (log.error if low else log.info)("volume %s: %.1f GB free (floor %s GB)", mount, free, floor)
    return {"path": mount, "pruned": pruned, "free_gb": free, "min_free_gb": floor, "low": low}


def record(conn, group: str | None, result: dict) -> None:
    """The measurement on this machine's own row (the worker role may write only these columns and its deadline)."""
    if not group:
        return
    conn.execute("UPDATE gpu_groups SET volume_free_gb = %s, volume_low = %s, volume_checked_at = now() "
                 "WHERE name = %s", (result["free_gb"], result["low"], group))
