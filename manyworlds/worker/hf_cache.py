"""Healing the Hugging Face cache on a shared weights volume (2026-11-26).

When the volume filled on 2026-11-24 ("Disk quota exceeded"), downloads in progress left zero-byte files in
HF_HOME/hub. huggingface_hub treats a present file as downloaded, so every later machine loaded an empty config.json
and failed (JSONDecodeError at from_pretrained). `heal(repo)` deletes a repo's zero-byte blobs and the
snapshot links that point at them, so the next load fetches just those files again; `forget(repo)` drops the
repo's whole cache entry, for a load that still fails on a corrupt file. Public model files only.

`load(repo, loader)` holds a per-repo lock on the volume while it heals and loads (worker.volume.exclusive): two
machines sharing the volume never fetch the same repo at once, and one machine's healing never deletes a file
another is still writing (huggingface_hub's own file locks are unreliable on a network filesystem). A blob still
being written (`*.incomplete`) is never touched."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

LOCK_WAIT_S = 1800   # a first download of the largest repo we load this way (Whisper large-v3, ~3 GB) fits in it


def _hub() -> Path:
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return Path(os.environ.get("HF_HUB_CACHE") or Path(home) / "hub")


def repo_dir(repo: str, hub: Path | None = None) -> Path:
    return (hub or _hub()) / f"models--{repo.replace('/', '--')}"


def heal(repo: str, hub: Path | None = None) -> list[str]:
    """Delete the repo's empty blobs and every snapshot entry that is empty or points at a missing or empty blob."""
    root = repo_dir(repo, hub)
    removed = []
    for blob in (root / "blobs").glob("*") if (root / "blobs").is_dir() else []:
        if blob.name.endswith(".incomplete"):
            continue   # being written: by this machine's own earlier attempt or, without the lock, another machine
        if blob.is_file() and blob.stat().st_size == 0:
            blob.unlink(missing_ok=True)
            removed.append(f"blobs/{blob.name}")
    for p in (root / "snapshots").rglob("*") if (root / "snapshots").is_dir() else []:
        if p.is_symlink() and not p.exists():
            p.unlink(missing_ok=True)
            removed.append(str(p.relative_to(root)))
        elif p.is_file() and not p.is_symlink() and p.stat().st_size == 0:
            p.unlink(missing_ok=True)
            removed.append(str(p.relative_to(root)))
    if removed:
        print(f"hf cache {repo}: removed {len(removed)} empty or dangling files", flush=True)
    return removed


def forget(repo: str, hub: Path | None = None) -> bool:
    root = repo_dir(repo, hub)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
        print(f"hf cache {repo}: dropped the whole entry", flush=True)
        return True
    return False


def load(repo: str, loader, *, corrupt=(ValueError,)):
    """`loader()` after healing the repo's cache, under the repo's lock; once more after forgetting the repo if it
    fails on a corrupt file (json.JSONDecodeError is a ValueError; safetensors header errors are too)."""
    from . import volume
    hub = _hub()
    with volume.exclusive(hub / ".mw-locks" / f"{repo_dir(repo, hub).name}.lock", wait_s=LOCK_WAIT_S,
                          what=f"loading {repo}"):
        heal(repo, hub)
        try:
            return loader()
        except corrupt:
            if not forget(repo, hub):
                raise
            return loader()
