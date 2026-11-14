"""Model weights from our own bucket instead of Hugging Face.

Rented machines start blank, so every one used to download its model (~20 GB for the
image model) from Hugging Face at boot. On 2026-11-14 three Vast machines in a row stalled
doing that. Now the weights live in our object store (R2: no egress fees, no rate limits):

    mirror(repo)   one `<family>.mirror` job downloads the repo from Hugging Face once
                   and uploads every file; the manifest is written last, so a half-finished
                   mirror is never used
    ensure(repo)   workers load from the bucket when the manifest exists, in parallel,
                   checking sizes; otherwise they fall back to Hugging Face as before
    ensure_mirrored  the producer queues a mirror job for every policy `models.mirror`
                   repo that has no manifest yet

Layout: models/<repo>/manifest.json and models/<repo>/files/<path>.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import queue
from ..storage import ObjectStore

log = logging.getLogger("manyworlds.models")


def _prefix(repo: str) -> str:
    return f"models/{repo}"


def manifest_key(repo: str) -> str:
    return f"{_prefix(repo)}/manifest.json"


def local_dir(repo: str) -> Path:
    return Path(os.environ.get("MW_MODEL_DIR", "/models/mw")) / repo.replace("/", "--")


def _threads() -> int:
    return int(os.environ.get("MW_MODEL_THREADS", "8"))


def _manifest(store: ObjectStore, repo: str, workdir: Path) -> dict | None:
    if not store.exists(manifest_key(repo)):
        return None
    return json.loads(store.get_file(store.uri_for(manifest_key(repo)), workdir / "manifest.json").read_text())


def ensure(repo: str, store: ObjectStore, workdir: Path) -> str:
    """A local path with the repo's files when it is mirrored, else the repo id itself
    (so from_pretrained fetches from Hugging Face as before)."""
    d = local_dir(repo)
    if (d / ".complete").exists():
        return str(d)
    m = _manifest(store, repo, Path(workdir))
    if not m:
        log.warning("%s is not mirrored yet; loading from Hugging Face", repo)
        return repo

    def get(f):
        dest = d / f["path"]
        if dest.exists() and dest.stat().st_size == f["bytes"]:
            return
        tmp = dest.with_name(dest.name + ".part")
        store.get_file(store.uri_for(f"{_prefix(repo)}/files/{f['path']}"), tmp)
        if tmp.stat().st_size != f["bytes"]:
            raise RuntimeError(f"{f['path']}: got {tmp.stat().st_size} bytes, expected {f['bytes']}")
        tmp.replace(dest)

    with ThreadPoolExecutor(_threads()) as ex:
        list(ex.map(get, m["files"]))
    (d / ".complete").write_text(json.dumps({"files": len(m["files"])}))
    log.info("loaded %s from the bucket: %d files, %.1f GB", repo, len(m["files"]),
             sum(f["bytes"] for f in m["files"]) / 1e9)
    return str(d)


def mirror(repo: str, store: ObjectStore, workdir: Path, *, source_dir: Path | None = None) -> dict:
    """Copy a Hugging Face repo into the bucket. Idempotent: files already there with the
    right size are skipped, and the manifest is written only when every file is up."""
    if source_dir is None:
        from huggingface_hub import snapshot_download

        source_dir = Path(snapshot_download(repo, token=os.environ.get("HF_TOKEN") or None, max_workers=_threads()))
    files = sorted(p for p in Path(source_dir).rglob("*") if p.is_file() and ".cache" not in p.parts)
    entries = [{"path": p.relative_to(source_dir).as_posix(), "bytes": p.stat().st_size} for p in files]

    def put(pair):
        p, e = pair
        store.put_file(p, f"{_prefix(repo)}/files/{e['path']}")

    with ThreadPoolExecutor(_threads()) as ex:
        list(ex.map(put, zip(files, entries)))
    out = Path(workdir) / "manifest.json"
    out.write_text(json.dumps({"repo": repo, "files": entries}))
    store.put_file(out, manifest_key(repo))
    return {"repo": repo, "files": len(entries), "gb": round(sum(e["bytes"] for e in entries) / 1e9, 2)}


def ensure_mirrored(conn, policy, store: ObjectStore) -> list[dict]:
    """Queue a mirror job for each policy `models.mirror` repo that has no manifest yet."""
    out = []
    for m in policy.get("models", "mirror", default=[]) or []:
        repo, kind = m["repo"], m["kind"]
        if store.exists(manifest_key(repo)):
            continue
        if conn.execute("SELECT 1 FROM jobs WHERE kind = %s AND payload->>'repo' = %s AND status IN ('queued', 'running')",
                        (kind, repo)).fetchone():
            continue
        tries = conn.execute("SELECT count(*) n FROM jobs WHERE kind = %s AND payload->>'repo' = %s",
                             (kind, repo)).fetchone()["n"]
        if tries >= int(policy.get("models", "max_mirror_jobs", default=3)):
            out.append({"repo": repo, "action": "gave_up", "jobs": tries})
            continue
        job = queue.enqueue(conn, kind, {"repo": repo}, priority=int(policy.get("models", "priority", default=10)),
                            idempotency_key=f"mirror:{repo}:{tries}")
        out.append({"repo": repo, "action": "queued", "job": str(job)})
    return out
