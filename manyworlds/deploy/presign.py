"""Presigned storage links: rented GPU machines hold no storage key (policy `security.worker_storage: presigned`).

Workers claim straight from Postgres, so nothing on the control plane runs at claim time. Instead the control
plane keeps a fresh set of links on every job a rented machine may claim:

  sign       for one job: a presigned GET for every object of our bucket its payload names (references, LoRA,
             images, audio; found by scanning the payload for `s3://<bucket>/...`), GETs for the model mirrors of
             its family that exist (manifest and files, worker.models), and a presigned PUT for every output key it
             may write. Output keys are the runner's `assets/<creator>/<job id>/<NN><suffix>`: deterministic per
             job, so a rerun (preemption, retry) overwrites the same objects and handlers stay idempotent. The set
             goes on `payload._storage` with `expires` (when the links die) and `claim_by` (the last moment a
             worker may start the job: expires less the kind's longest attempt, gpu_runs.max_attempt_minutes_*,
             plus the upload margin).
  refresh    every autoscaler tick: jobs a worker could claim (queued, or running with a lapsed lease: a retry, a
             release, a preempted machine's reclaim) whose links are missing or whose `claim_by` is within
             `refresh_ahead_seconds` get new ones.
  claim      a worker in this mode (MW_STORAGE_MODE=presigned, set by credentials.worker_env) claims only jobs
             whose `claim_by` is still ahead (queue.claim `need_links`), so it never starts one whose links would
             die mid-attempt, and never one the refresher hasn't signed yet.

Kinds that run on our own machines (`security.presigned.keyed_kinds`: the Fly finisher, hosted APIs and the Brain
on the control plane) keep the store's keys and get no links. `*.mirror` jobs write keys only Hugging Face knows
(the repo's file list): they need a keyed run and fail in this mode (storage.MissingLink).
"""

from __future__ import annotations

import fnmatch
import json
import logging
import time
from typing import Any

from psycopg.types.json import Jsonb

from ..policy import Policy
from ..storage import ObjectStore

log = logging.getLogger("manyworlds.presign")

FIELD = "_storage"            # the payload key that carries a job's links; the runner strips it before the handler
MODE_ENV = "MW_STORAGE_MODE"  # set on rented machines by credentials.worker_env
MODE = "presigned"


def enabled(policy: Policy) -> bool:
    return str(policy.get("security", "worker_storage", default="shared")) == MODE


def _cfg(policy: Policy) -> dict:
    return policy.get("security", "presigned", default={}) or {}


def keyed(policy: Policy, kind: str) -> bool:
    """Whether the kind runs on our own machines, with the store's keys (no links)."""
    return any(fnmatch.fnmatch(kind, p) for p in (_cfg(policy).get("keyed_kinds") or []))


def attempt_seconds(policy: Policy, kind: str) -> float:
    """The longest one attempt of this kind may run (the enforcer stops a machine past it) plus the upload margin."""
    runs = policy.get("gpu_runs", default={}) or {}
    by_kind = runs.get("max_attempt_minutes_by_kind") or {}
    by_family = runs.get("max_attempt_minutes_by_family") or {}
    minutes = by_kind.get(kind, by_family.get(policy.family(kind), by_family.get("default")))
    return float(minutes) * 60 + float(_cfg(policy)["upload_margin_seconds"])


def _uris(node: Any, prefix: str, out: set[str]) -> set[str]:
    if isinstance(node, str):
        if node.startswith(prefix):
            out.add(node[len(prefix):])
    elif isinstance(node, dict):
        for k, v in node.items():
            if k != FIELD:
                _uris(v, prefix, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _uris(v, prefix, out)
    return out


def output_keys(policy: Policy, job: dict) -> list[str]:
    """Every key the runner (worker.runner._store_outputs) or the handler may write for this job."""
    cfg = _cfg(policy)
    creator = str(job["creator_id"]) if job.get("creator_id") else "unassigned"
    per_job = int((policy.get("job_kinds", job["kind"], default={}) or {}).get("outputs_per_job") or 0)
    slots = max(per_job, int(cfg["output_slots"]))
    keys = [f"assets/{creator}/{job['id']}/{i:02d}{suffix}"
            for i in range(slots) for suffix in cfg["output_suffixes"]]
    for pattern in (cfg.get("extra_puts") or {}).get(job["kind"]) or []:
        keys.append(pattern.format(job_id=job["id"], creator=creator))
    return keys


class _Mirrors:
    """Model mirror manifests read once per refresh run (worker.models layout)."""

    def __init__(self, store: ObjectStore):
        self.store, self.seen = store, {}

    def files(self, repo: str) -> list[str] | None:
        """The mirror's keys (manifest first), or None when the repo is not mirrored."""
        from ..worker import models

        if repo not in self.seen:
            key = models.manifest_key(repo)
            if not self.store.exists(key):
                self.seen[repo] = None
            else:
                manifest = json.loads(self.store.read_bytes(key))
                self.seen[repo] = [key] + [f"models/{repo}/files/{f['path']}" for f in manifest["files"]]
        return self.seen[repo]


def _mirrored_repos(policy: Policy, kind: str) -> list[str]:
    family = policy.family(kind)
    return [m["repo"] for m in (policy.get("models", "mirror", default=[]) or [])
            if policy.family(m["kind"]) == family]


def sign(policy: Policy, store: ObjectStore, job: dict, *, now: float | None = None,
         mirrors: _Mirrors | None = None) -> dict:
    """A fresh link set for one job (a jobs row: id, kind, payload, creator_id)."""
    cfg = _cfg(policy)
    now = time.time() if now is None else now
    budget = attempt_seconds(policy, job["kind"])
    ahead = float(cfg["refresh_ahead_seconds"])
    lifetime = min(max(float(cfg["link_seconds"]), budget + 2 * ahead), float(cfg["max_link_seconds"]))
    prefix = store.uri_for("")
    mirrors = mirrors or _Mirrors(store)
    gets, absent = {}, []
    for key in sorted(_uris(job.get("payload") or {}, prefix, set())):
        gets[key] = store.presign(key, "GET", int(lifetime))
    for repo in _mirrored_repos(policy, job["kind"]):
        files = mirrors.files(repo)
        if files is None:
            from ..worker import models
            absent.append(models.manifest_key(repo))
            continue
        for key in files:
            gets[key] = store.presign(key, "GET", int(lifetime))
    puts = {key: store.presign(key, "PUT", int(lifetime)) for key in output_keys(policy, job)}
    return {"prefix": prefix, "get": gets, "put": puts, "absent": absent,
            "expires": round(now + lifetime, 3), "claim_by": round(now + lifetime - budget, 3)}


def refresh(conn, policy: Policy, store: ObjectStore, *, now: float | None = None) -> int:
    """Sign or re-sign the links of every job a rented machine could claim next. Returns how many were signed."""
    if not enabled(policy):
        return 0
    cfg = _cfg(policy)
    now = time.time() if now is None else now
    rows = conn.execute(
        """SELECT id, kind, payload, creator_id FROM jobs
           WHERE (status = 'queued' OR (status = 'running' AND lease_until < now()))
             AND attempts < max_attempts
             AND coalesce((payload->%(f)s->>'claim_by')::float8, 0) < %(due)s
           ORDER BY priority DESC, created_at LIMIT %(n)s""",
        {"f": FIELD, "due": now + float(cfg["refresh_ahead_seconds"]), "n": int(cfg["refresh_batch"])}).fetchall()
    mirrors, signed = _Mirrors(store), 0
    for job in rows:
        if keyed(policy, job["kind"]):
            continue
        try:
            links = sign(policy, store, job, now=now, mirrors=mirrors)
        except Exception as e:  # noqa: BLE001 - one job's links failing must not stop the others
            log.warning("could not sign links for job %s: %s", job["id"], e)
            continue
        # Only while still claimable: a worker that took it meanwhile already holds the old (still valid) links.
        done = conn.execute(
            """UPDATE jobs SET payload = payload || jsonb_build_object(%s::text, %s::jsonb), updated_at = now()
               WHERE id = %s AND (status = 'queued' OR (status = 'running' AND lease_until < now()))
               RETURNING id""", (FIELD, Jsonb(links), job["id"])).fetchone()
        signed += bool(done)
    return signed


def needs_links(env=None) -> bool:
    """Whether this worker holds no storage key and needs links (a rented machine in presigned mode)."""
    import os
    env = os.environ if env is None else env
    return env.get(MODE_ENV) == MODE
