"""Autoscaler: GPU machines follow the job queue, with no one starting or stopping them.

Each tick:
  1. enforce: meter billed time, stop machines past their deadline, over their spend
     cap, idle, over the day's GPU budget, or unregistered (manyworlds.deploy.enforce);
  2. scale up: for each job kind with waiting jobs and no machine serving it, start one
     in the first pool (policy job_kinds.<kind>.pools, or the job's own pinned pools)
     whose provider is configured and has capacity. Every machine is registered with a
     deadline and spend cap BEFORE it is created.

Three independent layers stop machines: this loop (run always-on), GitHub Actions running
the same tick on a schedule, and workers on Vast/RunPod destroying their own machine when
idle or past deadline. Concurrent ticks are serialised with a lease row (works through connection poolers).

    python -m manyworlds.deploy.autoscale --once      # one tick
    python -m manyworlds.deploy.autoscale             # loop every autoscale.tick_seconds
    python -m manyworlds.deploy.autoscale --once --no-launch   # enforce only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

from .. import db
from .. import policy as policy_mod
from ..policy import Policy
from . import credentials, enforce, presign, scoring
from .providers import (LaunchSpec, NoCapacity, Provider, ProviderError, all_providers,
                        image_published, redact)

LEASE = "autoscale.tick"  # name of the lease row, not a tunable


class MultiAPI:
    """The enforcer's view of every configured provider at once. A provider whose API
    fails this tick reports its registered machines as still live, so they are metered
    and kept under their limits rather than wrongly marked gone."""

    def __init__(self, conn, providers: dict[str, Provider]):
        self.where: dict[str, tuple[Provider, str | None]] = {}
        self.errors: dict[str, str] = {}
        self.prices: dict[str, float] = {}  # what a provider says a machine costs now
        for pname, p in providers.items():
            try:
                if not p.configured():
                    # No key here (e.g. a backstop without every provider's secret): we
                    # can't see its machines, which is not the same as them being gone.
                    # Marking them gone would tell their workers to self-destroy.
                    raise ProviderError(f"{pname} not configured in this environment")
                for name, ref in p.list_live().items():
                    self.where[name] = (p, ref)
                self.prices.update(getattr(p, "prices", None) or {})
            except ProviderError as e:
                if p.configured():
                    self.errors[pname] = str(e)[:300]
                for g in conn.execute("SELECT name, provider_ref FROM gpu_groups "
                                      "WHERE provider = %s AND stopped_at IS NULL", (pname,)).fetchall():
                    self.where[g["name"]] = (p, g["provider_ref"])

    def list_names(self) -> list[str]:
        return sorted(self.where)

    def stop_delete(self, name: str) -> None:
        p, ref = self.where[name]
        if not p.configured():
            raise ProviderError(f"cannot stop {name}: {p.name} not configured in this environment")
        p.destroy(name, ref)


def _family(policy: Policy, kind: str) -> str:
    return policy.family(kind)   # job_kinds.<kind>.family, else the prefix (image.qwen_edit -> edit)


def _family_kinds(policy: Policy, family: str) -> list[str]:
    group = merge_groups(policy).get(family)
    if group:   # a merged machine serves every kind of its member families
        return [k for k in policy.get("job_kinds", default={}) if _family(policy, k) in group["families"]]
    return [k for k in policy.get("job_kinds", default={}) if _family(policy, k) == family]


# Merged families (autoscale.merge_families, 2026-11-23): families that run on the same worker image can share ONE
# machine, so work that crosses them (walk-and-talk: voice, then comfy, then talk) pays one cold boot instead of three.
# {name: [family, ...]} or {name: {families: [...], tag, disk_gb, ram_gb, max_hours, max_usd, max_attempt_minutes}}.
# Absent (the shipped policy): every family has its own machines, as before. The breadth test's overlay sets it, and
# the autoscaler reads the overlay only for work in the overlay's batches (policy.for_batches).

def merge_groups(policy: Policy) -> dict[str, dict]:
    out = {}
    for name, cfg in (policy.get("autoscale", "merge_families", default={}) or {}).items():
        cfg = {"families": list(cfg)} if isinstance(cfg, (list, tuple)) else dict(cfg or {})
        if cfg.get("families"):
            out[str(name)] = {**cfg, "families": [str(f) for f in cfg["families"]]}
    return out


def machine_family(policy: Policy, kind: str) -> str:
    """The family of the machine that serves `kind`: its merge group when it is in one, else its own family."""
    fam = _family(policy, kind)
    return next((name for name, g in merge_groups(policy).items() if fam in g["families"]), fam)


def _tag_order(tag: str) -> tuple:
    import re
    return tuple(int(n) for n in re.findall(r"\d+", str(tag)))


def merged_image(policy: Policy, name: str) -> dict:
    """The worker image of a merged machine: its members' image (they must share one), the newest member tag unless
    the group names one, enough of every resource for any member, and every member's weights on its disk."""
    group = merge_groups(policy)[name]
    imgs = [policy.get("worker_images", f) for f in group["families"] if policy.get("worker_images", f)]
    if not imgs or len({i["image"] for i in imgs}) != 1:
        raise ProviderError(f"merged family {name!r}: its families must share one worker image")
    downloads = [float(i.get("download_gb", 0)) for i in imgs]
    disks = [int(i["disk_gb"]) for i in imgs]
    env: dict = {}
    for i in imgs:
        env.update(i.get("env") or {})
    cudas = [float(i["min_cuda"]) for i in imgs if i.get("min_cuda")]
    return {"image": imgs[0]["image"],
            "tag": str(group.get("tag") or max((str(i.get("tag", "latest")) for i in imgs), key=_tag_order)),
            "cpu": int(group.get("cpu") or max(int(i["cpu"]) for i in imgs)),
            "ram_gb": int(group.get("ram_gb") or max(int(i["ram_gb"]) for i in imgs)),
            # each member downloads its own weights on first use: the largest disk plus the others' downloads
            "disk_gb": int(group.get("disk_gb") or max(disks) + round(sum(downloads) - max(downloads))),
            # on a pool with a weights volume the weights live there, so the container needs only the largest disk
            "volume_disk_gb": int(group.get("volume_disk_gb") or max(disks)),
            "download_gb": sum(downloads), "env": env, **({"min_cuda": max(cudas)} if cudas else {})}


def _limit(policy: Policy, family: str, by_family_key: str, group_key: str, default_key: str, default: float) -> float:
    """A per-family machine limit (gpu_runs.<by_family_key>); a merged family without its own entry takes the group's
    `group_key`, else the largest of its members' limits."""
    runs = policy.get("gpu_runs", default={}) or {}
    by_family = runs.get(by_family_key) or {}
    fallback = float(by_family.get("default", runs.get(default_key, default)))
    if family in by_family:
        return float(by_family[family])
    group = merge_groups(policy).get(family)
    if group:
        if group.get(group_key) is not None:
            return float(group[group_key])
        return max(float(by_family.get(f, fallback)) for f in group["families"])
    return fallback


def waiting_work(conn, batches: list[str] | None = None) -> list[dict]:
    """Job kinds (and pinned pools) that have runnable jobs no worker holds; with `batches` (LIKE patterns),
    only jobs in those batches (paused operations: tests and benchmarks only). `batches` in each row: the batches
    the work is in (the test overlay applies to some)."""
    return conn.execute(
        """
        SELECT kind, pools, count(*) AS n, min(created_at) AS oldest, max(priority) AS top_priority,
               array_agg(DISTINCT coalesce(batch, '')) AS batches,
               bool_and(coalesce(batch, '') LIKE 'post-%%') AS production FROM jobs
        WHERE attempts < max_attempts
          AND (status = 'queued' OR (status = 'running' AND lease_until < now()))
          AND (%(b)s::text[] IS NULL OR coalesce(batch, '') LIKE ANY(%(b)s::text[]))
        GROUP BY kind, pools ORDER BY kind
        """, {"b": batches}).fetchall()


def _per_family(cfg: dict, key: str, family: str, default):
    v = cfg.get(key, default)
    return v.get(family, v.get("default", default)) if isinstance(v, dict) else v


def seconds_per_job(conn, policy: Policy, kind: str) -> float:
    """Typical wall time of one job of this kind, from recent finished attempts."""
    return job_seconds(conn, policy, kind)[0]


def job_seconds(conn, policy: Policy, kind: str) -> tuple[float, str]:
    """(typical wall time of one job of this kind, where it came from): "measured" (median of finished attempts
    within autoscale.scale.history_days) or "policy" (autoscale.scale.default_job_seconds)."""
    cfg = policy.get("autoscale", "scale", default={})
    row = conn.execute(
        """SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY a.gpu_seconds) AS s FROM attempts a
           JOIN jobs j ON j.id = a.job_id
           WHERE j.kind = %s AND a.outcome = 'done' AND a.gpu_seconds IS NOT NULL
             AND a.started_at > now() - make_interval(days => %s)""",
        (kind, int(cfg.get("history_days", 7)))).fetchone()
    if row and row["s"]:
        return float(row["s"]), "measured"
    return float(_per_family(cfg, "default_job_seconds", _family(policy, kind), 300)), "policy"


def batch_wait(conn, policy: Policy, family: str, work: list[dict], now: datetime) -> dict | None:
    """Why a family's waiting work should wait for more, or None to start a machine now.

    Every machine bills a boot, a model load and an idle tail (~7-10 min on RunPod) however many
    jobs it runs, and production makes posts lead_hours ahead, so a few minutes of work can wait
    to share a machine with more. Measured 2026-11-14: machines that ran one job each billed
    2-6x their job time. Work starts when min_work_minutes of it waits, when the oldest job has
    waited max_wait_minutes, or at once when anything other than production work waits (creator lab,
    benchmarks: someone is waiting on those) or a job has urgent_priority."""
    cfg = policy.get("autoscale", "batch", default={})
    if not cfg:
        return None
    rows = [w for w in work if _family(policy, w["kind"]) == family and not w["pools"]]
    if not rows:
        return None
    if not all(w["production"] for w in rows) or \
            max(int(w["top_priority"] or 0) for w in rows) >= int(cfg.get("urgent_priority", 1)):
        return None
    minutes = sum(int(w["n"]) * seconds_per_job(conn, policy, w["kind"]) for w in rows) / 60
    waited = (now - min(w["oldest"] for w in rows)).total_seconds() / 60
    need = float(_per_family(cfg, "min_work_minutes", family, 0))
    max_wait = float(_per_family(cfg, "max_wait_minutes", family, 0))
    if minutes >= need or waited >= max_wait:
        return None
    return {"work_minutes": round(minutes, 1), "need_minutes": need, "waited_minutes": round(waited),
            "max_wait_minutes": max_wait}


def machines_wanted(conn, policy: Policy, kind: str, waiting: int) -> int:
    """Enough machines to clear the backlog within the drain target. Each machine pays a
    boot (minutes) once, so small backlogs stay on one machine."""
    cfg = policy.get("autoscale", "scale", default={})
    drain_s = float(_per_family(cfg, "drain_target_minutes", _family(policy, kind), 30)) * 60
    return max(1, math.ceil(waiting * seconds_per_job(conn, policy, kind) / drain_s))


def attempt_limit_minutes(policy: Policy, family: str) -> float:
    """How long one job attempt may run on a machine of this family before the enforcers stop it (a merged family:
    the longest of its members, unless it has its own)."""
    return _limit(policy, family, "max_attempt_minutes_by_family", "max_attempt_minutes", "default_max_attempt_minutes", 20)


def launch_one(conn, policy: Policy, providers: dict[str, Provider], kind: str, pool: str,
               now: datetime, *, family: str | None = None) -> dict:
    """Register then create one worker machine for `kind` in `pool`. Raises NoCapacity. `family`: the machine's
    family when it is not the kind's own (a merged family, autoscale.merge_families: one machine for all its members'
    kinds, on their shared image)."""
    p = policy.pool(pool)
    provider = providers[p["provider"]]
    family = family or _family(policy, kind)
    img = merged_image(policy, family) if family in merge_groups(policy) else policy.get("worker_images", family)
    if not img:
        raise ProviderError(f"no worker_images entry for family {family!r}")
    gpu_name = policy.get("gpu_names", p["gpu"], provider.name)
    if not gpu_name:
        raise NoCapacity(f"no {provider.name} name for {p['gpu']} in policy gpu_names")
    runs = policy.get("gpu_runs", default={})
    hours = _limit(policy, family, "max_hours_by_family", "max_hours", "default_max_hours", 2)
    hours = min(hours, float(runs.get("max_hours_limit", 12)))
    max_usd = _limit(policy, family, "max_usd_by_family", "max_usd", "default_max_usd", 1.0)
    name = f"mw-{family}-{p['gpu'].removeprefix('rtx')}-{provider.name}-{uuid.uuid4().hex[:4]}"
    kinds = _family_kinds(policy, family)
    try:
        env = credentials.worker_env(policy)   # scoped credentials only: the host is not ours
    except credentials.MissingWorkerCredentials as e:
        raise ProviderError(str(e)) from None
    # Tunables workers read from the environment, per family then per pool (policy).
    env.update({k: str(v) for k, v in (img.get("env") or {}).items()})
    env.update({k: str(v) for k, v in (p.get("worker_env") or {}).items()})
    members = merge_groups(policy).get(family, {}).get("families") or [family]
    env.update({"MW_WORKER_KINDS": ",".join(kinds), "MW_WORKER_FAMILIES": ",".join(members),
                "MW_POOL": pool, "MW_GROUP": name,
                "MW_PROVIDER": provider.name,
                # so a worker image built before this pool existed still knows its price and provider
                "MW_POOL_INFO": json.dumps({k: v for k, v in p.items() if k != "worker_env"})})
    spec = LaunchSpec(image=f"{os.environ['MW_REGISTRY'].rstrip('/')}/{img['image']}:{img.get('tag', 'latest')}",
                      gpu_name=gpu_name, cpu=int(img["cpu"]), ram_gb=int(img["ram_gb"]),
                      disk_gb=int(img["disk_gb"]), max_usd_per_hour=float(p["usd_per_hour"]), env=env,
                      interruptible=bool(p.get("interruptible", False)),
                      priority=pool.split("_")[1] if p["provider"] == "salad" else "batch",
                      min_cuda=float(img["min_cuda"]) if img.get("min_cuda") else None,
                      cloud=str(p.get("cloud", "COMMUNITY")), download_gb=float(img.get("download_gb", 0)),
                      network_volume=p.get("network_volume") or None)
    if spec.network_volume:   # weights cached on the volume: downloaded once, not by every machine
        spec.env.update(MW_MODEL_DIR=f"{spec.network_volume['path']}/mw", HF_HOME=f"{spec.network_volume['path']}/hf")
        spec.download_gb = 0.0
        if img.get("volume_disk_gb"):   # a merged machine: no member's weights on the container disk
            spec.disk_gb = int(img["volume_disk_gb"])
    if not image_published(spec.image):
        raise ProviderError(f"worker image {spec.image} is not published; not renting a machine for it")
    # Registered first: if anything after this point dies, the enforcer still knows the
    # machine's deadline and cap and will stop it.
    conn.execute(
        """INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, replicas, job_kinds, deadline, max_usd,
                                   started_at, billed_until, last_busy_at, family, image, max_attempt_minutes)
           VALUES (%s, %s, %s, %s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (name, provider.name, pool, spec.max_usd_per_hour, kinds, now + timedelta(hours=hours), max_usd,
         now, now, now, family, f"{img['image']}:{img.get('tag', 'latest')}", attempt_limit_minutes(policy, family)))
    try:
        m = provider.launch(name, spec)
    except Exception:
        conn.execute("DELETE FROM gpu_groups WHERE name = %s", (name,))  # never existed
        raise
    conn.execute("UPDATE gpu_groups SET provider_ref = %s, usd_per_hour = %s, est_extra_usd = %s WHERE name = %s",
                 (m.ref, m.usd_per_hour, getattr(m, "upfront_usd", 0.0), name))
    return {"group": name, "action": "launched", "kind": kind, "pool": pool,
            "usd_per_hour": m.usd_per_hour, "deadline_hours": hours, "max_usd": max_usd}


def _work_pools(policy: Policy, w: dict) -> list[str]:
    return list(w["pools"] or policy.get("job_kinds", w["kind"], "pools", default=[]) or [])


def fly_work(policy: Policy, w: dict) -> bool:
    """Waiting work that only machines of this Fly app run (the finisher's finish.video)."""
    pools = _work_pools(policy, w)
    return bool(pools) and all(policy.get("gpu_pools", p, "provider") == "fly" for p in pools)


# The finisher (finish.video, worker/handlers/finish_video.py): a machine of this Fly app (fly.toml process group
# `finisher`, a larger VM than the control plane's), not a rented GPU. The GPU launch path assumes a worker image, a
# GPU name, a registry and scoped worker credentials, none of which apply, so it has this step of its own. Like every
# machine it is registered (deadline and spend cap: gpu_runs.*_by_family.finish) before it starts, metered into the
# ledger as gpu_billed by the enforcer every tick, and stopped by it (deadline, cap, idle, stuck, the day's budget);
# its worker also exits when its queue has been empty for the pool's idle_exit_seconds (the machine then stops and is
# marked gone) and stops its own machine past its deadline. It is started, never created: fly deploy makes the
# machine (its size and restart policy are fly.toml's).
def ensure_finisher(conn, policy: Policy, providers: dict[str, Provider], *, now: datetime) -> list[dict]:
    """Start the stopped finisher when its jobs wait and none is up; one machine per fly pool."""
    provider = providers.get("fly")
    pools = [name for name, p in (policy.get("gpu_pools", default={}) or {}).items() if p.get("provider") == "fly"]
    if provider is None or not pools or not provider.configured():
        return []
    from .. import operations
    work = [w for w in waiting_work(conn, operations.test_batch_patterns(policy)) if fly_work(policy, w)]
    cooldown = timedelta(minutes=float(policy.get("autoscale", "relaunch_cooldown_minutes", default=15)))
    actions: list[dict] = []
    for pool in pools:
        waiting = [w for w in work if pool in _work_pools(policy, w)]
        if not waiting:
            continue
        kinds = sorted({w["kind"] for w in waiting})
        if conn.execute("SELECT 1 FROM gpu_groups WHERE pool = %s AND stopped_at IS NULL", (pool,)).fetchone():
            continue   # already up (or starting): it serves the queue
        over = over_budget(conn, policy)
        if over:
            actions.append({"action": "not_launching", "reason": over, "pool": pool,
                            "waiting": {w["kind"]: w["n"] for w in waiting}})
            continue
        # A run that finished no job (it crashed at start, or failed everything) waits out the relaunch cooldown, so
        # a broken release can't start the finisher in a loop. A run that did work may stop for any reason: its worker
        # exits once the queue is empty, and the machine is then marked gone.
        recent = conn.execute(
            """SELECT g.name, g.stop_reason FROM gpu_groups g
               WHERE g.pool = %s AND g.stopped_at > %s AND coalesce(g.stop_reason, '') <> 'manual'
                 AND NOT EXISTS (SELECT 1 FROM attempts a WHERE a.gpu_group = g.name AND a.outcome = 'done')
               ORDER BY g.stopped_at DESC LIMIT 1""", (pool, now - cooldown)).fetchone()
        if recent:
            actions.append({"action": "cooldown", "kind": kinds[0], "after": recent["name"],
                            "reason": recent["stop_reason"]})
            continue
        family = _family(policy, kinds[0])
        streak = no_output_streak(conn, policy, family, now, [pool])
        if streak:
            actions.append({"action": "not_launching", "reason": "no_output_streak", "kind": kinds[0], **streak})
            continue
        try:
            actions.append(launch_fly(conn, policy, provider, pool, family, now))
        except (NoCapacity, ProviderError) as e:
            actions.append({"action": "no_capacity", "kind": kinds[0], "waiting": sum(int(w["n"]) for w in waiting),
                            "tried": [{"pool": pool, "error": redact(str(e))[:200]}]})
    return actions


def launch_fly(conn, policy: Policy, provider: Provider, pool: str, family: str, now: datetime) -> dict:
    """Register then start the stopped machine of a fly pool's process group. Raises NoCapacity / ProviderError (the
    registration is then removed)."""
    p = policy.pool(pool)
    runs = policy.get("gpu_runs", default={})
    hours = min(_limit(policy, family, "max_hours_by_family", "max_hours", "default_max_hours", 2),
                float(runs.get("max_hours_limit", 12)))
    max_usd = _limit(policy, family, "max_usd_by_family", "max_usd", "default_max_usd", 1.0)
    name = f"mw-{family}-{provider.name}-{uuid.uuid4().hex[:4]}"
    usd = float(p["usd_per_hour"])
    # Registered first, as every machine: if anything after this dies, the enforcer knows its deadline and cap.
    conn.execute(
        """INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, replicas, job_kinds, deadline, max_usd,
                                   started_at, billed_until, last_busy_at, family, image, max_attempt_minutes)
           VALUES (%s, %s, %s, %s, 1, %s, %s, %s, %s, %s, %s, %s, NULL, %s)""",
        (name, provider.name, pool, usd, _family_kinds(policy, family), now + timedelta(hours=hours), max_usd,
         now, now, now, family, attempt_limit_minutes(policy, family)))
    spec = LaunchSpec(image="", gpu_name=str(p.get("gpu") or "none"), cpu=0, ram_gb=0, disk_gb=0,
                      max_usd_per_hour=usd, process_group=p.get("process_group"))
    try:
        m = provider.launch(name, spec)
    except Exception:
        conn.execute("DELETE FROM gpu_groups WHERE name = %s", (name,))  # never started
        raise
    conn.execute("UPDATE gpu_groups SET provider_ref = %s, usd_per_hour = %s WHERE name = %s",
                 (m.ref, m.usd_per_hour, name))
    return {"group": name, "action": "launched", "kind": ",".join(_family_kinds(policy, family)), "pool": pool,
            "usd_per_hour": m.usd_per_hour, "deadline_hours": hours, "max_usd": max_usd}


def over_budget(conn, policy: Policy) -> str | None:
    """Why no machine may start now (today's GPU spend or today's whole bill at its cap), or None."""
    from .. import ledger
    daily_cap = float(policy.get("gpu_runs", "daily_gpu_usd", default=10))
    total_cap = policy.get("budgets", "daily_usd")
    return ("daily_gpu_budget" if enforce.billed_today(conn) >= daily_cap else
            "daily_total_budget" if total_cap is not None and ledger.bill_today(conn) >= float(total_cap) else None)


def overlay_missing(conn, kind: str, error: "policy_mod.OverlayMissing", now: datetime) -> dict:
    """Test work whose overlay file is gone gets no machine (policy.OverlayMissing). The owner hears once per
    missing file a day (a deduplicated owner event); other work goes on launching."""
    from .. import notify
    notify.owner_event(conn, "test_overlay_missing", subject={"path": error.path},
                       payload={"kind": kind, "batches": error.batches, "error": str(error)},
                       dedupe_key=f"test_overlay_missing:{error.path}:{now:%Y-%m-%d}")
    return {"action": "overlay_missing", "kind": kind, "path": error.path, "batches": error.batches}


def scale_up(conn, policy: Policy, providers: dict[str, Provider], *, now: datetime) -> list[dict]:
    cfg = policy.get("autoscale", default={})
    cooldown = timedelta(minutes=float(cfg.get("relaunch_cooldown_minutes", 15)))
    actions: list[dict] = []
    unfunded: set[str] = set()   # providers whose account is out of credit this tick
    from . import serverless
    if serverless.pools(policy):   # settle finished serverless requests whatever the caps say
        actions += serverless.poll(conn, policy, now=now)
    from .. import operations
    allowed = operations.test_batch_patterns(policy)   # None when live; else tests and benchmarks only
    over = over_budget(conn, policy)
    if over:
        work = waiting_work(conn, allowed)
        if work:
            actions.append({"action": "not_launching", "reason": over,
                            "waiting": {w["kind"]: w["n"] for w in work}})
        return actions
    # Work for a machine of this Fly app (the finisher) is ensure_finisher's: no rented machine, no worker credentials.
    work = [w for w in waiting_work(conn, allowed) if not fly_work(policy, w)]
    gaps = credentials.missing(policy)
    if gaps and work:   # say it once, plainly, instead of as every pool's launch error
        actions.append({"action": "not_launching", "reason": "worker_credentials", "missing": gaps,
                        "waiting": {w["kind"]: w["n"] for w in work}})
        return actions
    if serverless.pools(policy):   # serverless-served kinds get requests, not pods
        actions += serverless.dispatch(conn, policy, work, now=now)
    for w in work:
        kind = w["kind"]
        if _family(policy, kind) == "fake":
            continue
        pinned = bool(w["pools"])
        if serverless.serves(policy, kind, w["pools"]):
            continue   # a live serverless pool serves it (manyworlds.deploy.serverless)
        pools = list(w["pools"] or policy.get("job_kinds", kind, "pools", default=[]))
        if all(policy.get("gpu_pools", p, "provider") == "local" for p in pools):
            continue  # runs on the control host (e.g. reviews), not on rented GPUs
        # The machine family: the kind's own, or a merge group (autoscale.merge_families) the work's batches use.
        try:
            mpol = policy_mod.for_batches(policy, w.get("batches"))
        except policy_mod.OverlayMissing as e:   # never run test work on the plain policy: it boots more machines
            actions.append(overlay_missing(conn, kind, e, now))
            continue
        merged = machine_family(mpol, kind)
        merged = merged if merged in merge_groups(mpol) else None
        mfam = merged or _family(policy, kind)
        max_groups = int(_per_family(cfg, "max_groups_per_kind", mfam, 1))
        open_groups = conn.execute(
            "SELECT name, pool FROM gpu_groups WHERE stopped_at IS NULL AND %s = ANY(job_kinds)", (kind,)).fetchall()
        serving = [g for g in open_groups if g["pool"] in pools]
        if len(open_groups) >= max_groups:
            continue
        # Machines follow the backlog: one per drain-target's worth of work, one more per
        # tick at most, so a burst of jobs ramps up instead of renting a fleet at once.
        # A pinned benchmark measures one machine's speed and cost: a second machine only doubles the boot and
        # download it pays (2026-11-22: 3 talk jobs started two L40S, each downloading ~82 GB).
        wanted = int(policy.get("autoscale", "bench_machines", default=1)) if pinned \
            else machines_wanted(conn, policy, kind, int(w["n"]))
        if serving and len(serving) >= wanted:
            continue
        # No machine up for this work yet: wait until there is enough to be worth a boot.
        held = None if (pinned or open_groups) else batch_wait(conn, policy, _family(policy, kind), work, now)
        if held:
            actions.append({"action": "batching", "kind": kind, **held})
            continue
        # A machine for this work stopped for a reason other than finishing (crashed,
        # vanished, hit its cap): wait before paying for another, so a broken image or
        # job can't start machines in a loop.
        recent = conn.execute(
            """SELECT name, stop_reason FROM gpu_groups
               WHERE %s = ANY(job_kinds) AND pool = ANY(%s) AND stopped_at > %s
                 AND coalesce(stop_reason, '') NOT IN ('idle', 'manual')
               ORDER BY stopped_at DESC LIMIT 1""", (kind, pools, now - cooldown)).fetchone()
        if recent:
            actions.append({"action": "cooldown", "kind": kind, "after": recent["name"],
                            "reason": recent["stop_reason"]})
            continue
        streak = no_output_streak(conn, policy, mfam, now, pools)
        if streak:   # pinned benchmarks too: they are what ran the talk loop of 2026-11-20
            actions.append({"action": "not_launching", "reason": "no_output_streak", "kind": kind, **streak})
            continue
        # Pinned work (benchmarks) runs where it was pinned; everything else goes to the
        # pool with the lowest measured cost per accepted output, skipping paused pools.
        order, scores = (pools, []) if pinned else scoring.launch_order(conn, policy, mfam,
                                                                        pools, now)
        if not order:
            actions.append({"action": "no_pool", "kind": kind, "reason": "every pool is paused",
                            "scores": _brief(scores)})
            continue
        tried = []
        for pool in order:
            info = policy.get("gpu_pools", pool)
            if not info or info["provider"] not in providers or not providers[info["provider"]].configured():
                continue
            if info["provider"] in unfunded:
                continue
            try:
                launched = (launch_one(conn, mpol, providers, kind, pool, now, family=merged) if merged
                            else launch_one(conn, policy, providers, kind, pool, now))
                if scores:
                    launched["scores"] = _brief(scores)
                actions.append(launched)
                break
            except (NoCapacity, ProviderError) as e:
                tried.append({"pool": pool, "error": str(e)[:200]})
                # An account out of credit fails every pool of that provider: say so once, plainly,
                # and go straight to the next provider.
                if any(m in str(e).lower() for m in ("insufficient_credit", "lacks credit", "insufficient funds",
                                                     "balance too low")):
                    if info["provider"] not in unfunded:
                        unfunded.add(info["provider"])
                        actions.append({"action": "provider_unfunded", "provider": info["provider"]})
        else:
            actions.append({"action": "no_capacity", "kind": kind, "waiting": w["n"], "tried": tried})
    return actions


def no_output_streak(conn, policy: Policy, family: str, now: datetime, pools: list[str] | None = None) -> dict | None:
    """The family's last `machines` machines in these `pools` (within `hours`, on the current worker image) all
    tried work and finished none, or never started: renting another would most likely fail the same way. Seven
    talk machines in a row did exactly that on 2026-11-20. Counted per set of pools, since a failure on one card
    (out of memory on a 48 GB L40S) says nothing about another (an 80 GB H100). A new worker image tag, or the
    window passing, gives it a new try."""
    cfg = policy.get("autoscale", "no_output_guard", default={}) or {}
    n, hours = int(cfg.get("machines", 0)), float(cfg.get("hours", 24))
    if n <= 0:
        return None
    img = policy.get("worker_images", family) or {}
    image = f"{img.get('image')}:{img.get('tag', 'latest')}" if img else None
    rows = conn.execute(
        """SELECT g.name, g.image, g.stop_reason,
                  count(a.id) FILTER (WHERE a.outcome = 'done') AS done, count(a.id) AS tried
           FROM gpu_groups g LEFT JOIN attempts a ON a.gpu_group = g.name
           WHERE g.family = %s AND g.stopped_at > %s AND coalesce(g.stop_reason, '') <> 'manual'
             AND (%s::text[] IS NULL OR g.pool = ANY(%s::text[]))
           GROUP BY g.name, g.image, g.stop_reason, g.stopped_at
           ORDER BY g.stopped_at DESC LIMIT %s""", (family, now - timedelta(hours=hours), pools, pools, n)).fetchall()
    failed = [r for r in rows if not r["done"] and (r["tried"] or r["stop_reason"] == "never_started")
              and (image is None or r["image"] in (None, image))]
    if len(rows) < n or len(failed) < n:
        return None
    return {"machines": [r["name"] for r in failed], "image": image, "hours": hours}


def _brief(scores: list[dict]) -> list[dict]:
    return [{"pool": s["pool"], "status": s["status"], "outputs": s["outputs"], "billed_usd": s["billed_usd"],
             "usd_per_output": None if s["usd_per_output"] == float("inf") else round(s["usd_per_output"], 5)}
            for s in scores]


def update_prices(conn, prices: dict[str, float]) -> None:
    """Meter at what the provider says it charges now (a Vast instance costs more than the
    offer it was rented from: storage is added), not the price seen at launch."""
    for name, usd in prices.items():
        conn.execute("UPDATE gpu_groups SET usd_per_hour = %s WHERE name = %s AND stopped_at IS NULL "
                     "AND usd_per_hour <> %s", (usd, name, usd))


def refresh_links(conn, policy: Policy, store=None) -> list[dict]:
    """Presigned mode (security.worker_storage: presigned): sign or re-sign the storage links of every job a rented
    machine could claim (manyworlds.deploy.presign). A failure is reported, never fatal to the tick."""
    try:
        if store is None:
            from .. import storage
            store = storage.from_url()
        n = presign.refresh(conn, policy, store)
    except Exception as e:  # noqa: BLE001 - the enforcer and the rest of the tick must still run
        return [{"action": "links_failed", "error": f"{type(e).__name__}: {e}"[:300]}]
    return [{"action": "links_signed", "jobs": n}] if n else []


def tick(conn, policy: Policy, providers: dict[str, Provider], *, now: datetime | None = None,
         launch: bool = True, dry_run: bool = False, store=None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    holder = uuid.uuid4().hex
    lease_s = float(policy.get("autoscale", "tick_lease_seconds", default=600))
    got = conn.execute(
        """INSERT INTO leases (name, holder, until) VALUES (%(n)s, %(h)s, now() + make_interval(secs => %(s)s))
           ON CONFLICT (name) DO UPDATE SET holder = excluded.holder, until = excluded.until
           WHERE leases.until < now()
           RETURNING holder""", {"n": LEASE, "h": holder, "s": lease_s}).fetchone()
    if not got:
        return [{"action": "skipped", "reason": "another tick is running"}]
    try:
        api = MultiAPI(conn, providers)
        actions = [{"action": "provider_error", "provider": k, "error": v} for k, v in api.errors.items()]
        if not dry_run:
            update_prices(conn, api.prices)
        actions += enforce.tick(conn, policy, api, now=now, dry_run=dry_run)
        if not dry_run and presign.enabled(policy):   # before launching: a machine claims only jobs with links
            actions += refresh_links(conn, policy, store)
        if launch and not dry_run:
            actions += scale_up(conn, policy, providers, now=now)   # settles serverless requests too
            actions += ensure_finisher(conn, policy, providers, now=now)
        elif not dry_run:
            from . import serverless
            if serverless.pools(policy):   # the backstop (--no-launch) still cancels serverless requests run too long
                actions += serverless.poll(conn, policy, now=now)
        if not dry_run:
            actions += maybe_reconcile(conn, policy, providers)
        return actions
    finally:
        conn.execute("DELETE FROM leases WHERE name = %s AND holder = %s", (LEASE, holder))


def maybe_reconcile(conn, policy: Policy, providers: dict) -> list[dict]:
    """Pull providers' itemised charges into the ledger at most every `reconcile_every_minutes`."""
    every = float(policy.get("autoscale", "reconcile_every_minutes", default=60)) * 60
    due = conn.execute(
        """INSERT INTO leases (name, holder, until) VALUES ('reconcile', 'autoscaler', now() + make_interval(secs => %s))
           ON CONFLICT (name) DO UPDATE SET until = excluded.until WHERE leases.until < now()
           RETURNING name""", (every,)).fetchone()
    if not due:
        return []
    from . import reconcile
    try:
        out = reconcile.run(conn, policy, providers)
    except Exception as e:  # noqa: BLE001 - billing lookups must never stop enforcement or launches
        return [{"action": "reconcile_failed", "error": redact(str(e))[:300]}]
    return [{"action": "reconciled", **out}] if out["groups"] else []


def housekeeping(conn, policy: Policy | None = None, store=None, now: datetime | None = None) -> list[dict]:
    """Rules that hold whatever the master switch says, on the one loop that always runs: raw research
    items are deleted once past their retention date (research privacy), fixed monthly costs (policy
    `fixed_costs`, e.g. the weights volume) accrue to the ledger daily, and the JPEG copies made for Instagram
    are deleted once their post is out or given up (manyworlds.publish.public_copies)."""
    from .. import ledger
    from ..research.apify import purge_expired
    out: list[dict] = []
    try:
        n = purge_expired(conn)
        if policy is not None:
            ledger.accrue_fixed(conn, policy.get("fixed_costs", default=[]) or [])
        out += [{"action": "purged_research_items", "items": n}] if n else []
    except Exception as e:  # noqa: BLE001 - housekeeping never stops enforcement; retried next tick
        out.append({"action": "housekeeping_failed", "error": redact(f"{type(e).__name__}: {e}")[:300]})
    if policy is not None:
        from .. import storage
        from ..publish import public_copies
        try:
            r = public_copies.purge(conn, policy, store or storage.from_url, now=now)
            out += [{"action": "purged_public_copies", **r}] if r["posts"] or r["failed"] else []
        except Exception as e:  # noqa: BLE001 - as above: retried next tick
            out.append({"action": "housekeeping_failed", "error": redact(f"{type(e).__name__}: {e}")[:300]})
    return out


def daily_summary(conn, policy: Policy, now: datetime | None = None) -> list[dict]:
    """Once a day: the owner's summary of yesterday (manyworlds.summary), sent by notify_owner right after."""
    from .. import summary
    try:
        s = summary.send_if_due(conn, policy, now)
        return [{"action": "daily_summary", "day": s["day"]}] if s else []
    except Exception as e:  # noqa: BLE001 - the summary must never stop the loop that stops machines
        return [{"action": "daily_summary_failed", "error": str(e)[:200]}]


def storage_retention(conn, policy: Policy, store=None, now: datetime | None = None) -> list[dict]:
    """Once a day: delete the files of rejected media past the retention window (manyworlds.retention)."""
    from .. import retention, storage
    try:
        now = now or datetime.now(timezone.utc)
        if not retention.due(conn, policy, now=now):
            return []
        out = retention.run(conn, policy, store or storage.from_url(), now=now)
        return [{"action": "storage_retention", **out}] if out["deleted"] or out["failed"] else []
    except Exception as e:  # noqa: BLE001 - retention must never stop the loop that stops machines
        return [{"action": "storage_retention_failed", "error": str(e)[:200]}]


def model_selection(conn, policy: Policy, now: datetime | None = None) -> list[dict]:
    """Once every llm.selection.every_hours: propose model switches the shadow evidence supports (approvals, which
    notify_owner then pushes). Stored evidence only, so it spends nothing and runs while operations are paused."""
    from .. import selection
    try:
        if not selection.due(conn, policy, now=now):
            return []
        results = selection.run(conn, policy, now=now)
        selection.record(conn, policy, results)
        proposed = [r["role"] for r in results if r.get("proposed")]
        return [{"action": "model_selection", "proposed": proposed}] if proposed else []
    except Exception as e:  # noqa: BLE001 - selection must never stop the loop that stops machines
        return [{"action": "model_selection_failed", "error": str(e)[:200]}]


def notify_owner(conn, policy: Policy, actions: list[dict], now: datetime | None = None) -> list[dict]:
    """Record the actions that need the owner and push everything they haven't been told (manyworlds.notify).
    Runs on every loop, paused or not; a failure here never stops enforcement."""
    from .. import notify
    now = now or datetime.now(timezone.utc)
    try:
        notify.from_autoscaler(conn, policy, actions, now)
        sent = notify.deliver(conn, policy, now=now)
        return [{"action": "notified", "items": len(sent)}] if sent else []
    except Exception as e:  # noqa: BLE001
        return [{"action": "notify_failed", "error": redact(f"{type(e).__name__}: {e}")[:300]}]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="autoscale")
    p.add_argument("--once", action="store_true")
    p.add_argument("--no-launch", action="store_true", help="enforce limits only; start nothing")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    # docker stop sends SIGTERM; exiting normally runs the tick's finally, which releases the
    # tick lease, so a restart doesn't leave the next autoscaler waiting out tick_lease_seconds.
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    conn = db.connect()
    db.migrate(conn)
    while True:
        pol = policy_mod.load()
        try:
            actions = tick(conn, pol, all_providers(pol), launch=not a.no_launch, dry_run=a.dry_run)
        except Exception as e:  # noqa: BLE001 - a loop must survive one bad tick
            if a.once:
                raise
            actions = [{"action": "tick_failed", "error": redact(str(e))[:300]}]
            conn = db.connect()
        if not a.dry_run:
            actions += housekeeping(conn, pol)
            actions += model_selection(conn, pol)
            actions += storage_retention(conn, pol)
            actions += daily_summary(conn, pol)
            actions += notify_owner(conn, pol, actions)
        # Names, states, prices and errors only: GitHub Actions logs are public.
        interesting = [x for x in actions if x.get("action") != "ok"] if not a.once else actions
        if interesting:
            print(datetime.now(timezone.utc).strftime("%H:%M:%S"), json.dumps(interesting, default=str), flush=True)
        if a.once:
            return 0
        time.sleep(float(pol.get("autoscale", "tick_seconds", default=60)))


if __name__ == "__main__":
    sys.exit(main())
