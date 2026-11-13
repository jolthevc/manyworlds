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
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

from .. import db
from .. import policy as policy_mod
from ..policy import Policy
from . import enforce, scoring
from .providers import (LaunchSpec, NoCapacity, Provider, ProviderError, all_providers,
                        image_published, redact)

LEASE = "autoscale.tick"  # name of the lease row, not a tunable
PASS_THROUGH = ["DATABASE_URL", "STORAGE_URL", "S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY", "AWS_REGION", "HF_TOKEN", "MW_IMAGE_MODEL",
                "MW_VIDEO_MODEL", "MW_VIDEO_DISTILL_LORA"]


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


def _family(kind: str) -> str:
    return kind.split(".")[0]


def _family_kinds(policy: Policy, family: str) -> list[str]:
    return [k for k in policy.get("job_kinds", default={}) if _family(k) == family]


def waiting_work(conn) -> list[dict]:
    """Job kinds (and pinned pools) that have runnable jobs no worker holds."""
    return conn.execute(
        """
        SELECT kind, pools, count(*) AS n FROM jobs
        WHERE attempts < max_attempts
          AND (status = 'queued' OR (status = 'running' AND lease_until < now()))
        GROUP BY kind, pools ORDER BY kind
        """).fetchall()


def launch_one(conn, policy: Policy, providers: dict[str, Provider], kind: str, pool: str,
               now: datetime) -> dict:
    """Register then create one worker machine for `kind` in `pool`. Raises NoCapacity."""
    p = policy.pool(pool)
    provider = providers[p["provider"]]
    family = _family(kind)
    img = policy.get("worker_images", family)
    if not img:
        raise ProviderError(f"no worker_images entry for family {family!r}")
    gpu_name = policy.get("gpu_names", p["gpu"], provider.name)
    if not gpu_name:
        raise NoCapacity(f"no {provider.name} name for {p['gpu']} in policy gpu_names")
    runs = policy.get("gpu_runs", default={})
    hours = float(runs.get("max_hours_by_family", {}).get(family, runs.get("default_max_hours", 2)))
    hours = min(hours, float(runs.get("max_hours_limit", 12)))
    max_usd = float(runs.get("max_usd_by_family", {}).get(family, runs.get("default_max_usd", 1.0)))
    name = f"mw-{family}-{p['gpu'].removeprefix('rtx')}-{provider.name}-{uuid.uuid4().hex[:4]}"
    kinds = _family_kinds(policy, family)
    env = {k: os.environ[k] for k in PASS_THROUGH if os.environ.get(k)}
    # Tunables workers read from the environment, per family then per pool (policy).
    env.update({k: str(v) for k, v in (img.get("env") or {}).items()})
    env.update({k: str(v) for k, v in (p.get("worker_env") or {}).items()})
    env.update({"MW_WORKER_KINDS": ",".join(kinds), "MW_POOL": pool, "MW_GROUP": name,
                "MW_PROVIDER": provider.name})
    spec = LaunchSpec(image=f"{os.environ['MW_REGISTRY'].rstrip('/')}/{img['image']}:{img.get('tag', 'latest')}",
                      gpu_name=gpu_name, cpu=int(img["cpu"]), ram_gb=int(img["ram_gb"]),
                      disk_gb=int(img["disk_gb"]), max_usd_per_hour=float(p["usd_per_hour"]), env=env,
                      interruptible=bool(p.get("interruptible", False)),
                      priority=pool.split("_")[1] if p["provider"] == "salad" else "batch",
                      min_cuda=float(img["min_cuda"]) if img.get("min_cuda") else None)
    if not image_published(spec.image):
        raise ProviderError(f"worker image {spec.image} is not published; not renting a machine for it")
    # Registered first: if anything after this point dies, the enforcer still knows the
    # machine's deadline and cap and will stop it.
    conn.execute(
        """INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, replicas, job_kinds, deadline, max_usd,
                                   started_at, billed_until, last_busy_at, family)
           VALUES (%s, %s, %s, %s, 1, %s, %s, %s, %s, %s, %s, %s)""",
        (name, provider.name, pool, spec.max_usd_per_hour, kinds, now + timedelta(hours=hours), max_usd,
         now, now, now, family))
    try:
        m = provider.launch(name, spec)
    except Exception:
        conn.execute("DELETE FROM gpu_groups WHERE name = %s", (name,))  # never existed
        raise
    conn.execute("UPDATE gpu_groups SET provider_ref = %s, usd_per_hour = %s WHERE name = %s",
                 (m.ref, m.usd_per_hour, name))
    return {"group": name, "action": "launched", "kind": kind, "pool": pool,
            "usd_per_hour": m.usd_per_hour, "deadline_hours": hours, "max_usd": max_usd}


def scale_up(conn, policy: Policy, providers: dict[str, Provider], *, now: datetime) -> list[dict]:
    cfg = policy.get("autoscale", default={})
    max_groups = int(cfg.get("max_groups_per_kind", 1))
    cooldown = timedelta(minutes=float(cfg.get("relaunch_cooldown_minutes", 15)))
    daily_cap = float(policy.get("gpu_runs", "daily_gpu_usd", default=10))
    actions: list[dict] = []
    if enforce.billed_today(conn) >= daily_cap:
        work = waiting_work(conn)
        if work:
            actions.append({"action": "not_launching", "reason": "daily_gpu_budget",
                            "waiting": {w["kind"]: w["n"] for w in work}})
        return actions
    for w in waiting_work(conn):
        kind = w["kind"]
        if _family(kind) == "fake":
            continue
        pinned = bool(w["pools"])
        pools = list(w["pools"] or policy.get("job_kinds", kind, "pools", default=[]))
        open_groups = conn.execute(
            "SELECT name, pool FROM gpu_groups WHERE stopped_at IS NULL AND %s = ANY(job_kinds)", (kind,)).fetchall()
        if any(g["pool"] in pools for g in open_groups) or len(open_groups) >= max_groups:
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
        # Pinned work (benchmarks) runs where it was pinned; everything else goes to the
        # pool with the lowest measured cost per accepted output, skipping paused pools.
        order, scores = (pools, []) if pinned else scoring.launch_order(conn, policy, _family(kind), pools, now)
        if not order:
            actions.append({"action": "no_pool", "kind": kind, "reason": "every pool is paused",
                            "scores": _brief(scores)})
            continue
        tried = []
        for pool in order:
            info = policy.get("gpu_pools", pool)
            if not info or info["provider"] not in providers or not providers[info["provider"]].configured():
                continue
            try:
                launched = launch_one(conn, policy, providers, kind, pool, now)
                if scores:
                    launched["scores"] = _brief(scores)
                actions.append(launched)
                break
            except (NoCapacity, ProviderError) as e:
                tried.append({"pool": pool, "error": str(e)[:200]})
        else:
            actions.append({"action": "no_capacity", "kind": kind, "waiting": w["n"], "tried": tried})
    return actions


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


def tick(conn, policy: Policy, providers: dict[str, Provider], *, now: datetime | None = None,
         launch: bool = True, dry_run: bool = False) -> list[dict]:
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
        if launch and not dry_run:
            actions += scale_up(conn, policy, providers, now=now)
        return actions
    finally:
        conn.execute("DELETE FROM leases WHERE name = %s AND holder = %s", (LEASE, holder))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="autoscale")
    p.add_argument("--once", action="store_true")
    p.add_argument("--no-launch", action="store_true", help="enforce limits only; start nothing")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
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
        # Names, states, prices and errors only: GitHub Actions logs are public.
        interesting = [x for x in actions if x.get("action") != "ok"] if not a.once else actions
        if interesting:
            print(datetime.now(timezone.utc).strftime("%H:%M:%S"), json.dumps(interesting, default=str), flush=True)
        if a.once:
            return 0
        time.sleep(float(pol.get("autoscale", "tick_seconds", default=60)))


if __name__ == "__main__":
    sys.exit(main())
