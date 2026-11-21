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
from . import credentials, enforce, scoring
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


def _family(kind: str) -> str:
    return kind.split(".")[0]


def _family_kinds(policy: Policy, family: str) -> list[str]:
    return [k for k in policy.get("job_kinds", default={}) if _family(k) == family]


def waiting_work(conn, batches: list[str] | None = None) -> list[dict]:
    """Job kinds (and pinned pools) that have runnable jobs no worker holds; with `batches` (LIKE patterns),
    only jobs in those batches (paused operations: tests and benchmarks only)."""
    return conn.execute(
        """
        SELECT kind, pools, count(*) AS n, min(created_at) AS oldest, max(priority) AS top_priority,
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
    cfg = policy.get("autoscale", "scale", default={})
    row = conn.execute(
        """SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY a.gpu_seconds) AS s FROM attempts a
           JOIN jobs j ON j.id = a.job_id
           WHERE j.kind = %s AND a.outcome = 'done' AND a.gpu_seconds IS NOT NULL
             AND a.started_at > now() - make_interval(days => %s)""",
        (kind, int(cfg.get("history_days", 7)))).fetchone()
    if row and row["s"]:
        return float(row["s"])
    return float(_per_family(cfg, "default_job_seconds", _family(kind), 300))


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
    rows = [w for w in work if _family(w["kind"]) == family and not w["pools"]]
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
    drain_s = float(_per_family(cfg, "drain_target_minutes", _family(kind), 30)) * 60
    return max(1, math.ceil(waiting * seconds_per_job(conn, policy, kind) / drain_s))


def attempt_limit_minutes(policy: Policy, family: str) -> float:
    """How long one job attempt may run on a machine of this family before the enforcers stop it."""
    by_family = policy.get("gpu_runs", "max_attempt_minutes_by_family", default={}) or {}
    return float(by_family.get(family, by_family.get("default", 20)))


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
    try:
        env = credentials.worker_env(policy)   # scoped credentials only: the host is not ours
    except credentials.MissingWorkerCredentials as e:
        raise ProviderError(str(e)) from None
    # Tunables workers read from the environment, per family then per pool (policy).
    env.update({k: str(v) for k, v in (img.get("env") or {}).items()})
    env.update({k: str(v) for k, v in (p.get("worker_env") or {}).items()})
    env.update({"MW_WORKER_KINDS": ",".join(kinds), "MW_POOL": pool, "MW_GROUP": name,
                "MW_PROVIDER": provider.name,
                # so a worker image built before this pool existed still knows its price and provider
                "MW_POOL_INFO": json.dumps({k: v for k, v in p.items() if k != "worker_env"})})
    spec = LaunchSpec(image=f"{os.environ['MW_REGISTRY'].rstrip('/')}/{img['image']}:{img.get('tag', 'latest')}",
                      gpu_name=gpu_name, cpu=int(img["cpu"]), ram_gb=int(img["ram_gb"]),
                      disk_gb=int(img["disk_gb"]), max_usd_per_hour=float(p["usd_per_hour"]), env=env,
                      interruptible=bool(p.get("interruptible", False)),
                      priority=pool.split("_")[1] if p["provider"] == "salad" else "batch",
                      min_cuda=float(img["min_cuda"]) if img.get("min_cuda") else None,
                      cloud=str(p.get("cloud", "COMMUNITY")), download_gb=float(img.get("download_gb", 0)))
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


def scale_up(conn, policy: Policy, providers: dict[str, Provider], *, now: datetime) -> list[dict]:
    cfg = policy.get("autoscale", default={})
    cooldown = timedelta(minutes=float(cfg.get("relaunch_cooldown_minutes", 15)))
    daily_cap = float(policy.get("gpu_runs", "daily_gpu_usd", default=10))
    actions: list[dict] = []
    unfunded: set[str] = set()   # providers whose account is out of credit this tick
    from .. import operations
    allowed = operations.test_batch_patterns(policy)   # None when live; else tests and benchmarks only
    from .. import ledger
    total_cap = policy.get("budgets", "daily_usd")
    over = ("daily_gpu_budget" if enforce.billed_today(conn) >= daily_cap else
            "daily_total_budget" if total_cap is not None and ledger.bill_today(conn) >= float(total_cap) else None)
    if over:
        work = waiting_work(conn, allowed)
        if work:
            actions.append({"action": "not_launching", "reason": over,
                            "waiting": {w["kind"]: w["n"] for w in work}})
        return actions
    work = waiting_work(conn, allowed)
    gaps = credentials.missing(policy)
    if gaps and work:   # say it once, plainly, instead of as every pool's launch error
        actions.append({"action": "not_launching", "reason": "worker_credentials", "missing": gaps,
                        "waiting": {w["kind"]: w["n"] for w in work}})
        return actions
    for w in work:
        kind = w["kind"]
        if _family(kind) == "fake":
            continue
        pinned = bool(w["pools"])
        pools = list(w["pools"] or policy.get("job_kinds", kind, "pools", default=[]))
        if all(policy.get("gpu_pools", p, "provider") == "local" for p in pools):
            continue  # runs on the control host (e.g. reviews), not on rented GPUs
        max_groups = int(_per_family(cfg, "max_groups_per_kind", _family(kind), 1))
        open_groups = conn.execute(
            "SELECT name, pool FROM gpu_groups WHERE stopped_at IS NULL AND %s = ANY(job_kinds)", (kind,)).fetchall()
        serving = [g for g in open_groups if g["pool"] in pools]
        if len(open_groups) >= max_groups:
            continue
        # Machines follow the backlog: one per drain-target's worth of work, one more per
        # tick at most, so a burst of jobs ramps up instead of renting a fleet at once.
        if serving and len(serving) >= machines_wanted(conn, policy, kind, int(w["n"])):
            continue
        # No machine up for this work yet: wait until there is enough to be worth a boot.
        held = None if (pinned or open_groups) else batch_wait(conn, policy, _family(kind), work, now)
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
        streak = no_output_streak(conn, policy, _family(kind), now)
        if streak:   # pinned benchmarks too: they are what ran the talk loop of 2026-11-20
            actions.append({"action": "not_launching", "reason": "no_output_streak", "kind": kind, **streak})
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
            if info["provider"] in unfunded:
                continue
            try:
                launched = launch_one(conn, policy, providers, kind, pool, now)
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


def no_output_streak(conn, policy: Policy, family: str, now: datetime) -> dict | None:
    """The family's last `machines` machines (within `hours`, on the current worker image) all tried work and
    finished none, or never started: renting another would most likely fail the same way. Seven talk machines
    in a row did exactly that on 2026-11-20. A new worker image tag, or the window passing, gives it a new try."""
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
           GROUP BY g.name, g.image, g.stop_reason, g.stopped_at
           ORDER BY g.stopped_at DESC LIMIT %s""", (family, now - timedelta(hours=hours), n)).fetchall()
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


def housekeeping(conn) -> list[dict]:
    """Rules that hold whatever the master switch says, on the one loop that always runs: raw research
    items are deleted once past their retention date (research privacy)."""
    from ..research.apify import purge_expired
    try:
        n = purge_expired(conn)
        return [{"action": "purged_research_items", "items": n}] if n else []
    except Exception as e:  # noqa: BLE001 - housekeeping never stops enforcement; retried next tick
        return [{"action": "housekeeping_failed", "error": redact(f"{type(e).__name__}: {e}")[:300]}]


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
            actions += housekeeping(conn)
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
