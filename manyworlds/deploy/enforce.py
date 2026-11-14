"""Cloud-side GPU spend enforcer. Runs every 10 minutes on GitHub Actions (see
.github/workflows/gpu-enforcer.yml) so no limit depends on a laptop staying awake.

Each tick, for every registered group that is still up:
  1. meter billed instance time since the last tick into the ledger ('gpu_billed');
  2. stop it if it is past its deadline, over its own spend cap, idle (no open jobs
     for its kinds) after its boot grace, stuck (jobs waiting but no sign of life from
     its worker for gpu_runs.stuck_minutes), or if all groups together are over the
     day's GPU budget.
Any provider group named mw-* that was never registered (so has no deadline) is
stopped immediately. API or database failures exit non-zero so the run shows red.

    python -m manyworlds.deploy.enforce            # one tick
    python -m manyworlds.deploy.enforce --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Protocol

from .. import db, ledger
from .. import policy as policy_mod
from ..policy import Policy


class GroupAPI(Protocol):
    def list_names(self) -> list[str]: ...
    def stop_delete(self, name: str) -> None: ...


class SaladGroups:
    def list_names(self) -> list[str]:
        from .salad import _req

        res = _req("GET", "/organizations/{org}/projects/{project}/containers") or {}
        items = res.get("items", res) if isinstance(res, dict) else res
        return [g["name"] for g in items]

    def stop_delete(self, name: str) -> None:
        from .salad import _req

        base = f"/organizations/{{org}}/projects/{{project}}/containers/{name}"
        # Stop is refused while a group is Pending (still being placed); delete is what ends
        # billing and works in any state, so a failed stop must never skip it.
        try:
            _req("POST", f"{base}/stop")
        except Exception:  # noqa: BLE001
            pass
        _req("DELETE", base)


def _open_jobs(conn, kinds: list[str]) -> int:
    return conn.execute(
        "SELECT count(*) AS n FROM jobs WHERE kind = ANY(%s) AND status IN ('queued', 'running')",
        (kinds,)).fetchone()["n"]


def idle_stop_minutes(policy: Policy, provider: str | None) -> float:
    """How long a machine may sit with no work before it is stopped. Short where machines
    start in minutes (idling costs more than restarting), longer where starting is slow."""
    cfg = policy.get("gpu_runs", default={})
    by = cfg.get("idle_stop_minutes_by_provider") or {}
    return float(by.get(provider, cfg.get("idle_stop_minutes", 20)))


def billed_today(conn) -> float:
    row = conn.execute(
        "SELECT coalesce(sum(usd), 0) AS u FROM cost_entries "
        "WHERE category = 'gpu_billed' AND created_at >= date_trunc('day', now() AT TIME ZONE 'utc') AT TIME ZONE 'utc'"
    ).fetchone()
    return float(row["u"])


def tick(conn, policy: Policy, api: GroupAPI, *, now: datetime | None = None, dry_run: bool = False) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    cfg = policy.get("gpu_runs", default={})
    grace_min = float(cfg.get("boot_grace_minutes", 45))
    daily_cap = float(cfg.get("daily_gpu_usd", 10))
    stuck_min = float(cfg.get("stuck_minutes", 10))
    actions: list[dict] = []

    live = set(api.list_names())
    groups = conn.execute("SELECT * FROM gpu_groups WHERE stopped_at IS NULL ORDER BY started_at").fetchall()
    registered = {g["name"] for g in groups}

    # Meter every registered group first, so the daily check sees current spend. A group
    # that has disappeared (a worker destroyed its own machine, or the provider reclaimed
    # it) is billed up to now, which over-counts by at most one tick, then closed.
    for g in groups:
        seconds = max((now - g["billed_until"]).total_seconds(), 0)
        usd = seconds / 3600 * float(g["usd_per_hour"]) * int(g["replicas"])
        if seconds and not dry_run:
            ledger.record(conn, category="gpu_billed", provider=g["pool"], quantity=round(seconds, 1),
                          unit="instance_second", usd=usd, policy_version=policy.version)
            conn.execute("UPDATE gpu_groups SET billed_until = %s, billed_usd = billed_usd + %s WHERE name = %s",
                         (now, usd, g["name"]))
        g["billed_usd"] = float(g["billed_usd"]) + usd
        if g["name"] not in live:
            if not dry_run:
                conn.execute("UPDATE gpu_groups SET stopped_at = %s, stop_reason = 'gone' WHERE name = %s",
                             (now, g["name"]))
            actions.append({"group": g["name"], "action": "marked gone", "billed_usd": round(g["billed_usd"], 4)})

    spent_today = billed_today(conn) if not dry_run else None

    for g in groups:
        if g["name"] not in live:
            continue
        busy = _open_jobs(conn, list(g["job_kinds"])) > 0
        last_busy = now if busy else g["last_busy_at"]
        if busy and not dry_run:
            conn.execute("UPDATE gpu_groups SET last_busy_at = %s WHERE name = %s", (now, g["name"]))
        # Boot grace covers image pull and model download, until the machine starts its first
        # job. After that, idle time is just idle time.
        # Last sign of life: an attempt starting or finishing, or a heartbeat on a job it holds.
        last_seen = conn.execute(
            """SELECT max(greatest(a.started_at, a.finished_at,
                                   CASE WHEN j.status = 'running' AND j.worker_id = a.worker_id THEN j.updated_at END)) AS t
               FROM attempts a JOIN jobs j ON j.id = a.job_id WHERE a.gpu_group = %s""", (g["name"],)).fetchone()["t"]
        started_work = last_seen is not None
        booting = not started_work and (now - g["started_at"]).total_seconds() <= grace_min * 60
        idle_min = idle_stop_minutes(policy, g["provider"])
        reason = None
        if now >= g["deadline"]:
            reason = "deadline"
        elif g["billed_usd"] >= float(g["max_usd"]):
            reason = "max_usd"
        elif spent_today is not None and spent_today >= daily_cap:
            reason = "daily_gpu_budget"
        elif not booting and (now - last_busy).total_seconds() > idle_min * 60:
            reason = "idle"
        elif started_work and busy and (now - last_seen).total_seconds() > stuck_min * 60:
            # Work is waiting but this machine has gone silent (worker hung or died): it would
            # otherwise count as busy until its deadline and block a replacement.
            reason = "stuck"
        if reason:
            if not dry_run:
                try:
                    api.stop_delete(g["name"])
                except Exception as e:  # noqa: BLE001 - keep enforcing the others; retry next tick
                    actions.append({"group": g["name"], "action": "stop_failed", "reason": reason,
                                    "error": str(e)[:300]})
                    continue
                conn.execute("UPDATE gpu_groups SET stopped_at = %s, stop_reason = %s WHERE name = %s",
                             (now, reason, g["name"]))
            actions.append({"group": g["name"], "action": "stopped", "reason": reason,
                            "billed_usd": round(g["billed_usd"], 4)})
        else:
            actions.append({"group": g["name"], "action": "ok", "busy": busy,
                            "billed_usd": round(g["billed_usd"], 4), "deadline": g["deadline"].isoformat()})

    # Anything of ours running without a registered deadline gets stopped on sight.
    for name in sorted(live - registered):
        if name.startswith("mw-"):
            if not dry_run:
                try:
                    api.stop_delete(name)
                except Exception as e:  # noqa: BLE001
                    actions.append({"group": name, "action": "stop_failed", "reason": "unregistered",
                                    "error": str(e)[:300]})
                    continue
            actions.append({"group": name, "action": "stopped", "reason": "unregistered"})
    return actions


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="enforce")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    from .autoscale import MultiAPI
    from .providers import all_providers

    conn = db.connect()
    db.migrate(conn)
    pol = policy_mod.load()
    actions = tick(conn, pol, MultiAPI(conn, all_providers(pol)), dry_run=a.dry_run)
    # Group names, states and amounts only: these logs are public on a public repo.
    print(json.dumps(actions, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
