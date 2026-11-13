"""Which GPU pool to rent next, from what pools have actually cost.

A pool's score is billed dollars per accepted output over a rolling window, with every
cost counted: boot, model download, idle time and machines that never produced
anything. The hourly price alone misleads: a cheap host that is slow, or fails half
the time, costs more per image than a pricier one that works.

Order for a job family (policy autoscale.scoring):
  1. trial: pools without enough outputs to rank yet, whose trial spend in the window is
     under trial_usd. New pools and pools whose data aged out get measured, for a bounded
     amount; among themselves they keep the policy order.
  2. ranked: pools with at least min_outputs, cheapest per accepted output first.
  3. exhausted: spent their trial without reaching min_outputs, ranked by cost per output
     (which may be infinite). Kept last so they still run when nothing else has capacity.
Paused pools are left out: their last `pause_after_failed_machines` machines for this
family all stopped without producing an output, the newest less than pause_hours ago.

An output is accepted unless a review gate failed it. Until review gates exist (day 4),
every produced output counts.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from ..policy import Policy


def _cfg(policy: Policy, family: str, key: str, default):
    v = policy.get("autoscale", "scoring", key, default=default)
    if isinstance(v, dict):
        return v.get(family, v.get("default", default))
    return v


def machine_stats(conn, family: str, pools: list[str], since: datetime) -> list[dict]:
    """Every machine of this family in these pools started since `since`, newest first,
    with what it billed and how many accepted outputs it produced."""
    return conn.execute(
        """
        SELECT g.name, g.pool, g.billed_usd::float AS billed_usd, g.started_at, g.stopped_at, g.stop_reason,
               (SELECT count(*) FROM assets s
                  JOIN attempts a ON a.job_id = s.job_id AND a.outcome = 'done' AND a.gpu_group = g.name
                 WHERE NOT EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id = s.id AND NOT r.passed)
               ) AS outputs
        FROM gpu_groups g
        WHERE g.family = %s AND g.pool = ANY(%s) AND g.started_at >= %s
        ORDER BY g.started_at DESC
        """, (family, pools, since)).fetchall()


def score_pools(conn, policy: Policy, family: str, pools: list[str], now: datetime) -> list[dict]:
    """One row per pool, in the order the autoscaler should try them (paused rows last)."""
    window = timedelta(days=float(_cfg(policy, family, "window_days", 7)))
    min_outputs = int(_cfg(policy, family, "min_outputs", 20))
    trial_usd = float(_cfg(policy, family, "trial_usd", 0.30))
    fail_streak = int(_cfg(policy, family, "pause_after_failed_machines", 3))
    pause = timedelta(hours=float(_cfg(policy, family, "pause_hours", 24)))
    by_pool: dict[str, list[dict]] = {p: [] for p in pools}
    for m in machine_stats(conn, family, pools, now - window):
        by_pool[m["pool"]].append(dict(m))

    rows = []
    for order, pool in enumerate(pools):
        ms = by_pool[pool]
        billed = sum(m["billed_usd"] for m in ms)
        outputs = sum(m["outputs"] for m in ms)
        stopped = [m for m in ms if m["stopped_at"] is not None]
        recent = stopped[:fail_streak]
        paused = (fail_streak > 0 and len(recent) == fail_streak and all(m["outputs"] == 0 for m in recent)
                  and recent[0]["stopped_at"] > now - pause)
        per_output = billed / outputs if outputs else math.inf
        if paused:
            status = "paused"
        elif outputs >= min_outputs:
            status = "ranked"
        elif billed < trial_usd:
            status = "trial"
        else:
            status = "exhausted"
        rows.append({"pool": pool, "status": status, "machines": len(ms), "billed_usd": round(billed, 4),
                     "outputs": outputs, "usd_per_output": per_output, "policy_order": order,
                     "failed_machines": sum(1 for m in stopped if m["outputs"] == 0)})
    rank = {"trial": 0, "ranked": 1, "exhausted": 2, "paused": 3}
    rows.sort(key=lambda r: (rank[r["status"]],
                             r["policy_order"] if r["status"] == "trial" else r["usd_per_output"],
                             r["policy_order"]))
    return rows


def launch_order(conn, policy: Policy, family: str, pools: list[str], now: datetime) -> tuple[list[str], list[dict]]:
    """Pools to try, best first, without paused ones; plus the full scoring for the log."""
    if not policy.get("autoscale", "scoring", "enabled", default=True):
        return list(pools), []
    rows = score_pools(conn, policy, family, pools, now)
    return [r["pool"] for r in rows if r["status"] != "paused"], rows
