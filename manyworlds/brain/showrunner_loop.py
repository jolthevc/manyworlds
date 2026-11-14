"""Keeps every creator planned ahead: asks the showrunner for the next period when a creator has
less than `planner.plan_ahead_days` planned (the same rule and creator filter the planner used).
One showrunner job per creator and period (idempotency key)."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

from .. import queue
from ..policy import Policy
from . import planner


def ensure_batches(conn, policy: Policy, *, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    cfg = policy.get("planner", default={}) or {}
    ahead = timedelta(days=float(cfg.get("plan_ahead_days", 2)))
    out = []
    for c in conn.execute(
            "SELECT c.id, c.handle FROM creators c WHERE c.status = ANY(%s) AND c.profile_version IS NOT NULL "
            "AND jsonb_array_length(coalesce(c.anchors->'reference_images', '[]'::jsonb)) > 0",
            (list(cfg.get("auto_statuses", ["active"])),)).fetchall():
        start = planner.next_start(conn, c["id"], now.date())
        if datetime.combine(start, time(0), tzinfo=timezone.utc) - now > ahead:
            continue
        key = f"showrunner:{c['id']}:{start}"
        if conn.execute("SELECT 1 FROM jobs WHERE idempotency_key = %s", (key,)).fetchone():
            continue   # already asked for this period
        job = queue.enqueue(conn, "brain.showrunner", {"creator_id": str(c["id"]), "start": str(start)},
                            creator_id=c["id"], idempotency_key=key)
        out.append({"creator": c["handle"], "start": str(start), "job": str(job)})
    return out
