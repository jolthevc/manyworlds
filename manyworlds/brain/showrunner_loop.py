"""Keeps every creator planned ahead: asks the showrunner for the next period when a creator has
less than `planner.plan_ahead_days` planned (the same rule and creator filter the planner used).

One showrunner job per creator and period at a time. A period whose job failed, or that ended with
nothing scheduled (every idea invalid or dropped), is asked for again, up to
`brain.showrunner.max_period_attempts`; after that it is reported as given up (the creator would
otherwise stop being planned silently, since the next period starts after the last post).
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

from .. import queue
from ..policy import Policy
from . import planner


def _period_state(conn, creator_id, start) -> tuple[str, int]:
    """('active' | 'retry' | 'none', attempts so far) for this creator's period."""
    jobs = conn.execute("SELECT status FROM jobs WHERE kind = 'brain.showrunner' AND creator_id = %s "
                        "AND payload->>'start' = %s", (creator_id, str(start))).fetchall()
    if not jobs:
        return "none", 0
    if any(j["status"] in ("queued", "running") for j in jobs):
        return "active", len(jobs)
    ideas = conn.execute("SELECT count(*) FILTER (WHERE status IN ('proposed', 'scheduling', 'scheduled')) AS live "
                         "FROM ideas WHERE batch = %s", (f"{creator_id}:{start}",)).fetchone()
    return ("active" if ideas["live"] else "retry"), len(jobs)


def ensure_batches(conn, policy: Policy, *, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    cfg = policy.get("planner", default={}) or {}
    ahead = timedelta(days=float(cfg.get("plan_ahead_days", 2)))
    max_attempts = int(policy.get("brain", "showrunner", "max_period_attempts", default=3))
    out = []
    for c in conn.execute(
            "SELECT c.id, c.handle FROM creators c WHERE c.status = ANY(%s) AND c.profile_version IS NOT NULL "
            "AND jsonb_array_length(coalesce(c.anchors->'reference_images', '[]'::jsonb)) > 0",
            (list(cfg.get("auto_statuses", ["active"])),)).fetchall():
        start = planner.next_start(conn, c["id"], now.date())
        if datetime.combine(start, time(0), tzinfo=timezone.utc) - now > ahead:
            continue
        state, attempts = _period_state(conn, c["id"], start)
        if state == "active":
            continue
        if attempts >= max_attempts:
            out.append({"creator": c["handle"], "start": str(start), "gave_up": True})
            continue
        key = f"showrunner:{c['id']}:{start}" + (f":{attempts + 1}" if attempts else "")
        job = queue.enqueue(conn, "brain.showrunner", {"creator_id": str(c["id"]), "start": str(start)},
                            creator_id=c["id"], idempotency_key=key)
        out.append({"creator": c["handle"], "start": str(start), "job": str(job), "attempt": attempts + 1})
    return out
