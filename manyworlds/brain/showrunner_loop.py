"""Keeps every creator planned ahead: asks the showrunner for the next period when a creator has
less than `planner.plan_ahead_days` planned (the same rule and creator filter the planner used).

One showrunner job per creator and period at a time. A period whose job failed, or that ended with
nothing scheduled (every idea invalid or dropped), is asked for again, up to
`brain.showrunner.max_period_attempts` within `retry_after_hours`. A period that uses them raises one
`planning_stalled` event for the owner and is tried again once the cooldown has passed, so an unattended
creator never silently drops out; periods whose start has passed are replaced by the next one anyway.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

from .. import queue
from ..policy import Policy
from . import planner


def _period_state(conn, creator_id, start, retry_after_hours: float) -> tuple[str, int, int]:
    """('active' | 'retry' | 'none', attempts inside the cooldown window, all attempts) for this period."""
    jobs = conn.execute("SELECT status, created_at > now() - make_interval(secs => %s) AS recent FROM jobs "
                        "WHERE kind = 'brain.showrunner' AND creator_id = %s AND payload->>'start' = %s",
                        (retry_after_hours * 3600, creator_id, str(start))).fetchall()
    recent = sum(1 for j in jobs if j["recent"])
    if not jobs:
        return "none", 0, 0
    if any(j["status"] in ("queued", "running") for j in jobs):
        return "active", recent, len(jobs)
    ideas = conn.execute("SELECT count(*) FILTER (WHERE status IN ('proposed', 'scheduling', 'scheduled')) AS live "
                         "FROM ideas WHERE batch = %s", (f"{creator_id}:{start}",)).fetchone()
    return ("active" if ideas["live"] else "retry"), recent, len(jobs)


def _stalled(conn, creator, start, attempts: int) -> None:
    """One event per stalled period, for the owner's digest."""
    from psycopg.types.json import Jsonb
    subject = {"creator_id": str(creator["id"]), "handle": creator["handle"], "start": str(start)}
    if conn.execute("SELECT 1 FROM events WHERE type = 'planning_stalled' AND subject = %s", (Jsonb(subject),)).fetchone():
        return
    conn.execute("INSERT INTO events (type, subject, payload, routed_to) VALUES ('planning_stalled', %s, %s, 'owner')",
                 (Jsonb(subject), Jsonb({"attempts": attempts, "retry": "after the cooldown, automatically"})))


def ensure_batches(conn, policy: Policy, *, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    cfg = policy.get("planner", default={}) or {}
    ahead = timedelta(days=float(cfg.get("plan_ahead_days", 2)))
    max_attempts = int(policy.get("brain", "showrunner", "max_period_attempts", default=3))
    retry_after = float(policy.get("brain", "showrunner", "retry_after_hours", default=12))
    out = []
    for c in conn.execute(
            "SELECT c.id, c.handle FROM creators c WHERE c.status = ANY(%s) AND c.profile_version IS NOT NULL "
            "AND jsonb_array_length(coalesce(c.anchors->'reference_images', '[]'::jsonb)) > 0",
            (list(cfg.get("auto_statuses", ["active"])),)).fetchall():
        start = planner.next_start(conn, c["id"], now.date())
        if datetime.combine(start, time(0), tzinfo=timezone.utc) - now > ahead:
            continue
        state, recent, attempts = _period_state(conn, c["id"], start, retry_after)
        if state == "active":
            continue
        if recent >= max_attempts:
            _stalled(conn, c, start, recent)
            out.append({"creator": c["handle"], "start": str(start), "gave_up": True})
            continue
        key = f"showrunner:{c['id']}:{start}" + (f":{attempts + 1}" if attempts else "")
        job = queue.enqueue(conn, "brain.showrunner", {"creator_id": str(c["id"]), "start": str(start)},
                            creator_id=c["id"], idempotency_key=key)
        out.append({"creator": c["handle"], "start": str(start), "job": str(job), "attempt": attempts + 1})
    return out
