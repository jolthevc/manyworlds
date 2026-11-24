"""Paid GPU time goes to real posts, or the machine stops (2026-11-23, the owner's rule: maximise usable output per
dollar; never make extra takes just to fill seconds).

A machine bills its boot, model load and idle tail however little it runs. When a warm image machine is about to
run out of work, the producer starts the next posts that are already planned and scheduled beyond
`production.lead_hours` (up to `production.pull_forward.horizon_hours`), so their scenes are made on time that is
already being paid for. These are posts the showrunner planned and the publisher will publish at their own time:
net new content, not alternates. Nothing is pulled forward:
- while operations are paused (only test work runs then), or for test or breadth posts;
- unless a registered machine that serves the scene kind is up, has already started work (not booting), has at
  least `min_remaining_minutes` before its deadline and has spent less than `max_spend_share` of its cap;
- when that kind's open jobs are above `low_water_jobs` (the machine is still busy with today's work);
- when today's GPU spend is at `budget_share` of `gpu_runs.daily_gpu_usd` or more.

`demand()` is the other half: the enforcer stops an idle machine after `gpu_runs.idle_stop_minutes_no_demand`
instead of the usual idle allowance when no post is producing and none could be pulled forward.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from .. import operations
from ..policy import Policy
from .common import _cfg, _set, log


def _pf(policy: Policy) -> dict:
    return _cfg(policy).get("pull_forward") or {}


def scene_kind(policy: Policy) -> str:
    return str(_cfg(policy).get("image_kind", "image.zimage"))


def _candidates(conn, policy: Policy, now: datetime, limit: int) -> list[dict]:
    lead = timedelta(hours=float(_cfg(policy).get("lead_hours", 48)))
    horizon = timedelta(hours=float(_pf(policy).get("horizon_hours", 0)))
    if horizon <= lead or limit <= 0:
        return []
    formats = list(_cfg(policy).get("formats", ["photo", "carousel"]))
    rows = conn.execute(
        "SELECT p.* FROM posts p JOIN creators c ON c.id = p.creator_id WHERE p.status = 'planned' "
        "AND p.scheduled_for > %s AND p.scheduled_for <= %s AND p.format = ANY(%s) AND NOT p.body ? 'breadth_run' "
        "AND c.status NOT IN ('paused', 'retired') "
        "AND jsonb_array_length(coalesce(c.anchors->'reference_images', '[]'::jsonb)) > 0 "
        "ORDER BY p.scheduled_for LIMIT %s", (now + lead, now + horizon, formats, limit * 3)).fetchall()
    from ..publish import unpublishable
    return [r for r in rows if not unpublishable(conn, policy, r)][:limit]


def _warm_machine(conn, policy: Policy, kind: str, now: datetime) -> dict | None:
    cfg = _pf(policy)
    rows = conn.execute(
        "SELECT g.* FROM gpu_groups g WHERE g.stopped_at IS NULL AND %s = ANY(g.job_kinds) AND g.deadline > %s "
        "AND EXISTS (SELECT 1 FROM attempts a WHERE a.gpu_group = g.name) ORDER BY g.started_at",
        (kind, now + timedelta(minutes=float(cfg.get("min_remaining_minutes", 20))))).fetchall()
    share = float(cfg.get("max_spend_share", 0.8))
    for g in rows:
        if float(g["billed_usd"] or 0) + float(g.get("est_extra_usd") or 0) < share * float(g["max_usd"]):
            return g
    return None


def pull_forward(conn, policy: Policy, *, now: datetime, start) -> dict:
    """Start up to `max_posts_per_tick` planned posts early on a warm, nearly idle machine (see the module doc).
    `start` is production's own post start (scenes.start). Returns what it did and, when nothing, why."""
    cfg = _pf(policy)
    if not cfg.get("enabled"):
        return {"pulled": 0, "why": "disabled"}
    if not operations.is_live(policy):
        return {"pulled": 0, "why": "operations paused"}
    kind = scene_kind(policy)
    machine = _warm_machine(conn, policy, kind, now)
    if machine is None:
        return {"pulled": 0, "why": "no warm machine"}
    open_jobs = conn.execute("SELECT count(*) n FROM jobs WHERE kind = %s AND status IN ('queued', 'running')",
                             (kind,)).fetchone()["n"]
    if open_jobs > int(cfg.get("low_water_jobs", 2)):
        return {"pulled": 0, "why": "machine busy", "open_jobs": open_jobs}
    from ..deploy.enforce import billed_today
    cap = float(policy.get("gpu_runs", "daily_gpu_usd", default=10))
    if billed_today(conn) >= float(cfg.get("budget_share", 0.8)) * cap:
        return {"pulled": 0, "why": "daily GPU budget"}
    pulled = []
    for post in _candidates(conn, policy, now, int(cfg.get("max_posts_per_tick", 2))):
        row = conn.execute("SELECT * FROM posts WHERE id = %s AND status = 'planned'", (post["id"],)).fetchone()
        if row is None:
            continue
        try:
            start(conn, policy, row)
        except Exception:  # noqa: BLE001 - a post that can't start stays planned
            log.exception("pulling post %s forward failed", post["id"])
            continue
        body = conn.execute("SELECT body FROM posts WHERE id = %s", (post["id"],)).fetchone()["body"]
        _set(conn, post["id"], "producing", {**body, "pulled_forward": {"at": now.isoformat(),
                                                                         "machine": machine["name"]}})
        pulled.append(str(post["id"]))
    return {"pulled": len(pulled), "posts": pulled, "machine": machine["name"]}


def demand(conn, policy: Policy, *, now: datetime) -> bool:
    """Whether any post could still need GPU work soon: one is producing, or a planned one is due within the
    pull-forward horizon (or lead_hours, whichever is longer)."""
    if conn.execute("SELECT 1 FROM posts WHERE status = 'producing' LIMIT 1").fetchone():
        return True
    # Work queued anywhere (a creator lab's next step, a benchmark, another stage) may still feed this machine.
    if conn.execute("SELECT 1 FROM jobs WHERE status IN ('queued', 'running') LIMIT 1").fetchone():
        return True
    hours = max(float(_cfg(policy).get("lead_hours", 48)), float(_pf(policy).get("horizon_hours", 0))
                if _pf(policy).get("enabled") and operations.is_live(policy) else 0.0)
    return conn.execute("SELECT 1 FROM posts WHERE status = 'planned' AND coalesce(scheduled_for, now()) <= %s "
                        "LIMIT 1", (now + timedelta(hours=hours),)).fetchone() is not None
