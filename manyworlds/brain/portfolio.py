"""The portfolio in numbers, for the strategist (Brain M8, spec 4.6). Code, not judgement.

Conviction: a normal belief about a creator's weekly reach growth (slope of log weekly median reach, REAL
publications only, through manyworlds.brain.evidence in mode `real`). It starts at policy's prior and is updated
by the observed slope and its standard error (precision-weighted); with fewer than `min_weeks` of real data it
is the prior. Sandbox (simulated) outcomes never enter: simulated results are not evidence of demand.

Continue / stop, an expected-value comparison: continuing is worth the conviction mean plus a value-of-
information bonus (`voi_weight` x its sd: an uncertain creator can still surprise); the alternative is a new
launch at `alternative_mean`. Retirement is a case only when continuing is worth less than the alternative by
`retire_margin`, and the strategist's move is still checked against evidence windows and approval.

Budget: the month's master budget less the reserve, against the month's bill (ledger, every category except
attribution-only entries).
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta, timezone
from statistics import NormalDist

from .. import ledger, queue
from ..policy import Policy
from . import evidence


def _cfg(policy: Policy) -> dict:
    return policy.get("brain", "strategist", default={}) or {}


def _weekly_reach(conn, creator_id) -> list[tuple[int, float]]:
    posts = conn.execute(
        "SELECT p.id, p.published_at FROM posts p JOIN publications u ON u.post_id = p.id AND NOT u.simulated "
        "WHERE p.creator_id = %s ORDER BY p.published_at", (creator_id,)).fetchall()
    if not posts:
        return []
    first, weeks = posts[0]["published_at"], {}
    for p in posts:
        m = evidence.metrics(conn, p["id"], mode="real")
        if not m:
            continue
        reach = float(max(m, key=lambda r: r.get("post_age_s") or 0)["metrics"].get("reach") or 0)
        if reach > 0:
            weeks.setdefault((p["published_at"] - first).days // 7, []).append(reach)
    return [(w, statistics.median(v)) for w, v in sorted(weeks.items())]


def conviction(conn, policy: Policy, creator_id) -> dict:
    cfg = (_cfg(policy).get("conviction") or {})
    m0, s0 = float(cfg.get("prior_mean", 0.0)), float(cfg.get("prior_sd", 0.15))
    z = NormalDist().inv_cdf(0.95)
    pts = _weekly_reach(conn, creator_id)
    mean, sd, source = m0, s0, "prior"
    if len(pts) >= int(cfg.get("min_weeks", 3)):
        xs, ys = [w for w, _ in pts], [math.log(r) for _, r in pts]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx > 0:
            slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
            resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
            se = math.sqrt(max(sum(r * r for r in resid) / max(len(xs) - 2, 1), 1e-6) / sxx)
            w0, w1 = 1 / s0 ** 2, 1 / se ** 2
            mean, sd, source = (w0 * m0 + w1 * slope) / (w0 + w1), math.sqrt(1 / (w0 + w1)), "real"
    return {"mean": round(mean, 4), "sd": round(sd, 4), "low": round(mean - z * sd, 4), "high": round(mean + z * sd, 4),
            "weeks": len(pts), "source": source}


def value(policy: Policy, conv: dict) -> dict:
    cfg = (_cfg(policy).get("conviction") or {})
    ev = conv["mean"] + float(cfg.get("voi_weight", 0.5)) * conv["sd"]
    alt = float(cfg.get("alternative_mean", 0.0))
    return {"ev_continue": round(ev, 4), "ev_alternative": alt,
            "retire_case": ev < alt - float(cfg.get("retire_margin", 0.02))}


def budget(conn, policy: Policy, now: datetime) -> dict:
    b = policy.get("budget", default={}) or {}
    master, reserve = float(b.get("master_usd_per_month", 750)), float(b.get("reserve_share", 0.15))
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    spent = float(conn.execute("SELECT coalesce(sum(usd), 0) u FROM cost_entries WHERE category <> ALL(%s) "
                               "AND created_at >= %s", (list(ledger.ATTRIBUTION_ONLY), start)).fetchone()["u"])
    usable = master * (1 - reserve)
    return {"month": start.strftime("%Y-%m"), "master_usd": master, "usable_usd": round(usable, 2),
            "spent_usd": round(spent, 2), "remaining_usd": round(usable - spent, 2)}


def snapshot(conn, policy: Policy, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    rows = conn.execute(
        """SELECT c.id, c.handle, c.status, c.publish_mode, s.stage, s.entered_at, coalesce(s.adjustments, 0) adj,
                  (SELECT count(*) FROM publications u JOIN posts p ON p.id = u.post_id
                    WHERE p.creator_id = c.id AND NOT u.simulated) AS real_posts,
                  (SELECT min(u.published_at) FROM publications u JOIN posts p ON p.id = u.post_id
                    WHERE p.creator_id = c.id AND NOT u.simulated) AS first_real,
                  (SELECT coalesce(sum(usd), 0) FROM cost_entries e WHERE e.creator_id = c.id) AS cost,
                  (SELECT status FROM accounts a WHERE a.creator_id = c.id AND a.platform = 'instagram') AS account
           FROM creators c LEFT JOIN creator_states s ON s.creator_id = c.id
           WHERE c.status <> 'retired' ORDER BY c.created_at""").fetchall()
    creators = []
    for r in rows:
        conv = conviction(conn, policy, r["id"])
        creators.append({
            "creator_id": str(r["id"]), "handle": r["handle"], "status": r["status"], "publish_mode": r["publish_mode"],
            "stage": r["stage"] or "exploring", "adjustments": r["adj"], "real_posts": r["real_posts"],
            "days_live": (now - r["first_real"]).days if r["first_real"] else 0, "cost_usd": round(float(r["cost"]), 2),
            "account": r["account"], "conviction": conv, "value": value(policy, conv)})
    week = now - timedelta(days=7)
    launches = {
        "running": conn.execute("SELECT count(*) n FROM launches WHERE status = 'running'").fetchone()["n"],
        "pending_approvals": conn.execute("SELECT count(*) n FROM approvals WHERE kind = 'launch' AND status = 'pending'"
                                          ).fetchone()["n"],
        "last_7_days": conn.execute("SELECT count(*) n FROM approvals WHERE kind = 'launch' AND status IN "
                                    "('pending', 'approved') AND created_at > %s", (week,)).fetchone()["n"]}
    return {"creators": creators, "budget": budget(conn, policy, now), "launches": launches}


def ensure_daily(conn, policy: Policy, *, now: datetime | None = None):
    """One strategist run per day (the portfolio cycle); events and owner requests come on top."""
    now = now or datetime.now(timezone.utc)
    day = now.date().isoformat()
    return queue.enqueue(conn, "brain.strategist", {"day": day}, idempotency_key=f"strategist:{day}")
