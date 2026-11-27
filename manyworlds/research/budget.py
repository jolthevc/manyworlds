"""One monthly ceiling for all research and intelligence spend (owner 2026-11-26: up to $100 a month for discovery,
data acquisition, annotation, video understanding, web research, analysis and synthesis; a ceiling, not a target;
it replaced the $1/day research allowance).

Every research spender (the desk, the monitor, the trend watcher, the intelligence agent, research questions, the
vision bench) keeps its own cap and also asks `left()`: what this calendar month (UTC) still allows, paced so one day
takes at most `pace_multiple` times the even share of what is left (a burst never empties the month on day 2).
Spend is read from the ledger by agent tag (policy `research_budget.agents`: a tag counts when it equals one of them
or starts with it and a colon), so a new research agent is covered by naming its tag there."""

from __future__ import annotations

import calendar
from datetime import datetime, timezone

from ..policy import Policy


def cfg(policy: Policy) -> dict:
    return policy.get("research_budget", default={}) or {}


def _month_start(now: datetime) -> datetime:
    return now.astimezone(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _day_start(now: datetime) -> datetime:
    return now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def spent(conn, policy: Policy, since: datetime) -> float:
    tags = [str(t) for t in cfg(policy).get("agents") or []]
    if not tags:
        return 0.0
    return float(conn.execute(
        "SELECT coalesce(sum(usd), 0) AS u FROM cost_entries WHERE created_at >= %s "
        "AND (agent = ANY(%s) OR agent LIKE ANY(%s))",
        (since, tags, [t + ":%" for t in tags])).fetchone()["u"])


def status(conn, policy: Policy, now: datetime | None = None) -> dict:
    """{monthly_usd, month_spent, month_left, today_spent, today_allowance, left}: `left` is what a spender may still
    use now (the smaller of the month's remainder and today's paced allowance); inf when no ceiling is set."""
    now = now or datetime.now(timezone.utc)
    c = cfg(policy)
    if c.get("monthly_usd") is None:
        return {"monthly_usd": None, "left": float("inf")}
    monthly = float(c["monthly_usd"])
    month = spent(conn, policy, _month_start(now))
    today = spent(conn, policy, _day_start(now))
    days = calendar.monthrange(now.year, now.month)[1]
    days_left = days - now.astimezone(timezone.utc).day + 1           # today included
    month_left = max(0.0, monthly - month)
    allowance = (month_left + today) / days_left * float(c.get("pace_multiple", 1.0))
    left = max(0.0, min(month_left, allowance - today))
    return {"monthly_usd": monthly, "month_spent": round(month, 4), "month_left": round(month_left, 4),
            "today_spent": round(today, 4), "today_allowance": round(allowance, 4), "left": round(left, 4)}


def left(conn, policy: Policy, now: datetime | None = None) -> float:
    return float(status(conn, policy, now)["left"])
