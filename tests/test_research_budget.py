"""One monthly ceiling over all research spend (owner 2026-11-26: up to $100/month; manyworlds.research.budget)."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

from manyworlds import ledger
from manyworlds.research import budget


def _with(policy, **c):
    return dataclasses.replace(policy, body={**policy.body, "research_budget": {**policy.body["research_budget"], **c}})


def _spend(conn, agent, usd, at):
    ledger.record(conn, category="llm", provider="t", quantity=1, unit="call", usd=usd, agent=agent, at=at)


def test_the_shipped_ceiling_is_the_owners_hundred_dollars(policy):
    c = budget.cfg(policy)
    assert c["monthly_usd"] == 100.0 and {"research", "intelligence", "trends"} <= set(c["agents"])


def test_every_research_tag_counts_and_others_do_not(conn, policy):
    now = datetime(2026, 12, 10, 12, tzinfo=timezone.utc)
    _spend(conn, "research:desk", 1.0, now)
    _spend(conn, "intelligence:scout", 2.0, now)
    _spend(conn, "trends", 0.5, now)
    _spend(conn, "showrunner", 9.0, now)                                   # production planning: not research
    _spend(conn, "researcher_of_other_things", 9.0, now)                   # a prefix without the colon: not research
    _spend(conn, "research:desk", 50.0, datetime(2026, 11, 30, tzinfo=timezone.utc))   # last month
    s = budget.status(conn, policy, now)
    assert s["month_spent"] == 3.5 and s["month_left"] == 96.5


def test_a_day_is_paced_and_the_month_is_a_hard_ceiling(conn, policy):
    now = datetime(2026, 12, 2, 12, tzinfo=timezone.utc)                   # 30 days left, today included
    pol = _with(policy, monthly_usd=30.0, pace_multiple=2.0)
    assert budget.left(conn, pol, now) == 2.0                              # 30 / 30 x 2
    _spend(conn, "research:desk", 1.5, now)
    assert abs(budget.left(conn, pol, now) - 0.5) < 1e-9
    _spend(conn, "intelligence", 40.0, datetime(2026, 12, 1, tzinfo=timezone.utc))
    assert budget.left(conn, pol, now) == 0.0                              # the month is spent
    assert budget.left(conn, _with(policy, monthly_usd=None), now) == float("inf")


def test_the_desk_stops_at_the_monthly_ceiling(conn, policy):
    from manyworlds.research import desk
    pol = _with(policy, monthly_usd=0.0)
    out = desk.ask(conn, pol, None, "what is trending in quiet luxury?", depth="quick")   # no model is called
    assert "monthly research budget" in str(out.get("skipped"))
