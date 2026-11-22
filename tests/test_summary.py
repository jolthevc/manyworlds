"""The owner's daily summary: yesterday's spend, outputs and problems, once a day, readable on a phone."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from manyworlds import ledger, notify, summary
from manyworlds.deploy import autoscale


def test_yesterdays_summary_goes_out_once_after_its_hour(conn, policy):
    hour = policy.get("notify", "daily_summary_hour")
    now = datetime.now(timezone.utc).replace(hour=hour, minute=5, second=0, microsecond=0)
    yesterday = now - timedelta(days=1)
    ledger.record(conn, category="llm", provider="claude-haiku-5-5", quantity=1, unit="token", usd=1.25, at=yesterday)
    ledger.record(conn, category="hosted_api", provider="together:wan", quantity=1, unit="clip", usd=0.10, at=yesterday)
    assert summary.due(conn, policy, now - timedelta(hours=1)) is None             # before the hour: not yet
    assert autoscale.daily_summary(conn, policy, now) == [{"action": "daily_summary", "day": str(yesterday.date())}]
    assert autoscale.daily_summary(conn, policy, now + timedelta(hours=2)) == []      # once a day
    sent = []
    notify.deliver(conn, policy, send=lambda title, body, priority: sent.append((title, body, priority)), now=now)
    title, body, priority = next(s for s in sent if "daily summary" in s[0])
    assert body.startswith("Spend $1.35") and "claude-haiku-5-5 $1.25" in body and priority == "low"
