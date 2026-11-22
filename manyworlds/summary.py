"""The owner's daily summary: what the platform spent, made and got stuck on yesterday (UTC), in a few lines.

Sent once a day (policy notify.daily_summary_hour) as an owner event, so it reaches the phone through the normal
alert channel (manyworlds.notify). Built from the ledger, the queue, reviews and the machine report only: it costs
nothing to make.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

from psycopg.types.json import Jsonb

from . import notify
from .policy import Policy


def build(conn, policy: Policy, day: date) -> dict:
    start = datetime.combine(day, time(0), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    win = {"s": start, "e": end}
    spend = {r["category"]: float(r["u"]) for r in conn.execute(
        "SELECT category, sum(usd) AS u FROM cost_entries WHERE created_at >= %(s)s AND created_at < %(e)s "
        "AND category <> 'gpu' GROUP BY 1", win)}
    llm = conn.execute(
        "SELECT provider, sum(usd) AS u FROM cost_entries WHERE category = 'llm' AND created_at >= %(s)s "
        "AND created_at < %(e)s GROUP BY 1 ORDER BY 2 DESC LIMIT 3", win).fetchall()
    kinds = conn.execute(
        """SELECT j.kind, count(*) FILTER (WHERE j.status = 'done') AS done,
                  count(*) FILTER (WHERE j.status IN ('failed', 'dead')) AS failed
           FROM jobs j WHERE j.updated_at >= %(s)s AND j.updated_at < %(e)s AND j.kind NOT LIKE 'brain.%%'
           GROUP BY 1 ORDER BY 2 DESC""", win).fetchall()
    accepted = {r["kind"]: int(r["n"]) for r in conn.execute(
        """SELECT kind, count(*) AS n FROM (
             SELECT j.kind, a.id FROM assets a JOIN jobs j ON j.id = a.job_id JOIN reviews r ON r.asset_id = a.id
             WHERE j.updated_at >= %(s)s AND j.updated_at < %(e)s GROUP BY j.kind, a.id
             HAVING bool_and(r.passed)) x GROUP BY 1""", win)}
    machines = conn.execute(
        "SELECT count(*) AS n, coalesce(sum(cost_usd), 0) AS u FROM gpu_groups WHERE started_at >= %(s)s "
        "AND started_at < %(e)s", win).fetchone()
    brain_failed = conn.execute(
        "SELECT count(*) AS n FROM jobs WHERE kind LIKE 'brain.%%' AND status IN ('failed', 'dead') "
        "AND updated_at >= %(s)s AND updated_at < %(e)s", win).fetchone()["n"]
    published = conn.execute(
        "SELECT count(*) AS n FROM publications WHERE NOT simulated AND published_at >= %(s)s AND published_at < %(e)s",
        win).fetchone()["n"]
    approvals = conn.execute("SELECT count(*) AS n FROM approvals WHERE status = 'pending'").fetchone()["n"]

    total = sum(spend.values())
    lines = [f"Spend ${total:.2f}: " + ", ".join(f"{k} ${v:.2f}" for k, v in sorted(spend.items(), key=lambda x: -x[1])
                                                if v >= 0.005) if total else "Spend $0.00"]
    if llm:
        lines.append("LLM: " + ", ".join(f"{r['provider']} ${float(r['u']):.2f}" for r in llm))
    for r in kinds:
        acc = accepted.get(r["kind"])
        lines.append(f"{r['kind']}: {r['done']} done" + (f", {acc} accepted" if acc is not None else "")
                     + (f", {r['failed']} failed" if r["failed"] else ""))
    lines.append(f"GPU machines: {machines['n']} (${float(machines['u']):.2f})")
    lines.append(f"Published: {published}" + (f"; Brain jobs failed: {brain_failed}" if brain_failed else "")
                 + (f"; approvals waiting: {approvals}" if approvals else ""))
    return {"day": str(day), "text": "\n".join(lines), "spend_usd": round(total, 2), "published": published,
            "pending_approvals": approvals}


def due(conn, policy: Policy, now: datetime) -> date | None:
    """Yesterday's date once the day's summary hour has passed and it hasn't been sent, else None."""
    hour = policy.get("notify", "daily_summary_hour", default=None)
    if hour is None or now.hour < int(hour):
        return None
    day = (now - timedelta(days=1)).date()
    sent = conn.execute("SELECT 1 FROM events WHERE type = 'daily_summary' AND dedupe_key = %s",
                        (f"summary:{day}",)).fetchone()
    return None if sent else day


def send_if_due(conn, policy: Policy, now: datetime | None = None) -> dict | None:
    now = now or datetime.now(timezone.utc)
    day = due(conn, policy, now)
    if day is None:
        return None
    s = build(conn, policy, day)
    notify.owner_event(conn, "daily_summary", subject={"day": s["day"]}, payload=s, dedupe_key=f"summary:{day}")
    conn.execute("INSERT INTO decisions (kind, inputs, output, policy_version, agent) VALUES "
                 "('daily_summary', %s, %s, %s, 'summary')", (Jsonb({"day": s["day"]}), Jsonb(s), policy.version))
    return s
