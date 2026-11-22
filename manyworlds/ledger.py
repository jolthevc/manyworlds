"""Cost ledger: every GPU-second, token and data pull is written here.

Cost per *accepted* output is the headline metric, so failed attempts and rejected
candidates are charged too; they just don't count as accepted.
"""

from __future__ import annotations

from typing import Any


def record(
    conn,
    *,
    category: str,
    provider: str,
    quantity: float,
    unit: str,
    usd: float,
    job_id=None,
    attempt_id=None,
    creator_id=None,
    post_id=None,
    policy_version: str | None = None,
    agent: str | None = None,
    at=None,
) -> None:
    """`at` dates an entry that belongs to the past (a provider's charges reconciled later), so daily caps and
    reports count it on the day it was incurred; default now."""
    conn.execute(
        """
        INSERT INTO cost_entries
            (job_id, attempt_id, creator_id, post_id, category, provider,
             quantity, unit, usd, policy_version, agent, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, coalesce(%s, now()))
        """,
        (job_id, attempt_id, creator_id, post_id, category, provider, quantity, unit, usd, policy_version, agent, at),
    )


# 'gpu' entries attribute GPU time to the jobs that used it (for cost per output);
# 'gpu_billed' entries are the provider's actual instance time, idle included.
# The real bill is everything except 'gpu', which would double count.
ATTRIBUTION_ONLY = ("gpu",)


def bill_today(conn) -> float:
    """Everything on today's bill (UTC): machines, LLM calls, data; not the 'gpu' attribution entries."""
    row = conn.execute(
        "SELECT coalesce(sum(usd), 0) AS u FROM cost_entries WHERE category <> ALL(%s) "
        "AND created_at >= date_trunc('day', now() AT TIME ZONE 'utc') AT TIME ZONE 'utc'",
        (list(ATTRIBUTION_ONLY),)).fetchone()
    return float(row["u"])


def bill_total(conn) -> float:
    row = conn.execute("SELECT coalesce(sum(usd), 0) AS u FROM cost_entries WHERE category <> ALL(%s)",
                       (list(ATTRIBUTION_ONLY),)).fetchone()
    return float(row["u"])


def totals(conn, *, batch: str | None = None) -> list[dict[str, Any]]:
    """Spend grouped by category, provider and source: `brain` (an agent's LLM calls, entries with
    an agent) or `production` (making and reviewing outputs, machines, data). Optionally for one batch."""
    return conn.execute(
        """
        SELECT c.category, c.provider, CASE WHEN c.agent IS NOT NULL THEN 'brain' ELSE 'production' END AS source,
               sum(c.quantity) AS quantity, min(c.unit) AS unit,
               round(sum(c.usd)::numeric, 6) AS usd, c.category <> 'gpu' AS on_bill
        FROM cost_entries c
        LEFT JOIN jobs j ON j.id = c.job_id
        WHERE %(batch)s::text IS NULL OR j.batch = %(batch)s
        GROUP BY c.category, c.provider, 3
        ORDER BY usd DESC
        """,
        {"batch": batch},
    ).fetchall()


def cost_per_output(conn, *, batch: str | None = None) -> list[dict[str, Any]]:
    """Per production job kind: GPU time and review LLM calls charged to its jobs, outputs made,
    outputs accepted (reviewed, every review row passed) and the cost per accepted output. Brain
    agents' spend is excluded here and reported by `brain_costs`; keyframes are not outputs."""
    return conn.execute(
        """
        WITH spend AS (
            SELECT j.kind, sum(c.usd) FILTER (WHERE c.category = 'gpu') AS gpu_usd,
                   sum(c.usd) FILTER (WHERE c.category = 'llm') AS review_usd, sum(c.usd) AS usd
            FROM cost_entries c JOIN jobs j ON j.id = c.job_id
            WHERE c.agent IS NULL AND j.kind NOT LIKE 'brain.%%'
              AND (%(batch)s::text IS NULL OR j.batch = %(batch)s)
            GROUP BY j.kind
        ), outputs AS (
            SELECT j.kind, count(a.id) AS n, count(a.id) FILTER (WHERE r.n > 0 AND r.ok) AS accepted
            FROM assets a JOIN jobs j ON j.id = a.job_id
            LEFT JOIN LATERAL (SELECT count(*) AS n, bool_and(passed) AS ok FROM reviews WHERE asset_id = a.id) r ON true
            WHERE a.kind <> 'keyframe' AND (%(batch)s::text IS NULL OR j.batch = %(batch)s)
            GROUP BY j.kind
        )
        SELECT s.kind, o.n AS outputs, coalesce(o.accepted, 0) AS accepted,
               round(coalesce(s.gpu_usd, 0)::numeric, 6) AS gpu_usd, round(coalesce(s.review_usd, 0)::numeric, 6) AS review_usd,
               round(s.usd::numeric, 6) AS usd,
               round((s.usd / NULLIF(o.n, 0))::numeric, 8) AS usd_per_output,
               round((s.usd / NULLIF(o.accepted, 0))::numeric, 8) AS usd_per_accepted
        FROM spend s LEFT JOIN outputs o USING (kind)
        ORDER BY s.kind
        """,
        {"batch": batch},
    ).fetchall()


def brain_costs(conn, *, since=None) -> list[dict[str, Any]]:
    """The Brain's own spend (agents' LLM calls), by agent; part of the bill, not of any output's cost."""
    return conn.execute(
        """SELECT agent, count(*) AS calls, round(sum(usd)::numeric, 6) AS usd FROM cost_entries
           WHERE agent IS NOT NULL AND (%(since)s::timestamptz IS NULL OR created_at >= %(since)s)
           GROUP BY agent ORDER BY usd DESC""", {"since": since}).fetchall()


def accrue_fixed(conn, items: list[dict], *, day=None) -> int:
    """Fixed monthly charges (a storage volume, a subscription) into the ledger, one entry per item per UTC day at
    usd_per_month / 30, so the bill and the daily caps include them. Idempotent per day. Returns entries written."""
    from datetime import datetime, timezone
    day = day or datetime.now(timezone.utc).date()
    n = 0
    for it in items or []:
        provider = str(it["provider"])
        if conn.execute("SELECT 1 FROM cost_entries WHERE category = 'fixed' AND provider = %s "
                        "AND (created_at AT TIME ZONE 'utc')::date = %s", (provider, day)).fetchone():
            continue
        record(conn, category="fixed", provider=provider, quantity=1, unit="day",
               usd=round(float(it["usd_per_month"]) / 30, 6))
        n += 1
    return n
