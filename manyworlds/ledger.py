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
) -> None:
    conn.execute(
        """
        INSERT INTO cost_entries
            (job_id, attempt_id, creator_id, post_id, category, provider,
             quantity, unit, usd, policy_version)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (job_id, attempt_id, creator_id, post_id, category, provider, quantity, unit, usd, policy_version),
    )


# 'gpu' entries attribute GPU time to the jobs that used it (for cost per output);
# 'gpu_billed' entries are the provider's actual instance time, idle included.
# The real bill is everything except 'gpu', which would double count.
ATTRIBUTION_ONLY = ("gpu",)


def bill_total(conn) -> float:
    row = conn.execute("SELECT coalesce(sum(usd), 0) AS u FROM cost_entries WHERE category <> ALL(%s)",
                       (list(ATTRIBUTION_ONLY),)).fetchone()
    return float(row["u"])


def totals(conn, *, batch: str | None = None) -> list[dict[str, Any]]:
    """Spend grouped by category and provider, optionally for one batch of jobs."""
    return conn.execute(
        """
        SELECT c.category, c.provider, sum(c.quantity) AS quantity, min(c.unit) AS unit,
               round(sum(c.usd)::numeric, 6) AS usd, c.category <> 'gpu' AS on_bill
        FROM cost_entries c
        LEFT JOIN jobs j ON j.id = c.job_id
        WHERE %(batch)s::text IS NULL OR j.batch = %(batch)s
        GROUP BY c.category, c.provider
        ORDER BY usd DESC
        """,
        {"batch": batch},
    ).fetchall()


def cost_per_output(conn, *, batch: str | None = None) -> list[dict[str, Any]]:
    """GPU spend per produced asset, by job kind (accepted-only comes with review, day 4)."""
    return conn.execute(
        """
        WITH spend AS (
            SELECT j.kind, sum(c.usd) AS usd
            FROM cost_entries c JOIN jobs j ON j.id = c.job_id
            WHERE %(batch)s::text IS NULL OR j.batch = %(batch)s
            GROUP BY j.kind
        ), outputs AS (
            SELECT j.kind, count(a.id) AS n
            FROM assets a JOIN jobs j ON j.id = a.job_id
            WHERE %(batch)s::text IS NULL OR j.batch = %(batch)s
            GROUP BY j.kind
        )
        SELECT s.kind, o.n AS outputs, round(s.usd::numeric, 6) AS usd,
               round((s.usd / NULLIF(o.n, 0))::numeric, 8) AS usd_per_output
        FROM spend s LEFT JOIN outputs o USING (kind)
        ORDER BY s.kind
        """,
        {"batch": batch},
    ).fetchall()
