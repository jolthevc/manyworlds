"""Reconcile machine costs with what the provider actually charged.

The enforcers meter machines by the hour, which is all most providers charge. Vast also charges per GB of
download and upload and for disk, set per host; on 2026-11-20 that was $36 of a $40 bill the ledger never saw.
`run` pulls the provider's itemised charges for recent days into `provider_charges`, sets each machine's
`actual_usd` (its provider total), and writes the difference from what was metered to the ledger as a
`gpu_billed` adjustment, once (tracked in `gpu_groups.reconciled_usd`), so the bill and daily GPU cap see it.
Only providers with a `charges` method take part. Machines still running are reconciled on a later run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .. import ledger
from ..policy import Policy


def run(conn, policy: Policy, providers: dict, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    days = int(policy.get("autoscale", "reconcile_days", default=3))
    since, until = now - timedelta(days=days), now
    fetched, groups = 0, 0
    for name, p in providers.items():
        if not hasattr(p, "charges") or not p.configured():
            continue
        rows = p.charges(since, until)
        fetched += len(rows)
        for r in rows:
            conn.execute(
                """INSERT INTO provider_charges (provider, ref, group_name, day, component, usd)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (provider, ref, day, component) DO UPDATE
                   SET usd = excluded.usd, group_name = excluded.group_name, updated_at = now()""",
                (name, str(r["ref"]), r.get("label"), r["day"], r["component"], float(r["usd"])))
        for g in conn.execute(
                """SELECT g.name, g.pool, g.billed_usd, g.reconciled_usd, g.stopped_at, sum(c.usd) AS actual
                   FROM gpu_groups g JOIN provider_charges c ON c.provider = %s
                        AND (c.group_name = g.name OR c.ref = g.provider_ref)
                   WHERE g.provider = %s AND g.stopped_at IS NOT NULL
                   GROUP BY g.name, g.pool, g.billed_usd, g.reconciled_usd, g.stopped_at""", (name, name)).fetchall():
            actual = float(g["actual"])
            delta = round(actual - float(g["billed_usd"]) - float(g["reconciled_usd"]), 6)
            if abs(delta) >= 0.0005:
                ledger.record(conn, category="gpu_billed", provider=g["pool"], quantity=0, unit="reconciliation",
                              usd=delta, policy_version=policy.version, at=g["stopped_at"])   # its own day
                conn.execute("UPDATE gpu_groups SET reconciled_usd = reconciled_usd + %s WHERE name = %s",
                             (delta, g["name"]))
            conn.execute("UPDATE gpu_groups SET actual_usd = %s WHERE name = %s", (actual, g["name"]))
            groups += 1
    return {"charges": fetched, "groups": groups}
