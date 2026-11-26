"""Where the money went, development apart from recurring production (owner 2026-11-25).

`mw economics --days N` splits the bill (every ledger category but the `gpu` attribution rows) into:

    production     posts outside test and bench batches: their calls and their jobs' all-in machine share
    development    posts in test or bench batches (breadth, casting auditions, pilots): experiments
    machines_idle  machine bills no job absorbed (a family whose machines all ran nothing)
    orchestration  calls booked to no post by the production agents (capability_card.spread_agents)
    research       the research desk, intelligence agent and monitor (agent research:* / intelligence:*)
    agency         every other untagged call: strategy, talent managers, chief of staff, shadows
    other          anything left (e.g. serverless execution not booked to a job)

Shares of machine bills come from deploy.report.machine_cost_by_job, so the buckets add up to the bill."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .policy import Policy


def report(conn, policy: Policy, *, days: float = 30, now: datetime | None = None) -> dict:
    from .deploy import report as machines
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    dev = [f"{p}%" for p in policy.get("operations", "test_batch_prefixes", default=["bench-", "test-"]) or []]
    spread = list(policy.get("strategy", "capability_card", "spread_agents", default=[]) or [])
    out = {k: 0.0 for k in ("production", "development", "machines_idle", "orchestration", "research", "agency",
                            "other")}
    by_job = machines.machine_cost_by_job(conn, since=since, now=now)
    jobs = conn.execute(
        "SELECT j.id::text AS id, (j.batch LIKE ANY(%s) OR EXISTS (SELECT 1 FROM jobs k WHERE k.post_id = j.post_id "
        "AND j.post_id IS NOT NULL AND k.batch LIKE ANY(%s))) AS dev FROM jobs j WHERE j.id::text = ANY(%s)",
        (dev, dev, [k for k in by_job if not k.startswith("_")])).fetchall()
    for j in jobs:
        out["development" if j["dev"] else "production"] += float(by_job[j["id"]])
    out["machines_idle"] += sum(float(v) for v in (by_job.get("_unallocated") or {}).values())
    gpu_billed = float(conn.execute(
        "SELECT coalesce(sum(usd), 0) u FROM cost_entries WHERE category = 'gpu_billed' AND created_at > %s",
        (since,)).fetchone()["u"])
    rows = conn.execute(
        """SELECT c.usd, c.agent, c.post_id IS NOT NULL OR c.job_id IS NOT NULL AS tagged,
                  EXISTS (SELECT 1 FROM jobs j WHERE (j.post_id = c.post_id OR j.id = c.job_id)
                          AND j.batch LIKE ANY(%s)) AS dev
           FROM cost_entries c WHERE c.created_at > %s AND c.category NOT IN ('gpu', 'gpu_billed')""",
        (dev, since)).fetchall()
    for r in rows:
        usd, agent = float(r["usd"] or 0), (r["agent"] or "").split(":")[0]
        if r["tagged"]:
            out["development" if r["dev"] else "production"] += usd
        elif agent in spread:
            out["orchestration"] += usd
        elif agent in ("research", "intelligence"):
            out["research"] += usd
        elif agent:
            out["agency"] += usd
        else:
            out["other"] += usd
    # Machines started before the window bill inside it; what the split could not place is shown, not hidden.
    placed = sum(float(v) for k, v in by_job.items() if not k.startswith("_")) + out["machines_idle"]
    out["machines_outside_split"] = round(gpu_billed - placed, 4)
    total = sum(v for k, v in out.items() if k != "machines_outside_split") + max(out["machines_outside_split"], 0)
    return {"days": days, "bill_usd": round(total, 2), **{k: round(v, 4) for k, v in out.items()},
            "recurring_share": round((out["production"] + out["orchestration"]) / total, 3) if total else None}
