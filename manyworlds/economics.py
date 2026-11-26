"""Where the money went, in the owner's five categories (2026-11-25), reconciled to what the providers billed.

`mw economics --days N`:

  categories        every bill dollar in exactly one bucket (they add up to `bill_usd`):
    production        posts outside test and bench batches: their calls plus their jobs' whole-machine share
    orchestration     production agents' calls booked to no post (capability_card.spread_agents)
    development       posts and jobs in test or bench batches (breadth, casting auditions, pilots, benchmarks)
    research          the research desk, intelligence agent, monitor and vision benchmark (agent research:* or
                      intelligence:*), and data purchases
    agency            other untagged calls: strategy, talent managers, chief of staff, shadow tests
    fixed             accrued fixed charges (policy fixed_costs, ledger category `fixed`)
    machines_unplaced machine bills no job absorbed (a family whose machines ran nothing) or that fall outside the
                      split window
  per_format        production posts only, per accepted post:
    marginal          what one more post costs on a warm machine: its attempts' seconds at the machine's hourly rate
                      plus its own calls (no boot, idle or failed machines)
    fully_loaded      its whole-machine share plus its calls plus its share of orchestration and of fixed charges
  reconciliation    the machines' bills (gpu_groups.cost_usd) against the ledger's `gpu_billed`, and what the split
                    placed: they should agree; `gpu` attribution rows are never counted (they would double count)
  measured          what is a provider's own figure vs our estimate (metered hours at list price, token prices)
  missing           infrastructure the bill doesn't carry yet (policy `infrastructure` items without a monthly figure)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .policy import Policy

RESEARCH_AGENTS = ("research", "intelligence")


def report(conn, policy: Policy, *, days: float = 30, now: datetime | None = None) -> dict:
    from .deploy import report as machines
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    dev = [f"{p}%" for p in policy.get("operations", "test_batch_prefixes", default=["bench-", "test-"]) or []]
    spread = list(policy.get("strategy", "capability_card", "spread_agents", default=[]) or [])
    ok = list((policy.get("intelligence", "feedback", default={}) or {}).get("accepted_statuses")
              or ["ready", "approved", "scheduled", "published"])
    cat = {k: 0.0 for k in ("production", "orchestration", "development", "research", "agency", "fixed",
                            "machines_unplaced")}

    # --- machines: whole bills split over jobs (boot, idle, failures and dead machines included)
    by_job = machines.machine_cost_by_job(conn, since=since, now=now)
    job_ids = [k for k in by_job if not k.startswith("_")]
    jobs = {r["id"]: r for r in conn.execute(
        """SELECT j.id::text AS id, j.post_id::text AS post_id,
                  (j.batch LIKE ANY(%(dev)s) OR EXISTS (SELECT 1 FROM jobs k WHERE j.post_id IS NOT NULL
                     AND k.post_id = j.post_id AND k.batch LIKE ANY(%(dev)s))) AS dev,
                  (SELECT coalesce(sum(extract(epoch FROM least(coalesce(a.finished_at, g.stopped_at, %(now)s),
                                                                coalesce(g.stopped_at, %(now)s)) - a.started_at)
                                   / 3600 * g.usd_per_hour), 0)
                     FROM attempts a JOIN gpu_groups g ON g.name = a.gpu_group
                    WHERE a.job_id = j.id AND a.started_at IS NOT NULL) AS marginal_gpu
           FROM jobs j WHERE j.id::text = ANY(%(ids)s)""",
        {"dev": dev, "ids": job_ids, "now": now}).fetchall()}
    for jid in job_ids:
        cat["development" if jobs.get(jid, {}).get("dev") else "production"] += float(by_job[jid])
    cat["machines_unplaced"] += sum(float(v) for v in (by_job.get("_unallocated") or {}).values())

    machine_bills = conn.execute(
        """SELECT coalesce(sum(cost_usd), 0)::float AS cost, coalesce(sum(billed_usd), 0)::float AS metered,
                  coalesce(sum(est_extra_usd), 0)::float AS est_extra,
                  coalesce(sum(actual_usd) FILTER (WHERE actual_usd IS NOT NULL), 0)::float AS provider_actual,
                  count(*) FILTER (WHERE actual_usd IS NOT NULL) AS reconciled, count(*) AS machines
           FROM gpu_groups WHERE started_at > %s""", (since,)).fetchone()
    ledger_gpu = float(conn.execute(
        "SELECT coalesce(sum(usd), 0) u FROM cost_entries WHERE category = 'gpu_billed' AND created_at > %s",
        (since,)).fetchone()["u"])
    placed = sum(float(by_job[j]) for j in job_ids) + cat["machines_unplaced"]
    # The ledger's gpu_billed is the bill; machines started before the window bill inside it, so the split covers
    # what it can and the remainder is shown, never dropped.
    outside = round(ledger_gpu - placed, 4)
    if outside > 0:
        cat["machines_unplaced"] += outside

    # --- every other bill entry (never the `gpu` attribution rows, never gpu_billed: machines are counted above)
    post_costs: dict[str, float] = {}
    rows = conn.execute(
        """SELECT c.usd, c.category, c.agent, c.post_id::text AS post_id,
                  coalesce(c.post_id::text, (SELECT j.post_id::text FROM jobs j WHERE j.id = c.job_id)) AS any_post,
                  c.post_id IS NOT NULL OR c.job_id IS NOT NULL AS tagged,
                  EXISTS (SELECT 1 FROM jobs j WHERE (j.post_id = c.post_id OR j.id = c.job_id)
                          AND j.batch LIKE ANY(%s)) AS dev
           FROM cost_entries c WHERE c.created_at > %s AND c.category NOT IN ('gpu', 'gpu_billed')""",
        (dev, since)).fetchall()
    llm_usd = 0.0
    for r in rows:
        usd, agent = float(r["usd"] or 0), (r["agent"] or "").split(":")[0]
        llm_usd += usd if r["category"] == "llm" else 0.0
        if r["category"] == "fixed":
            cat["fixed"] += usd
        elif r["tagged"]:
            cat["development" if r["dev"] else "production"] += usd
            if not r["dev"] and r["any_post"]:
                post_costs[r["any_post"]] = post_costs.get(r["any_post"], 0.0) + usd
        elif agent in spread:
            cat["orchestration"] += usd
        elif agent in RESEARCH_AGENTS or r["category"] == "data":
            cat["research"] += usd
        else:
            cat["agency"] += usd

    # --- per format, production posts only
    per_format: dict[str, dict] = {}
    posts = conn.execute(
        """SELECT p.id::text AS id, p.format, p.status IN (SELECT unnest(%s::text[])) AS accepted
           FROM posts p WHERE p.updated_at > %s AND NOT EXISTS
             (SELECT 1 FROM jobs j WHERE j.post_id = p.id AND j.batch LIKE ANY(%s))
             AND EXISTS (SELECT 1 FROM jobs j WHERE j.post_id = p.id)""", (ok, since, dev)).fetchall()
    jobs_of: dict[str, list[str]] = {}
    for jid, j in jobs.items():
        if j["post_id"]:
            jobs_of.setdefault(j["post_id"], []).append(jid)
    n_posts = len(posts) or 1
    overhead_each = (cat["orchestration"] + cat["fixed"]) / n_posts
    for p in posts:
        f = per_format.setdefault(p["format"], {"posts": 0, "accepted": 0, "marginal": 0.0, "fully_loaded": 0.0})
        mine = jobs_of.get(p["id"], [])
        calls = post_costs.get(p["id"], 0.0)
        f["posts"] += 1
        f["accepted"] += bool(p["accepted"])
        f["marginal"] += sum(float(jobs[j]["marginal_gpu"] or 0) for j in mine) + calls
        f["fully_loaded"] += sum(float(by_job.get(j, 0)) for j in mine) + calls + overhead_each
    for f in per_format.values():
        acc = f["accepted"]
        f["marginal_per_accepted"] = round(f["marginal"] / acc, 4) if acc else None
        f["fully_loaded_per_accepted"] = round(f["fully_loaded"] / acc, 4) if acc else None
        f["marginal"], f["fully_loaded"] = round(f["marginal"], 4), round(f["fully_loaded"], 4)

    total = sum(cat.values())
    infra = list(policy.get("infrastructure", default=[]) or [])
    return {
        "days": days, "bill_usd": round(total, 2),
        "categories": {k: round(v, 4) for k, v in cat.items()},
        "recurring_production_usd": round(cat["production"] + cat["orchestration"], 4),
        "per_format": per_format,
        "reconciliation": {
            "ledger_gpu_billed_usd": round(ledger_gpu, 4), "machines_cost_usd": round(machine_bills["cost"], 4),
            "placed_on_jobs_usd": round(sum(float(by_job[j]) for j in job_ids), 4),
            "outside_split_usd": outside,
            "note": "machines_cost_usd counts machines started in the window (provider actuals where reconciled); "
                    "the ledger counts billing inside the window; `gpu` attribution rows are excluded everywhere"},
        "measured": {
            "gpu_provider_actual_usd": round(machine_bills["provider_actual"], 4),
            "gpu_machines_reconciled": f"{machine_bills['reconciled']}/{machine_bills['machines']}",
            "gpu_estimated_usd": round(max(machine_bills["cost"] - machine_bills["provider_actual"], 0), 4),
            "gpu_estimate_basis": "metered hours at policy list price plus estimated download extras until the "
                                  "provider's itemised charges are reconciled (deploy/reconcile.py)",
            "llm_usd": round(llm_usd, 4),
            "llm_basis": "estimated from token counts at policy prices; compare with the Anthropic console invoice",
            "fixed_basis": "policy fixed_costs (contract price), accrued daily"},
        "missing": [{"item": i.get("item"), "needs": i.get("needs") or "monthly invoice figure"}
                    for i in infra if i.get("usd_per_month") is None],
    }
