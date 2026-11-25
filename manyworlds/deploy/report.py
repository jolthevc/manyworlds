"""Where GPU money went: what each machine billed against the work it finished.

`mw machines --hours 72` lists every machine; `mw status` carries the summary. A machine bills
from start to stop; job time is the attempts it finished (done or not), clipped to its life.
Billed money on machines that finished no job at all is the clearest waste: hosts that never
came up, and software that broke on real hardware. Busy share is finished-job time over billed
time across machines that worked: what batching and short idle windows improve.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def machines(conn, *, hours: float = 72, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    return conn.execute(
        """
        WITH g AS (SELECT *, coalesce(stopped_at, %(now)s) AS ended FROM gpu_groups WHERE started_at > %(since)s)
        SELECT g.name, g.provider, g.pool, g.family, g.stop_reason, round(g.cost_usd::numeric, 4)::float AS billed_usd,
               round(extract(epoch FROM g.ended - g.started_at)::numeric / 60, 1)::float AS billed_minutes,
               round(extract(epoch FROM min(a.started_at) - g.started_at)::numeric / 60, 1)::float AS boot_minutes,
               round(coalesce(sum(extract(epoch FROM least(a.finished_at, g.ended) - a.started_at))
                     FILTER (WHERE a.outcome = 'done'), 0)::numeric / 60, 1)::float AS done_minutes,
               count(a.id) FILTER (WHERE a.outcome = 'done') AS jobs_done,
               count(a.id) FILTER (WHERE a.outcome IS DISTINCT FROM 'done') AS jobs_not_done
        FROM g LEFT JOIN attempts a ON a.gpu_group = g.name
        GROUP BY g.name, g.provider, g.pool, g.family, g.stop_reason, g.cost_usd, g.started_at, g.ended
        ORDER BY g.started_at""", {"now": now, "since": now - timedelta(hours=hours)}).fetchall()


def summary(rows: list[dict]) -> dict:
    billed = sum(r["billed_usd"] for r in rows)
    idle = [r for r in rows if not r["jobs_done"]]
    worked = [r for r in rows if r["jobs_done"]]
    worked_minutes = sum(r["billed_minutes"] for r in worked)
    return {"machines": len(rows), "billed_usd": round(billed, 2),
            "no_output_machines": len(idle), "no_output_usd": round(sum(r["billed_usd"] for r in idle), 2),
            "no_output_share": round(sum(r["billed_usd"] for r in idle) / billed, 2) if billed else None,
            "busy_share_of_working_machines": (round(sum(r["done_minutes"] for r in worked) / worked_minutes, 2)
                                               if worked_minutes else None),
            "no_output_by_stop_reason": _by_reason(idle)}


def _by_reason(rows: list[dict]) -> dict:
    out: dict[str, float] = {}
    for r in rows:
        key = (r["stop_reason"] or "running").split(":")[0]
        out[key] = round(out.get(key, 0) + r["billed_usd"], 2)
    return out


def throughput(conn, *, hours: float = 72, now: datetime | None = None) -> list[dict]:
    """What we pay for, per worker family and pool, over machines started in the last `hours` (the North Star:
    accepted outputs per dollar and per GPU-hour).

    billed_usd is each machine's metered cost (gpu_billed, plus reconciled provider extras: gpu_groups.cost_usd), split
    by time into boot (start to its first attempt; all of it on a machine that never began one), busy (attempts of any
    outcome) and idle (the rest: gaps and the tail before it stopped). An output is accepted unless a review failed it
    (as deploy/scoring.py counts). First pass: jobs whose first attempt ran here, finished, and had no output rejected,
    over all jobs whose first attempt ran here. Retries: attempts here that were not their job's first.

    The factory's clock (medians over the group's machines; None where nothing was recorded): provision_minutes
    (start to its worker up: provider start and image pull; recorded by workers built after 2026-11-25), boot_minutes
    (start to its first attempt), cold_start_seconds (its first finished job's time less the median warm job of that
    kind here: weights download and model load, estimated), tail_minutes (last attempt to stop), queue_minutes (a job's
    creation to its first attempt, for jobs first tried here; production work made lead_hours ahead waits on
    purpose)."""
    now = now or datetime.now(timezone.utc)
    machines_ = conn.execute(
        """SELECT name, coalesce(family, '') AS family, pool, started_at, coalesce(stopped_at, %(now)s) AS ended,
                  stopped_at, worker_ready_at, coalesce(cost_usd, 0)::float AS usd
           FROM gpu_groups WHERE started_at > %(since)s""",
        {"now": now, "since": now - timedelta(hours=hours)}).fetchall()
    names = [m["name"] for m in machines_]
    attempts = conn.execute(
        """WITH n AS (SELECT a.*, row_number() OVER (PARTITION BY a.job_id ORDER BY a.started_at, a.id) AS nth
                      FROM attempts a WHERE a.job_id IN (SELECT job_id FROM attempts WHERE gpu_group = ANY(%(g)s)))
           SELECT n.id, n.job_id, n.gpu_group, n.started_at, n.finished_at, n.outcome, n.nth, n.gpu_seconds,
                  (SELECT j.kind FROM jobs j WHERE j.id = n.job_id) AS kind,
                  (SELECT j.created_at FROM jobs j WHERE j.id = n.job_id) AS job_created_at,
                  EXISTS (SELECT 1 FROM assets s JOIN reviews r ON r.asset_id = s.id
                          WHERE s.job_id = n.job_id AND NOT r.passed) AS rejected,
                  (SELECT count(*) FROM assets s WHERE s.job_id = n.job_id AND n.outcome = 'done'
                     AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id = s.id AND NOT r.passed)) AS accepted,
                  (SELECT coalesce(sum((s.meta->>'seconds')::float), 0) FROM assets s
                    WHERE s.job_id = n.job_id AND n.outcome = 'done' AND s.kind = 'video'
                      AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id = s.id AND NOT r.passed))
                    AS accepted_video_s
           FROM n WHERE n.gpu_group = ANY(%(g)s)""", {"g": names}).fetchall()
    by_machine: dict[str, list[dict]] = {}
    for a in attempts:
        by_machine.setdefault(a["gpu_group"], []).append(a)
    pool_of = {m["name"]: (m["family"], m["pool"]) for m in machines_}
    # A warm job: a finished attempt that was not its machine's first, per (family, pool, kind).
    warm: dict[tuple, list[float]] = {}
    for name, mine in by_machine.items():
        first_id = min(mine, key=lambda a: a["started_at"])["id"]
        for a in mine:
            if a["outcome"] == "done" and a["gpu_seconds"] is not None and a["id"] != first_id:
                warm.setdefault((*pool_of[name], a["kind"]), []).append(float(a["gpu_seconds"]))

    groups: dict[tuple, dict] = {}
    for m in machines_:
        life = max((m["ended"] - m["started_at"]).total_seconds(), 0)
        mine = by_machine.get(m["name"], [])
        first = min((a["started_at"] for a in mine), default=None)
        boot = min(max((first - m["started_at"]).total_seconds(), 0), life) if first else life
        busy = sum(max((min(a["finished_at"] or m["ended"], m["ended"]) - a["started_at"]).total_seconds(), 0)
                   for a in mine)
        busy = min(busy, life - boot)
        idle = life - boot - busy
        g = groups.setdefault((m["family"], m["pool"]), {
            "family": m["family"], "pool": m["pool"], "machines": 0, "billed_usd": 0.0, "boot_usd": 0.0,
            "busy_usd": 0.0, "idle_usd": 0.0, "billed_hours": 0.0, "attempts": 0, "retries": 0, "jobs_first": 0,
            "jobs_first_pass": 0, "accepted_outputs": 0, "accepted_video_seconds": 0.0,
            "done": 0, "provision_min": [], "boot_min": [], "cold_s": [], "tail_min": [], "queue_min": []})
        if m["worker_ready_at"] is not None:
            g["provision_min"].append(max((m["worker_ready_at"] - m["started_at"]).total_seconds(), 0) / 60)
        if first is not None:
            g["boot_min"].append(boot / 60)
            opening = min(mine, key=lambda a: a["started_at"])
            ref = warm.get((m["family"], m["pool"], opening["kind"]))
            if opening["outcome"] == "done" and opening["gpu_seconds"] is not None and ref:
                g["cold_s"].append(max(float(opening["gpu_seconds"]) - _median(ref), 0.0))
            if m["stopped_at"] is not None:
                last = max(max(a["started_at"], a["finished_at"] or a["started_at"]) for a in mine)
                g["tail_min"].append(max((m["stopped_at"] - last).total_seconds(), 0) / 60)
        g["queue_min"] += [max((a["started_at"] - a["job_created_at"]).total_seconds(), 0) / 60
                           for a in mine if a["nth"] == 1 and a["job_created_at"] is not None]
        g["machines"] += 1
        g["billed_usd"] += m["usd"]
        g["billed_hours"] += life / 3600
        if life:
            g["boot_usd"] += m["usd"] * boot / life
            g["busy_usd"] += m["usd"] * busy / life
            g["idle_usd"] += m["usd"] * idle / life
        for a in mine:
            g["attempts"] += 1
            g["done"] += a["outcome"] == "done"
            g["accepted_outputs"] += int(a["accepted"])
            g["accepted_video_seconds"] += float(a["accepted_video_s"])
            if a["nth"] == 1:
                g["jobs_first"] += 1
                g["jobs_first_pass"] += a["outcome"] == "done" and not a["rejected"]
            else:
                g["retries"] += 1

    out = []
    for g in sorted(groups.values(), key=lambda r: (r["family"], r["pool"])):
        usd, acc, secs, hrs = g["billed_usd"], g["accepted_outputs"], g["accepted_video_seconds"], g["billed_hours"]
        out.append({
            "family": g["family"], "pool": g["pool"], "machines": g["machines"], "billed_usd": round(usd, 4),
            "boot_usd": round(g["boot_usd"], 4), "busy_usd": round(g["busy_usd"], 4),
            "idle_usd": round(g["idle_usd"], 4), "billed_hours": round(hrs, 3),
            "busy_share": round(g["busy_usd"] / usd, 3) if usd else None,
            "attempts": g["attempts"], "retries": g["retries"], "accepted_outputs": acc,
            "first_pass_rate": round(g["jobs_first_pass"] / g["jobs_first"], 3) if g["jobs_first"] else None,
            "usd_per_accepted_output": round(usd / acc, 5) if acc else None,
            "accepted_video_seconds": round(secs, 1),
            "usd_per_accepted_video_second": round(usd / secs, 5) if secs else None,
            "accepted_per_gpu_hour": round(acc / hrs, 2) if hrs else None,
            "jobs_per_machine": round(g["done"] / g["machines"], 2) if g["machines"] else None,
            "provision_minutes_p50": _round(_median(g["provision_min"]), 1),
            "boot_minutes_p50": _round(_median(g["boot_min"]), 1),
            "cold_start_seconds_p50": _round(_median(g["cold_s"]), 0),
            "tail_minutes_p50": _round(_median(g["tail_min"]), 1),
            "queue_minutes_p50": _round(_median(g["queue_min"]), 1),
            "queue_minutes_p90": _round(_quantile(g["queue_min"], 0.9), 1)})
    return out


def _quantile(values: list[float], q: float) -> float | None:
    """Linear interpolation between closest ranks (as Postgres percentile_cont); None for no values."""
    if not values:
        return None
    v = sorted(values)
    pos = q * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def _median(values: list[float]) -> float | None:
    return _quantile(values, 0.5)


def _round(x: float | None, digits: int) -> float | None:
    return None if x is None else round(x, digits)
