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
        SELECT g.name, g.provider, g.pool, g.family, g.stop_reason, round(g.billed_usd::numeric, 4)::float AS billed_usd,
               round(extract(epoch FROM g.ended - g.started_at)::numeric / 60, 1)::float AS billed_minutes,
               round(extract(epoch FROM min(a.started_at) - g.started_at)::numeric / 60, 1)::float AS boot_minutes,
               round(coalesce(sum(extract(epoch FROM least(a.finished_at, g.ended) - a.started_at))
                     FILTER (WHERE a.outcome = 'done'), 0)::numeric / 60, 1)::float AS done_minutes,
               count(a.id) FILTER (WHERE a.outcome = 'done') AS jobs_done,
               count(a.id) FILTER (WHERE a.outcome IS DISTINCT FROM 'done') AS jobs_not_done
        FROM g LEFT JOIN attempts a ON a.gpu_group = g.name
        GROUP BY g.name, g.provider, g.pool, g.family, g.stop_reason, g.billed_usd, g.started_at, g.ended
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
