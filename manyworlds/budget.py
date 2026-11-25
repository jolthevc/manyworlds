"""Daily spend caps: starting new work vs finishing work already started (owner, 2026-11-25).

Two caps gate new machines each UTC day: the whole bill (`budgets.daily_usd`) and GPU machines alone
(`gpu_runs.daily_gpu_usd`). Two things move them, and nothing else:

- An owner grant (`budget_grants`, `mw budget grant --usd X --reason ...`): a one-day exception that raises both caps
  for that UTC day only. The policy number never changes; the next day is back to it. Code never writes a grant:
  only the owner's command does.
- Completion headroom (`budgets.completion_usd`): once a cap is reached, a machine may still start (and a draining
  one keep going) for work that finishes a post already in production, i.e. one with at least one finished job
  (2026-11-25: the fully edited Maren video stopped at its voice lines, every photo already paid for, a few dollars
  from done). Nothing new starts in the headroom, and the headroom is a hard ceiling above the cap. The owner hears
  once a day when it is used.
"""

from __future__ import annotations

from .policy import Policy

_TODAY = "date_trunc('day', now() AT TIME ZONE 'utc') AT TIME ZONE 'utc'"


def granted_today(conn) -> float:
    """Owner grants for today (UTC)."""
    row = conn.execute("SELECT coalesce(sum(usd), 0) AS u FROM budget_grants "
                       "WHERE day = (now() AT TIME ZONE 'utc')::date").fetchone()
    return float(row["u"])


def caps(conn, policy: Policy) -> dict:
    """Today's caps for starting new work: {total, gpu, granted, completion_usd}. total is None without
    budgets.daily_usd."""
    g = granted_today(conn)
    total = policy.get("budgets", "daily_usd")
    return {"total": None if total is None else float(total) + g,
            "gpu": float(policy.get("gpu_runs", "daily_gpu_usd", default=10)) + g,
            "granted": g,
            "completion_usd": float(policy.get("budgets", "completion_usd", default=0) or 0)}


def spent(conn) -> dict:
    from . import ledger
    from .deploy import enforce
    return {"total": ledger.bill_today(conn), "gpu": enforce.billed_today(conn)}


def over(conn, policy: Policy, *, headroom: float = 0.0) -> str | None:
    """Why no machine may start now: 'daily_gpu_budget' or 'daily_total_budget' when today's spend is at its cap
    (plus `headroom`), else None."""
    c, s = caps(conn, policy), spent(conn)
    if s["gpu"] >= c["gpu"] + headroom:
        return "daily_gpu_budget"
    if c["total"] is not None and s["total"] >= c["total"] + headroom:
        return "daily_total_budget"
    return None


def completion_kinds(conn, batches: list[str] | None = None) -> set[str]:
    """Job kinds with runnable jobs that finish a post already in production (status producing, at least one of its
    jobs done). With `batches` (LIKE patterns): only jobs in those batches."""
    rows = conn.execute(
        """SELECT DISTINCT j.kind FROM jobs j JOIN posts p ON p.id = j.post_id
           WHERE p.status = 'producing' AND j.attempts < j.max_attempts
             AND (j.status = 'queued' OR (j.status = 'running' AND j.lease_until < now()))
             AND (%(b)s::text[] IS NULL OR coalesce(j.batch, '') LIKE ANY(%(b)s::text[]))
             AND EXISTS (SELECT 1 FROM jobs d WHERE d.post_id = p.id AND d.status = 'done')""",
        {"b": batches}).fetchall()
    return {r["kind"] for r in rows}


def completion_open(conn, policy: Policy, batches: list[str] | None = None) -> bool:
    """Whether finishing started posts may still spend: headroom configured, not used up, and such work waiting or
    running."""
    c = caps(conn, policy)
    if c["completion_usd"] <= 0 or over(conn, policy, headroom=c["completion_usd"]):
        return False
    if completion_kinds(conn, batches):
        return True
    row = conn.execute(
        """SELECT 1 FROM jobs j JOIN posts p ON p.id = j.post_id
           WHERE p.status = 'producing' AND j.status = 'running' AND j.lease_until >= now() LIMIT 1""").fetchone()
    return row is not None


def note_completion(conn, reason: str, kinds: list[str]) -> None:
    """Tell the owner (once a UTC day) that a cap was reached and only started posts are being finished."""
    from datetime import datetime, timezone

    from . import notify
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    notify.owner_event(conn, "budget_completion", subject={"day": day},
                       payload={"cap": reason, "kinds": sorted(kinds),
                                "note": "daily cap reached; finishing posts already in production within "
                                        "budgets.completion_usd, nothing new starts"},
                       dedupe_key=f"budget_completion:{day}")


def grant(conn, usd: float, reason: str, *, by: str = "owner") -> dict:
    """An owner's one-day exception: raises today's caps by usd (UTC day). Only `mw budget grant` calls this."""
    if usd <= 0:
        raise ValueError("a grant is a positive amount")
    if not reason.strip():
        raise ValueError("a grant needs a reason")
    row = conn.execute("INSERT INTO budget_grants (day, usd, reason, granted_by) "
                       "VALUES ((now() AT TIME ZONE 'utc')::date, %s, %s, %s) RETURNING day, usd",
                       (usd, reason.strip(), by)).fetchone()
    return {"day": str(row["day"]), "usd": float(row["usd"]), "granted_today": granted_today(conn)}


def status(conn, policy: Policy) -> dict:
    c, s = caps(conn, policy), spent(conn)
    return {"caps": c, "spent": {k: round(v, 2) for k, v in s.items()}, "over": over(conn, policy),
            "completion_open": completion_open(conn, policy)}
