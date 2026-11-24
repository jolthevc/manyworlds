"""The chief of staff: keeps the agency running. Code, not an agent: it owns the reliability of the organisation, and
none of that needs judgement.

Each tick (the producer's Brain steps, while operations are live):
  - due predictions are resolved against outcomes (manyworlds.brain.predictions)
  - requests nobody decided by their due date expire, and the owner hears how many
  - an agent whose jobs keep failing (>= `failing_jobs` failures in `window_hours`) is reported once a day
  - an agent that spent LLM money in the window and recorded no decision is reported (spend with nothing to show)
  - requests waiting on the owner for longer than `owner_wait_days` are reported again, once a day
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .. import notify, queue
from ..policy import Policy
from . import predictions, proposals


def _cfg(policy: Policy) -> dict:
    return policy.get("brain", "chief_of_staff", default={}) or {}


def tick(conn, policy: Policy, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    cfg = _cfg(policy)
    if not cfg.get("enabled", True):
        return {}
    day = now.date().isoformat()
    window = now - timedelta(hours=float(cfg.get("window_hours", 24)))
    out = {"predictions": predictions.resolve_due(conn, policy, now=now),
           "expired_requests": proposals.expire_overdue(conn, now)}
    if out["expired_requests"]:
        notify.owner_event(conn, "requests_expired", payload={"n": out["expired_requests"]},
                           dedupe_key=f"requests_expired:{now:%Y-%m-%dT%H}")
    failing = conn.execute(
        "SELECT kind, count(*) n FROM jobs WHERE kind LIKE 'brain.%%' AND status IN ('failed', 'dead') "
        "AND coalesce(error, '') NOT LIKE %s "   # a cancelled job (queue.cancel) did not fail
        "AND updated_at > %s GROUP BY 1 HAVING count(*) >= %s",
        (queue.CANCELLED + "%", window, int(cfg.get("failing_jobs", 3)))).fetchall()
    for r in failing:
        notify.owner_event(conn, "agent_failing", subject={"agent": r["kind"]}, payload={"failed": r["n"]},
                           dedupe_key=f"agent_failing:{r['kind']}:{day}")
    idle_spend = conn.execute(
        "SELECT c.agent, round(sum(c.usd)::numeric, 2)::float usd FROM cost_entries c WHERE c.category = 'llm' "
        "AND split_part(regexp_replace(c.agent, '^sandbox:', ''), ':', 1) = ANY(%s) "
        "AND c.created_at > %s AND NOT EXISTS (SELECT 1 FROM decisions d "
        "WHERE d.agent = split_part(regexp_replace(c.agent, '^sandbox:', ''), ':', 1) AND d.created_at > %s) "
        "GROUP BY 1 HAVING sum(c.usd) >= %s",
        (list(cfg.get("decision_agents") or []), window, window, float(cfg.get("idle_spend_usd", 0.5)))).fetchall()
    for r in idle_spend:
        notify.owner_event(conn, "agent_spend_without_output", subject={"agent": r["agent"]}, payload={"usd": r["usd"]},
                           dedupe_key=f"agent_spend_without_output:{r['agent']}:{day}")
    waiting = conn.execute("SELECT count(*) n FROM proposals WHERE to_role = 'owner' AND status = 'open' "
                           "AND created_at < %s", (now - timedelta(days=float(cfg.get("owner_wait_days", 3))),)
                           ).fetchone()["n"]
    if waiting:
        notify.owner_event(conn, "requests_waiting_for_owner", payload={"n": waiting},
                           dedupe_key=f"requests_waiting_for_owner:{day}")
    out.update(failing_agents=len(failing), idle_spend_agents=len(idle_spend), waiting_for_owner=waiting)
    return out
