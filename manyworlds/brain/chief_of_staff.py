"""The chief of staff: keeps the agency running. Code, not an agent: it owns the reliability of the organisation, and
none of that needs judgement.

Each tick (the producer's Brain steps, while operations are live):
  - due predictions are resolved against outcomes (manyworlds.brain.predictions)
  - pattern cards get their production acceptance and real outcomes (manyworlds.brain.patterns.feedback)
  - the production capability card is measured again (manyworlds.brain.capability_card) and the commercial registry
    seeded where it is empty (manyworlds.brain.commerce)
  - requests nobody decided by their due date expire, and the owner hears how many
  - an agent whose jobs keep failing (>= `failing_jobs` failures in `window_hours`) is reported once a day
  - an agent that spent LLM money in the window and recorded no decision is reported (spend with nothing to show)
  - requests waiting on the owner for longer than `owner_wait_days` are reported again, once a day
  - performance calibration takes one step from reliable observations (manyworlds.performance.refresh_calibration,
    while policy performance.calibration.enabled; nothing changes until judge agreement is measured)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .. import notify, queue
from ..policy import Policy
from . import capability_card, commerce, patterns, predictions, proposals


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
           "expired_requests": proposals.expire_overdue(conn, now),
           # pattern cards learn from what was made with them: production acceptance, real outcomes (evidence.py)
           "pattern_feedback": patterns.feedback(conn, policy)}
    # what production does reliably and cheaply now, per format (casting, talent managers and the CEO read it)
    card = capability_card.refresh(conn, policy, now=now)
    out["capability_card"] = {f: c.get("source") for f, c in card.items()}
    commerce.categories(conn, policy)   # an empty registry is seeded from strategy.commercial.seed
    from .. import performance
    cal = performance.refresh_calibration(conn, policy)   # one step, only from reliable observations
    out["performance_calibration"] = {k: cal.get(k) for k in ("enabled", "judge", "reason") if cal.get(k)} | {
        "changed": len(cal.get("changed") or [])}
    out["research"] = research_tick(conn, policy, now)
    out["verify"] = verify_tick(conn, policy, now)
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


def research_tick(conn, policy: Policy, now: datetime) -> dict:
    """The research loop's code steps (no model): mirror the research and production tables into the analytical
    spine every intelligence.spine.sync_every_hours, then file knowledge gaps as pre-registered questions
    (research_questions.file_gaps; the intelligence agent's `questions` mode answers them). Each step in its own
    savepoint: a failure is reported, never breaks the tick."""
    from ..research import spine
    from . import inquiry, research_questions
    out: dict = {}
    every = float((policy.get("intelligence", "spine", default={}) or {}).get("sync_every_hours", 6))
    last = conn.execute("SELECT max(created_at) t FROM analysis_runs WHERE method = 'spine_sync'").fetchone()["t"]
    if last is None or last <= now - timedelta(hours=every):
        try:
            with conn.transaction():
                got = spine.sync(conn, policy)
                inquiry.record(conn, policy, requester="chief_of_staff", method="spine_sync", spec={}, result=got)
            out["spine"] = {k: got.get(k) for k in ("external", "own", "normalised")}
        except Exception as e:  # noqa: BLE001 - the tick goes on
            out["spine_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        try:   # gaps on the same cadence: each looks at the freshly synced spine
            with conn.transaction():
                out["gaps"] = research_questions.file_gaps(conn, policy)
        except Exception as e:  # noqa: BLE001
            out["gaps_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out or {"skipped": "synced recently"}


def verify_tick(conn, policy: Policy, now: datetime) -> dict:
    """The live end-to-end checks (manyworlds.verify) every verify.every_hours; failing checks reach the owner."""
    from .. import verify
    from . import inquiry
    every = float((policy.get("verify", default={}) or {}).get("every_hours", 24))
    last = conn.execute("SELECT max(created_at) t FROM analysis_runs WHERE method = 'verify'").fetchone()["t"]
    if last is not None and last > now - timedelta(hours=every):
        return {"skipped": "ran recently"}
    try:
        with conn.transaction():
            got = verify.run(conn, policy, now=now)
            inquiry.record(conn, policy, requester="chief_of_staff", method="verify", spec={}, result=got)
            alerted = verify.alert(conn, policy, got, now=now)
        return {"failing": got["failing"], "alerted": alerted}
    except Exception as e:  # noqa: BLE001 - the tick goes on
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}
