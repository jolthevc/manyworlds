"""Requests between roles: how the agency's agents ask each other for things and get a decision.

A talent manager asks the CEO for a better production tier, a market lead proposes a cohort, anyone escalates to
the owner. Every request has one owner who decides it once (policy `brain.rights.routes`), and code, not the
model, checks what each role may decide and what the money allows (`brain.rights`, `brain.tiers`, the month's
budget). That is the whole debate mechanism: an argument with evidence, a decision with a reason, no loops.

    file()    record a request (refused: unknown kind, too many open for the creator, the same request rejected
              within `cooldown_days` and filed again with no new evidence, over the requester's allowance)
    decide()  the owner role accepts or rejects; an accepted request is applied when its kind has an effect
              (tier_change sets creator_states.tier; research commissions the analyst with the question;
              escalate and anything over the decider's rights go to the owner)
    open_for() what a role has to decide next run

Requests about a creator in the sandbox carry `evidence.mode = sandbox`: simulated outcomes argue nothing about
real demand, and the decider is told so.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from .. import notify, queue
from ..policy import Policy
from . import evidence

KINDS = ("tier_change", "budget", "research", "experiment", "cohort", "production_upgrade", "escalate")


def _rights(policy: Policy) -> dict:
    return policy.get("brain", "rights", default={}) or {}


def route(policy: Policy, kind: str) -> str:
    return str((_rights(policy).get("routes") or {}).get(kind, "owner"))


def _key(kind: str, creator_id, market_id, subject: dict) -> str:
    return f"{kind}:{creator_id or '-'}:{market_id or '-'}:{json.dumps(subject, sort_keys=True, default=str)}"


def file(conn, policy: Policy, *, kind: str, from_role: str, subject: dict, argument: str, evidence_: dict | None = None,
         creator_id=None, market_id=None, job_id=None, now: datetime | None = None) -> dict:
    """{id, to_role} for a recorded request, or {refused: why}."""
    now = now or datetime.now(timezone.utc)
    rights = _rights(policy)
    if kind not in KINDS:
        return {"refused": f"unknown request kind {kind!r}"}
    key = _key(kind, creator_id, market_id, subject)
    if conn.execute("SELECT 1 FROM proposals WHERE dedupe_key = %s AND status = 'open'", (key,)).fetchone():
        return {"refused": "the same request is already open"}
    ev = dict(evidence_ or {})
    if creator_id is not None:
        ev.setdefault("mode", evidence.mode_for(conn, creator_id))
        n_open = conn.execute("SELECT count(*) n FROM proposals WHERE creator_id = %s AND status = 'open'",
                              (creator_id,)).fetchone()["n"]
        if n_open >= int(rights.get("max_open_per_creator", 3)):
            return {"refused": f"{n_open} requests already open for this creator"}
    last = conn.execute("SELECT evidence, decided_at FROM proposals WHERE dedupe_key = %s AND status = 'rejected' "
                        "ORDER BY decided_at DESC LIMIT 1", (key,)).fetchone()
    if last and last["decided_at"] and now - last["decided_at"] < timedelta(days=float(rights.get("cooldown_days", 14))) \
            and _same_evidence(last["evidence"] or {}, ev):
        return {"refused": "rejected recently; file again with new evidence or after the cooldown"}
    allowance = (rights.get(from_role) or {}).get(f"{kind}_per_week")
    if allowance is not None and creator_id is not None:
        n = conn.execute("SELECT count(*) n FROM proposals WHERE kind = %s AND creator_id = %s AND from_role = %s "
                         "AND created_at > %s", (kind, creator_id, from_role, now - timedelta(days=7))).fetchone()["n"]
        if n >= int(allowance):
            return {"refused": f"{from_role} may file {allowance} {kind} requests a week per creator"}
    to_role = route(policy, kind)
    due = now + timedelta(days=float(rights.get("due_days", 3)))
    row = conn.execute(
        "INSERT INTO proposals (kind, from_role, to_role, creator_id, market_id, subject, argument, evidence, "
        "dedupe_key, job_id, due_at, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "RETURNING id", (kind, from_role, to_role, creator_id, market_id, Jsonb(subject), argument[:2000], Jsonb(ev),
                         key, job_id, due, now)).fetchone()
    if to_role == "owner":
        _tell_owner(conn, row["id"], kind, subject, argument, creator_id)
    if kind in (rights.get("auto_accept") or []):   # the requester's own call, within its allowance
        done = decide(conn, policy, row["id"], decided_by=to_role, accept=True,
                      reason="within the requester's rights", now=now)
        return {"id": str(row["id"]), "to_role": to_role, "status": done.get("status")}
    return {"id": str(row["id"]), "to_role": to_role}


def _same_evidence(old: dict, new: dict) -> bool:
    """No new evidence: the same finding ids and conviction (rounded) as the rejected request."""
    def view(e):
        return (sorted(map(str, e.get("finding_ids") or [])), round(float((e.get("conviction") or {}).get("mean") or 0), 2))
    return view(old) == view(new)


def _tell_owner(conn, pid, kind, subject, argument, creator_id) -> None:
    notify.owner_event(conn, "request_for_owner", subject={"proposal_id": str(pid), "kind": kind},
                       payload={"subject": subject, "argument": argument[:400]}, creator_id=creator_id,
                       dedupe_key=f"proposal:{pid}")


def open_for(conn, to_role: str, *, creator_id=None, market_id=None, limit: int = 20) -> list[dict]:
    rows = conn.execute(
        "SELECT id, kind, from_role, creator_id, market_id, subject, argument, evidence, created_at FROM proposals "
        "WHERE to_role = %s AND status = 'open' AND (%s::uuid IS NULL OR creator_id = %s) "
        "AND (%s::uuid IS NULL OR market_id = %s) ORDER BY created_at LIMIT %s",
        (to_role, creator_id, creator_id, market_id, market_id, limit)).fetchall()
    return [{**dict(r), "id": str(r["id"]), "creator_id": str(r["creator_id"]) if r["creator_id"] else None,
             "market_id": str(r["market_id"]) if r["market_id"] else None,
             "label": "SIMULATED (sandbox: not evidence of demand)" if (r["evidence"] or {}).get("mode") == "sandbox"
             else "real"} for r in rows]


def decide(conn, policy: Policy, proposal_id, *, decided_by: str, accept: bool, reason: str,
           now: datetime | None = None) -> dict:
    """Accept or reject an open request as its owner role; code applies an accepted one within rights and money."""
    now = now or datetime.now(timezone.utc)
    p = conn.execute("SELECT * FROM proposals WHERE id = %s", (proposal_id,)).fetchone()
    if not p or p["status"] != "open":
        return {"refused": "no such open request"}
    if decided_by not in (p["to_role"], "owner"):
        return {"refused": f"{decided_by} does not decide {p['kind']} requests ({p['to_role']} does)"}
    applied: dict = {}
    status = "rejected"
    if accept:
        applied = _apply(conn, policy, p, decided_by, now)
        status = "accepted" if not applied.get("refused") else ("open" if applied.get("escalated") else "rejected")
    resolution = {"by": decided_by, "accept": accept, "reason": reason[:1000], **applied}
    conn.execute("UPDATE proposals SET status = %s, resolution = %s, decided_at = CASE WHEN %s = 'open' THEN NULL "
                 "ELSE %s END, to_role = CASE WHEN %s THEN 'owner' ELSE to_role END WHERE id = %s",
                 (status, Jsonb(resolution), status, now, bool(applied.get("escalated")), proposal_id))
    return {"status": status, **resolution}


def _apply(conn, policy: Policy, p: dict, decided_by: str, now: datetime) -> dict:
    kind, subject, rights = p["kind"], p["subject"] or {}, _rights(policy)
    if kind in ("tier_change", "production_upgrade"):
        tiers = policy.get("brain", "tiers", default={}) or {}
        tier = str(subject.get("tier") or "")
        if tier not in (tiers.get("levels") or {}):
            return {"refused": f"unknown tier {tier!r}"}
        cost = _tier_month_delta(conn, policy, p["creator_id"], tier)
        left = _budget_left(conn, policy, now)
        if decided_by != "owner" and cost > left:
            return {"refused": f"the tier adds ~${cost:.2f} this month; ${left:.2f} is left", "escalated": True}
        conn.execute("INSERT INTO creator_states (creator_id, tier) VALUES (%s, %s) ON CONFLICT (creator_id) "
                     "DO UPDATE SET tier = EXCLUDED.tier, updated_at = now()", (p["creator_id"], tier))
        return {"applied": {"tier": tier, "usd_this_month": round(cost, 2)}}
    if kind == "budget":
        usd = float(subject.get("usd") or 0)
        cap = float((rights.get(decided_by) or {}).get("max_usd_per_request", 0)) if decided_by != "owner" else usd
        if usd > cap:
            _tell_owner(conn, p["id"], kind, subject, p["argument"], p["creator_id"])
            return {"refused": f"${usd:.2f} is above the {decided_by}'s ${cap:.2f} per request", "escalated": True}
        if usd > _budget_left(conn, policy, now):
            return {"refused": "not enough budget left this month"}
        return {"applied": {"usd": usd}}
    if kind == "research":
        job = queue.enqueue(conn, "brain.analyst", {"creator_id": str(p["creator_id"]),
                                                    "question": str(subject.get("question") or "")[:1000],
                                                    "proposal_id": str(p["id"])},
                            creator_id=p["creator_id"], idempotency_key=f"proposal:{p['id']}")
        return {"applied": {"analyst_job": str(job)}}
    if kind == "escalate":
        _tell_owner(conn, p["id"], kind, subject, p["argument"], p["creator_id"])
        return {"applied": {"told_owner": True}}
    return {"applied": {}}   # experiment, cohort: recorded; the requester's next plan or the launch flow acts on it


def _tier_month_delta(conn, policy: Policy, creator_id, tier: str) -> float:
    levels = (policy.get("brain", "tiers", default={}) or {}).get("levels") or {}
    cur = conn.execute("SELECT tier FROM creator_states WHERE creator_id = %s", (creator_id,)).fetchone()
    now_tier = (cur or {}).get("tier") or (policy.get("brain", "tiers", "default", default="economy"))
    day = float((levels.get(tier) or {}).get("usd_per_day", 0)) - float((levels.get(now_tier) or {}).get("usd_per_day", 0))
    return max(day, 0.0) * 30


def _budget_left(conn, policy: Policy, now: datetime) -> float:
    from . import portfolio
    b = portfolio.budget(conn, policy, now)
    return float(b.get("remaining_usd", b.get("available_usd", 0)) or 0)


def tier_of(conn, policy: Policy, creator_id) -> str:
    row = conn.execute("SELECT tier FROM creator_states WHERE creator_id = %s", (creator_id,)).fetchone()
    return (row or {}).get("tier") or str(policy.get("brain", "tiers", "default", default="economy"))


def tier_dials(policy: Policy, tier: str, dials: dict) -> dict:
    """A creator's dials within their tier: posts per day capped, video share capped (images_per_video floored)."""
    lv = ((policy.get("brain", "tiers", default={}) or {}).get("levels") or {}).get(tier) or {}
    out = dict(dials or {})
    if lv.get("posts_per_day_max") is not None:
        out["posts_per_day"] = {p: min(float(n), float(lv["posts_per_day_max"]))
                                for p, n in (out.get("posts_per_day") or {}).items()}
    if lv.get("video_share_max"):
        floor = (1 - float(lv["video_share_max"])) / float(lv["video_share_max"])   # photos per video
        out["images_per_video"] = max(float(out.get("images_per_video") or 0), round(floor, 2))
    return out


def expire_overdue(conn, now: datetime | None = None) -> int:
    """Requests their owner never decided are expired (the chief of staff reports them)."""
    now = now or datetime.now(timezone.utc)
    return len(conn.execute("UPDATE proposals SET status = 'expired', decided_at = %s WHERE status = 'open' "
                            "AND to_role <> 'owner' AND due_at < %s RETURNING id", (now, now)).fetchall())
