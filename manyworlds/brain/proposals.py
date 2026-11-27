"""Requests between roles: how the agency's agents ask each other for things and get a decision.

A talent manager asks the CEO for a better production tier, a market lead proposes a cohort, anyone escalates to
the owner. Every request has one owner who decides it once (policy `brain.rights.routes`), and code, not the
model, checks what each role may decide and what the money allows (`brain.rights`, `brain.tiers`, the month's
budget). That is the whole debate mechanism: an argument with evidence, a decision with a reason, no loops.

    file()    record a request (refused: unknown kind, too many open for the creator, the same request rejected
              within `cooldown_days` and filed again with no new evidence, over the requester's allowance)
    decide()  the owner role accepts or rejects; an accepted request is applied when its kind has an effect
              (tier_change sets creator_states.tier; research commissions the analyst with the question;
              intelligence commissions the IIA (brain.intelligence) within its cap;
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

KINDS = ("tier_change", "budget", "research", "experiment", "cohort", "production_upgrade", "escalate", "artist_concept",
         "intelligence")


def _rights(policy: Policy) -> dict:
    return policy.get("brain", "rights", default={}) or {}


def route(policy: Policy, kind: str) -> str:
    return str((_rights(policy).get("routes") or {}).get(kind, "owner"))


# The part of a subject that makes two requests the same request: a reworded argument or an extra field written by
# a model is not a new request (it would walk round the cooldown).
KEY_FIELDS = {"tier_change": ("tier",), "production_upgrade": ("tier",), "budget": ("usd",), "experiment": ("effect",)}


def _key(kind: str, creator_id, market_id, subject: dict) -> str:
    fields = KEY_FIELDS.get(kind)
    view = {k: subject.get(k) for k in fields} if fields else subject
    if "usd" in view:
        view["usd"] = round(float(view["usd"] or 0))
    return f"{kind}:{creator_id or '-'}:{market_id or '-'}:{json.dumps(view, sort_keys=True, default=str)}"


def file(conn, policy: Policy, *, kind: str, from_role: str, subject: dict, argument: str, evidence_: dict | None = None,
         creator_id=None, market_id=None, job_id=None, now: datetime | None = None) -> dict:
    """{id, to_role} for a recorded request, or {refused: why}."""
    now = now or datetime.now(timezone.utc)
    rights = _rights(policy)
    if kind not in KINDS:
        return {"refused": f"unknown request kind {kind!r}"}
    key = _key(kind, creator_id, market_id, subject)
    if job_id is not None:   # a rerun of the job that filed it (interrupted after filing): the same request
        again = conn.execute("SELECT id, to_role, status FROM proposals WHERE dedupe_key = %s AND job_id = %s",
                             (key, job_id)).fetchone()
        if again:
            return {"id": str(again["id"]), "to_role": again["to_role"], "status": again["status"], "repeat": True}
    if conn.execute("SELECT 1 FROM proposals WHERE dedupe_key = %s AND status = 'open'", (key,)).fetchone():
        return {"refused": "the same request is already open"}
    ev = dict(evidence_ or {})
    if creator_id is not None:
        ev.setdefault("mode", evidence.mode_for(conn, creator_id))
        # Requests waiting on the owner don't count: an absent owner must not stop a creator's other requests.
        n_open = conn.execute("SELECT count(*) n FROM proposals WHERE creator_id = %s AND status = 'open' "
                              "AND to_role <> 'owner'", (creator_id,)).fetchone()["n"]
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
        conv = e.get("conviction") or {}
        return (sorted(map(str, e.get("finding_ids") or [])),
                round(float(conv.get("probability", conv.get("mean")) or 0), 2))
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
    with conn.transaction():   # locked: the owner and the strategist can't both apply one request
        p = conn.execute("SELECT * FROM proposals WHERE id = %s FOR UPDATE", (proposal_id,)).fetchone()
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
        if applied.get("escalated"):   # now the owner's to decide: tell them at once, not at the 3-day nag
            _tell_owner(conn, p["id"], p["kind"], p["subject"] or {}, p["argument"] or "", p["creator_id"])
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
            return {"refused": f"${usd:.2f} is above the {decided_by}'s ${cap:.2f} per request", "escalated": True}
        if usd > _budget_left(conn, policy, now):
            return {"refused": "not enough budget left this month"}
        return {"applied": {"usd": usd, "usd_this_month": round(usd, 2)}}   # counted against the month's budget
    if kind == "research":
        job = queue.enqueue(conn, "brain.analyst", {"creator_id": str(p["creator_id"]),
                                                    "question": str(subject.get("question") or "")[:1000],
                                                    "proposal_id": str(p["id"])},
                            creator_id=p["creator_id"], idempotency_key=f"proposal:{p['id']}")
        return {"applied": {"analyst_job": str(job)}}
    if kind == "intelligence":
        return _commission_intelligence(conn, policy, p)
    if kind == "escalate":
        if decided_by == "owner" and subject.get("profile_changes"):
            return {"applied": {"profile_changes": _apply_profile_changes(conn, policy, p)}}
        _tell_owner(conn, p["id"], kind, subject, p["argument"], p["creator_id"])
        return {"applied": {"told_owner": True}}
    if kind == "experiment":
        return _open_experiment(conn, policy, p)
    if kind == "cohort":
        return _cohort_theses(conn, p)
    if kind == "artist_concept":
        return _artist_thesis(conn, p)
    return {"applied": {}}


def _commission_intelligence(conn, policy: Policy, p: dict) -> dict:
    """An accepted `intelligence` request: a brain.intelligence job (the IIA) on the question or hypothesis, with any
    named reference handles, capped at intelligence.agent.max_usd_per_commission (a larger ask is cut to the cap, not
    sent to the owner: no approval gate, the cap is the control)."""
    subject = p["subject"] or {}
    from ..research import external
    if not external.allowed(policy):   # owner 2026-11-27: no third-party creator research, whoever asks
        return {"applied": None, "declined": external.reason(policy)}
    cap = float(policy.get("intelligence", "agent", "max_usd_per_commission", default=5.0))
    usd = min(float(subject.get("usd") or cap), cap)
    text = str(subject.get("question") or subject.get("hypothesis") or p["argument"] or "")[:1000]
    payload = {"mode": str(subject.get("mode") or "pilot"), "hypothesis": text, "usd": usd,
               "proposal_id": str(p["id"]), "requested_by": p["from_role"]}
    for k in ("thesis_id", "handles", "platforms", "archetype"):
        if subject.get(k):
            payload[k] = subject[k]
    if payload["mode"] not in ("scout", "annotate", "distill", "pilot"):
        payload["mode"] = "pilot"
    job = queue.enqueue(conn, "brain.intelligence", payload, creator_id=p["creator_id"],
                        idempotency_key=f"proposal:{p['id']}")
    return {"applied": {"intelligence_job": str(job), "usd_cap": usd}}


def _apply_profile_changes(conn, policy: Policy, p: dict) -> list[dict]:
    """The owner accepted a talent manager's profile changes outside its rights: applied, except the locked paths
    (identity, the never-list, disclosure, the handle), which only the owner edits by hand."""
    from ..creator import profile as profile_mod
    from .agents.talent_manager import LOCKED, _value
    out = []
    for ch in (p["subject"] or {}).get("profile_changes") or []:
        path = str(ch.get("path") or "")
        if not path or any(path == lock or path.startswith(lock + ".") for lock in LOCKED):
            out.append({"path": path, "skipped": "locked"})
            continue
        try:
            profile_mod.set_path(conn, policy, p["creator_id"], path, _value(ch.get("value")), author="owner",
                                 reason=str(ch.get("reason") or "accepted request"))
            out.append({"path": path, "applied": True})
        except (ValueError, LookupError) as e:
            out.append({"path": path, "skipped": str(e)})
    return out


def _open_experiment(conn, policy: Policy, p: dict) -> dict:
    """An accepted experiment request opens an experiment the showrunner's slots follow (brain.experiments), when
    its effect is one an idea can be told to follow: attribute:<key>=<value>, dimension:<name> or format:<name>."""
    from . import experiments
    subject = p["subject"] or {}
    key = str(subject.get("effect") or "")
    arms = experiments.arms_for(key)
    if not arms:
        return {"refused": "an experiment needs an effect: attribute:<key>=<value>, dimension:<name> or format:<name>"}
    mode = evidence.mode_for(conn, p["creator_id"])
    if len(experiments.open_for(conn, p["creator_id"])) >= int(experiments._cfg(policy).get("max_open_per_creator", 1)):
        return {"refused": "the creator already runs as many experiments as policy allows"}
    need = experiments.sample_size(policy, experiments._outcome_sd(conn, policy, p["creator_id"], mode))
    row = conn.execute(
        "INSERT INTO experiments (creator_id, hypothesis, arms, metric, min_sample, status, effect_key, mode) "
        "VALUES (%s, %s, %s, 'reach_lift', %s, 'open', %s, %s) "
        "ON CONFLICT (creator_id, effect_key, mode) WHERE effect_key IS NOT NULL DO NOTHING RETURNING id",
        (p["creator_id"], str(subject.get("hypothesis") or p["argument"] or key)[:1000], Jsonb(arms), need, key,
         mode)).fetchone()
    if not row:
        return {"refused": "this effect was already tested for the creator"}
    return {"applied": {"experiment_id": str(row["id"])}}


def _cohort_theses(conn, p: dict) -> dict:
    """An accepted cohort becomes proposed theses (one per concept). Launching any of them still takes the
    strategist's launch slots and the owner's approval (brain.launches), like every thesis."""
    ids = []
    for c in ((p["subject"] or {}).get("concepts") or [])[:10]:
        c = c if isinstance(c, dict) else {"concept": str(c)}
        body = {"title": str(c.get("concept") or "")[:80], "archetype": str(c.get("archetype") or ""),
                "audience": "", "why_now": str(p["argument"] or "")[:1000], "format_mix": [],
                "differentiation": str(c.get("why_distinct") or ""), "risks": "", "production_fit": "",
                "concept": c.get("concept"), "market_id": str(p["market_id"]) if p["market_id"] else None,
                "source": {"proposal_id": str(p["id"])}}
        row = conn.execute("INSERT INTO theses (summary, evidence, body) VALUES (%s, %s, %s) RETURNING id",
                           (body["title"] or "cohort concept", Jsonb({"proposal_id": str(p["id"])}), Jsonb(body))).fetchone()
        ids.append(str(row["id"]))
    return {"applied": {"theses": ids}} if ids else {"refused": "the cohort named no concepts"}


def _artist_thesis(conn, p: dict) -> dict:
    """An accepted artist concept (A&R to head_of_music) becomes a proposed thesis for a music artist (vertical music).
    Launching it still takes the strategist's launch slots and the owner's approval (brain.launches)."""
    c = p["subject"] or {}
    if not c.get("concept"):
        return {"refused": "the pitch has no concept"}
    body = {"title": str(c.get("concept"))[:80], "archetype": "singer_songwriter", "vertical": "music",
            "audience": "", "why_now": str(c.get("why_now") or p["argument"] or "")[:1000], "format_mix": [],
            "differentiation": str(c.get("why_distinct") or ""), "risks": "", "production_fit": "",
            "concept": c.get("concept"), "music": {k: c.get(k) for k in ("genre", "subgenre", "personality",
                                                                         "sonic_identity", "story")},
            "source": {"proposal_id": str(p["id"]), "finding_ids": c.get("finding_ids") or []}}
    row = conn.execute("INSERT INTO theses (summary, evidence, body) VALUES (%s, %s, %s) RETURNING id",
                       (body["title"], Jsonb({"proposal_id": str(p["id"])}), Jsonb(body))).fetchone()
    return {"applied": {"thesis_id": str(row["id"])}}


def _tier_month_delta(conn, policy: Policy, creator_id, tier: str) -> float:
    levels = (policy.get("brain", "tiers", default={}) or {}).get("levels") or {}
    cur = conn.execute("SELECT tier FROM creator_states WHERE creator_id = %s", (creator_id,)).fetchone()
    now_tier = (cur or {}).get("tier") or (policy.get("brain", "tiers", "default", default="economy"))
    day = float((levels.get(tier) or {}).get("usd_per_day", 0)) - float((levels.get(now_tier) or {}).get("usd_per_day", 0))
    return max(day, 0.0) * 30


def _budget_left(conn, policy: Policy, now: datetime) -> float:
    """The month's budget left, less what requests accepted this month already committed (a tier upgrade's cost
    is spent over the month: five upgrades in one run must not each pass against the same remainder)."""
    from . import portfolio
    b = portfolio.budget(conn, policy, now)
    left = float(b.get("remaining_usd", b.get("available_usd", 0)) or 0)
    month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    committed = conn.execute(
        "SELECT coalesce(sum((resolution->'applied'->>'usd_this_month')::numeric), 0) s FROM proposals "
        "WHERE status = 'accepted' AND decided_at >= %s", (month,)).fetchone()["s"]
    return left - float(committed or 0)


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
