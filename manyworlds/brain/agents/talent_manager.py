"""The talent manager: one per creator, the coordinator of that creator's career, reports to the CEO (strategist).

It is a role with durable state (the profile and dossier, its briefs, its conviction, its open requests and its
predictions), activated as a job when it is due; nothing runs per creator in between. Three modes (payload
{creator_id, mode}):

  onboard   builds the dossier sections the profile lacks (manyworlds.creator.dossier: archetypes, positioning,
            audience insight, story, content system), one remembered step per section, each a profile version
            authored by the talent manager; a positioning too close to a sibling (word overlap past
            `brain.talent_manager.max_overlap`) is rewritten once against it. Then a first review.
  review    the career review, on a cadence by stage (`review_days_by_stage`): dossier, playbooks, recent posts and
            scores, what production made and dropped and what it cost, findings and the analyst's guidance (labelled
            real or simulated), its market's learnings, its tier, its last brief, and what became of its requests.
  diagnose  the same review when something happened (a `drop` event routes here: brain.events.routes), with the
            event in front of it and the instruction to find out why before changing course.

A review writes:
  - the brief the showrunner plans from (decision `talent_brief`; the strategist reads the latest per creator)
  - its conviction: the probability she reaches the next stage within a horizon, with an interval and the case
    for it (a `convictions` row). A creator with little traction but real upside can be argued for here.
  - typed requests (manyworlds.brain.proposals): a tier change to the CEO, research to the analyst,
    escalations to the owner. Code checks the rights and the money; a rejected request comes back only with new
    evidence.
  - predictions (manyworlds.brain.predictions): checkable claims about her next weeks, scored later
  - profile changes within `brain.talent_manager.editable` (identity, the never-list, disclosure and handle never);
    anything else goes to the owner as an escalation
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Literal

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from ... import direction, queue
from ...creator import dossier
from ...creator import profile as profile_mod
from ...worker.runner import NonRetryable
from .. import archetypes, evidence, predictions, proposals
from ..agent import AgentContext, record_decision, register, remember
from . import analyst

LOCKED = ("identity", "persona.never", "disclosure_line", "handle")
ROLE = "talent_manager"


class ProfileChange(BaseModel):
    path: str = Field(description="dotted profile path, e.g. content_system.series or look.wardrobe")
    value: str = Field(description="the new value as JSON (a string, list or object)")
    reason: str


class Conviction(BaseModel):
    probability: float = Field(description="0-1: how likely she reaches her next stage within horizon_weeks")
    low: float = Field(description="0-1: the pessimistic end of that")
    high: float = Field(description="0-1: the optimistic end of that")
    horizon_weeks: int
    case: str = Field(description="the argument: why she can become influential, or why not, with the evidence")


class Request(BaseModel):
    kind: Literal["tier_change", "research", "experiment", "production_upgrade", "escalate"]
    subject: str = Field(description='what is asked, as JSON: {"tier": "standard"}, {"question": "..."}, '
                                     '{"effect": "format:carousel" | "attribute:<key>=<value>" | "dimension:<name>", '
                                     '"hypothesis": "..."}, {"ask": "..."}')
    argument: str = Field(description="why, with the evidence; for money, what it buys and what it should show")


class Claim(BaseModel):
    claim: str = Field(description="what you expect, in words")
    metric: Literal["median_reach", "reach_growth"]
    comparator: Literal["above", "below"]
    threshold: float
    probability: float = Field(description="0-1: how sure you are")
    horizon_days: int


class CareerReview(BaseModel):
    state: str = Field(description="where her brand stands now, in two sentences")
    diagnosis: str = Field(description="why things are going as they are (for a diagnosis: the most likely causes, "
                                       "ranked, and what would tell them apart); empty when nothing needs explaining")
    working: list[str]
    not_working: list[str]
    focus: list[str] = Field(description="what the next batches are about (pillars, series, a thread to build)")
    series_next: list[str] = Field(description="series names to run next, from her content system")
    capabilities_to_try: list[str] = Field(description="capability ids (group.name) worth trying next; may be empty")
    stop: list[str] = Field(description="what to stop doing")
    hypotheses: list[str] = Field(description="what the next batches should test about her audience")
    brief_for_showrunner: str = Field(description="the brief the showrunner writes the next ideas from: concrete, "
                                                  "in her voice's terms, what to make and what to avoid")
    conviction: Conviction
    predictions: list[Claim] = Field(description="1-3 checkable claims about her next weeks; none before she posts")
    requests: list[Request] = Field(description="requests to other roles; often none")
    profile_changes: list[ProfileChange] = Field(description="changes to her profile; often none")


def _cfg(policy) -> dict:
    return policy.get("brain", "talent_manager", default={}) or {}


def latest_brief(conn, creator_id) -> dict | None:
    """The creator's latest brief: {id, created_at, output} or None."""
    return conn.execute("SELECT id, created_at, output FROM decisions WHERE kind = 'talent_brief' AND creator_id = %s "
                        "ORDER BY created_at DESC LIMIT 1", (creator_id,)).fetchone()


def editable(policy, path: str) -> bool:
    if any(path == lock or path.startswith(lock + ".") for lock in LOCKED):
        return False
    return any(path == p or path.startswith(p + ".") for p in _cfg(policy).get("editable") or [])


def _value(raw: str):
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def production_stats(conn, creator_id, days: int) -> dict:
    """What production made of her ideas lately: per format planned/ready/dropped, the top drop reasons, and
    spend per ready post (the ledger, every category but attribution-only `gpu`)."""
    rows = conn.execute(
        "SELECT format, status, count(*) n FROM posts WHERE creator_id = %s AND created_at > now() - %s "
        "AND status <> 'simulated' GROUP BY 1, 2", (creator_id, timedelta(days=days))).fetchall()
    by: dict = {}
    for r in rows:
        by.setdefault(r["format"], {})[r["status"]] = r["n"]
    drops = [r["why"] for r in conn.execute(
        "SELECT body->>'drop_reason' why, count(*) n FROM posts WHERE creator_id = %s AND status = 'dropped' "
        "AND created_at > now() - %s GROUP BY 1 ORDER BY 2 DESC LIMIT 5", (creator_id, timedelta(days=days))).fetchall()
        if r["why"]]
    spend = conn.execute("SELECT coalesce(sum(usd), 0) u FROM cost_entries WHERE creator_id = %s AND category <> 'gpu' "
                         "AND created_at > now() - %s", (creator_id, timedelta(days=days))).fetchone()["u"]
    ready = sum(v.get("ready", 0) + v.get("published", 0) for v in by.values())
    return {"by_format": by, "top_drop_reasons": drops, "usd": round(float(spend), 2),
            "usd_per_ready_post": round(float(spend) / ready, 2) if ready else None}


def _recent(conn, creator_id, n: int) -> list[dict]:
    rows = conn.execute(
        """SELECT p.format, p.status, p.caption, p.body->>'series' series, p.body->>'pillar' pillar, e.scores
           FROM posts p LEFT JOIN ideas i ON i.post_id = p.id
           LEFT JOIN LATERAL (SELECT scores FROM evaluations WHERE idea_id = i.id AND role = 'evaluator'
                              ORDER BY created_at DESC LIMIT 1) e ON true
           WHERE p.creator_id = %s AND p.status <> 'simulated'
           ORDER BY coalesce(p.scheduled_for, p.created_at) DESC LIMIT %s""", (creator_id, n)).fetchall()
    return [{k: v for k, v in dict(r).items() if v} for r in rows]


def _requests_history(conn, creator_id, n: int = 10) -> list[dict]:
    """Its recent requests and what became of them: the loop closes here."""
    rows = conn.execute("SELECT kind, subject, status, resolution, created_at FROM proposals WHERE creator_id = %s "
                        "AND from_role = %s ORDER BY created_at DESC LIMIT %s", (creator_id, ROLE, n)).fetchall()
    return [{"kind": r["kind"], "subject": r["subject"], "status": r["status"],
             "reason": (r["resolution"] or {}).get("reason") or (r["resolution"] or {}).get("refused")} for r in rows]


def _market_learnings(conn, creator_id) -> dict | None:
    row = conn.execute("SELECT m.name, m.dossier FROM creators c JOIN markets m ON m.id = c.market_id WHERE c.id = %s",
                       (creator_id,)).fetchone()
    if not row:
        return None
    d = row["dossier"] or {}
    return {"market": row["name"], "thesis": d.get("thesis"), "learnings": d.get("learnings")}


def _onboard(ctx: AgentContext, creator_id) -> dict:
    conn, policy = ctx.conn, ctx.policy
    built = []
    for section in dossier.SECTIONS:
        cur = profile_mod.current(conn, creator_id)
        prof = cur[1]
        if prof.get(section):
            continue

        def compute(section=section, prof=prof):
            return {"kind": "dossier_section", "creator_id": creator_id, "inputs": {"section": section},
                    "output": dossier.build_section(conn, policy, ctx.llm, creator_id, section, prof),
                    "rationale": f"dossier section {section}", "model": ctx.llm.model_for(_cfg(policy).get("role", ROLE))}
        patch = remember(ctx, f"section:{section}", compute)
        profile_mod.update(conn, policy, creator_id, patch, author=ROLE, reason=f"dossier: {section}")
        built.append(section)
    prof = profile_mod.current(conn, creator_id)[1]
    diff = dossier.differentiation(conn, policy, creator_id, prof)
    if diff["too_close"]:
        sib = next(s for s in dossier.siblings(conn, creator_id) if s["handle"] == diff["closest"])
        extra = (f"Her current positioning overlaps {diff['closest']} too much ({diff['overlap']} word overlap). "
                 f"Their positioning: {json.dumps(sib['profile'].get('positioning'), default=str)}. Rewrite hers so "
                 f"the two are clearly different accounts.")

        def rewrite():
            return {"kind": "dossier_section", "creator_id": creator_id, "inputs": {"section": "positioning", **diff},
                    "output": dossier.build_section(conn, policy, ctx.llm, creator_id, "positioning", prof, extra=extra),
                    "rationale": "positioning too close to a sibling"}
        profile_mod.update(conn, policy, creator_id, remember(ctx, "differentiate", rewrite), author=ROLE,
                           reason=f"dossier: positioning apart from {diff['closest']}")
        prof = profile_mod.current(conn, creator_id)[1]
        diff = {**dossier.differentiation(conn, policy, creator_id, prof), "rewritten": True}
    return {"built": built, "differentiation": diff}


def _event(conn, event_id) -> dict | None:
    if not event_id:
        return None
    row = conn.execute("SELECT type, payload, created_at FROM events WHERE id = %s", (event_id,)).fetchone()
    return {"type": row["type"], "payload": row["payload"], "at": row["created_at"].isoformat()} if row else None


def _review_prompt(ctx: AgentContext, creator_id, prof: dict, mode: str, event: dict | None) -> str:
    conn, policy = ctx.conn, ctx.policy
    cfg = _cfg(policy)
    ev_mode = evidence.mode_for(conn, creator_id)
    findings = [{"finding": f["statement"], "grade": f["grade"]}
                for f in evidence.findings(conn, mode=ev_mode, creator_id=creator_id)]
    last = analyst.latest(conn, creator_id)
    prev = latest_brief(conn, creator_id)
    who = {k: v for k, v in prof.items() if k not in ("dials", "identity", "commercialization")}
    label = "SIMULATED (sandbox test, not evidence of demand)" if ev_mode == "sandbox" else "real"
    tier = proposals.tier_of(conn, policy, creator_id)
    tiers = (policy.get("brain", "tiers", "levels", default={}) or {})
    parts = [
        f"Her profile and dossier:\n{json.dumps(who, default=str)}",
        f"Her archetypes' playbooks: {json.dumps(archetypes.playbooks(policy, prof), default=str)}",
        f"Recent posts, newest first, with idea scores where scored:\n"
        f"{json.dumps(_recent(conn, creator_id, int(cfg.get('recent_posts', 30))), default=str)}",
        f"What production made of her ideas in the last {int(cfg.get('stats_days', 14))} days (dropped posts cost "
        f"money and made nothing): {json.dumps(production_stats(conn, creator_id, int(cfg.get('stats_days', 14))))}",
        f"Findings about her audience ({label}): {json.dumps(findings) if findings else 'none yet'}",
        f"The analyst's latest guidance ({label}): {json.dumps((last['output'] or {}).get('guidance')) if last else 'none'}",
        f"Her market's thesis and learnings (from its market lead): {json.dumps(_market_learnings(conn, creator_id), default=str)}",
        f"Her production tier: {tier} (tiers: {json.dumps(tiers)}). The CEO sets tiers; you may ask for a change "
        f"with the evidence that it would pay off.",
        f"Your previous brief: {json.dumps(prev['output'], default=str) if prev else 'none (first review)'}",
        f"Your recent requests and what became of them: {json.dumps(_requests_history(conn, creator_id), default=str)}",
        f"Profile paths you may change yourself (prefixes): {cfg.get('editable')}; never: {list(LOCKED)}. Anything "
        f"else is an `escalate` request. Never touch: {policy.get('scope', 'excluded_topics')}.",
    ]
    if mode == "diagnose" and event:
        parts.append(f"WHAT HAPPENED (diagnose this before changing course): {json.dumps(event, default=str)}. Rank the "
                     f"likely causes (content, production quality, cadence, audience fit, distribution), say what "
                     f"would tell them apart, and ask for research where the data can't.")
    parts.append("Review her career as her talent manager. Be specific and forensic: what is building her brand, what "
                 "is diluting it, what the next batches are for. State your own conviction about her future and argue "
                 "it, even against how she is doing so far, when the evidence supports upside. Predictions must be "
                 "checkable within their horizon. Simulated findings only show the loop works; never treat them as "
                 "what real people want.")
    return "\n\n".join(parts)


def _review(ctx: AgentContext, creator_id, mode: str, event: dict | None) -> dict:
    conn, policy = ctx.conn, ctx.policy
    version, prof = profile_mod.current(conn, creator_id)
    role = _cfg(policy).get("role", ROLE)

    def compute():
        out = ctx.llm.structured(role, system=profile_mod.persona_system(policy, dossier.SYSTEM),
                                 content=_review_prompt(ctx, creator_id, prof, mode, event), schema=CareerReview,
                                 effort=_cfg(policy).get("diagnose_effort" if mode == "diagnose" else "effort",
                                                         "medium"),
                                 creator_id=creator_id)
        return {"kind": "talent_review", "creator_id": creator_id,
                "inputs": {"profile_version": version, "mode": mode, "event": event},
                "output": out.model_dump(), "rationale": out.state, "model": ctx.llm.model_for(role)}
    review = remember(ctx, "review", compute)
    applied, needs_owner = [], []
    for ch in review.get("profile_changes") or []:
        if not editable(policy, ch["path"]):
            needs_owner.append(ch)
            continue
        def change(ch=ch):
            saved = dossier.checked_change(policy, prof, ch["path"], _value(ch["value"]))
            for path, value in saved.items():
                profile_mod.set_path(conn, policy, creator_id, path, value, author=ROLE,
                                     reason=ch["reason"] or "career review")
            return {"kind": "profile_change", "creator_id": creator_id, "inputs": {"path": ch["path"]},
                    "output": {"saved": sorted(saved)}, "rationale": ch["reason"] or ""}
        try:   # once per job: a rerun finds the recorded change and saves no duplicate version
            remember(ctx, f"change:{ch['path']}", change)
            applied.append(ch)
        except (ValueError, LookupError) as e:
            needs_owner.append({**ch, "refused": str(e)})
    conv = review.get("conviction") or {}
    finding_ids = [str(f["id"]) for f in evidence.findings(conn, mode=evidence.mode_for(conn, creator_id),
                                                           creator_id=creator_id)]
    filed = []
    if needs_owner:
        filed.append(proposals.file(conn, policy, kind="escalate", from_role=ROLE, creator_id=creator_id,
                                    subject={"profile_changes": needs_owner},
                                    argument="profile changes outside the talent manager's rights",
                                    evidence_={"conviction": conv}, job_id=ctx.job.id))
    for r in review.get("requests") or []:
        filed.append(proposals.file(conn, policy, kind=r["kind"], from_role=ROLE, creator_id=creator_id,
                                    subject=_subject(r.get("subject")), argument=r.get("argument") or "",
                                    evidence_={"conviction": conv, "finding_ids": finding_ids}, job_id=ctx.job.id))
    ok_caps = set(direction.capability_ids(policy, available_only=True))
    brief = {k: review.get(k) for k in ("state", "diagnosis", "focus", "series_next", "stop", "hypotheses",
                                        "brief_for_showrunner", "conviction")}
    brief["capabilities_to_try"] = [c for c in review.get("capabilities_to_try") or [] if c in ok_caps]
    brief["requests"] = filed
    decision = record_decision(ctx, "talent_brief", step="brief", creator_id=creator_id, inputs={"mode": mode},
                               output=brief, rationale=review.get("state", ""), model=ctx.llm.model_for(role))
    if conv and not conn.execute("SELECT 1 FROM convictions WHERE inputs->>'decision_id' = %s",
                                 (str(decision),)).fetchone():
        conn.execute("INSERT INTO convictions (subject, subject_id, mean, low, high, inputs) VALUES "
                     "('creator', %s, %s, %s, %s, %s)",
                     (creator_id, _p(conv.get("probability")), _p(conv.get("low")), _p(conv.get("high")),
                      Jsonb({"decision_id": str(decision), "role": ROLE, "horizon_weeks": conv.get("horizon_weeks"),
                             "case": conv.get("case")})))
    made = 0
    if not conn.execute("SELECT 1 FROM predictions WHERE decision_id = %s", (decision,)).fetchone():
        for c in review.get("predictions") or []:
            made += predictions.record(conn, role=ROLE, creator_id=creator_id, decision_id=decision,
                                       claim=c["claim"], metric=c["metric"], comparator=c["comparator"],
                                       threshold=c["threshold"], probability=c["probability"],
                                       horizon_days=c["horizon_days"]) is not None
    return {"applied": len(applied), "needs_owner": len(needs_owner),
            "requests": sum(1 for f in filed if f.get("id")), "refused": [f["refused"] for f in filed if f.get("refused")],
            "predictions": made}


def _p(v) -> float:
    try:
        return min(max(float(v), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


def _subject(raw) -> dict:
    v = _value(raw) if isinstance(raw, str) else raw
    return v if isinstance(v, dict) else {"text": str(raw or "")}


class TalentManager:
    name = ROLE

    def run(self, ctx: AgentContext) -> dict:
        p = ctx.job.payload or {}
        creator_id = p.get("creator_id")
        mode = p.get("mode") or ("diagnose" if p.get("event_id") else "review")
        if not creator_id or mode not in ("onboard", "review", "diagnose"):
            raise NonRetryable("talent_manager payload needs creator_id and mode onboard|review|diagnose")
        if not profile_mod.current(ctx.conn, creator_id):
            raise NonRetryable("the creator has no profile yet (the launch builds it first)")
        out = {"mode": mode}
        if mode == "onboard":
            out["onboard"] = _onboard(ctx, creator_id)
        out["review"] = _review(ctx, creator_id, mode, _event(ctx.conn, p.get("event_id")))
        return out


register(TalentManager())


def ensure_reviews(conn, policy, *, now: datetime | None = None) -> list:
    """A review for each active portfolio creator when its stage's cadence is due (`review_days_by_stage`, days;
    `default`): exploring and growing creators often, paused ones rarely. One job per creator per period."""
    now = now or datetime.now(timezone.utc)
    cfg = _cfg(policy)
    if not cfg.get("enabled"):
        return []
    cadence = cfg.get("review_days_by_stage") or {}
    rows = conn.execute(
        "SELECT c.id, coalesce(s.stage, 'exploring') stage FROM creators c LEFT JOIN creator_states s ON s.creator_id = c.id "
        "WHERE c.status = 'active' AND c.anchors->'simulation' IS NULL "
        "AND EXISTS (SELECT 1 FROM creator_profiles p WHERE p.creator_id = c.id)").fetchall()
    out = []
    for r in rows:
        days = max(int(cadence.get(r["stage"], cadence.get("default", 7))), 1)
        period = int(now.timestamp() // (days * 86400))
        cur = profile_mod.current(conn, r["id"])
        # A creator from before the agency (or whose launch skipped it) has no dossier: build it first.
        mode = "onboard" if cur and dossier.missing(cur[1]) else "review"
        out.append(queue.enqueue(conn, "brain.talent_manager", {"creator_id": str(r["id"]), "mode": mode},
                                 creator_id=r["id"], idempotency_key=f"talent:{r['id']}:{days}d:{period}"))
    return out
