"""The talent manager: one per creator, owns their identity and career, reports to the strategist (the CEO).

Two modes, one job each (payload {creator_id, mode}):

  onboard  builds the dossier sections the profile lacks (manyworlds.creator.dossier: archetypes, positioning,
           audience insight, story, content system), one remembered step per section, each saved as a profile
           version authored by the talent manager. If the positioning ends too close to a sibling (word overlap past
           `brain.talent_manager.max_overlap`), it is rewritten once against that sibling. Then a first brief.
  review   the weekly career review: reads the dossier, the archetype playbooks, recent posts and their scores, what
           production accepted and dropped, the analyst's findings (labelled real or simulated) and its own last
           brief; writes the brief the showrunner plans the next batches from, changes the profile within its
           authority (policy `brain.talent_manager.editable`: path prefixes; identity, never-list and disclosure are
           never its call), and puts anything else to the owner.

The brief (decision kind `talent_brief`) is what the agency runs on: the showrunner reads the latest one for the
creator, the strategist reads every creator's latest one in its portfolio snapshot.
"""

from __future__ import annotations

import json
from datetime import timedelta

from pydantic import BaseModel, Field

from ... import direction, notify, queue
from ...creator import dossier
from ...creator import profile as profile_mod
from ...worker.runner import NonRetryable
from .. import archetypes, evidence
from ..agent import AgentContext, record_decision, register, remember
from . import analyst

LOCKED = ("identity", "persona.never", "disclosure_line", "handle")


class ProfileChange(BaseModel):
    path: str = Field(description="dotted profile path, e.g. content_system.series or look.wardrobe")
    value: str = Field(description="the new value as JSON (a string, list or object)")
    reason: str


class Ask(BaseModel):
    to: str = Field(description="strategist or owner")
    ask: str
    why: str


class CareerReview(BaseModel):
    state: str = Field(description="where her brand stands now, in two sentences")
    working: list[str]
    not_working: list[str]
    focus: list[str] = Field(description="what the next batches are about (pillars, series, a thread to build)")
    series_next: list[str] = Field(description="series names to run next, from her content system")
    capabilities_to_try: list[str] = Field(description="capability ids (group.name) worth trying next; may be empty")
    stop: list[str] = Field(description="what to stop doing")
    hypotheses: list[str] = Field(description="what the next batches should test about her audience")
    brief_for_showrunner: str = Field(description="the brief the showrunner writes the next ideas from: concrete, "
                                                  "in her voice's terms, what to make and what to avoid")
    profile_changes: list[ProfileChange] = Field(description="changes to her profile; often none")
    asks: list[Ask] = Field(description="requests to the strategist or the owner; often none")


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
                         "AND at > now() - %s", (creator_id, timedelta(days=days))).fetchone()["u"]
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
                    "rationale": f"dossier section {section}", "model": ctx.llm.model_for(_cfg(policy).get("role", "talent_manager"))}
        patch = remember(ctx, f"section:{section}", compute)
        profile_mod.update(conn, policy, creator_id, patch, author="talent_manager", reason=f"dossier: {section}")
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
        profile_mod.update(conn, policy, creator_id, remember(ctx, "differentiate", rewrite), author="talent_manager",
                           reason=f"dossier: positioning apart from {diff['closest']}")
        prof = profile_mod.current(conn, creator_id)[1]
        diff = {**dossier.differentiation(conn, policy, creator_id, prof), "rewritten": True}
    return {"built": built, "differentiation": diff}


def _review_prompt(ctx: AgentContext, creator_id, prof: dict) -> str:
    conn, policy = ctx.conn, ctx.policy
    cfg = _cfg(policy)
    mode = evidence.mode_for(conn, creator_id)
    findings = [{"finding": f["statement"], "grade": f["grade"]}
                for f in evidence.findings(conn, mode=mode, creator_id=creator_id)]
    last = analyst.latest(conn, creator_id)
    prev = latest_brief(conn, creator_id)
    who = {k: v for k, v in prof.items() if k not in ("dials", "identity", "commercialization")}
    label = "SIMULATED (sandbox test, not evidence of demand)" if mode == "sandbox" else "real"
    return "\n\n".join([
        f"Her profile and dossier:\n{json.dumps(who, default=str)}",
        f"Her archetypes' playbooks: {json.dumps(archetypes.playbooks(policy, prof), default=str)}",
        f"Recent posts, newest first, with idea scores where scored:\n"
        f"{json.dumps(_recent(conn, creator_id, int(cfg.get('recent_posts', 30))), default=str)}",
        f"What production made of her ideas in the last {int(cfg.get('stats_days', 14))} days (dropped posts cost "
        f"money and made nothing): {json.dumps(production_stats(conn, creator_id, int(cfg.get('stats_days', 14))))}",
        f"Findings about her audience ({label}): {json.dumps(findings) if findings else 'none yet'}",
        f"The analyst's latest guidance ({label}): {json.dumps((last['output'] or {}).get('guidance')) if last else 'none'}",
        f"Your previous brief: {json.dumps(prev['output'], default=str) if prev else 'none (first review)'}",
        f"Profile paths you may change yourself (prefixes): {cfg.get('editable')}; never: {list(LOCKED)}. "
        f"Anything else goes in `asks` to the owner. Never touch: {policy.get('scope', 'excluded_topics')}.",
        "Review her career as her talent manager. Be specific and forensic: what is building her brand, what is "
        "diluting it, what the next batches are for. Simulated findings only show the loop works; never treat them "
        "as what real people want."])


def _review(ctx: AgentContext, creator_id) -> dict:
    conn, policy = ctx.conn, ctx.policy
    version, prof = profile_mod.current(conn, creator_id)
    role = _cfg(policy).get("role", "talent_manager")

    def compute():
        out = ctx.llm.structured(role, system=profile_mod.persona_system(policy, dossier.SYSTEM),
                                 content=_review_prompt(ctx, creator_id, prof), schema=CareerReview,
                                 effort=_cfg(policy).get("effort", "medium"), creator_id=creator_id)
        return {"kind": "talent_review", "creator_id": creator_id, "inputs": {"profile_version": version},
                "output": out.model_dump(), "rationale": out.state, "model": ctx.llm.model_for(role)}
    review = remember(ctx, "review", compute)
    applied, needs_owner = [], []
    for ch in review.get("profile_changes") or []:
        if not editable(policy, ch["path"]):
            needs_owner.append(ch)
            continue
        try:
            profile_mod.set_path(conn, policy, creator_id, ch["path"], _value(ch["value"]), author="talent_manager",
                                 reason=ch["reason"] or "career review")
            applied.append(ch)
        except (ValueError, LookupError) as e:
            needs_owner.append({**ch, "refused": str(e)})
    owner_asks = [a for a in review.get("asks") or [] if a.get("to") == "owner"]
    if needs_owner or owner_asks:
        notify.owner_event(conn, "talent_needs_owner", subject={"creator_id": str(creator_id)},
                           payload={"changes": needs_owner, "asks": owner_asks}, creator_id=creator_id,
                           dedupe_key=f"talent_needs_owner:{ctx.job.id}")
    ok_caps = set(direction.capability_ids(policy, available_only=True))
    brief = {k: review.get(k) for k in ("state", "focus", "series_next", "stop", "hypotheses", "brief_for_showrunner")}
    brief["capabilities_to_try"] = [c for c in review.get("capabilities_to_try") or [] if c in ok_caps]
    brief["asks_for_strategist"] = [a for a in review.get("asks") or [] if a.get("to") == "strategist"]
    record_decision(ctx, "talent_brief", step="brief", creator_id=creator_id, inputs={"review": "review"},
                    output=brief, rationale=review.get("state", ""), model=ctx.llm.model_for(role))
    return {"applied": len(applied), "needs_owner": len(needs_owner) + len(owner_asks)}


class TalentManager:
    name = "talent_manager"

    def run(self, ctx: AgentContext) -> dict:
        p = ctx.job.payload or {}
        creator_id, mode = p.get("creator_id"), p.get("mode", "review")
        if not creator_id or mode not in ("onboard", "review"):
            raise NonRetryable("talent_manager payload needs creator_id and mode onboard|review")
        if not profile_mod.current(ctx.conn, creator_id):
            raise NonRetryable("the creator has no profile yet (the launch builds it first)")
        out = {"mode": mode}
        if mode == "onboard":
            out["onboard"] = _onboard(ctx, creator_id)
        out["review"] = _review(ctx, creator_id)
        return out


register(TalentManager())


def ensure_reviews(conn, policy, *, now) -> list:
    """One review per active portfolio creator per ISO week (weekday `review_weekday`, 0 = Monday)."""
    cfg = _cfg(policy)
    if not cfg.get("enabled") or now.weekday() != int(cfg.get("review_weekday", 0)):
        return []
    week = now.strftime("%G-W%V")
    rows = conn.execute("SELECT c.id FROM creators c WHERE c.status = 'active' AND c.anchors->'simulation' IS NULL "
                        "AND EXISTS (SELECT 1 FROM creator_profiles p WHERE p.creator_id = c.id)").fetchall()
    return [queue.enqueue(conn, "brain.talent_manager", {"creator_id": str(r["id"]), "mode": "review"},
                          creator_id=r["id"], idempotency_key=f"talent:{r['id']}:{week}") for r in rows]
