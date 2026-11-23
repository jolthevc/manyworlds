"""The market lead: the owner of one market's thesis (fitness, fashion, slow living...), a division head under the
CEO. Activated, not appointed: the CEO covers every market until one holds `brain.market_lead.min_creators`
portfolio creators, or the CEO opens it (markets.dossier.lead = "open"). Then a run every `review_days`.

It reads the market's trend briefs (manyworlds.research.trends), market-scope findings, and every creator's
latest talent brief and conviction in the market. It writes:
  - the market thesis, audiences, what works, the gaps, and its conviction (markets.dossier, markets.conviction_*,
    a `convictions` row) that the strategist reads
  - learnings for the market's creators (markets.dossier.learnings: their talent managers read them, so a discovery
    by one creator reaches the others)
  - requests to the CEO (manyworlds.brain.proposals): a cohort of differentiated creators to launch, a budget for
    market experiments. Creators in one market are not competing for a fixed audience; the case is for many
    distinct personalities, not one winner.
  - predictions about the market's creators (manyworlds.brain.predictions)
Market leads recommend; the CEO decides.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from ... import queue
from ...worker.runner import NonRetryable
from .. import evidence, predictions, proposals
from ..agent import AgentContext, record_decision, register, remember
from .talent_manager import latest_brief

ROLE = "market_lead"
SYSTEM = """You lead one market at Manyworlds, an AI-native talent agency whose creators are original fictional people,
openly labeled as AI. You think like a division head at a top talent agency: you know this market's audiences, which
creator archetypes and aesthetics win, what is underserved, and where new personalities could grow. You argue for
investment when the upside is real and say plainly when it is not. Followers follow many creators in a market: the
opportunity is many distinct personalities, never clones. Learn from real creators' patterns; never copy a real
person or their content. Simulated data only shows the machinery works; it is never evidence of demand."""


class CohortConcept(BaseModel):
    concept: str = Field(description="a distinct personality and angle, in one or two sentences")
    archetype: str
    why_distinct: str


class MarketRequest(BaseModel):
    kind: Literal["cohort", "escalate"]
    subject: str = Field(description='as JSON: {"size": 5, "concepts": [...]} or {"ask": "..."}')
    argument: str


class MarketClaim(BaseModel):
    creator_handle: str
    claim: str
    metric: Literal["median_reach", "reach_growth"]
    comparator: Literal["above", "below"]
    threshold: float
    probability: float
    horizon_days: int


class MarketReview(BaseModel):
    thesis: str = Field(description="why this market is or isn't attractive for us now, in three sentences")
    audiences: list[str]
    working: list[str] = Field(description="archetypes, aesthetics, formats and hooks that win here")
    gaps: list[str] = Field(description="underserved audiences and angles")
    conviction_mean: float = Field(description="0-1: how likely we build at least one breakout creator here in 3 months")
    conviction_low: float
    conviction_high: float
    learnings: list[str] = Field(description="what one creator here found that the others should know")
    cohort: list[CohortConcept] = Field(description="new creator concepts worth launching, distinct from each other "
                                                    "and from the market's creators; may be empty")
    requests: list[MarketRequest] = Field(description="requests to the CEO; often none")
    predictions: list[MarketClaim] = Field(description="0-3 checkable claims about the market's creators")


def _cfg(policy) -> dict:
    return policy.get("brain", "market_lead", default={}) or {}


def _creators(conn, market_id) -> list[dict]:
    rows = conn.execute("SELECT c.id, c.handle, coalesce(s.stage, 'exploring') stage, s.tier FROM creators c "
                        "LEFT JOIN creator_states s ON s.creator_id = c.id WHERE c.market_id = %s "
                        "AND c.status <> 'retired' AND c.anchors->'simulation' IS NULL", (market_id,)).fetchall()
    out = []
    for r in rows:
        b = latest_brief(conn, r["id"])
        mode = evidence.mode_for(conn, r["id"])
        if mode == "sandbox":   # simulated outcomes are not evidence: a sandbox creator's brief never shapes a market
            continue
        out.append({"handle": r["handle"], "creator_id": str(r["id"]), "stage": r["stage"], "tier": r["tier"],
                    "mode": mode,
                    "brief": {k: (b["output"] or {}).get(k) for k in ("state", "focus", "conviction")} if b else None})
    return out


def _prompt(conn, policy, market: dict) -> str:
    briefs = [dict(r) for r in conn.execute("SELECT day, brief FROM trend_briefs WHERE market = %s ORDER BY day DESC "
                                            "LIMIT %s", (market["name"], int(_cfg(policy).get("trend_briefs", 3)))).fetchall()]
    findings = [{"finding": f["statement"], "grade": f["grade"]}
                for f in evidence.findings(conn, mode="real", scope="market", market_id=market["id"])]
    return "\n\n".join([
        f"The market: {market['name']}. Its current dossier: {json.dumps(market['dossier'] or {}, default=str)}",
        f"Its creators (briefs from their talent managers; mode sandbox means no real audience yet): "
        f"{json.dumps(_creators(conn, market['id']), default=str)}",
        f"Trend briefs (public posts in this niche: correlation, never proof): {json.dumps(briefs, default=str)}",
        f"Real market findings: {json.dumps(findings) if findings else 'none yet'}",
        f"Cohort limits the CEO works within: {policy.get('brain', 'strategist', 'max_active_creators')} active "
        f"creators, {policy.get('brain', 'strategist', 'launches_per_week')} launches a week now. Never touch: "
        f"{policy.get('scope', 'excluded_topics')}.",
        "Review this market as its lead: update the thesis and your conviction, pass learnings between its creators, "
        "and propose what the CEO should invest in, if anything."])


def _subject(raw) -> dict:
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        v = None
    return v if isinstance(v, dict) else {"text": str(raw or "")}


class MarketLead:
    name = ROLE

    def run(self, ctx: AgentContext) -> dict:
        p = ctx.job.payload or {}
        if not p.get("market_id"):
            raise NonRetryable("market_lead payload needs market_id")
        conn, policy = ctx.conn, ctx.policy
        market = conn.execute("SELECT * FROM markets WHERE id = %s", (p["market_id"],)).fetchone()
        if not market:
            raise NonRetryable("no such market")
        role = _cfg(policy).get("role", ROLE)

        def compute():
            out = ctx.llm.structured(role, system=SYSTEM, content=_prompt(conn, policy, market), schema=MarketReview,
                                     effort=_cfg(policy).get("effort", "medium"))
            return {"kind": "market_review", "inputs": {"market_id": str(market["id"])}, "output": out.model_dump(),
                    "rationale": out.thesis, "model": ctx.llm.model_for(role)}
        review = remember(ctx, "review", compute)
        clamp = lambda v: min(max(float(v or 0), 0.0), 1.0)  # noqa: E731
        dossier = {**(market["dossier"] or {}), **{k: review.get(k) for k in ("thesis", "audiences", "working", "gaps",
                                                                              "learnings")},
                   "cohort_concepts": review.get("cohort"), "reviewed_at": ctx.now.isoformat()}
        conn.execute("UPDATE markets SET dossier = %s, conviction_mean = %s, conviction_low = %s, conviction_high = %s, "
                     "updated_at = now() WHERE id = %s",
                     (Jsonb(dossier), clamp(review.get("conviction_mean")), clamp(review.get("conviction_low")),
                      clamp(review.get("conviction_high")), market["id"]))
        decision = record_decision(ctx, "market_thesis", step="thesis", inputs={"market_id": str(market["id"])},
                                   output={"thesis": review.get("thesis"), "conviction": review.get("conviction_mean")},
                                   rationale=review.get("thesis", ""), model=ctx.llm.model_for(role))
        if not conn.execute("SELECT 1 FROM convictions WHERE inputs->>'decision_id' = %s", (str(decision),)).fetchone():
            conn.execute("INSERT INTO convictions (subject, subject_id, mean, low, high, inputs) VALUES "
                         "('market', %s, %s, %s, %s, %s)",
                         (market["id"], clamp(review.get("conviction_mean")), clamp(review.get("conviction_low")),
                          clamp(review.get("conviction_high")), Jsonb({"decision_id": str(decision), "role": ROLE})))
        filed = []
        for r in review.get("requests") or []:
            subject = _subject(r.get("subject"))
            if r["kind"] == "cohort" and review.get("cohort"):
                subject.setdefault("concepts", review["cohort"])
            filed.append(proposals.file(conn, policy, kind=r["kind"], from_role=ROLE, market_id=market["id"],
                                        subject=subject, argument=r.get("argument") or "",
                                        evidence_={"conviction": {"mean": review.get("conviction_mean")}},
                                        job_id=ctx.job.id))
        handles = {c["handle"]: c["creator_id"] for c in _creators(conn, market["id"])}
        made = 0
        if not conn.execute("SELECT 1 FROM predictions WHERE decision_id = %s", (decision,)).fetchone():
            for c in review.get("predictions") or []:
                cid = handles.get(c.get("creator_handle"))
                made += cid is not None and predictions.record(
                    conn, role=ROLE, creator_id=cid, market_id=market["id"], decision_id=decision, claim=c["claim"],
                    metric=c["metric"], comparator=c["comparator"], threshold=c["threshold"],
                    probability=c["probability"], horizon_days=c["horizon_days"]) is not None
        return {"market": market["name"], "requests": filed, "predictions": made}


register(MarketLead())


def ensure_reviews(conn, policy, *, now: datetime | None = None) -> list:
    """A run for each market that warrants a lead (enough portfolio creators, or opened by the CEO), every
    `review_days`."""
    now = now or datetime.now(timezone.utc)
    cfg = _cfg(policy)
    if not cfg.get("enabled"):
        return []
    days = max(int(cfg.get("review_days", 7)), 1)
    period = int(now.timestamp() // (days * 86400))
    rows = conn.execute(
        "SELECT m.id FROM markets m WHERE m.dossier->>'lead' = 'open' OR (SELECT count(*) FROM creators c "
        "WHERE c.market_id = m.id AND c.status = 'active' AND c.anchors->'simulation' IS NULL) >= %s",
        (int(cfg.get("min_creators", 3)),)).fetchall()
    return [queue.enqueue(conn, "brain.market_lead", {"market_id": str(r["id"])},
                          idempotency_key=f"market_lead:{r['id']}:{days}d:{period}") for r in rows]
