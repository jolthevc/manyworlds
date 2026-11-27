"""The strategist: the daily portfolio cycle (Brain M8, spec 4.6).

It sees the portfolio in numbers (manyworlds.brain.portfolio: stage, real posts, cost, conviction, the
continue/stop values, budget, launch pacing) and the best unlaunched theses, and proposes stage moves and
launches with reasons. Code then checks each proposal against policy `brain.strategist` and applies only what
passes:

  stage moves  to a stage in `stages`; a move into `downward_stages` needs the evidence window
               (`min_posts_before_downward` real posts and `min_days_before_downward` days live); retirement
               also needs `min_adjustments_before_retire` and the expected-value case; moves in
               `approval_moves` wait for the owner
  launches     a proposed (unused) thesis; within `max_active_creators`, `launches_per_week` and the month's
               remaining budget (`est_launch_usd` each); while `launch_requires_approval` the owner approves,
               otherwise the strategist approves within its authority and manyworlds.brain.launches takes over

The whole run is one decision with the numbers it saw, what it proposed, and what was applied or refused.
It reads real evidence only. Its calls come with checkable predictions about the creators they concern
(manyworlds.brain.predictions, role `strategist`, at most `max_predictions` a run, linked to the run's decision), so
`mw brain scorecard` scores the CEO as it scores the talent managers; it sees its own scorecard each run.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from ...llm import as_task
from ...worker.runner import NonRetryable
from .. import portfolio, predictions, proposals
from ..agent import AgentContext, register, remember

SYSTEM = """You are the CEO of Manyworlds, a talent, media and consumer business built on openly AI-generated
lifestyle personalities, not merely a content studio. Think like a talent agency executive, commercial strategist
and portfolio manager: where is the largest demonstrated demand, which personalities could build the most valuable
audiences with the broadest commercial adjacencies, what each costs to produce reliably, and where the long-term
value is captured (sponsorship, affiliate, licensing, then owned products distributed across our own network).
Commercially informed casting, growth-led execution, opportunistic monetisation: commercial intelligence decides
which personalities to develop; once launched, the job is to make each one extraordinarily successful (reach,
follower growth, shares, saves), because a very large engaged audience creates the commercial opportunities.
Never trade growth for early monetisation. Allocate attention, budget and formats by expected return per dollar
(growth first, then brand and commercial value), not by habit. Repeating a commercially proven archetype with distinct personalities is
good strategy; being different only for its own sake is not. Be patient with creators: judge them on evidence
windows, not early noise, and prefer adjusting a strategy to giving up. Every number you see is computed from real
outcomes only (a creator still in the sandbox has none yet). Propose only what the numbers and theses support,
with a reason for each, including the commercial case; code will check your proposals against budget, pacing and
evidence rules."""


class StageMove(BaseModel):
    creator_id: str
    to: str
    reason: str


class LaunchProposal(BaseModel):
    thesis_id: str
    cohort: str = Field(description="a short label shared by the creators launched together")
    reason: str


class RequestDecision(BaseModel):
    proposal_id: str
    accept: bool
    reason: str = Field(description="why, against the evidence and the budget; a rejection says what would change it")


class Claim(BaseModel):
    creator_id: str = Field(description="the creator the claim is about")
    claim: str = Field(description="what you expect, in words")
    metric: Literal["median_reach", "reach_growth"]
    comparator: Literal["above", "below"]
    threshold: float
    probability: float = Field(description="0-1: how sure you are")
    horizon_days: int


class PortfolioPlan(BaseModel):
    stage_moves: list[StageMove]
    launches: list[LaunchProposal]
    requests: list[RequestDecision] = Field(default_factory=list, description="a decision on every open request addressed to you")
    predictions: list[Claim] = Field(default_factory=list, description=(
        "checkable claims behind your calls (a stage move, a tier, keeping a creator): what her reach will do over "
        "the horizon, with your probability; scored later against real outcomes. None for a creator with no "
        "published posts"))
    notes: str = Field(description="one or two sentences on the portfolio and why these choices")


def _cfg(policy) -> dict:
    return policy.get("brain", "strategist", default={}) or {}


def _strategy_text(ctx: AgentContext) -> str:
    """The agency's thesis, the portfolio map (coverage, overlaps, gaps; code), what production does reliably (the
    capability card) and the best commercial categories (the registry): what a launch should fill (owner 2026-11-25)."""
    from .. import capability_card, commerce, portfolio_map
    from ...creator import dossier
    shown = int(ctx.policy.get("strategy", "commercial", "shown", default=8))
    nl = "\n"
    return (f"{dossier.strategy_text(ctx.policy)}{nl}"
            f"Portfolio map (archetypes x audience cells x commercial categories; launch into the gaps, never a copy of "
            f"a creator we have):{nl}{portfolio_map.text(portfolio_map.build(ctx.conn, ctx.policy))}{nl}"
            f"What production makes reliably and cheaply now (the capability card):{nl}"
            f"{capability_card.brief(capability_card.card(ctx.conn, ctx.policy))}{nl}"
            f"Commercial categories (registry, best first; estimates until researched):{nl}"
            f"{commerce.brief(commerce.categories(ctx.conn, ctx.policy), n=shown)}"
            + _desk(ctx))


def _desk(ctx: AgentContext) -> str:
    """Research desk findings on creator business models, archetype demand and commerce (manyworlds.research.desk)."""
    from ...research import desk
    lines = "\n".join(x for x in (desk.brief(ctx.conn, ctx.policy, use="casting", tags=["strategy", "casting"]),
                                   desk.brief(ctx.conn, ctx.policy, use="commerce")) if x)
    return (f"\nResearch desk (current sources, evidence mix shown; context, not our own results):\n{lines}"
            if lines else "")


def _apply_moves(ctx: AgentContext, snap: dict, moves: list[StageMove]) -> dict:
    cfg, conn = _cfg(ctx.policy), ctx.conn
    stages, down = list(cfg.get("stages") or []), set(cfg.get("downward_stages") or [])
    by = {c["creator_id"]: c for c in snap["creators"]}
    out = {"applied": [], "pending": [], "refused": []}
    for m in moves:
        c, why = by.get(m.creator_id), None
        if not c:
            why = "unknown or retired creator"
        elif m.to not in stages:
            why = f"unknown stage {m.to!r}"
        elif m.to == c["stage"]:
            why = "already in that stage"
        elif m.to in down and (c["real_posts"] < int(cfg.get("min_posts_before_downward", 30))
                               or c["days_live"] < int(cfg.get("min_days_before_downward", 21))):
            why = (f"inside the evidence window ({c['real_posts']} real posts, {c['days_live']} days live; "
                   f"needs {cfg.get('min_posts_before_downward')} and {cfg.get('min_days_before_downward')})")
        elif m.to == "retired" and c["adjustments"] < int(cfg.get("min_adjustments_before_retire", 2)):
            why = f"only {c['adjustments']} strategy adjustments tried"
        elif m.to == "retired" and not c["value"]["retire_case"]:
            why = (f"continuing is not worth less in expected value ({c['value']['ev_continue']} vs alternative "
                   f"{c['value']['ev_alternative']})")
        if why:
            out["refused"].append({**m.model_dump(), "why": why})
            continue
        if m.to in set(cfg.get("approval_moves") or []):
            if not conn.execute("SELECT 1 FROM approvals WHERE kind = 'stage_move' AND status = 'pending' "
                                "AND subject->>'creator_id' = %s", (m.creator_id,)).fetchone():   # one at a time
                conn.execute("INSERT INTO approvals (kind, subject, reason, requested_by) VALUES ('stage_move', %s, "
                             "%s, 'strategist')", (Jsonb({**m.model_dump(), "from": c["stage"]}), m.reason))
            out["pending"].append(m.model_dump())
            continue
        portfolio.apply_stage_move(conn, m.creator_id, from_stage=c["stage"], to=m.to, reason=m.reason, downward=down)
        out["applied"].append(m.model_dump())
    return out


def _apply_launches(ctx: AgentContext, snap: dict, launches: list[LaunchProposal], decision_hint: str) -> dict:
    cfg, conn = _cfg(ctx.policy), ctx.conn
    out = {"requested": 0, "approved": 0, "refused": []}
    active = sum(1 for c in snap["creators"] if c["status"] in ("draft", "active"))
    taken = active + snap["launches"]["running"] + snap["launches"]["pending_approvals"]
    this_week = snap["launches"]["last_7_days"]
    remaining = snap["budget"]["remaining_usd"]
    need = ctx.policy.get("brain", "strategist", "est_launch_usd", default=3.0)
    for p in launches:
        t = conn.execute("SELECT id, status FROM theses WHERE id::text = %s", (p.thesis_id,)).fetchone()
        why = None
        if not t:
            why = "unknown thesis"
        elif t["status"] != "proposed":
            why = f"thesis is {t['status']}, not available"
        elif taken + 1 > int(cfg.get("max_active_creators", 5)):
            why = f"at the cap of {cfg.get('max_active_creators')} active creators (incl. launching)"
        elif this_week + 1 > int(cfg.get("launches_per_week", 2)):
            why = f"over {cfg.get('launches_per_week')} launches per week"
        elif remaining < float(need):
            why = f"month budget remaining ${remaining} < ${need} per launch"
        if why:
            out["refused"].append({**p.model_dump(), "why": why})
            continue
        needs_owner = bool(cfg.get("launch_requires_approval", True))
        conn.execute(
            "INSERT INTO approvals (kind, subject, reason, requested_by, status, decided_by, decided_at) "
            "VALUES ('launch', %s, %s, 'strategist', %s, %s, %s)",
            (Jsonb({"thesis_id": p.thesis_id, "cohort": p.cohort, "decision": decision_hint}), p.reason,
             "pending" if needs_owner else "approved", None if needs_owner else "strategist",
             None if needs_owner else datetime.now(timezone.utc)))
        conn.execute("UPDATE theses SET status = 'chosen' WHERE id = %s", (t["id"],))
        out["requested" if needs_owner else "approved"] += 1
        taken, this_week, remaining = taken + 1, this_week + 1, remaining - float(need)
    return out


class Strategist:
    name = "strategist"

    def run(self, ctx: AgentContext) -> dict:
        p = ctx.job.payload or {}
        if not p.get("day"):
            raise NonRetryable("strategist payload needs a day")
        cfg = _cfg(ctx.policy)

        def compute() -> dict:
            snap = portfolio.snapshot(ctx.conn, ctx.policy)
            theses = [{"thesis_id": str(t["id"]), "summary": t["summary"], **(t["body"] or {})}
                      for t in ctx.conn.execute(
                          "SELECT id, summary, body FROM theses WHERE status = 'proposed' "
                          "ORDER BY score DESC NULLS LAST, created_at LIMIT %s",
                          (int(cfg.get("thesis_candidates", 8)),)).fetchall()]
            open_requests = proposals.open_for(ctx.conn, "strategist",
                                               limit=int(cfg.get("max_requests_per_run", 20)))
            markets = [dict(m) for m in ctx.conn.execute(
                "SELECT id::text, name, conviction_mean, conviction_low, conviction_high, dossier->'thesis' thesis "
                "FROM markets WHERE dossier ? 'thesis' ORDER BY conviction_mean DESC NULLS LAST LIMIT 10").fetchall()]
            card = predictions.scorecard(ctx.conn, role=ROLE)
            prompt = (f"{_strategy_text(ctx)}\n"
                      f"Portfolio (numbers from real outcomes only): {json.dumps(snap, default=str)}\n"
                      f"Your own scored predictions so far (Brier: lower is better, 0.25 is a coin flip): "
                      f"{json.dumps(card, default=str) if card else 'none scored yet'}\n"
                      f"Market theses from the market leads: {json.dumps(markets, default=str) if markets else 'none yet'}\n"
                      f"Requests addressed to you (decide every one; a talent manager's conviction is an argument to "
                      f"weigh, not a fact; requests labelled SIMULATED rest on sandbox data, which shows nothing about "
                      f"real demand): {json.dumps(open_requests, default=str) if open_requests else 'none'}\n"
                      f"Unlaunched theses: {json.dumps(theses, default=str)}\n"
                      f"Stages: {cfg.get('stages')}. Moves into {cfg.get('downward_stages')} need an evidence window "
                      f"({cfg.get('min_posts_before_downward')} real posts, {cfg.get('min_days_before_downward')} days live). "
                      f"Launch limits: {cfg.get('max_active_creators')} active creators, {cfg.get('launches_per_week')} "
                      f"launches a week, about ${cfg.get('est_launch_usd')} each. Propose stage moves and launches "
                      f"(by thesis_id) with reasons; propose nothing where nothing is warranted. Back your calls "
                      f"with up to {cfg.get('max_predictions')} checkable predictions about the creators concerned "
                      f"(median_reach or reach_growth, above or below a threshold, with a probability).")
            with as_task(ctx.llm, "strategist_plan"):
                plan = ctx.llm.structured("brain", system=SYSTEM, content=prompt, schema=PortfolioPlan,
                                          effort=cfg.get("effort", "medium"))
            moves = _apply_moves(ctx, snap, plan.stage_moves)
            decided = _decide_requests(ctx, open_requests, plan.requests)
            launches = _apply_launches(ctx, snap, plan.launches, f"strategist:{p['day']}")
            return {"kind": "portfolio", "rationale": plan.notes, "model": ctx.llm.model_for("brain"),
                    "inputs": {"creators": snap["creators"], "budget": snap["budget"], "launches": snap["launches"],
                               "theses": [t["thesis_id"] for t in theses]},
                    "output": {"proposed": plan.model_dump(), "stage_moves": moves, "launches": launches,
                               "requests": decided}}

        out = remember(ctx, "portfolio", compute)
        return {**out, "predictions_recorded": _record_predictions(ctx, out)}


ROLE = "strategist"


def _record_predictions(ctx: AgentContext, out: dict) -> int:
    """The run's claims (at most brain.strategist.max_predictions, each about a creator the run saw) into the
    predictions ledger under role strategist, linked to the run's decision. A rerun of the job records none twice."""
    conn, cap = ctx.conn, int(_cfg(ctx.policy).get("max_predictions", 5))
    decision = conn.execute("SELECT id, inputs FROM decisions WHERE job_id = %s AND step = 'portfolio'",
                            (ctx.job.id,)).fetchone()
    if decision is None or conn.execute("SELECT 1 FROM predictions WHERE decision_id = %s",
                                        (decision["id"],)).fetchone():
        return 0
    seen = {str(c.get("creator_id")) for c in (decision["inputs"] or {}).get("creators") or []}
    made = 0
    for c in ((out.get("proposed") or {}).get("predictions") or [])[:cap]:
        if str(c.get("creator_id")) not in seen:
            continue   # a claim about a creator it never saw (or made up) is not recorded
        made += predictions.record(conn, role=ROLE, creator_id=c["creator_id"], decision_id=decision["id"],
                                   claim=c["claim"], metric=c["metric"], comparator=c["comparator"],
                                   threshold=c["threshold"], probability=c["probability"],
                                   horizon_days=c["horizon_days"]) is not None
    return made


def _decide_requests(ctx: AgentContext, open_requests: list[dict], decisions: list) -> list[dict]:
    """Each decision applied through manyworlds.brain.proposals (rights and money checked there); requests it left
    out stay open for the next run, and expire when due. At most `tier_changes_per_day` tier changes a run."""
    known = {r["id"]: r for r in open_requests}
    cap = int((ctx.policy.get("brain", "rights", "strategist", default={}) or {}).get("tier_changes_per_day", 5))
    out, tiers = [], 0
    for d in decisions or []:
        r = known.get(d.proposal_id)
        if r is None:
            out.append({"proposal_id": d.proposal_id, "refused": "not an open request addressed to the strategist"})
            continue
        if d.accept and r["kind"] in ("tier_change", "production_upgrade"):
            tiers += 1
            if tiers > cap:
                out.append({"proposal_id": d.proposal_id, "refused": f"over {cap} tier changes a day; left open"})
                continue
        out.append({"proposal_id": d.proposal_id,
                    **proposals.decide(ctx.conn, ctx.policy, d.proposal_id, decided_by="strategist", accept=d.accept,
                                       reason=d.reason)})
    return out


register(Strategist())
