"""The analyst: turns a creator's outcomes into evidence-qualified hypotheses and guidance.

The numbers come from code (manyworlds.brain.analysis): normalised outcomes, effects of rubric
dimensions, formats and attributes with corrected intervals, breakouts. The LLM sees only that
table (never captions or raw posts it could pattern-match on) and writes hypotheses that cite the
effects supporting them, plus guidance for the next batch that honours the uncertainty. With too
little data there is no LLM call and no finding: the guidance is to keep exploring.

Findings are recorded through manyworlds.brain.evidence (scope `creator`, with the posts and, for
sandbox creators, the simulated source), replacing the creator's previous ones. They are written so
the same evidence can later be pooled for clusters, markets and the strategist (M8/M9). Sandbox
findings are SIMULATED: an engineering test of the loop, never evidence of demand.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ...worker.runner import NonRetryable
from .. import analysis, evidence
from ..agent import AgentContext, commission, record_decision, register

SYSTEM = """You are the analyst for an AI creator studio. You are given statistics about how one creator's
posts performed: effects of creative characteristics on reach (log-lift over the creator's own
baseline), each with an interval, the number of posts, and a grade (tentative = the interval clears a
correction for testing many things at once; speculative = it doesn't). Write hypotheses only from these
numbers. Say what the evidence supports and how strongly; never claim more than the grade allows; prefer
"may" for speculative effects. Then advise the next batch: what to do more of, less of, and what to
explore, keeping a healthy share of exploration while evidence is thin. Never recommend turning the feed
into a formula: ordinary moments stay part of a creator's life."""


class Hypothesis(BaseModel):
    statement: str = Field(description="one sentence, evidence-qualified")
    direction: Literal["positive", "negative"] = Field(description="the sign of the effect the statement claims")
    supports: list[str] = Field(description="effect ids from the table, as kind:key; the FIRST is what the statement is about")
    how_to_confirm: str = Field(description="what the next batches could do to test it")


class Guidance(BaseModel):
    more_of: list[str]
    less_of: list[str]
    explore: list[str]
    notes: str = Field(description="one or two sentences on confidence and what's still unknown")


class Analysis(BaseModel):
    hypotheses: list[Hypothesis]
    guidance: Guidance


def _table(result: dict) -> list[dict]:
    return [{"id": f"{e['kind']}:{e['key']}", "effect": e["effect"], "interval": [e["low"], e["high"]],
             "n": e["n"], "grade": e["grade"]} for e in result["effects"]]


def analyse(ctx: AgentContext, creator_id) -> dict:
    mode = evidence.mode_for(ctx.conn, creator_id)
    rows = analysis.normalise(analysis.dataset(ctx.conn, creator_id, mode=mode, policy=ctx.policy), ctx.policy)
    result = analysis.effects(rows, ctx.policy, seed=0)
    label = "SIMULATED (sandbox engineering test, not real demand)" if mode == "sandbox" else "REAL Instagram data"
    if result["insufficient"]:
        need = ctx.policy.get("brain", "analysis", "min_n_any", default=8)
        guidance = {"more_of": [], "less_of": [], "explore": ["keep the mix of formats, settings and moods broad"],
                    "notes": f"Insufficient evidence ({result['n']} posts with outcomes, {need} needed): keep the mix broad and explore; "
                             f"no pattern is reliable yet."}
        return {"mode": mode, "n": result["n"], "insufficient": True, "effects": [], "breakouts": result["breakouts"],
                "hypotheses": [], "guidance": guidance, "finding_ids": [], "rationale": guidance["notes"]}
    by_id = {f"{e['kind']}:{e['key']}": e for e in result["effects"]}
    content = (f"DATA: {label}. Posts with outcomes: {result['n']}; effects tested: {result.get('tested')}.\n"
               f"Effects (strongest first): {_table(result)[:40]}\n"
               f"Breakouts (reach far above baseline): {result['breakouts']}")
    out = ctx.llm.structured("brain", system=SYSTEM, content=content, schema=Analysis,
                             effort=ctx.policy.get("brain", "analyst", "effort", default="medium"),
                             creator_id=creator_id)
    evidence_kind = "simulated" if mode == "sandbox" else "observational"
    source = [{"creator_id": str(creator_id)}] + ([{"table": "simulated_metrics"}] if mode == "sandbox" else [])
    finding_ids = []
    with ctx.conn.transaction():
        evidence.supersede(ctx.conn, creator_id=creator_id)
        for h in out.hypotheses:
            effects = [by_id[s] for s in h.supports if s in by_id]
            if not effects:
                continue   # a hypothesis citing nothing in the table isn't a finding
            if any((e["effect"] > 0) != (h.direction == "positive") for e in effects):
                continue   # a statement that contradicts the numbers it cites isn't a finding
            # Graded by its subject (the first effect it cites), never by the strongest citation: a story-only
            # claim citing a general effect must not inherit that effect's grade (live run, 2026-11-14).
            best = by_id.get(h.supports[0]) if h.supports else None
            if best is None:
                continue
            fid = evidence.record_finding(
                ctx.conn, scope="creator", creator_id=creator_id, statement=h.statement, evidence=evidence_kind,
                grade=best["grade"], effect=best["effect"], interval=(best["low"], best["high"]), n=best["n"],
                sources=source + [{"effects": h.supports, "how_to_confirm": h.how_to_confirm,
                                   "posts": [r["post_id"] for r in rows if r.get("lift") is not None]}])
            finding_ids.append(str(fid))
    return {"mode": mode, "n": result["n"], "insufficient": False, "effects": result["effects"],
            "breakouts": result["breakouts"], "hypotheses": [h.model_dump() for h in out.hypotheses],
            "guidance": out.guidance.model_dump(), "finding_ids": finding_ids, "rationale": out.guidance.notes}


def latest(conn, creator_id) -> dict | None:
    """The creator's most recent analysis made in their current evidence mode: a creator who left the
    sandbox never plans on simulated guidance."""
    return conn.execute("SELECT id, output FROM decisions WHERE kind = 'analysis' AND creator_id = %s "
                        "AND output->>'mode' = %s ORDER BY created_at DESC LIMIT 1",
                        (creator_id, evidence.mode_for(conn, creator_id))).fetchone()


class Analyst:
    name = "analyst"

    def run(self, ctx: AgentContext) -> dict:
        p = ctx.job.payload or {}
        if not p.get("creator_id"):
            raise NonRetryable("analyst payload needs creator_id")
        done = ctx.conn.execute("SELECT output FROM decisions WHERE job_id = %s AND step = 'analyse'",
                                (ctx.job.id,)).fetchone()
        output = done["output"] if done else analyse(ctx, p["creator_id"])
        if not done:
            record_decision(ctx, "analysis", step="analyse", creator_id=p["creator_id"],
                            model=None if output["insufficient"] else ctx.llm.model_for("brain"),
                            inputs={"mode": output["mode"], "n": output["n"]},
                            output={k: v for k, v in output.items() if k != "rationale"}, rationale=output["rationale"])
        if p.get("then_plan"):   # the next period is planned with this analysis in hand
            commission(ctx, "showrunner", {"creator_id": p["creator_id"], "start": p["then_plan"]},
                       reason="plan the next period after analysis", key=f"plan:{p['then_plan']}")
        return {k: output[k] for k in ("n", "insufficient", "finding_ids", "guidance")}


register(Analyst())
