"""The idea reviser: one cheap rewrite of an idea that scored just under a feed-QA floor (manyworlds.brain.revise).

Asked for by scheduling (manyworlds.brain.ideas.schedule_ready), never by an agent's own choice. It reads the plan,
the evaluator's reasons for the weak dimensions and the creator's voice rules, and returns the same idea schema as
the showrunner (PlannedIdea), improved on those dimensions. The rewrite becomes a new idea in the same batch (format,
day, platform and library names kept from the original), keeps the original's created_at (the batch's wait counts
from planning), carries the original and the reason on its body, and is sent to the evaluator like any idea. The
original is kept with status `revised`. One call, remembered per job: a rerun never pays twice.
"""

from __future__ import annotations

import json
import uuid
from datetime import timedelta

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from ...creator import profile as profile_mod
from ...worker.runner import NonRetryable
from .. import planner, revise
from ..agent import AgentContext, commission, record_decision, register, remember
from . import evaluator
from .showrunner import PlannedIdea

AGENT = "idea_reviser"

SYSTEM = planner.SYSTEM + """

You are the creator's showrunner, revising one of your own post ideas. An independent critic scored it just
under the bar on the dimensions named below and said why. Rewrite the same idea so it clears the bar: keep its
moment, format and day, fix what the critic named (usually the caption, the script or how the scenes are
written), and change nothing that already works. It must still sound exactly like the creator. Return the whole
idea, every field filled as the original was.""" + planner.DIRECTION

# Fields the rewrite may change; everything else on the body (day, platform, library names, flags) is the original's.
_CONTENT = ("scenes", "script", "on_screen_text", "hashtags", "canon", "canon_key", "pillar", "tone", "series",
            "caption")
_ORIGINAL = ("scenes", "script", "on_screen_text", "caption", "hashtags", "canon", "pillar", "tone", "series")


class IdeaRevision(BaseModel):
    changes: str = Field(description="what you changed and why it answers the critic, in one or two sentences")
    idea: PlannedIdea


def _critique(conn, idea_id, dims: list[str], floors: dict) -> dict:
    """The latest volume-evaluator scores, and its reasons for the weak dimensions."""
    ev = conn.execute("SELECT scores, rationale FROM evaluations WHERE idea_id = %s AND role = 'evaluator' "
                      "ORDER BY created_at DESC LIMIT 1", (idea_id,)).fetchone()
    if ev is None:
        raise NonRetryable(f"idea_reviser: idea {idea_id} has no evaluation to revise from")
    scores, why = ev["scores"] or {}, ev["rationale"] or {}
    return {"weak": {d: {"score": scores.get(d), "floor": floors.get(d), "why": why.get(d) or []} for d in dims},
            "scores": scores, "notes": why.get("_notes") or []}


def _prompt(conn, policy, idea, critique: dict) -> str:
    plan = {k: v for k, v in (idea["body"] or {}).items() if k not in revise.BOOKKEEPING}
    _, who = evaluator.creator_summary(conn, idea["creator_id"])
    scope = policy.get("scope", default={}) or {}
    return (f"{who}\n"
            f"THE IDEA ({idea['format']}): {idea['concept']}\n"
            f"PLAN: {json.dumps(plan, ensure_ascii=False, default=str)}\n"
            f"ATTRIBUTES: {json.dumps(idea['attributes'] or {}, ensure_ascii=False)}\n"
            f"THE CRITIC (1-10 per dimension; the floor is the bar it must clear): "
            f"{json.dumps(critique['weak'], ensure_ascii=False)}\n"
            f"All its scores: {json.dumps(critique['scores'])}; its overall notes: {json.dumps(critique['notes'])}\n"
            f"Keep the format ({idea['format']}), day ({(idea['body'] or {}).get('day', 0)}) and platform "
            f"({(idea['body'] or {}).get('platform', 'instagram')}). Never touch: {scope.get('excluded_topics')}. "
            f"No sponsored, product or review posts, no selling.")


class IdeaReviser:
    name = AGENT

    def run(self, ctx: AgentContext) -> dict:
        p = ctx.job.payload or {}
        try:
            idea_id = str(uuid.UUID(str(p.get("idea_id"))))
        except ValueError:
            raise NonRetryable(f"idea_reviser: idea_id {p.get('idea_id')!r} is not a uuid") from None
        conn, policy = ctx.conn, ctx.policy
        done = conn.execute("SELECT output FROM decisions WHERE job_id = %s AND step = 'apply'",
                            (ctx.job.id,)).fetchone()
        if done:   # a rerun returns what this job already did
            return done["output"]
        idea = conn.execute("SELECT id, creator_id, batch, concept, format, body, attributes, status, created_at "
                            "FROM ideas WHERE id = %s", (idea_id,)).fetchone()
        if idea is None:
            raise NonRetryable(f"idea_reviser: no idea {idea_id}")
        if idea["status"] != "proposed":   # scheduled or dropped meanwhile (the batch stopped waiting)
            return {"revised_idea_id": None, "skipped": f"idea is {idea['status']}"}
        floors = policy.get("brain", "qa", "min_scores", default={}) or {}
        dims = [d for d in (p.get("dims") or []) if d in floors] or list(floors)
        role = str(policy.get("brain", "revise", "role", default="idea_revise"))

        def compute() -> dict:
            critique = _critique(conn, idea_id, dims, floors)
            out = ctx.llm.structured(role, system=SYSTEM + planner.voice_rules(policy),
                                     content=_prompt(conn, policy, idea, critique), schema=IdeaRevision,
                                     effort=str(policy.get("brain", "revise", "effort", default="low")),
                                     creator_id=idea["creator_id"])
            revised, changes = getattr(out, "idea", None), str(getattr(out, "changes", "") or "")
            return {"kind": "idea_revision", "creator_id": idea["creator_id"], "model": ctx.llm.model_for(role),
                    "inputs": {"idea_id": idea_id, "dims": dims, "scores": critique["scores"],
                               "reason": p.get("reason")},
                    "output": {"idea": revised.model_dump(mode="json") if revised is not None else None,
                               "changes": changes},
                    "rationale": changes or "rewrote a near-floor idea"}
        answer = remember(ctx, "rewrite", compute)
        return self._apply(ctx, idea, answer, dims)

    def _apply(self, ctx: AgentContext, idea, answer: dict, dims: list[str]) -> dict:
        conn, policy = ctx.conn, ctx.policy
        body = idea["body"] or {}
        rejected = None
        q = None
        if not answer.get("idea"):
            rejected = "the rewrite returned no idea"
        else:
            try:
                q = PlannedIdea.model_validate({**answer["idea"], "format": idea["format"],
                                                "day": int(body.get("day", 0)),
                                                "platform": body.get("platform", "instagram")})
            except ValueError as e:
                rejected = f"the rewrite is not a valid idea: {str(e)[:200]}"
        if q is not None:
            days = int(policy.get("brain", "showrunner", "days", default=3))
            cur = profile_mod.current(conn, idea["creator_id"])
            prof = cur[1] if cur else None
            platforms = list((policy.get("scope", default={}) or {}).get("platforms") or ["instagram"])
            dials = (prof or {}).get("dials") or policy.get("creators", "dials_default", default={}) or {}
            q, why = planner.validate(policy, q, platforms=platforms, days=days, prof=prof,
                                      hashtags_max=dials.get("hashtags_max") or {})
            if q is None:
                rejected = f"the rewrite failed the plan checks: {why}"
        max_wait = timedelta(hours=float(policy.get("brain", "showrunner", "max_wait_hours", default=6)))
        with conn.transaction():
            # The original, locked: scheduling drops or schedules it under this lock, never halfway through a write.
            cur_idea = conn.execute("SELECT status, created_at FROM ideas WHERE id = %s FOR UPDATE",
                                    (idea["id"],)).fetchone()
            revised_id = None
            if cur_idea["status"] != "proposed":
                rejected = rejected or f"idea is {cur_idea['status']}"
            elif ctx.now - cur_idea["created_at"] >= max_wait:
                rejected = rejected or "the batch is past brain.showrunner.max_wait_hours"
            if rejected is None:
                revised_id = self._insert(ctx, idea, q, answer, dims)
            output = {"revised_idea_id": str(revised_id) if revised_id else None, "rejected": rejected}
            record_decision(ctx, "idea_revision_applied", step="apply", creator_id=idea["creator_id"],
                            inputs={"idea_id": str(idea["id"]), "dims": dims}, output=output,
                            rationale=answer.get("changes") or (rejected or ""))
        return output

    def _insert(self, ctx: AgentContext, idea, q: PlannedIdea, answer: dict, dims: list[str]):
        conn = ctx.conn
        body = dict(idea["body"] or {})
        original = {k: body.get(k) for k in _ORIGINAL if k in body}
        new_body = {k: v for k, v in body.items() if k not in revise.BOOKKEEPING and k not in ("shots",)}
        new_body.update({k: getattr(q, k) for k in _CONTENT})
        if q.slides:
            new_body["slides"] = [s.model_dump(mode="json") for s in q.slides]
        new_body.update({"revision": int(body.get("revision") or 0) + 1, "revision_of": str(idea["id"]),
                         "original": {**original, "concept": idea["concept"]},
                         "revise_reason": body.get("revise_reason") or f"near floor on {', '.join(dims)}",
                         "revise_dims": dims, "reasoning": body.get("reasoning", "")})
        attributes = {**(idea["attributes"] or {}), **{k: v for k, v in (q.attributes or {}).items() if v}}
        concept = " | ".join(q.scenes) + (f" | script: {q.script}" if q.script else "") + f" | caption: {q.caption}"
        new_id = conn.execute(
            "INSERT INTO ideas (creator_id, batch, concept, format, body, attributes, status, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, 'proposed', %s) RETURNING id",
            (idea["creator_id"], idea["batch"], f"{idea['format']}: {concept}", idea["format"], Jsonb(new_body),
             Jsonb(attributes), idea["created_at"])).fetchone()["id"]
        conn.execute("UPDATE ideas SET status = 'revised', updated_at = now(), body = body || %s WHERE id = %s",
                     (Jsonb({"revised_by": str(new_id)}), idea["id"]))
        commission(ctx, "evaluator", {"idea_id": str(new_id)}, reason="score a revised idea before scheduling",
                   key=f"eval:{new_id}")
        return new_id


register(IdeaReviser())
