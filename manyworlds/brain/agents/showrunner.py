"""The showrunner: plans a creator's next few days as ideas, the way the person would actually post.

It knows the creator (profile and memory), sees how its recent ideas scored and what has been
learned about this creator (findings, read in the creator's evidence mode), and spends a share of
each batch exploring things the creator hasn't tried (policy `brain.exploration_share`, which the
Brain may tune). Ideas carry attributes for learning (setting, outfit, activity, mood, trend) but
no strategic category; ordinary moments are welcome. Its reasoning is stored on each idea and never
shown to the evaluator. Every idea is sent to the evaluator; only evaluated ideas become posts
(manyworlds.brain.ideas.schedule_ready).
"""

from __future__ import annotations

import json
from datetime import date, timedelta

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from ...creator import profile as profile_mod
from ...worker.runner import NonRetryable
from .. import evidence, memory, planner
from ..agent import AgentContext, commission, record_decision, register
from . import analyst

SYSTEM = planner.SYSTEM + """

You are the creator's showrunner. Plan the next few days as a natural feed: everyday moments,
outfits, food, places, small opinions, the occasional trend or arc, and Stories for throwaway
moments. Not every post needs a hook or a purpose; a lovely ordinary moment is a good post. Keep the
creator's life coherent with what is known about them, and learn from how recent ideas scored and
from any findings, without turning the feed into a formula."""


class PlannedIdea(planner.PlannedPost):
    attributes: dict[str, str] = Field(default_factory=dict,
                                       description="setting, outfit, activity, mood, trend (yes/no): plain words")
    reasoning: str = Field(default="", description="why this post now, in one or two sentences")
    exploratory: bool = Field(default=False, description="true if this tries something the creator hasn't done")
    canon_key: str = Field(default="", description="if `canon` replaces a single fact (home, partner, job, pet), "
                                                   "that fact's name; otherwise empty")


class IdeaBatch(BaseModel):
    rationale: str
    ideas: list[PlannedIdea]


def _recent_with_scores(conn, creator_id, n: int) -> list[dict]:
    rows = conn.execute(
        """SELECT p.format, p.caption, p.body->'scenes' AS scenes, p.body->>'canon' AS canon, e.scores
           FROM posts p LEFT JOIN ideas i ON i.post_id = p.id
           LEFT JOIN LATERAL (SELECT scores FROM evaluations WHERE idea_id = i.id AND role = 'evaluator'
                              ORDER BY created_at DESC LIMIT 1) e ON true
           WHERE p.creator_id = %s AND p.status <> 'dropped'
           ORDER BY coalesce(p.scheduled_for, p.created_at) DESC LIMIT %s""", (creator_id, n)).fetchall()
    return [{k: v for k, v in dict(r).items() if v} for r in rows]


def _prompt(ctx: AgentContext, creator_id, start: date, days: int) -> tuple[str, dict]:
    conn, policy = ctx.conn, ctx.policy
    cfg = policy.get("brain", "showrunner", default={}) or {}
    memory.seed_from_profile(conn, creator_id)   # idempotent: new creators and new profile versions reach memory
    c = conn.execute("SELECT handle, bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
    known = [f"{m['kind']}: {m['text']}" for m in memory.facts(conn, creator_id, limit=int(cfg.get("memory_facts", 60)))]
    day_names = {i: (start + timedelta(days=i)).strftime("%A %d %B") for i in range(days)}
    cur = profile_mod.current(conn, creator_id)
    version, prof = cur if cur else (None, None)
    dials = (prof or {}).get("dials") or policy.get("creators", "dials_default", default={})
    counts = planner._counts(policy, dials, days)
    formats = planner._formats(policy)
    who = {k: prof[k] for k in planner.ANALYSIS_FIELDS if k in prof} if prof else \
        {k: c["bible"].get(k) for k in planner.BIBLE_FIELDS if c["bible"].get(k)}
    mode = evidence.mode_for(conn, creator_id)
    findings = [{"finding": f["statement"], "grade": f["grade"], **({"label": f["label"]} if f["label"] else {})}
                for f in evidence.findings(conn, mode=mode, creator_id=creator_id)]
    last = analyst.latest(conn, creator_id)
    guidance = (last["output"] or {}).get("guidance") if last else None
    hypotheses = [h["statement"] for h in ((last["output"] or {}).get("hypotheses") or [])] if last else []
    share = float(policy.get("brain", "exploration_share", default=0.3))
    scope = policy.get("scope", default={})
    hashtags_max = dials.get("hashtags_max") or {}
    notes = policy.get("planner", "format_notes", default={}) or {}
    prompt = (
        f"The creator:\n{json.dumps(who, default=str)}\n"
        f"What we know about their life (keep it consistent):\n"
        f"{json.dumps(known)}\n"
        f"Recent posts, newest first, with how their ideas scored (1-10) where scored (don't repeat scenes, jokes "
        f"or captions):\n{json.dumps(_recent_with_scores(conn, creator_id, int(cfg.get('recent_posts', 30))), default=str)}\n"
        f"What has been learned about this creator so far (grades: proven > tentative > speculative):\n"
        f"{json.dumps(findings) if findings else 'nothing yet'}\n"
        f"The analyst's latest guidance for this batch (follow it where the evidence is strong, keep exploring where "
        f"it isn't; say in an idea's reasoning when a finding shaped it): "
        f"{json.dumps({'label': 'SIMULATED (sandbox test)' if mode == 'sandbox' else 'real', 'guidance': guidance, 'hypotheses': hypotheses}) if guidance else 'no analysis yet'}\n\n"
        f"Days: {day_names} (weekly rituals happen on their weekday). "
        f"Plan about {counts['total']} posts and Stories over {days} days (day 0 to {days - 1}): "
        f"{counts['per_platform']} per platform; about {counts['videos']} videos "
        f"({planner._video_formats(policy)}), each in one of the creator's video series. "
        f"Formats and images per post: { {f: f'{lo}-{hi} images' for f, (lo, hi) in formats.items()} }. "
        f"What each format is: {json.dumps({f: n for f, n in notes.items() if f in formats})}. "
        f"About {round(share * 100)}% of the ideas should try something this creator hasn't done yet (mark them "
        f"exploratory); the rest build on what works. Hashtags per post at most {hashtags_max or 1}; often none. "
        f"Never touch: {scope.get('excluded_topics')}. No sponsored, product or review posts, no selling. "
        f"The creator's face is visible in most images, wearing their signature item.")
    meta = {"profile_version": version, "counts": counts, "hashtags_max": hashtags_max, "prof": prof,
            "analysis_decision_id": str(last["id"]) if last else None,
            "finding_ids": [str(f["id"]) for f in evidence.findings(conn, mode=mode, creator_id=creator_id)],
            "platforms": list(scope.get("platforms") or ["instagram"])}
    return prompt, meta


class Showrunner:
    name = "showrunner"

    def run(self, ctx: AgentContext) -> dict:
        p = ctx.job.payload or {}
        if not p.get("creator_id") or not p.get("start"):
            raise NonRetryable("showrunner payload needs creator_id and start")
        done = ctx.conn.execute("SELECT output FROM decisions WHERE job_id = %s AND step = 'plan'",
                                (ctx.job.id,)).fetchone()
        if done:   # a rerun returns what this job already planned
            return done["output"]
        creator_id, start = p["creator_id"], date.fromisoformat(p["start"])
        days = int(ctx.policy.get("brain", "showrunner", "days", default=3))
        prompt, meta = _prompt(ctx, creator_id, start, days)
        out = ctx.llm.structured("brain", system=SYSTEM, content=prompt, schema=IdeaBatch,
                                 effort=ctx.policy.get("brain", "showrunner", "effort", default="medium"),
                                 creator_id=creator_id)
        kept, dropped = [], []
        for item in out.ideas:
            q, why = planner.validate(ctx.policy, item, platforms=meta["platforms"], days=days, prof=meta["prof"],
                                      hashtags_max=meta["hashtags_max"])
            (kept.append(q) if q else dropped.append({"caption": item.caption[:80], "why": why}))
        if not kept:
            raise NonRetryable(f"the showrunner returned no usable ideas: {dropped[:5]}")
        batch = f"{creator_id}:{start}"
        with ctx.conn.transaction():   # the batch, its evaluation requests and the decision land together
            # Serialise runs of this job: if another worker (after a lost lease) already wrote the
            # batch during our LLM call, return its result and write nothing.
            ctx.conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"showrunner:{ctx.job.id}",))
            done = ctx.conn.execute("SELECT output FROM decisions WHERE job_id = %s AND step = 'plan'",
                                    (ctx.job.id,)).fetchone()
            if done:
                return done["output"]
            ids = []
            for q in kept:
                body = {"scenes": q.scenes, "script": q.script, "on_screen_text": q.on_screen_text,
                        "hashtags": q.hashtags, "canon": q.canon, "pillar": q.pillar, "tone": q.tone,
                        "series": q.series, "caption": q.caption, "day": q.day, "platform": q.platform,
                        "exploratory": q.exploratory, "reasoning": q.reasoning, "canon_key": q.canon_key}
                concept = " | ".join(q.scenes) + (f" | script: {q.script}" if q.script else "") + f" | caption: {q.caption}"
                iid = ctx.conn.execute(
                    "INSERT INTO ideas (creator_id, batch, concept, format, body, attributes, status) "
                    "VALUES (%s, %s, %s, %s, %s, %s, 'proposed') RETURNING id",
                    (creator_id, batch, f"{q.format}: {concept}", q.format, Jsonb(body),
                     Jsonb({**q.attributes, **({"exploratory": "yes"} if q.exploratory else {})}))).fetchone()["id"]
                commission(ctx, "evaluator", {"idea_id": str(iid)}, reason="score before scheduling", key=f"eval:{iid}")
                ids.append(str(iid))
            output = {"batch": batch, "ideas": len(ids), "idea_ids": ids, "dropped": dropped}
            record_decision(ctx, "content_batch", step="plan", creator_id=creator_id, model=ctx.llm.model_for("brain"),
                            inputs={"start": str(start), "days": days, "profile_version": meta["profile_version"],
                                    "requested": meta["counts"], "analysis_decision_id": meta["analysis_decision_id"],
                                    "finding_ids": meta["finding_ids"]},
                            output=output, rationale=out.rationale)
        return output


register(Showrunner())
