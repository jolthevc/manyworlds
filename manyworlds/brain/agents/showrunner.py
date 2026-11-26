"""The showrunner: plans a creator's next few days as ideas, the way the person would actually post.

It knows the creator (profile and memory), sees how its recent ideas scored and what has been
learned about this creator (findings, read in the creator's evidence mode, each with its level, replications and
age: manyworlds.brain.hypotheses), and spends an exact part of each batch exploring: every idea is labelled refine,
adjacent or novel, the counts come from her exploration temperature (manyworlds.brain.exploration.batch_targets,
policy `brain.exploration_mix` and the portfolio total `brain.exploration_share`, which the Brain may tune), and
code fits the batch to them (exploration.reconcile: trim or flag, no extra call). Hypotheses comparable creators proved
are offered as experiments, never rules. Ideas carry attributes for learning (setting, outfit, activity, mood, trend) but
no strategic category; ordinary moments are welcome. Its reasoning is stored on each idea and never
shown to the evaluator. Every idea is sent to the evaluator; only evaluated ideas become posts
(manyworlds.brain.ideas.schedule_ready).
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone

from psycopg.types.json import Jsonb
from typing import Literal

from pydantic import BaseModel, Field

from ...creator import dossier, profile as profile_mod
from ...worker.runner import NonRetryable
from .. import (archetypes, events, evidence, experiments, exploration, hypotheses, memory, patterns, planner,
               proposals, series)
from ..agent import AgentContext, commission, record_decision, register
from . import analyst, comment_themes, style_director, talent_manager

SYSTEM = planner.SYSTEM + """

You are the creator's showrunner. Plan the next few days as a natural feed: everyday moments,
outfits, food, places, small opinions, the occasional trend or arc, and Stories for throwaway
moments. Not every post needs a hook or a purpose; a lovely ordinary moment is a good post. Keep the
creator's life coherent with what is known about them, and learn from how recent ideas scored and
from any findings, without turning the feed into a formula.""" + planner.DIRECTION


class Slide(BaseModel):
    template: Literal["cover", "list_item", "step", "quote", "comparison", "recap", "cta"]
    heading: str = Field(description="the slide's headline: 8 words or fewer on a cover")
    text: str = Field(default="", description="one or two short lines; a comparison's two sides split by '|'")
    items: list[str] = Field(default_factory=list, description="a recap's points, or a comparison's two labels")
    photo: int = Field(default=-1, description="the index of the scene whose photo this slide shows, or -1")


class PlannedIdea(planner.PlannedPost):
    attributes: dict[str, str] = Field(default_factory=dict,
                                       description="setting, outfit, activity, mood, trend (yes/no): plain words")
    reasoning: str = Field(default="", description="why this post now, in one or two sentences")
    exploration: Literal["refine", "adjacent", "novel"] = Field(
        default="refine", description="refine: builds on what works; adjacent: a step from it (a new setting, hook or "
                                      "format on her proven ground); novel: something she hasn't done")
    canon_key: str = Field(default="", description="if `canon` replaces a single fact (home, partner, job, pet, "
                                                   "employer), that fact's name; otherwise empty")
    staging: str = Field(default="", description="the name of a staging from the library, or empty (most posts)")
    flavor: str = Field(default="", description="the name of a flavor from the library, or empty (most posts)")
    move: str = Field(default="", description="the name of a move from the library, or empty (most posts)")
    pattern: str = Field(default="", description="the name of the pattern card this idea follows, or empty")
    experiment: str = Field(default="", description="the slot id when this idea fills an experiment slot, else empty")
    slides: list[Slide] = Field(default_factory=list, description="a carousel's text slides in her brand (capability "
                                                                  "edits.text_slides), in order; empty for most posts")
    # Series continuity (manyworlds.brain.series): flat fields only (a nested list once broke a grammar limit).
    episode: int = Field(default=0, description="in a series: this episode's number (the series' next_episode, one "
                                                "more for each later episode in this batch); else 0")
    episode_premise: str = Field(default="", description="in a series: this episode's premise in one line; else empty")
    opens_thread: str = Field(default="", description="a thread this post leaves open (a promised part 2, a running "
                                                      "joke, a question to the audience), in one line; usually empty")
    closes_threads: list[str] = Field(default_factory=list, description="ids of open threads this post pays off")


class IdeaBatch(BaseModel):
    rationale: str
    ideas: list[PlannedIdea]


def _recent_with_scores(conn, creator_id, n: int, policy=None) -> list[dict]:
    """Recent posts, newest first, with how their ideas scored and, once published and old enough, how they did
    (evidence.post_outcomes: reach at the analysis target age, in the creator's evidence mode, labelled SIMULATED
    for a sandbox creator)."""
    rows = conn.execute(
        """SELECT p.id, p.format, p.caption, p.body->'scenes' AS scenes, p.body->>'canon' AS canon,
                  p.body->>'series' AS series, e.scores
           FROM posts p
           -- one row per post: a post linked to several ideas (from_post and a scheduled idea) shows the
           -- latest evaluation of any of them, never a second copy of itself
           LEFT JOIN LATERAL (SELECT ev.scores FROM ideas i JOIN evaluations ev ON ev.idea_id = i.id
                              WHERE i.post_id = p.id AND ev.role = 'evaluator'
                              ORDER BY ev.created_at DESC LIMIT 1) e ON true
           WHERE p.creator_id = %s AND p.status <> 'dropped'
           ORDER BY coalesce(p.scheduled_for, p.created_at) DESC LIMIT %s""", (creator_id, n)).fetchall()
    outcomes = evidence.post_outcomes(conn, creator_id, n, evidence.mode_for(conn, creator_id), policy=policy)
    out = []
    for r in rows:
        d = {k: v for k, v in dict(r).items() if v and k != "id"}
        if str(r["id"]) in outcomes:
            d["outcome"] = outcomes[str(r["id"])]
        out.append(d)
    return out


def stagings_for(policy, formats, section: str = "stagings") -> dict:
    """The stagings (or flavors) whose formats production makes now."""
    lib = policy.get(section, "library", default={}) or {}
    return {k: v for k, v in lib.items() if set(v.get("formats") or []) & set(formats)}


def moves_for(policy, formats) -> dict:
    """The moves (policy `moves`) whose formats production makes now and whose readiness the models can carry
    (policy `direction.offer_readiness`): a move the models can't do yet is never offered."""
    offered = set(policy.get("direction", "offer_readiness", default=[]) or [])
    return {k: v for k, v in stagings_for(policy, formats, "moves").items() if v.get("readiness") in offered}


def recipe_view(moves: dict) -> dict:
    """The recipes as the showrunner sees them: what each is, its formats and readiness, the archetypes it suits,
    the capabilities it uses and its length; its shot defaults are the director's business."""
    keys = ("note", "formats", "readiness", "suits", "uses", "length_s")
    return {name: {k: m[k] for k in keys if m.get(k)} for name, m in moves.items()}


def _strategy_line(policy) -> str:
    """The agency's content guidance and commercial rule (policy strategy): what to build on, and how objects appear
    in her world until a brand deal exists."""
    s = policy.get("strategy", default={}) or {}
    parts = [str(s.get("content") or "").strip(), str((s.get("commercial") or {}).get("content_rule") or "").strip(),
             f"Platform guidelines: {json.dumps(s.get('platform_guidelines') or {})}."]
    return " ".join(p for p in parts if p) + (" " if any(parts) else "")


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
    from ...publish import instagram
    dials = instagram.ramp_dials(conn, policy, creator_id, dials,
                                 datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc))
    dials = proposals.tier_dials(policy, proposals.tier_of(conn, policy, creator_id), dials)   # within her tier
    counts = planner._counts(policy, dials, days)
    formats = planner._formats(policy)
    who = {k: prof[k] for k in planner.ANALYSIS_FIELDS if k in prof} if prof else \
        {k: c["bible"].get(k) for k in planner.BIBLE_FIELDS if c["bible"].get(k)}
    mode = evidence.mode_for(conn, creator_id)
    outcome_age_h = round(int(policy.get("brain", "analysis", "target_age_s", default=259200)) / 3600)
    found = evidence.findings(conn, mode=mode, creator_id=creator_id)
    findings = hypotheses.views(policy, found)   # level, replications, age, weight; stale ones due for retest
    offered = hypotheses.planning_views(conn, policy, creator_id)
    offered_text = (f"Hypotheses from comparable creators (each held in the creators counted, never tested on her: "
                    f"suggestions an experiment may test, not rules; never copy their posts): {json.dumps(offered)}\n"
                    if offered else "")
    last = analyst.latest(conn, creator_id)
    guidance = (last["output"] or {}).get("guidance") if last else None
    hypotheses = [h["statement"] for h in ((last["output"] or {}).get("hypotheses") or [])] if last else []
    mix = exploration.batch_targets(conn, policy, creator_id, counts["total"])
    t = mix["targets"]
    scope = policy.get("scope", default={})
    hashtags_max = dials.get("hashtags_max") or {}
    notes = policy.get("planner", "format_notes", default={}) or {}
    stagings = stagings_for(policy, formats)
    flavors = stagings_for(policy, formats, "flavors")
    flavor_text = (f"Flavors you may give an idea (set `flavor` to its name; most posts need none): "
                   f"{json.dumps(flavors)}. " if flavors else "")
    moves = moves_for(policy, formats)
    move_text = (f"Recipes (moves) you may give an idea (set `move` to its name; most posts need none; examples composed "
                 f"from the capabilities, each with the archetypes it suits and the length that works; only in a "
                 f"format the recipe lists; the scene that performs it stays short and simple where its readiness is "
                 f"partial): {json.dumps(recipe_view(moves))}. " if moves else "")
    capability_text = planner.capability_brief(policy, who)
    plays = archetypes.playbooks(policy, prof or c["bible"])
    play_text = (f"Her archetype playbooks (how this kind of creator wins: mix, series patterns, hooks, edit grammar, "
                 f"pitfalls; a guide, not a template): {json.dumps(plays)}\n" if plays else "")
    look = style_director.planning_view(policy, prof, today=start)
    hemisphere = ((prof or {}).get("world") or {}).get("hemisphere") or "north"
    season = _season(start, hemisphere)
    style_text = (f"Her style director's look (keep her style, palette, grade and signature items; vary outfits from her "
                  f"capsule, places and shot types; plan photo dumps and carousels in the dump order and by the rules; "
                  f"most frames show her world, not her face to camera, roughly as her face share says): "
                  f"{json.dumps(look, default=str)}\n" if look else "")
    brief = talent_manager.latest_brief(conn, creator_id)
    brief_text = (f"Her talent manager's brief for these batches (follow it: it sets the focus, the series to run and "
                  f"what to stop): {json.dumps(brief['output'], default=str)}\n" if brief else "")
    continuity_text = _continuity_text(conn, policy, creator_id, prof)
    staging_text = (f"Stagings you may give an idea (set `staging` to its name; most posts need none; the scene must "
                    f"show its setting, framing and wardrobe and the script follow its conventions): "
                    f"{json.dumps(stagings)}. {policy.get('stagings', 'rules', default='')} " if stagings else "")
    slots = experiments.assign(conn, policy, creator_id, start=start, days=days)
    effects = ((last["output"] or {}).get("effects") or []) if last else []
    explore = exploration.directions(policy, effects=effects, seed_key=f"{creator_id}:{start}",
                                     untried=exploration.untried(conn, creator_id, formats=list(formats),
                                                                 stagings=list(stagings), flavors=list(flavors)))
    slot_text = (f"Experiment slots (randomised: write exactly one idea per slot, on its day, that follows its "
                 f"instruction and is still a natural post; set `experiment` to the slot id): "
                 f"{json.dumps([{k: s[k] for k in ('slot', 'day', 'platform', 'instruction')} for s in slots])}. " if slots else "")
    qa_window = int(policy.get("brain", "qa", "window", default=12))
    qa_drops = [r["why"] for r in conn.execute(
        "SELECT body->>'drop_reason' AS why FROM ideas WHERE creator_id = %s AND status = 'dropped' "
        "AND body->>'drop_reason' ~ '^(repetitive feed|unoriginal|[a-z_]+ [0-9.]+ below)' "
        "ORDER BY updated_at DESC LIMIT %s", (creator_id, qa_window)).fetchall()]
    niche = evidence.niche_trends(conn, creator_id, policy)
    niche_text = (f"What is working in this niche on public TikTok and Instagram posts right now (other people's posts: "
                  f"correlation, not proof; use it for hooks, formats and pacing, never copy, and stay in the creator's "
                  f"voice): {json.dumps(niche)}\n" if niche else "")
    cards = pattern_cards(conn, policy, prof or c["bible"], formats)
    from ...research import desk
    culture_lines = desk.brief(conn, policy, use="content")
    culture_text = (f"Research desk now (findings from current sources, with their evidence mix; ideas may ride one "
                    f"when it suits her, never forced):\n"
                    f"{culture_lines}\n" if culture_lines else "")
    card_text = (f"Pattern cards from the creative pattern library (photo and video patterns that make creators like "
                 f"her worth following; ILLUSTRATIVE ones are craft knowledge, researched ones carry evidence and "
                 f"weigh more). An idea that follows one sets `pattern` to its name and takes its format, hook, "
                 f"shot types and recipe (move, edit grammar) in her own voice; most ideas still need none, and none "
                 f"is ever a copy of anyone's post:\n{patterns.brief(cards, detail='plan')}\n" if cards else "")
    breakouts = events.for_planning(conn, creator_id, policy=policy)
    breakout_text = (f"Recent posts that did far better than usual (a natural follow-up may suit; never a copy; one "
                     f"marked unreplicated is a single observation, not a pattern: never build a formula on it): "
                     f"{json.dumps(breakouts)}. " if breakouts else "")
    qa_text = (f"Recent ideas the feed check dropped (don't repeat these mistakes): {json.dumps(qa_drops)}. "
               if qa_drops else "")
    prompt = (
        f"The creator:\n{json.dumps(who, default=str)}\n"
        f"What we know about their life (keep it consistent):\n"
        f"{json.dumps(known)}\n"
        f"Recent posts, newest first, with how their ideas scored (1-10) where scored and, once published and "
        f"old enough, how they did (`outcome`: reach at {outcome_age_h} h; {'SIMULATED, a sandbox test, not evidence of demand' if mode == 'sandbox' else 'real'}). "
        f"Learn from what reached people, not only from the scores; don't repeat scenes, jokes or captions:\n"
        f"{json.dumps(_recent_with_scores(conn, creator_id, int(cfg.get('recent_posts', 30)), policy), default=str)}\n"
        f"Her series (a video belongs to one of these, by name): {json.dumps(planner.series_catalogue(prof), default=str)}\n"
        f"{continuity_text}"
        f"What has been learned about this creator so far (hypotheses, strongest first; grades: proven > tentative > "
        f"speculative; our own experiments outrank observation; one marked due_for_retest is stale and stays "
        f"revisable):\n"
        f"{json.dumps(findings) if findings else 'nothing yet'}\n"
        f"{offered_text}{niche_text}{play_text}{card_text}{culture_text}{brief_text}{style_text}"
        f"The analyst's latest guidance for this batch (follow it where the evidence is strong, keep exploring where "
        f"it isn't; say in an idea's reasoning when a finding shaped it): "
        f"{json.dumps({'label': 'SIMULATED (sandbox test)' if mode == 'sandbox' else 'real', 'guidance': guidance, 'hypotheses': hypotheses}) if guidance else 'no analysis yet'}\n\n"
        f"Days: {day_names} (weekly rituals happen on their weekday; it is {season} where she lives: the light, "
        f"weather, food and things to do follow the season). "
        f"Plan about {counts['total']} posts and Stories over {days} days (day 0 to {days - 1}): "
        f"{counts['per_platform']} per platform; about {counts['videos']} videos "
        f"({planner._video_formats(policy)}), each in one of the creator's video series. "
        f"Formats and images per post: { {f: f'{lo}-{hi} images' for f, (lo, hi) in formats.items()} }. "
        f"What each format is: {json.dumps({f: n for f, n in notes.items() if f in formats})}. "
        f"{capability_text}{staging_text}{flavor_text}{move_text}"
        f"Exploration: of the {counts['total']} ideas, exactly {t['refine']} refine (build on what works), "
        f"{t['adjacent']} adjacent (a step from what works: a new setting, hook or format on her proven ground) and "
        f"{t['novel']} novel (something she hasn't done yet); set `exploration` on each. Her identity stays "
        f"coherent in all of them; only the content moves. Point the adjacent and novel ones at these directions, "
        f"chosen by sampling over what is known and untried: {json.dumps([d['describe'] for d in explore])}. "
        f"{slot_text}{qa_text}{breakout_text}Hashtags per post at most {hashtags_max or 1}; often none. "
        f"Never touch: {scope.get('excluded_topics')}. No sponsored, product or review posts, no selling. "
        f"{_strategy_line(policy)}{dossier.presentation_line(prof)}"
        + ("Covers and talking show her face; details, places, food, her back and her hands carry the rest. "
           if look else "") + planner.face_rule(prof))
    meta = {"profile_version": version, "counts": counts, "hashtags_max": hashtags_max, "prof": prof,
            "analysis_decision_id": str(last["id"]) if last else None,
            "talent_brief_id": str(brief["id"]) if brief else None,
            "finding_ids": [str(f["id"]) for f in found],
            "platforms": list(scope.get("platforms") or ["instagram"]), "stagings": stagings, "flavors": flavors,
            "moves": moves, "slots": {s["slot"]: s for s in slots}, "exploration": explore, "mix": mix,
            "cards": {c["key"]: c for c in cards}}
    return prompt, meta


def pattern_cards(conn, policy, who: dict | None, formats) -> list[dict]:
    """The pattern cards the showrunner is offered: by her archetypes and style's shot families, for the formats it
    may plan (manyworlds.brain.patterns.retrieve, stage showrunner)."""
    media = sorted({m for f in formats for m in (patterns.media_for(policy, f) or [])}) or None
    families = ((who or {}).get("style") or {}).get("shot_families") or None
    return patterns.retrieve(conn, policy, stage="showrunner", who=who, media=media, shot_types=families)


def apply_pattern(policy, body: dict, fmt: str, cards: dict, chosen: str, moves: dict) -> dict:
    """An idea's pattern card into its body: the card's move (when the idea has none and the move is offered for its
    format), a talking post's edit grammar and its clamped talk-edit overrides, and `pattern_cards` (every card
    offered, and which one the idea applied) for attribution. A card whose medium doesn't fit the format is ignored."""
    card = cards.get(chosen) if chosen else None
    if card and card.get("medium") not in (patterns.media_for(policy, fmt) or [card.get("medium")]):
        card = None
    recipe = (card or {}).get("recipe") or {}
    if card:
        move = recipe.get("move")
        if move and not body.get("move") and fmt in ((moves.get(move) or {}).get("formats") or []):
            body["move"] = move
        if fmt == "talking_video" and recipe.get("grammar"):
            body["edit_grammar"] = recipe["grammar"]
            over = patterns.talk_overrides(policy, card)
            if over:
                body["edit_overrides"] = over
    offered = [c for c in cards.values() if c.get("medium") in (patterns.media_for(policy, fmt) or [c.get("medium")])]
    if offered:
        body["pattern_cards"] = patterns.usage(offered, "showrunner", body=body if card else None,
                                               chosen=[card["key"]] if card else [])
    return body


def _continuity_text(conn, policy, creator_id, prof) -> str:
    """Where each series stands (manyworlds.brain.series: next episode number, last episodes, open threads, what its
    audience asked for) and the latest comment themes (agents/comment_themes); empty when there is neither."""
    view = series.planning_view(conn, policy, creator_id, planner.series_catalogue(prof))
    themes = comment_themes.for_planning(conn, policy, creator_id)
    for t in themes:
        if t.get("series") in view and t["series"] and t.get("kind") in ("question", "request"):
            view[t["series"]].setdefault("audience_asked", []).append(t["text"])
    text = ""
    if any(v.get("last_episodes") or v.get("open_threads") for v in view.values()):
        text += (f"Where her series stand (an idea in a series sets `series`, `episode` (its next_episode; one more for "
                 f"each later episode in this batch) and `episode_premise`; follow on from the last episodes without "
                 f"repeating them, call back to them, answer what the audience asked; a post that pays off an open "
                 f"thread lists its id in `closes_threads`, and one that leaves something open says so in "
                 f"`opens_thread`; threads under \"\" are hers, not one series'): {json.dumps(view, default=str)}\n")
    if themes:
        text += (f"What her audience keeps saying in comments on her posts (themes, in our words; let it shape ideas, "
                 f"answer a question or grant a request where it fits her; never quote or name a commenter): "
                 f"{json.dumps([{k: t[k] for k in ('kind', 'text', 'series') if t.get(k)} for t in themes])}\n")
    return text


def _season(day: date, hemisphere: str) -> str:
    """The season on a day ("late autumn"), by meteorological seasons, flipped for the southern hemisphere."""
    names = ["winter", "winter", "spring", "spring", "spring", "summer", "summer", "summer", "autumn", "autumn",
             "autumn", "winter"]
    starts = {"winter": 11, "spring": 2, "summer": 5, "autumn": 8}
    m = (day.month - 1) if hemisphere != "south" else (day.month + 5) % 12
    part = ("early", "mid", "late")[(m - starts[names[m]]) % 12]
    return f"{part} {names[m]}"


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
        role = ctx.policy.get("brain", "showrunner", "role", default="brain")
        out = ctx.llm.structured(role,
                                 system=SYSTEM + planner.voice_rules(ctx.policy),
                                 content=prompt, schema=IdeaBatch,
                                 effort=ctx.policy.get("brain", "showrunner", "effort", default="medium"),
                                 creator_id=creator_id)
        kept, dropped = [], []
        for item in out.ideas:
            q, why = planner.validate(ctx.policy, item, platforms=meta["platforms"], days=days, prof=meta["prof"],
                                      hashtags_max=meta["hashtags_max"])
            (kept.append(q) if q else dropped.append({"caption": item.caption[:80], "why": why}))
        if not kept:
            raise NonRetryable(f"the showrunner returned no usable ideas: {dropped[:5]}")
        # The exploration temperature, in code: trim a batch far over its size (the most over-target level first,
        # never an experiment slot) and flag ideas over their level's count; labels are never rewritten.
        mix = meta["mix"]
        total = int(meta["counts"]["total"]) or len(kept)
        over = float(ctx.policy.get("brain", "exploration_mix", "trim_over_share", default=0.25))
        size = max(total, math.ceil(total * (1 + over)))
        fit = exploration.reconcile([q.exploration for q in kept], mix["targets"], size,
                                    keep={i for i, q in enumerate(kept) if q.experiment})
        dropped += [{"caption": kept[i].caption[:80],
                     "why": f"exploration mix: batch over {size} ideas, {kept[i].exploration} over its "
                            f"{mix['targets'][kept[i].exploration]}"} for i in fit["trimmed"]]
        flagged = {fit["kept"].index(i) for i in fit["flagged"]}
        kept = [kept[i] for i in fit["kept"]]
        batch = f"{creator_id}:{start}"
        with ctx.conn.transaction():   # the batch, its evaluation requests and the decision land together
            # Serialise runs of this job: if another worker (after a lost lease) already wrote the
            # batch during our LLM call, return its result and write nothing.
            ctx.conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"showrunner:{ctx.job.id}",))
            done = ctx.conn.execute("SELECT output FROM decisions WHERE job_id = %s AND step = 'plan'",
                                    (ctx.job.id,)).fetchone()
            if done:
                return done["output"]
            ids, used = [], set()
            for qi, q in enumerate(kept):
                slot = meta["slots"].get(q.experiment) if q.experiment else None
                if slot and (slot["slot"] in used or slot["day"] != q.day or slot["platform"] != q.platform):
                    slot = None   # a slot is filled once, on its randomised day
                if slot:
                    used.add(slot["slot"])
                staging = q.staging if q.staging in meta["stagings"] else ""   # unknown names are dropped
                flavor = q.flavor if q.flavor in meta["flavors"] else ""
                offered = meta["moves"].get(q.move) or {}
                move = q.move if q.format in (offered.get("formats") or []) else ""   # unknown or off-format: dropped
                body = {"scenes": q.scenes, "script": q.script, "on_screen_text": q.on_screen_text,
                        "hashtags": q.hashtags, "canon": q.canon, "pillar": q.pillar, "tone": q.tone,
                        "series": q.series, "caption": q.caption, "day": q.day, "platform": q.platform,
                        "exploratory": q.exploration != "refine", "exploration": q.exploration,
                        "reasoning": q.reasoning, "canon_key": q.canon_key, "staging": staging, "flavor": flavor}
                if qi in flagged:   # over its level's count in this batch: kept; the next batch's carry corrects it
                    body["exploration_off_target"] = True
                if q.series:   # read back at publication, when the post becomes an episode (manyworlds.brain.series)
                    body.update(episode=max(int(q.episode or 0), 0), episode_premise=q.episode_premise.strip())
                if q.opens_thread.strip():
                    body["opens_thread"] = q.opens_thread.strip()
                closes = [t.strip().lower() for t in q.closes_threads if t.strip()]
                if closes:
                    body["closes_threads"] = closes
                if move:   # the director (manyworlds.brain.director) starts that scene from the move's shot
                    body["move"] = move
                apply_pattern(ctx.policy, body, q.format, meta["cards"], q.pattern.strip(), meta["moves"])
                if q.slides:   # carousel text slides (manyworlds.slides), carried to the post (ideas._DIRECTED_FIELDS)
                    body["slides"] = [s.model_dump(mode="json") for s in q.slides]
                concept = " | ".join(q.scenes) + (f" | script: {q.script}" if q.script else "") + f" | caption: {q.caption}"
                iid = ctx.conn.execute(
                    "INSERT INTO ideas (creator_id, batch, concept, format, body, attributes, status, experiment_id, arm) "
                    "VALUES (%s, %s, %s, %s, %s, %s, 'proposed', %s, %s) RETURNING id",
                    (creator_id, batch, f"{q.format}: {concept}", q.format, Jsonb(body),
                     Jsonb({**q.attributes, "exploration": q.exploration,
                            **({"exploratory": "yes"} if q.exploration != "refine" else {}),
                            **({"staging": staging} if staging else {}), **({"flavor": flavor} if flavor else {}),
                            **({"move": move} if move else {})}),
                     slot["experiment_id"] if slot else None, slot["arm"] if slot else None)).fetchone()["id"]
                commission(ctx, "evaluator", {"idea_id": str(iid)}, reason="score before scheduling", key=f"eval:{iid}")
                ids.append(str(iid))
            output = {"batch": batch, "ideas": len(ids), "idea_ids": ids, "dropped": dropped}
            realised = {"stage": mix["stage"], "plateau": mix["plateau"], "own": mix["own"], "shares": mix["shares"],
                        "correction": mix["correction"], "targets": mix["targets"], "planned": fit["planned"],
                        "shortfall": fit["shortfall"], "trimmed": len(fit["trimmed"]), "flagged": len(fit["flagged"])}
            exploration.record_window(ctx.conn, creator_id, mix, fit["planned"])
            record_decision(ctx, "content_batch", step="plan", creator_id=creator_id, model=ctx.llm.model_for(role),
                            inputs={"start": str(start), "days": days, "profile_version": meta["profile_version"],
                                    "requested": meta["counts"], "analysis_decision_id": meta["analysis_decision_id"],
                                    "talent_brief_id": meta["talent_brief_id"],
                                    "pattern_cards": sorted(meta["cards"]),
                                    "finding_ids": meta["finding_ids"], "exploration": meta["exploration"],
                                    "exploration_mix": realised,
                                    "experiment_slots": {"offered": len(meta["slots"]), "filled": len(used)}},
                            output=output, rationale=out.rationale)
        return output


register(Showrunner())
