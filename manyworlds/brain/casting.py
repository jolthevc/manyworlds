"""Casting with auditions (owner 2026-11-25; spec docs/superpowers/specs/2026-11-25-talent-agency.md).

A brief (a thesis) becomes 3-5 candidate creators, the best of them are auditioned with a cheap mini-feed rendered
through normal production, and the winner is picked from the rubric, what production actually accepted and what it
cost. Code around three kinds of model call; everything else is deterministic:

    cast        one call (role `casting`) develops brain.casting.candidates compact concepts (a bible draft: who, look
                and styling direction, audience; the brand bible core; commercial categories; content mix), informed by
                the thesis, the strategy, the portfolio map's gaps, the capability card, the commercial registry and the
                pattern cards for casting. One judge call (role `casting_judge`) scores every concept on the rubric
                (strategy.rubric: A desirability, B audience surface area, C commercial viability, D production
                feasibility, E distinctiveness and longevity) and the restrictions (strategy.desirability). Code then:
                  - D is the lower of the judge's score and code's (the concept's content mix weighted by the capability
                    card's reliability) and is a hard gate;
                  - C is the mean of the judge's and the registry's index of the concept's categories;
                  - E is capped (too_close_cap) when the concept's positioning words overlap a sibling past
                    brain.talent_manager.max_overlap;
                  - a guardrail flag, a name or handle that impersonates a named reference (manyworlds.brain.impersonation),
                    or any criterion under its floor puts the concept out.
                The best brain.casting.audition_top eligible concepts are shortlisted; each concept is an `auditions` row
                (arms A-E) with its scores; the whole casting is a `casting` decision.
    audition    per shortlisted concept: one bible call (role `creator`, the concept in hand) with the mini-feed's scenes;
                a sandbox creator (anchors.simulation.audition: never a portfolio creator, never published); a small
                reference pack in a test batch; the anchor pick (vision); then brain.casting.audition.posts photo posts
                through normal production in the test batch <batch_prefix><run>-<arm> (body.breadth_run marks them:
                the producer and the publisher leave them alone; this module advances them).
    judge       when every post is accepted or dropped (or the audition's deadline passed): acceptance (accepted /
                posts) and cost (every ledger entry of the audition creator and its jobs) per accepted post.
    pick        the winner maximises rubric x acceptance^acceptance_weight x (cost_ref_usd / $ per accepted)^cost_weight
                among auditions with acceptance >= min_acceptance; a `casting_pick` decision. The winner is promoted
                (promote: linked to the thesis, marker removed) and becomes the launched creator; the others retire.

A launch runs it as its first step (manyworlds.brain.launches, step `casting`, while brain.casting.enabled): the model
steps are `brain.creator_lab` jobs; `advance` (code) moves auditions along each producer tick. `mw brain casting
<thesis>` runs the same flow as a sandbox test (the model steps inline), reports the winner and launches nothing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from .. import queue
from ..creator import dossier, lab
from ..policy import Policy
from . import archetypes, capability_card, commerce, impersonation, portfolio_map

log = logging.getLogger("manyworlds.casting")

CRITERIA = ("desirability", "surface_area", "commercial", "feasibility", "distinctiveness")
ARMS = ("A", "B", "C", "D", "E")
NOBODY = "00000000-0000-0000-0000-000000000000"   # siblings of a creator not made yet: every portfolio creator
OPEN = ("shortlisted", "bible", "references", "anchors", "feed")

SYSTEM = """You are the head of casting at Manyworlds, an AI-native modeling agency, talent development company and
lifestyle media network that owns its talent. Every creator is original, fictional and openly labeled as AI. You
develop candidate talent the way the best agencies scout: desirable and aspirational, specific enough to be
memorable, built for an audience and for commercial work from day one, and designed around what our production
renders reliably. Candidates differ from each other in a real way, not by a detail. Real creators may be references
for appeal, but a candidate is never a real person's name, handle, face or life. Adults only. Desirability is the
goal: attractive, magnetic, stylish and aspirational in each candidate's own register (beautiful, rugged, elegant,
playful, mysterious), to the standard of the best mainstream fashion and lifestyle work; never explicit."""

JUDGE = """You judge candidate talent for an AI-native modeling agency and lifestyle media network, strictly and
independently: you did not write these candidates. Score each criterion 0-10 against the rubric with a reason grounded
in the candidate. Say plainly when a candidate is generic (broad but bland), when production could not render its
world, and when it falls under the restrictions (explicit, visible nudity or a deliberately staged nudity simulation,
fetish, soliciting, simulated intimacy with followers, gratification-first). Judge attractiveness in context: swimwear, shirtlessness and revealing
fashion are not problems in themselves, and a sterile candidate is as weak as a crude one."""


class Audience(BaseModel):
    core: str = Field(description="the core audience, one sentence")
    age_range: str = Field(description="e.g. 28-45")
    gender: str = Field(description="women, men, or mixed")
    regions: str
    adjacent_categories: list[str] = Field(default_factory=list, description="adjacent lifestyle interests")


class ContentMix(BaseModel):
    photo: float = 0.0
    carousel: float = 0.0
    story: float = 0.0
    narrated_edit: float = 0.0
    story_clip: float = 0.0
    talking_video: float = 0.0


class Concept(BaseModel):
    display_name: str
    handle: str = Field(description="lowercase, letters, digits, underscores, periods")
    one_line: str
    archetype: str = Field(description="one playbook name from the library: how her content is built")
    propositions: list[dossier.PropositionShare] = Field(default_factory=list, description="WHY people would "
                                                         "follow: 1-3 audience propositions with shares")
    verticals: list[str] = Field(default_factory=list, description="WHAT she posts about: 1-3 verticals")
    desire: str = Field(description="the desire or aspiration they embody")
    why_follow: str
    signature: str = Field(description="the memorable element: a look, ritual, place or point of view")
    apparent_age: int
    gender_presentation: str
    look: str = Field(description="appearance sketch: face, hair, build, bearing")
    styling: str = Field(description="look and styling direction: silhouettes, palette, signature pieces")
    environments: list[str]
    audience: Audience
    brand_core: dossier.BrandCore
    commercial_categories: list[str] = Field(description="2-5 keys from the commercial registry")
    content_mix: ContentMix = Field(description="share of posts by format; photography first")
    core_formats: list[str] = Field(default_factory=list, description="the 1-3 formats that directly deliver her "
                                    "appeal (her creator-content fit's core); a personality need not talk to camera")
    production_risks: list[str] = Field(default_factory=list, description="what production may struggle with")


class Concepts(BaseModel):
    concepts: list[Concept]


class Score(BaseModel):
    score: float = Field(description="0-10")
    why: str


class ConceptScore(BaseModel):
    candidate: int = Field(description="the candidate's number")
    desirability: Score
    surface_area: Score
    commercial: Score
    feasibility: Score
    distinctiveness: Score
    guardrail_ok: bool = Field(description="false only if the concept falls under the restrictions given (explicit, "
                                           "nude, fetish, soliciting, simulated intimacy, gratification-first, could "
                                           "read as under age); never for attractiveness, swimwear or fashion alone")
    guardrail_note: str = ""
    generic_risk: str = Field(default="", description="how it could turn broad but bland")


class Scores(BaseModel):
    scores: list[ConceptScore]
    note: str = ""


class AuditionDraft(BaseModel):
    bible: lab.CreatorBible
    feed: list[str] = Field(description="the mini-feed: one scene per photo post, showing her world at its best")


def _cfg(policy: Policy) -> dict:
    return policy.get("brain", "casting", default={}) or {}


def enabled(policy: Policy) -> bool:
    return bool(_cfg(policy).get("enabled"))


def _acfg(policy: Policy) -> dict:
    return _cfg(policy).get("audition") or {}


def batch(policy: Policy, run: str, arm: str) -> str:
    return f"{_acfg(policy).get('batch_prefix', 'test-casting-')}{run}-{arm}"


def refs_batch(policy: Policy, run: str, arm: str) -> str:
    """Where an audition's reference pack renders: a test batch the image review skips (review.gates.pending_jobs)."""
    return f"{batch(policy, run, arm)}.refpack"


def thesis(conn, thesis_id) -> dict:
    t = conn.execute("SELECT id, summary, body FROM theses WHERE id::text = %s", (str(thesis_id),)).fetchone()
    if not t:
        raise LookupError(f"no thesis {thesis_id}")
    return {"id": str(t["id"]), "summary": t["summary"], **(t["body"] or {})}


# --- context --------------------------------------------------------------------------------------------------------
def _references(conn, t: dict) -> list[dict]:
    """The named references of the brief (body.references: handles), as the impersonation check wants them."""
    out = []
    for h in t.get("references") or []:
        handle = str(h.get("handle") if isinstance(h, dict) else h).split(":")[-1].lstrip("@")
        row = conn.execute("SELECT id, handle, display_name FROM reference_creators WHERE lower(handle) = lower(%s) "
                           "LIMIT 1", (handle,)).fetchone()
        out.append({"id": str(row["id"]), "handle": row["handle"], "display_name": row["display_name"]} if row
                   else {"id": handle, "handle": handle, "display_name": None})
    return out


def _context(conn, policy: Policy, t: dict) -> str:
    """What both calls see: the thesis, the strategy and rubric, the portfolio map, the capability card, the registry."""
    s = policy.get("strategy", default={}) or {}
    rubric = {k: {"weight": v.get("weight"), "floor": v.get("floor"), "ask": v.get("ask")}
              for k, v in ((s.get("rubric") or {}).get("criteria") or {}).items()}
    return "\n\n".join([
        f"The brief (thesis): {json.dumps({k: v for k, v in t.items() if k != 'id'}, default=str)}",
        dossier.strategy_text(policy),
        f"Desirability takes many forms (examples, not a taxonomy): {s.get('examples')}",
        f"The casting rubric (0-10 each): {json.dumps(rubric)}",
        f"The portfolio now (cast into its gaps; never a copy of a creator we have):\n"
        f"{portfolio_map.text(portfolio_map.build(conn, policy))}",
        f"What production renders reliably and cheaply (design every candidate's content around it):\n"
        f"{capability_card.brief(capability_card.card(conn, policy))}",
        f"Content guidance: {str(s.get('content') or '').strip()}",
        f"Commercial registry (keys, bands, index; estimates until researched):\n"
        f"{commerce.brief(commerce.categories(conn, policy))}",
    ])


def _hypotheses_text(conn, policy: Policy) -> str:
    """What has held for which archetype across our own creators (manyworlds.brain.hypotheses, real evidence only), with
    how many creators it held and failed in: hypotheses about an archetype, never proof about a new creator."""
    from . import hypotheses
    rows = hypotheses.casting_views(conn, policy)
    return (f"Archetype-level hypotheses from our own creators' experiments (replicated_in: creators where it held; "
            f"suggestions about an archetype, never proof for a new creator): {json.dumps(rows, default=str)}"
            if rows else "")


# --- cast: concepts, judge, code scores -----------------------------------------------------------------------------
def develop(conn, policy: Policy, llm, t: dict) -> list[dict]:
    """The concepts (one call)."""
    from . import patterns
    cfg = _cfg(policy)
    n = int(cfg.get("candidates", 4))
    scope = policy.get("scope", default={}) or {}
    cards = lab.casting_cards(conn, policy, t)
    taken = [r["handle"] for r in conn.execute("SELECT handle FROM creators").fetchall()]
    prompt = "\n\n".join([
        _context(conn, policy, t),
        f"The playbook library (name: how content is built): {json.dumps(archetypes.catalogue(policy))}",
        f"Audience propositions (name: why people follow; give each concept a mix, and vary them across concepts): "
        f"{json.dumps(archetypes.proposition_catalogue(policy))}",
        (f"Pattern cards for casting (what makes creators like this worth following; patterns, never a person):\n"
         f"{patterns.brief(cards)}" if cards else ""),
        _hypotheses_text(conn, policy),
        _culture(conn, policy),
        f"Handles already taken: {taken}. Minimum apparent age {scope.get('creator_min_age', 21)}. Never touch: "
        f"{scope.get('excluded_topics')}.",
        f"Develop exactly {n} candidates for this brief, clearly different from each other and each distinct enough "
        f"from the portfolio's creators to own its own audience (sharing a proven archetype with them is fine). "
        f"Each a compact concept: who they are, the desire they embody, their look and styling direction, their "
        f"audience and why it could grow very large (viral potential, years of things to post), the core of their brand "
        f"bible, 2-5 commercial categories (registry keys), their commercial case (why a large audience of theirs "
        f"would be valuable, its signature associations), and a content mix that "
        f"leans on what production does reliably (photography first; a minority of cheap motion for discovery)."])
    out = llm.structured(str(cfg.get("role", "casting")), system=lab.persona_system(policy, SYSTEM), content=prompt,
                         schema=Concepts, effort=str(cfg.get("effort", "high")))
    concepts = [c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in (out.concepts or [])][:len(ARMS)]
    return concepts


def judge(conn, policy: Policy, llm, t: dict, concepts: list[dict]) -> dict:
    """The rubric scores (one call), by candidate number."""
    cfg = _cfg(policy)
    numbered = [{"candidate": i + 1, **{k: v for k, v in c.items()}} for i, c in enumerate(concepts)]
    prompt = "\n\n".join([
        _context(conn, policy, t),
        dossier.desirability_text(policy),
        f"The candidates:\n{json.dumps(numbered, default=str)}",
        "Score every candidate on every criterion (0-10, with the reason), say whether it keeps the guardrail, and "
        "name its generic risk."])
    out = llm.structured(str(cfg.get("judge_role", "casting_judge")), system=JUDGE, content=prompt, schema=Scores,
                         effort=str(cfg.get("judge_effort", "medium")))
    got = {}
    for sc in out.scores or []:
        d = sc.model_dump() if hasattr(sc, "model_dump") else dict(sc)
        got[int(d.get("candidate") or 0)] = d
    return {"by_candidate": got, "note": getattr(out, "note", "") or ""}


def _pseudo_profile(c: dict) -> dict:
    core = c.get("brand_core") or {}
    return {"positioning": {"why_follow": c.get("why_follow", ""), "the_gap": c.get("desire", ""),
                            "point_of_view": core.get("ethos", ""), "signature_angle": c.get("signature", "")},
            "niche": c.get("archetype", ""), "one_line": c.get("one_line", ""),
            "propositions": archetypes.clean_mix(None, c.get("propositions")), "verticals": c.get("verticals") or []}


def score(conn, policy: Policy, concepts: list[dict], judged: dict, references: list[dict]) -> list[dict]:
    """Code's verdict per concept: final criterion scores, total (weighted, 0-10), eligibility and why not."""
    s = policy.get("strategy", default={}) or {}
    rub = s.get("rubric") or {}
    crit = rub.get("criteria") or {}
    card = capability_card.card(conn, policy)
    max_overlap = float(policy.get("brain", "talent_manager", "max_overlap", default=0.35))
    sibs = dossier.siblings(conn, NOBODY)
    lib = set(archetypes.names(policy))
    out = []
    for i, c in enumerate(concepts):
        j = (judged.get("by_candidate") or {}).get(i + 1) or {}
        scores, reasons, why_not = {}, {}, []
        for k in CRITERIA:
            v = j.get(k) or {}
            try:
                scores[k] = max(0.0, min(10.0, float(v.get("score"))))
            except (TypeError, ValueError):
                scores[k] = 0.0
                why_not.append(f"the judge gave no {k} score")
            reasons[k] = v.get("why", "")
        code = {"feasibility": capability_card.feasibility(card, c.get("content_mix")),
                "commercial": commerce.mean_index(conn, policy, c.get("commercial_categories")),
                "delivery": capability_card.delivery(card, c.get("core_formats"))}
        if code["feasibility"] is not None:
            scores["feasibility"] = min(scores["feasibility"], code["feasibility"])
        if code["delivery"] is not None:   # her appeal rests on her core formats: as feasible as the weakest
            scores["feasibility"] = min(scores["feasibility"], code["delivery"]["score"])
        if code["commercial"] is not None:
            scores["commercial"] = round((scores["commercial"] + code["commercial"]) / 2, 2)
        closest = max(((dossier.overlap(_pseudo_profile(c), sb["profile"]), sb["handle"]) for sb in sibs),
                      default=(0.0, None))
        code["closest_sibling"], code["overlap"] = closest[1], closest[0]
        if closest[0] > max_overlap:
            scores["distinctiveness"] = min(scores["distinctiveness"], float(rub.get("too_close_cap", 4)))
            why_not.append(f"too close to {closest[1]} (overlap {closest[0]})")
        for k in CRITERIA:
            floor = float((crit.get(k) or {}).get("floor", 0))
            if scores[k] < floor:
                gate = " (hard gate)" if (crit.get(k) or {}).get("hard_gate") else ""
                why_not.append(f"{k} {scores[k]} under its floor {floor:g}{gate}")
        if j and not j.get("guardrail_ok", False):
            why_not.append(f"restricted content: {j.get('guardrail_note') or 'flagged by the judge'}")
        if not j:
            why_not.append("not scored by the judge")
        if c.get("archetype") not in lib:
            code["archetype_unknown"] = c.get("archetype")
        if int(c.get("apparent_age") or 0) < int(policy.get("scope", "creator_min_age", default=21)):
            why_not.append(f"apparent age {c.get('apparent_age')} under the minimum")
        checks = impersonation.check({"name": c.get("display_name"), "handle": c.get("handle")}, references, policy)
        if checks["blocked"]:
            why_not.append(f"impersonation: {checks['blocks']}")
        weights = {k: float((crit.get(k) or {}).get("weight", 0)) for k in CRITERIA}
        total = round(sum(weights[k] * scores[k] for k in CRITERIA) / (sum(weights.values()) or 1.0), 3)
        out.append({"arm": ARMS[i], "scores": {k: round(v, 2) for k, v in scores.items()}, "reasons": reasons,
                    "total": total, "eligible": not why_not, "why_not": why_not, "code": code, "checks": checks,
                    "generic_risk": j.get("generic_risk", ""), "guardrail_ok": bool(j.get("guardrail_ok", False))})
    return out


def cast(conn, policy: Policy, llm, thesis_id, *, run: str, launch_id=None) -> dict:
    """Concepts, judge and code scores into `auditions` rows (once per run: a rerun returns what exists)."""
    existing = conn.execute("SELECT id, arm, stage, scores FROM auditions WHERE run = %s ORDER BY arm", (run,)).fetchall()
    if existing:
        return {"run": run, "auditions": [{"id": str(r["id"]), "arm": r["arm"], "stage": r["stage"]} for r in existing],
                "again": True}
    t = thesis(conn, thesis_id)
    concepts = develop(conn, policy, llm, t)
    if not concepts:
        raise ValueError("the casting call returned no concepts")
    judged = judge(conn, policy, llm, t, concepts)
    refs = _references(conn, t)
    verdicts = score(conn, policy, concepts, judged, refs)
    top = int(_cfg(policy).get("audition_top", 2))
    ranked = sorted((v for v in verdicts if v["eligible"]), key=lambda v: -v["total"])
    shortlisted = {v["arm"] for v in ranked[:top]}
    rows = []
    for c, v in zip(concepts, verdicts):
        stage = "shortlisted" if v["arm"] in shortlisted else "not_shortlisted"
        rid = conn.execute(
            """INSERT INTO auditions (thesis_id, arm, candidate, "references", checks, scores, decision, run, launch_id,
                                      stage, est_usd)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (t["id"], v["arm"], Jsonb(c), Jsonb([r["id"] for r in refs]), Jsonb(v["checks"]),
             Jsonb({k: v[k] for k in ("scores", "reasons", "total", "eligible", "why_not", "code", "generic_risk",
                                      "guardrail_ok")}),
             "pending" if stage == "shortlisted" else "reject", run, launch_id, stage,
             estimate_audition(policy) if stage == "shortlisted" else None)).fetchone()["id"]
        rows.append({"id": str(rid), "arm": v["arm"], "stage": stage, "total": v["total"], "why_not": v["why_not"]})
    conn.execute("INSERT INTO decisions (kind, inputs, output, rationale, policy_version) VALUES "
                 "('casting', %s, %s, %s, %s)",
                 (Jsonb({"thesis_id": t["id"], "run": run, "launch_id": str(launch_id) if launch_id else None}),
                  Jsonb({"candidates": [{"arm": v["arm"], "name": c.get("display_name"), "total": v["total"],
                                         "eligible": v["eligible"], "why_not": v["why_not"]}
                                        for c, v in zip(concepts, verdicts)],
                         "shortlisted": sorted(shortlisted), "judge_note": judged.get("note")}),
                  f"{len(shortlisted)} of {len(concepts)} candidates shortlisted for auditions", policy.version))
    return {"run": run, "auditions": rows}


# --- audition steps (model calls) -----------------------------------------------------------------------------------
def audition_bible(conn, policy: Policy, llm, audition_id) -> dict:
    """The bible and mini-feed for a shortlisted concept, its sandbox creator and its reference pack (stage
    references). Idempotent: an audition that has its creator only queues what is missing."""
    a = conn.execute("SELECT * FROM auditions WHERE id = %s", (audition_id,)).fetchone()
    if a["creator_id"] is None:
        acfg = _acfg(policy)
        posts = int(acfg.get("posts", 9))
        refs = int(acfg.get("reference_prompts", 6))
        scope = policy.get("scope", default={}) or {}
        taken = [r["handle"] for r in conn.execute("SELECT handle FROM creators").fetchall()]
        prompt = (f"The concept to write up (keep it: who they are, look, styling, audience and brand core):\n"
                  f"{json.dumps(a['candidate'], default=str)}\n\n{dossier.strategy_text(policy)}\n\n"
                  f"Minimum apparent age {scope.get('creator_min_age', 21)}. Off-limits topics: "
                  f"{scope.get('excluded_topics')}. Disclosure must say '{scope.get('disclosure_line')}'. Handles "
                  f"already taken: {taken}. Kokoro voices to choose from: {lab.KOKORO_VOICES}. Write exactly {refs} "
                  f"reference_prompts, and a feed of exactly {posts} photo scenes: her world at its best, the kind of "
                  f"feed that would make a stranger follow (portraits, places, details, style), photography only.")
        draft = llm.structured("creator", system=lab.persona_system(policy, lab.SYSTEM), content=prompt,
                               schema=AuditionDraft, effort="high")
        bible = lab.check_bible(policy, draft.bible)
        row = lab.create_creator(conn, bible, a["thesis_id"], policy_version=policy.version,
                                 voice_blend=policy.get("creator_lab", "voice_blend"), link_thesis=False,
                                 anchors={"simulation": {"audition": str(a["id"])},
                                          "audition": {"id": str(a["id"]), "run": a["run"]}})
        feed = [str(x) for x in (draft.feed or [])][:posts]
        conn.execute("UPDATE auditions SET creator_id = %s, candidate = candidate || %s, updated_at = now() "
                     "WHERE id = %s", (row["id"], Jsonb({"feed": feed}), a["id"]))
        a = conn.execute("SELECT * FROM auditions WHERE id = %s", (audition_id,)).fetchone()
    acfg = _acfg(policy)
    lab.enqueue_reference_pack(conn, policy, a["creator_id"], batch=refs_batch(policy, a["run"], a["arm"]),
                               limit=int(acfg.get("reference_prompts", 6)),
                               per_prompt=int(acfg.get("candidates_per_reference", 2)), key="audition-ref")
    _stage(conn, a["id"], "references")
    return {"creator_id": str(a["creator_id"]), "stage": "references"}


def audition_anchors(conn, policy: Policy, llm, store, audition_id, workdir) -> dict:
    """Anchors from the audition's pack, then the mini-feed's posts through normal production (stage feed)."""
    from .. import production
    a = conn.execute("SELECT * FROM auditions WHERE id = %s", (audition_id,)).fetchone()
    anchors = (conn.execute("SELECT anchors FROM creators WHERE id = %s", (a["creator_id"],)).fetchone()["anchors"]
               or {})
    if not anchors.get("reference_images"):
        picked = lab.select_anchors(conn, policy, llm, store, a["creator_id"], workdir, key="audition-ref")
        if not picked.get("selected"):
            return _judged(conn, policy, a, reason="no usable reference image")
    b = batch(policy, a["run"], a["arm"])
    made = 0
    for i, scene in enumerate((a["candidate"] or {}).get("feed") or []):
        key = f"{a['id']}:{i}"
        post = conn.execute("SELECT * FROM posts WHERE body->>'audition_post' = %s", (key,)).fetchone()
        if post is None:
            body = {"scenes": [scene], "hashtags": [], "platform_ai_label": True, "pillar": "audition",
                    "breadth_run": f"casting-{a['run']}", "audition_id": str(a["id"]), "audition_post": key}
            post = conn.execute("INSERT INTO posts (creator_id, platform, format, status, caption, body, scheduled_for) "
                                "VALUES (%s, 'instagram', 'photo', 'planned', '', %s, NULL) RETURNING *",
                                (a["creator_id"], Jsonb(body))).fetchone()
        if post["status"] == "planned":
            production.start(conn, policy, post)
            made += 1
    _adopt(conn, a["creator_id"], b)
    conn.execute("UPDATE auditions SET scores = scores || %s WHERE id = %s",
                 (Jsonb({"feed_started_at": datetime.now(timezone.utc).isoformat()}), a["id"]))
    _stage(conn, a["id"], "feed")
    return {"posts_started": made, "batch": b}


# --- the code that moves auditions ---------------------------------------------------------------------------------
def _stage(conn, audition_id, stage: str) -> None:
    conn.execute("UPDATE auditions SET stage = %s, updated_at = now() WHERE id = %s", (stage, audition_id))


def _adopt(conn, creator_id, b: str) -> int:
    """The audition's queued post jobs into its test batch (production queues them in post-<id> batches)."""
    return conn.execute(
        "UPDATE jobs SET batch = %s, updated_at = now() WHERE status = 'queued' AND creator_id = %s "
        "AND post_id IS NOT NULL AND coalesce(batch, '') <> %s", (b, creator_id, b)).rowcount


def _posts(conn, audition_id) -> list[dict]:
    return conn.execute("SELECT * FROM posts WHERE body->>'audition_id' = %s ORDER BY body->>'audition_post'",
                        (str(audition_id),)).fetchall()


def _cost(conn, creator_id) -> float:
    if creator_id is None:
        return 0.0
    return float(conn.execute(
        "SELECT coalesce(sum(usd), 0) AS u FROM cost_entries WHERE creator_id = %s "
        "OR job_id IN (SELECT id FROM jobs WHERE creator_id = %s)", (creator_id, creator_id)).fetchone()["u"])


def _judged(conn, policy: Policy, a: dict, *, reason: str | None = None) -> dict:
    """Acceptance and cost of a finished (or stopped) audition."""
    ok, bad = capability_card._statuses(policy)
    posts = _posts(conn, a["id"])
    planned = int(_acfg(policy).get("posts", 9))
    accepted = sum(p["status"] in ok for p in posts)
    usd = _cost(conn, a["creator_id"])
    res = {"posts": len(posts), "accepted": accepted, "dropped": sum(p["status"] in bad for p in posts),
           "acceptance": round(accepted / max(len(posts), planned, 1), 3), "usd": round(usd, 4),
           "usd_per_accepted": round(usd / accepted, 4) if accepted else None}
    if reason:
        res["stopped"] = reason
    conn.execute("UPDATE auditions SET stage = 'judged', actual_usd = %s, scores = scores || %s, updated_at = now() "
                 "WHERE id = %s", (round(usd, 4), Jsonb({"audition": res}), a["id"]))
    # posts still in production are left where they are; their jobs are cancelled so no machine renders them
    for p in posts:
        if p["status"] in ("planned", "producing"):
            conn.execute("UPDATE posts SET status = 'dropped', body = body || %s, updated_at = now() WHERE id = %s",
                         (Jsonb({"drop_reason": "casting audition closed"}), p["id"]))
    if a["creator_id"]:
        queue.cancel(conn, [r["id"] for r in conn.execute(
            "SELECT id FROM jobs WHERE creator_id = %s AND status = 'queued'", (a["creator_id"],)).fetchall()],
            "casting audition closed")
    return {"stage": "judged", **res}


def _job_state(conn, kind: str, key: str) -> str | None:
    r = conn.execute("SELECT status FROM jobs WHERE kind = %s AND idempotency_key = %s", (kind, key)).fetchone()
    return r["status"] if r else None


def _llm_step(conn, policy, a: dict, step: str, *, llm=None, store=None, workdir=None) -> dict:
    """A model step: inline with an llm (the sandbox driver), else a brain.creator_lab job (once; a failed job closes
    the audition)."""
    if llm is not None:
        if step == "audition":
            return audition_bible(conn, policy, llm, a["id"])
        return audition_anchors(conn, policy, llm, store, a["id"], workdir)
    key = f"casting:{a['id']}:{step}"
    state = _job_state(conn, "brain.creator_lab", key)
    if state is None:
        queue.enqueue(conn, "brain.creator_lab", {"audition_id": str(a["id"]), "step": step},
                      creator_id=a["creator_id"], idempotency_key=key)
        return {"queued": step}
    if state in ("failed", "dead"):
        return _judged(conn, policy, a, reason=f"{step} job {state}")
    return {"waiting": step}


def advance(conn, policy: Policy, *, run: str | None = None, llm=None, store=None, workdir=None,
            now: datetime | None = None) -> dict:
    """Move every open audition (of `run`, else all) one step, then pick the winner of each run whose auditions are
    all judged. Returns what happened."""
    from .. import production
    from ..production.locks import post_lease
    now = now or datetime.now(timezone.utc)
    acfg = _acfg(policy)
    out: dict = {"moved": [], "picked": []}
    rows = conn.execute("SELECT * FROM auditions WHERE stage = ANY(%s) AND (%s::text IS NULL OR run = %s) "
                        "ORDER BY run, arm", (list(OPEN), run, run)).fetchall()
    for a in rows:
        st = a["stage"]
        if st == "shortlisted" or (st == "bible" and llm is not None):
            if llm is None:
                _stage(conn, a["id"], "bible")
            out["moved"].append({"arm": a["arm"], **_llm_step(conn, policy, a, "audition", llm=llm)})
        elif st == "bible":
            out["moved"].append({"arm": a["arm"], **_llm_step(conn, policy, a, "audition")})
        elif st == "references":
            jobs = conn.execute("SELECT status FROM jobs WHERE batch = %s", (refs_batch(policy, a["run"], a["arm"]),)
                                ).fetchall()
            if any(j["status"] in ("queued", "running") for j in jobs) or not jobs:
                continue
            done = sum(j["status"] == "done" for j in jobs)
            if done < int(acfg.get("min_reference_renders", 4)):
                out["moved"].append({"arm": a["arm"], **_judged(conn, policy, a, reason=f"{done} reference renders")})
                continue
            _stage(conn, a["id"], "anchors")
            out["moved"].append({"arm": a["arm"], **_llm_step(conn, policy, {**a, "stage": "anchors"},
                                                              "audition_anchors", llm=llm, store=store,
                                                              workdir=workdir)})
        elif st == "anchors":
            out["moved"].append({"arm": a["arm"], **_llm_step(conn, policy, a, "audition_anchors", llm=llm,
                                                              store=store, workdir=workdir)})
        elif st == "feed":
            b = batch(policy, a["run"], a["arm"])
            for p in _posts(conn, a["id"]):
                if p["status"] != "producing":
                    continue
                with post_lease(conn, policy, p["id"]) as held:
                    if not held:
                        continue
                    row = conn.execute("SELECT * FROM posts WHERE id = %s AND status = 'producing'",
                                       (p["id"],)).fetchone()
                    if row is not None:
                        try:
                            production.advance_post(conn, policy, row, store)
                        except Exception:  # noqa: BLE001 - one broken audition post never stops the others
                            log.exception("advancing audition post %s failed", p["id"])
            _adopt(conn, a["creator_id"], b)
            posts = _posts(conn, a["id"])
            started = (a["scores"] or {}).get("feed_started_at")
            late = started and now - datetime.fromisoformat(started) > timedelta(
                hours=float(acfg.get("deadline_hours", 24)))
            if posts and all(p["status"] not in ("planned", "producing") for p in posts):
                out["moved"].append({"arm": a["arm"], **_judged(conn, policy, a)})
            elif late:
                out["moved"].append({"arm": a["arm"], **_judged(conn, policy, a, reason="deadline")})
    runs = {r["run"] for r in rows} | ({run} if run else set())
    for r in sorted(x for x in runs if x):
        got = pick(conn, policy, r)
        if got and got.get("picked_now"):
            out["picked"].append(got)
    return out


def pick(conn, policy: Policy, run: str) -> dict | None:
    """The run's winner once every shortlisted audition is judged (None while one is still open). Idempotent."""
    rows = conn.execute("SELECT * FROM auditions WHERE run = %s ORDER BY arm", (run,)).fetchall()
    if not rows or any(r["stage"] in OPEN for r in rows):
        return None
    chosen = next((r for r in rows if r["decision"] == "select"), None)
    if chosen is not None:
        return {"run": run, "winner": str(chosen["id"]), "creator_id": str(chosen["creator_id"]), "picked_now": False}
    pcfg = _cfg(policy).get("pick") or {}
    min_acc = float(pcfg.get("min_acceptance", 0.5))
    ref, floor = float(pcfg.get("cost_ref_usd", 0.05)), float(pcfg.get("cost_floor_usd", 0.01))
    ranked = []
    for r in rows:
        if r["stage"] != "judged":
            continue
        s = r["scores"] or {}
        aud = s.get("audition") or {}
        acc = float(aud.get("acceptance") or 0)
        if acc < min_acc or r["creator_id"] is None:
            continue
        per = max(float(aud.get("usd_per_accepted") or ref), floor)
        value = float(s.get("total") or 0) * acc ** float(pcfg.get("acceptance_weight", 1.0)) * \
            (ref / per) ** float(pcfg.get("cost_weight", 0.3))
        ranked.append((round(value, 4), r))
    ranked.sort(key=lambda x: (-x[0], x[1]["arm"]))
    winner = ranked[0][1] if ranked else None
    for r in rows:
        if r["stage"] == "judged":
            conn.execute("UPDATE auditions SET decision = %s, stage = 'done', updated_at = now() WHERE id = %s",
                         ("select" if winner is not None and r["id"] == winner["id"] else "reject", r["id"]))
    conn.execute("INSERT INTO decisions (kind, creator_id, inputs, output, rationale, policy_version) VALUES "
                 "('casting_pick', %s, %s, %s, %s, %s)",
                 (winner["creator_id"] if winner else None, Jsonb({"run": run}),
                  Jsonb({"ranking": [{"arm": r["arm"], "value": v, "total": (r["scores"] or {}).get("total"),
                                      "audition": (r["scores"] or {}).get("audition")} for v, r in ranked],
                         "winner": winner["arm"] if winner else None}),
                  f"winner {winner['arm']}" if winner else "no audition reached the minimum acceptance",
                  policy.version))
    return {"run": run, "winner": str(winner["id"]) if winner else None,
            "creator_id": str(winner["creator_id"]) if winner else None, "picked_now": True}


def promote(conn, policy: Policy, audition_id, thesis_id) -> dict:
    """The winner becomes the launched creator: linked to the thesis, no longer an audition, her brand core and
    concept in the profile's seed; the other auditions' creators retire."""
    a = conn.execute("SELECT * FROM auditions WHERE id = %s", (audition_id,)).fetchone()
    cid = a["creator_id"]
    c = conn.execute("SELECT handle, anchors FROM creators WHERE id = %s", (cid,)).fetchone()
    anchors = {k: v for k, v in (c["anchors"] or {}).items() if k not in ("simulation", "audition")}
    anchors["cast_from"] = {"audition_id": str(a["id"]), "run": a["run"]}
    anchors.pop("reference_images", None)   # the launch's full pack and anchor pick replace the audition's
    conn.execute("UPDATE creators SET anchors = %s WHERE id = %s", (Jsonb(anchors), cid))
    lab.link(conn, cid, c["handle"], thesis_id, policy_version=policy.version)
    concept = a["candidate"] or {}
    conn.execute("UPDATE creators SET bible = bible || %s WHERE id = %s",
                 (Jsonb({"casting": {k: concept.get(k) for k in ("desire", "signature", "audience", "styling",
                                                                 "commercial_categories", "brand_core")}}), cid))
    for other in conn.execute("SELECT creator_id FROM auditions WHERE run = %s AND id <> %s AND creator_id IS NOT NULL",
                              (a["run"], a["id"])).fetchall():
        conn.execute("UPDATE creators SET status = 'retired' WHERE id = %s", (other["creator_id"],))
    return {"creator_id": str(cid), "handle": c["handle"]}


def seed_profile(conn, policy: Policy, creator_id) -> bool:
    """After the launch builds profile v1: the cast brand core goes in as is (the talent manager then builds only the
    sections still missing), so the bible the audition proved is the one she keeps."""
    from ..creator import profile as profile_mod
    bible = (conn.execute("SELECT bible FROM creators WHERE id = %s", (creator_id,)).fetchone() or {}).get("bible") or {}
    core = (bible.get("casting") or {}).get("brand_core")
    cur = profile_mod.current(conn, creator_id)
    if not core or not cur or cur[1].get("brand_core"):
        return False
    profile_mod.update(conn, policy, creator_id, {"brand_core": core}, author="casting", reason="brand core from casting")
    return True


# --- estimate -------------------------------------------------------------------------------------------------------
def estimate_audition(policy: Policy) -> float:
    cfg, acfg = _cfg(policy), _acfg(policy)
    e = cfg.get("estimate") or {}
    refs = int(acfg.get("reference_prompts", 6)) * int(acfg.get("candidates_per_reference", 2))
    images = int(acfg.get("posts", 9)) * int(policy.get("production", "candidates_per_scene", default=4))
    return round(float(e.get("bible_call", 0)) + int(acfg.get("reference_prompts", 6)) * float(e.get("anchor_call", 0))
                 + (refs + images) * float(e.get("image", 0)) + images * float(e.get("image_review", 0)), 4)


def estimate(policy: Policy) -> dict:
    """What one casting costs at most on its first try (the dry run's answer): calls and renders, from policy."""
    cfg = _cfg(policy)
    e = cfg.get("estimate") or {}
    top = int(cfg.get("audition_top", 2))
    lines = {"concepts_call": float(e.get("concepts_call", 0)), "judge_call": float(e.get("judge_call", 0)),
             "auditions": round(top * estimate_audition(policy), 4)}
    return {"candidates": int(cfg.get("candidates", 4)), "auditions": top,
            "posts_per_audition": int(_acfg(policy).get("posts", 9)), "lines": lines,
            "total_usd": round(sum(lines.values()), 2)}


# --- launches ------------------------------------------------------------------------------------------------------
def launch_run(launch) -> str:
    return f"l{str(launch['id'])[:8]}"


def launch_tick(conn, policy: Policy, launch: dict) -> tuple[str, str | None]:
    """A launch at step `casting`: ('waiting', None), ('done', creator_id) or ('failed', why)."""
    run = launch_run(launch)
    key = f"launch:{launch['id']}:casting"
    if not conn.execute("SELECT 1 FROM auditions WHERE run = %s", (run,)).fetchone():
        state = _job_state(conn, "brain.creator_lab", key)
        if state is None:
            queue.enqueue(conn, "brain.creator_lab", {"launch_id": str(launch["id"]), "step": "casting"},
                          idempotency_key=key)
            return "waiting", None
        if state in ("failed", "dead"):
            return "failed", f"the casting job {state}"
        return "waiting", None
    advance(conn, policy, run=run)
    got = pick(conn, policy, run)
    if got is None:
        return "waiting", None
    if not got.get("winner"):
        return "failed", "no audition passed (rubric floors, guardrail, or minimum acceptance)"
    win = promote(conn, policy, got["winner"], launch["thesis_id"]) if launch.get("creator_id") is None else \
        {"creator_id": str(launch["creator_id"])}
    return "done", win["creator_id"]


def summary(conn, run: str) -> dict:
    """A run's auditions: arm, concept name, stage, decision, rubric total, acceptance and cost."""
    rows = conn.execute("SELECT * FROM auditions WHERE run = %s ORDER BY arm", (run,)).fetchall()
    return {"run": run, "auditions": [
        {"arm": r["arm"], "name": (r["candidate"] or {}).get("display_name"), "stage": r["stage"],
         "decision": r["decision"], "total": (r["scores"] or {}).get("total"),
         "why_not": (r["scores"] or {}).get("why_not"), "audition": (r["scores"] or {}).get("audition"),
         "creator_id": str(r["creator_id"]) if r["creator_id"] else None,
         "usd": float(r["actual_usd"]) if r["actual_usd"] is not None else None} for r in rows]}



def _culture(conn, policy: Policy) -> str:
    """Culture signals for casting and commerce (manyworlds.research.desk): context, not evidence of what performs."""
    from ..research import desk
    lines = "\n".join(x for x in (desk.brief(conn, policy, use="casting"), desk.brief(conn, policy, use="commerce"))
                      if x)
    return (f"What is moving in culture now (research desk findings from current sources, with their evidence mix; "
            f"our own synthesis, context not proof):\n{lines}" if lines else "")
