"""Brain step 1: decide what to research, pull it, and propose creator opportunities.

    plan_research -> run_research -> write_theses -> choose_thesis

Every step is a recorded decision (inputs, output, model, policy version), so later
days can learn which research choices led to creators that worked.
"""

from __future__ import annotations

import json
from typing import Literal

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from ..llm import StructuredLLM
from ..policy import Policy
from ..research import signals
from ..research.apify import ApifySource

SYSTEM = """You are the research lead of Manyworlds, a studio that runs a portfolio of openly
AI-generated social media creators. Every creator is clearly labeled as AI. The goal is many
inexpensive, well-chosen experiments: find formats and archetypes with real audience pull, then
design original fictional creators that use those transferable patterns without copying any real
person's identity, face, voice or distinctive posts. Be concrete and numbers-first, and say when
evidence is thin."""


class PlatformTags(BaseModel):
    platform: Literal["instagram", "tiktok"]
    hashtags: list[str] = Field(description="hashtags without '#', lowercase")


class ResearchPlan(BaseModel):
    rationale: str
    plans: list[PlatformTags]


class Thesis(BaseModel):
    title: str = Field(description="short name for the opportunity")
    archetype: str = Field(description="the creator archetype being tested, e.g. 'cozy city-apartment routines'")
    audience: str
    why_now: str = Field(description="evidence from the research numbers")
    format_mix: list[str] = Field(description="post formats in priority order, e.g. 'photo dump', 'talking-head story'")
    differentiation: str = Field(description="what makes it original rather than a copy of existing creators")
    risks: str
    production_fit: str = Field(description="how well cheap image/short-video generation can execute it")
    score: float = Field(description="0-10 expected value of testing this, all things considered")


class ThesisSet(BaseModel):
    theses: list[Thesis]


class ThesisChoice(BaseModel):
    chosen_index: int = Field(description="0-based index into the candidate list")
    reasoning: str


def _record(conn, kind: str, inputs: dict, output: dict, model: str, policy: Policy, creator_id=None):
    conn.execute(
        "INSERT INTO decisions (kind, creator_id, inputs, output, model, policy_version) VALUES (%s,%s,%s,%s,%s,%s)",
        (kind, creator_id, Jsonb(inputs), Jsonb(output), model, policy.version))


def _portfolio(conn) -> list[dict]:
    return conn.execute(
        "SELECT handle, status, bible->>'archetype' AS archetype, bible->>'niche' AS niche FROM creators"
    ).fetchall()


def plan_research(conn, policy: Policy, llm: StructuredLLM) -> ResearchPlan:
    scope = policy.get("scope", default={})
    n = policy.get("research", "hashtags_per_platform", default=6)
    prompt = (
        f"Scope: {scope.get('niche')}\nPlatforms: {scope.get('platforms')}\n"
        f"Never touch: {scope.get('excluded_topics')}\n"
        f"Existing creators in the portfolio: {json.dumps(_portfolio(conn), default=str)}\n\n"
        f"Choose {n} hashtags per platform to research. Mix proven high-volume lifestyle tags with "
        f"a few rising or underserved ones, and avoid ones that overlap the existing portfolio. "
        f"Return lowercase tags without '#'.")
    plan = llm.structured("research", system=SYSTEM, content=prompt, schema=ResearchPlan, effort="medium")
    for p in plan.plans:
        p.hashtags = [h.strip().lstrip("#").lower() for h in p.hashtags if h.strip()][:n]
    _record(conn, "research_plan", {"scope": scope}, plan.model_dump(), getattr(llm, "model_for", lambda r: "?")("research"), policy)
    return plan


def run_research(conn, policy: Policy, plan: ResearchPlan, source: ApifySource) -> list[dict]:
    per_tag = int(policy.get("research", "items_per_hashtag", default=30))
    budget = int(policy.get("research", "max_items_per_run", default=400))
    results, pulled = [], 0
    for p in plan.plans:
        for tag in p.hashtags:
            if pulled + per_tag > budget:
                results.append({"platform": p.platform, "tag": tag, "skipped": "run item budget reached"})
                continue
            r = source.pull(p.platform, tag, per_tag)
            pulled += r.get("items", 0)
            results.append(r)
    return results


def write_theses(conn, policy: Policy, llm: StructuredLLM) -> list[dict]:
    summary = signals.summarize(conn)
    k = int(policy.get("research", "theses_per_run", default=5))
    scope = policy.get("scope", default={})
    prompt = (
        f"Research summary (last 30 days, anonymised):\n{json.dumps(summary, default=str)}\n\n"
        f"Existing creators: {json.dumps(_portfolio(conn), default=str)}\n"
        f"Scope: {scope.get('niche')}. Never touch: {scope.get('excluded_topics')}.\n\n"
        f"Propose {k} distinct creator opportunities to test. Each must be an original fictional "
        f"adult creator archetype, executable with AI photos and short videos, labeled as AI. "
        f"Ground 'why_now' in the numbers above. Score honestly; spread the bets across different "
        f"audiences rather than five variations of one idea.")
    theses = llm.structured("research", system=SYSTEM, content=prompt, schema=ThesisSet, effort="high").theses
    rows = []
    for t in theses:
        row = conn.execute(
            "INSERT INTO theses (summary, evidence, score, body) VALUES (%s, %s, %s, %s) RETURNING id",
            (t.title, Jsonb({"research_total_posts": summary["total_posts"]}), t.score, Jsonb(t.model_dump())),
        ).fetchone()
        rows.append({"id": row["id"], **t.model_dump()})
    return rows


def choose_thesis(conn, policy: Policy, llm: StructuredLLM) -> dict:
    cands = conn.execute(
        "SELECT id, body FROM theses WHERE status = 'proposed' ORDER BY score DESC NULLS LAST LIMIT 10"
    ).fetchall()
    if not cands:
        raise LookupError("no proposed theses; run discovery first")
    listing = [{"index": i, **c["body"]} for i, c in enumerate(cands)]
    prompt = (f"Candidate opportunities:\n{json.dumps(listing, default=str)}\n\n"
              f"Existing creators: {json.dumps(_portfolio(conn), default=str)}\n"
              "Pick the one to launch next: best expected learning per dollar, most different from "
              "what the portfolio already tests, and most executable with cheap generation.")
    choice = llm.structured("brain", system=SYSTEM, content=prompt, schema=ThesisChoice, effort="medium")
    idx = min(max(choice.chosen_index, 0), len(cands) - 1)
    chosen = cands[idx]
    conn.execute("UPDATE theses SET status = 'chosen' WHERE id = %s", (chosen["id"],))
    _record(conn, "thesis_choice", {"candidates": [str(c["id"]) for c in cands]},
            {"chosen": str(chosen["id"]), "reasoning": choice.reasoning},
            getattr(llm, "model_for", lambda r: "?")("brain"), policy)
    return {"id": chosen["id"], "reasoning": choice.reasoning, **chosen["body"]}


def discover(conn, policy: Policy, llm: StructuredLLM, source: ApifySource) -> dict:
    plan = plan_research(conn, policy, llm)
    pulls = run_research(conn, policy, plan, source)
    theses = write_theses(conn, policy, llm)
    return {"plan": plan.model_dump(), "pulls": pulls, "theses": theses}
