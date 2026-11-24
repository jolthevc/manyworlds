"""A&R (spec section 3): original artist concepts from public, logged-out research (findings read through
brain.evidence: real only) and the gaps in our roster (brain.music_portfolio). Each pitch is filed as an
`artist_concept` request to head_of_music (brain.proposals); one head_of_music accepts becomes a thesis, which is
launched only through the strategist's launch slots and the owner's approval, like every creator.

Never "the next <real artist>": a pitch naming a real artist, or pointing at someone else's work, is dropped in code
(music_roles.word_issues). Concepts are adults only (policy creator_min_age) and original people.
`brain.a_and_r` ({}); it runs only when commissioned (nothing schedules it while music is off)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .. import evidence, music_portfolio, proposals
from ..agent import AgentContext, register
from ..music_roles import StrictInput, ask, cfg, clean_words

NAME = "a_and_r"

SYSTEM = """You are A&R for an AI music studio whose artists are openly AI-generated original people (adults, never
based on anyone real). From the research findings (public data: what is rising) and the gaps in our roster, pitch
original artist concepts: genre and subgenre, a personality, a sonic identity in words, her story, why now (cite the
finding ids you rely on) and why she is distinct from our artists. Never pitch "the next <someone>", never name a
real artist, band or song, never "in the style of". Fewer, sharper pitches beat many."""


class Input(StrictInput):
    findings: list[dict] = Field(default_factory=list, description="research findings: id, statement, grade")
    roster: list[dict] = Field(default_factory=list)
    gaps: list[dict] = Field(default_factory=list)
    max_pitches: int = 3
    min_age: int = 18


class Pitch(BaseModel):
    concept: str = ""
    genre: str = ""
    subgenre: str = ""
    personality: str = ""
    sonic_identity: str = ""
    story: str = ""
    why_now: str = ""
    why_distinct: str = ""
    finding_ids: list[str] = Field(default_factory=list)


class Output(BaseModel):
    pitches: list[Pitch] = Field(default_factory=list)
    why: str = ""


def pitch(ctx: AgentContext, *, step: str = "pitch") -> dict:
    conn, policy = ctx.conn, ctx.policy
    c = cfg(policy, NAME)
    found = [f for f in evidence.findings(conn, mode="real") if f.get("evidence") == "research"
             or f.get("domain") == "music"][-int(c.get("max_findings", 30)):]
    snap = music_portfolio.snapshot(conn, policy)
    inp = Input(findings=[{"id": str(f["id"]), "statement": f["statement"], "grade": f["grade"]} for f in found],
                roster=[{"handle": a["handle"], "genre": a["genre"]} for a in snap["artists"]], gaps=snap["gaps"],
                max_pitches=int(c.get("max_pitches", 3)),
                min_age=int(policy.get("scope", "creator_min_age", default=21)))
    out = ask(ctx, NAME, system=SYSTEM, inp=inp, out=Output, creator_id=None, step=step)
    known = {str(f["id"]) for f in found}
    filed, dropped = [], []
    for p in out.pitches[:inp.max_pitches]:
        problems = clean_words(policy, p.model_dump())
        if problems or not p.concept:
            dropped.append({"concept": p.concept, "why": problems or ["no concept"]})
            continue
        body = {**p.model_dump(), "finding_ids": [i for i in p.finding_ids if i in known]}
        filed.append(proposals.file(conn, policy, kind="artist_concept", from_role=NAME, subject=body,
                                    argument=p.why_now or p.concept, evidence_={"finding_ids": body["finding_ids"]},
                                    job_id=ctx.job.id))
    return {"filed": filed, "dropped": dropped}


class AandR:
    name = NAME

    def run(self, ctx: AgentContext) -> dict:
        return pitch(ctx)


register(AandR())
