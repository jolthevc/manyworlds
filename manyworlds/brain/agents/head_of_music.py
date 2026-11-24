"""Head of music (spec sections 3 and 6): the music slate under the CEO. Reads the music portfolio (artists, genres,
feature-space distance between artists, gaps, the music budget: brain.music_portfolio), the portfolio-wide music
findings, the scorecard of the music roles' predictions and the artist concepts A&R filed. Decides:

  - genre priorities (where to grow, what to leave), each with why;
  - the budget split of the music share across artists (shares, normalised in code; the music share itself is
    policy music.portfolio.budget_share of the month's usable budget: the CEO's allocation, never raised here);
  - each artist's release cadence (songs per week, capped at music.roles.head_of_music.max_songs_per_week);
  - A&R's concepts: accept or reject (brain.proposals.decide as head_of_music; an accepted one becomes a thesis that
    launches only through the strategist and the owner's approval);
  - tier changes for artists, filed to the strategist as requests (brain.proposals; rights in brain.rights).

Recorded as one decision (music_head_of_music) with what was applied and refused. `brain.head_of_music` ({}).
Real evidence only for money: the portfolio's outcomes are real; sandbox artists appear only in sandbox runs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .. import music_analysis, music_portfolio, predictions, proposals
from ..agent import AgentContext, register
from ..music_roles import ROLES, StrictInput, ask, cfg, clean_words

NAME = "head_of_music"

SYSTEM = """You run the music department of an AI studio whose artists are openly AI-generated original people.
Under the CEO's allocation you set the music slate: which genres to grow and why, how the music budget is split
across artists, each artist's release cadence (songs per week), decisions on A&R's artist concepts (accept only
originals that fill a gap and are distinct from our artists; give the reason), and production tier changes to ask
the CEO for. The numbers are real outcomes only; with little data, explore broadly and say so."""


class GenrePriority(BaseModel):
    genre: str = ""
    move: str = Field(default="hold", description="grow, hold or leave")
    why: str = ""


class Share(BaseModel):
    creator_id: str = ""
    share: float = 0.0


class Cadence(BaseModel):
    creator_id: str = ""
    songs_per_week: float = 1.0


class ConceptDecision(BaseModel):
    proposal_id: str = ""
    accept: bool = False
    reason: str = ""


class TierAsk(BaseModel):
    creator_id: str = ""
    tier: str = ""
    argument: str = ""


class Input(StrictInput):
    portfolio: dict
    findings: list[str] = Field(default_factory=list)
    scorecard: list[dict] = Field(default_factory=list)
    concepts: list[dict] = Field(default_factory=list, description="open artist_concept requests to decide")
    tiers: list[str] = Field(default_factory=list)


class Slate(BaseModel):
    genre_priorities: list[GenrePriority] = Field(default_factory=list)
    budget_split: list[Share] = Field(default_factory=list)
    cadence: list[Cadence] = Field(default_factory=list)
    concepts: list[ConceptDecision] = Field(default_factory=list)
    tier_requests: list[TierAsk] = Field(default_factory=list)
    why: str = ""


def plan(ctx: AgentContext, *, include_sandbox: bool = False, step: str = "slate") -> dict:
    conn, policy = ctx.conn, ctx.policy
    c = cfg(policy, NAME)
    snap = music_portfolio.snapshot(conn, policy, include_sandbox=include_sandbox)
    open_concepts = proposals.open_for(conn, NAME, limit=int(c.get("max_concepts", 10)))
    roles = [r for r in ROLES if r != NAME]
    card = [r for r in predictions.scorecard(conn, mode="real") if r["role"] in roles]
    found = [f["statement"] for f in music_analysis.findings(conn, mode="real", scope="portfolio")] + \
        [f["statement"] for f in music_analysis.findings(conn, mode="real", scope="genre")]
    tiers = sorted(((policy.get("brain", "tiers", default={}) or {}).get("levels") or {}))
    inp = Input(portfolio=snap, findings=found, scorecard=card, tiers=tiers,
                concepts=[{"proposal_id": p["id"], "subject": p["subject"], "argument": p["argument"]}
                          for p in open_concepts])
    out = ask(ctx, NAME, system=SYSTEM, inp=inp, out=Slate, creator_id=None, step=step)
    artists = {a["creator_id"] for a in snap["artists"]}
    shares = {s.creator_id: max(float(s.share), 0.0) for s in out.budget_split if s.creator_id in artists}
    total = sum(shares.values())
    music_usd = float(snap["budget"]["usd"])
    split = [{"creator_id": k, "share": round(v / total, 3), "usd": round(music_usd * v / total, 2)}
             for k, v in shares.items()] if total > 0 else []
    top = float(c.get("max_songs_per_week", 3))
    cadence = [{"creator_id": x.creator_id, "songs_per_week": min(max(float(x.songs_per_week), 0.0), top)}
               for x in out.cadence if x.creator_id in artists]
    decided = []
    ids = {p["id"] for p in open_concepts}
    for d in out.concepts:
        if d.proposal_id not in ids:
            continue
        subject = next(p["subject"] for p in open_concepts if p["id"] == d.proposal_id)
        accept = d.accept and not clean_words(policy, subject)
        decided.append({"proposal_id": d.proposal_id,
                        **proposals.decide(conn, policy, d.proposal_id, decided_by=NAME, accept=accept,
                                           reason=d.reason or ("accepted" if accept else "rejected"))})
    filed = [proposals.file(conn, policy, kind="tier_change", from_role=NAME, creator_id=t.creator_id,
                            subject={"tier": t.tier}, argument=t.argument, job_id=ctx.job.id)
             for t in out.tier_requests if t.creator_id in artists and t.tier in tiers]
    applied = {"budget_split": split, "cadence": cadence, "concepts": decided, "tier_requests": filed,
               "genre_priorities": [g.model_dump() for g in out.genre_priorities]}
    ctx.conn.execute("UPDATE decisions SET output = output || jsonb_build_object('applied', %s::jsonb) "
                     "WHERE job_id = %s AND step = %s", (_json(applied), ctx.job.id, f"{NAME}:{step}"))
    return applied


def _json(x) -> str:
    import json
    return json.dumps(x, default=str)


def latest_slate(conn) -> dict | None:
    """The newest slate's applied split and cadence (what the pipeline's scheduler reads)."""
    return conn.execute("SELECT id, created_at, output->'applied' AS applied FROM decisions "
                        "WHERE kind = 'music_head_of_music' ORDER BY created_at DESC LIMIT 1").fetchone()


class HeadOfMusic:
    name = NAME

    def run(self, ctx: AgentContext) -> dict:
        return plan(ctx, include_sandbox=bool((ctx.job.payload or {}).get("include_sandbox")))


register(HeadOfMusic())
