"""The music talent manager, one per artist (spec section 3): the artist's coordinator, like talent_manager.py for
content creators, with a music dossier.

`brain.music_talent_manager` ({creator_id, mode: onboard | review}):
  onboard  writes the music dossier: sonic identity (genre, subgenre, the fixed voice and its tags, palette, sound
           described in words, never by naming anyone), story, release plan (singles toward an EP), fan relationship.
  review   reads the dossier, the artist's songs (verdicts, outcomes), the music findings and its own scored
           predictions; decides the career arc and what to drop next and why: the next song briefs (what the
           pipeline makes, music_pipeline.next_briefs), a release plan step, predictions and requests (a tier change
           to the strategist through brain.proposals, within music_talent_manager's rights).
Every text passes the originality check (music_roles.word_issues): a dossier or brief naming a real artist is
refused (recorded, not used). Decisions: music_dossier, music_career.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ...worker.runner import NonRetryable
from .. import evidence, music_analysis, predictions, proposals
from ..agent import AgentContext, register
from ..music_roles import SongBrief, StrictInput, ask, cfg, clean_words

NAME = "music_talent_manager"

DOSSIER_SYSTEM = """You are the talent manager of one openly AI-generated music artist (an original person, never
based on anyone real). Write her music dossier: her sonic identity (genre and subgenre, her voice described in words
plus a few lower-case voice tags that stay the same on every song, a palette of instruments and textures, what her
sound is in words: never by naming a real artist, band or song), her story (where the songs come from), a release
plan (singles toward a first EP) and how she relates to her listeners. Make her distinct from the other artists
listed."""

REVIEW_SYSTEM = """You are the talent manager of one openly AI-generated music artist. Read her dossier, her songs (the
critic's verdicts and how songs did; SIMULATED data is a test, never demand), the music findings and your own past
predictions with how they turned out. Decide the next step of her career: the state in a sentence, the next songs to
make (briefs: concept, theme, point of view, mood, genre, target listener, which series pattern, why), the next
release step, checkable predictions about her reach, and requests (only a production tier change, with the
argument). Prefer what the evidence supports; keep exploring while it is thin. Never name a real artist or song."""


class SonicIdentity(BaseModel):
    genre: str = ""
    subgenre: str = ""
    voice: str = ""
    voice_tags: list[str] = Field(default_factory=list)
    palette: list[str] = Field(default_factory=list)
    sound_in_words: str = ""


class DossierInput(StrictInput):
    artist: dict
    roster: list[dict] = Field(default_factory=list, description="the other music artists: handle and genre")
    archetype: dict = Field(default_factory=dict, description="the musician playbook that fits her")


class Dossier(BaseModel):
    sonic_identity: SonicIdentity = Field(default_factory=SonicIdentity)
    story: str = ""
    release_plan: list[str] = Field(default_factory=list)
    fan_relationship: str = ""
    why: str = ""


class Claim(BaseModel):
    claim: str = ""
    metric: Literal["median_reach", "reach_growth"] = "median_reach"
    comparator: Literal["above", "below"] = "above"
    threshold: float = 0.0
    probability: float = 0.5
    horizon_days: int = 21


class Request(BaseModel):
    tier: str = ""
    argument: str = ""


class ReviewInput(StrictInput):
    dossier: dict
    songs: list[dict] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    track_record: list[dict] = Field(default_factory=list)
    mode: str = "sandbox"
    songs_wanted: int = 1


class Career(BaseModel):
    state: str = ""
    next_songs: list[SongBrief] = Field(default_factory=list)
    release_step: str = ""
    predictions: list[Claim] = Field(default_factory=list)
    requests: list[Request] = Field(default_factory=list)
    why: str = ""


def latest_dossier(conn, creator_id) -> dict | None:
    return conn.execute("SELECT id, created_at, output FROM decisions WHERE kind = 'music_talent_manager' "
                        "AND creator_id = %s AND inputs->>'step' LIKE 'dossier%%' AND output->>'refused' IS NULL "
                        "ORDER BY created_at DESC LIMIT 1", (creator_id,)).fetchone()


def latest_career(conn, creator_id) -> dict | None:
    return conn.execute("SELECT id, created_at, output FROM decisions WHERE kind = 'music_talent_manager' "
                        "AND creator_id = %s AND inputs->>'step' LIKE 'career%%' ORDER BY created_at DESC LIMIT 1",
                        (creator_id,)).fetchone()


def _artist(conn, creator_id) -> dict:
    row = conn.execute("SELECT handle, bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
    if row is None:
        raise NonRetryable(f"no creator {creator_id}")
    b = row["bible"] or {}
    return {"handle": row["handle"], **{k: b.get(k) for k in ("display_name", "persona", "voice", "interests",
                                                             "one_line", "niche", "archetype", "music", "age")
                                       if b.get(k)}}


def onboard(ctx: AgentContext, creator_id) -> dict:
    from .. import archetypes, music_portfolio
    conn, policy = ctx.conn, ctx.policy
    roster = [{"handle": a["handle"], "genre": a["genre"]}
              for a in music_portfolio.artists(conn, policy, include_sandbox=True) if a["creator_id"] != str(creator_id)]
    artist = _artist(conn, creator_id)
    try:
        books = archetypes.playbooks(policy, artist)
    except Exception:  # noqa: BLE001 - the playbook is context, not a requirement
        books = []
    out = ask(ctx, NAME, system=DOSSIER_SYSTEM,
              inp=DossierInput(artist=artist, roster=roster, archetype=(books or [{}])[0] or {}),
              out=Dossier, creator_id=creator_id, step="dossier")
    problems = clean_words(policy, out.model_dump())
    if problems:   # kept in the decision log, never used
        conn.execute("UPDATE decisions SET output = output || jsonb_build_object('refused', %s::text) "
                     "WHERE job_id = %s AND step = %s", ("; ".join(problems), ctx.job.id, f"{NAME}:dossier"))
        return {"dossier": None, "refused": problems}
    if out.sonic_identity.genre:
        conn.execute("UPDATE creators SET vertical = 'music' WHERE id = %s", (creator_id,))
    return {"dossier": out.model_dump(), "refused": []}


def review(ctx: AgentContext, creator_id, *, songs_wanted: int | None = None) -> dict:
    conn, policy = ctx.conn, ctx.policy
    c = cfg(policy, NAME)
    d = latest_dossier(conn, creator_id)
    if d is None:
        raise NonRetryable("the artist has no music dossier yet (mode onboard first)")
    mode = evidence.mode_for(conn, creator_id)
    rows = music_analysis.song_rows(conn, policy, mode=mode, creator_id=creator_id)[-int(c.get("recent_songs", 20)):]
    verdicts = {str(r["song_id"]): r["verdict"] for r in conn.execute(
        "SELECT DISTINCT ON (song_id) song_id, verdict FROM song_assessments WHERE song_id = ANY(%s::uuid[]) "
        "ORDER BY song_id, created_at DESC", ([r["song_id"] for r in rows],)).fetchall()}
    song_list = [{"title": r["title"], "version": r["version"], "verdict": verdicts.get(r["song_id"]),
                  "outcome": r["outcome"]} for r in rows]
    found = [f["statement"] + (" [SIMULATED]" if f.get("label") else "")
             for f in music_analysis.findings(conn, mode=mode, creator_id=creator_id)]
    want = int(songs_wanted if songs_wanted is not None else c.get("songs_per_review", 2))
    inp = ReviewInput(dossier=d["output"], songs=song_list, findings=found, mode=mode, songs_wanted=want,
                      track_record=predictions.resolved_for(conn, creator_id, role=NAME,
                                                            n=int(c.get("resolved_predictions", 10))))
    out = ask(ctx, NAME, system=REVIEW_SYSTEM, inp=inp, out=Career, creator_id=creator_id,
              step=f"career:{ctx.job.id}")
    briefs = [b for b in out.next_songs if not clean_words(policy, b.model_dump())][:want]
    decision = conn.execute("SELECT id FROM decisions WHERE job_id = %s AND step = %s",
                            (ctx.job.id, f"{NAME}:career:{ctx.job.id}")).fetchone()["id"]
    made = 0
    if not conn.execute("SELECT 1 FROM predictions WHERE decision_id = %s", (decision,)).fetchone():
        for p in out.predictions[:int(c.get("max_predictions", 2))]:
            made += predictions.record(conn, role=NAME, creator_id=creator_id, decision_id=decision, claim=p.claim,
                                       metric=p.metric, comparator=p.comparator, threshold=p.threshold,
                                       probability=p.probability, horizon_days=p.horizon_days) is not None
    filed = [proposals.file(conn, policy, kind="tier_change", from_role=NAME, creator_id=creator_id,
                            subject={"tier": r.tier}, argument=r.argument, job_id=ctx.job.id)
             for r in out.requests[:1] if r.tier]
    return {"briefs": [b.model_dump() for b in briefs], "predictions": made, "requests": filed,
            "dropped_briefs": len(out.next_songs) - len(briefs), "state": out.state}


class MusicTalentManager:
    name = NAME

    def run(self, ctx: AgentContext) -> dict:
        p = ctx.job.payload or {}
        cid, mode = p.get("creator_id"), p.get("mode")
        if not cid or mode not in ("onboard", "review"):
            raise NonRetryable("music_talent_manager payload needs creator_id and mode onboard|review")
        if mode == "onboard":
            return onboard(ctx, cid)
        return review(ctx, cid, songs_wanted=p.get("songs"))


register(MusicTalentManager())
