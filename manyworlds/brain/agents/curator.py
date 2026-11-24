"""The curator (spec section 3): which of an artist's finished songs make a release (single, EP or album), in what
order, and why it holds together. Reads the artist's songs that are made and not yet released (latest version of
each lineage, dropped takes left out) with their measurements, the critic's verdicts and outcomes. Code checks the
plan (only the artist's own unreleased songs, a track count within music.roles.curator.tracks.<kind>) and writes it
to the releases table (manyworlds.songs.new_release, place). `brain.curator` ({creator_id, kind?})."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ... import songs
from ...worker.runner import NonRetryable
from .. import evidence, music_analysis
from ..agent import AgentContext, register
from ..music_roles import StrictInput, ask, cfg, clean_words

NAME = "curator"

SYSTEM = """You are the curator of an AI music studio. From one artist's finished, unreleased songs, choose a release:
a single, an EP or an album, its title (original, never a real artist's or record's) and its tracks in order, with
why each is there and why the order works (an opener that hooks fast, energy that moves, a closer). Prefer songs the
critic would ship and songs that did well; cohesion matters more than count. Use only the song ids listed."""

KIND = Literal["single", "ep", "album"]


class Candidate(BaseModel):
    song_id: str
    title: str = ""
    version: int = 1
    verdict: str | None = None
    outcome: dict | None = None
    features: dict = Field(default_factory=dict)


class Input(StrictInput):
    songs: list[Candidate] = Field(default_factory=list)
    released: list[dict] = Field(default_factory=list, description="the artist's earlier releases")
    kind_hint: KIND | None = None


class Track(BaseModel):
    song_id: str = ""
    why: str = ""


class Output(BaseModel):
    kind: KIND = "single"
    title: str = ""
    tracks: list[Track] = Field(default_factory=list)
    why: str = ""


def candidates(conn, policy, creator_id) -> list[dict]:
    """The artist's made, unreleased songs: the newest version of each lineage, takes not chosen left out."""
    rows = conn.execute(
        """SELECT s.id, s.title, s.version FROM songs s WHERE s.creator_id = %s AND s.status = 'made'
             AND s.release_id IS NULL AND NOT EXISTS (SELECT 1 FROM songs r WHERE r.parent_id = s.id)
           ORDER BY s.created_at""", (creator_id,)).fetchall()
    ids = [str(r["id"]) for r in rows]
    feats = songs.latest_features(conn, ids)
    keep = ("time_to_first_chorus_s", "chorus_repeats", "tempo_bpm", "key", "lufs_integrated", "duration_s",
            "chorus_energy_lift_db")
    mode = evidence.mode_for(conn, creator_id)
    outcomes = {r["song_id"]: r["outcome"] for r in music_analysis.song_rows(conn, policy, mode=mode, song_ids=ids)}
    verdicts = {str(r["song_id"]): r["verdict"] for r in conn.execute(
        "SELECT DISTINCT ON (song_id) song_id, verdict FROM song_assessments WHERE song_id = ANY(%s::uuid[]) "
        "ORDER BY song_id, created_at DESC", (ids,)).fetchall()}
    return [{"song_id": str(r["id"]), "title": r["title"], "version": r["version"], "verdict": verdicts.get(str(r["id"])),
             "outcome": outcomes.get(str(r["id"])),
             "features": {k: v for k, v in (feats.get(str(r["id"])) or {}).items() if k in keep}} for r in rows]


def curate(ctx: AgentContext, creator_id, *, kind: str | None = None, step: str | None = None) -> dict:
    conn, policy = ctx.conn, ctx.policy
    c = cfg(policy, NAME)
    pool = candidates(conn, policy, creator_id)
    if not pool:
        return {"release_id": None, "why": "no finished unreleased songs"}
    released = [dict(r) for r in conn.execute("SELECT kind, title, status FROM releases WHERE creator_id = %s "
                                              "ORDER BY created_at", (creator_id,)).fetchall()]
    out = ask(ctx, NAME, system=SYSTEM, inp=Input(songs=[Candidate.model_validate(s) for s in pool], released=released,
                                                  kind_hint=kind), out=Output, creator_id=creator_id,
              step=step or "release")
    ids = {s["song_id"] for s in pool}
    tracks, seen = [], set()
    for t in out.tracks:
        if t.song_id in ids and t.song_id not in seen:
            tracks.append(t)
            seen.add(t.song_id)
    lo, hi = (c.get("tracks") or {}).get(out.kind) or (1, 1)
    problems = clean_words(policy, {"title": out.title})
    if not (int(lo) <= len(tracks) <= int(hi)):
        problems.append(f"a {out.kind} has {lo}-{hi} tracks; the plan had {len(tracks)} usable")
    if problems:
        return {"release_id": None, "refused": problems}
    same = conn.execute("SELECT id FROM releases WHERE creator_id = %s AND title = %s", (creator_id, out.title)).fetchone()
    if same:   # a rerun of the same step (a resumed run): the release it made
        placed = [str(r["id"]) for r in conn.execute("SELECT id FROM songs WHERE release_id = %s ORDER BY track_number",
                                                     (same["id"],)).fetchall()]
        return {"release_id": str(same["id"]), "kind": out.kind, "title": out.title, "tracks": placed, "reused": True}
    rel = songs.new_release(conn, creator_id, out.kind, out.title or "untitled")
    for n, t in enumerate(tracks, start=1):
        songs.place(conn, t.song_id, rel, n)
    return {"release_id": str(rel), "kind": out.kind, "title": out.title, "tracks": [t.song_id for t in tracks]}


class Curator:
    name = NAME

    def run(self, ctx: AgentContext) -> dict:
        p = ctx.job.payload or {}
        if not p.get("creator_id"):
            raise NonRetryable("curator payload needs creator_id")
        return curate(ctx, p["creator_id"], kind=p.get("kind"))


register(Curator())
