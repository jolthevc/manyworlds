"""The producer (spec section 3): arrangement, tempo, key, instrumentation and the engine's style tags for one
song, from the brief, the concept, the genre's bands and the critic's fixes. Tags pass the lyric checks (genre, mood
and instrument words only). Runs in the pipeline or alone as `brain.producer` ({song_id})."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ... import music
from ..agent import register
from ..music_roles import SongBrief, SongRole, SongRoleAgent, StrictInput, draft_fixes, register_song_role

SYSTEM = """You are the producer of an AI music studio. For ONE song, set the arrangement and the engine settings:
style tags (genre, subgenre, mood, instruments, production texture, tempo feel: lower-case words only, never a
person, band, song or label), bpm, key (one of the twelve major or minor keys, e.g. 'A minor'), beats per bar, the
length in seconds, the arrangement (sections in order) and the energy plan (where it builds). Stay within the bands
you are given unless the concept needs otherwise, and say why. Address every fix from the critic."""


class Input(StrictInput):
    brief: SongBrief
    concept: dict = Field(default_factory=dict)
    sonic_identity: dict = Field(default_factory=dict, description="the artist's sound (music dossier)")
    bands: dict = Field(default_factory=dict)
    max_seconds: float = 30.0
    fixes: list[str] = Field(default_factory=list)


class Output(BaseModel):
    tags: list[str] = Field(default_factory=list)
    bpm: int = 96
    key: str = "C major"
    time_signature: str = "4"
    duration_s: float = 30.0
    arrangement: list[str] = Field(default_factory=list)
    instruments: list[str] = Field(default_factory=list)
    energy_plan: str = ""
    why: str = ""


class Producer(SongRole):
    name = "producer"
    Input, Output, SYSTEM = Input, Output, SYSTEM

    def build(self, ctx, draft):
        c = draft.get("context") or {}
        return Input(brief=SongBrief.model_validate(draft.get("brief") or {}), concept=draft.get("concept") or {},
                     sonic_identity=(draft.get("artist") or {}).get("sonic_identity") or {},
                     bands=(c.get("bands") or {}).get("bands") or {},
                     max_seconds=float(ctx.policy.get("music", "pipeline", "song_seconds", default=30)),
                     fixes=draft_fixes(draft, self.name))

    def apply(self, ctx, draft, out):
        p = out.model_dump()
        p["tags"] = [t for t in p["tags"] if not music.issues({"title": "", "lyrics": "", "tags": t}, ctx.policy)]
        draft["production"] = p


ROLE = register_song_role(Producer())
register(SongRoleAgent(ROLE))
