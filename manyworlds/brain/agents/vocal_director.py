"""The vocal director (spec section 3): delivery and style for one song over the artist's fixed voice. The voice
itself never changes song to song (the dossier's voice tags always lead the vocal tags; voice conversion into the
artist's voice asset is a later branch's step); the director sets delivery: intensity, phrasing, ad-libs, harmonies,
register. Runs in the pipeline or alone as `brain.vocal_director` ({song_id})."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ... import music
from ..agent import register
from ..music_roles import SongBrief, SongRole, SongRoleAgent, StrictInput, draft_fixes, register_song_role

SYSTEM = """You are the vocal director of an AI music studio. The artist's voice is fixed (given): never change who
she sounds like. For ONE song, direct the delivery: a few lower-case vocal tags for the engine (intensity, phrasing,
breathiness, harmonies, ad-libs, register; never a person's name), notes for the performance, and how strongly a
later voice conversion should hold her timbre (0-1). Address every fix from the critic."""


class Input(StrictInput):
    brief: SongBrief
    concept: dict = Field(default_factory=dict)
    voice: dict = Field(default_factory=dict, description="the artist's fixed voice: description and tags")
    fixes: list[str] = Field(default_factory=list)


class Output(BaseModel):
    vocal_tags: list[str] = Field(default_factory=list)
    delivery_notes: str = ""
    conversion_strength: float = 1.0
    why: str = ""


class VocalDirector(SongRole):
    name = "vocal_director"
    Input, Output, SYSTEM = Input, Output, SYSTEM

    def build(self, ctx, draft):
        ident = (draft.get("artist") or {}).get("sonic_identity") or {}
        return Input(brief=SongBrief.model_validate(draft.get("brief") or {}), concept=draft.get("concept") or {},
                     voice={"description": ident.get("voice") or "", "tags": ident.get("voice_tags") or []},
                     fixes=draft_fixes(draft, self.name))

    def apply(self, ctx, draft, out):
        v = out.model_dump()
        fixed = list(((draft.get("artist") or {}).get("sonic_identity") or {}).get("voice_tags") or [])
        delivery = [t for t in v["vocal_tags"] if t not in fixed
                    and not music.issues({"title": "", "lyrics": "", "tags": t}, ctx.policy)]
        v["vocal_tags"] = fixed + delivery
        v["conversion_strength"] = min(max(float(v["conversion_strength"]), 0.0), 1.0)
        draft["vocal"] = v


ROLE = register_song_role(VocalDirector())
register(SongRoleAgent(ROLE))
