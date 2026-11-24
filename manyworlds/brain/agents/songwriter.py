"""The songwriter (music department, spec section 3): concept, theme, point of view, structure and the hook idea of
one song, from the talent manager's brief, the artist's dossier, the nearest past songs and the open findings.
Runs in the pipeline (manyworlds.music_pipeline) or alone as `brain.songwriter` ({song_id}: a revision with the
critic's fixes). Its words pass the originality check (music_roles.word_issues); problems go to the lyricist."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..agent import register
from ..music_roles import SongBrief, SongRole, SongRoleAgent, StrictInput, clean_words, draft_fixes, register_song_role

SYSTEM = """You are the songwriter of an AI music studio whose artists are openly AI-generated original people.
From the brief, write the idea of ONE song for this artist: what it is about, from whose point of view, its
structure (sections with target lengths in seconds) and its hook: the one line people will repeat. Short-form
listening decides fast: the hook should land early (see the bands and findings you are given; they are advice).
Original only: never name a real artist or song, never "the next <someone>", never "in the style of". If fixes from
the critic are listed, address each of them."""


class Section(BaseModel):
    name: str = Field(default="verse", description="verse, pre-chorus, chorus, bridge, intro, outro, hook...")
    seconds: float = 8.0


class Input(StrictInput):
    brief: SongBrief
    artist: dict = Field(default_factory=dict, description="the artist's sonic identity and story (music dossier)")
    nearest: list[dict] = Field(default_factory=list, description="the most alike past songs and how they did")
    findings: list[str] = Field(default_factory=list, description="open music findings for this artist or genre")
    bands: dict = Field(default_factory=dict, description="what good looks like (advisory)")
    fixes: list[str] = Field(default_factory=list, description="the critic's fixes for the songwriter")


class Output(BaseModel):
    title_idea: str = ""
    concept: str = ""
    theme: str = ""
    pov: str = ""
    mood: str = ""
    structure: list[Section] = Field(default_factory=list)
    hook_line: str = ""
    hook_at_s: float = Field(default=0.0, description="when the hook first lands, seconds from the start")
    why: str = ""


class Songwriter(SongRole):
    name = "songwriter"
    Input, Output, SYSTEM = Input, Output, SYSTEM

    def build(self, ctx, draft):
        c = draft.get("context") or {}
        return Input(brief=SongBrief.model_validate(draft.get("brief") or {}), artist=draft.get("artist") or {},
                     nearest=c.get("nearest") or [], findings=[f["statement"] for f in c.get("findings") or []],
                     bands=(c.get("bands") or {}).get("bands") or {}, fixes=draft_fixes(draft, self.name))

    def apply(self, ctx, draft, out):
        concept = out.model_dump()
        concept["problems"] = clean_words(ctx.policy, concept)
        draft["concept"] = concept


ROLE = register_song_role(Songwriter())
register(SongRoleAgent(ROLE))
