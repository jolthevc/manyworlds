"""Melody (spec section 3): candidate takes of one song (seeds and variations, music.song jobs, at most
music.roles.melody.max_takes) and the pick among them. The takes are made by the pipeline's take maker
(manyworlds.music_pipeline: queued GPU jobs, or stubbed in the sandbox); once measured, melody reads each take's
measurements against the bands and picks the strongest hook. A pick outside the takes falls back to code's choice
(the take with the most features inside their bands, then the earliest chorus). Sections to regenerate are noted for
an engine that can repaint (not ACE-Step's ComfyUI graph yet)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..agent import register
from ..music_roles import SongRole, SongRoleAgent, StrictInput, draft_fixes, register_song_role

SYSTEM = """You pick the strongest take of ONE song for an AI music studio. Each candidate take is the same song made
with a different seed; you see each take's measurements (when the chorus lands, how often it repeats, vocal entry
and clarity, loudness, energy lift into the chorus) and where it sits against the bands. Pick the take whose hook is
strongest and earliest without breaking the rest, and say why from the numbers. Note sections worth regenerating."""


class Take(BaseModel):
    index: int
    song_id: str
    features: dict = Field(default_factory=dict)
    position: list[dict] = Field(default_factory=list)


class Input(StrictInput):
    concept: dict = Field(default_factory=dict)
    takes: list[Take] = Field(default_factory=list)
    fixes: list[str] = Field(default_factory=list)


class Output(BaseModel):
    take_index: int = -1
    regenerate_sections: list[str] = Field(default_factory=list)
    why: str = ""


def code_pick(takes: list[dict]) -> int:
    """The take with the most features within their bands, then the earliest first chorus, then the first."""
    def score(t):
        inside = sum(1 for p in t.get("position") or [] if p["position"] == "within")
        chorus = (t.get("features") or {}).get("time_to_first_chorus_s")
        return (-inside, chorus if isinstance(chorus, (int, float)) else 1e9, t["index"])
    return min(takes, key=score)["index"] if takes else -1


class Melody(SongRole):
    name = "melody"
    Input, Output, SYSTEM = Input, Output, SYSTEM

    def build(self, ctx, draft):
        return Input(concept=draft.get("concept") or {},
                     takes=[Take.model_validate(t) for t in draft.get("takes") or []],
                     fixes=draft_fixes(draft, self.name))

    def apply(self, ctx, draft, out):
        takes = draft.get("takes") or []
        idx = {t["index"] for t in takes}
        ok = out.take_index in idx
        pick = out.take_index if ok else code_pick(takes)
        draft["pick"] = {"take_index": pick, "by": "melody" if ok else "code", "why": out.why,
                         "regenerate_sections": out.regenerate_sections}
        chosen = next((t for t in takes if t["index"] == pick), None)
        draft["chosen_song_id"] = chosen["song_id"] if chosen else None


ROLE = register_song_role(Melody())
register(SongRoleAgent(ROLE))
