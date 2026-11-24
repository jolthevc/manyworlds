"""The mix and master (spec section 3): loudness, balance and polish of the chosen take, from its measurements, the
bands and the critic's mix fixes. Writes a mix plan (target LUFS, true-peak ceiling, EQ, dynamics, stereo); the
DSP that applies it on the control plane is not built yet, so the plan is kept on the song (songs.spec.mix_plan).
Targets are clamped to policy music.roles.mix_master (lufs_range, max_true_peak_dbtp)."""

from __future__ import annotations

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from ..agent import register
from ..music_roles import SongRole, SongRoleAgent, StrictInput, cfg, draft_fixes, register_song_role

SYSTEM = """You are the mix and mastering engineer of an AI music studio. For the chosen take of ONE song, read its
measurements (loudness, true peak, crest factor, clipping, spectral balance, stereo width, vocal clarity) against the
bands, and write the mix plan: target integrated loudness, true-peak ceiling, EQ moves, dynamics, stereo, and why.
Address every fix from the critic. Fix what the numbers show; don't change what already works."""


class Input(StrictInput):
    features: dict = Field(default_factory=dict)
    position: list[dict] = Field(default_factory=list, description="the song against its bands")
    fixes: list[str] = Field(default_factory=list)


class Output(BaseModel):
    target_lufs: float = -10.0
    true_peak_ceiling_dbtp: float = -1.0
    eq: list[str] = Field(default_factory=list)
    dynamics: str = ""
    stereo: str = ""
    why: str = ""


class MixMaster(SongRole):
    name = "mix_master"
    Input, Output, SYSTEM = Input, Output, SYSTEM

    def build(self, ctx, draft):
        c = draft.get("take_context") or {}
        return Input(features=c.get("features") or {}, position=c.get("position") or [],
                     fixes=draft_fixes(draft, self.name))

    def apply(self, ctx, draft, out):
        c = cfg(ctx.policy, self.name)
        lo, hi = c.get("lufs_range") or (-14, -8)
        plan = out.model_dump()
        plan["target_lufs"] = min(max(float(plan["target_lufs"]), float(lo)), float(hi))
        plan["true_peak_ceiling_dbtp"] = min(float(plan["true_peak_ceiling_dbtp"]),
                                             float(c.get("max_true_peak_dbtp", -1.0)))
        draft["mix_plan"] = plan
        if draft.get("chosen_song_id"):
            ctx.conn.execute("UPDATE songs SET spec = spec || %s, updated_at = now() WHERE id = %s",
                             (Jsonb({"mix_plan": plan}), draft["chosen_song_id"]))


ROLE = register_song_role(MixMaster())
register(SongRoleAgent(ROLE))
