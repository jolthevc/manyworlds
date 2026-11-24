"""The lyricist as a music role (spec section 3): the words of one song in the artist's voice, written by
manyworlds.brain.lyrics (its checks: no real names, no quoted lines, no denial of being AI-made...) from the
songwriter's concept, the producer's length and the critic's fixes. Agent name `lyricist`; its llm role is the
lyricist's (llm.roles.lyricist). Recorded once per job step like every role (decision kind music_lyricist)."""

from __future__ import annotations

from ... import music
from .. import budget, lyrics
from ..agent import register, remember
from ..music_roles import SongRole, SongRoleAgent, _guard, draft_fixes, register_song_role


def idea_from(draft: dict, fixes: list[str]) -> dict:
    """The lyricist's idea (brain.lyrics reads concept, caption, tone, script) from the songwriter's concept."""
    c = draft.get("concept") or {}
    parts = [c.get("concept") or (draft.get("brief") or {}).get("concept") or "",
             f"theme: {c['theme']}" if c.get("theme") else "",
             f"point of view: {c['pov']}" if c.get("pov") else "",
             "structure: " + ", ".join(f"{s['name']} ({s['seconds']} s)" for s in c["structure"])
             if c.get("structure") else "",
             f"the hook, sung by {c.get('hook_at_s')} s: {c['hook_line']}" if c.get("hook_line") else ""]
    notes = [f"fix: {f}" for f in fixes] + [f"avoid: {p}" for p in c.get("problems") or []]
    return {"concept": "; ".join(p for p in parts if p), "caption": c.get("title_idea") or "",
            "tone": c.get("mood") or "", "script": "\n".join(notes)}


class Lyricist(SongRole):
    name = "lyricist"

    def run(self, ctx, draft, *, step):
        idea = idea_from(draft, draft_fixes(draft, self.name))
        seconds = float((draft.get("production") or {}).get("duration_s")
                        or ctx.policy.get("music", "pipeline", "song_seconds", default=30))

        def compute():
            tag = budget.tag_for(ctx.conn, lyrics.AGENT, draft.get("creator_id"))
            try:
                spec = lyrics.write_song(ctx.conn, ctx.policy, ctx.llm, creator_id=draft.get("creator_id"), idea=idea,
                                         bible=draft.get("bible"), max_seconds=seconds,
                                         before_call=_guard(ctx.conn, ctx.policy, tag))
                out, why = {"spec": spec, "refused": None}, "written and checked"
            except music.SongRefused as e:
                out, why = {"spec": None, "refused": str(e)[:1000]}, f"refused: {e}"[:500]
            return {"kind": "music_lyricist", "creator_id": draft.get("creator_id"),
                    "inputs": {"step": step, "idea": idea}, "output": out, "rationale": why,
                    "model": ctx.llm.model_for(lyrics.AGENT)}
        out = remember(ctx, f"lyricist:{step}", compute)
        if out.get("spec"):
            draft["words"] = {k: out["spec"][k] for k in ("title", "lyrics")}
        draft["lyrics_refused"] = out.get("refused")
        draft.setdefault("trace", []).append({"role": self.name, "step": step})
        return out


ROLE = register_song_role(Lyricist())
register(SongRoleAgent(ROLE))
