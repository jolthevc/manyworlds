"""Original lyrics for a song post (policy `music`, OFF; llm role `lyricist`).

For a scheduled music_video post (manyworlds.music.SUNG_FORMATS) the lyricist writes the song from the creator's bible
(her artist voice, persona, world) and the idea (concept, caption, scenes): a title, section-tagged lyrics, the style
tags (genre, mood, instruments, vocal), bpm, key and length. Then code checks it (manyworlds.music.issues): no real
artist's name, nothing that points at someone else's work or voice ("in the style of", "feat.", "type beat"), no quoted
passage (someone else's line), no line of a real song the owner listed, no denial of being AI-made, nothing in
scope.excluded_topics, known section tags, line limits. A refused draft is rewritten with its problems listed, up to
music.lyrics.max_attempts; a draft still refused raises music.SongRefused and the post gets no song (production then
drops it before any GPU: "check the brief before the GPU").

The song spec goes to posts.body.song; production makes it (production.song). Captions and the bio keep the policy's
disclosure line as for every post (set at publish); lyrics never deny it.
"""

from __future__ import annotations

import json
import logging

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from .. import music
from ..policy import Policy

log = logging.getLogger("manyworlds.lyrics")
AGENT = "lyricist"


class SongAnswer(BaseModel):
    title: str = Field(description="the song's title, a few words, original")
    lyrics: str = Field(description="the lyrics, every section starting with its tag on its own line, e.g. [Verse], "
                                    "[Chorus], [Bridge]; one sung line per line")
    tags: list[str] = Field(description="style tags: genre, mood, instruments, vocal type and tempo feel, lower case; "
                                        "never a person, band, song or label")
    bpm: int = Field(description="tempo in beats per minute")
    key: str = Field(description="the key, e.g. 'A minor' or 'Eb major'")
    time_signature: str = Field(description="beats per bar: 2, 3, 4 or 6")
    duration_s: float = Field(description="the song's length in seconds")


SYSTEM = (
    "You write original songs for {name}, an openly AI-generated musician (an original person, not based on anyone "
    "real). Write in her own artist voice, from her life and world as the bible describes it, for the idea given.\n"
    "Rules, all of them hard:\n"
    "- Every word is new and yours. Never quote, paraphrase or echo lines of existing songs, poems or slogans; no "
    "quotation marks around lines.\n"
    "- Never name a real artist, band, songwriter, producer, song or album, and never write 'in the style of', "
    "'sounds like', 'feat.', 'type beat' or anything that points at someone else's work or voice, in the lyrics, the "
    "title or the tags. Style tags are genre, mood, instruments, vocal and tempo words only.\n"
    "- Never say or imply she is human or not AI. Never touch these topics: {excluded}.\n"
    "- Section tags on their own lines, from: {sections}. Keep lines singable: at most {max_words} words a line, at "
    "most {max_lines} lines. A song for a post is short: {seconds} seconds, built around one strong hook.\n"
    "- Key from the twelve major or minor keys (e.g. 'A minor'); bpm between {bpm_lo} and {bpm_hi}; language {lang}.")


def _system(policy: Policy, bible: dict, max_seconds: float) -> str:
    c = music.cfg(policy)
    lc, lim, d = c.get("lyrics") or {}, c.get("limits") or {}, c.get("defaults") or {}
    bpm = lim.get("bpm") or [50, 200]
    return SYSTEM.format(
        name=bible.get("display_name") or "the creator",
        excluded=", ".join(policy.get("scope", "excluded_topics", default=[]) or []) or "none",
        sections=", ".join(f"[{str(t).title()}]" for t in lc.get("section_tags") or []),
        max_words=lc.get("max_line_words", 16), max_lines=lc.get("max_lines", 40),
        seconds=int(min(float(d.get("duration_s", 30)), max_seconds)), bpm_lo=bpm[0], bpm_hi=bpm[1],
        lang=d.get("language", "en"))


def _content(bible: dict, idea: dict, problems: list[str] | None) -> str:
    keep = {k: bible.get(k) for k in ("display_name", "persona", "voice", "interests", "home_and_places", "one_line",
                                      "niche", "archetype", "music") if bible.get(k)}
    brief = {k: idea.get(k) for k in ("concept", "caption", "scenes", "script", "series", "tone", "pillar")
             if idea.get(k)}
    text = f"The creator (bible):\n{json.dumps(keep, ensure_ascii=False)}\n\nThe post:\n{json.dumps(brief, ensure_ascii=False)}"
    if problems:
        text += ("\n\nYour last draft was refused for these reasons; write a new song that has none of them:\n- "
                 + "\n- ".join(problems))
    return text


def write_song(conn, policy: Policy, llm, *, creator_id, idea: dict, bible: dict | None = None,
               max_seconds: float | None = None) -> dict:
    """The song spec for an idea (music.Song fields), checked; raises music.SongRefused when every draft fails."""
    from . import budget
    if bible is None:
        row = conn.execute("SELECT bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
        bible = (row or {}).get("bible") or {}
    c = music.cfg(policy)
    lc = c.get("lyrics") or {}
    cap = float(max_seconds if max_seconds is not None else policy.get("music", "video", "max_seconds", default=30))
    agent = budget.tag_for(conn, AGENT, creator_id)
    problems: list[str] = []
    for _ in range(max(1, int(lc.get("max_attempts", 2)))):
        with llm.tagged(agent, None, before_call=lambda: budget.check(conn, policy, agent)):
            out = llm.structured(AGENT, system=_system(policy, bible, cap), content=_content(bible, idea, problems),
                                 schema=SongAnswer, effort=str(lc.get("effort", "medium")), creator_id=creator_id)
        spec = {"title": out.title, "lyrics": out.lyrics, "tags": music.tags_text(out.tags), "bpm": out.bpm,
                "key": out.key, "time_signature": out.time_signature, "duration_s": out.duration_s}
        problems = music.issues(spec, policy)
        if not problems:
            try:
                s = music.normalise(spec, policy, max_seconds=cap)
            except music.SongRefused as e:
                problems = [str(e)]
                continue
            return {"title": s.title, "lyrics": s.lyrics, "tags": s.tags, "bpm": s.bpm, "key": s.key,
                    "time_signature": s.time_signature, "language": s.language, "duration_s": s.duration_s,
                    "instrumental": s.instrumental}
    raise music.SongRefused("; ".join(problems))


def write_songs(conn, policy: Policy, llm, creator_id, posts) -> int:
    """Songs for newly scheduled sung posts ([(post_id, format, body)]) while music.enabled: each gets body.song, or
    body.song_error when every draft was refused (production then drops it before any GPU work). Sandbox test
    creators are skipped. Returns how many songs were written."""
    from . import budget
    if llm is None or not music.enabled(policy):
        return 0
    todo = [(pid, body) for pid, fmt, body in posts or [] if fmt in music.SUNG_FORMATS and not (body or {}).get("song")]
    if not todo or budget.tag_for(conn, AGENT, creator_id).startswith(budget.SANDBOX):
        return 0
    written = 0
    for post_id, body in todo:
        try:
            spec = write_song(conn, policy, llm, creator_id=creator_id, idea=body or {})
            patch = {"song": spec}
            written += 1
        except Exception as e:  # noqa: BLE001 - a post without a song is dropped by production, never made wrong
            log.warning("no song for post %s (%s: %s)", post_id, type(e).__name__, str(e)[:300])
            patch = {"song_error": f"{type(e).__name__}: {str(e)[:300]}"}
        conn.execute("UPDATE posts SET body = body || %s, updated_at = now() WHERE id = %s AND status = 'planned'",
                     (Jsonb(patch), post_id))
    return written
