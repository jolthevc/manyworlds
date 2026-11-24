"""Original songs (policy `music`, OFF): the song spec, the music.song job payload, lyric checks and generated beds.

  spec       a song as production and the lyricist hold it (body.song): {title, lyrics, tags, bpm, key,
             time_signature, language, duration_s, instrumental}. `lyrics` carries section tags on their own lines
             ("[Verse]", "[Chorus]", "[Bridge 2]"); an instrumental song has none (its lyrics are "[Instrumental]").
             `normalise(spec, policy)` fills the policy defaults (music.defaults), clamps the length to
             music.limits and refuses what the model cannot take (a key not in KEYS, a time signature or language
             not in the limits).
  payload    `song_payload(policy, spec, seed=, check=)`: the music.song job (worker/handlers/music_song.py). The
             engine settings travel in the payload (music.engines.<engine>), so the worker never reads the policy.
  lyrics     `parse(lyrics)` -> [(tag, lines)]; `sung_words(lyrics)`: what a listener hears (no tags, no
             parenthesised directions), the reference for the bench's lyric WER.
             `issues(spec, policy)`: why a song may not be made: a real artist's name (music.lyrics.banned_names),
             a phrase that points at someone else's work or voice ("in the style of", "feat.", "type beat"...), a
             quoted passage of min_quote_words or more (someone else's line), a line from music.lyrics.known_lines,
             a lyric that denies being AI-made, an excluded topic (scope.excluded_topics), unknown section tags, too
             many lines or words. Tags and title are checked like the lyrics.
  beds       `generated_beds(conn, policy)`: instrumental music.song outputs that passed their check, as sound
             library entries (licence own-generated), while sound.library.generated.enabled; `with_generated_beds`
             returns the policy with them in sound.library.entries (the finish plan carries it to the finisher).

No model calls and no heavy imports here: the brain writes lyrics (brain/lyrics.py), the worker makes the song.
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata

from .policy import Policy

KIND = "music.song"
SUNG_FORMATS = ("music_video",)          # post formats whose audio is a song (production)
ROLES = ("mix", "vocals", "instrumental")   # a music.song job's outputs, in this order (deterministic keys)
INSTRUMENTAL = "[Instrumental]"
# ACE-Step 1.5's keyscale choices (ComfyUI TextEncodeAceStepAudio1.5: 17 roots x major/minor).
ROOTS = ("C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B")
KEYS = tuple(f"{r} {m}" for r in ROOTS for m in ("major", "minor"))
_TAG = re.compile(r"^\s*\[([^\]]+)\]\s*$")


class SongRefused(ValueError):
    """A song spec that may not be made (see `issues`, `normalise`)."""


def cfg(policy: Policy | None) -> dict:
    return (policy.get("music", default={}) or {}) if policy is not None else {}


def enabled(policy: Policy | None) -> bool:
    return bool(cfg(policy).get("enabled"))


def kind(policy: Policy | None) -> str:
    return str(cfg(policy).get("kind") or KIND)


# --- lyrics ------------------------------------------------------------------------------------------------------

def _plain(text: str) -> str:
    """Lower case, accents folded, punctuation to spaces: what name and line matching compares."""
    t = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()
    return " ".join(re.findall(r"[a-z0-9']+", t))


def parse(lyrics: str) -> list[tuple[str, list[str]]]:
    """[(section tag as written, its lines)] in order; lines before any tag fall under ''. Blank lines dropped."""
    out: list[tuple[str, list[str]]] = []
    for raw in str(lyrics or "").splitlines():
        m = _TAG.match(raw)
        if m:
            out.append((m.group(1).strip(), []))
            continue
        line = raw.strip()
        if not line:
            continue
        if not out:
            out.append(("", []))
        out[-1][1].append(line)
    return out


def section_name(tag: str) -> str:
    """'Verse 2' -> 'verse', 'Pre-Chorus' -> 'pre-chorus', 'chorus x2' -> 'chorus'."""
    t = re.sub(r"\s*(x\s*\d+|\d+)\s*$", "", str(tag or "").strip().lower())
    return re.sub(r"\s+", "-", t)


def is_instrumental(spec: dict) -> bool:
    lyr = str(spec.get("lyrics") or "").strip()
    return bool(spec.get("instrumental")) or not lyr or lyr.lower() == INSTRUMENTAL.lower()


def sung_words(lyrics: str) -> str:
    """The words a listener hears: lines without section tags and without (parenthesised) directions."""
    lines = [line for _, lines in parse(lyrics) for line in lines]
    return " ".join(w for line in lines for w in re.sub(r"\([^)]*\)", " ", line).split())


def line_coverage(lyrics: str, heard: str, *, min_line_ratio: float) -> float | None:
    """The share of the written lines that were sung: a line counts when some stretch of the transcript as long as
    the line matches it at `min_line_ratio` or better (word sequence similarity). Unlike WER, a song that repeats
    its chorus or holds long instrumental parts is not penalised for words that are sung again (2026-11-23: a
    2-minute song that sang every line scored WER 6.0). None without lyrics."""
    from difflib import SequenceMatcher
    lines = [_plain(re.sub(r"\([^)]*\)", " ", line)).split() for _, ls in parse(lyrics) for line in ls]
    lines = [w for w in lines if w]
    if not lines:
        return None
    said = _plain(heard).split()
    found = 0
    for words in lines:
        n = len(words)
        best = 0.0
        for start in range(0, max(1, len(said) - n + 1)):
            best = max(best, SequenceMatcher(None, words, said[start:start + n + 2]).ratio())
            if best >= min_line_ratio:
                break
        found += best >= min_line_ratio
    return round(found / len(lines), 3)


def tags_text(tags) -> str:
    """The engine's style prompt: tags as one comma-separated line (a string is taken as it is)."""
    if isinstance(tags, str):
        return ", ".join(t.strip() for t in tags.split(",") if t.strip())
    return ", ".join(str(t).strip() for t in tags or [] if str(t).strip())


def _phrase_hits(text: str, phrases) -> list[str]:
    low = str(text or "").lower()
    return [p for p in phrases or [] if re.search(r"(?<![a-z0-9])" + re.escape(str(p).lower()), low)]


def _name_hits(text: str, names) -> list[str]:
    plain = f" {_plain(text)} "
    return [n for n in names or [] if f" {_plain(n)} " in plain and _plain(n)]


def _quotes(text: str, min_words: int) -> list[str]:
    found = re.findall(r"[\"“”]([^\"“”]+)[\"“”]", str(text or ""))
    return [q.strip() for q in found if len(q.split()) >= int(min_words)]


def issues(spec: dict, policy: Policy) -> list[str]:
    """Why this song may not be made (empty when it may): see the module docstring."""
    lc = cfg(policy).get("lyrics") or {}
    lyrics = str(spec.get("lyrics") or "")
    title = str(spec.get("title") or "")
    tags = tags_text(spec.get("tags"))
    out: list[str] = []
    for where, text in (("lyrics", lyrics), ("title", title), ("tags", tags)):
        for name in _name_hits(text, lc.get("banned_names")):
            out.append(f"{where} name a real artist ({name!r}): original artists only")
        for p in _phrase_hits(text, lc.get("banned_phrases")):
            out.append(f"{where} point at someone else's work or voice ({p!r})")
        for p in _phrase_hits(text, lc.get("denial_phrases")):
            out.append(f"{where} deny being AI-made ({p!r}): the disclosure is non-negotiable")
        for topic in _phrase_hits(text, policy.get("scope", "excluded_topics", default=[]) or []):
            out.append(f"{where} touch an excluded topic ({topic!r})")
    for q in _quotes(lyrics, int(lc.get("min_quote_words", 4))):
        out.append(f"the lyrics quote a line ({q[:60]!r}): every line must be the creator's own")
    known = {_plain(k) for k in lc.get("known_lines") or [] if _plain(k)}
    sections = parse(lyrics)
    for _, lines in sections:
        for line in lines:
            if _plain(line) in known:
                out.append(f"a line is from a real song ({line[:60]!r})")
    if not is_instrumental(spec):
        allowed = {str(t).lower() for t in lc.get("section_tags") or []}
        unknown = sorted({tag for tag, _ in sections if tag and section_name(tag) not in allowed})
        if unknown and allowed:
            out.append(f"unknown section tags {unknown} (music.lyrics.section_tags)")
        if not any(tag for tag, _ in sections):
            out.append("the lyrics have no section tags ([Verse], [Chorus], ...)")
        lines = [line for _, ls in sections for line in ls]
        if len(lines) > int(lc.get("max_lines", 40)):
            out.append(f"{len(lines)} lines, more than music.lyrics.max_lines {lc.get('max_lines')}")
        long = [line for line in lines if len(line.split()) > int(lc.get("max_line_words", 16))]
        if long:
            out.append(f"{len(long)} lines longer than {lc.get('max_line_words')} words")
        if not sung_words(lyrics):
            out.append("the lyrics have no sung words")
    return out


# --- the spec and the payload ---------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Song:
    title: str
    lyrics: str
    tags: str
    bpm: int
    key: str
    time_signature: str
    language: str
    duration_s: float
    instrumental: bool


def normalise(spec: dict, policy: Policy, *, max_seconds: float | None = None) -> Song:
    """The spec with the policy defaults filled in and its length clamped (music.limits; `max_seconds` lowers the
    cap, e.g. a music video's music.video.max_seconds). Raises SongRefused for what the engine can't take."""
    c = cfg(policy)
    d, lim = c.get("defaults") or {}, c.get("limits") or {}
    instrumental = is_instrumental(spec)
    lyrics = INSTRUMENTAL if instrumental else str(spec.get("lyrics") or "").strip()
    tags = tags_text(spec.get("tags"))
    if not tags:
        raise SongRefused("a song needs style tags (genre, mood, instruments, vocal)")
    bpm = int(round(float(spec.get("bpm") or d.get("bpm", 96))))
    lo, hi = (lim.get("bpm") or (50, 200))
    if not int(lo) <= bpm <= int(hi):
        raise SongRefused(f"bpm {bpm} outside music.limits.bpm {[lo, hi]}")
    key = " ".join(str(spec.get("key") or d.get("key", "C major")).split())
    key = next((k for k in KEYS if k.lower() == key.lower()), None) or ""
    if not key:
        raise SongRefused(f"key {spec.get('key')!r} is not one of the engine's keys (music.KEYS)")
    ts = str(spec.get("time_signature") or d.get("time_signature", "4")).split("/")[0].strip()
    if ts not in [str(x) for x in lim.get("time_signatures") or ["4"]]:
        raise SongRefused(f"time signature {ts!r} not in music.limits.time_signatures")
    lang = str(spec.get("language") or d.get("language", "en")).strip().lower()
    if lang not in [str(x) for x in lim.get("languages") or ["en"]]:
        raise SongRefused(f"language {lang!r} not in music.limits.languages")
    seconds = float(spec.get("duration_s") or d.get("duration_s", 30))
    top = float(lim.get("max_seconds", 240))
    if max_seconds is not None:
        top = min(top, float(max_seconds))
    seconds = round(max(float(lim.get("min_seconds", 10)), min(top, seconds)), 2)
    return Song(title=str(spec.get("title") or "").strip(), lyrics=lyrics, tags=tags, bpm=bpm, key=key,
                time_signature=ts, language=lang, duration_s=seconds, instrumental=instrumental)


def engine_settings(policy: Policy, engine: str | None = None, variant: str | None = None) -> dict:
    """What the worker needs of the engine (music.engines.<engine>), with the variant resolved: {engine, variant,
    licence, repo, diffusion_model, steps, cfg, shift, sampler, scheduler, text_encoders, vae, lm, timeout_s}."""
    c = cfg(policy)
    name = str(engine or c.get("engine") or "ace_step_1_5")
    e = dict((c.get("engines") or {}).get(name) or {})
    if not e:
        raise SongRefused(f"no music engine {name!r} in music.engines")
    out = {"engine": name, **{k: v for k, v in e.items() if k not in ("variants", "variant")}}
    if e.get("variants"):
        v = str(variant or e.get("variant") or "")
        if v not in e["variants"]:
            raise SongRefused(f"music engine {name!r} has no variant {v!r}")
        out.update(variant=v, **dict(e["variants"][v] or {}))
    return out


def song_payload(policy: Policy, spec: dict, *, seed: int, check: bool = False, engine: str | None = None,
                 variant: str | None = None, max_seconds: float | None = None) -> dict:
    """The music.song job for a song spec (checked: raises SongRefused when `issues` finds any)."""
    problems = issues(spec, policy)
    if problems:
        raise SongRefused("; ".join(problems))
    s = normalise(spec, policy, max_seconds=max_seconds)
    c = cfg(policy)
    payload = {"title": s.title, "lyrics": s.lyrics, "tags": s.tags, "bpm": s.bpm, "key": s.key,
               "time_signature": s.time_signature, "language": s.language, "duration_s": s.duration_s,
               "instrumental": s.instrumental, "seed": int(seed) % (2**31),
               "engine": engine_settings(policy, engine, variant),
               "separation": dict(c.get("separation") or {}),
               "library_licence": str(c.get("library_licence") or "own-generated")}
    if check:
        payload["check"] = dict(c.get("check") or {})
    return payload


# --- generated beds in the sound library ---------------------------------------------------------------------------

def _generated_cfg(policy: Policy) -> dict:
    return policy.get("sound", "library", "generated", default={}) or {}


def generated_beds(conn, policy: Policy) -> list[dict]:
    """Instrumental music.song outputs that may join the sound library (sound.library.generated, off as shipped):
    the `instrumental` output of an instrumental song, its job done, its check passed (while require_check), newest
    first, at most max_entries within `days`. Each is its own `library_entry` with the stored uri."""
    g = _generated_cfg(policy)
    if not g.get("enabled"):
        return []
    rows = conn.execute(
        """SELECT a.uri, a.meta FROM assets a JOIN jobs j ON j.id = a.job_id
           WHERE j.kind = %s AND j.status = 'done' AND a.kind = 'audio'
             AND a.meta->>'role' = 'instrumental' AND (a.meta->>'instrumental')::boolean IS TRUE
             AND a.created_at > now() - make_interval(days => %s)
           ORDER BY a.created_at DESC, a.uri LIMIT %s""",
        (str(g.get("kind") or KIND), int(g.get("days", 90)), int(g.get("max_entries", 40)))).fetchall()
    out = []
    for r in rows:
        meta = r["meta"] or {}
        entry = meta.get("library_entry")
        if not entry:
            continue
        if g.get("require_check", True) and not (meta.get("check") or {}).get("passed"):
            continue
        out.append({**entry, "uri": r["uri"]})
    return out


def with_generated_beds(conn, policy: Policy) -> Policy:
    """The policy with the generated beds appended to sound.library.entries (unchanged when there are none)."""
    beds = generated_beds(conn, policy)
    if not beds:
        return policy
    sound = dict(policy.body.get("sound") or {})
    lib = dict(sound.get("library") or {})
    lib["entries"] = [*(lib.get("entries") or []), *beds]
    sound["library"] = lib
    return dataclasses.replace(policy, body={**policy.body, "sound": sound})
