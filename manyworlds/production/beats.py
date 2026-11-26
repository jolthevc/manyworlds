"""Audio-first blocking: a talking take's beats re-timed to its real voice line (performance direction 3.7).

The director times a shot's beats (`blocking[].at_s`) before any voice exists, against an estimate of how long the
lines take. Once the take's designed voice line is made, its real timing is known: each line's and clause's span
(voice.speak.<engine> meta `lines` / `clauses`, start_s/end_s), its word timestamps (meta.words, transcribed once for
the captions and cached there: production.captioning.voice_words) and the script's *emphasis* marks. The voice then
sets the timing, rhythm and emotional structure of the picture: before the talk job is built (production.talk), the
beats are moved onto it, and the talk prompt carries the moved times ("at 2.1 s: ...", direction.compile_prompts).

How a beat moves (policy `production.beats`; every move is within `snap_window_s` of where the director's estimate
lands on the real audio, so nothing is dragged across the take):
  - the director's time is first warped from the estimate (generation.talk_long.words_per_second plus each line's
    pause) onto the real line starts, so a beat keeps its place relative to the lines;
  - the first beat sits in the lead-in (clip time 0): she is already doing it when the line starts, never a still
    face that starts moving with the first word;
  - a beat that names a line's words (`relate_min_letters`, minus `relate_ignore`) moves to that line: its stressed
    word for a gesture, else its start;
  - a gesture (`kinds.gesture` words) lands on the nearest stressed word: an *emphasised* one first, else a word the
    voice itself stretched (duration per syllable at least `stress_ratio` x the line's median: lengthening is the
    cheapest acoustic mark of stress, and the only one the voice meta carries; no loudness or pitch contour per word
    is stored, so none is used);
  - head movement (`kinds.head`) lands on a stressed word of a rising phrase (a line ending in "?"; with none
    stressed, `rising_tail_s` before its end, where the pitch rises), else as a gesture;
  - a reaction (`kinds.reaction`) lands at the start of the nearest pause (a silence of `min_pause_s` or more between
    words or clauses), where a listener would see it;
  - the last beat (of two or more) sits in the end hold, `hold_offset_s` after the last word, so the take ends on her
    reaction and never on a return to a blank stare; unless it is a gesture already on a stressed word of the last
    line (the punchline's own gesture stays on it);
  - beats keep their order, at least `min_gap_s` apart, rounded to `round_s`, inside the clip.

Assumptions (unverified on a GPU): InfiniteTalk and LTX follow timed beats in a prompt only loosely; the gain is that
what the prompt asks for at a time agrees with what the voice does then. The director's own clock is assumed to be
the script at words_per_second with each line's pause_after_s (else `default_pause_s`) between lines. Only a take
that says exactly its shot's lines is re-timed (a shot split across takes for length, or an undirected take, keeps
its beats). A walk-and-talk take's silent motion clip is made before or alongside its voice line, so its own prompt
is not re-timed here (its lip-sync pass is, with no lead-in or hold).
"""

from __future__ import annotations

import copy
import re
from typing import Any, Sequence

from ..policy import Policy



def beats_cfg(policy: Policy) -> dict:
    return policy.get("production", "beats", default={}) or {}


def take_rolls(params: dict, *, segment: int | None = None, of: int = 1, moving: bool = False) -> tuple[float, float]:
    """The take's lead-in and end hold in seconds, as the talk worker renders them (worker talk_comfy.rolls: only a
    long take's first segment gets the lead-in and its last the hold; a walk-and-talk take none)."""
    if moving:
        return 0.0, 0.0
    i, n = int(segment or 0), int(of or 1)
    lead = float(params.get("lead_in_s") or 0) if i == 0 else 0.0
    hold = float(params.get("end_hold_s") or 0) if i == n - 1 else 0.0
    return max(lead, 0.0), max(hold, 0.0)


# --- the words -------------------------------------------------------------------------------------------------

def _timed_items(items) -> list[dict]:
    return [x for x in items or [] if isinstance(x, dict) and x.get("start_s") is not None
            and x.get("end_s") is not None]


def timed_words(shot: dict, meta: dict, words: Sequence[dict] | None, lead: float, caption_policy=None) -> list:
    """The take's words in clip time (captions.Word, with emphasis from the shot's *marked* lines): from the word
    timestamps when there are any, else estimated over the voice's line spans (captions.estimate_timings)."""
    from .. import captions
    from ..direction import spoken_lines
    marked = [ln["text"] for ln in spoken_lines(shot)]
    script = " ".join(marked)
    if words:
        try:
            return captions.word_timings(list(words), script=script or None, offset=lead)
        except Exception:  # noqa: BLE001 - a malformed cache entry: estimate instead
            pass
    lines = _timed_items(meta.get("lines"))
    if not lines:
        return []
    if len(lines) == len(marked):   # the voice's lines are the shot's: their marks give the emphasis
        lines = [{**ln, "text": text} for ln, text in zip(lines, marked)]
    return captions.estimate_timings(lines, offset=lead, policy=caption_policy)


def stressed(words: Sequence[Any], c: dict, *, measured: bool) -> set[int]:
    """Indices of stressed words: every *emphasised* one and, where the timings were measured (word timestamps, not
    estimates), the words the voice stretched (cuts.stressed_indices: duration per syllable at least `stress_ratio`
    x its stretch's median, `stress_min_letters` letters or more)."""
    from ..cuts import stressed_indices
    out = {i for i, w in enumerate(words) if getattr(w, "emphasis", False)}
    if measured and words:
        out |= stressed_indices(words, ratio=float(c.get("stress_ratio", 1.3)),
                                min_letters=int(c.get("stress_min_letters", 4)),
                                pause_s=float(c.get("min_pause_s", 0.25)))
    return out


def pauses(words: Sequence[Any], meta: dict, lead: float, c: dict) -> list[tuple[float, float]]:
    """Silences inside the speech, clip time: between words, and between the voice's clauses and lines, of at least
    `min_pause_s`. The lead-in and the end hold are not among them."""
    gap = float(c.get("min_pause_s", 0.25))
    spans = [(float(x["start_s"]) + lead, float(x["end_s"]) + lead)
             for x in _timed_items(meta.get("clauses")) or _timed_items(meta.get("lines"))]
    spans += [(float(w.start), float(w.end)) for w in words]
    spans.sort()
    out, end = [], None
    for a, b in spans:
        if end is not None and a - end >= gap:
            out.append((round(end, 3), round(a, 3)))
        end = b if end is None else max(end, b)
    return out


# --- the warp from the director's estimate onto the real voice ------------------------------------------------------

def _beat_warp(shot: dict, meta: dict, lead: float, c: dict, words_per_second: float):
    """A function from the director's time to clip time: piecewise linear through each line's estimated start ->
    its real start (when the voice has as many lines as the shot), and the estimated end -> the real end."""
    from ..direction import spoken_lines
    lines = _timed_items(meta.get("lines"))
    mine = spoken_lines(shot)
    wps = max(float(words_per_second or 2.5), 0.1)
    default_pause = float(c.get("default_pause_s", 0.4))
    est, t = [], 0.0
    for k, ln in enumerate(mine):
        est.append(t)
        t += len(ln["text"].split()) / wps
        if k < len(mine) - 1:
            t += float(ln["pause_after_s"]) if ln.get("pause_after_s") is not None else default_pause
    est_end = t
    real_start, real_end = lead + float(lines[0]["start_s"]), lead + float(lines[-1]["end_s"])
    if len(lines) == len(mine) and mine:
        pts = [(e, lead + float(ln["start_s"])) for e, ln in zip(est, lines)] + [(est_end, real_end)]
    else:
        pts = [(0.0, real_start), (est_end, real_end)]
    pts = sorted({p for p in pts})

    def f(x: float) -> float:
        if x <= pts[0][0]:
            return pts[0][1] + (x - pts[0][0])
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x <= x1:
                return y0 + (y1 - y0) * ((x - x0) / (x1 - x0) if x1 > x0 else 0.0)
        return pts[-1][1] + (x - pts[-1][0])
    return f


# --- kinds and relations ---------------------------------------------------------------------------------------

def _mentions_any(text: str, words) -> bool:
    return any(re.search(rf"\b{re.escape(str(w).lower())}", text) for w in words or [] if str(w).strip())


def beat_kind(beat: dict, c: dict) -> str:
    """'reaction', 'head', 'gesture' or 'other', by the first of `kinds` whose words start a word of the beat's
    action or hands (in that order: a laugh with a nod is a reaction)."""
    text = " ".join(str(beat.get(k) or "") for k in ("action", "hands")).lower()
    kinds = c.get("kinds") or {}
    for name in ("reaction", "head", "gesture"):
        if _mentions_any(text, kinds.get(name)):
            return name
    return "other"


def _content_words(text: str, c: dict) -> set[str]:
    from .. import captions
    ignore = {captions.key(w) for w in c.get("relate_ignore") or []}
    n = int(c.get("relate_min_letters", 4))
    return {k for k in (captions.key(w) for w in str(text or "").split()) if len(k) >= n and k not in ignore}


def related_line(beat: dict, shot: dict, c: dict) -> int | None:
    """The shot line whose words the beat names ("holds up the *mug*" and "this mug changed my mornings"), the
    first with the most words in common; None when none shares one."""
    from ..direction import spoken_lines
    mine = _content_words(" ".join(str(beat.get(k) or "") for k in ("action", "hands")), c)
    if not mine:
        return None
    best, score = None, 0
    for k, ln in enumerate(spoken_lines(shot)):
        n = len(mine & _content_words(ln["text"].replace("*", ""), c))
        if n > score:
            best, score = k, n
    return best


# --- the re-timing ---------------------------------------------------------------------------------------------

def _nearest_time(candidates: list[float], target: float, window: float) -> float | None:
    near = [t for t in candidates if abs(t - target) <= window]
    return min(near, key=lambda t: abs(t - target)) if near else None


def retime(shot: dict | None, meta: dict | None, words: Sequence[dict] | None, rolls: tuple[float, float],
           c: dict, *, words_per_second: float, caption_policy=None) -> dict:
    """The shot with its blocking re-timed to the voice (see the module docstring): {shot, moves}; `moves` lists
    each beat that moved {beat, action, kind, from, to, why}. The shot comes back unchanged (moves []) when it has no
    beats or the voice line has no line timings. Pure: no database, no files."""
    shot = shot if isinstance(shot, dict) else {}
    meta = meta or {}
    blocking = [b for b in shot.get("blocking") or [] if isinstance(b, dict)]
    lines = _timed_items(meta.get("lines"))
    if not blocking or not lines:
        return {"shot": shot, "moves": []}
    lead, tail = (float(r or 0) for r in rolls)
    ws = timed_words(shot, meta, words, lead, caption_policy)
    stress = stressed(ws, c, measured=bool(words))
    gaps = pauses(ws, meta, lead, c)
    speech_end = lead + float(lines[-1]["end_s"])
    seconds = meta.get("seconds")
    clip_end = lead + float(seconds) + tail if seconds is not None else speech_end + tail
    window, step = float(c.get("snap_window_s", 1.5)), float(c.get("round_s", 0.1))
    warp = _beat_warp(shot, meta, lead, c, words_per_second)
    starts = [lead + float(ln["start_s"]) for ln in lines]
    stressed_at = [ws[i].start for i in sorted(stress)]
    rising = []
    from ..direction import spoken_lines
    mine = spoken_lines(shot)
    for k, ln in enumerate(lines):
        text = mine[k]["text"] if len(mine) == len(lines) else str(ln.get("text") or "")
        if text.rstrip().rstrip('"\'').endswith("?"):
            a, b = lead + float(ln["start_s"]), lead + float(ln["end_s"])
            rising += [t for t in stressed_at if a <= t <= b] or [max(a, b - float(c.get("rising_tail_s", 0.3)))]
    last_line = (starts[-1], speech_end)

    order = sorted(range(len(blocking)), key=lambda i: float(blocking[i].get("at_s") or 0))
    placed: dict[int, tuple[float, str, str]] = {}
    for rank, i in enumerate(order):
        b = blocking[i]
        k = beat_kind(b, c)
        w = warp(float(b.get("at_s") or 0))
        t, why = None, ""
        if rank == 0:
            t, why = 0.0, "first beat: in the lead-in, already moving when the line starts"
        else:
            line = related_line(b, shot, c)
            if line is not None and line < len(lines):
                a, z = lead + float(lines[line]["start_s"]), lead + float(lines[line]["end_s"])
                on = [s for s in stressed_at if a <= s <= z]
                if k in ("gesture", "head") and on:
                    t, why = min(on, key=lambda s: abs(s - w)), f"on the stressed word of the line it names ({line})"
                else:
                    t, why = a, f"at the start of the line it names ({line})"
            elif k == "reaction":
                p = _nearest_time([a for a, _ in gaps], w, window)
                if p is not None:
                    t, why = p, "in a pause"
            elif k == "head":
                p = _nearest_time(rising, w, window) or _nearest_time(stressed_at, w, window)
                if p is not None:
                    t, why = p, "head movement on a rising phrase" if p in rising else "on a stressed word"
            elif k == "gesture":
                p = _nearest_time(stressed_at, w, window)
                if p is not None:
                    t, why = p, "on a stressed word"
            else:
                p = _nearest_time(starts, w, window)
                if p is not None:
                    t, why = p, "at a line start"
            last = rank == len(order) - 1
            punch = k == "gesture" and t is not None and last_line[0] <= t <= last_line[1] and t in stressed_at
            if last and tail > 0 and not punch:
                t, why = speech_end + float(c.get("hold_offset_s", 0.1)), "last beat: in the end hold, landing the line"
        if t is None:
            t, why = w, "kept where the director's time lands on the voice"
        placed[i] = (t, k, why)

    min_gap = float(c.get("min_gap_s", 0.4))
    top = max(0.0, clip_end - step)
    out_blocking, moves, prev = [], [], None
    for i in order:
        t, k, why = placed[i]
        if prev is not None:
            t = max(t, prev + min_gap)
        t = min(max(0.0, t), top)
        t = round(round(t / step) * step, 3) if step > 0 else round(t, 3)
        prev = t
        old = float(blocking[i].get("at_s") or 0)
        out_blocking.append({**copy.deepcopy(blocking[i]), "at_s": t})
        if abs(t - old) > 1e-6:
            moves.append({"beat": i, "action": str(blocking[i].get("action") or "")[:80], "kind": k,
                          "from": old, "to": t, "why": why})
    return {"shot": {**shot, "blocking": out_blocking}, "moves": moves}


# --- production ------------------------------------------------------------------------------------------------

def _said_words(text: str) -> list[str]:
    from .. import captions
    return [k for k in (captions.key(w) for w in str(text or "").replace("*", "").split()) if k]


def says_its_shot(shot: dict | None, meta: dict) -> bool:
    """Whether the voice line says exactly the shot's lines (so the shot's beats and the voice share one clock)."""
    from ..direction import spoken
    lines = _timed_items(meta.get("lines"))
    heard = _said_words(" ".join(str(ln.get("text") or "") for ln in lines))
    return bool(shot) and bool(heard) and heard == _said_words(spoken(shot))


def take_key(segment: int | None) -> str:
    return "take" if segment is None else str(segment)


def for_take(conn, policy: Policy, post: dict, body: dict, shot: dict | None, audio: dict | None, params: dict, *,
             segment: int | None = None, moving: bool = False, store=None) -> dict | None:
    """The take's shot with its beats re-timed to its voice line (retime), kept in body.talk_beats[take] with the
    moves; the shot as it was when re-timing is off, the take has no designed voice line with line timings, or the
    line doesn't say exactly the shot's lines. Word timestamps are read from the line's cache (meta.words) or, when
    the format's captions are timed from them anyway (captions.word_timestamps_for), transcribed once and cached
    (captioning.voice_words: the captions then reuse them for free); otherwise estimated from the line spans. A
    single beat only moves into the lead-in, which needs no words."""
    c = beats_cfg(policy)
    meta = (audio or {}).get("meta") or {}
    if not c.get("enabled") or not shot or not (shot.get("blocking") or []) or not says_its_shot(shot, meta):
        return shot
    from .. import captions
    words = meta.get("words") or None
    if (not words and audio.get("id") and len(shot["blocking"]) > 1
            and captions.word_timestamps_for(policy, post.get("format"))):
        from .captioning import voice_words
        words = voice_words(conn, policy, audio, store)
    of = len(body.get("talk_segments") or []) or 1
    rolls = take_rolls(params, segment=segment, of=of, moving=moving)
    wps = float(policy.get("generation", "talk_long", "words_per_second", default=2.5))
    made = retime(shot, meta, words, rolls, c, words_per_second=wps, caption_policy=policy)
    body.setdefault("talk_beats", {})[take_key(segment)] = {
        "blocking": made["shot"]["blocking"], "moves": made["moves"], "audio_uri": audio.get("uri"),
        "words": "transcript" if words else "estimated", "rolls": list(rolls)}
    return made["shot"]
