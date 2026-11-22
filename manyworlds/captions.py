"""Burned-in captions: the spoken words on the video, word-timed, styled, and inside the platforms' safe zones.

Owner 2026-11-22: captions are "the words that appear on the video itself, that match the words being verbally
spoken ... nice fonts and styles ... everything must be nicely done, look professional."

    word timings -> chunks (short phrases, at most two lines) -> layout (safe zones) -> ASS script -> ffmpeg burn

* Word timings come from a transcript with word timestamps (`word_timings`), or are estimated from the voice
  line's per-line spans (`estimate_timings`; voice.speak returns meta.lines with start_s/end_s per line), each
  line's time shared between its words by syllables and letters, with a little room after commas and full stops.
* Chunks (`chunk_words`) break at sentence and clause ends and before phrase starters, never after an article or
  a preposition, keep within the style's words, characters and lines, and never leave one word on its own unless
  it is a whole utterance by itself ("Okay." between pauses).
* Layout keeps text out of Reels/TikTok UI: the top and bottom bands and the right-side buttons (policy
  `burned_captions.safe_zone`), centred, in the lower third by default or upper-middle when the face is low
  (`placement_for`). A line too wide for the safe width is scaled down, never let out.
* Styles (`burned_captions.styles`) are presets: font, size (share of frame height), colours, outline, soft
  shadow, rounded box, letter and line spacing, case (incl. small caps), per-phrase / word-highlight (karaoke) /
  word-reveal modes, fade and pop. A creator's default is just a style name.
* Emphasis: *words in asterisks* in the script, or a list of words, take the style's accent colour.
* Rendering: an ASS script for libass (ffmpeg's `ass` filter, fonts from manyworlds/fonts via `fontsdir`). Where
  an ffmpeg has no libass, the same layout is drawn with Pillow into transparent PNGs and overlaid
  (`render_overlays`; static, no fade or pop).

Everything here is pure except `render_overlays` and `burn`.
"""

from __future__ import annotations

import itertools
import re
import struct
from dataclasses import dataclass, replace as _replace
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

PACKAGE_FONTS = Path(__file__).resolve().parent / "fonts"

_SENTENCE_END = re.compile(r"[.!?…]['\"”’)\]]*$")
_CLAUSE_END = re.compile(r"[,;:—–]['\"”’)\]]*$|^[-—–]+$")
_EMPHASIS = re.compile(r"\*([^*]+)\*")
_WORDLIKE = re.compile(r"\w")

# Fallbacks for a policy without the block's numbers; the shipped values live in policy/default.yaml.
_DEFAULT_SAFE = {"top": 0.14, "bottom": 0.25, "left": 0.12, "right": 0.12}
_DEFAULT_PLACEMENTS = {"lower": {"anchor": "bottom", "y": 0.74}, "upper": {"anchor": "top", "y": 0.24},
                       "center": {"anchor": "middle", "y": 0.5}}
_DEFAULT_COSTS = {"after_clause": 1.0, "before_starter": 2.0, "mid_phrase": 5.0, "after_sticky": 12.0,
                  "internal_sentence": 6.0, "size": 0.6, "over_duration_per_s": 8.0, "orphan": 100.0,
                  "line_balance": 0.15, "line_break_factor": 0.5, "two_lines": 1.0, "single_word_line": 2.0}


# --- policy ---------------------------------------------------------------------------------------------------

def _cfg(policy) -> dict:
    """The `burned_captions` block from a Policy, a policy body, or the block itself."""
    if policy is None:
        return {}
    if hasattr(policy, "body"):
        return policy.get("burned_captions", default={}) or {}
    if isinstance(policy, dict):
        return policy.get("burned_captions", policy) or {}
    return {}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def styles(policy) -> list[str]:
    return list((_cfg(policy).get("styles") or {}).keys())


def resolve_style(policy, name: str | None = None) -> tuple[str, dict]:
    """(name, style) for a style name (a creator's default caption style, say); an unknown or missing name gets
    `burned_captions.default_style`. Each style is laid over `style_defaults`."""
    cfg = _cfg(policy)
    table = cfg.get("styles") or {}
    if not table:
        raise ValueError("no caption styles in policy (burned_captions.styles)")
    default = cfg.get("default_style")
    chosen = name if name in table else default if default in table else next(iter(table))
    return chosen, _merge(cfg.get("style_defaults") or {}, table[chosen])


def fonts_dir(policy=None) -> Path | None:
    """Where the caption fonts are: policy `burned_captions.fonts_dir` if set, the package's fonts/ directory, or
    manyworlds/fonts under the working directory (a non-editable install leaves the TTFs out of site-packages
    unless pyproject's package-data lists them; the images run from /app, where the source tree is)."""
    given = _cfg(policy).get("fonts_dir")
    for p in ([Path(given)] if given else []) + [PACKAGE_FONTS, Path.cwd() / "manyworlds" / "fonts"]:
        if p.is_dir() and any(p.glob("*.ttf")):
            return p
    return None


# --- words and timings ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Word:
    text: str            # as shown (punctuation kept, emphasis asterisks gone)
    start: float         # seconds in the video
    end: float
    emphasis: bool = False


def key(text: str) -> str:
    """A word as compared: lower case, letters, digits and apostrophes only."""
    return re.sub(r"[^a-z0-9']", "", str(text or "").lower().replace("’", "'"))


def _glue(prev: str, tok: str) -> str:
    return f"{prev} {tok}" if tok[:1] in "—–-" else prev + tok


def parse_emphasis(text: str) -> list[tuple[str, bool]]:
    """Words of `text` with their emphasis: "*so* good" -> [("so", True), ("good", False)]. Spans of several
    words work ("*this whole thing*"); punctuation on its own joins the word before it."""
    text = text or ""
    raw: list[tuple[str, bool]] = []
    pos = 0
    for m in _EMPHASIS.finditer(text):
        raw += [(w, False) for w in text[pos:m.start()].split()]
        raw += [(w, True) for w in m.group(1).split()]
        pos = m.end()
    raw += [(w, False) for w in text[pos:].split()]
    out: list[tuple[str, bool]] = []
    for w, emph in raw:
        w = w.replace("*", "")
        if not w:
            continue
        if not _WORDLIKE.search(w) and out:
            out[-1] = (_glue(out[-1][0], w), out[-1][1])
            continue
        out.append((w, emph))
    return out


def _marks(emphasis: Iterable[str] | None) -> set[str]:
    return {key(w) for phrase in (emphasis or ()) for w in str(phrase).split() if key(w)}


def syllables(word: str) -> int:
    """A rough syllable count (vowel groups, a silent final e), at least 1; digits count one each."""
    w = re.sub(r"[^a-z]", "", (word or "").lower())
    digits = len(re.sub(r"\D", "", word or ""))
    if not w:
        return max(digits, 1)
    n = len(re.findall(r"[aeiouy]+", w))
    if n > 1 and w.endswith("e") and not w.endswith(("le", "ee", "ye")):
        n -= 1
    return max(n, 1) + digits


def estimate_timings(lines: Sequence[dict], *, emphasis: Iterable[str] | None = None, offset: float = 0.0,
                     policy=None) -> list[Word]:
    """Word timings from known line spans: [{text, start_s, end_s}] (voice.speak's meta.lines; `start`/`end` also
    read). Each line's span is shared between its words by syllables and letters (`timing.syllable_weight`,
    `char_weight`), with room after clause and sentence ends (`clause_pause_weight`, `sentence_pause_weight`).
    The first word starts at its line's start and the last ends at its line's end. `offset` shifts every time
    (where the voice line starts in the video). Words in *asterisks*, or in `emphasis`, are emphasised."""
    tc = _cfg(policy).get("timing") or {}
    syl_w, chr_w = float(tc.get("syllable_weight", 1.0)), float(tc.get("char_weight", 0.08))
    clause_p, sent_p = float(tc.get("clause_pause_weight", 0.8)), float(tc.get("sentence_pause_weight", 1.4))
    marks = _marks(emphasis)
    spans = []
    for line in lines or []:
        a = line.get("start_s", line.get("start"))
        b = line.get("end_s", line.get("end"))
        toks = parse_emphasis(str(line.get("text") or ""))
        if a is not None and b is not None and toks:
            spans.append((float(a), float(b), toks))
    spans.sort(key=lambda s: s[0])
    out: list[Word] = []
    floor = float("-inf")
    for a, b, toks in spans:
        a = max(a, floor)
        b = max(b, a)
        weights = [max(syl_w * syllables(w) + chr_w * len(key(w)), 1e-6) for w, _ in toks]
        pauses = [sent_p if _SENTENCE_END.search(w) else clause_p if _CLAUSE_END.search(w) else 0.0
                  for w, _ in toks[:-1]] + [0.0]
        unit = (b - a) / (sum(weights) + sum(pauses))
        t = a
        for i, ((w, emph), wt, p) in enumerate(zip(toks, weights, pauses)):
            e = b if i == len(toks) - 1 else t + wt * unit
            out.append(Word(w, round(t + offset, 3), round(e + offset, 3), emph or key(w) in marks))
            t = e + p * unit
        floor = b
    return out


def word_timings(words: Sequence[dict], *, script: str | None = None, emphasis: Iterable[str] | None = None,
                 offset: float = 0.0) -> list[Word]:
    """Word timings from a transcript with word timestamps: [{word, start, end}] (`text`, `start_s`, `end_s`
    also read; Whisper's verbose_json `words`). Punctuation-only tokens join the word before; times are made
    monotonic (each word starts no earlier than the one before and ends no later than the next starts).
    With `script`, the script's spelling, casing, punctuation and *emphasis* replace the transcript's wherever
    the words line up (the transcript is lower-case and unpunctuated on some hosts)."""
    items: list[list[Any]] = []
    for it in words or []:
        text = str(it.get("word", it.get("text")) or "").strip()
        a = it.get("start", it.get("start_s"))
        b = it.get("end", it.get("end_s"))
        if text and a is not None and b is not None:
            items.append([text, float(a), float(b)])
    items.sort(key=lambda x: x[1])
    merged: list[list[Any]] = []
    for text, a, b in items:
        if not _WORDLIKE.search(text):
            if merged:
                merged[-1][0] = _glue(merged[-1][0], text)
            continue
        merged.append([text, a, b])
    marks = _marks(emphasis)
    out = []
    for i, (text, a, b) in enumerate(merged):
        e = max(b, a)
        if i + 1 < len(merged):
            e = min(e, merged[i + 1][1])
        out.append(Word(text, round(a + offset, 3), round(e + offset, 3), key(text) in marks))
    return _align_script(out, script) if script else out


def _align_script(words: list[Word], script: str) -> list[Word]:
    toks = parse_emphasis(script)
    if not toks:
        return words
    a, b = [key(w.text) for w in words], [key(t) for t, _ in toks]
    out = list(words)
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            for i, j in zip(range(i1, i2), range(j1, j2)):
                out[i] = _replace(out[i], text=toks[j][0], emphasis=out[i].emphasis or toks[j][1])
    return out


# --- chunks ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    lines: tuple[tuple[Word, ...], ...]   # one or two lines
    start: float                          # on screen from
    end: float                            # until

    @property
    def words(self) -> tuple[Word, ...]:
        return tuple(w for line in self.lines for w in line)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def _ends_sentence(w: Word) -> bool:
    return bool(_SENTENCE_END.search(w.text))


def _break_cost(prev: Word, nxt: Word | None, p: dict) -> float:
    """What a break between `prev` and `nxt` costs: free at a sentence end or a pause, cheap after a clause end
    or before a phrase starter, dear in mid-phrase, dearest after an article or preposition."""
    if nxt is None or _ends_sentence(prev) or nxt.start - prev.end >= p["pause_s"]:
        return 0.0
    c = p["costs"]
    if _CLAUSE_END.search(prev.text):
        return c["after_clause"]
    if key(prev.text) in p["sticky"]:
        return c["after_sticky"]
    if key(nxt.text) in p["starters"]:
        return c["before_starter"]
    return c["mid_phrase"]


def standalone(words: Sequence[Word], i: int, pause_s: float) -> bool:
    """Is word i an utterance by itself (a sentence or a pause on both sides)? Only such a word may be a chunk
    on its own."""
    n = len(words)
    before = i == 0 or _ends_sentence(words[i - 1]) or words[i].start - words[i - 1].end >= pause_s
    after = i == n - 1 or _ends_sentence(words[i]) or words[i + 1].start - words[i].end >= pause_s
    return before and after


def _shown(text: str, case: str) -> str:
    return text.upper() if case in ("upper", "smallcaps") else text.lower() if case == "lower" else text


def _wrap(seg: Sequence[Word], p: dict) -> tuple[tuple[tuple[Word, ...], ...] | None, float]:
    """The chunk as one line, or two balanced lines; None if it fits neither. A chunk of four words or more puts
    at least two on each line; two or three words may stack one over the other (narrow, big styles). A line of
    one over-long word is allowed (layout scales it down)."""
    texts = [_shown(w.text, p["case"]) for w in seg]
    if len(seg) == 1 or len(" ".join(texts)) <= p["max_chars"]:
        return (tuple(seg),), 0.0
    if p["max_lines"] < 2:
        return None, 0.0
    c, best = p["costs"], None
    least = 2 if len(seg) >= 4 else 1
    for k in range(least, len(seg) - least + 1):
        l1, l2 = len(" ".join(texts[:k])), len(" ".join(texts[k:]))
        if (l1 > p["max_chars"] and k > 1) or (l2 > p["max_chars"] and len(seg) - k > 1):
            continue
        cost = c["line_balance"] * abs(l1 - l2) + c["line_break_factor"] * _break_cost(seg[k - 1], seg[k], p)
        if k == 1 or len(seg) - k == 1:
            cost += c["single_word_line"]
        if best is None or cost < best[0]:
            best = (cost, k)
    if best is None:
        return None, 0.0
    cost, k = best
    return (tuple(seg[:k]), tuple(seg[k:])), cost + c["two_lines"]


def _chunk_params(policy, style: dict | None) -> dict:
    cfg = _cfg(policy)
    ch = cfg.get("chunking") or {}
    style = style or {}
    return {
        "max_words": max(1, int(style.get("max_words", ch.get("max_words", 6)))),
        "target": float(style.get("target_words", ch.get("target_words", 4))),
        "max_chars": int(style.get("max_chars_per_line", ch.get("max_chars_per_line", 22))),
        "max_lines": min(2, int(style.get("max_lines", ch.get("max_lines", 2)))),
        "max_dur": float(ch.get("max_duration_s", 2.8)),
        "gap_break": float(ch.get("gap_break_s", 0.6)),
        "pause_s": float(ch.get("pause_s", 0.3)),
        "costs": {**_DEFAULT_COSTS, **(ch.get("costs") or {})},
        "starters": {key(w) for w in ch.get("phrase_starters") or []},
        "sticky": {key(w) for w in ch.get("sticky_words") or []},
        "case": str(style.get("case", "none")),
    }


def chunk_words(words: Sequence[Word], *, policy=None, style: dict | None = None) -> list[Chunk]:
    """Group timed words into caption chunks (a dynamic programme over break points). Hard limits: the style's
    `max_words`, `max_chars_per_line` (a single over-long word excepted; layout scales it down), at most two
    lines, and no chunk across a silence longer than `chunking.gap_break_s`. Costs (`chunking.costs`) prefer
    sentence and clause ends, phrase starters, chunks of about `target_words`, at most `max_duration_s`, and
    make a lone word that is not an utterance by itself (an orphan) a last resort.
    Timing (`timing`): a chunk shows `lead_s` before its first word, holds `hold_s` after its last, stays at least
    `min_display_s`, never overlaps the next, and runs on to the next when the gap is under `bridge_s`."""
    words = list(words)
    n = len(words)
    if not n:
        return []
    p = _chunk_params(policy, style)
    c = p["costs"]
    inf = float("inf")
    best, back = [inf] * (n + 1), [0] * (n + 1)
    lines_at: list[Any] = [None] * (n + 1)
    best[0] = 0.0
    for i in range(n):
        if best[i] == inf:
            continue
        for j in range(i + 1, min(n, i + p["max_words"]) + 1):
            if j - i > 1 and words[j - 1].start - words[j - 2].end > p["gap_break"]:
                break
            seg = words[i:j]
            lines, cost = _wrap(seg, p)
            if lines is None:
                continue
            cost += _break_cost(words[j - 1], words[j] if j < n else None, p)
            cost += sum(c["internal_sentence"] for w in seg[:-1] if _ends_sentence(w))
            cost += c["size"] * (len(seg) - p["target"]) ** 2
            dur = seg[-1].end - seg[0].start
            if dur > p["max_dur"]:
                cost += c["over_duration_per_s"] * (dur - p["max_dur"])
            if len(seg) == 1 and not standalone(words, i, p["pause_s"]):
                cost += c["orphan"]
            if best[i] + cost < best[j]:
                best[j], back[j], lines_at[j] = best[i] + cost, i, lines
    spans, j = [], n
    while j > 0:
        spans.append((back[j], j, lines_at[j]))
        j = back[j]
    spans.reverse()
    return _timed(words, spans, _cfg(policy).get("timing") or {})


def _timed(words: list[Word], spans: list, tc: dict) -> list[Chunk]:
    lead, hold = float(tc.get("lead_s", 0.05)), float(tc.get("hold_s", 0.25))
    min_disp, bridge = float(tc.get("min_display_s", 0.6)), float(tc.get("bridge_s", 0.35))
    starts, prev_end = [], 0.0
    for i, j, _ in spans:
        starts.append(max(words[i].start - lead, prev_end, 0.0))
        prev_end = words[j - 1].end
    out = []
    for k, (i, j, lines) in enumerate(spans):
        last = words[j - 1].end
        end = max(last + hold, starts[k] + min_disp)
        if k + 1 < len(spans):
            nxt = starts[k + 1]
            end = min(end, nxt)
            if nxt - end < bridge:
                end = nxt
        out.append(Chunk(lines=lines, start=round(starts[k], 3), end=round(max(end, last, starts[k]), 3)))
    return out


# --- fonts and layout ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FontMetrics:
    upem: int
    ascender: int        # hhea, font units (libass's line box)
    descender: int       # hhea, positive
    win_ascent: int      # OS/2 (libass sizes text so win_ascent + win_descent = the ASS font size)
    win_descent: int
    cap_height: int
    x_height: int


_DEFAULT_METRICS = FontMetrics(1000, 1050, 350, 1135, 627, 700, 550)


@lru_cache(maxsize=None)
def font_metrics(path: str) -> FontMetrics:
    """The few metrics captions need, read from a TTF's head, hhea and OS/2 tables."""
    data = Path(path).read_bytes()
    tables = {}
    for i in range(struct.unpack_from(">H", data, 4)[0]):
        tag, _, off, _ = struct.unpack_from(">4sIII", data, 12 + 16 * i)
        tables[tag.decode("latin-1")] = off
    upem = struct.unpack_from(">H", data, tables["head"] + 18)[0]
    asc, desc = struct.unpack_from(">hh", data, tables["hhea"] + 4)
    os2 = tables.get("OS/2")
    if os2 is None:
        return FontMetrics(upem, asc, -desc, asc, -desc, int(upem * 0.7), int(upem * 0.5))
    wa, wd = struct.unpack_from(">HH", data, os2 + 74)
    version = struct.unpack_from(">H", data, os2)[0]
    xh, ch = struct.unpack_from(">hh", data, os2 + 86) if version >= 2 else (int(upem * 0.5), int(upem * 0.7))
    return FontMetrics(upem, asc, -desc, wa or asc, wd or -desc, ch, xh)


@lru_cache(maxsize=128)
def _pil_font(path: str | None, size: int):
    try:
        from PIL import ImageFont
        if path:
            return ImageFont.truetype(path, size)
        return ImageFont.load_default(size=size)
    except Exception:
        return None


@dataclass(frozen=True)
class _Setup:
    em: float            # px, the font's em at scale 1
    ass_size: float      # the ASS Fontsize that renders that em
    metrics: FontMetrics
    font_path: str | None
    line_h: float        # baseline to baseline, px
    asc: float           # px above a baseline (hhea)
    desc: float          # px below
    spacing: float       # letter spacing, px
    outline: float
    shadow: float
    soften: float
    char_w: float        # em per character when the font cannot be measured


def _setup(style: dict, height: int, fonts: Path | None) -> _Setup:
    em = float(style.get("size", 0.036)) * height
    path = fonts / style["file"] if fonts is not None and style.get("file") else None
    path_s = str(path) if path is not None and path.exists() else None
    m = font_metrics(path_s) if path_s else _DEFAULT_METRICS
    return _Setup(em=em, ass_size=em * (m.win_ascent + m.win_descent) / m.upem, metrics=m, font_path=path_s,
                  line_h=em * float(style.get("line_spacing", 1.18)), asc=em * m.ascender / m.upem,
                  desc=em * m.descender / m.upem, spacing=em * float(style.get("letter_spacing", 0.0)),
                  outline=em * float((style.get("outline") or {}).get("width", 0.0)),
                  shadow=em * float((style.get("shadow") or {}).get("depth", 0.0)),
                  soften=em * float(style.get("soften", 0.0)), char_w=float(style.get("char_width", 0.58)))


def _runs(text: str, style: dict) -> list[tuple[str, float]]:
    """A word as drawn: [(text, size factor)]; small caps are capitals at `smallcaps_scale` for lower case."""
    case = style.get("case", "none")
    if case != "smallcaps":
        return [(_shown(text, case), 1.0)]
    sc = float(style.get("smallcaps_scale", 0.8))
    return [("".join(g).upper(), sc if low else 1.0) for low, g in itertools.groupby(text, key=str.islower)]


def _advance(text: str, st: _Setup, size: float) -> float:
    font = _pil_font(st.font_path, max(1, int(round(size)))) if st.font_path else None
    if font is not None:
        return float(font.getlength(text))
    return len(text) * st.char_w * size


def _line_width(words: Sequence[Word], style: dict, st: _Setup) -> float:
    total, chars = 0.0, 0
    for i, w in enumerate(words):
        runs = ([(" ", 1.0)] if i else []) + _runs(w.text, style)
        for text, k in runs:
            total += _advance(text, st, st.em * k)
            chars += len(text)
    return total + st.spacing * max(chars - 1, 0)


@dataclass(frozen=True)
class CaptionPlan:
    chunks: tuple[Chunk, ...]
    style_name: str
    style: dict
    width: int
    height: int
    placement: str
    cfg: dict
    fonts_dir: Path | None


@dataclass(frozen=True)
class LineBox:
    words: tuple[Word, ...]
    first: int           # index of the line's first word in its chunk
    x0: float            # the text's extent, px: centre +- width/2, ascender to descender
    y0: float
    x1: float
    y1: float
    baseline: float
    width: float


@dataclass(frozen=True)
class ChunkLayout:
    chunk: Chunk
    lines: tuple[LineBox, ...]
    scale: float         # 1, or less when a line would be wider than the safe width


def safe_area(width: int, height: int, policy=None) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) in px where captions may go: the frame minus `burned_captions.safe_zone` (shares of the
    frame: Reels/TikTok put the account, caption and buttons in the bottom ~25%, the top bar in ~14%, the like /
    comment / share column on the right ~12%; the left margin matches it, so centred text clears the buttons)."""
    z = {**_DEFAULT_SAFE, **(_cfg(policy).get("safe_zone") or {})}
    return (width * float(z["left"]), height * float(z["top"]),
            width * (1 - float(z["right"])), height * (1 - float(z["bottom"])))


def layout(plan: CaptionPlan) -> list[ChunkLayout]:
    """Where each chunk's lines go, in px of the plan's frame."""
    st = _setup(plan.style, plan.height, plan.fonts_dir)
    sx0, sy0, sx1, sy1 = safe_area(plan.width, plan.height, plan.cfg)
    places = plan.cfg.get("placements") or _DEFAULT_PLACEMENTS
    place = places.get(plan.placement) or _DEFAULT_PLACEMENTS.get(plan.placement) or _DEFAULT_PLACEMENTS["lower"]
    box = plan.style.get("box") or {}
    pad = st.em * float(box.get("pad_x", 0.0)) if box.get("enabled") else 0.0
    avail = (sx1 - sx0) - 2 * (pad + st.outline)
    cx = (sx0 + sx1) / 2
    out = []
    for chunk in plan.chunks:
        widths = [_line_width(line, plan.style, st) for line in chunk.lines]
        widest = max(widths) if widths else 0.0
        scale = min(1.0, avail / widest) if widest > 0 else 1.0
        asc, desc, line_h = st.asc * scale, st.desc * scale, st.line_h * scale
        block = asc + desc + (len(chunk.lines) - 1) * line_h
        y = float(place.get("y", 0.74)) * plan.height
        anchor = place.get("anchor", "bottom")
        top = y if anchor == "top" else y - block / 2 if anchor == "middle" else y - block
        top = max(sy0, min(top, sy1 - block))
        boxes, first = [], 0
        for i, (line, w) in enumerate(zip(chunk.lines, widths)):
            base = top + asc + i * line_h
            boxes.append(LineBox(line, first, cx - w * scale / 2, base - asc, cx + w * scale / 2, base + desc, base,
                                 w * scale))
            first += len(line)
        out.append(ChunkLayout(chunk, tuple(boxes), scale))
    return out


def placement_for(face_box: Sequence[float] | None, policy=None) -> str:
    """'upper' when the face (x0, y0, x1, y1 as shares of the frame) reaches into the lower caption band and
    clears the upper one; else the default placement. The bands are `face_band` tall under / over the
    placements' anchors."""
    cfg = _cfg(policy)
    default = cfg.get("default_placement", "lower")
    if not face_box:
        return default
    places = cfg.get("placements") or _DEFAULT_PLACEMENTS
    band = float(cfg.get("face_band", 0.14))
    lo = float((places.get("lower") or {}).get("y", 0.74))
    up = float((places.get("upper") or {}).get("y", 0.24))
    _, fy0, _, fy1 = (float(v) for v in face_box)
    hits_lower = fy1 > lo - band and fy0 < lo
    hits_upper = fy1 > up and fy0 < up + band
    return "upper" if hits_lower and not hits_upper and "upper" in places else default


def make_plan(words: Sequence[Word], policy=None, *, width: int, height: int, style: str | None = None,
              placement: str | None = None, fonts: Path | None = None) -> CaptionPlan:
    """Chunks, style and placement for one video of `width` x `height` (1080x1920, 720x1280, ...)."""
    cfg = _cfg(policy)
    name, st = resolve_style(policy, style)
    places = cfg.get("placements") or _DEFAULT_PLACEMENTS
    default = cfg.get("default_placement", "lower")
    place = placement if placement in places else default if default in places else next(iter(places))
    return CaptionPlan(tuple(chunk_words(words, policy=policy, style=st)), name, st, int(width), int(height),
                       place, cfg, Path(fonts) if fonts is not None else fonts_dir(policy))


def plan_from_lines(lines: Sequence[dict], policy=None, *, width: int, height: int, script: str | None = None,
                    words: Sequence[dict] | None = None, emphasis: Iterable[str] | None = None, offset: float = 0.0,
                    style: str | None = None, placement: str | None = None) -> CaptionPlan:
    """The join's entry point: word timestamps from a transcript when there are any (`words`), else estimated
    from the voice line's `lines`; `script` (with *emphasis*) supplies the shown text and emphasis."""
    if words:
        timed = word_timings(words, script=script, emphasis=emphasis, offset=offset)
    else:
        timed = estimate_timings(lines, emphasis=emphasis, offset=offset, policy=policy)
        if script:
            timed = _align_script(timed, script)
    return make_plan(timed, policy, width=width, height=height, style=style, placement=placement)


# --- ASS ------------------------------------------------------------------------------------------------------

def _rgb(colour: str) -> tuple[int, int, int]:
    s = str(colour).lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _alpha(opacity: float) -> int:
    return max(0, min(255, int(round((1 - float(opacity)) * 255))))


def ass_colour(colour: str) -> str:
    """'#RRGGBB' as an ASS override colour, '&HBBGGRR&'."""
    r, g, b = _rgb(colour)
    return f"&H{b:02X}{g:02X}{r:02X}&"


def _style_colour(colour: str, opacity: float = 1.0) -> str:
    r, g, b = _rgb(colour)
    return f"&H{_alpha(opacity):02X}{b:02X}{g:02X}{r:02X}"


def ass_time(seconds: float) -> str:
    """h:mm:ss.cc"""
    return _cs_time(int(round(max(0.0, seconds) * 100)))


def _cs_time(cs: int) -> str:
    h, rem = divmod(max(cs, 0), 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _esc(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")")


def _n(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _states(chunk: Chunk, mode: str) -> list[tuple[float, float, int | None]]:
    """(start, end, active word) per rendering: one for a phrase, one per word for karaoke and reveal."""
    if mode not in ("karaoke", "reveal"):
        return [(chunk.start, chunk.end, None)]
    words = chunk.words
    times = [chunk.start] + [max(w.start, chunk.start) for w in words[1:]] + [chunk.end]
    return [(times[k], max(times[k + 1], times[k]), k) for k in range(len(words))]


def _rounded_rect(w: float, h: float, r: float) -> str:
    k = 0.5523 * r
    pts = [("m", r, 0), ("l", w - r, 0), ("b", w - r + k, 0, w, r - k, w, r), ("l", w, h - r),
           ("b", w, h - r + k, w - r + k, h, w - r, h), ("l", r, h), ("b", r - k, h, 0, h - r + k, 0, h - r),
           ("l", 0, r), ("b", 0, r - k, r - k, 0, r, 0)]
    return " ".join(p[0] + " " + " ".join(str(int(round(v))) for v in p[1:]) for p in pts)


def _box_geometry(line: LineBox, cl: ChunkLayout, st: _Setup, box: dict) -> tuple[float, float, float, float, float]:
    """A line's rounded box: (x0, y0, w, h, radius) px, centred on the line's mid-letter height."""
    em, m = st.em * cl.scale, st.metrics
    pad, h = em * float(box.get("pad_x", 0.35)), em * float(box.get("height", 1.3))
    mid = line.baseline - em * (m.x_height + m.cap_height) / (4 * m.upem)
    return line.x0 - pad, mid - h / 2, line.width + 2 * pad, h, min(em * float(box.get("radius", 0.3)), h / 2)


def _pop_tags(pop: dict, base: float) -> str:
    frm, peak, ms = float(pop.get("from", 1.0)), float(pop.get("peak", 1.0)), int(pop.get("ms", 0))
    if ms <= 0 or (frm == 1.0 and peak == 1.0):
        return ""
    tags = f"\\fscx{_n(frm * base)}\\fscy{_n(frm * base)}"
    if peak == 1.0:
        return tags + f"\\t(0,{ms},\\fscx{_n(base)}\\fscy{_n(base)})"
    t1 = int(round(ms * 0.55))
    return (tags + f"\\t(0,{t1},\\fscx{_n(peak * base)}\\fscy{_n(peak * base)})"
            f"\\t({t1},{ms},\\fscx{_n(base)}\\fscy{_n(base)})")


def _line_body(line: LineBox, active: int | None, style: dict, st: _Setup, base: float) -> str:
    mode = style.get("mode", "phrase")
    primary = ass_colour(style["color"])
    accent = ass_colour(style.get("accent") or style["color"])
    highlight = ass_colour(style.get("highlight") or style.get("accent") or style["color"])
    a1 = f"&H{_alpha(1.0):02X}&"
    outline = style.get("outline") or {}
    a3 = f"&H{_alpha(outline.get('opacity', 1.0) if st.outline > 0 else 0.0):02X}&"
    a4 = f"&H{_alpha((style.get('shadow') or {}).get('opacity', 1.0)):02X}&"
    word_pop = style.get("word_pop") or {}
    wp, wp_ms = float(word_pop.get("from", 1.0)), int(word_pop.get("ms", 0))
    parts = []
    for j, w in enumerate(line.words):
        g = line.first + j
        if j:
            parts.append(" ")
        opened, closed = [], []
        is_active = mode == "karaoke" and active is not None and g == active
        colour = highlight if is_active else accent if w.emphasis else primary
        if colour != primary:
            opened.append(f"\\1c{colour}")
            closed.append(f"\\1c{primary}")
        if mode == "reveal" and active is not None and g > active:
            opened.append("\\1a&HFF&\\3a&HFF&\\4a&HFF&")
            closed.append(f"\\1a{a1}\\3a{a3}\\4a{a4}")
        if is_active and wp != 1.0 and wp_ms > 0:
            opened.append(f"\\fscx{_n(wp * base)}\\fscy{_n(wp * base)}\\t(0,{wp_ms},\\fscx{_n(base)}\\fscy{_n(base)})")
            closed.append(f"\\fscx{_n(base)}\\fscy{_n(base)}")
        text = "".join(_esc(t) if k == 1.0 else f"{{\\fs{_n(st.ass_size * k)}}}{_esc(t)}{{\\fs{_n(st.ass_size)}}}"
                       for t, k in _runs(w.text, style))
        parts.append(("{" + "".join(opened) + "}" if opened else "") + text
                     + ("{" + "".join(closed) + "}" if closed else ""))
    return "".join(parts)


def to_ass(plan: CaptionPlan) -> str:
    """The plan as an ASS script at the video's own resolution (PlayRes = width x height, so every number is a
    pixel). One Style line; per chunk one Dialogue per line (per line and word for karaoke / reveal), each placed
    with \\an2\\pos at its line's bottom; rounded boxes are drawings on layer 0 under the text on layer 1."""
    s, st = plan.style, _setup(plan.style, plan.height, plan.fonts_dir)
    name = re.sub(r"[^A-Za-z0-9_-]", "_", plan.style_name) or "captions"
    outline, shadow, box = s.get("outline") or {}, s.get("shadow") or {}, s.get("box") or {}
    outline_px = st.outline
    if st.soften > 0 and outline_px <= 0:
        outline_px = 1.0      # libass blurs the border; without one it would blur the letters (border kept invisible)
    outline_colour = _style_colour(outline.get("color", "#000000"),
                                   outline.get("opacity", 1.0) if st.outline > 0 else 0.0)
    head = [
        "[Script Info]",
        f"; manyworlds captions, style {plan.style_name}",
        "ScriptType: v4.00+",
        f"PlayResX: {plan.width}",
        f"PlayResY: {plan.height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "Kerning: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: {name},{s.get('font', 'Poppins')},{_n(st.ass_size)},{_style_colour(s['color'])},"
        f"{_style_colour(s['color'])},{outline_colour},"
        f"{_style_colour(shadow.get('color', '#000000'), shadow.get('opacity', 0.0))},"
        f"{-1 if s.get('bold') else 0},{-1 if s.get('italic') else 0},0,0,100,100,{_n(st.spacing)},0,1,"
        f"{_n(outline_px)},{_n(st.shadow)},2,0,0,0,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    mode = s.get("mode", "phrase")
    fade_in, fade_out = int(s.get("fade_in_ms", 0)), int(s.get("fade_out_ms", 0))
    events: list[tuple[int, int, str]] = []
    for cl in layout(plan):
        states = [(int(round(a * 100)), int(round(b * 100)), k) for a, b, k in _states(cl.chunk, mode)]
        states = [x for x in states if x[1] > x[0]]
        if not states:
            continue
        c0, c1 = states[0][0], states[-1][1]
        base = 100.0 * cl.scale
        if box.get("enabled"):
            fi, fo = min(fade_in, (c1 - c0) * 5), min(fade_out, (c1 - c0) * 5)
            fad = f"\\fad({fi},{fo})" if fi or fo else ""
            for line in cl.lines:
                x0, y0, w, h, r = _box_geometry(line, cl, st, box)
                text = (f"{{\\an7\\pos({_n(x0)},{_n(y0)})\\bord0\\shad0\\blur0\\1c{ass_colour(box.get('color', '#000000'))}"
                        f"\\1a&H{_alpha(box.get('opacity', 0.6)):02X}&{fad}\\p1}}{_rounded_rect(w, h, r)}{{\\p0}}")
                events.append((c0, 0, f"Dialogue: 0,{_cs_time(c0)},{_cs_time(c1)},{name},,0,0,0,,{text}"))
        for si, (a, b, k) in enumerate(states):
            dur = (b - a) * 10
            fi = min(fade_in, dur // (2 if len(states) == 1 else 1)) if si == 0 else 0
            fo = min(fade_out, dur - fi) if si == len(states) - 1 else 0
            for line in cl.lines:
                tags = f"\\an2\\pos({_n((line.x0 + line.x1) / 2)},{_n(line.baseline + st.desc * cl.scale)})"
                if fi or fo:
                    tags += f"\\fad({fi},{fo})"
                if st.soften > 0:
                    tags += f"\\blur{_n(st.soften * cl.scale)}"
                pop = _pop_tags(s.get("pop") or {}, base) if si == 0 else ""
                tags += pop or (f"\\fscx{_n(base)}\\fscy{_n(base)}" if cl.scale < 1.0 else "")
                text = "{" + tags + "}" + _line_body(line, k, s, st, base)
                events.append((a, 1, f"Dialogue: 1,{_cs_time(a)},{_cs_time(b)},{name},,0,0,0,,{text}"))
    events.sort(key=lambda e: (e[0], e[1]))
    return "\n".join(head + [e[2] for e in events]) + "\n"


def write_ass(plan: CaptionPlan, path: Path) -> Path:
    path = Path(path)
    path.write_text(to_ass(plan), encoding="utf-8")
    return path


# --- fallback renderer (an ffmpeg without libass) ---------------------------------------------------------------

def _rgba(colour: str, opacity: float = 1.0) -> tuple[int, int, int, int]:
    return (*_rgb(colour), 255 - _alpha(opacity))


def _draw_state(plan: CaptionPlan, cl: ChunkLayout, active: int | None, st: _Setup):
    """One rendering of a chunk as a transparent full-frame PNG image: box, soft shadow, outline, fill."""
    from PIL import Image, ImageDraw, ImageFilter

    s, W, H = plan.style, plan.width, plan.height
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    em = st.em * cl.scale
    margin = int(em * 2)
    top = max(0, int(min(l.y0 for l in cl.lines)) - margin)
    bottom = min(H, int(max(l.y1 for l in cl.lines)) + margin)
    if bottom <= top:
        return img
    mode = s.get("mode", "phrase")
    colours = {"primary": _rgba(s["color"]), "accent": _rgba(s.get("accent") or s["color"]),
               "highlight": _rgba(s.get("highlight") or s.get("accent") or s["color"])}

    def runs():
        for line in cl.lines:
            x = line.x0
            for j, w in enumerate(line.words):
                g = line.first + j
                pieces = ([(" ", 1.0)] if j else []) + _runs(w.text, s)
                hidden = mode == "reveal" and active is not None and g > active
                fill = (colours["highlight"] if mode == "karaoke" and g == active
                        else colours["accent"] if w.emphasis else colours["primary"])
                for text, k in pieces:
                    size = max(1, int(round(em * k)))
                    font = _pil_font(st.font_path, size)
                    adv = float(font.getlength(text)) if font is not None else len(text) * st.char_w * size
                    if text.strip() and not hidden and font is not None:
                        yield x, line.baseline - top, text, font, fill
                    x += adv + st.spacing * cl.scale * len(text)

    def layer():
        return Image.new("RGBA", (W, bottom - top), (0, 0, 0, 0))

    region = layer()
    box = s.get("box") or {}
    if box.get("enabled"):
        d = ImageDraw.Draw(region)
        for line in cl.lines:
            x0, y0, w, h, r = _box_geometry(line, cl, st, box)
            d.rounded_rectangle((x0, y0 - top, x0 + w, y0 - top + h), radius=r,
                                fill=_rgba(box.get("color", "#000000"), box.get("opacity", 0.6)))
    blur = st.soften * cl.scale
    stroke = int(round(st.outline * cl.scale))
    shadow, outline = s.get("shadow") or {}, s.get("outline") or {}
    if st.shadow > 0 and float(shadow.get("opacity", 0)) > 0:
        lay = layer()
        d = ImageDraw.Draw(lay)
        off = st.shadow * cl.scale
        col = _rgba(shadow.get("color", "#000000"), shadow.get("opacity", 0.5))
        for x, y, text, font, _ in runs():
            d.text((x + off, y + off), text, font=font, fill=col, anchor="ls", stroke_width=stroke, stroke_fill=col)
        region.alpha_composite(lay.filter(ImageFilter.GaussianBlur(blur)) if blur > 0 else lay)
    if stroke > 0 and float(outline.get("opacity", 1.0)) > 0:
        lay = layer()
        d = ImageDraw.Draw(lay)
        col = _rgba(outline.get("color", "#000000"), outline.get("opacity", 1.0))
        for x, y, text, font, _ in runs():
            d.text((x, y), text, font=font, fill=col, anchor="ls", stroke_width=stroke, stroke_fill=col)
        region.alpha_composite(lay.filter(ImageFilter.GaussianBlur(blur)) if blur > 0 else lay)
    d = ImageDraw.Draw(region)
    for x, y, text, font, fill in runs():
        d.text((x, y), text, font=font, fill=fill, anchor="ls")
    img.alpha_composite(region, (0, top))
    return img


def render_overlays(plan: CaptionPlan, workdir: Path) -> Path:
    """The plan drawn with Pillow: one transparent PNG per chunk rendering (and a blank one for the gaps), and an
    ffconcat listing that shows each for its time. For an ffmpeg without libass (media.burn_captions overlays it);
    the layout, fonts, colours, box, outline and shadow match the ASS; fades and pops are not drawn."""
    from PIL import Image

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    st = _setup(plan.style, plan.height, plan.fonts_dir)
    blank = workdir / "blank.png"
    Image.new("RGBA", (plan.width, plan.height), (0, 0, 0, 0)).save(blank)
    entries: list[tuple[str, float]] = []
    t, n = 0.0, 0
    mode = plan.style.get("mode", "phrase")
    for cl in layout(plan):
        for a, b, k in _states(cl.chunk, mode):
            a = max(a, t)
            if b - a < 0.01:
                continue
            if a - t >= 0.01:
                entries.append((blank.name, a - t))
            png = workdir / f"cap_{n:05d}.png"
            n += 1
            _draw_state(plan, cl, k, st).save(png)
            entries.append((png.name, b - a))
            t = b
    entries.append((blank.name, 0.1))
    text = ["ffconcat version 1.0"]
    for f, d in entries:
        text += [f"file '{f}'", f"duration {d:.3f}"]
    text.append(f"file '{entries[-1][0]}'")   # the concat demuxer drops the last duration without a repeat
    listing = workdir / "overlay.ffconcat"
    listing.write_text("\n".join(text) + "\n", encoding="utf-8")
    return listing


def burn(video: Path, plan: CaptionPlan, out: Path, *, workdir: Path, policy=None, use_ass: bool | None = None,
         timeout: float = 900) -> Path:
    """Burn the plan into `video` -> `out` (audio copied). libass when this ffmpeg has it, else the Pillow
    overlay. Refuses when the style's font is missing: a caption in a fallback system font is worse than none."""
    from . import media

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if plan.fonts_dir is None or not (plan.fonts_dir / str(plan.style.get("file", ""))).exists():
        raise FileNotFoundError(f"caption font {plan.style.get('file')!r} not found (fonts dir {plan.fonts_dir})")
    ass = write_ass(plan, workdir / "captions.ass")
    enc = _cfg(policy).get("encode") or plan.cfg.get("encode") or {}
    return media.burn_captions(Path(video), ass, Path(out), plan.fonts_dir,
                               overlay=lambda: render_overlays(plan, workdir / "overlay"), use_ass=use_ass,
                               crf=int(enc.get("crf", 18)), preset=str(enc.get("preset", "veryfast")),
                               timeout=timeout)
