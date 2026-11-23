"""On-screen text that is not the spoken words: labels, the hook, day-in-life timestamps and question bubbles.

Top clips (docs/research/2026-11-23-readiness-and-craft.md) put a short LABEL on outfit, fashion and day-in-life
videos, not karaoke: one line in the upper third or centre, TikTok-native (white sans with a soft shadow, or black
on a white rounded box), numbered looks "1." "2.", tips "Tight on baggy", timestamps "7:30 AM Hygiene", and on GRWM
a question in a white bubble at the top ("What's your skin type?", like a reply to a comment; we never name a real
user, so there is no "Reply to @..." line). The hook is big text over the first second or two, then gone.

    items (shot sheet edit.on_screen_text) -> overlay_events (wrap, style, safe zone, off the face) -> make_script

* An item is {kind, text, start_s, end_s, style, position} (+ `number` for a numbered look, `time` for a timestamp
  whose time is not at the start of its text). Kinds, styles and positions are policy (`on_screen_text`): each
  kind has a default style and position; a style is laid over `style_defaults` with the same keys as a caption
  style (font, size as a share of the frame height, colours, outline, shadow, soften, box, case, spacing, fades,
  pop), plus `box.mode` (one box per line, or one `block` round all lines) and `box.tail` (the bubble's tail).
* Everything is in shares of the frame, so 1080x1920 and 720x1280 look the same. Text stays inside
  `on_screen_text.safe_zone` (top bar ~250 px, bottom rows ~320 px and the right-side buttons at 1080x1920); a line
  too wide is scaled down, never let out. Lines wrap at the style's `max_chars_per_line`, balanced, and never
  leave one word alone on the last line when there are three or more.
* An item that would cover the face (a face box from the keyframes) or another item showing at the same time
  moves to the first of its position's `alternatives` that is clear.
* Characters the font cannot draw (emoji) are dropped (`missing_glyphs: strip`).
* Rendering: Dialogue lines in the captions' ASS script (`make_script`: one script, one burn pass; captions on
  layers 0-1, overlays on 2-3), and a Pillow drawing of each item for an ffmpeg without libass (`burn`).

Everything here is pure except `burn`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

from . import captions as cap

KINDS = ("label", "hook", "timestamp", "question")

# A time at the start of a timestamp's text: "7:30 AM Hygiene", "7am - coffee", "19:45 dinner".
_TIME = re.compile(r"^\s*(\d{1,2}:\d{2}(?:\s*[ap]\.?m\.?)?|\d{1,2}\s*[ap]\.?m\.?)(?=[\s,.:\-–—|]|$)[\s,.:\-–—|]*(.*)$",
                   re.I)
_WORDS_EXHAUSTIVE = 14     # longer texts are wrapped greedily (labels are a few words; a search is cheap below)

# Fallbacks for a policy without the block's numbers; the shipped values live in policy/default.yaml.
_DEFAULT_SAFE = {"top": 0.13, "bottom": 0.17, "left": 0.08, "right": 0.12}
_DEFAULT_POSITIONS = {"upper_third": {"anchor": "middle", "y": 0.3, "align": "center"}}


# --- policy ---------------------------------------------------------------------------------------------------

def _cfg(policy) -> dict:
    """The `on_screen_text` block from a Policy, a policy body, or the block itself."""
    if policy is None:
        return {}
    if hasattr(policy, "body"):
        return policy.get("on_screen_text", default={}) or {}
    if isinstance(policy, dict):
        return policy.get("on_screen_text", policy) or {}
    return {}


def styles(policy) -> list[str]:
    return list((_cfg(policy).get("styles") or {}).keys())


def resolve_style(policy, name: str | None = None, *, kind: str | None = None) -> tuple[str, dict]:
    """(name, style) for an on-screen text style; an unknown or missing name gets the kind's default style, then
    `default_style`. Each style is laid over `style_defaults`."""
    cfg = _cfg(policy)
    table = cfg.get("styles") or {}
    if not table:
        raise ValueError("no on-screen text styles in policy (on_screen_text.styles)")
    kind_style = ((cfg.get("kinds") or {}).get(kind) or {}).get("style")
    chosen = next((s for s in (name, kind_style, cfg.get("default_style")) if s in table), next(iter(table)))
    return chosen, cap._merge(cfg.get("style_defaults") or {}, table[chosen])


def safe_area(width: int, height: int, policy=None) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) px where on-screen text may go: the frame minus `on_screen_text.safe_zone`."""
    z = {**_DEFAULT_SAFE, **(_cfg(policy).get("safe_zone") or {})}
    return (width * float(z["left"]), height * float(z["top"]),
            width * (1 - float(z["right"])), height * (1 - float(z["bottom"])))


def _position(cfg: dict, name: str | None, kind: str) -> str:
    places = cfg.get("positions") or _DEFAULT_POSITIONS
    name = str(name or "").strip().lower().replace(" ", "_")
    name = (cfg.get("position_aliases") or {}).get(name, name)
    if name in places:
        return name
    default = ((cfg.get("kinds") or {}).get(kind) or {}).get("position")
    return default if default in places else next(iter(places))


# --- items ----------------------------------------------------------------------------------------------------

def items_for_shot(edit: dict | None, start_s: float, end_s: float | None = None) -> list[dict]:
    """A shot's `edit.on_screen_text` as items timed in the whole video: each item's start_s / end_s (relative to
    the shot) shifted by the shot's start, an open end closed at the shot's end. A plain string (the older sheet
    field) is one label for the whole shot."""
    raw = (edit or {}).get("on_screen_text")
    if isinstance(raw, str):
        raw = [{"kind": "label", "text": raw}] if raw.strip() else []
    out = []
    for it in raw or []:
        it = dict(it.model_dump() if hasattr(it, "model_dump") else it)
        a = float(it.get("start_s") or 0.0) + float(start_s)
        b = it.get("end_s")
        b = float(b) + float(start_s) if b is not None else end_s
        if end_s is not None and b is not None:
            b = min(float(b), float(end_s))
        out.append({**it, "start_s": a, "end_s": b})
    return out


def split_time(text: str) -> tuple[str | None, str]:
    """('7:30 AM', 'Hygiene') from '7:30 AM Hygiene'; (None, text) when the text starts with no time."""
    m = _TIME.match(text or "")
    return (re.sub(r"\s+", " ", m.group(1)).strip(), m.group(2).strip()) if m else (None, (text or "").strip())


def wrap(text: str, max_chars: int, max_lines: int, orphan_cost: float = 50.0) -> list[str]:
    """`text` as at most `max_lines` balanced lines of at most `max_chars` (a single longer word stands alone; the
    layout scales it down). Fewest lines first; among those the most even; a last line of one word when there are
    three or more costs `orphan_cost` (it reads as an afterthought). When nothing fits, `max_lines` balanced lines
    anyway (scaled down by the layout)."""
    words = (text or "").split()
    if not words:
        return []
    if len(" ".join(words)) <= max_chars or len(words) == 1:
        return [" ".join(words)]
    n = len(words)
    if n > _WORDS_EXHAUSTIVE:
        return _greedy(words, max_chars)
    fallback = None
    for k in range(2, max(2, max_lines) + 1):
        if k > n:
            break
        best = None
        for cuts in combinations(range(1, n), k - 1):
            bounds = (0, *cuts, n)
            lines = [" ".join(words[a:b]) for a, b in zip(bounds, bounds[1:])]
            lens = [len(s) for s in lines]
            cost = max(lens) - min(lens) + (orphan_cost if n >= 3 and " " not in lines[-1] else 0.0)
            fits = all(len(s) <= max_chars or " " not in s for s in lines)
            if fits and (best is None or cost < best[0]):
                best = (cost, lines)
            if k == max_lines and (fallback is None or (max(lens), cost) < fallback[0]):
                fallback = ((max(lens), cost), lines)
        if best is not None:
            return best[1]
    return fallback[1] if fallback else [" ".join(words)]


def _greedy(words: list[str], max_chars: int) -> list[str]:
    lines: list[str] = []
    for w in words:
        if lines and len(lines[-1]) + 1 + len(w) <= max_chars:
            lines[-1] += " " + w
        else:
            lines.append(w)
    if len(lines) > 1 and " " not in lines[-1] and " " in lines[-2]:   # no lone last word
        head, _, moved = lines[-2].rpartition(" ")
        lines[-2:] = [head, f"{moved} {lines[-1]}"]
    return lines


# --- layout ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TextLine:
    text: str
    factor: float        # size against the style's em (a timestamp's caption is smaller than its time)
    x0: float            # px, the text's extent
    x1: float
    baseline: float
    asc: float           # px above / below the baseline, scaled
    desc: float


@dataclass(frozen=True)
class Overlay:
    kind: str
    text: str
    start: float
    end: float
    style_name: str
    style: dict
    position: str
    align: str           # center | left
    lines: tuple[TextLine, ...]
    scale: float         # 1, or less when a line would be wider than the safe width
    boxes: tuple[tuple[float, float, float, float, float], ...]   # (x0, y0, w, h, radius) px per box
    tail: tuple[float, float, float] | None                       # (x, width, height) px under the block box
    extent: tuple[float, float, float, float]                     # all that is drawn, (x0, y0, x1, y1) px
    width: int
    height: int
    fonts_dir: Path | None


def _pieces(item: dict, kind: str, style: dict, cfg: dict, font: str | None) -> list[tuple[str, float]]:
    """The item's text as (line, size factor), wrapped, cased, and with undrawable characters dropped."""
    mode = str(cfg.get("missing_glyphs", "strip"))
    text = cap.drawable(str(item.get("text") or "").strip(), font, mode)
    if item.get("number") not in (None, ""):
        text = f"{item['number']}. {text}".strip()
    parts: list[tuple[str, float]] = [(text, 1.0)]
    if kind == "timestamp":
        time, rest = (str(item["time"]).strip(), text) if item.get("time") else split_time(text)
        if time:
            scale = float(((cfg.get("kinds") or {}).get("timestamp") or {}).get("caption_scale", 0.8))
            parts = [(cap.drawable(time, font, mode), 1.0)] + ([(rest, scale)] if rest else [])
    max_chars = int(style.get("max_chars_per_line", 24))
    max_lines = int(style.get("max_lines", 3))
    orphan = float((cfg.get("wrap") or {}).get("orphan_cost", 50.0))
    case = str(style.get("case", "none"))
    out = []
    for piece, factor in parts:
        out += [(cap._shown(line, case), factor) for line in wrap(piece, int(max_chars / factor), max_lines, orphan)]
    return [(t, k) for t, k in out if t.strip()]


def _text_width(text: str, factor: float, st) -> float:
    return cap._advance(text, st, st.em * factor) + st.spacing * factor * max(len(text) - 1, 0)


def _layout(pieces: list[tuple[str, float]], style: dict, place: dict, width: int, height: int,
            safe: tuple[float, float, float, float], fonts: Path | None) -> dict:
    """Lines, boxes and extent of one item at one position, in px."""
    st = cap._setup(style, height, fonts)
    m = st.metrics
    box = style.get("box") or {}
    boxed = bool(box.get("enabled"))
    block = boxed and box.get("mode") == "block"
    tail_cfg = box.get("tail") or {}
    sx0, sy0, sx1, sy1 = safe
    align = "left" if place.get("align") == "left" else "center"
    cx = width / 2
    pad0 = st.em * float(box.get("pad_x", 0.0)) if boxed else 0.0
    room = (sx1 - sx0) if align == "left" else 2 * min(cx - sx0, sx1 - cx)
    avail = room - 2 * (pad0 + st.outline)
    widths = [_text_width(t, k, st) for t, k in pieces]
    widest = max(widths) if widths else 0.0
    scale = min(1.0, avail / widest) if widest > 0 else 1.0
    # Lines from a top of 0; moved into place below.
    rows, base = [], 0.0
    for i, ((text, k), w) in enumerate(zip(pieces, widths)):
        asc, desc = st.asc * k * scale, st.desc * k * scale
        base = asc if i == 0 else base + st.line_h * k * scale
        x0 = sx0 + pad0 * scale + st.outline if align == "left" else cx - w * scale / 2
        rows.append([text, k, x0, x0 + w * scale, base, asc, desc])
    boxes, tail = [], None
    if boxed:
        pad, radius = float(box.get("pad_x", 0.35)), float(box.get("radius", 0.3))
        if block:
            pad_y = st.em * scale * float(box.get("pad_y", 0.3))
            x0 = min(r[2] for r in rows) - pad * st.em * scale
            x1 = max(r[3] for r in rows) + pad * st.em * scale
            y0, y1 = -pad_y, rows[-1][4] + rows[-1][6] + pad_y
            boxes.append((x0, y0, x1 - x0, y1 - y0, min(st.em * scale * radius, (y1 - y0) / 2)))
            if tail_cfg.get("enabled"):
                tw, th = st.em * scale * float(tail_cfg.get("width", 0.5)), st.em * scale * float(tail_cfg.get("height", 0.4))
                tx = x0 + (x1 - x0) * float(tail_cfg.get("x", 0.15))
                tail = (tx, tw, th)
        else:
            for text, k, x0, x1, b, _, _ in rows:
                em = st.em * k * scale
                h = em * float(box.get("height", 1.3))
                mid = b - em * (m.x_height + m.cap_height) / (4 * m.upem)
                boxes.append((x0 - pad * em, mid - h / 2, x1 - x0 + 2 * pad * em, h, min(em * radius, h / 2)))
    o = st.outline * scale
    ex0 = min([r[2] - o for r in rows] + [b[0] for b in boxes])
    ex1 = max([r[3] + o for r in rows] + [b[0] + b[2] for b in boxes])
    ey0 = min([r[4] - r[5] - o for r in rows] + [b[1] for b in boxes])
    ey1 = max([r[4] + r[6] + o for r in rows] + [b[1] + b[3] + (tail[2] if tail else 0.0) for b in boxes])
    # Into place: the anchor holds the whole drawn extent, kept inside the safe area.
    ext_h = ey1 - ey0
    ref = float(place.get("y", 0.3)) * height
    anchor = place.get("anchor", "middle")
    top = ref if anchor == "top" else ref - ext_h / 2 if anchor == "middle" else ref - ext_h
    top = max(sy0, min(top, sy1 - ext_h))
    dy = top - ey0
    lines = tuple(TextLine(t, k, x0, x1, b + dy, a, d) for t, k, x0, x1, b, a, d in rows)
    return {"lines": lines, "scale": scale, "align": align,
            "boxes": tuple((x, y + dy, w, h, r) for x, y, w, h, r in boxes), "tail": tail,
            "extent": (ex0, ey0 + dy, ex1, ey1 + dy)}


def _clash(extent, other: Overlay, start: float, end: float) -> bool:
    if start >= other.end or end <= other.start:
        return False
    x0, y0, x1, y1 = extent
    ox0, oy0, ox1, oy1 = other.extent
    return x0 < ox1 and ox0 < x1 and y0 < oy1 and oy0 < y1


def _on_face(extent, face_box, width: int, height: int, margin: float) -> bool:
    if not face_box:
        return False
    fx0, _, fx1, _ = (float(v) for v in face_box)
    x0, y0, x1, y1 = extent
    across = x0 / width < fx1 + margin and fx0 - margin < x1 / width
    return across and cap.covers_face(y0 / height, y1 / height, face_box, margin)


def overlay_events(items: Iterable[dict], plan_or_size, policy=None, *, duration: float | None = None,
                   face_box: Sequence[float] | None = None) -> list[Overlay]:
    """The items laid out for a video: a captions plan (its frame size) or (width, height). Each item gets its
    kind's style and position unless it names known ones; times default per kind (a hook from `start_s` for
    `duration_s`, at most `max_duration_s`; others to the end of the video, else `default_duration_s`) and are
    clipped to `duration`. An item covering `face_box` (x0, y0, x1, y1 shares) or an earlier item showing at the
    same time moves to the first clear position in its position's `alternatives`. Empty when
    `on_screen_text.enabled` is off. Items with no drawable text, or no time on screen, are left out."""
    cfg = _cfg(policy)
    if not cfg.get("enabled", True):
        return []
    if isinstance(plan_or_size, cap.CaptionPlan):
        width, height = plan_or_size.width, plan_or_size.height
    else:
        width, height = (int(v) for v in plan_or_size)
    fonts = cap.fonts_dir(policy)
    safe = safe_area(width, height, policy)
    places = cfg.get("positions") or _DEFAULT_POSITIONS
    margin = float(cfg.get("face_margin", 0.0))
    default_s = float(cfg.get("default_duration_s", 3.0))
    out: list[Overlay] = []
    for raw in items or []:
        item = dict(raw.model_dump() if hasattr(raw, "model_dump") else raw)
        kind = str(item.get("kind") or "label").lower()
        kind = kind if kind in KINDS else "label"
        kcfg = (cfg.get("kinds") or {}).get(kind) or {}
        want = item.get("style") or (kcfg.get("numbered_style") if item.get("number") not in (None, "") else None)
        style_name, style = resolve_style(policy, want, kind=kind)
        font = fonts / style["file"] if fonts is not None and style.get("file") else None
        pieces = _pieces(item, kind, style, cfg, str(font) if font is not None and font.exists() else None)
        if not pieces:
            continue
        start = max(0.0, float(item.get("start_s") if item.get("start_s") is not None else kcfg.get("start_s", 0.0)))
        end = item.get("end_s")
        if end is None:
            end = (start + float(kcfg["duration_s"]) if kcfg.get("duration_s") is not None
                   else duration if duration is not None else start + default_s)
        end = float(end)
        if kcfg.get("max_duration_s") is not None:
            end = min(end, start + float(kcfg["max_duration_s"]))
        if duration is not None:
            end = min(end, float(duration))
        if end - start < 0.05:
            continue
        first = _position(cfg, item.get("position"), kind)
        tried = [first] + [p for p in (cfg.get("alternatives") or {}).get(first) or [] if p in places and p != first]
        laid, where = None, first
        for name in tried:
            cand = _layout(pieces, style, places[name], width, height, safe, fonts)
            if laid is None:
                laid, where = cand, name
            if not _on_face(cand["extent"], face_box, width, height, margin) and \
                    not any(_clash(cand["extent"], o, start, end) for o in out):
                laid, where = cand, name
                break
        out.append(Overlay(kind=kind, text=" / ".join(t for t, _ in pieces), start=round(start, 3),
                           end=round(end, 3), style_name=style_name, style=style, position=where,
                           align=laid["align"], lines=laid["lines"], scale=laid["scale"], boxes=laid["boxes"],
                           tail=laid["tail"], extent=laid["extent"], width=width, height=height, fonts_dir=fonts))
    return out


# --- ASS ------------------------------------------------------------------------------------------------------

def _path(points: list[tuple]) -> str:
    return " ".join(p[0] + " " + " ".join(str(int(round(v))) for v in p[1:]) for p in points)


def bubble_path(w: float, h: float, r: float, tail: tuple[float, float, float] | None = None) -> str:
    """A rounded rectangle w x h (corner radius r) as an ASS drawing, with an optional speech-bubble tail
    (x from the box's left, width, height) hanging from its bottom edge, drawn as one outline (no seam)."""
    k = 0.5523 * r
    bottom = [("l", r, h)]
    if tail:
        tx, tw, th = tail
        tx = max(r, min(tx, w - r - tw))
        bottom = [("l", tx + tw, h), ("l", tx + tw * 0.15, h + th), ("l", tx, h), ("l", r, h)]
    pts = [("m", r, 0), ("l", w - r, 0), ("b", w - r + k, 0, w, r - k, w, r), ("l", w, h - r),
           ("b", w, h - r + k, w - r + k, h, w - r, h), *bottom, ("b", r - k, h, 0, h - r + k, 0, h - r),
           ("l", 0, r), ("b", 0, r - k, r - k, 0, r, 0)]
    return _path(pts)


def style_name(ov: Overlay) -> str:
    return "ost_" + cap.ass_name(ov.style_name, "label")


def _fades(ov: Overlay) -> str:
    dur = int(round((ov.end - ov.start) * 1000))
    fi = min(int(ov.style.get("fade_in_ms", 0)), dur // 2)
    fo = min(int(ov.style.get("fade_out_ms", 0)), dur - fi)
    return f"\\fad({fi},{fo})" if fi or fo else ""


def events(ov: Overlay) -> list[tuple[int, int, str]]:
    """The item's Dialogue lines as (start in centiseconds, layer, line): its box drawings on layer 2 and its
    text on layer 3, over the captions (layers 0 and 1)."""
    s, st = ov.style, cap._setup(ov.style, ov.height, ov.fonts_dir)
    name = style_name(ov)
    c0, c1 = int(round(ov.start * 100)), int(round(ov.end * 100))
    a, b = cap._cs_time(c0), cap._cs_time(c1)
    fad = _fades(ov)
    out = []
    box = s.get("box") or {}
    for i, (x, y, w, h, r) in enumerate(ov.boxes):
        shape = bubble_path(w, h, r, (ov.tail[0] - x, ov.tail[1], ov.tail[2]) if ov.tail and i == 0 else None)
        text = (f"{{\\an7\\pos({cap._n(x)},{cap._n(y)})\\bord0\\shad0\\blur0"
                f"\\1c{cap.ass_colour(box.get('color', '#FFFFFF'))}\\1a&H{cap._alpha(box.get('opacity', 1.0)):02X}&"
                f"{fad}\\p1}}{shape}{{\\p0}}")
        out.append((c0, 2, f"Dialogue: 2,{a},{b},{name},,0,0,0,,{text}"))
    base = 100.0 * ov.scale
    pop = cap._pop_tags(s.get("pop") or {}, base)
    for line in ov.lines:
        x = line.x0 if ov.align == "left" else (line.x0 + line.x1) / 2
        tags = f"\\an{1 if ov.align == 'left' else 2}\\pos({cap._n(x)},{cap._n(line.baseline + line.desc)}){fad}"
        if st.soften > 0:
            tags += f"\\blur{cap._n(st.soften * ov.scale)}"
        tags += pop or (f"\\fscx{cap._n(base)}\\fscy{cap._n(base)}" if ov.scale < 1.0 else "")
        if line.factor != 1.0:
            tags += f"\\fs{cap._n(st.ass_size * line.factor)}"
        out.append((c0, 3, f"Dialogue: 3,{a},{b},{name},,0,0,0,,{{{tags}}}{cap._esc(line.text)}"))
    return out


def make_script(plan: cap.CaptionPlan | None, overlays: Sequence[Overlay], policy=None, *,
                size: tuple[int, int] | None = None) -> str:
    """One ASS script with the spoken captions (if any) and the on-screen texts, for one burn pass. The frame is
    the plan's, else the overlays', else `size`."""
    if plan is not None:
        width, height = plan.width, plan.height
    elif overlays:
        width, height = overlays[0].width, overlays[0].height
    elif size:
        width, height = size
    else:
        raise ValueError("make_script needs a captions plan, overlays or a size")
    lines = [cap.caption_style_line(plan)] if plan is not None else []
    seen: set[str] = set()
    for ov in overlays:
        if style_name(ov) not in seen:
            seen.add(style_name(ov))
            lines.append(cap.style_line(style_name(ov), ov.style, cap._setup(ov.style, ov.height, ov.fonts_dir)))
    evs = cap.caption_events(plan) if plan is not None else []
    for ov in overlays:
        evs += events(ov)
    evs.sort(key=lambda e: (e[0], e[1]))
    what = f"captions in style {plan.style_name}" if plan is not None else "no captions"
    head = cap.script_head(width, height, f"manyworlds {what}, {len(overlays)} on-screen texts", lines)
    return "\n".join(head + [e[2] for e in evs]) + "\n"


def write_script(path: Path, plan: cap.CaptionPlan | None, overlays: Sequence[Overlay], policy=None, *,
                 size: tuple[int, int] | None = None) -> Path:
    path = Path(path)
    path.write_text(make_script(plan, overlays, policy, size=size), encoding="utf-8")
    return path


# --- fallback drawing and burning ------------------------------------------------------------------------------

def draw(ov: Overlay):
    """The item as a transparent full-frame RGBA image (Pillow): boxes and tail, soft shadow, outline, text. For
    an ffmpeg without libass; fades and pops are not drawn."""
    from PIL import Image, ImageDraw, ImageFilter

    s, W, H = ov.style, ov.width, ov.height
    st = cap._setup(s, H, ov.fonts_dir)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    box = s.get("box") or {}
    d = ImageDraw.Draw(img)
    fill = cap._rgba(box.get("color", "#FFFFFF"), box.get("opacity", 1.0))
    for i, (x, y, w, h, r) in enumerate(ov.boxes):
        d.rounded_rectangle((x, y, x + w, y + h), radius=r, fill=fill)
        if ov.tail and i == 0:
            tx, tw, th = ov.tail
            tx = max(x + r, min(tx, x + w - r - tw))
            d.polygon([(tx, y + h - 1), (tx + tw, y + h - 1), (tx + tw * 0.15, y + h + th)], fill=fill)

    def runs():
        for line in ov.lines:
            font = cap._pil_font(st.font_path, max(1, int(round(st.em * line.factor * ov.scale))))
            if font is not None:
                yield line.x0, line.baseline, line.text, font

    shadow, outline = s.get("shadow") or {}, s.get("outline") or {}
    blur = st.soften * ov.scale
    stroke = int(round(st.outline * ov.scale))
    if st.shadow > 0 and float(shadow.get("opacity", 0)) > 0:
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lay)
        off = st.shadow * ov.scale
        col = cap._rgba(shadow.get("color", "#000000"), shadow.get("opacity", 0.5))
        for x, y, text, font in runs():
            ld.text((x + off, y + off), text, font=font, fill=col, anchor="ls", stroke_width=stroke, stroke_fill=col)
        img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(blur)) if blur > 0 else lay)
    if stroke > 0 and float(outline.get("opacity", 1.0)) > 0:
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lay)
        col = cap._rgba(outline.get("color", "#000000"), outline.get("opacity", 1.0))
        for x, y, text, font in runs():
            ld.text((x, y), text, font=font, fill=col, anchor="ls", stroke_width=stroke, stroke_fill=col)
        img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(blur)) if blur > 0 else lay)
    d = ImageDraw.Draw(img)
    for x, y, text, font in runs():
        d.text((x, y), text, font=font, fill=cap._rgba(s["color"]), anchor="ls")
    return img


def burn(video: Path, out: Path, *, workdir: Path, plan: cap.CaptionPlan | None = None,
         overlays: Sequence[Overlay] = (), policy=None, use_ass: bool | None = None, timeout: float = 900) -> Path:
    """Burn the captions plan and the on-screen texts into `video` -> `out` in one pass (audio copied): libass
    when this ffmpeg has it, else the Pillow drawings. Refuses when a style's font is missing: text in a fallback
    system font is worse than none."""
    from . import media

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    fonts = plan.fonts_dir if plan is not None else cap.fonts_dir(policy)
    needed = ([plan.style] if plan is not None else []) + [ov.style for ov in overlays]
    for style in needed:
        if fonts is None or not (fonts / str(style.get("file", ""))).exists():
            raise FileNotFoundError(f"on-screen text font {style.get('file')!r} not found (fonts dir {fonts})")
    if plan is not None:
        size = (plan.width, plan.height)
    elif overlays:
        size = (overlays[0].width, overlays[0].height)
    else:
        raise ValueError("nothing to burn: no captions plan and no overlays")
    ass = write_script(workdir / "text.ass", plan, overlays, policy)
    extra = [(ov.start, ov.end, (lambda ov=ov: draw(ov))) for ov in overlays]
    enc = cap._cfg(policy).get("encode") or (plan.cfg.get("encode") if plan is not None else None) or {}
    return media.burn_captions(Path(video), ass, Path(out), fonts,
                               overlay=lambda: cap.render_overlays(plan, workdir / "overlay", extra=extra, size=size),
                               use_ass=use_ass, crf=int(enc.get("crf", 18)), preset=str(enc.get("preset", "veryfast")),
                               timeout=timeout)
