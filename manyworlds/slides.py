"""Carousel text slides in a creator's brand: covers, numbered tips, steps, quotes, comparisons, recaps, asks.

The renderer the carousel templates of the aesthetics library describe (policy/aesthetics). A creator's brand is
`profile.world.carousel` (the style director's: headline and body font, background, text and accent colours, a
layout); every slide of every carousel of hers uses it, so her carousels look like hers over months.

    render(slide, brand, photo=None) -> PIL image (1080x1350; text inside the centre 1080x1080 the grid shows)
    render_post(conn, policy, store, post, body, chosen) -> the post's slides as image assets, in order

A slide: {template, heading, text, items (recap, comparison: two labels), photo (index of one of the post's
photos, or -1)}. Text is drawn, never generated inside an image (generated text reads as AI). CPU only: the
producer renders slides itself, with Pillow and the OFL fonts in manyworlds/fonts.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .policy import Policy

log = logging.getLogger("manyworlds.slides")
FONTS = Path(__file__).resolve().parent / "fonts"
FONT_FILES = {"DMSerifDisplay": "DMSerifDisplay-Regular.ttf", "Poppins": "Poppins-SemiBold.ttf",
              "Anton": "Anton-Regular.ttf", "TikTokSans": "TikTokSans-SemiBold.ttf"}
BOLD = {"Poppins": "Poppins-Bold.ttf", "TikTokSans": "TikTokSans-Bold.ttf"}
TEMPLATES = ("cover", "list_item", "step", "quote", "comparison", "recap", "cta")
DEFAULT_BRAND = {"headline_font": "DMSerifDisplay", "body_font": "Poppins", "background": "#f4efe8",
                 "text": "#2b2622", "accent": "#b5654a", "layout": "editorial"}


def _cfg(policy: Policy | None) -> dict:
    return (policy.get("slides", default={}) if policy else {}) or {}


def _font(name: str, size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    file = (BOLD.get(name) if bold else None) or FONT_FILES.get(name) or FONT_FILES["Poppins"]
    return ImageFont.truetype(str(FONTS / file), size)


def _rgb(hex_: str, fallback: str = "#000000") -> tuple[int, int, int]:
    h = (hex_ or fallback).lstrip("#")
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return _rgb(fallback)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    lines, line = [], ""
    for word in (text or "").split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + ([line] if line else [])


def _fit(draw, text: str, name: str, start: int, smallest: int, width: int, max_lines: int, *, bold=False):
    """The largest font size (start down to smallest) at which the text wraps into max_lines."""
    size = start
    while size > smallest:
        font = _font(name, size, bold=bold)
        lines = _wrap(draw, text, font, width)
        if len(lines) <= max_lines:
            return font, lines
        size -= 4
    font = _font(name, smallest, bold=bold)
    return font, _wrap(draw, text, font, width)[:max_lines]


def _block(draw, lines: list[str], font, x: int, y: int, fill, *, spacing: float = 1.2, centre: int | None = None) -> int:
    for ln in lines:
        lx = x if centre is None else centre - int(draw.textlength(ln, font=font)) // 2
        draw.text((lx, y), ln, font=font, fill=fill)
        y += int(font.size * spacing)
    return y


def _cover_photo(photo: Image.Image, w: int, h: int) -> Image.Image:
    """The photo filling w x h, cropped from its centre."""
    p = photo.convert("RGB")
    scale = max(w / p.width, h / p.height)
    p = p.resize((max(w, int(p.width * scale)), max(h, int(p.height * scale))), Image.LANCZOS)
    left, top = (p.width - w) // 2, (p.height - h) // 2
    return p.crop((left, top, left + w, top + h))


def render(slide: dict, brand: dict | None = None, photo: Image.Image | None = None, *, policy: Policy | None = None,
           number: int | None = None, of: int | None = None, handle: str = "") -> Image.Image:
    cfg = _cfg(policy)
    W, H = int(cfg.get("width", 1080)), int(cfg.get("height", 1350))
    margin = int(cfg.get("margin", 96))
    b = {**DEFAULT_BRAND, **(brand or {})}
    bg, ink, accent = _rgb(b["background"]), _rgb(b["text"]), _rgb(b["accent"])
    head_name, body_name = b["headline_font"], b["body_font"]
    template = slide.get("template") if slide.get("template") in TEMPLATES else "list_item"
    heading, text = str(slide.get("heading") or "").strip(), str(slide.get("text") or "").strip()
    items = [str(i) for i in (slide.get("items") or []) if str(i).strip()]
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    inner = W - 2 * margin
    safe_top, safe_bottom = (H - W) // 2 + margin // 2, (H + W) // 2 - margin // 2   # the grid's centre square

    if template == "cover":
        if photo is not None:
            img.paste(_cover_photo(photo, W, H))
            scrim = Image.new("RGBA", (W, H // 2), (0, 0, 0, 0))
            sd = ImageDraw.Draw(scrim)
            for y in range(H // 2):   # a soft scrim, dark at the bottom, so the headline reads on any photo
                sd.line([(0, y), (W, y)], fill=(0, 0, 0, int(150 * (y / (H // 2)) ** 1.5)))
            img.paste(scrim, (0, H - H // 2), scrim)
            draw = ImageDraw.Draw(img)
            colour = (255, 255, 255)
        else:
            colour = ink
        font, lines = _fit(draw, heading, head_name, int(cfg.get("cover_size", 104)), 64, inner, 4)
        height = int(font.size * 1.15) * len(lines)
        y = safe_bottom - height - (60 if text else 0)
        _block(draw, lines, font, margin, y, colour, spacing=1.15)
        if text:
            sub = _font(body_name, 36)
            _block(draw, _wrap(draw, text, sub, inner)[:2], sub, margin, safe_bottom - 30, colour)
        cue = _font(body_name, 30)
        draw.text((W - margin - draw.textlength("swipe →", font=cue), H - margin), "swipe →", font=cue, fill=colour)
        return img

    top = safe_top
    if photo is not None and template in ("list_item", "step"):
        ph = int(W * 0.42)
        img.paste(_cover_photo(photo, inner, ph), (margin, top))
        top += ph + 48
    if template in ("list_item", "step"):
        label = (f"{number}" if template == "list_item" else f"step {number} of {of}") if number else ""
        if label:
            nf = _font(head_name, 120 if template == "list_item" else 40)
            draw.text((margin, top), label, font=nf, fill=accent)
            top += int(nf.size * 1.1) + 12
        hf, hl = _fit(draw, heading, head_name, 64, 44, inner, 3)
        top = _block(draw, hl, hf, margin, top, ink, spacing=1.15) + 16
        bf = _font(body_name, 38)
        _block(draw, _wrap(draw, text, bf, inner)[:5], bf, margin, top, ink, spacing=1.35)
    elif template == "quote":
        qf, ql = _fit(draw, f"“{heading or text}”", head_name, 84, 52, inner, 6)
        height = int(qf.size * 1.2) * len(ql)
        _block(draw, ql, qf, margin, (H - height) // 2, ink, spacing=1.2)
        if heading and text:
            bf = _font(body_name, 34)
            draw.text((margin, (H + height) // 2 + 30), f"— {text}", font=bf, fill=accent)
    elif template == "comparison":
        hf, hl = _fit(draw, heading, head_name, 64, 44, inner, 2)
        top = _block(draw, hl, hf, margin, top, ink) + 30
        half = (inner - 40) // 2
        labels = (items + ["", ""])[:2]
        parts = (text.split("|") + [""])[:2] if "|" in text else [text, ""]
        bf, lf = _font(body_name, 34), _font(body_name, 40, bold=True)
        for i, (lab, part) in enumerate(zip(labels, parts)):
            x = margin + i * (half + 40)
            draw.rounded_rectangle((x, top, x + half, safe_bottom), radius=28, outline=accent, width=4)
            draw.text((x + 28, top + 28), lab, font=lf, fill=accent)
            _block(draw, _wrap(draw, part.strip(), bf, half - 56)[:8], bf, x + 28, top + 100, ink, spacing=1.3)
    elif template == "recap":
        hf, hl = _fit(draw, heading or "the recap", head_name, 64, 44, inner, 2)
        top = _block(draw, hl, hf, margin, top, ink) + 24
        bf = _font(body_name, 38)
        for it in items[:8]:
            draw.rounded_rectangle((margin, top + 8, margin + 30, top + 38), radius=6, outline=accent, width=4)
            top = _block(draw, _wrap(draw, it, bf, inner - 60)[:2], bf, margin + 56, top, ink, spacing=1.25) + 14
    else:   # cta
        hf, hl = _fit(draw, heading or "save this for later", head_name, 84, 52, inner, 4)
        height = int(hf.size * 1.2) * len(hl)
        y = _block(draw, hl, hf, 0, (H - height) // 2 - 40, ink, centre=W // 2) + 20
        if text:
            bf = _font(body_name, 38)
            _block(draw, _wrap(draw, text, bf, inner)[:3], bf, 0, y, accent, centre=W // 2)
    if handle:
        hf = _font(body_name, 26)
        draw.text((margin, H - margin + 20), f"@{handle}", font=hf, fill=accent)
    return img


def brand_of(prof: dict | None) -> dict:
    return {**DEFAULT_BRAND, **(((prof or {}).get("world") or {}).get("carousel") or {})}


def render_post(conn, policy: Policy, store, post: dict, body: dict, chosen: list[dict]) -> list[dict]:
    """The post's slides drawn in her brand, uploaded and recorded as image assets, in slide order: a slide with a
    photo index uses that chosen photo; photos no slide uses follow the slides they sit between, so a carousel can
    mix her photos and text slides. Idempotent: the same slides and photos give the same asset uris."""
    from .creator import profile as profile_mod
    slides = [s for s in body.get("slides") or [] if isinstance(s, dict)][:int(_cfg(policy).get("max_slides", 8))]
    if not slides:
        return chosen
    cur = profile_mod.current(conn, post["creator_id"])
    brand = brand_of(cur[1] if cur else None)
    handle = (conn.execute("SELECT handle FROM creators WHERE id = %s", (post["creator_id"],)).fetchone() or {}).get("handle", "")
    numbered = [s for s in slides if s.get("template") in ("list_item", "step")]
    used: set[int] = set()
    drawn: list[tuple[dict, dict]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, s in enumerate(slides):
            k = s.get("photo")
            k = int(k) if isinstance(k, (int, float)) or str(k).lstrip("-").isdigit() else -1
            photo = None
            if 0 <= k < len(chosen):
                used.add(k)
                photo = Image.open(store.get_file(chosen[k]["uri"], Path(tmp) / f"p{k}.png"))
            n = next((j + 1 for j, x in enumerate(numbered) if x is s), None)
            img = render(s, brand, photo, policy=policy, number=n, of=len(numbered), handle=handle)
            key = hashlib.sha256(repr((s, brand, chosen[k]["uri"] if photo is not None else "")).encode()).hexdigest()[:16]
            path = Path(tmp) / f"slide_{i}.png"
            img.save(path, "PNG")
            obj = store.put_file(path, f"slides/{post['creator_id']}/{post['id']}/{i}_{key}.png")
            row = conn.execute("SELECT id, uri FROM assets WHERE uri = %s", (obj.uri,)).fetchone() or conn.execute(
                "INSERT INTO assets (creator_id, kind, uri, sha256, bytes, meta) VALUES (%s, 'image', %s, %s, %s, %s) "
                "RETURNING id, uri", (post["creator_id"], obj.uri, obj.sha256, obj.bytes,
                                      _json({"slide": i, "template": s.get("template")}))).fetchone()
            drawn.append((s, dict(row)))
    # Order: the text slides, her photos no slide used (a dump between the cover and the end), then the closing
    # recap and ask.
    closing = [r for s, r in drawn if s.get("template") in ("recap", "cta")]
    opening = [r for s, r in drawn if s.get("template") not in ("recap", "cta")]
    return opening + [chosen[j] for j in range(len(chosen)) if j not in used] + closing


def _json(v: Any):
    from psycopg.types.json import Jsonb
    return Jsonb(v)
