"""Structural control for photos: a pose skeleton drawn from a shot sheet (policy generation.control).

The 2026-11-27 range audit found the words of a shot reach the image for place and garments, but the camera
(distance, height, angle) and composition land only 54-61% of the time. A pose map fixes where her body is, how big
it is in the frame, how it is posed and how the camera's height bends it, which words alone do not. The skeleton is
synthetic: drawn here from a small library of pose templates (policy/layout/poses-v1.yaml), never extracted from
anyone's photo. It is rendered in the OpenPose convention the Z-Image Fun ControlNet Union was trained on.

`pose_map(shot, policy, width, height)` -> a PIL image, or None for a shot with no body to place (close-ups,
details, hands-only, overhead flat lays). `plan(shot, policy)` says what it would draw, for tests and reports.
"""

from __future__ import annotations

import functools
import math
import re
from pathlib import Path
from typing import Any

import yaml

# OpenPose (COCO-18) limbs, keypoint index pairs, and the colours ControlNet pose models were trained on.
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
         (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
COLOURS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
           (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
           (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def _cfg(policy) -> dict:
    return policy.get("generation", "control", default={}) or {}


def _policy_dir() -> Path:
    for candidate in (Path(__file__).resolve().parents[1] / "policy", Path.cwd() / "policy", Path("/app/policy")):
        if candidate.is_dir():
            return candidate
    return Path.cwd() / "policy"


@functools.lru_cache(maxsize=4)
def library(name: str) -> dict:
    data = yaml.safe_load((_policy_dir() / name).read_text(encoding="utf-8"))
    for pose, pts in (data.get("poses") or {}).items():
        if len(pts) != 18:
            raise ValueError(f"pose {pose}: {len(pts)} keypoints, expected 18")
    return data


def _text(shot: dict) -> str:
    beats = shot.get("blocking") or []
    return " ".join(str(b.get("action") or "") + " " + str(b.get("hands") or "") for b in beats).lower()


def _first(rules, text: str):
    for pattern, value in rules or []:
        if re.search(str(pattern), text):
            return value
    return None


def plan(shot: dict, policy) -> dict | None:
    """What the skeleton would be: pose, size rule, camera bend, horizontal centre. None: no body to place."""
    c = _cfg(policy)
    framing = shot.get("framing") or {}
    size = str(framing.get("shot_size") or "").lower()
    angle = str(framing.get("angle") or "").lower()
    if framing.get("device") == "pov" or "overhead" in angle and "flat" in _text(shot):
        return None
    rule = None
    for key, spec in c.get("sizes") or []:
        if str(key) in size:
            rule = spec
            break
    if rule is None:
        return None
    lib = library(str(c.get("poses", "layout/poses-v1.yaml")))
    text = _text(shot)
    pose = _first(lib.get("words"), text) or str(c.get("default_pose", "stand"))
    bend = float(_first([(re.escape(str(k)), v) for k, v in c.get("angles") or []], angle) or 0.0)
    x = float(_first(c.get("placements"), text + " " + size) or 0.5)
    return {"pose": pose, "scale": float(rule["scale"]), "top": float(rule["top"]), "bend": bend, "x": x}


def _bent(y: float, bend: float) -> float:
    """Camera height as perspective on the body's height axis: with bend > 0 (a low camera) the parts near the feet
    take up more of the body's drawn height and the head less; bend < 0 the reverse. The head stays at the top."""
    if abs(bend) < 1e-9:
        return y
    top = 1.0 - bend / 2.0
    return y * (1.0 - bend * y / 2.0) / top


def keypoints(p: dict, policy, width: int, height: int) -> list[tuple[float, float] | None]:
    """The pose's keypoints in pixels for the frame, after size, camera bend and placement."""
    c = _cfg(policy)
    pts = library(str(c.get("poses", "layout/poses-v1.yaml")))["poses"][p["pose"]]
    bend = p["bend"]
    xs = [q[0] for q in pts if q]
    ys = [_bent(q[1], bend) for q in pts if q]
    head_top = max(ys) + 0.06          # a little above the nose and eyes
    body_px = p["scale"] * height
    span = (max(xs) - min(xs)) * body_px
    limit = float(c.get("max_width", 0.95)) * width
    if span > limit:
        body_px *= limit / span
    out: list[tuple[float, float] | None] = []
    for q in pts:
        if not q:
            out.append(None)
            continue
        widen = 1.0 + bend * (0.5 - q[1]) * 0.5       # nearer parts are a little wider too
        x = p["x"] * width + q[0] * widen * body_px
        y = p["top"] * height + (head_top - _bent(q[1], bend)) * body_px
        out.append((x, y))
    return out


def draw(points, width: int, height: int, body_px: float, stick: float):
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    w = max(2.0, stick * body_px)
    for i, (a, b) in enumerate(LIMBS):
        pa, pb = points[a], points[b]
        if pa is None or pb is None:
            continue
        layer = Image.new("RGB", (width, height), (0, 0, 0))
        mask = Image.new("L", (width, height), 0)
        (x1, y1), (x2, y2) = pa, pb
        length = math.hypot(x2 - x1, y2 - y1)
        ang = math.atan2(y2 - y1, x2 - x1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        poly = []
        for t in range(0, 360, 15):    # an ellipse along the limb, as OpenPose draws it
            r = math.radians(t)
            ex, ey = math.cos(r) * length / 2, math.sin(r) * w
            poly.append((cx + ex * math.cos(ang) - ey * math.sin(ang), cy + ex * math.sin(ang) + ey * math.cos(ang)))
        ImageDraw.Draw(layer).polygon(poly, fill=COLOURS[i])
        ImageDraw.Draw(mask).polygon(poly, fill=int(255 * 0.6))
        canvas.paste(layer, (0, 0), mask)
    d = ImageDraw.Draw(canvas)
    for i, p in enumerate(points):
        if p is not None:
            d.ellipse([p[0] - w, p[1] - w, p[0] + w, p[1] + w], fill=COLOURS[i])
    return canvas


def pose_map(shot: dict, policy, width: int, height: int):
    """The control image for a shot, or None when the shot has no body to place."""
    p = plan(shot, policy)
    if p is None:
        return None
    pts = keypoints(p, policy, width, height)
    drawn = [q for q in pts if q]
    body_px = (max(y for _, y in drawn) - min(y for _, y in drawn)) or p["scale"] * height
    return draw(pts, width, height, body_px, float(_cfg(policy).get("stick_width", 0.012)))


def control_payload(policy, uri: str, plan_: dict[str, Any]) -> dict:
    """The image.zimage payload's `control` block for a pose map stored at `uri`."""
    c = _cfg(policy)
    return {"kind": "pose", "uri": uri, "scale": float(c.get("scale", 0.75)), "repo": str(c["repo"]),
            "file": str(c["file"]), "plan": plan_}
