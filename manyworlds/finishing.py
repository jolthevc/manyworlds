"""Finishing: what a finished video gets on the control plane before it is published (no GPU).

  rate       the delivery frame rate (`finishing.fps`, 30): a slower video (Wan 2.2 clips at 16 fps, talk at 25) is
             motion-interpolated (`finishing.interpolate`, media.fps_filter) at its own size, before the upscale
             (cheaper); a faster one drops frames
  scale      a video smaller than `finishing.output` (1080x1920; talk renders at 480x832) is upscaled
             (`output.flags`, lanczos) until it covers it, keeping its aspect, and centre-cropped, so the grade,
             grain and burned-in captions all work at the delivery size; never scaled down. An AI upscaler on a
             GPU worker (`finishing.upscaler`, off) would run before this stage; then nothing is left to scale here
  grade      a gentle natural look per creator or set (policy `finishing.looks`: neutral, daylight, warm_night),
             built from ffmpeg eq, colorbalance and curves; every value is clamped to `finishing.limits`, so no
             look can oversaturate or crush the image
  sharpen    a slight luma unsharp mask, off by default
  vignette   optional, off by default
  grain      subtle temporal film grain (ffmpeg `noise`, strength in policy), so frames don't read as too clean
  hook       a hook-text overlay for the first 1-2 s: an ASS file written by the captions module in one of its
             named styles (policy `finishing.hook.style`); this module only burns it in, or skips it
  loudness   -14 LUFS integrated, -1 dBTP true peak, two-pass loudnorm (measure, then apply linearly); one pass
             when the measurement can't be used (silence)
  cover      the cover frame: the sharpest keyframe with the face found, eyes open and mouth closed
             (Laplacian variance, plus MediaPipe face landmarks where that is installed)

The builders (`scale_filter`, `filters`, `audio_filter`, `loudnorm_filter`, `parse_loudnorm`) are pure strings,
tested in CI. `finish` runs ffmpeg (manyworlds.media.ffmpeg, the imageio-ffmpeg static build): scale, grade, grain
and text in one pass, plus a loudness measurement. Production's finish stage (production.finish_video, for the formats
in `finishing.formats`) calls it once a post's video is chosen: the burned-in captions (manyworlds.captions, planned
at `finished_size`) ride in the same pass as the hook text where this ffmpeg has libass, else a second Pillow pass;
the cover is picked from the pieces' keyframes and kept as the post's `body.cover` (the Reel's cover on Instagram).
The finished file replaces the post's video (`body.video`; the unfinished one stays as `body.video_raw`).
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Callable

from .policy import Policy

FONTS_DIR = Path(__file__).parent / "fonts"   # the captions module's OFL fonts, when present


def cfg(policy: Policy | None) -> dict:
    return (policy.get("finishing", default={}) or {}) if policy is not None else {}


def _clamp(value, lo_hi, default: float) -> float:
    v = float(default if value is None else value)
    if lo_hi:
        lo, hi = float(lo_hi[0]), float(lo_hi[1])
        v = min(hi, max(lo, v))
    return v


def _num(v: float) -> str:
    """A filter number without float noise: 1.0 -> '1', 0.035 -> '0.035'."""
    s = f"{float(v):.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def look_for(policy: Policy, *, creator: str | None = None, set_id: str | None = None) -> str:
    """The look name for a video: the set's (`finishing.look_by_set`), else the creator's (`look_by_creator`,
    keyed by handle), else `default_look`."""
    c = cfg(policy)
    return ((set_id and (c.get("look_by_set") or {}).get(set_id))
            or (creator and (c.get("look_by_creator") or {}).get(creator))
            or c.get("default_look") or "neutral")


def grade(look: str | dict | None, policy: Policy) -> list[str]:
    """The grade's filters for a named look (or a look dict), clamped to `finishing.limits`. An unknown name
    falls back to `default_look`; a look with no settings is no filter at all."""
    c = cfg(policy)
    looks, limits = c.get("looks") or {}, c.get("limits") or {}
    if isinstance(look, dict):
        spec = look
    else:
        spec = looks.get(look or "") or looks.get(c.get("default_look") or "neutral") or {}
    out = []
    eq = {k: _clamp(spec.get(k), limits.get(k), d)
          for k, d in (("contrast", 1.0), ("brightness", 0.0), ("saturation", 1.0), ("gamma", 1.0))}
    if eq != {"contrast": 1.0, "brightness": 0.0, "saturation": 1.0, "gamma": 1.0}:
        out.append("eq=" + ":".join(f"{k}={_num(v)}" for k, v in eq.items()))
    bal_limit = limits.get("balance")
    balance = {}
    for k, v in (spec.get("balance") or {}).items():
        if k not in ("rs", "gs", "bs", "rm", "gm", "bm", "rh", "gh", "bh") or v is None:
            continue
        v = float(v)
        if bal_limit is not None:
            v = min(float(bal_limit), max(-float(bal_limit), v))
        if v:
            balance[k] = v
    if balance:
        out.append("colorbalance=" + ":".join(f"{k}={_num(v)}" for k, v in sorted(balance.items())))
    curves = {k: str(v).strip() for k, v in (spec.get("curves") or {}).items()
              if k in ("master", "red", "green", "blue") and v and re.fullmatch(r"[0-9./ ]+", str(v).strip())}
    if curves:
        out.append("curves=" + ":".join(f"{k}='{v}'" for k, v in curves.items()))
    return out


def _filter_path(path: Path | str) -> str:
    """A file path as a filtergraph option value: single-quoted for the graph, with ':' escaped for the filter's
    own option parser (a Windows drive letter). Paths are ours (workdirs), so a quote in one is refused."""
    p = Path(path).resolve().as_posix()
    if "'" in p:
        raise ValueError(f"a quote in a filter path is not supported: {p}")
    return "'" + p.replace(":", "\\:") + "'"


def output_size(policy: Policy | None) -> tuple[int, int] | None:
    """The delivery frame size (`finishing.output` width x height), or None when none is set."""
    o = cfg(policy).get("output") or {}
    try:
        w, h = int(o.get("width") or 0), int(o.get("height") or 0)
    except (TypeError, ValueError):
        return None
    return (w, h) if w > 0 and h > 0 else None


def scale_filter(width: int | None, height: int | None, policy: Policy) -> str | None:
    """The upscale for a video of this frame size: when it is smaller than `finishing.output` in either dimension,
    scaled up (`finishing.output.flags`, lanczos) until it covers the output size, keeping its aspect, then
    centre-cropped to it. None when no output size is set, the size is unknown, or the video is already that big
    (it is never scaled down)."""
    out = output_size(policy)
    if not out or not width or not height:
        return None
    (ow, oh), (w, h) = out, (int(width), int(height))
    if w >= ow and h >= oh:
        return None
    s = max(ow / w, oh / h)
    sw = max(ow, 2 * math.ceil(round(w * s / 2, 6)))   # even sizes for yuv420p; rounding drops float noise
    sh = max(oh, 2 * math.ceil(round(h * s / 2, 6)))
    flags = str((cfg(policy).get("output") or {}).get("flags") or "lanczos")
    f = f"scale={sw}:{sh}:flags={flags}"
    return f if (sw, sh) == (ow, oh) else f"{f},crop={ow}:{oh}"


def finished_size(width: int, height: int, policy: Policy) -> tuple[int, int]:
    """The frame size the finish stage delivers for a video of this size (what burned-in captions are planned at)."""
    out = output_size(policy)
    return out if out and scale_filter(width, height, policy) else (width, height)


def target_fps(policy: Policy | None) -> float | None:
    """The delivery frame rate (`finishing.fps`), or None when the video keeps its own."""
    try:
        v = float(cfg(policy).get("fps") or 0)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def rate_filter(fps: float | None, policy: Policy, *, seconds: float | None = None) -> str | None:
    """The step to the delivery rate for a video at `fps` (0 = unknown: plain `fps=`), by `finishing.interpolate`
    (media.fps_filter). None when no rate is set or the video is already at it."""
    from .media import fps_filter
    return fps_filter(fps, target_fps(policy), cfg(policy).get("interpolate"), seconds=seconds)


def upscaler(policy: Policy | None) -> dict | None:
    """The AI upscaler to run on a GPU worker before the finish stage (`finishing.upscaler`), or None when it is
    off or names no job kind. Off: it needs a GPU benchmark (time and cost per output second, a gate pass rate
    against lanczos) before production queues it; until then the finish pass's lanczos does the scaling."""
    u = cfg(policy).get("upscaler") or {}
    return dict(u) if u.get("enabled") and u.get("kind") else None


def filters(look: str | dict | None, *, policy: Policy, grain: bool | float | None = None,
            sharpen: bool | float | None = None, vignette: bool | float | None = None,
            hook_ass: Path | str | None = None, size: tuple[int, int] | None = None,
            fps: float | None = None, seconds: float | None = None) -> str:
    """The video filter chain: the frame rate (when `fps`, the input's rate, is given: 0 = unknown), the upscale to
    `finishing.output` (when `size`, the input's frame size, is smaller), grade, sharpen, vignette, grain, then the
    hook text (so text gets no grain), and yuv420p. The rate comes first, so motion interpolation works on the
    small source frames. `grain`, `sharpen` and `vignette`: None = policy, False or 0 = off, True = on at the
    policy strength, a number = that strength."""
    c = cfg(policy)
    rate = rate_filter(fps, policy, seconds=seconds) if fps is not None else None
    scale = scale_filter(*size, policy) if size else None
    chain = ([rate] if rate else []) + ([scale] if scale else []) + grade(look, policy)

    def strength(arg, key: str, field: str) -> float:
        block = c.get(key) or {}
        if arg is None:
            return float(block.get(field, 0)) if block.get("enabled") else 0.0
        if arg is True:
            return float(block.get(field, 0))
        return float(arg or 0)

    amount = strength(sharpen, "sharpen", "amount")
    if amount > 0:
        size = int((c.get("sharpen") or {}).get("size", 5))
        chain.append(f"unsharp=luma_msize_x={size}:luma_msize_y={size}:luma_amount={_num(amount)}")
    angle = strength(vignette, "vignette", "angle")
    if angle > 0:
        chain.append(f"vignette=angle={_num(angle)}")
    g = strength(grain, "grain", "strength")
    g = min(g, float((c.get("grain") or {}).get("max_strength", g)))
    if g > 0:
        chain.append(f"noise=alls={int(round(g))}:allf=t+u")
    if hook_ass:
        opt = f"ass=filename={_filter_path(hook_ass)}"
        fonts = Path(c.get("fonts_dir") or FONTS_DIR)
        if fonts.is_dir():
            opt += f":fontsdir={_filter_path(fonts)}"
        chain.append(opt)
    chain.append("format=yuv420p")
    return ",".join(chain)


def _loud_cfg(policy: Policy) -> dict:
    return cfg(policy).get("loudness") or {}


def loudnorm_filter(policy: Policy, measured: dict | None = None) -> str:
    """loudnorm at the policy target. With a first pass's measurement it runs linear (a constant gain, no pumping)."""
    lc = _loud_cfg(policy)
    base = (f"loudnorm=I={_num(lc.get('lufs', -14))}:TP={_num(lc.get('true_peak', -1.0))}"
            f":LRA={_num(lc.get('lra', 11))}")
    if not measured:
        return base
    return (base + f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
                   f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
                   f":offset={measured['target_offset']}:linear=true")


def audio_filter(policy: Policy, measured: dict | None = None) -> str:
    """loudnorm, then back to the delivery sample rate (loudnorm works at 192 kHz internally)."""
    return f"{loudnorm_filter(policy, measured)},aresample={int(_loud_cfg(policy).get('sample_rate', 48000))}"


_MEASURED = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")


def parse_loudnorm(stderr: str) -> dict | None:
    """The first pass's JSON (loudnorm print_format=json, the last {...} block on stderr), or None when it is
    missing or unusable (silence measures -inf)."""
    blocks = re.findall(r"\{[^{}]*\}", stderr or "")
    for block in reversed(blocks):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        if not all(k in data for k in _MEASURED):
            continue
        try:
            values = {k: float(data[k]) for k in _MEASURED}
        except (TypeError, ValueError):
            return None
        if any(v != v or v in (float("inf"), float("-inf")) for v in values.values()):
            return None
        return {k: str(data[k]).strip() for k in _MEASURED}
    return None


def measure(video: Path, policy: Policy, *, timeout: float = 600) -> dict | None:
    from .media import ffmpeg
    r = subprocess.run([ffmpeg(), "-hide_banner", "-nostats", "-i", str(video), "-vn",
                        "-af", loudnorm_filter(policy) + ":print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=timeout)
    return parse_loudnorm(r.stderr) if r.returncode == 0 else None


def finish(video: Path, out: Path, *, look: str | dict | None, grain: bool | float | None = None,
           hook_ass: Path | str | None = None, policy: Policy, timeout: float | None = None) -> Path:
    """The delivery frame rate (`finishing.fps`, interpolated up when slower), the upscale to `finishing.output`
    (when smaller), grade, grain (and sharpen / vignette per policy), the hook text, and loudness, into `out`
    (H.264 + AAC), in one ffmpeg pass (plus a loudness measurement)."""
    from .media import ffmpeg, probe
    c = cfg(policy)
    enc = c.get("encode") or {}
    timeout = float(timeout or enc.get("timeout_seconds", 900))
    video, out = Path(video), Path(out)
    info = probe(video, timeout=timeout)
    size = (info["width"], info["height"]) if info["width"] and info["height"] else None
    vf = filters(look, policy=policy, grain=grain, hook_ass=hook_ass, size=size, fps=info.get("fps") or 0.0,
                 seconds=info.get("seconds"))
    cmd = [ffmpeg(), "-y", "-v", "error", "-i", str(video), "-map", "0:v:0", "-vf", vf,
           "-c:v", "libx264", "-preset", str(enc.get("preset", "veryfast")), "-crf", str(enc.get("crf", 18)),
           "-pix_fmt", "yuv420p"]
    if info["audio"]:
        measured = measure(video, policy, timeout=timeout) if _loud_cfg(policy).get("two_pass", True) else None
        cmd += ["-map", "0:a:0", "-af", audio_filter(policy, measured), "-c:a", "aac",
                "-b:a", str(enc.get("audio_bitrate", "160k")), "-ac", "2"]
    cmd += ["-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg could not finish {video.name}: {(r.stderr or '')[-500:]}")
    return out


# --- cover frame ------------------------------------------------------------------------------------------------

def sharpness(gray) -> float:
    """Variance of the 4-neighbour Laplacian: higher is sharper."""
    import numpy as np
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2 or min(g.shape) < 3:
        return 0.0
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1])
    return float(lap.var())


def _gray(path: Path, width: int):
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("L")
        if width and im.width > width:
            im = im.resize((width, max(1, round(im.height * width / im.width))))
        return np.asarray(im, dtype=np.float64)


# Face-mesh landmark ids: eyelids (upper, lower) and eye corners per eye; inner lips and mouth corners.
_EYES = ((159, 145, 33, 133), (386, 374, 362, 263))
_MOUTH = (13, 14, 78, 308)
_landmarker: dict = {}


def mediapipe_face(path: Path) -> dict | None:
    """{eyes_open, mouth_open, box} from MediaPipe's face landmarker, or None when there is no face, or no
    mediapipe / model here (the control-plane image has neither today: the cover is then picked on sharpness)."""
    model = os.environ.get("MW_FACE_LANDMARKER", "/opt/models/face_landmarker.task")
    try:
        import mediapipe as mp
        import numpy as np
        from mediapipe.tasks.python import BaseOptions, vision
        from PIL import Image
    except ImportError:
        return None
    if not Path(model).exists():
        return None
    try:
        if "marker" not in _landmarker:
            _landmarker["marker"] = vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model), num_faces=1))
        with Image.open(path) as im:
            data = np.ascontiguousarray(np.asarray(im.convert("RGB")))
        found = _landmarker["marker"].detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=data)).face_landmarks
    except Exception:  # noqa: BLE001 - a cover is still picked without it
        return None
    if not found:
        return None
    p = found[0]

    def ratio(a, b, left, right):
        width = abs(p[right].x - p[left].x)
        return abs(p[b].y - p[a].y) / width if width else 0.0

    xs, ys = [q.x for q in p], [q.y for q in p]
    return {"eyes_open": min(ratio(*e) for e in _EYES), "mouth_open": ratio(*_MOUTH),
            "box": (max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys)))}


def cover_scores(frames: list[Path], *, policy: Policy | None = None,
                 face: Callable[[Path], dict | None] | None = None) -> list[dict]:
    """Per frame: sharpness (of the face box when a face is found), the face reading, and whether it is a good
    cover face (eyes open over `cover.eyes_open_min`, mouth closed under `cover.mouth_closed_max`)."""
    cc = cfg(policy).get("cover") or {}
    width = int(cc.get("analysis_width", 512))
    eyes_min, mouth_max = float(cc.get("eyes_open_min", 0.18)), float(cc.get("mouth_closed_max", 0.08))
    if face is None:
        face = mediapipe_face if cc.get("mediapipe", True) else (lambda _p: None)
    out = []
    for f in frames:
        f = Path(f)
        try:
            g = _gray(f, width)
        except Exception:  # noqa: BLE001 - an unreadable frame is never the cover
            out.append({"frame": f, "sharpness": -1.0, "face": None, "good_face": False, "readable": False})
            continue
        found = face(f)
        region = g
        if found and found.get("box"):
            x0, y0, x1, y1 = found["box"]
            h, w = g.shape
            crop = g[int(y0 * h):max(int(y0 * h) + 3, int(y1 * h)), int(x0 * w):max(int(x0 * w) + 3, int(x1 * w))]
            region = crop if min(crop.shape) >= 3 else g
        good = bool(found) and float(found.get("eyes_open", 0)) >= eyes_min and float(found.get("mouth_open", 1)) <= mouth_max
        out.append({"frame": f, "sharpness": sharpness(region), "face": found, "good_face": good, "readable": True})
    return out


def pick_cover(frames: list[Path], *, policy: Policy | None = None,
               face: Callable[[Path], dict | None] | None = None) -> Path:
    """The cover: among readable frames, a good cover face (eyes open, mouth closed) beats any face, which
    beats no face; within that, the sharpest. With no face reader it is simply the sharpest frame."""
    if not frames:
        raise ValueError("no frames to pick a cover from")
    scored = cover_scores(frames, policy=policy, face=face)
    best = max(scored, key=lambda s: (s["readable"], s["good_face"], s["face"] is not None, s["sharpness"]))
    return best["frame"]
