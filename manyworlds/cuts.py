"""Cut-based talking edits: which edit grammar a talking post follows, and the punch-ins its finish adds.

Social talking videos cut every few seconds; long takes read as AI (a frozen hand, a slow drift) and cost the most
GPU. So a talking post follows an edit grammar (policy `generation.talk_edit.grammars`): `story_time` (one
lip-synced take per speaking shot, punched in and out at word gaps so it reads as jump cuts, occasional cutaways under
the same voice), `commercial` (several angles of one scene, each its own first frame; the voice carries over the
shots that don't show her face to camera, which are not lip-synced) or `long_take` (one take, no punch-ins).

The grammar is the idea's own (`body.edit_grammar`), else her primary archetype's (`by_archetype`), else the
`default`. It tells the director how to cut (its `director` rules join the brief), production which shots need lip
sync (`lip_sync: on_camera`: a faceless shot, or an angle in `narrated_angles`, is a cutaway under the voice), and
the finish where to punch in (`punch_in`), all done on the CPU: a crop and scale per stretch, cut only between words,
the voice and its timing untouched, so the captions' word timings still hold.
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

GRAMMARS_KEY = ("generation", "talk_edit")


def _cfg(policy) -> dict:
    return policy.get(*GRAMMARS_KEY, default={}) or {}


def grammar_names(policy) -> list[str]:
    return list((_cfg(policy).get("grammars") or {}).keys())


def grammar(policy, name: str | None) -> dict:
    """A grammar's settings ({} for an unknown name)."""
    return dict(((_cfg(policy).get("grammars") or {}).get(name or "")) or {})


def grammar_for(policy, name: str | None, body: dict | None) -> dict:
    """A grammar's settings with the post's pattern-card overrides (body.edit_overrides, already clamped by
    manyworlds.brain.patterns.talk_overrides): the punch-in rhythm `every_s` when the grammar punches in at all."""
    g = grammar(policy, name)
    over = (body or {}).get("edit_overrides") or {}
    every = over.get("punch_every_s")
    if g.get("punch_in") and isinstance(every, (list, tuple)) and len(every) == 2:
        g["punch_in"] = {**g["punch_in"], "every_s": [float(every[0]), float(every[1])]}
    return g


def choose(policy, body: dict | None, who: dict | None = None) -> str:
    """The grammar a talking post follows: the idea's `edit_grammar` when it names a known one, else her first
    archetype's (`by_archetype`, archetypes.for_creator on `who`: her profile or bible), else the `default`."""
    cfg = _cfg(policy)
    known = grammar_names(policy)
    asked = str((body or {}).get("edit_grammar") or "")
    if asked in known:
        return asked
    if who:
        from .brain import archetypes
        by = cfg.get("by_archetype") or {}
        for name in archetypes.for_creator(policy, who):
            if by.get(name) in known:
                return str(by[name])
    default = str(cfg.get("default") or "")
    return default if default in known else (known[0] if known else "")


def choose_for(conn, policy, body: dict, creator_id) -> str:
    """`choose` with the creator read from the database: her current profile, else her bible."""
    from .creator import profile as profile_mod
    who = None
    try:
        cur = profile_mod.current(conn, creator_id)
        who = cur[1] if cur else None
    except Exception:  # noqa: BLE001 - no profile: the bible
        who = None
    if not who:
        row = conn.execute("SELECT bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
        who = (row or {}).get("bible") or {}
    return choose(policy, body, who)


def narrates(shot: Any, policy, name: str | None) -> bool:
    """Whether a speaking shot is a cutaway under the voice instead of a lip-synced take, by the grammar: with
    `lip_sync: on_camera`, a shot without her face to camera (direction.faceless) or framed from an angle in
    `narrated_angles` (over the shoulder, from behind); with `all`, only the director's own cutaways."""
    from . import direction
    g = grammar(policy, name)
    if str(g.get("lip_sync") or "all") != "on_camera":
        return False
    s = direction._coerce(shot)
    if direction.faceless(s):
        return True
    angle = (s.framing.angle or "").lower()
    return any(str(a).lower() in angle for a in g.get("narrated_angles") or [] if str(a).strip())


def director_rules(policy, name: str) -> str:
    """The grammar's rules for the director's brief, one line each; "" without any."""
    g = grammar(policy, name)
    rules = [str(r).strip() for r in g.get("director") or [] if str(r).strip()]
    if not rules:
        return ""
    return f"Edit grammar for this talking post: {name}.\n" + "\n".join(f"- {r}" for r in rules) + "\n"


# --- punch-ins (the finish) ---------------------------------------------------------------------------------------

def _gaps(words: Sequence[Any], a: float, b: float) -> list[float]:
    """Where a cut may go inside [a, b]: between two words (the middle of the silence between them, or the first
    word's end when they touch), never inside one."""
    out = []
    timed = sorted((float(_get(w, "start")), float(_get(w, "end"))) for w in words)
    for (s0, e0), (s1, _e1) in zip(timed, timed[1:]):
        t = (e0 + s1) / 2 if s1 > e0 else e0
        if a < t < b and not any(s < t < e for s, e in timed):
            out.append(round(t, 3))
    return out


def _get(w: Any, k: str):
    return w[k] if isinstance(w, dict) else getattr(w, k)


def punch_plan(words: Sequence[Any], windows: Iterable[tuple[float, float]], duration: float, cfg: dict, *,
               seed: int = 0, fps: float | None = None) -> list[dict]:
    """The finish's punch-ins: stretches {start, end, scale} covering [0, duration] in order. Inside each window (a
    lip-synced take's span) the frame changes scale about every `every_s` [lo, hi] seconds (seeded), cycling
    through `scales` (the first stretch of a window at scales[0]), each cut moved to the nearest gap between words
    and kept at least `min_shot_s` from the last cut and the window's end; outside the windows, scale 1. Cut times
    land on frame boundaries at `fps`. [] when nothing would change (no scales above 1, no words, no window)."""
    scales = [float(s) for s in cfg.get("scales") or []]
    if not words or not duration or not scales or max(scales) <= 1.0:
        return []
    lo, hi = (float(v) for v in cfg.get("every_s") or (2.0, 4.0))
    min_shot = float(cfg.get("min_shot_s", 1.0))
    rng = random.Random(seed)

    def snap(t: float) -> float:
        return round(round(t * fps) / fps, 3) if fps else round(t, 3)

    cuts: list[tuple[float, float]] = []   # (time, scale from then on)
    for a, b in sorted((max(0.0, float(a)), min(float(duration), float(b))) for a, b in windows):
        if b - a < 2 * min_shot:
            continue
        gaps = _gaps(words, a, b)
        cuts.append((snap(a), scales[0]))
        k, last = 0, a
        while True:
            target = last + rng.uniform(lo, hi)
            ok = [g for g in gaps if g - last >= min_shot and b - g >= min_shot]
            if not ok or target >= b - min_shot:
                break
            t = min(ok, key=lambda g: abs(g - target))
            if t <= last:
                break
            k += 1
            cuts.append((snap(t), scales[k % len(scales)]))
            last = t
        cuts.append((snap(b), 1.0))
    if not cuts:
        return []
    out: list[dict] = []
    t, scale = 0.0, 1.0
    for at, nxt in cuts:
        if at > t:
            out.append({"start": t, "end": at, "scale": scale})
        t, scale = max(t, at), nxt
    if t < duration:
        out.append({"start": t, "end": round(float(duration), 3), "scale": scale})
    merged: list[dict] = []
    for s in out:   # neighbours at one scale are one stretch
        if merged and merged[-1]["scale"] == s["scale"]:
            merged[-1]["end"] = s["end"]
        else:
            merged.append(dict(s))
    return merged if any(s["scale"] > 1.0 for s in merged) else []


def crop_box(size: tuple[int, int], scale: float, center: tuple[float, float]) -> tuple[int, int, int, int]:
    """(w, h, x, y) in pixels of a frame punched in by `scale` round `center` (shares of the frame), kept inside the
    frame, even sizes (yuv420p)."""
    width, height = size
    w = max(2, int(width / scale) // 2 * 2)
    h = max(2, int(height / scale) // 2 * 2)
    x = int(round(center[0] * width - w / 2))
    y = int(round(center[1] * height - h / 2))
    return w, h, min(max(0, x), width - w), min(max(0, y), height - h)


def center_for(face: Sequence[float] | None, cfg: dict) -> tuple[float, float]:
    """Where to punch in: the middle of the face box (shares of the frame) when there is one, else `center`."""
    if face:
        x0, y0, x1, y1 = (float(v) for v in face)
        return ((x0 + x1) / 2, (y0 + y1) / 2)
    c = cfg.get("center") or (0.5, 0.4)
    return (float(c[0]), float(c[1]))


def punched_face_box(face: Sequence[float] | None, plan: list[dict], size: tuple[int, int],
                     center: tuple[float, float]) -> tuple[float, float, float, float] | None:
    """The face box as it shows after the punch-ins: the union of the box at every stretch's scale, so captions and
    labels laid out once stay off the face in the widest and the tightest framing."""
    if not face or not plan:
        return tuple(face) if face else None
    width, height = size
    x0, y0, x1, y1 = (float(v) for v in face)
    boxes = []
    for scale in sorted({float(s["scale"]) for s in plan}):
        w, h, cx, cy = crop_box(size, scale, center)
        fx = lambda v: min(1.0, max(0.0, (v * width - cx) / w))   # noqa: E731
        fy = lambda v: min(1.0, max(0.0, (v * height - cy) / h))   # noqa: E731
        boxes.append((fx(x0), fy(y0), fx(x1), fy(y1)))
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def punch_filter(plan: list[dict], size: tuple[int, int], center: tuple[float, float]) -> str:
    """The filter graph: each stretch trimmed, cropped and scaled back to `size`, then joined. Video only: the
    audio is copied as it is, so the voice and the captions' timing never move."""
    width, height = size
    parts, labels = [f"[0:v]split={len(plan)}" + "".join(f"[s{i}]" for i in range(len(plan)))], []
    for i, s in enumerate(plan):
        chain = f"[s{i}]trim=start={s['start']}:end={s['end']},setpts=PTS-STARTPTS"
        if float(s["scale"]) > 1.0:
            w, h, x, y = crop_box(size, float(s["scale"]), center)
            chain += f",crop={w}:{h}:{x}:{y},scale={width}:{height}:flags=lanczos"
        chain += f",setsar=1[p{i}]"
        parts.append(chain)
        labels.append(f"[p{i}]")
    parts.append("".join(labels) + f"concat=n={len(plan)}:v=1:a=0,format=yuv420p[v]")
    return ";".join(parts)


def render_punch_ins(src: Path, out: Path, plan: list[dict], size: tuple[int, int], center: tuple[float, float], *,
                     timeout: float = 600) -> Path:
    """The video with its punch-ins (punch_filter), re-encoded; the audio stream copied untouched."""
    from .media import ffmpeg
    cmd = [ffmpeg(), "-y", "-v", "error", "-i", str(src), "-filter_complex", punch_filter(plan, size, center),
           "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
           "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg could not punch in {Path(src).name}: {(r.stderr or '')[-500:]}")
    return out


def talk_windows(body: dict, meta: dict, pieces: list[dict], duration: float | None) -> list[tuple[float, float]]:
    """The spans of the made video that are lip-synced takes (punch-ins happen only there, never over a cutaway):
    each piece of a joined video whose `talk_modes` entry is 'talk' (placed by the join's measured lengths, else
    the piece's own); a single piece, the whole video. [] when a length is unknown."""
    if not duration:
        return []
    modes = body.get("talk_modes") or []
    if len(pieces) <= 1:
        return [(0.0, float(duration))] if not modes or modes[0] == "talk" else []
    measured = (meta or {}).get("segment_seconds") or []
    out, t = [], 0.0
    for i, piece in enumerate(pieces):
        length = measured[i] if i < len(measured) and measured[i] is not None else (piece.get("meta") or {}).get("seconds")
        if length is None:
            return out
        end = min(float(duration), t + float(length))
        if (modes[i] if i < len(modes) else "talk") == "talk":
            out.append((t, end))
        t = end
    return out
