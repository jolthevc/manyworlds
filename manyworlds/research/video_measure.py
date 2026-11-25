"""Video measurement for research references: the cut grammar, pace and sampled frames of a public video, on the CPU.

A port of the 2026-11-24 talking-video calibration (docs/research/2026-11-24-talking-video-calibration.md; the
scripts ran by hand on the control plane), as the job `finish.measure_video` on the Fly finisher. The handler
(worker/handlers/finish_video.py) downloads the research copy from the research store, measures it, puts the
sampled frames and a numbered grid back in the research store (for the vision read and for internal review, under
the post's media retention), deletes the local file and returns the measure in the job's result. The control plane
stores it (`collect`: reference_annotations.measured, the frames on reference_posts.media).

    probe(path)                       duration, width, height, fps (ffmpeg's header)
    scene_scores(path, cfg)           [(t, score)] from ffmpeg's scene filter on a small copy
    cuts_at(scores, threshold, gap)   cut times above a threshold, at least `gap` apart
    shot_lengths(cuts, duration)      the shots between cuts
    speech_from_words(words)          pace from word timestamps (Whisper's, when the job carries them)
    speech_from_vtt(text)             pace from a platform's subtitle file (TikTok's native captions)
    measure(path, cfg, ...)           all of it; `frames` sampled evenly in time
    grid(frames, cfg)                 the numbered frame grid the vision read looks at (Pillow)

Hard cuts are scene scores above `intelligence.measure.hard_cut` (0.3 in the calibration), soft ones above
`soft_cut` (0.15: punch-ins). Every number is policy `intelligence.measure`. Nothing is learned from the footage
but these descriptions and parameters; the footage is never a production input (manyworlds.research.references).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from ..policy import Policy

VERSION = "v1"            # bump when a measurement changes; stored with the measure
KIND = "finish.measure_video"


def cfg(policy: Policy | None) -> dict:
    return (policy.get("intelligence", "measure", default={}) or {}) if policy is not None else {}


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: str) -> dict:
    info = subprocess.run([_ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                          timeout=60).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info)
    if not m:
        raise ValueError("not a readable video (no duration)")
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    mv = re.search(r"Video:.*?(\d{2,5})x(\d{2,5}).*?([\d.]+) fps", info)
    return {"duration_s": round(dur, 2), "width": int(mv.group(1)) if mv else None,
            "height": int(mv.group(2)) if mv else None, "fps": float(mv.group(3)) if mv else None,
            "has_audio": "Audio:" in info}


def scene_scores(path: str, c: dict) -> list[tuple[float, float]]:
    """ffmpeg's scene-change score at every frame above `scene_floor`, on a copy scaled to `scale_width`."""
    proc = subprocess.run(
        [_ffmpeg(), "-v", "error", "-hide_banner", "-threads", "2", "-i", str(path), "-an", "-vf",
         f"scale={int(c.get('scale_width', 192))}:-2,select='gt(scene,{float(c.get('scene_floor', 0.08))})',"
         "metadata=print:file=-", "-f", "null", "-"],
        capture_output=True, text=True, timeout=int(c.get("timeout_s", 600)), check=True)
    pts, out = None, []
    for line in proc.stdout.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            pts = float(m.group(1))
        m = re.search(r"scene_score=([\d.]+)", line)
        if m and pts is not None:
            out.append((pts, float(m.group(1))))
    return out


def cuts_at(scores: list[tuple[float, float]], threshold: float, gap: float) -> list[float]:
    res: list[float] = []
    for t, s in scores:
        if s > threshold and (not res or t - res[-1] > gap):
            res.append(round(t, 3))
    return res


def shot_lengths(cuts: list[float], duration: float, *, min_s: float = 0.05) -> list[float]:
    b = [0.0] + [c for c in cuts if 0 < c < duration] + [duration]
    return [round(b[i + 1] - b[i], 2) for i in range(len(b) - 1) if b[i + 1] - b[i] > min_s]


def _q(xs: list[float], p: float) -> float | None:
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo = int(k)
    return round(xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo), 2)


def shots_summary(shots: list[float], cuts: list[float], duration: float, short_s: float) -> dict:
    return {"cuts": len(cuts), "cuts_per_10s": round(len(cuts) / duration * 10, 2) if duration else None,
            "shot_median_s": _q(shots, 0.5), "shot_p25_s": _q(shots, 0.25), "shot_p75_s": _q(shots, 0.75),
            "short_share": round(sum(s < short_s for s in shots) / len(shots), 2) if shots else None,
            "first_cut_s": cuts[0] if cuts else round(duration, 2)}


def speech_from_words(words: list[dict], *, early_s: float = 2.0) -> dict | None:
    """Pace from word timestamps [{start, end, word}]: words per second of speech, not of the video."""
    ws = [w for w in words or [] if w.get("start") is not None and w.get("end") is not None]
    if not ws:
        return None
    speech = sum(max(0.0, float(w["end"]) - float(w["start"])) for w in ws)
    span = float(ws[-1]["end"]) - float(ws[0]["start"])
    return {"words": len(ws), "first_word_s": round(float(ws[0]["start"]), 2), "speech_s": round(speech, 2),
            "span_s": round(span, 2), "words_per_s": round(len(ws) / span, 2) if span > 0 else None,
            "words_in_first_2s": sum(float(w["start"]) < early_s for w in ws), "source": "words"}


_CUE = re.compile(r"(\d+):(\d+):([\d.]+) --> (\d+):(\d+):([\d.]+)[^\n]*\n(.*?)(?:\n\n|\Z)", re.S)


def speech_from_vtt(text: str, *, early_s: float = 2.0) -> dict | None:
    """Pace from a WebVTT subtitle file (the calibration's source on TikTok). Cue time, not word time."""
    cues = []
    for m in _CUE.finditer(text or ""):
        g = m.groups()
        a = int(g[0]) * 3600 + int(g[1]) * 60 + float(g[2])
        b = int(g[3]) * 3600 + int(g[4]) * 60 + float(g[5])
        cues.append((a, b, " ".join(g[6].split())))
    if not cues:
        return None
    words = sum(len(c[2].split()) for c in cues)
    speech = sum(b - a for a, b, _ in cues)
    return {"words": words, "first_word_s": round(cues[0][0], 2), "speech_s": round(speech, 2),
            "span_s": round(cues[-1][1] - cues[0][0], 2),
            "words_per_s": round(words / speech, 2) if speech else None,
            "words_in_first_2s": sum(len(c[2].split()) for c in cues if c[0] < early_s), "source": "vtt"}


def frames(path: str, duration: float, c: dict, workdir: Path) -> list[Path]:
    """`frames` stills sampled evenly in time (the middle of each slice), each scaled into `frame_size`."""
    n = int(c.get("frames", 8))
    w, h = (int(x) for x in (c.get("frame_size") or [180, 320]))
    workdir.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        t = duration * (i + 0.5) / n
        fp = workdir / f"f{i + 1:02d}.jpg"
        subprocess.run([_ffmpeg(), "-v", "error", "-y", "-threads", "2", "-ss", f"{t:.2f}", "-i", str(path),
                        "-frames:v", "1", "-vf",
                        f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2", str(fp)],
                       capture_output=True, timeout=120)
        if fp.exists() and fp.stat().st_size:
            out.append(fp)
    return out


def grid(paths: list[Path], c: dict, dest: Path) -> Path | None:
    """The frames as one numbered grid (1 = earliest, left to right, top to bottom), as the calibration's read."""
    from PIL import Image, ImageDraw
    if not paths:
        return None
    cols = int(c.get("grid_cols", 4))
    w, h = (int(x) for x in (c.get("frame_size") or [180, 320]))
    rows = (len(paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            canvas.paste(im.convert("RGB").resize((w, h)), ((i % cols) * w, (i // cols) * h))
        x, y = (i % cols) * w, (i // cols) * h
        draw.rectangle([x, y, x + 30, y + 22], fill=(0, 0, 0))
        draw.text((x + 6, y + 5), str(i + 1), fill=(255, 255, 255))
    canvas.save(dest, "JPEG", quality=80)
    return dest


def measure(path: str, c: dict, *, words: list[dict] | None = None, vtt: str | None = None,
            workdir: Path | None = None) -> dict:
    """Everything the calibration measured for one video. `workdir` set: frames and the grid are written there."""
    info = probe(path)
    dur = info["duration_s"]
    gap = float(c.get("min_gap_s", 0.25))
    sc = scene_scores(path, c)
    hard, soft = cuts_at(sc, float(c.get("hard_cut", 0.3)), gap), cuts_at(sc, float(c.get("soft_cut", 0.15)), gap)
    short = float(c.get("short_shot_s", 1.0))
    hard_shots, soft_shots = shot_lengths(hard, dur), shot_lengths(soft, dur)
    early = float(c.get("hook_window_s", 2.0))
    speech = speech_from_words(words or [], early_s=early) if words else (
        speech_from_vtt(vtt, early_s=early) if vtt else None)
    out: dict[str, Any] = {
        "version": VERSION, **info,
        "hard": {**shots_summary(hard_shots, hard, dur, short), "at": hard},
        "soft": {**shots_summary(soft_shots, soft, dur, short), "at": soft},
        "speech": speech,
        "speech_share": round(min(1.0, speech["speech_s"] / dur), 2) if speech and dur else None,
    }
    if workdir is not None:
        fs = frames(path, dur, c, Path(workdir) / "frames")
        g = grid(fs, c, Path(workdir) / "grid.jpg")
        out["_frames"] = [str(p) for p in fs]
        out["_grid"] = str(g) if g else None
    return out


# --- the job, on both sides -----------------------------------------------------------------------------------
def payload(policy: Policy, post: dict, *, words: list[dict] | None = None, vtt: str | None = None) -> dict:
    """The finish.measure_video payload: everything the finisher needs, no database there."""
    video = next((m for m in post.get("media") or [] if m.get("kind") == "video"), None)
    if not video:
        raise ValueError(f"reference post {post['id']} has no research copy of its video")
    return {"post_id": str(post["id"]), "media_uri": video["uri"], "words": words, "vtt": vtt,
            "policy": {"intelligence": {"measure": cfg(policy)}}, "policy_version": policy.version}


def measure_files(p: dict, store, workdir: Path) -> dict:
    """The handler's work: download, measure, upload the frames and grid to the research store, delete local files."""
    import shutil
    import tempfile
    from .references import media_key
    c = (p.get("policy") or {}).get("intelligence", {}).get("measure") or {}
    workdir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="mw-vid-", dir=str(workdir)))
    try:
        src = store.get_file(p["media_uri"], tmp / ("video" + (Path(p["media_uri"]).suffix or ".mp4")))
        out = measure(str(src), c, words=p.get("words"), vtt=p.get("vtt"), workdir=tmp)
        media = []
        for fp in out.pop("_frames", []):
            o = store.put_file(Path(fp), media_key(p["post_id"], f"frames/{Path(fp).name}"))
            media.append({"uri": o.uri, "kind": "frame", "bytes": o.bytes})
        g = out.pop("_grid", None)
        if g:
            o = store.put_file(Path(g), media_key(p["post_id"], "frames/grid.jpg"))
            media.append({"uri": o.uri, "kind": "grid", "bytes": o.bytes})
        out["media"] = media
        out["policy_version"] = p.get("policy_version")
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def enqueue(conn, policy: Policy, post: dict, *, batch: str | None = None, **speech):
    from .. import queue
    return queue.enqueue(conn, KIND, payload(policy, post, **speech), batch=batch or f"research-{post['id']}",
                         idempotency_key=f"measure-video:{post['id']}:{VERSION}")


def record(conn, policy: Policy, post_id, out: dict) -> None:
    """A measure stored: reference_annotations.measured (schema measure_<version>), its frames added to the post's
    research media under the post's retention."""
    from .references import media_purge_after
    media = out.pop("media", [])
    conn.execute(
        """INSERT INTO reference_annotations (post_id, schema, model, measured) VALUES (%s, %s, 'ffmpeg', %s)
           ON CONFLICT (post_id, schema) DO UPDATE SET measured = EXCLUDED.measured, updated_at = now()""",
        (post_id, f"measure_{out.get('version') or VERSION}", Jsonb(out)))
    if media:
        conn.execute("""UPDATE reference_posts SET media = media || %s,
                        media_purge_after = coalesce(media_purge_after, %s) WHERE id = %s""",
                     (Jsonb(media), media_purge_after(policy), post_id))


def collect(conn, policy: Policy) -> dict[str, int]:
    """Finished finish.measure_video jobs into the research tables, each once."""
    res = {"measured": 0, "failed": 0}
    for j in conn.execute(
            "SELECT id, status, result, error, payload->>'post_id' AS post_id FROM jobs WHERE kind = %s "
            "AND status IN ('done', 'failed', 'dead') AND NOT coalesce((result->>'recorded')::boolean, false)",
            (KIND,)).fetchall():
        r = j["result"] or {}
        if j["status"] == "done" and r.get("video_measure"):
            record(conn, policy, j["post_id"], dict(r["video_measure"]))
            res["measured"] += 1
        else:
            res["failed"] += 1
        conn.execute("UPDATE jobs SET result = coalesce(result, '{}'::jsonb) || jsonb_build_object('recorded', true) "
                     "WHERE id = %s", (j["id"],))
    return res
