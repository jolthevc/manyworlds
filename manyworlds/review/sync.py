"""Lip-sync score of the finished file, on the CPU: does the mouth open when the voice is loud?

The clip gate (manyworlds.review.video) reads each talking take's score from its worker (worker/lipsync.py), before
the take is joined, mixed and finished. Nothing until now measured the file that is posted: long talks joined from
segments on the control plane (media.py), walk-and-talk takes lip-synced onto a motion clip (V2V), and whatever the
sound mix, trim and finish did to the timing. This does, once per post, from final.mp4:

  frames    decoded with ffmpeg at `sample_fps` (fewer when the file is longer than `max_frames` allows), scaled to
            `frame_height` px
  openness  per frame, the inner-lip gap over the mouth width, from a landmark extractor (injectable; the real one is
            MediaPipe's face landmarker, Apache-2.0 code and model, imported only when it runs)
  loudness  the voice's log RMS in one window per sampled frame (mono, `audio_rate` Hz, from ffmpeg)
  score     the best Pearson (normalised cross-) correlation of openness with loudness within +-`max_lag_ms`
            (worker/lipsync.sync_score) and the lag where it peaks (positive: the mouth follows the voice)
  shares    silent_mouth_moving: of the frames with a face and no voice (loudness more than `speech_db_below_p95` dB
            under its 95th percentile), the share with the mouth open; talking_mouth_closed: of the frames with a
            face and voice, the share with the mouth shut. Open is above max(`open_min`, `open_relative_to_p90` x the
            clip's 90th-percentile openness); shut is at most max(`closed_max`, `closed_relative_to_p90` x it)

Report-only by default (`review.sync.enforce: false`): the verdict is recorded and `passed` is True whatever it says,
as review.final and review.detail do. The result lands on the post (body.sync_review), as one `reviews` row (gate
`final_lip_sync`) on the finished video asset when it is found, and as a `review.sync` decision. It never raises: a
file that can't be scored records why. Policy `review.sync`.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable, Iterable

from psycopg.types.json import Jsonb

from ..policy import Policy

# Extractor: (RGB frames as HxWx3 uint8 arrays, their fps) -> openness per frame (NaN where no face).
Extractor = Callable[[Iterable, float], "object"]

# Face-mesh landmark ids: inner upper and lower lip, mouth corners (as worker/lipsync.py).
UPPER, LOWER, LEFT, RIGHT = 13, 14, 78, 308


def cfg(policy: Policy | None) -> dict:
    return (policy.get("review", "sync", default={}) or {}) if policy is not None else {}


def applies(policy: Policy | None, post: dict) -> bool:
    c = cfg(policy)
    return bool(c.get("enabled")) and post.get("format") in set(c.get("formats") or [])


# ------------------------------------------------------------------------------------------- frames and audio

def sample_rate(seconds: float, fps: float | None, policy: Policy | None) -> float:
    """Frames per second to sample: the file's own rate up to `sample_fps`, lower when that would exceed
    `max_frames` over `seconds`."""
    c = cfg(policy)
    rate = min(float(fps or c.get("sample_fps", 25)), float(c.get("sample_fps", 25)))
    cap = int(c.get("max_frames", 900))
    if seconds and seconds * rate > cap:
        rate = cap / seconds
    return max(rate, 1e-3)


def iter_frames(video: Path, rate: float, width: int, height: int, timeout: float):
    """RGB frames of `video` at `rate` fps, `width` x `height`, streamed from ffmpeg (never all held at once)."""
    import numpy as np

    from .. import media
    cmd = [media.ffmpeg(), "-v", "error", "-nostdin", "-i", str(video), "-an",
           "-vf", f"fps={rate:.6f},scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    size = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()


def read_audio(video: Path, rate: int, timeout: float):
    """The file's sound as mono float samples at `rate` Hz; None when it has no audio stream."""
    import numpy as np

    from .. import media
    r = subprocess.run([media.ffmpeg(), "-v", "error", "-nostdin", "-i", str(video), "-vn", "-ac", "1",
                        "-ar", str(int(rate)), "-f", "s16le", "-"], capture_output=True, timeout=timeout)
    if r.returncode != 0 or not r.stdout:
        return None
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float64) / 32768.0


# ------------------------------------------------------------------------------------------------ the extractor

def mediapipe_extractor(model_path: str) -> Extractor:
    """The real extractor: MediaPipe's face landmarker (VIDEO mode, one face) on each frame. mediapipe is imported
    only when the extractor runs (the control-plane image's `sync` extra); without it, the import error is raised to
    review_sync, which records it."""

    def extract(frames: Iterable, fps: float):
        import mediapipe as mp
        import numpy as np
        from mediapipe.tasks.python import BaseOptions, vision

        if not Path(model_path).exists():
            raise FileNotFoundError(f"no face landmarker model at {model_path}")
        options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
                                               running_mode=vision.RunningMode.VIDEO, num_faces=1)
        values = []
        with vision.FaceLandmarker.create_from_options(options) as marker:
            for i, frame in enumerate(frames):
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
                found = marker.detect_for_video(image, int(round(i * 1000 / fps))).face_landmarks
                if not found:
                    values.append(np.nan)
                    continue
                p = found[0]
                width = abs(p[RIGHT].x - p[LEFT].x)
                values.append(abs(p[LOWER].y - p[UPPER].y) / width if width else np.nan)
        return np.array(values, dtype=float)

    return extract


def default_extractor(policy: Policy | None) -> Extractor:
    path = os.environ.get("MW_FACE_LANDMARKER") or str(cfg(policy).get("landmarker_path"))
    return mediapipe_extractor(path)


# ----------------------------------------------------------------------------------------------------- the maths

def score(openness, loud, fps: float, policy: Policy | None) -> dict:
    """The sync score of one series pair (same length, one value per sampled frame): {corr, lag_ms, offset_frames,
    face_share, frames, speech_share, silent_mouth_moving, talking_mouth_closed, ...}. Pure numpy."""
    import numpy as np

    from ..worker import lipsync
    c = cfg(policy)
    n = min(len(openness), len(loud))
    a = np.asarray(openness[:n], dtype=float)
    b = np.asarray(loud[:n], dtype=float)
    max_off = int(round(float(c.get("max_lag_ms", 200)) / 1000.0 * fps))
    out = lipsync.sync_score(a, b, max_offset_frames=max_off, min_frames=int(c.get("min_frames", 25)))
    out["lag_ms"] = (round(out["offset_frames"] * 1000.0 / fps, 1) if out.get("offset_frames") is not None else None)
    out["max_lag_ms"] = float(c.get("max_lag_ms", 200))
    db = b * (20.0 / np.log(10.0))   # natural-log RMS -> dB
    speech = db >= (float(np.percentile(db, 95)) - float(c.get("speech_db_below_p95", 30))) if n else np.zeros(0, bool)
    face = ~np.isnan(a)
    valid = a[face]
    p90 = float(np.percentile(valid, 90)) if len(valid) else 0.0
    open_t = max(float(c.get("open_min", 0.08)), float(c.get("open_relative_to_p90", 0.35)) * p90)
    shut_t = max(float(c.get("closed_max", 0.05)), float(c.get("closed_relative_to_p90", 0.15)) * p90)
    filled = np.where(face, a, 0.0)
    silent, talking = face & ~speech, face & speech
    out.update(
        speech_share=round(float(speech.mean()), 3) if n else None,
        silent_mouth_moving=(round(float((filled[silent] > open_t).mean()), 3) if silent.any() else None),
        talking_mouth_closed=(round(float((filled[talking] <= shut_t).mean()), 3) if talking.any() else None),
        open_threshold=round(open_t, 4), closed_threshold=round(shut_t, 4), fps=round(fps, 3))
    return out


def decide(s: dict, policy: Policy | None) -> tuple[bool, list[str]]:
    """(would pass, reasons it wouldn't) against `min_corr`, `min_face_share`, `max_silent_mouth_moving` and
    `max_talking_mouth_closed`."""
    c = cfg(policy)
    why = []
    if s.get("corr") is None or float(s["corr"]) < float(c.get("min_corr", 0.3)):
        why.append(f"corr {s.get('corr')} under {c.get('min_corr')}")
    if float(s.get("face_share") or 0) < float(c.get("min_face_share", 0.8)):
        why.append(f"face in {s.get('face_share')} of frames, under {c.get('min_face_share')}")
    if (s.get("silent_mouth_moving") or 0) > float(c.get("max_silent_mouth_moving", 0.3)):
        why.append(f"mouth moving in silence {s['silent_mouth_moving']}")
    if (s.get("talking_mouth_closed") or 0) > float(c.get("max_talking_mouth_closed", 0.5)):
        why.append(f"mouth shut while talking {s['talking_mouth_closed']}")
    return not why, why


def measure(video: Path, policy: Policy | None, extractor: Extractor | None = None) -> dict:
    """The score of one file (no database): samples frames and sound, extracts openness, scores. Raises on an
    unreadable file or a failing extractor; {"skipped": "no audio"} without sound."""
    import numpy as np

    from .. import media
    from ..worker import lipsync
    c = cfg(policy)
    video = Path(video)
    timeout = float(c.get("timeout_seconds", 300))
    info = media.probe(video)
    if not info.get("audio"):
        return {"skipped": "no audio"}
    if not (info.get("width") and info.get("height")):
        raise RuntimeError(f"can't read the frame size of {video.name}")
    seconds = float(info.get("seconds") or 0)
    rate = sample_rate(seconds, info.get("fps"), policy)
    h = int(c.get("frame_height", 480))
    h = min(h, int(info["height"])) // 2 * 2
    w = max(2, int(round(info["width"] * h / info["height"])) // 2 * 2)
    openness = np.asarray((extractor or default_extractor(policy))(iter_frames(video, rate, w, h, timeout), rate),
                          dtype=float)
    samples = read_audio(video, int(c.get("audio_rate", 16000)), timeout)
    if samples is None:
        return {"skipped": "no audio"}
    loud = lipsync.loudness(samples, int(c.get("audio_rate", 16000)), rate, len(openness))
    return {**score(openness, loud, rate, policy), "seconds": seconds, "size": [w, h]}


# ---------------------------------------------------------------------------------------------------- the review

def review_sync(conn, policy: Policy, post: dict, video_path, *, extractor: Extractor | None = None) -> dict:
    """Score one post's finished talking video once. Returns {passed, accepted, enforced, corr, lag_ms, ...};
    `passed` is True while report-only, and whenever the file can't be scored. Off or not a listed format:
    {"skipped": "off", "passed": True} (not stored). Stored on the post (body.sync_review; the caller's
    `post["body"]` dict too)."""
    if not applies(policy, post):
        return {"skipped": "off", "passed": True}
    c = cfg(policy)
    enforce = bool(c.get("enforce"))
    try:
        s = measure(Path(video_path), policy, extractor)
    except Exception as e:  # noqa: BLE001 - a score that can't be taken is recorded, never fatal
        out = {"error": f"{type(e).__name__}: {e}"[:500], "passed": True, "enforced": enforce,
               "policy_version": policy.version}
        _store(conn, post, out)
        return out
    if s.get("skipped"):
        out = {**s, "passed": True, "enforced": enforce, "policy_version": policy.version}
        _store(conn, post, out)
        return out
    ok, why = decide(s, policy)
    out = {**s, "passed": ok or not enforce, "accepted": ok, "enforced": enforce, "reasons": why,
           "policy_version": policy.version}
    _store(conn, post, out)
    asset = conn.execute("SELECT id FROM assets WHERE kind = 'video' AND uri LIKE %s ORDER BY created_at DESC LIMIT 1",
                         (f"%/post-{post['id']}/final.mp4",)).fetchone()
    if asset:
        conn.execute("INSERT INTO reviews (asset_id, gate, passed, score, details) VALUES (%s, 'final_lip_sync', %s, "
                     "%s, %s)", (asset["id"], ok or not enforce, s.get("corr"),
                                 Jsonb({**out, "would_pass": ok, "stage": "final"})))
    conn.execute("INSERT INTO decisions (kind, creator_id, inputs, output, model, policy_version) "
                 "VALUES ('review.sync', %s, %s, %s, %s, %s)",
                 (post["creator_id"], Jsonb({"post_id": str(post["id"]), "frames": s.get("frames"),
                                             "fps": s.get("fps"), "seconds": s.get("seconds")}),
                  Jsonb({k: out.get(k) for k in ("passed", "accepted", "enforced", "corr", "lag_ms", "face_share",
                                                 "silent_mouth_moving", "talking_mouth_closed", "reasons")}),
                  "mediapipe-face-landmarker" if extractor is None else "injected", policy.version))
    return out


def _store(conn, post: dict, out: dict) -> None:
    conn.execute("UPDATE posts SET body = body || %s, updated_at = now() WHERE id = %s",
                 (Jsonb({"sync_review": out}), post["id"]))
    if isinstance(post.get("body"), dict):
        post["body"]["sync_review"] = out
