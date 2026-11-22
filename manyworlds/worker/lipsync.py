"""A lip-sync score that needs no restricted weights: does the mouth open when the voice is loud?

The video gate judges keyframes, so it cannot see sync. SyncNet's weights are research-only and LatentSync's
licence is unread (model review, 2026-11-21), so this measures sync directly:

  openness  per video frame, the inner-lip gap over the mouth width, from MediaPipe's face landmarker
            (Apache-2.0 code and model)
  loudness  the voice's RMS energy (log), in windows at the video's frame rate

A clip in sync has the mouth tracking the loudness at a small lag; a frozen, mumbling or drifting mouth does
not. The score is their best Pearson correlation within `max_offset_frames`, and the lag where it peaks.
It is uncalibrated until talk clips pass and fail by eye, so the gate records it and only rejects on it
when policy `review.video.lip_sync.enforce` is on.

`sync_score` is pure numpy (tested in CI); `mouth_openness` needs mediapipe, in the talk worker image only.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

LANDMARKER = os.environ.get("MW_FACE_LANDMARKER", "/opt/models/face_landmarker.task")
# Face-mesh landmark ids: inner upper and lower lip, mouth corners.
UPPER, LOWER, LEFT, RIGHT = 13, 14, 78, 308


def loudness(samples: np.ndarray, rate: int, fps: float, frames: int) -> np.ndarray:
    """Log RMS energy of the audio in one window per video frame."""
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    hop = rate / fps
    out = np.zeros(frames)
    for i in range(frames):
        w = samples[int(i * hop):int((i + 1) * hop)]
        out[i] = np.log(np.sqrt(np.mean(w.astype(np.float64) ** 2)) + 1e-4) if len(w) else np.log(1e-4)
    return out


def sync_score(openness: np.ndarray, loud: np.ndarray, *, max_offset_frames: int = 5,
               min_frames: int = 25) -> dict:
    """Best Pearson correlation of mouth openness with loudness over lags of up to `max_offset_frames`
    (positive lag: the mouth follows the voice). Frames with no face are NaN and left out."""
    n = min(len(openness), len(loud))
    a, b = np.asarray(openness[:n], dtype=float), np.asarray(loud[:n], dtype=float)
    face = float(np.mean(~np.isnan(a))) if n else 0.0
    best = {"corr": None, "offset_frames": None, "face_share": round(face, 3), "frames": n}
    for lag in range(-max_offset_frames, max_offset_frames + 1):
        x, y = (a[lag:], b[:n - lag]) if lag >= 0 else (a[:n + lag], b[-lag:])
        keep = ~np.isnan(x) & ~np.isnan(y)
        # a mouth that never moves (or silent audio) has no correlation to measure, not a correlation of 0
        if keep.sum() < min_frames or np.std(x[keep]) < 1e-6 or np.std(y[keep]) < 1e-6:
            continue
        r = float(np.corrcoef(x[keep], y[keep])[0, 1])
        if not np.isfinite(r):
            continue
        if best["corr"] is None or r > best["corr"]:
            best.update(corr=round(r, 3), offset_frames=lag)
    return best


BILABIALS = "bpm"


def bilabial_onsets(words: list[dict]) -> list[float]:
    """Seconds at which a b, p or m sound starts, from words with times ({text, start_s, end_s}). Spelling, not
    phonemes: each letter's time is its share of the word's span; a run of bilabial letters is one closure
    ("summer", "jumped"); "ph" is an f and a word-final "mb" is silent ("lamb"), so neither counts."""
    out = []
    for w in words:
        text = "".join(ch for ch in str(w.get("text", "")).lower() if ch.isalpha())
        start, end = float(w.get("start_s", 0)), float(w.get("end_s", 0))
        if not text or end < start:
            continue
        prev = False
        for i, ch in enumerate(text):
            is_b = ch in BILABIALS
            if ch == "p" and text[i + 1:i + 2] == "h":
                is_b = False
            if ch == "b" and i == len(text) - 1 and i > 0 and text[i - 1] == "m":
                is_b = False
            if is_b and not prev:
                out.append(start + (end - start) * i / len(text))
            prev = is_b
    return out


def estimate_words(script: str = "", seconds: float = 0.0, lines: list[dict] | None = None) -> list[dict]:
    """Word times when the voice has none: each line's span (voice.speak's per-line timings) or, without lines,
    the script over `seconds`, shared out by word length."""
    spans = ([(str(ln.get("text", "")), float(ln.get("start_s", 0)), float(ln.get("end_s", 0))) for ln in lines]
             if lines else [(script or "", 0.0, float(seconds))])
    out = []
    for text, start, end in spans:
        words = text.split()
        weights = [max(1, sum(ch.isalpha() for ch in w)) + 1 for w in words]
        total, at = sum(weights), start
        for w, k in zip(words, weights):
            dur = (end - start) * k / total if total else 0.0
            out.append({"text": w, "start_s": round(at, 3), "end_s": round(at + dur, 3)})
            at += dur
    return out


def lip_closure(openness, fps: float, onsets_s: list[float], *, tolerance_frames: int = 2,
                closed_max: float = 0.05, relative_to_p90: float = 0.0, offset_frames: int = 0,
                min_bilabials: int = 1) -> dict:
    """Do the lips close at each bilabial? A closure is the minimum openness within ±`tolerance_frames` of the
    onset's frame (shifted by the sync lag `offset_frames`) at most the closed threshold: `closed_max`, or
    `relative_to_p90` x the clip's 90th-percentile openness when that is larger (a mouth that opens wide).
    Onsets with no face in their window are not counted. `share` is None under `min_bilabials`."""
    a = np.array([np.nan if v is None else float(v) for v in openness], dtype=float)
    valid = a[~np.isnan(a)]
    thresh = float(closed_max)
    if relative_to_p90 and len(valid):
        thresh = max(thresh, float(relative_to_p90) * float(np.percentile(valid, 90)))
    checked = closed = 0
    misses = []
    for t in onsets_s:
        f = int(round(t * fps)) + int(offset_frames)
        lo, hi = max(0, f - tolerance_frames), min(len(a), f + tolerance_frames + 1)
        if lo >= hi:
            continue
        w = a[lo:hi]
        w = w[~np.isnan(w)]
        if not len(w):
            continue
        checked += 1
        if float(w.min()) <= thresh:
            closed += 1
        elif len(misses) < 20:
            misses.append(round(float(t), 2))
    return {"bilabials": checked, "closed": closed,
            "share": round(closed / checked, 3) if checked >= max(1, min_bilabials) else None,
            "threshold": round(thresh, 4), "tolerance_frames": tolerance_frames, "missed_at_s": misses}


def series(values: np.ndarray, digits: int = 3) -> list:
    """A per-frame series as JSON (NaN is not JSON: no face is null)."""
    return [None if not np.isfinite(v) else round(float(v), digits) for v in np.asarray(values, dtype=float)]


def mouth_openness(video: Path) -> tuple[np.ndarray, float]:
    """(openness per frame, NaN where no face was found; the video's fps)."""
    import imageio.v3 as iio
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    fps = float(iio.immeta(str(video)).get("fps") or 25)
    options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=LANDMARKER),
                                           running_mode=vision.RunningMode.VIDEO, num_faces=1)
    values = []
    with vision.FaceLandmarker.create_from_options(options) as marker:
        for i, frame in enumerate(iio.imiter(str(video))):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame[..., :3]))
            found = marker.detect_for_video(image, int(i * 1000 / fps)).face_landmarks
            if not found:
                values.append(np.nan)
                continue
            p = found[0]
            width = abs(p[RIGHT].x - p[LEFT].x)
            values.append(abs(p[LOWER].y - p[UPPER].y) / width if width else np.nan)
    return np.array(values), fps


def score_clip(video: Path, audio: Path, *, max_offset_frames: int = 5, words: list[dict] | None = None) -> dict:
    """The sync score, plus the per-frame mouth series and fps (so the reviewer can check lip closure on b/p/m
    against the script with its own policy), plus that closure here when timed `words` are given."""
    import soundfile as sf
    openness, fps = mouth_openness(video)
    samples, rate = sf.read(str(audio))
    out = sync_score(openness, loudness(np.asarray(samples), rate, fps, len(openness)),
                     max_offset_frames=max_offset_frames)
    out.update(fps=round(fps, 3), openness=series(openness))
    if words:
        out["bilabial"] = lip_closure(openness, fps, bilabial_onsets(words), offset_frames=out["offset_frames"] or 0)
    return out


def score_in_venv(video: Path, audio: Path, *, max_offset_frames: int = 5, timeout: float = 600,
                  words: list[dict] | None = None) -> dict:
    """Run the score in the talk image's own mediapipe environment (MW_LIPSYNC_PYTHON): mediapipe's pinned
    numpy/protobuf would clash with InfiniteTalk's stack. Never raises: a failed score is recorded, not fatal.
    `words` (timed, {text, start_s, end_s}) go to the subprocess as a JSON file."""
    import json
    import subprocess
    python = os.environ.get("MW_LIPSYNC_PYTHON", "/opt/lipsync/bin/python")
    if not Path(python).exists():
        return {"error": f"no lip-sync environment at {python}"}
    try:
        args = [python, __file__, str(video), str(audio), str(max_offset_frames)]
        if words:
            listing = Path(video).with_suffix(".words.json")
            listing.write_text(json.dumps(words), encoding="utf-8")
            args.append(str(listing))
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return {"error": (proc.stderr or proc.stdout)[-500:]}
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001 - the clip stands without a score
        return {"error": f"{type(e).__name__}: {e}"[:500]}


if __name__ == "__main__":   # /opt/lipsync/bin/python lipsync.py VIDEO AUDIO [MAX_OFFSET [WORDS_JSON]]
    import json
    import sys
    timed = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8")) if len(sys.argv) > 4 else None
    print(json.dumps(score_clip(Path(sys.argv[1]), Path(sys.argv[2]),
                                max_offset_frames=int(sys.argv[3]) if len(sys.argv) > 3 else 5, words=timed)))
