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


def score_clip(video: Path, audio: Path, *, max_offset_frames: int = 5) -> dict:
    import soundfile as sf
    openness, fps = mouth_openness(video)
    samples, rate = sf.read(str(audio))
    return sync_score(openness, loudness(np.asarray(samples), rate, fps, len(openness)),
                      max_offset_frames=max_offset_frames)


def score_in_venv(video: Path, audio: Path, *, max_offset_frames: int = 5, timeout: float = 600) -> dict:
    """Run the score in the talk image's own mediapipe environment (MW_LIPSYNC_PYTHON): mediapipe's pinned
    numpy/protobuf would clash with InfiniteTalk's stack. Never raises: a failed score is recorded, not fatal."""
    import json
    import subprocess
    python = os.environ.get("MW_LIPSYNC_PYTHON", "/opt/lipsync/bin/python")
    if not Path(python).exists():
        return {"error": f"no lip-sync environment at {python}"}
    try:
        proc = subprocess.run([python, __file__, str(video), str(audio), str(max_offset_frames)],
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            return {"error": (proc.stderr or proc.stdout)[-500:]}
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001 - the clip stands without a score
        return {"error": f"{type(e).__name__}: {e}"[:500]}


if __name__ == "__main__":   # /opt/lipsync/bin/python lipsync.py VIDEO AUDIO [MAX_OFFSET]
    import json
    import sys
    print(json.dumps(score_clip(Path(sys.argv[1]), Path(sys.argv[2]),
                                max_offset_frames=int(sys.argv[3]) if len(sys.argv) > 3 else 5)))
