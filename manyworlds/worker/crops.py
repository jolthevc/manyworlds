"""Close-up crops of hands and the mouth from a clip's keyframes, for the judge (policy `review.detail.crops`).

Hands and teeth are where generated video breaks first, and on a full 480x832 keyframe they are a few dozen pixels.
The talk worker already has MediaPipe (Apache-2.0 code and models) in its own environment for the lip-sync score
(worker/lipsync.py, MW_LIPSYNC_PYTHON), so the crops are computed there too: the face landmarker's lip points and
the hand landmarker's 21 points per hand give a box, squared and padded, cut from the keyframe and scaled to
`size` px. Each crop is an output of kind `crop` with meta {of: keyframe index, part: hands|mouth}. No crops (no
MediaPipe environment, no hand model, nothing found) is normal: the judge then scores the full keyframes as before.

`square_box` is pure Python (tested in CI); `crop_frames` needs mediapipe and Pillow (the lip-sync environment).
This file runs as a script in that environment, so it imports nothing from the package.
"""

from __future__ import annotations

import os
from pathlib import Path

FACE_LANDMARKER = os.environ.get("MW_FACE_LANDMARKER", "/opt/models/face_landmarker.task")
HAND_LANDMARKER = os.environ.get("MW_HAND_LANDMARKER", "/opt/models/hand_landmarker.task")
# Face-mesh lip landmarks: outer and inner contour (corners 61/291, top 0, bottom 17, inner 13/14/78/308).
LIPS = (0, 17, 13, 14, 37, 267, 39, 269, 40, 270, 61, 291, 78, 308, 84, 314, 91, 321, 146, 375, 181, 405)
PARTS = ("hands", "mouth")


def square_box(points, width: int, height: int, *, margin: float = 0.5, min_side: int = 32):
    """A square pixel box (x0, y0, x1, y1) around normalised points [(x, y)], padded by `margin` x the larger
    side on each side, at least `min_side` px, moved (not shrunk) to stay inside the frame where it fits. None
    when there are no points in the frame."""
    pts = [(float(x), float(y)) for x, y in points if 0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0]
    if not pts or width <= 0 or height <= 0:
        return None
    xs, ys = [x * width for x, _ in pts], [y * height for _, y in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys))
    side = min(max(side * (1 + 2 * margin), float(min_side)), float(min(width, height)))
    x0 = min(max(cx - side / 2, 0.0), width - side)
    y0 = min(max(cy - side / 2, 0.0), height - side)
    return int(round(x0)), int(round(y0)), int(round(x0 + side)), int(round(y0 + side))


def crop_frames(frames: list[tuple[int, Path]], out_dir: Path, *, hands: bool = True, mouth: bool = True,
                size: int = 384, margin: float = 0.5, max_hands: int = 2) -> list[dict]:
    """Crops of each keyframe (index, path): [{path, of, part}]. Needs mediapipe; a part whose model file is
    missing is skipped."""
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import BaseOptions, vision
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    face = hand = None
    if mouth and Path(FACE_LANDMARKER).exists():
        face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_LANDMARKER), running_mode=vision.RunningMode.IMAGE,
            num_faces=1))
    if hands and Path(HAND_LANDMARKER).exists():
        hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_LANDMARKER), running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands))
    out = []
    try:
        for of, path in frames:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(img)))
            boxes = []
            if face is not None:
                found = face.detect(image).face_landmarks
                if found:
                    boxes.append(("mouth", square_box([(found[0][i].x, found[0][i].y) for i in LIPS], w, h,
                                                      margin=margin)))
            if hand is not None:
                for pts in (hand.detect(image).hand_landmarks or [])[:max_hands]:
                    boxes.append(("hands", square_box([(p.x, p.y) for p in pts], w, h, margin=margin)))
            n = {}
            for part, box in boxes:
                if box is None:
                    continue
                n[part] = n.get(part, 0) + 1
                p = out_dir / f"crop_{int(of)}_{part}_{n[part]}.png"
                img.crop(box).resize((int(size), int(size)), Image.LANCZOS).save(p)
                out.append({"path": str(p), "of": int(of), "part": part, "box": list(box)})
    finally:
        for m in (face, hand):
            if m is not None:
                m.close()
    return out


def options(raw: dict | None) -> dict | None:
    """The crop options a job asks for (payload `crops`: {enabled, hands, mouth, size}), or None when off."""
    if not isinstance(raw, dict) or not raw.get("enabled") or not (raw.get("hands") or raw.get("mouth")):
        return None
    return {"hands": bool(raw.get("hands")), "mouth": bool(raw.get("mouth")), "size": int(raw.get("size", 384)),
            "margin": float(raw.get("margin", 0.5))}


def run_script(spec: Path, *, timeout: float = 300) -> list[dict]:
    """Run this file on a spec in the lip-sync environment (MW_LIPSYNC_PYTHON); its JSON list. Raises on failure
    (the smoke test wants to know; crops_in_venv does not)."""
    import json
    import subprocess
    python = os.environ.get("MW_LIPSYNC_PYTHON", "/opt/lipsync/bin/python")
    if not Path(python).exists():
        raise RuntimeError(f"no lip-sync environment at {python}")
    proc = subprocess.run([python, __file__, str(spec)], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"crops failed: {(proc.stderr or proc.stdout)[-500:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def write_spec(frames: list[tuple[int, str]], workdir: Path, opts: dict) -> Path:
    import json
    spec = Path(workdir) / "crops.json"
    spec.write_text(json.dumps({"frames": frames, "out_dir": str(Path(workdir) / "crops"), **opts}), encoding="utf-8")
    return spec


def crops_in_venv(keyframes, workdir: Path, raw: dict | None, *, timeout: float = 300, run=None) -> list:
    """Crops of the keyframe outputs (OutputFile kind keyframe, meta.index) as OutputFile kind `crop`, computed
    in the lip-sync environment (run_script). Never raises: no crops is a normal result. `run` stands in for
    run_script in tests: run(spec_path) -> the script's JSON list."""
    from .runner import OutputFile
    opts = options(raw)
    if opts is None or not keyframes:
        return []
    try:
        frames = [(int((k.meta or {}).get("index", i)), str(k.path)) for i, k in enumerate(keyframes)]
        spec = write_spec(frames, workdir, opts)
        found = run(spec) if run is not None else run_script(spec, timeout=timeout)
        return [OutputFile(Path(c["path"]), "crop", {"of": int(c["of"]), "part": c["part"], "box": c.get("box")})
                for c in found if c.get("part") in PARTS and Path(c["path"]).exists()]
    except Exception:  # noqa: BLE001 - the clip stands without crops
        return []


if __name__ == "__main__":   # /opt/lipsync/bin/python crops.py SPEC_JSON
    import json
    import sys
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(json.dumps(crop_frames([(int(i), Path(p)) for i, p in spec["frames"]], Path(spec["out_dir"]),
                                 hands=bool(spec.get("hands", True)), mouth=bool(spec.get("mouth", True)),
                                 size=int(spec.get("size", 384)), margin=float(spec.get("margin", 0.5)))))
