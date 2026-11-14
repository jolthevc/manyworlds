"""Checks a worker image before it is pushed, with no GPU and nothing rented.

    python -m manyworlds.worker.smoke image      # run inside the built image

Handlers import their heavy libraries lazily, so a missing package only shows up on a rented
machine, after its boot and model download. Between 2026-11-12 and 2026-11-14 that cost about
$1.46 of machines: missing ftfy and imageio, a keyframe reader that ran out of memory, and more.
This imports what each kind needs and runs the CPU parts for real. scripts/build-worker.sh
pushes an image only when it passes.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
from pathlib import Path


def _imports(names: list[str]) -> None:
    for name in names:
        importlib.import_module(name)


def check_image() -> list[str]:
    done = []
    _imports(["torch", "diffusers", "transformers", "accelerate", "peft", "safetensors", "PIL", "soundfile",
              "kokoro", "ftfy", "imageio", "imageio_ffmpeg"])
    from diffusers import AutoencoderKLWan, DiffusionPipeline, WanImageToVideoPipeline  # noqa: F401
    done.append("imports")

    from .runner import handlers_for
    from .. import policy
    kinds = [k for k in policy.load().get("job_kinds", default={}) if k.split(".")[0] in ("image", "video")]
    handlers_for(kinds)
    done.append(f"handlers: {', '.join(kinds)}")

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg missing (narrated edits need it)")
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean   # uses ftfy, as the Wan pipeline does
    prompt_clean("a woman pours tea &amp; smiles")
    done.append("ffmpeg, prompt cleaning")

    import numpy as np
    from diffusers.utils import export_to_video
    from .handlers.video_wan import extract_keyframes, keyframe_indices
    d = Path(tempfile.mkdtemp())
    frames = [np.full((64, 48, 3), i / 121, dtype=np.float32) for i in range(121)]
    export_to_video(frames, str(d / "clip.mp4"), fps=24)
    outs = extract_keyframes(d / "clip.mp4", d, 4)
    if [o.meta["frame"] for o in outs] != keyframe_indices(121, 4):
        raise RuntimeError(f"keyframes came out wrong: {[o.meta for o in outs]}")
    done.append("clip export and keyframes")
    return done


def check_train() -> list[str]:
    _imports(["torch", "yaml"])
    from .runner import handlers_for
    handlers_for(["train.lora"])
    return ["imports", "handlers: train.lora"]


CHECKS = {"image": check_image, "train": check_train}


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    family = args[0] if args else "image"
    try:
        for step in CHECKS[family]():
            print(f"ok  {step}")
    except Exception as e:  # noqa: BLE001 - report any failure and exit non-zero
        print(f"FAIL {type(e).__name__}: {e}")
        return 1
    print(f"smoke test passed for the {family} worker image")
    return 0


if __name__ == "__main__":
    sys.exit(main())
