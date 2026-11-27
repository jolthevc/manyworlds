"""Story clips (job kind comfy.wan_i2v) on ComfyUI + WanVideoWrapper: Wan 2.2 I2V A14B with the lightx2v LoRA
(manyworlds.worker.comfy.graph_i2v), on the same worker image as talking video.

The story clip job: one approved photo animated into a short clip, with keyframes for the video
gate. Self-hosted FastWan (5B, DMD) rendered in ~1 minute but passed 0 of 10 clips (2026-11-22); this is the
community's proven 14B graph, at hourly GPU prices instead of a hosted API's per-clip price.

Payload: image_uri, prompt, negative_prompt (the shot's negatives, manyworlds.direction.compile_prompts; else
comfy.negative_text's fallbacks), keyframes, seed; params{width, height, frames, fps, steps, high_steps, shift,
high_lora_strength, low_lora_strength, high_cfg, swap_below_gb, blocks_to_swap, timeout_s, fit_stretch_tolerance}
(a photo at the frame's aspect is scaled whole, not cropped: talk_comfy.fit_box).
"""

from __future__ import annotations

import random
import shutil
import time
from pathlib import Path
from typing import Any

from .. import comfy, crops
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile
from .talk_comfy import _blocks_to_swap, _models_dir, _prepare_image
from ..keyframes import extract_keyframes

REQUIRED = ("width", "height", "timeout_s")
_ready: dict = {"weights": False}


def generate(payload: dict, ctx: JobContext) -> HandlerResult:
    if not payload.get("image_uri") or not payload.get("prompt"):
        raise NonRetryable("payload needs image_uri and prompt")
    params: dict[str, Any] = payload.get("params") or {}
    missing = [k for k in REQUIRED if params.get(k) is None]
    if missing:
        raise NonRetryable(f"params missing {missing}")
    seed = payload.get("seed")
    seed = int(seed) if seed is not None else random.randrange(1 << 31)
    work = Path(ctx.workdir)

    models = _models_dir()
    if not _ready["weights"]:
        comfy.write_model_paths(models)
        comfy.ensure_weights(models, comfy.I2V_WEIGHTS, prune=True)
        _ready["weights"] = True
    comfy.start()

    start = ctx.fetch(payload["image_uri"], "start_frame.png")
    inp = comfy.COMFY_DIR / "input"
    inp.mkdir(parents=True, exist_ok=True)
    image_name = f"mw_{ctx.job.id}.png"
    _prepare_image(start, inp / image_name, int(params["width"]), int(params["height"]),
                   stretch_tolerance=float(params.get("fit_stretch_tolerance") or 0))   # a 9:16 photo is not cropped
    g = comfy.graph_i2v(image=image_name, prompt=payload["prompt"], seed=seed, params=params,
                        blocks_to_swap=_blocks_to_swap(params), negative=payload.get("negative_prompt"))
    t0 = time.monotonic()
    out = comfy.run(g, timeout_s=float(params["timeout_s"]))
    render_s = round(time.monotonic() - t0, 1)
    clip = work / "clip.mp4"
    shutil.move(str(out), clip)
    (inp / image_name).unlink(missing_ok=True)

    seconds = round(int(params.get("frames", 81)) / float(params.get("fps", 16)), 2)
    keyframes = extract_keyframes(clip, work, int(payload.get("keyframes", 4)))
    keyframes += crops.crops_in_venv(keyframes, work, payload.get("crops"))
    recipe = {"model": [comfy.HIGH, comfy.LOW], "lora": comfy.LORA, "engine": "comfy+WanVideoWrapper",
              "prompt": payload["prompt"], "negative_prompt": comfy.negative_text(params, payload.get("negative_prompt")),
              "image_uri": payload["image_uri"], "seed": seed, "params": params}
    return HandlerResult([OutputFile(clip, "video", {"seconds": seconds, "width": params["width"],
                                                     "height": params["height"], "engine": "comfy", "render": comfy.last_run()}), *keyframes],
                         recipe=recipe, meta={"output_seconds": seconds, "render_seconds": render_s})


HANDLERS = {"comfy.wan_i2v": generate}

def release() -> None:
    """Give the GPU back before another family's job (manyworlds.worker.runner._switch_to)."""
    from .. import comfy
    comfy.free()
