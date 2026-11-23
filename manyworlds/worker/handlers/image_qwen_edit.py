"""Reference-conditioned scene photos: Qwen-Image-Edit-2511 with the 4-step Lightning LoRA.

Licences (checked 2026-11-22 on the official Hugging Face cards): Qwen/Qwen-Image-Edit-2511 is Apache-2.0 ("Qwen-Image
is licensed under Apache 2.0."); lightx2v/Qwen-Image-Edit-2511-Lightning is Apache-2.0.

The pipeline is diffusers' QwenImageEditPlusPipeline (the model's model_index.json). Its transformer config sets
`zero_cond_t`, which diffusers reads from 0.37.0 on (0.36 lacks it and would silently run the model wrong), so the
handler refuses an older diffusers. bf16 only: an fp8 base with the Lightning LoRA gave grid artifacts (research
2026-11-22, section 1). The Lightning sampler uses the scheduler config ModelTC distilled it with (shift 3,
generate_with_diffusers.py in ModelTC/Qwen-Image-Lightning), 4 steps, true CFG 1.0.

Payload: images [uri, ...] (1-3, in order: creator anchor, set plate, outfit reference), prompt, n, seed, width,
height, params{steps, true_cfg, negative_prompt, lightning}. Outputs n images. Weights are loaded once per process,
from our bucket when mirrored (manyworlds.worker.models), else Hugging Face (~58 GB: transformer ~41 GB, text
encoder ~17 GB). They fit an 80 GB card; below MW_EDIT_OFFLOAD_BELOW_GB free (an L40S's 48 GB) components are moved
to the GPU one at a time.

Not run on a GPU yet: `sets.reference_conditioning` stays off until `mw bench image.qwen_edit` passes the image gate.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from typing import Any

from .. import models
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

MODEL_ID = os.environ.get("MW_EDIT_MODEL", "Qwen/Qwen-Image-Edit-2511")
LORA_REPO = os.environ.get("MW_EDIT_LORA_REPO", "lightx2v/Qwen-Image-Edit-2511-Lightning")
LORA_FILE = os.environ.get("MW_EDIT_LORA_FILE", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
OFFLOAD_BELOW_GB = float(os.environ.get("MW_EDIT_OFFLOAD_BELOW_GB", "64"))
MIN_DIFFUSERS = "0.37.0"     # QwenImageTransformer2DModel(zero_cond_t), which 2511's config needs
MAX_IMAGES = 3               # anchor, set plate, outfit reference
SIZE_MULTIPLE = 16           # the VAE's 8x downsampling and the transformer's 2x2 patches

# The sampler the Lightning LoRA was distilled with (ModelTC/Qwen-Image-Lightning, generate_with_diffusers.py).
# Fixed by the model, not a tunable.
LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False, "max_image_seq_len": 8192,
    "max_shift": math.log(3), "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": None,
    "stochastic_sampling": False, "time_shift_type": "exponential", "use_beta_sigmas": False,
    "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False,
}

_state: dict[str, Any] = {"pipe": None, "offload": None, "fallback": False, "base_scheduler": None,
                          "lightning": None}
log = logging.getLogger("manyworlds.image_edit")


def settings(payload: dict) -> dict:
    """The job's settings, checked: what cannot work raises NonRetryable (a rerun would fail the same way)."""
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise NonRetryable("payload has no prompt")
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise NonRetryable("payload has no input images")
    if len(images) > MAX_IMAGES:
        raise NonRetryable(f"at most {MAX_IMAGES} input images, got {len(images)}")
    if not all(isinstance(u, str) and u.strip() for u in images):
        raise NonRetryable("every input image must be a uri")
    params = payload.get("params") or {}
    try:
        n = int(payload.get("n", 4))
        width, height = int(payload.get("width", 896)), int(payload.get("height", 1152))
        steps = int(params.get("steps", 4))
        true_cfg = float(params.get("true_cfg", 1.0))
        seed = payload.get("seed")
        seed = int(seed) if seed is not None else random.randrange(1 << 31)
    except (TypeError, ValueError) as e:
        raise NonRetryable(f"bad number in payload: {e}") from None
    if n < 1 or steps < 1 or true_cfg < 1.0:
        raise NonRetryable(f"n, steps must be >= 1 and true_cfg >= 1.0 (got {n}, {steps}, {true_cfg})")
    if width % SIZE_MULTIPLE or height % SIZE_MULTIPLE or width < 256 or height < 256:
        raise NonRetryable(f"width and height must be multiples of {SIZE_MULTIPLE}, at least 256 "
                           f"(got {width}x{height})")
    negative = params.get("negative_prompt", " ")
    return {"prompt": prompt, "images": list(images), "n": n, "width": width, "height": height, "steps": steps,
            "true_cfg": true_cfg, "seed": seed, "negative_prompt": negative if isinstance(negative, str) else " ",
            "lightning": bool(params.get("lightning", True))}


def check_diffusers() -> str:
    import diffusers
    from packaging.version import Version

    if Version(diffusers.__version__) < Version(MIN_DIFFUSERS):
        raise NonRetryable(f"diffusers {diffusers.__version__} is older than {MIN_DIFFUSERS}: Qwen-Image-Edit-2511 "
                           "needs zero_cond_t; rebuild the image worker")
    return diffusers.__version__


def lightning_scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)


def _pipe(ctx: JobContext):
    if _state["pipe"] is None:
        import torch
        from diffusers import QwenImageEditPlusPipeline

        version = check_diffusers()
        free, total = torch.cuda.mem_get_info()
        offload = free / 1e9 < OFFLOAD_BELOW_GB
        log.info("GPU %s: %.1f GB free of %.1f GB; cpu offload=%s; diffusers %s",
                 torch.cuda.get_device_name(0), free / 1e9, total / 1e9, offload, version)
        source = models.ensure(MODEL_ID, ctx.store, ctx.workdir)
        pipe = QwenImageEditPlusPipeline.from_pretrained(source, torch_dtype=torch.bfloat16)
        lora = models.ensure(LORA_REPO, ctx.store, ctx.workdir)
        pipe.load_lora_weights(lora, weight_name=LORA_FILE, adapter_name="lightning")
        if offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        _state.update(pipe=pipe, offload=offload, free_gb=round(free / 1e9, 1), gpu=torch.cuda.get_device_name(0),
                      base_scheduler=pipe.scheduler, lightning=None, diffusers=version)   # set per job
    return _state["pipe"]


def _use_lightning(pipe, on: bool) -> None:
    """The Lightning LoRA and its sampler on (4 steps), or off (the base model's own sampler, ~40 steps)."""
    if _state["lightning"] == on:
        return
    if on:
        pipe.enable_lora()
        pipe.scheduler = lightning_scheduler()
    else:
        pipe.disable_lora()
        pipe.scheduler = _state["base_scheduler"]
    _state["lightning"] = on


def _fall_back(pipe) -> None:
    import torch

    log.warning("out of GPU memory with the full edit model loaded; switching to cpu offload")
    pipe.to("cpu")
    torch.cuda.empty_cache()
    pipe.enable_model_cpu_offload()
    _state.update(offload=True, fallback=True)


def generate(payload: dict, ctx: JobContext) -> HandlerResult:
    s = settings(payload)
    import torch
    from PIL import Image

    inputs = []
    for i, uri in enumerate(s["images"]):
        with Image.open(ctx.fetch(uri, f"edit_in_{i}.png")) as im:
            inputs.append(im.convert("RGB"))
    pipe = _pipe(ctx)
    _use_lightning(pipe, s["lightning"])
    outs, timings = [], []
    for i in range(s["n"]):
        def render():
            with torch.inference_mode():
                return pipe(image=inputs, prompt=s["prompt"], negative_prompt=s["negative_prompt"],
                            true_cfg_scale=s["true_cfg"], guidance_scale=1.0, num_inference_steps=s["steps"],
                            width=s["width"], height=s["height"], num_images_per_prompt=1,
                            generator=torch.Generator("cuda").manual_seed(s["seed"] + i)).images[0]
        t0 = time.monotonic()
        try:
            img = render()
        except torch.cuda.OutOfMemoryError:
            if _state["offload"]:
                raise
            _fall_back(pipe)
            t0 = time.monotonic()
            img = render()
        timings.append(round(time.monotonic() - t0, 2))
        path = ctx.workdir / f"edit_{i}.png"
        img.save(path)
        outs.append(OutputFile(path, "image", {"width": img.width, "height": img.height, "seed": s["seed"] + i}))
    recipe = {"model": MODEL_ID, "lora": f"{LORA_REPO}/{LORA_FILE}" if s["lightning"] else None,
              "prompt": s["prompt"], "images": s["images"], "seed": s["seed"], "steps": s["steps"],
              "true_cfg": s["true_cfg"], "negative_prompt": s["negative_prompt"], "width": s["width"],
              "height": s["height"]}
    return HandlerResult(outs, recipe=recipe, meta={
        "images": len(outs), "sec_per_image": timings, "cpu_offload": _state["offload"],
        "offload_fallback": _state["fallback"], "gpu": _state.get("gpu"), "free_gb_at_load": _state.get("free_gb"),
        "diffusers": _state.get("diffusers")})


HANDLERS = {"image.qwen_edit": generate}


def release() -> None:
    """Give the GPU back before another family's job (manyworlds.worker.runner._switch_to)."""
    if _state.get("pipe") is None:
        return
    _state["pipe"] = None
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - no torch or no GPU: nothing to free
        pass
