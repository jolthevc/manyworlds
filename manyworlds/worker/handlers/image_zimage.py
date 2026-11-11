"""Photo candidates: Z-Image Turbo (Apache 2.0) plus an optional per-creator LoRA.

Runs on RTX 4090/5090 workers. The pipeline loads once per process; the LoRA is swapped
only when the creator changes, which is why the scheduler batches jobs by creator.
Payload: prompt, n, width, height, seed, lora_uri, lora_scale, params{steps, guidance}.

Not yet verified on a GPU: run `mw bench` on day 1 and check speed and identity.
The 4-bit Nunchaku variant is a follow-up once the BF16 baseline is measured.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

MODEL_ID = os.environ.get("MW_IMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
# Consumer cards on SaladCloud run Windows/WSL: the desktop holds VRAM, and when a model
# doesn't fit, WSL spills to shared system memory and runs ~10x slower instead of failing.
# Below this much free VRAM we let diffusers move components on and off the GPU.
OFFLOAD_BELOW_GB = float(os.environ.get("MW_OFFLOAD_BELOW_GB", "26"))
_state: dict[str, Any] = {"pipe": None, "lora_uri": None, "offload": None}
log = logging.getLogger("manyworlds.image")


def _pipe():
    if _state["pipe"] is None:
        import torch
        from diffusers import DiffusionPipeline

        free, total = torch.cuda.mem_get_info()
        offload = free / 1e9 < OFFLOAD_BELOW_GB
        log.info("GPU %s: %.1f GB free of %.1f GB; cpu offload=%s",
                 torch.cuda.get_device_name(0), free / 1e9, total / 1e9, offload)
        pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        if offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        _state.update(pipe=pipe, offload=offload, free_gb=round(free / 1e9, 1),
                      gpu=torch.cuda.get_device_name(0))
    return _state["pipe"]


def _set_lora(pipe, ctx: JobContext, lora_uri: str | None, scale: float) -> None:
    if lora_uri == _state["lora_uri"]:
        if lora_uri:
            pipe.set_adapters(["creator"], adapter_weights=[scale])
        return
    if _state["lora_uri"]:
        pipe.unload_lora_weights()
        _state["lora_uri"] = None
    if lora_uri:
        path = ctx.fetch(lora_uri, "creator.safetensors")
        pipe.load_lora_weights(str(path), adapter_name="creator")
        pipe.set_adapters(["creator"], adapter_weights=[scale])
        _state["lora_uri"] = lora_uri


def generate(payload: dict, ctx: JobContext) -> HandlerResult:
    import torch

    prompt = payload.get("prompt")
    if not prompt:
        raise NonRetryable("payload has no prompt")
    params = payload.get("params") or {}
    n = int(payload.get("n", 4))
    width, height = int(payload.get("width", 896)), int(payload.get("height", 1152))
    steps = int(params.get("steps", 9))
    guidance = float(params.get("guidance", 0.0))
    seed = payload.get("seed")
    seed = int(seed) if seed is not None else random.randrange(1 << 31)

    pipe = _pipe()
    _set_lora(pipe, ctx, payload.get("lora_uri"), float(payload.get("lora_scale", 1.0)))
    # One image at a time: batching multiplies activation memory, which is exactly what
    # 24 GB consumer cards don't have to spare. Seeds stay seed+i, so outputs are unchanged.
    outs, timings = [], []
    for i in range(n):
        t0 = time.monotonic()
        img = pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps,
                   guidance_scale=guidance, generator=torch.Generator("cuda").manual_seed(seed + i)).images[0]
        timings.append(round(time.monotonic() - t0, 2))
        path = ctx.workdir / f"img_{i}.png"
        img.save(path)
        outs.append(OutputFile(path, "image", {"width": width, "height": height, "seed": seed + i}))
    recipe = {"model": MODEL_ID, "prompt": prompt, "seed": seed, "steps": steps, "guidance": guidance,
              "width": width, "height": height, "lora_uri": payload.get("lora_uri"),
              "lora_scale": payload.get("lora_scale", 1.0)}
    return HandlerResult(outs, recipe=recipe, meta={
        "images": len(outs), "sec_per_image": timings, "cpu_offload": _state["offload"],
        "gpu": _state.get("gpu"), "free_gb_at_load": _state.get("free_gb")})


HANDLERS = {"image.zimage": generate}
