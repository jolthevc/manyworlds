"""Photo candidates: Z-Image Turbo (Apache 2.0) plus an optional per-creator LoRA.

Runs on RTX 4090/5090 workers. The pipeline loads once per process; the LoRA is swapped
only when the creator changes, which is why the scheduler batches jobs by creator.
Payload: prompt, n, width, height, seed, lora_uri, lora_scale, negative_prompt (used only with guidance on),
params{steps, guidance}.

Not yet verified on a GPU: run `mw bench` on day 1 and check speed and identity.
The 4-bit Nunchaku variant is a follow-up once the BF16 baseline is measured.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

from .. import models
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

MODEL_ID = os.environ.get("MW_IMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
# Below this much free VRAM, diffusers moves components on and off the GPU for every image
# (slow: speed then depends on the host's memory bus, not the GPU). The autoscaler sets it
# per pool from policy (worker_images.image.env, gpu_pools.<pool>.worker_env). SaladCloud's
# Windows/WSL nodes need a high value: when a model doesn't fit there, WSL spills to shared
# system memory and runs ~10x slower instead of failing. On Linux hosts an over-full GPU
# raises out-of-memory instead, and the handler falls back to offload (see _fall_back).
# The default only applies to workers started by hand; it is the safe setting.
OFFLOAD_BELOW_GB = float(os.environ.get("MW_OFFLOAD_BELOW_GB", "26"))
_state: dict[str, Any] = {"pipe": None, "lora_uri": None, "offload": None, "fallback": False, "control": None}
log = logging.getLogger("manyworlds.image")


def control_spec(payload: dict) -> dict | None:
    """The payload's structural control (manyworlds.layout.control_payload), checked; None for a plain photo."""
    c = payload.get("control")
    if not c:
        return None
    if not isinstance(c, dict) or c.get("kind") != "pose" or not c.get("uri") or not c.get("repo") or not c.get("file"):
        raise NonRetryable(f"control needs kind 'pose', uri, repo and file: {c!r}")
    return {"uri": str(c["uri"]), "repo": str(c["repo"]), "file": str(c["file"]), "scale": float(c.get("scale", 0.75))}


def _controlnet_file(repo: str, filename: str) -> str:
    """One file of the ControlNet repo, kept with the machine's models (the repo is 53 GB of variants; only the
    file named is fetched)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, local_dir=str(models.local_dir(repo)),
                           token=os.environ.get("HF_TOKEN") or None)


def _pipe(ctx: JobContext, control: dict | None = None):
    """The pipeline for this job: plain Z-Image, or Z-Image with the ControlNet named by `control`. Switching
    between the two reloads (from the machine's disk): a test batch is all one or the other."""
    want = (control["repo"], control["file"]) if control else None
    if _state["pipe"] is not None and _state["control"] != want:
        release()
    if _state["pipe"] is None:
        import torch
        from diffusers import DiffusionPipeline

        free, total = torch.cuda.mem_get_info()
        offload = free / 1e9 < OFFLOAD_BELOW_GB
        log.info("GPU %s: %.1f GB free of %.1f GB; cpu offload=%s",
                 torch.cuda.get_device_name(0), free / 1e9, total / 1e9, offload)
        # Our bucket's copy when it exists (manyworlds.worker.models), else Hugging Face.
        source = models.ensure(MODEL_ID, ctx.store, ctx.workdir)
        if want:
            from diffusers import ZImageControlNetModel, ZImageControlNetPipeline

            net = ZImageControlNetModel.from_single_file(_controlnet_file(*want), torch_dtype=torch.bfloat16)
            pipe = ZImageControlNetPipeline.from_pretrained(source, controlnet=net, torch_dtype=torch.bfloat16)
        else:
            pipe = DiffusionPipeline.from_pretrained(source, torch_dtype=torch.bfloat16)
        if offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        _state.update(pipe=pipe, offload=offload, free_gb=round(free / 1e9, 1), control=want,
                      gpu=torch.cuda.get_device_name(0))
    return _state["pipe"]


def _fall_back(pipe) -> None:
    """The whole model didn't fit: keep this machine useful by offloading from now on."""
    import torch

    log.warning("out of GPU memory with the full model loaded; switching to cpu offload")
    pipe.to("cpu")
    torch.cuda.empty_cache()
    pipe.enable_model_cpu_offload()
    _state.update(offload=True, fallback=True)


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


def negative_kwargs(payload: dict, guidance: float) -> dict:
    """The negative prompt, only where the pipeline reads one: with guidance on (Z-Image Turbo ships at guidance 0,
    where there is no negative branch and a negative prompt does nothing). A directed photo carries none at all
    (manyworlds.direction: photo has no negative channel, so its negatives are phrased positively in the prompt)."""
    negative = payload.get("negative_prompt")
    if guidance > 0 and isinstance(negative, str) and negative.strip():
        return {"negative_prompt": negative.strip()}
    return {}


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
    negative = negative_kwargs(payload, guidance)
    seed = payload.get("seed")
    seed = int(seed) if seed is not None else random.randrange(1 << 31)

    control = control_spec(payload)
    pipe = _pipe(ctx, control)
    _set_lora(pipe, ctx, payload.get("lora_uri"), float(payload.get("lora_scale", 1.0)))
    extra: dict[str, Any] = {}
    if control:
        from PIL import Image

        with Image.open(ctx.fetch(control["uri"], "control.png")) as im:
            extra = {"control_image": im.convert("RGB").resize((width, height)),
                     "controlnet_conditioning_scale": control["scale"]}
    # One image at a time: batching multiplies activation memory, which is exactly what
    # 24 GB consumer cards don't have to spare. Seeds stay seed+i, so outputs are unchanged.
    outs, timings = [], []
    for i in range(n):
        t0 = time.monotonic()
        def render():
            return pipe(prompt=prompt, height=height, width=width, num_inference_steps=steps,
                        guidance_scale=guidance, generator=torch.Generator("cuda").manual_seed(seed + i),
                        **negative, **extra).images[0]
        try:
            img = render()
        except torch.cuda.OutOfMemoryError:
            if _state["offload"]:
                raise
            _fall_back(pipe)
            t0 = time.monotonic()
            img = render()
        timings.append(round(time.monotonic() - t0, 2))
        path = ctx.workdir / f"img_{i}.png"
        img.save(path)
        outs.append(OutputFile(path, "image", {"width": width, "height": height, "seed": seed + i}))
    recipe = {"model": MODEL_ID, "prompt": prompt, "seed": seed, "steps": steps, "guidance": guidance,
              "width": width, "height": height, "lora_uri": payload.get("lora_uri"),
              "negative_prompt": negative.get("negative_prompt"),
              "lora_scale": payload.get("lora_scale", 1.0),
              "control": ({"file": control["file"], "scale": control["scale"], "uri": control["uri"]}
                          if control else None)}
    return HandlerResult(outs, recipe=recipe, meta={
        "images": len(outs), "sec_per_image": timings, "cpu_offload": _state["offload"],
        "offload_fallback": _state["fallback"],
        "gpu": _state.get("gpu"), "free_gb_at_load": _state.get("free_gb")})


def mirror_model(payload: dict, ctx: JobContext) -> HandlerResult:
    """Copy a model repo (default: this handler's model) into our bucket, once."""
    info = models.mirror(payload.get("repo") or MODEL_ID, ctx.store, ctx.workdir)
    return HandlerResult([], recipe={"mirror": info["repo"]}, meta=info)


HANDLERS = {"image.zimage": generate, "image.mirror": mirror_model}


def release() -> None:
    """Give the GPU back before another family's job (manyworlds.worker.runner._switch_to)."""
    if _state.get("pipe") is None:
        return
    _state["pipe"], _state["lora_uri"], _state["control"] = None, None, None   # a reload starts without the LoRA
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - no torch or no GPU: nothing to free
        pass
