"""Story clips: Wan 2.2 image-to-video from one approved creator photo.

Backends, chosen with MW_VIDEO_BACKEND:

* `diffusers` (default): diffusers' WanImageToVideoPipeline with MW_VIDEO_MODEL, by default
  Wan-AI/Wan2.2-TI2V-5B-Diffusers (Apache 2.0): 5B parameters, runs on one 24 GB card, 24 fps.
  Weights load from our bucket once mirrored (manyworlds.worker.models). The whole pipeline
  stays on the GPU when MW_VIDEO_OFFLOAD_BELOW_GB of memory is free, otherwise components are
  offloaded to CPU (slower, but fits 24 GB cards).
* `lightx2v`: LightX2V's distilled Wan 2.2 14B as a subprocess (MW_LIGHTX2V_CMD template).
  Kept for a later quality/speed comparison; not verified.

Payload: image_uri, prompt, width, height, frames, seed, params{steps, guidance, fps}.
Kind video.mirror copies the model into the bucket (see models.mirror).
"""

from __future__ import annotations

import logging
import os
import random
import shlex
import subprocess
import time
from typing import Any

from .. import models
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

BACKEND = os.environ.get("MW_VIDEO_BACKEND", "diffusers")
MODEL_ID = os.environ.get("MW_VIDEO_MODEL", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
OFFLOAD_BELOW_GB = float(os.environ.get("MW_VIDEO_OFFLOAD_BELOW_GB", "40"))
LIGHTX2V_CMD = os.environ.get(
    "MW_LIGHTX2V_CMD",
    "python -m lightx2v.infer --model_cls wan2.2_moe_distill --task i2v "
    "--model_path {model_path} --config_json {config_json} --prompt {prompt} "
    "--image_path {image} --save_result_path {output} --seed {seed}",
)
_state: dict[str, Any] = {"pipe": None}
log = logging.getLogger("manyworlds.video")


def _diffusers_pipe(ctx: JobContext):
    if _state["pipe"] is None:
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

        source = models.ensure(MODEL_ID, ctx.store, ctx.workdir)
        free, total = torch.cuda.mem_get_info()
        offload = free / 1e9 < OFFLOAD_BELOW_GB
        log.info("GPU %s: %.1f GB free of %.1f GB; cpu offload=%s", torch.cuda.get_device_name(0),
                 free / 1e9, total / 1e9, offload)
        # The model card loads the VAE in float32 for quality.
        vae = AutoencoderKLWan.from_pretrained(source, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained(source, vae=vae, torch_dtype=torch.bfloat16)
        if offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        _state.update(pipe=pipe, offload=offload, gpu=torch.cuda.get_device_name(0), free_gb=round(free / 1e9, 1))
    return _state["pipe"]


def _run_diffusers(ctx, image_path, out_path, prompt, w, h, frames, steps, guidance, fps, seed):
    import torch
    from diffusers.utils import export_to_video, load_image

    pipe = _diffusers_pipe(ctx)
    image = load_image(str(image_path)).resize((w, h))
    frames_out = pipe(image=image, prompt=prompt, height=h, width=w, num_frames=frames,
                      num_inference_steps=steps, guidance_scale=guidance,
                      generator=torch.Generator("cpu").manual_seed(seed)).frames[0]
    export_to_video(frames_out, str(out_path), fps=fps)


def _run_lightx2v(image_path, out_path, prompt, seed):
    cmd = LIGHTX2V_CMD.format(
        model_path=shlex.quote(os.environ.get("MW_LIGHTX2V_MODEL_PATH", "/models/LightWan2.2-A14B")),
        config_json=shlex.quote(os.environ.get("MW_LIGHTX2V_CONFIG", "/configs/wan22_i2v_nvfp4_4step.json")),
        prompt=shlex.quote(prompt), image=shlex.quote(str(image_path)),
        output=shlex.quote(str(out_path)), seed=seed,
    )
    proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(f"lightx2v exited {proc.returncode}: {proc.stderr[-2000:]}")


def generate(payload: dict, ctx: JobContext) -> HandlerResult:
    if not payload.get("image_uri") or not payload.get("prompt"):
        raise NonRetryable("payload needs image_uri and prompt")
    params = payload.get("params") or {}
    w, h = int(payload.get("width", 704)), int(payload.get("height", 1280))
    frames = int(payload.get("frames", 121))
    steps, guidance = int(params.get("steps", 30)), float(params.get("guidance", 5.0))
    fps = int(params.get("fps", 24))
    seed = payload.get("seed")
    seed = int(seed) if seed is not None else random.randrange(1 << 31)

    image_path = ctx.fetch(payload["image_uri"], "start_frame.png")
    out_path = ctx.workdir / "clip.mp4"
    t0 = time.monotonic()
    if BACKEND == "diffusers":
        _run_diffusers(ctx, image_path, out_path, payload["prompt"], w, h, frames, steps, guidance, fps, seed)
    elif BACKEND == "lightx2v":
        _run_lightx2v(image_path, out_path, payload["prompt"], seed)
    else:
        raise NonRetryable(f"unknown MW_VIDEO_BACKEND {BACKEND!r}")
    render_s = round(time.monotonic() - t0, 1)

    seconds = round(frames / fps, 2)
    recipe = {"backend": BACKEND, "model": MODEL_ID if BACKEND == "diffusers" else "LightWan2.2-A14B",
              "prompt": payload["prompt"], "image_uri": payload["image_uri"], "seed": seed, "width": w, "height": h,
              "frames": frames, "steps": steps, "guidance": guidance, "fps": fps}
    return HandlerResult([OutputFile(out_path, "video", {"seconds": seconds, "width": w, "height": h})],
                         recipe=recipe, meta={"output_seconds": seconds, "render_seconds": render_s,
                                              "cpu_offload": _state.get("offload"), "gpu": _state.get("gpu")})


def mirror_model(payload: dict, ctx: JobContext) -> HandlerResult:
    info = models.mirror(payload.get("repo") or MODEL_ID, ctx.store, ctx.workdir)
    return HandlerResult([], recipe={"mirror": info["repo"]}, meta=info)


HANDLERS = {"video.wan_i2v": generate, "video.mirror": mirror_model}
