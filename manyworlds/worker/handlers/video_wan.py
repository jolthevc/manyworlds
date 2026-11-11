"""Story clips: Wan 2.2 image-to-video from a creator still.

Two backends, chosen with MW_VIDEO_BACKEND:

* `lightx2v` (target): LightX2V's NVFP4 + 4-step + sparse-attention build of Wan 2.2 14B,
  ~11 s per 5 s 480p clip on an RTX 5090 per its model card. It runs as a subprocess so
  the worker can use LightX2V's own container image. The command is a template in
  MW_LIGHTX2V_CMD because LightX2V's CLI changes between releases; confirm it against
  the image you deploy.
* `diffusers` (baseline): diffusers' WanImageToVideoPipeline with an optional 4-step
  distill LoRA. Slower and memory-hungry (CPU offload on 32 GB cards), but a known-good
  reference for checking LightX2V's quality.

Payload: image_uri, prompt, width, height, frames, seed, params{steps, guidance, fps}.
Not yet verified on a GPU.
"""

from __future__ import annotations

import os
import random
import shlex
import subprocess
from typing import Any

from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

BACKEND = os.environ.get("MW_VIDEO_BACKEND", "lightx2v")
MODEL_ID = os.environ.get("MW_VIDEO_MODEL", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
DISTILL_LORA = os.environ.get("MW_VIDEO_DISTILL_LORA", "")
LIGHTX2V_CMD = os.environ.get(
    "MW_LIGHTX2V_CMD",
    "python -m lightx2v.infer --model_cls wan2.2_moe_distill --task i2v "
    "--model_path {model_path} --config_json {config_json} --prompt {prompt} "
    "--image_path {image} --save_result_path {output} --seed {seed}",
)
_state: dict[str, Any] = {"pipe": None}


def _diffusers_pipe():
    if _state["pipe"] is None:
        import torch
        from diffusers import WanImageToVideoPipeline

        pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        if DISTILL_LORA:
            pipe.load_lora_weights(DISTILL_LORA, adapter_name="distill")
        pipe.enable_model_cpu_offload()
        _state["pipe"] = pipe
    return _state["pipe"]


def _run_diffusers(image_path, out_path, prompt, w, h, frames, steps, guidance, fps, seed):
    import torch
    from diffusers.utils import export_to_video, load_image

    pipe = _diffusers_pipe()
    frames_out = pipe(image=load_image(str(image_path)), prompt=prompt, height=h, width=w,
                      num_frames=frames, num_inference_steps=steps, guidance_scale=guidance,
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
    w, h = int(payload.get("width", 480)), int(payload.get("height", 832))
    frames = int(payload.get("frames", 81))
    steps, guidance = int(params.get("steps", 4)), float(params.get("guidance", 1.0))
    fps = int(params.get("fps", 16))
    seed = payload.get("seed")
    seed = int(seed) if seed is not None else random.randrange(1 << 31)

    image_path = ctx.fetch(payload["image_uri"], "start_frame.png")
    out_path = ctx.workdir / "clip.mp4"
    if BACKEND == "diffusers":
        _run_diffusers(image_path, out_path, payload["prompt"], w, h, frames, steps, guidance, fps, seed)
    elif BACKEND == "lightx2v":
        _run_lightx2v(image_path, out_path, payload["prompt"], seed)
    else:
        raise NonRetryable(f"unknown MW_VIDEO_BACKEND {BACKEND!r}")

    seconds = round(frames / fps, 2)
    recipe = {"backend": BACKEND, "model": MODEL_ID if BACKEND == "diffusers" else "LightWan2.2-A14B",
              "distill_lora": DISTILL_LORA or None, "prompt": payload["prompt"], "image_uri": payload["image_uri"],
              "seed": seed, "width": w, "height": h, "frames": frames, "steps": steps, "guidance": guidance, "fps": fps}
    return HandlerResult([OutputFile(out_path, "video", {"seconds": seconds, "width": w, "height": h})],
                         recipe=recipe, meta={"output_seconds": seconds})


HANDLERS = {"video.wan_i2v": generate}
