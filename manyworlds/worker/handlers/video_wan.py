"""Story clips: Wan 2.2 image-to-video from one approved creator photo.

Diffusers' WanImageToVideoPipeline with MW_VIDEO_MODEL, by default Wan-AI/Wan2.2-TI2V-5B-Diffusers
(Apache 2.0): 5B parameters, runs on one 24 GB card, 24 fps. Weights load from our bucket once
mirrored (manyworlds.worker.models). The whole pipeline stays on the GPU when
MW_VIDEO_OFFLOAD_BELOW_GB of memory is free, otherwise components are offloaded to CPU (slower,
but fits 24 GB cards).

Payload: image_uri, prompt, width, height, frames, seed, params{steps, guidance, fps}, keyframes.
Besides the clip, `keyframes` evenly spaced frames are stored as JPEG `keyframe` assets: the
video review gate (manyworlds.review.video) judges identity drift and motion from them.
Kind video.mirror copies the model into the bucket (see models.mirror).
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

from .. import models
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

MODEL_ID = os.environ.get("MW_VIDEO_MODEL", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
OFFLOAD_BELOW_GB = float(os.environ.get("MW_VIDEO_OFFLOAD_BELOW_GB", "40"))
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
            # 2026-11-14: 704x1280x121 on a 24 GB 3090 denoised fine, then ran out of memory in
            # the float32 VAE decode. Tiled decoding keeps it within the card.
            pipe.vae.enable_tiling()
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


def keyframe_indices(frames: int, n: int) -> list[int]:
    """n frames spread over the clip, ending on the last (where drift is worst); frame 0 is the
    approved start photo itself, so it is left out."""
    n = max(1, min(n, frames - 1))
    return sorted({round((i + 1) * (frames - 1) / n) for i in range(n)})


def extract_keyframes(video_path, workdir, n: int) -> list[OutputFile]:
    import imageio.v2 as iio
    from PIL import Image

    # The reader's length is unknown (inf) until counted: list(reader) raises MemoryError.
    reader = iio.get_reader(str(video_path), "ffmpeg")
    total = reader.count_frames()
    wanted = keyframe_indices(total, n)
    outs = []
    for idx, frame in enumerate(reader.iter_data()):
        if idx in wanted:
            p = workdir / f"keyframe_{len(outs)}.jpg"
            Image.fromarray(frame).convert("RGB").save(p, quality=90)
            outs.append(OutputFile(p, "keyframe", {"index": len(outs), "frame": idx, "of_frames": total}))
    reader.close()
    return outs


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
    _run_diffusers(ctx, image_path, out_path, payload["prompt"], w, h, frames, steps, guidance, fps, seed)
    render_s = round(time.monotonic() - t0, 1)

    keyframes = extract_keyframes(out_path, ctx.workdir, int(payload.get("keyframes", 4)))
    seconds = round(frames / fps, 2)
    recipe = {"model": MODEL_ID,
              "prompt": payload["prompt"], "image_uri": payload["image_uri"], "seed": seed, "width": w, "height": h,
              "frames": frames, "steps": steps, "guidance": guidance, "fps": fps}
    return HandlerResult([OutputFile(out_path, "video", {"seconds": seconds, "width": w, "height": h}), *keyframes],
                         recipe=recipe, meta={"output_seconds": seconds, "render_seconds": render_s,
                                              "cpu_offload": _state.get("offload"), "gpu": _state.get("gpu")})


def mirror_model(payload: dict, ctx: JobContext) -> HandlerResult:
    info = models.mirror(payload.get("repo") or MODEL_ID, ctx.store, ctx.workdir)
    return HandlerResult([], recipe={"mirror": info["repo"]}, meta=info)


HANDLERS = {"video.wan_i2v": generate, "video.mirror": mirror_model}
