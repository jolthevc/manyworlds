"""Lip-synced talking video (job kind talk.infinitetalk) on ComfyUI + WanVideoWrapper (manyworlds.worker.comfy).

Same job as handlers/talk_infinitetalk.py (approved photo + voice line or script, keyframes for the video gate, the
lip-sync score, the spoken line as its own output), rendered by the community's optimised stack: fp8-scaled
InfiniteTalk with the 4-step lightx2v LoRA and SageAttention. The talk worker image chooses this engine
(MW_TALK_ENGINE=comfy). Weights download once into MW_MODEL_DIR/comfy (the pool's network volume if it has one).

Payload as for talk_infinitetalk; params{model, width, height, steps, shift, cfg, scheduler, audio_scale, audio_cfg,
frame_window, motion_frame, overlap, longcat_version, lora_strength, attention, swap_below_gb, blocks_to_swap,
timeout_s}. params.model picks the graph: infinitetalk (the default) or longcat (comfy.graph_longcat, LongCat-Video-
Avatar; its weights download only when a LongCat job runs). LongCat saves one clip per window; they are joined
here, trimmed to the voice line and muxed with it.
"""

from __future__ import annotations

import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

from .. import comfy, lipsync
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile
from .talk_infinitetalk import _speak, audio_seconds
from .video_wan import extract_keyframes

REQUIRED = ("width", "height", "steps", "timeout_s")
MODELS = ("infinitetalk", "longcat")
_ready: dict = {"weights": False}   # per model: "weights" (InfiniteTalk), "weights:longcat:<version>"


def _models_dir() -> Path:
    return Path(os.environ.get("MW_MODEL_DIR", "/models/mw")) / "comfy"


def _prepare_image(src: Path, dest: Path, w: int, h: int) -> None:
    """Cover-crop the approved photo to the clip's frame, centred (a 4:5 photo into a 9:16 talking shot)."""
    from PIL import Image
    im = Image.open(src).convert("RGB")
    scale = max(w / im.width, h / im.height)
    im = im.resize((math.ceil(im.width * scale), math.ceil(im.height * scale)), Image.LANCZOS)
    left, top = (im.width - w) // 2, (im.height - h) // 2
    im.crop((left, top, left + w, top + h)).save(dest)


def _blocks_to_swap(params: dict) -> int:
    """Swap transformer blocks to CPU only on cards too small to hold the fp8 model with room to work."""
    import torch
    _, total = torch.cuda.mem_get_info()
    return 0 if total / 1e9 >= float(params.get("swap_below_gb", 40)) else int(params.get("blocks_to_swap", 20))


def model_of(params: dict) -> str:
    model = str(params.get("model") or "infinitetalk")
    if model not in MODELS:
        raise NonRetryable(f"unknown talk model {model!r}; known: {list(MODELS)}")
    return model


def join_windows(clips: list[Path], audio: Path, seconds: float, dest: Path) -> Path:
    """LongCat's per-window clips, in order, as one clip with the voice line, cut to the line's length (the last
    window pads past it)."""
    import subprocess

    from ...media import ffmpeg
    listing = dest.with_suffix(".txt")
    listing.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips))
    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(audio),
           "-map", "0:v:0", "-map", "1:a:0", "-t", f"{seconds:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"joining LongCat windows failed: {proc.stderr[-800:]}")
    listing.unlink(missing_ok=True)
    return dest


def generate(payload: dict, ctx: JobContext) -> HandlerResult:
    if not payload.get("image_uri"):
        raise NonRetryable("payload needs image_uri")
    if not payload.get("audio_uri") and not (payload.get("script") or "").strip():
        raise NonRetryable("payload needs audio_uri or script")
    if not payload.get("keyframes"):
        raise NonRetryable("payload needs keyframes (the video gate judges them)")
    params: dict[str, Any] = payload.get("params") or {}
    missing = [k for k in REQUIRED if params.get(k) is None]
    if missing:
        raise NonRetryable(f"params missing {missing}")
    model = model_of(params)
    seed = payload.get("seed")
    seed = int(seed) if seed is not None else random.randrange(1 << 31)
    work = Path(ctx.workdir)

    models = _models_dir()
    if model == "longcat":
        try:
            version = comfy.longcat_version(params)
        except ValueError as e:
            raise NonRetryable(str(e)) from None
        weights, ready_key = comfy.LONGCAT_WEIGHTS[version], f"weights:longcat:{version}"
    else:
        version, weights, ready_key = None, None, "weights"
    if not _ready.get(ready_key):
        comfy.write_model_paths(models)
        comfy.ensure_weights(models, weights, prune=True)
        _ready[ready_key] = True
    comfy.start()

    start = ctx.fetch(payload["image_uri"], "start_frame.png")
    spoken = None
    if payload.get("audio_uri"):
        audio = ctx.fetch(payload["audio_uri"], "voice.wav")
    else:
        audio = spoken = _speak(payload, work)
    seconds = audio_seconds(audio)
    tag = ctx.job.id
    inp = comfy.COMFY_DIR / "input"
    inp.mkdir(parents=True, exist_ok=True)
    image_name, audio_name = f"mw_{tag}.png", f"mw_{tag}{audio.suffix or '.wav'}"
    _prepare_image(start, inp / image_name, int(params["width"]), int(params["height"]))
    shutil.copy(audio, inp / audio_name)

    prompt = payload.get("prompt") or "a person talks to the camera"
    clip = work / "talk.mp4"
    t0 = time.monotonic()
    if model == "longcat":
        g = comfy.graph_longcat(image=image_name, audio=audio_name, prompt=prompt,
                                frames=math.ceil(seconds * comfy.LONGCAT_FPS[version]), seed=seed, params=params,
                                blocks_to_swap=_blocks_to_swap(params))
        windows = comfy.run_all(g, timeout_s=float(params["timeout_s"]))
        join_windows(windows, audio, seconds, clip)
        for f in windows:
            f.unlink(missing_ok=True)
    else:
        g = comfy.graph(image=image_name, audio=audio_name, prompt=prompt, frames=math.ceil(seconds * 25), seed=seed,
                        params=params, blocks_to_swap=_blocks_to_swap(params))
        shutil.move(str(comfy.run(g, timeout_s=float(params["timeout_s"]))), clip)
    render_s = round(time.monotonic() - t0, 1)
    for f in (inp / image_name, inp / audio_name):
        f.unlink(missing_ok=True)

    sync = lipsync.score_in_venv(clip, audio, max_offset_frames=int(params.get("sync_max_offset_frames", 5)))
    outputs = [OutputFile(clip, "video", {"seconds": seconds, "width": params["width"], "height": params["height"],
                                          "lip_sync": True, "sync": sync, "engine": "comfy", "model": model}),
               *extract_keyframes(clip, work, int(payload["keyframes"]))]
    if spoken is not None:
        outputs.append(OutputFile(spoken, "audio", {"seconds": seconds}))
    if model == "longcat":
        names = {"model": weights["diffusion_models"][0][1], "lora": weights["loras"][0][1],
                 "adapter": f"LongCat-Video-Avatar {version}"}
    else:
        names = {"model": comfy.MODEL, "adapter": comfy.ADAPTER, "lora": comfy.LORA}
    recipe = {**names, "engine": "comfy+WanVideoWrapper", "talk_model": model, "prompt": payload.get("prompt"),
              "image_uri": payload["image_uri"], "audio_uri": payload.get("audio_uri"), "script": payload.get("script"), "voice": payload.get("voice"), "seed": seed, "params": params}
    return HandlerResult(outputs, recipe=recipe, meta={"output_seconds": seconds, "render_seconds": render_s,
                                                        "seconds_per_output_second": round(render_s / max(seconds, 0.1), 1)})


HANDLERS = {"talk.infinitetalk": generate}
