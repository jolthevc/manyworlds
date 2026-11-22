"""Lip-synced talking video (job kind talk.infinitetalk) on ComfyUI + WanVideoWrapper (manyworlds.worker.comfy).

Same job as handlers/talk_infinitetalk.py (approved photo + voice line or script, keyframes for the video gate, the
lip-sync score, the spoken line as its own output), rendered by the community's optimised stack: fp8-scaled
InfiniteTalk with the 4-step lightx2v LoRA and SageAttention. The talk worker image chooses this engine
(MW_TALK_ENGINE=comfy). Weights download once into MW_MODEL_DIR/comfy (the pool's network volume if it has one).

Payload as for talk_infinitetalk; params{width, height, steps, shift, cfg, scheduler, audio_scale, audio_cfg,
frame_window, motion_frame, swap_below_gb, blocks_to_swap, timeout_s}.
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
_ready: dict = {"weights": False}


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
    seed = payload.get("seed")
    seed = int(seed) if seed is not None else random.randrange(1 << 31)
    work = Path(ctx.workdir)

    models = _models_dir()
    if not _ready["weights"]:
        comfy.write_model_paths(models)
        comfy.ensure_weights(models, prune=True)
        _ready["weights"] = True
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

    g = comfy.graph(image=image_name, audio=audio_name, prompt=payload.get("prompt") or "a person talks to the camera",
                    frames=math.ceil(seconds * 25), seed=seed, params=params, blocks_to_swap=_blocks_to_swap(params))
    t0 = time.monotonic()
    out = comfy.run(g, timeout_s=float(params["timeout_s"]))
    render_s = round(time.monotonic() - t0, 1)
    clip = work / "talk.mp4"
    shutil.move(str(out), clip)
    for f in (inp / image_name, inp / audio_name):
        f.unlink(missing_ok=True)

    sync = lipsync.score_in_venv(clip, audio, max_offset_frames=int(params.get("sync_max_offset_frames", 5)))
    outputs = [OutputFile(clip, "video", {"seconds": seconds, "width": params["width"], "height": params["height"],
                                          "lip_sync": True, "sync": sync, "engine": "comfy"}),
               *extract_keyframes(clip, work, int(payload["keyframes"]))]
    if spoken is not None:
        outputs.append(OutputFile(spoken, "audio", {"seconds": seconds}))
    recipe = {"model": comfy.MODEL, "adapter": comfy.ADAPTER, "lora": comfy.LORA, "engine": "comfy+WanVideoWrapper",
              "prompt": payload.get("prompt"), "image_uri": payload["image_uri"], "audio_uri": payload.get("audio_uri"),
              "script": payload.get("script"), "voice": payload.get("voice"), "seed": seed, "params": params}
    return HandlerResult(outputs, recipe=recipe, meta={"output_seconds": seconds, "render_seconds": render_s,
                                                        "seconds_per_output_second": round(render_s / max(seconds, 0.1), 1)})


HANDLERS = {"talk.infinitetalk": generate}
