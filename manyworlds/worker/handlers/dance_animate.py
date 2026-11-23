"""Dance and motion transfer (job kind comfy.wan_animate2): Wan-Animate-2 in ComfyUI core (worker/comfy_animate.py)
animates the creator's approved photo with the motion of a reference clip. OFF: refused unless the policy on this
host says `dance.enabled: true`, and the reference must pass manyworlds.dance.reference (ours or licensed, never a
downloaded platform video). Production has no format that uses it; `mw bench comfy.wan_animate2 --extra
'{"image_uri": ..., "reference": "<dance.references id>"}'` is the only way in.

Runs on the ComfyUI worker image (mw-worker-talkcomfy), family `dance` (job_kinds), so a story-clip machine never
loads it. The output keeps the reference's frame rate and never its audio (the graph takes only its frames).

Payload: image_uri, reference, prompt, keyframes, seed, crops; params override policy dance.generation
(width, height, frames, steps, ..., timeout_s).
"""

from __future__ import annotations

import random
import shutil
import time
from pathlib import Path
from typing import Any

from ...policy import Policy
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

KIND = "comfy.wan_animate2"
_ready: dict = {"weights": False}


def _policy(policy: Policy | None) -> Policy:
    if policy is not None:
        return policy
    from ... import policy as policy_mod
    return policy_mod.load()


def check(payload: dict, policy: Policy) -> tuple[dict, dict, int]:
    """(the reference's catalogue entry, the graph params, frames), or NonRetryable: dance off, no photo, a
    reference that isn't ours or licensed, or a length the node can't take. Pure (tested in CI)."""
    from ... import dance
    from .. import comfy_animate
    if not dance.enabled(policy):
        raise NonRetryable("dance is off (policy dance.enabled)")
    if not payload.get("image_uri") or not payload.get("reference"):
        raise NonRetryable("payload needs image_uri and reference (a dance.references id)")
    try:
        ref = dance.reference(str(payload["reference"]), policy)
    except dance.RefusedReference as e:
        raise NonRetryable(str(e)) from None
    params: dict[str, Any] = {**(dance.cfg(policy).get("generation") or {}), **(payload.get("params") or {})}
    try:
        frames = comfy_animate.check_length(int(params.get("frames", comfy_animate.MAX_WINDOW)))
        for k in ("width", "height"):
            if int(params[k]) % 16:
                raise ValueError(f"{k} {params[k]} is not a multiple of 16")
    except (KeyError, ValueError) as e:
        raise NonRetryable(f"dance params: {e}") from None
    return ref, params, frames


def _seconds(clip: Path, frames: int, fps: float) -> float:
    try:
        import imageio.v3 as iio
        meta = iio.immeta(str(clip))
        if meta.get("duration"):
            return round(float(meta["duration"]), 2)
        if meta.get("fps"):
            return round(frames / float(meta["fps"]), 2)
    except Exception:  # noqa: BLE001 - an estimate is enough for the ledger's output seconds
        pass
    return round(frames / fps, 2)


def generate(payload: dict, ctx: JobContext, *, policy: Policy | None = None) -> HandlerResult:
    policy = _policy(policy)
    ref, params, frames = check(payload, policy)
    from .. import comfy, comfy_animate, crops
    from .talk_comfy import _models_dir, _prepare_image
    from ..keyframes import extract_keyframes
    seed = payload.get("seed")
    seed = int(seed) if seed is not None else random.randrange(1 << 31)
    work = Path(ctx.workdir)

    models = _models_dir()
    if not _ready["weights"]:
        comfy.write_model_paths(models)
        comfy.ensure_weights(models, comfy_animate.WEIGHTS, prune=True)
        _ready["weights"] = True
    comfy.start()

    inp = comfy.COMFY_DIR / "input"
    inp.mkdir(parents=True, exist_ok=True)
    image_name, video_name = f"mw_{ctx.job.id}.png", f"mw_{ctx.job.id}_drive.mp4"
    _prepare_image(ctx.fetch(payload["image_uri"], "start_frame.png"), inp / image_name,
                   int(params["width"]), int(params["height"]))
    shutil.copy(ctx.fetch(ref["uri"], "drive.mp4"), inp / video_name)
    prompt = payload.get("prompt") or "a person dances, full body, natural movement"
    g = comfy_animate.graph(image=image_name, video=video_name, prompt=prompt, frames=frames, seed=seed,
                            params=params)
    t0 = time.monotonic()
    out = comfy.run(g, timeout_s=float(params.get("timeout_s", 1800)))
    render_s = round(time.monotonic() - t0, 1)
    clip = work / "dance.mp4"
    shutil.move(str(out), clip)
    for f in (inp / image_name, inp / video_name):
        f.unlink(missing_ok=True)

    seconds = _seconds(clip, frames, float(params.get("fps", 16)))
    keyframes = extract_keyframes(clip, work, int(payload.get("keyframes", 4)))
    keyframes += crops.crops_in_venv(keyframes, work, payload.get("crops"))
    recipe = {"model": comfy_animate.MODEL, "lora": comfy_animate.LORA, "engine": "comfy core (Wan-Animate-2)",
              "prompt": prompt, "image_uri": payload["image_uri"], "seed": seed, "params": params,
              "reference": {k: ref.get(k) for k in ("id", "owner", "licence")}}
    return HandlerResult([OutputFile(clip, "video", {"seconds": seconds, "width": params["width"],
                                                     "height": params["height"], "engine": "comfy",
                                                     "dance": True, "audio": False}), *keyframes],
                         recipe=recipe, meta={"output_seconds": seconds, "render_seconds": render_s})


HANDLERS = {KIND: generate}

def release() -> None:
    """Give the GPU back before another family's job (manyworlds.worker.runner._switch_to)."""
    from .. import comfy
    comfy.free()
