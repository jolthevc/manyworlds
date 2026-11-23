"""Lip-synced talking video: InfiniteTalk (MeiGen, Apache 2.0) from one approved photo and a voice line.

The model is Wan2.1-I2V-14B-480P plus InfiniteTalk's audio adapter, driven by chinese-wav2vec2-base
audio features. It runs as a subprocess of its own repo (cloned into the talk worker image at a
pinned commit, MW_INFINITETALK_REPO), because its code is a script, not a library. Streaming mode
makes the clip as long as the audio; output is 25 fps with the audio track muxed in.

Payload: image_uri, prompt, and audio_uri or script (+ voice, speed, lang: spoken here with Kokoro,
which InfiniteTalk's script imports anyway); params{size, steps, audio_cfg, text_cfg, shift, extra,
lora}; seed; keyframes. `extra` is a list of extra CLI flags; "{infinitetalk}" in it is replaced by
the adapter's weight directory (for --quant_dir). Weights download onto the GPU machine (bucket
mirror first, else Hugging Face), never anywhere else. Keyframes go to the video gate as for clips.
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .. import crops, lipsync, models
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile
from ..keyframes import extract_keyframes

MODEL = "MeiGen-AI/InfiniteTalk"
BASE = "Wan-AI/Wan2.1-I2V-14B-480P"
WAV2VEC = "TencentGameMate/chinese-wav2vec2-base"
REPO_DIR = Path(os.environ.get("MW_INFINITETALK_REPO", "/opt/InfiniteTalk"))
# The adapter repo is 169 GB, mostly quantised multi-person builds: fetch the single-person adapter
# (10 GB) and only the quant files a run's flags name ("{infinitetalk}/quant_models/...").
ADAPTER_FILES = ["single/*"]
REQUIRED_PARAMS = ("size", "steps", "audio_cfg", "text_cfg", "motion_frame", "timeout_s")
log = logging.getLogger("manyworlds.talk")


def _local(repo: str, store, workdir: Path, **download) -> str:
    """The repo's files on this machine: from our bucket when mirrored, else Hugging Face."""
    found = models.ensure(repo, store, workdir)
    if found != repo:
        return found
    from huggingface_hub import snapshot_download

    d = models.local_dir(repo)
    snapshot_download(repo, local_dir=str(d), token=os.environ.get("HF_TOKEN") or None, **download)
    return str(d)


def adapter_files(extra: list[str]) -> list[str]:
    """The adapter files a run needs: the single-person adapter, any file its flags name, and with --quant the
    matching quantised text encoder, which InfiniteTalk loads from the same folder (quant_models/t5_<q>...)."""
    named = [a.split("{infinitetalk}/", 1)[1] for a in extra if "{infinitetalk}/" in a]
    files = ADAPTER_FILES + [n.rsplit(".", 1)[0] + "*" for n in named]
    if "--quant" in extra:
        q = extra[extra.index("--quant") + 1]
        files += [f"quant_models/t5_{q}*", f"quant_models/t5_map_{q}*"]
    return files


def lora_path(lora: dict | None, workdir: Path) -> dict | None:
    """A LoRA named by Hugging Face repo and file is fetched onto this machine."""
    if not lora or "repo" not in lora:
        return lora
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(lora["repo"], lora["file"], local_dir=str(models.local_dir(lora["repo"])),
                           token=os.environ.get("HF_TOKEN") or None)
    return {"path": path, "scale": lora.get("scale", 1.0)}


# With --quant, InfiniteTalk builds the transformer and text encoder from the quantised files and never reads
# these from the base model (wan/multitalk.py, wan/modules/t5.py): ~68 GB not downloaded per machine.
QUANT_SKIPS_BASE = ["diffusion_pytorch_model*", "models_t5_umt5-xxl-enc-bf16.pth"]


def weights(store, workdir: Path, extra: list[str] = ()) -> dict[str, str]:
    base = _local(BASE, store, workdir, **({"ignore_patterns": QUANT_SKIPS_BASE} if "--quant" in extra else {}))
    wav2vec = _local(WAV2VEC, store, workdir)
    if not (Path(wav2vec) / "model.safetensors").exists():
        # The README takes the safetensors build from an open PR of the same repo.
        from huggingface_hub import hf_hub_download
        hf_hub_download(WAV2VEC, "model.safetensors", revision="refs/pr/1", local_dir=wav2vec)
    adapter = _local(MODEL, store, workdir, allow_patterns=adapter_files(list(extra)))
    return {"base": base, "wav2vec": wav2vec, "infinitetalk": adapter}


def build_command(*, base: str, wav2vec: str, infinitetalk: str, input_json: Path, out_stem: Path, size: str,
                  steps: int, audio_cfg: float, text_cfg: float, seed: int, extra: list[str],
                  lora: dict | None, motion_frame: int, shift: float | None = None) -> list[str]:
    cmd = [sys.executable, "generate_infinitetalk.py",
           "--ckpt_dir", base, "--wav2vec_dir", wav2vec,
           "--infinitetalk_dir", f"{infinitetalk}/single/infinitetalk.safetensors",
           "--input_json", str(input_json), "--size", size, "--mode", "streaming", "--motion_frame", str(motion_frame),
           "--sample_steps", str(steps), "--sample_text_guide_scale", str(text_cfg),
           "--sample_audio_guide_scale", str(audio_cfg), "--base_seed", str(seed),
           "--save_file", str(out_stem), "--audio_save_dir", str(Path(out_stem).parent / "audio")]
    if shift is not None:
        cmd += ["--sample_shift", str(shift)]
    if lora:
        cmd += ["--lora_dir", lora["path"], "--lora_scale", str(lora.get("scale", 1.0))]
    return cmd + [a.replace("{infinitetalk}", infinitetalk) for a in extra]


def write_input(workdir: Path, *, image: Path, audio: Path, prompt: str) -> Path:
    p = Path(workdir) / "input.json"
    p.write_text(json.dumps({"prompt": prompt, "cond_video": str(image), "cond_audio": {"person1": str(audio)}}))
    return p


def run(cmd: list[str], cwd: Path, timeout: float | None = None) -> None:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # A hung render heartbeats fine; without this only the enforcers' attempt limit would stop it.
        raise RuntimeError(f"InfiniteTalk timed out after {timeout:.0f} s") from e
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-3000:]
        log.error("InfiniteTalk failed (%s):\n%s", proc.returncode, tail)
        # The text carries "CUDA out of memory" etc., so the runner counts machine faults correctly.
        raise RuntimeError(f"InfiniteTalk exited {proc.returncode}: {tail[-800:]}")


def audio_seconds(path: Path) -> float:
    import soundfile as sf
    return round(sf.info(str(path)).duration, 2)


def _speak(payload: dict, workdir: Path) -> Path:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    pipe = KPipeline(lang_code=payload.get("lang", "a"))
    from ... import voices   # a creator's own blend of stock voices, or one stock voice
    chunks = [a for _, _, a in pipe(payload["script"], voice=voices.voice_arg(pipe, payload.get("voice", "af_heart")),
                                    speed=float(payload.get("speed", 1.0)))]
    path = workdir / "voice.wav"
    sf.write(str(path), np.concatenate([np.asarray(c) for c in chunks]), 24000)
    return path


def generate(payload: dict, ctx: JobContext) -> HandlerResult:
    if not payload.get("image_uri"):
        raise NonRetryable("payload needs image_uri")
    if not payload.get("audio_uri") and not (payload.get("script") or "").strip():
        raise NonRetryable("payload needs audio_uri or script")
    if not payload.get("keyframes"):
        raise NonRetryable("payload needs keyframes (the video gate judges them)")
    params: dict[str, Any] = payload.get("params") or {}
    missing = [k for k in REQUIRED_PARAMS if params.get(k) is None]
    if missing:   # settings come from policy generation.talk; no defaults here to drift from it
        raise NonRetryable(f"params missing {missing}")
    seed = payload.get("seed")
    seed = int(seed) if seed is not None else random.randrange(1 << 31)
    work = Path(ctx.workdir)

    image = ctx.fetch(payload["image_uri"], "start_frame.png")
    spoken = None
    if payload.get("audio_uri"):
        audio = ctx.fetch(payload["audio_uri"], "voice.wav")
    else:
        audio = spoken = _speak(payload, work)
    seconds = audio_seconds(audio)

    extra = list(params.get("extra") or [])
    w = weights(ctx.store, work, extra=extra)
    prompt = payload.get("prompt") or "a person talks to the camera"
    out_stem = work / "talk"
    cmd = build_command(base=w["base"], wav2vec=w["wav2vec"], infinitetalk=w["infinitetalk"],
                        input_json=write_input(work, image=image, audio=audio, prompt=prompt), out_stem=out_stem,
                        size=params["size"], steps=int(params["steps"]), audio_cfg=float(params["audio_cfg"]),
                        text_cfg=float(params["text_cfg"]), motion_frame=int(params["motion_frame"]), seed=seed,
                        extra=extra, lora=lora_path(params.get("lora"), work), shift=params.get("shift"))
    t0 = time.monotonic()
    run(cmd, REPO_DIR, timeout=float(params["timeout_s"]))
    render_s = round(time.monotonic() - t0, 1)
    clip = Path(f"{out_stem}.mp4")
    if not clip.exists():
        raise RuntimeError(f"InfiniteTalk wrote no {clip.name}")

    # Sync from the frames and the voice (worker/lipsync.py): keyframes can't show it, so the gate reads this.
    sync = lipsync.score_in_venv(clip, audio, max_offset_frames=int(params.get("sync_max_offset_frames", 5)))
    outputs = [OutputFile(clip, "video", {"seconds": seconds, "size": params["size"],
                                          "lip_sync": True, "sync": sync}),
               *extract_keyframes(clip, work, int(payload["keyframes"]))]
    outputs += crops.crops_in_venv([o for o in outputs if o.kind == "keyframe"], work, payload.get("crops"))
    if spoken is not None:
        outputs.append(OutputFile(spoken, "audio", {"seconds": seconds}))
    recipe = {"model": MODEL, "base": BASE, "prompt": prompt, "image_uri": payload["image_uri"],
              "audio_uri": payload.get("audio_uri"), "script": payload.get("script"), "voice": payload.get("voice"),
              "seed": seed, "params": params}
    return HandlerResult(outputs, recipe=recipe, meta={"output_seconds": seconds, "render_seconds": render_s})


HANDLERS = {"talk.infinitetalk": generate}
