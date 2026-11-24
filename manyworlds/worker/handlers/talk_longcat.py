"""Lip-synced talking video as ONE continuous take: LongCat-Video-Avatar 1.6 (Meituan, MIT) run by its own repo.

Kijai's ComfyUI-WanVideoWrapper supports LongCat-Video-Avatar 1.0 and 1.5 only (newest commit a7439db, 2026-11-13;
1.6 came out 2026-11-15) and no 1.6 checkpoint in the wrapper's format exists, so 1.6 runs natively: the LongCat repo
(cloned into the talk worker image at a pinned commit, MW_LONGCAT_REPO) in its own Python environment
(MW_LONGCAT_PYTHON, docker/constraints-longcat.txt), launched by torchrun on one GPU, as a subprocess of this worker.
ComfyUI stays the engine for every other talk model.

One job renders the whole voice line (up to params.max_seconds): the script generates its first 93 frames from the
photo (ai2v), then continues autoregressively, each segment from the previous one's last 13 frames and re-anchored
to the photo's latent (`--ref_img_index`, `--mask_frame_range`), in one process: no seams, no re-encode between our
jobs. 1.6 = the 1.5 architecture (Whisper-large-v3 audio, 25 fps) plus two distillation LoRAs (loraA DMD, loraB OPSD,
blended over the first `mix_steps` of 8 steps). Precision (params.quantization): int8 (`--use_int8`, the 15.9 GB
INT8 DiT), bf16 (31.7 GB), fp8 (Transformer Engine, not in our image: refused), or auto: int8 on cards under
params.bf16_min_vram_gb, else bf16. The script fixes its own seed (42 + rank) and negative prompt.

Job kinds: `talk.longcat` (its own attempt limit, gpu_runs.max_attempt_minutes_by_kind: a 90 s take renders for
over an hour), and `talk.infinitetalk` with params.model `longcat_native` (talk_comfy dispatches here), so a talk
profile can pick it. Payload as for talk_comfy: image_uri, audio_uri or script, prompt, keyframes, crops,
audio_window; params from generation.talk_profiles.longcat16. Weights download onto the GPU machine (weights_root),
never anywhere else.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .. import crops, lipsync
from ..keyframes import extract_keyframes
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

REPO_DIR = Path(os.environ.get("MW_LONGCAT_REPO", "/opt/LongCat-Video"))
PYTHON = os.environ.get("MW_LONGCAT_PYTHON", "/opt/longcat/bin/python")
SCRIPT = "run_demo_avatar_single_audio_to_video.py"
AVATAR = "meituan-longcat/LongCat-Video-Avatar-1.6"
BASE = "meituan-longcat/LongCat-Video"   # tokenizer, UMT5 text encoder and VAE: read from <checkpoint>/../LongCat-Video
MODEL_TYPE = "avatar-v1.6"
# File names and sizes from the Hugging Face file lists (2026-11-23). Only what the single-audio script reads:
# not the 6 GB flax / fp32 / .bin Whisper copies, the demo videos, or the other precision's DiT.
DIT_FILES = {"int8": ["base_model_int8/*"],     # 4 shards, 15.88 GB
             "bf16": ["base_model/*"]}          # 6 shards, 31.71 GB
AVATAR_FILES = ["config.json", "model_index.json", "scheduler/*",
                "loraA/*", "loraB/*",           # DMD and OPSD LoRAs, 2.52 GB each
                "vocal_separator/*",            # Kim_Vocal_2.onnx (67 MB) and the separator's model lists
                *[f"whisper-large-v3/{n}" for n in (           # fp16 safetensors, 3.09 GB
                    "config.json", "model.safetensors", "preprocessor_config.json", "generation_config.json",
                    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
                    "special_tokens_map.json", "normalizer.json")]]
BASE_FILES = ["config.json", "model_index.json", "tokenizer/*",   # 21 MB
              "text_encoder/*",                                  # UMT5-XXL, 5 fp32 shards, 22.72 GB (loaded bf16)
              "vae/*"]                                           # 0.51 GB
DOWNLOAD_GB = {"int8": 47.4, "bf16": 63.2}   # the sums of the sizes above
QUANTIZATIONS = ("int8", "bf16", "fp8", "auto")
REQUIRED = ("width", "height", "timeout_s", "max_seconds", "quantization", "resolution", "frames_per_segment",
            "cond_frames", "fps", "mix_steps", "mix_alpha", "lora_path", "ref_img_index", "mask_frame_range")
log = logging.getLogger("manyworlds.talk.longcat")


def weights_root(params: dict) -> Path:
    """Where the weights go: the machine's weights cache (MW_MODEL_DIR: a pool's network volume where it has one)
    with params.weights_on_volume, else the container's own disk (MW_LONGCAT_DIR), so ~47 GB never fills a shared
    volume that already holds the other talk weights."""
    if params.get("weights_on_volume"):
        return Path(os.environ.get("MW_MODEL_DIR", "/models/mw")) / "longcat"
    return Path(os.environ.get("MW_LONGCAT_DIR", "/models/longcat"))


def resolve_quantization(params: dict, vram_gb: float | None) -> str:
    """int8 / bf16 / fp8 as params say; auto: int8 on a card under bf16_min_vram_gb (a 48 GB L40S), else bf16."""
    q = str(params["quantization"])
    if q not in QUANTIZATIONS:
        raise NonRetryable(f"unknown LongCat quantization {q!r}; known: {list(QUANTIZATIONS)}")
    if q == "auto":
        floor = float(params.get("bf16_min_vram_gb") or 0)
        return "bf16" if vram_gb is not None and floor and vram_gb >= floor else "int8"
    return q


def segments_for(seconds: float, *, fps: int, frames_per_segment: int, cond_frames: int) -> int:
    """How many segments the script must make to cover `seconds`: the first gives frames_per_segment frames, each
    later one frames_per_segment - cond_frames new ones (the script's own arithmetic)."""
    frames = math.ceil(float(seconds) * fps)
    if frames <= frames_per_segment:
        return 1
    return 1 + math.ceil((frames - frames_per_segment) / (frames_per_segment - cond_frames))


def avatar_patterns(quantization: str) -> list[str]:
    return DIT_FILES["bf16" if quantization == "fp8" else quantization] + AVATAR_FILES


def ensure_weights(root: Path, quantization: str, *, min_free_gb: float = 0) -> tuple[Path, Path]:
    """The avatar and base checkpoints under `root`, side by side as the script expects; downloaded once (a rerun
    skips files already there). Refuses, as a machine fault, a disk without room for them."""
    from huggingface_hub import snapshot_download

    avatar, base = root / AVATAR.split("/")[1], root / BASE.split("/")[1]
    root.mkdir(parents=True, exist_ok=True)
    marker = root / f".complete-{quantization}"
    if not marker.exists():
        need = DOWNLOAD_GB["bf16" if quantization == "fp8" else quantization] + float(min_free_gb or 0)
        free = shutil.disk_usage(root).free / 1e9
        have = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e9
        if free + have < need:
            raise RuntimeError(f"No space left on device for LongCat weights: {free:.0f} GB free, "
                               f"{have:.0f} GB there, {need:.0f} GB needed")
        token = os.environ.get("HF_TOKEN") or None
        snapshot_download(AVATAR, local_dir=str(avatar), allow_patterns=avatar_patterns(quantization), token=token)
        snapshot_download(BASE, local_dir=str(base), allow_patterns=BASE_FILES, token=token)
        marker.write_text("ok")
    return avatar, base


def write_input(workdir: Path, *, image: Path, audio: Path, prompt: str) -> Path:
    p = Path(workdir) / "longcat_input.json"
    p.write_text(json.dumps({"prompt": prompt, "cond_image": str(image), "cond_audio": {"person1": str(audio)}}))
    return p


def build_command(*, checkpoint: Path, input_json: Path, out_dir: Path, segments: int, quantization: str,
                  params: dict, python: str = PYTHON, repo: Path = REPO_DIR) -> list[str]:
    """torchrun on one GPU (the script initialises torch.distributed and reads RANK) running the single-audio
    avatar script: image-to-video, then `segments - 1` continuations, at 1.6's distilled 8 steps."""
    cmd = [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", str(Path(repo) / SCRIPT),
           "--context_parallel_size=1", f"--checkpoint_dir={checkpoint}", "--stage_1=ai2v",
           f"--input_json={input_json}", f"--output_dir={out_dir}", f"--resolution={params['resolution']}",
           f"--num_segments={int(segments)}", f"--ref_img_index={int(params['ref_img_index'])}",
           f"--mask_frame_range={int(params['mask_frame_range'])}", f"--model_type={MODEL_TYPE}", "--use_distill",
           "--lora_path", str(params["lora_path"]), f"--mix_steps={int(params['mix_steps'])}",
           f"--mix_alpha={float(params['mix_alpha'])}"]
    if quantization == "int8":
        cmd.append("--use_int8")
    elif quantization == "fp8":
        cmd.append("--use_fp8")
    if params.get("compile"):
        cmd += ["--compile_dit", "--compile_vae"]
    return cmd


def run(cmd: list[str], cwd: Path, timeout: float) -> None:
    env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"LongCat timed out after {timeout:.0f} s") from e
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-3000:]
        log.error("LongCat failed (%s):\n%s", proc.returncode, tail)
        # The text carries "CUDA out of memory" etc., so the runner counts machine faults correctly.
        raise RuntimeError(f"LongCat exited {proc.returncode}: {tail[-800:]}")


def newest_video(out_dir: Path) -> Path:
    """The script's final video: the last cumulative continuation (video_continue_N), or ai2v_demo_1 for one segment."""
    videos = sorted(Path(out_dir).glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not videos:
        raise RuntimeError(f"LongCat wrote no video into {out_dir}")
    return videos[-1]


def _vram_gb() -> float | None:
    try:
        import torch
        return round(torch.cuda.mem_get_info()[1] / 1e9, 1) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - no CUDA here: auto picks the smaller build
        return None


def generate(payload: dict, ctx: JobContext) -> HandlerResult:
    from .talk_comfy import _prepare_image, join_windows, voice_line
    if not payload.get("image_uri"):
        raise NonRetryable("payload needs image_uri")
    if not payload.get("audio_uri") and not (payload.get("script") or "").strip():
        raise NonRetryable("payload needs audio_uri or script")
    if not payload.get("keyframes"):
        raise NonRetryable("payload needs keyframes (the video gate judges them)")
    params: dict[str, Any] = payload.get("params") or {}
    missing = [k for k in REQUIRED if params.get(k) is None]
    if missing:   # settings come from policy generation.talk_profiles.longcat16; no defaults here to drift from it
        raise NonRetryable(f"params missing {missing}")
    quantization = resolve_quantization(params, _vram_gb())
    if quantization == "fp8":
        probe = subprocess.run([PYTHON, "-c", "import transformer_engine"], capture_output=True, text=True)
        if probe.returncode != 0:
            raise NonRetryable("LongCat fp8 needs Transformer Engine, which the talk image doesn't install: "
                               "use quantization int8 or bf16")
    work = Path(ctx.workdir)
    audio, seconds, extra_outputs = voice_line(payload, ctx, work)
    if seconds > float(params["max_seconds"]):
        raise NonRetryable(f"the voice line is {seconds:.1f} s; one LongCat take is at most {params['max_seconds']} s")
    fps = int(params["fps"])
    n = segments_for(seconds, fps=fps, frames_per_segment=int(params["frames_per_segment"]),
                     cond_frames=int(params["cond_frames"]))

    avatar, _ = ensure_weights(weights_root(params), quantization, min_free_gb=float(params.get("min_free_disk_gb") or 0))
    image = work / "longcat_start.png"
    _prepare_image(ctx.fetch(payload["image_uri"], "start_frame.png"), image, int(params["width"]),
                   int(params["height"]), stretch_tolerance=float(params.get("fit_stretch_tolerance") or 0))
    prompt = payload.get("prompt") or "a person talks to the camera"
    out_dir = work / "longcat_out"
    shutil.rmtree(out_dir, ignore_errors=True)   # a rerun on the same machine never picks up a stale video
    cmd = build_command(checkpoint=avatar, input_json=write_input(work, image=image, audio=audio, prompt=prompt),
                        out_dir=out_dir, segments=n, quantization=quantization, params=params)
    t0 = time.monotonic()
    run(cmd, work, timeout=float(params["timeout_s"]))
    render_s = round(time.monotonic() - t0, 1)
    clip = join_windows([newest_video(out_dir)], audio, seconds, work / "talk.mp4")   # cut to the line, our audio

    sync = lipsync.score_in_venv(clip, audio, max_offset_frames=int(params.get("sync_max_offset_frames", 5)))
    outputs = [OutputFile(clip, "video", {"seconds": seconds, "width": params["width"], "height": params["height"],
                                          "lip_sync": True, "sync": sync, "engine": "native",
                                          "model": "longcat16", "segments": n, "quantization": quantization,
                                          "single_take": True}),
               *extract_keyframes(clip, work, int(payload["keyframes"]))]
    outputs += crops.crops_in_venv([o for o in outputs if o.kind == "keyframe"], work, payload.get("crops"))
    outputs += extra_outputs
    recipe = {"model": AVATAR, "base": BASE, "engine": "LongCat-Video native (torchrun, 1 GPU)",
              "repo_commit": os.environ.get("MW_LONGCAT_COMMIT"), "talk_model": "longcat16",
              "quantization": quantization, "segments": n, "prompt": payload.get("prompt"),
              "image_uri": payload["image_uri"], "audio_uri": payload.get("audio_uri"),
              "audio_window": payload.get("audio_window"), "script": payload.get("script"),
              "voice": payload.get("voice"), "seed": "fixed by the LongCat script (42 + rank)", "params": params}
    return HandlerResult(outputs, recipe=recipe, meta={"output_seconds": seconds, "render_seconds": render_s,
                                                        "seconds_per_output_second": round(render_s / max(seconds, 0.1), 1)})


HANDLERS = {"talk.longcat": generate}
