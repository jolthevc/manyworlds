"""Lip-synced talking video through ComfyUI + Kijai's WanVideoWrapper, the stack the community runs it on.

InfiniteTalk's own research script loads the 14B model at full precision and can't combine its fp8 build with the
4-step distillation LoRA, so on one card it was either slow (fp8, 40 steps: ~145 s of compute per output second on
an L40S) or out of memory (2026-11-22). WanVideoWrapper (Apache-2.0) runs the fp8-scaled model and InfiniteTalk
adapter with the lightx2v LoRA, SageAttention and block swapping, on 24-48 GB cards.

The graph follows Kijai's example workflow (example_workflows/wanvideo_2_1_14B_I2V_InfiniteTalk_example_03.json at
the pinned commit), in ComfyUI's API format: editor-only nodes (Set/Get, notes, previews) are gone, the optional
vocal separator and CLIP vision encoder are left out (our audio is a clean voice line; the note there says CLIP
vision is not needed), and the video is written by ComfyUI's own CreateVideo/SaveVideo. `graph` is pure (tested in
CI); the server helpers run inside the talk worker image, where ComfyUI is installed.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

COMFY_DIR = Path(os.environ.get("MW_COMFY_DIR", "/opt/ComfyUI"))
PORT = int(os.environ.get("MW_COMFY_PORT", "8188"))
URL = f"http://127.0.0.1:{PORT}"

# Weights by ComfyUI model folder: (Hugging Face repo, file). All Apache-2.0 (checked 2026-11-22).
WEIGHTS = {
    "diffusion_models": [("Kijai/WanVideo_comfy_fp8_scaled", "I2V/Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors"),
                         ("Kijai/WanVideo_comfy_fp8_scaled",
                          "InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors")],
    "vae": [("Kijai/WanVideo_comfy", "Wan2_1_VAE_bf16.safetensors")],
    "text_encoders": [("Kijai/WanVideo_comfy", "umt5-xxl-enc-bf16.safetensors")],
    "loras": [("Kijai/WanVideo_comfy", "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors")],
}
MODEL, ADAPTER = (f for _, f in WEIGHTS["diffusion_models"])
VAE, TEXT_ENCODER, LORA = WEIGHTS["vae"][0][1], WEIGHTS["text_encoders"][0][1], WEIGHTS["loras"][0][1]
WAV2VEC = "TencentGameMate/chinese-wav2vec2-base"
NEGATIVE = ("bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
            "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
            "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still "
            "picture, messy background, three legs, many people in the background, walking backwards")


def graph(*, image: str, audio: str, prompt: str, frames: int, seed: int, params: dict, blocks_to_swap: int) -> dict:
    """The InfiniteTalk graph in ComfyUI's API format. `image` and `audio` are file names in ComfyUI's input folder;
    `frames` is the clip length at 25 fps (the audio's)."""
    p = params
    return {
        "vae": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": VAE, "precision": "bf16"}},
        "lora": {"class_type": "WanVideoLoraSelect",
                 "inputs": {"lora": LORA, "strength": float(p.get("lora_strength", 1.0)), "low_mem_load": False,
                            "merge_loras": False}},
        "swap": {"class_type": "WanVideoBlockSwap",
                 "inputs": {"blocks_to_swap": int(blocks_to_swap), "offload_img_emb": False, "offload_txt_emb": False,
                            "use_non_blocking": True, "vace_blocks_to_swap": 0, "prefetch_blocks": 1,
                            "block_swap_debug": False}},
        "adapter": {"class_type": "MultiTalkModelLoader", "inputs": {"model": ADAPTER}},
        "model": {"class_type": "WanVideoModelLoader",
                  "inputs": {"model": MODEL, "base_precision": p.get("base_precision", "fp16_fast"),
                             "quantization": p.get("quantization", "fp8_e4m3fn_scaled"), "load_device": "offload_device",
                             "attention_mode": p.get("attention", "sageattn"), "block_swap_args": ["swap", 0],
                             "lora": ["lora", 0], "multitalk_model": ["adapter", 0]}},
        "wav2vec": {"class_type": "DownloadAndLoadWav2VecModel",
                    "inputs": {"model": WAV2VEC, "base_precision": "fp16", "load_device": "main_device"}},
        "audio": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "embeds": {"class_type": "MultiTalkWav2VecEmbeds",
                   "inputs": {"wav2vec_model": ["wav2vec", 0], "audio_1": ["audio", 0], "normalize_loudness": True,
                              "num_frames": int(frames), "fps": 25, "audio_scale": float(p.get("audio_scale", 1.0)),
                              "audio_cfg_scale": float(p.get("audio_cfg", 1.0)), "multi_audio_type": "para"}},
        "text": {"class_type": "WanVideoTextEncodeCached",
                 "inputs": {"model_name": TEXT_ENCODER, "precision": "bf16", "positive_prompt": prompt,
                            "negative_prompt": p.get("negative_prompt", NEGATIVE), "quantization": "disabled",
                            "use_disk_cache": False, "device": "gpu"}},
        "image": {"class_type": "LoadImage", "inputs": {"image": image}},
        "i2v": {"class_type": "WanVideoImageToVideoMultiTalk",
                "inputs": {"vae": ["vae", 0], "width": int(p["width"]), "height": int(p["height"]),
                           "frame_window_size": int(p.get("frame_window", 81)),
                           "motion_frame": int(p.get("motion_frame", 9)), "force_offload": False,
                           "colormatch": p.get("colormatch", "disabled"), "start_image": ["image", 0],
                           "tiled_vae": False, "mode": "infinitetalk", "output_path": ""}},
        "sampler": {"class_type": "WanVideoSampler",
                    "inputs": {"model": ["model", 0], "image_embeds": ["i2v", 0], "text_embeds": ["text", 0],
                               "multitalk_embeds": ["embeds", 0], "steps": int(p["steps"]),
                               "cfg": float(p.get("cfg", 1.0)), "shift": float(p.get("shift", 11.0)),
                               "seed": int(seed) % (2**63), "force_offload": True,
                               "scheduler": p.get("scheduler", "dpm++_sde"), "riflex_freq_index": 0,
                               "denoise_strength": 1.0, "batched_cfg": False, "rope_function": "comfy",
                               "start_step": 0, "end_step": -1, "add_noise_to_samples": False}},
        "frames": {"class_type": "WanVideoPassImagesFromSamples", "inputs": {"samples": ["sampler", 0]}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["frames", 0], "fps": 25.0, "audio": ["audio", 0]}},
        "save": {"class_type": "SaveVideo",
                 "inputs": {"video": ["video", 0], "filename_prefix": "mw_talk", "format": "mp4", "codec": "h264"}},
    }


def missing_inputs(g: dict, object_info: dict) -> list[str]:
    """Every node class the graph uses exists, and every required input is given: what ComfyUI would reject."""
    problems = []
    for node_id, node in g.items():
        info = object_info.get(node["class_type"])
        if not info:
            problems.append(f"{node_id}: no node {node['class_type']}")
            continue
        for name in (info.get("input", {}).get("required") or {}):
            if name not in node["inputs"]:
                problems.append(f"{node_id} ({node['class_type']}): missing required input {name}")
        known = {**(info.get("input", {}).get("required") or {}), **(info.get("input", {}).get("optional") or {})}
        problems += [f"{node_id} ({node['class_type']}): unknown input {k}" for k in node["inputs"] if k not in known]
    return problems


# --- the server, inside the talk worker -------------------------------------------------------------------------
_server: dict = {"proc": None}


def _get(path: str, timeout: float = 10):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.loads(r.read())


def write_model_paths(models_dir: Path) -> Path:
    """ComfyUI reads model folders from extra_model_paths.yaml: point them at the weights cache (a network volume
    where the pool has one), so weights are downloaded once."""
    cfg = COMFY_DIR / "extra_model_paths.yaml"
    lines = ["mw:", f"  base_path: {models_dir}"] + [f"  {folder}: {folder}" for folder in WEIGHTS]
    cfg.write_text("\n".join(lines) + "\n")
    return cfg


def ensure_weights(models_dir: Path, weights: dict | None = None) -> None:
    from huggingface_hub import hf_hub_download
    for folder, files in (weights or WEIGHTS).items():
        for repo, name in files:
            hf_hub_download(repo, name, local_dir=str(models_dir / folder), token=os.environ.get("HF_TOKEN") or None)


def start(*, args: tuple[str, ...] = (), wait_s: float = 600) -> None:
    """Start ComfyUI once per worker and wait until it answers."""
    proc = _server["proc"]
    if proc is not None and proc.poll() is None:
        return
    log = open(Path(os.environ.get("MW_COMFY_LOG", "/tmp/comfyui.log")), "ab")
    _server["proc"] = subprocess.Popen(
        ["python", "main.py", "--listen", "127.0.0.1", "--port", str(PORT), "--disable-auto-launch", *args],
        cwd=str(COMFY_DIR), stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if _server["proc"].poll() is not None:
            raise RuntimeError(f"ComfyUI exited at start ({_server['proc'].returncode}); see its log")
        try:
            _get("/system_stats")
            return
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(2)
    raise RuntimeError(f"ComfyUI did not answer within {wait_s:.0f} s")


def stop() -> None:
    proc = _server["proc"]
    if proc is not None and proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=30)
    _server["proc"] = None


def run(g: dict, *, timeout_s: float, poll_s: float = 3) -> Path:
    """Queue the graph, wait for it, return the saved video's path."""
    req = urllib.request.Request(URL + "/prompt", data=json.dumps({"prompt": g}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            prompt_id = json.loads(r.read())["prompt_id"]
    except urllib.error.HTTPError as e:   # 400: the graph was rejected (node_errors say why)
        raise RuntimeError(f"ComfyUI rejected the graph: {e.read().decode(errors='replace')[:1500]}") from None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        h = _get(f"/history/{prompt_id}").get(prompt_id)
        if h:
            status = h.get("status") or {}
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI failed: {execution_error(status)}")
            for out in (h.get("outputs") or {}).values():
                for key in ("videos", "images", "gifs"):
                    for f in out.get(key) or []:
                        if str(f.get("filename", "")).endswith(".mp4"):
                            return COMFY_DIR / (f.get("type") or "output") / f.get("subfolder", "") / f["filename"]
            if status.get("completed"):
                raise RuntimeError(f"ComfyUI finished with no video: {json.dumps(h.get('outputs'))[:800]}")
        if _server["proc"] is not None and _server["proc"].poll() is not None:
            raise RuntimeError("ComfyUI exited while rendering (out of memory?); see its log")
        time.sleep(poll_s)
    raise RuntimeError(f"ComfyUI render not done after {timeout_s:.0f} s")


# --- story clips: Wan 2.2 I2V A14B, the two experts with the lightx2v LoRA ------------------------------------------
# Kijai's example workflow (example_workflows/wanvideo_2_2_I2V_A14B_example_WIP.json at the pinned commit): the
# high-noise expert takes the first steps with a short guidance schedule, the low-noise expert finishes; both carry
# the lightx2v distillation LoRA (strength 3 on high, 1 on low), 6 steps in all, dpm++_sde, shift 8, 81 frames at 16
# fps. fp8-scaled weights (Apache-2.0); the T5 encoder is the one the talk graph uses.
I2V_WEIGHTS = {
    "diffusion_models": [("Kijai/WanVideo_comfy_fp8_scaled", "I2V/Wan2_2-I2V-A14B-HIGH_fp8_e4m3fn_scaled_KJ.safetensors"),
                         ("Kijai/WanVideo_comfy_fp8_scaled", "I2V/Wan2_2-I2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors")],
    "vae": WEIGHTS["vae"], "text_encoders": WEIGHTS["text_encoders"], "loras": WEIGHTS["loras"],
}
HIGH, LOW = (f for _, f in I2V_WEIGHTS["diffusion_models"])


def graph_i2v(*, image: str, prompt: str, seed: int, params: dict, blocks_to_swap: int) -> dict:
    """A story clip from one approved photo, in ComfyUI's API format."""
    p = params
    steps, split = int(p.get("steps", 6)), int(p.get("high_steps", 3))

    def expert(name: str, weights: str, lora_strength: float) -> dict:
        return {
            f"{name}_model": {"class_type": "WanVideoModelLoader",
                              "inputs": {"model": weights, "base_precision": p.get("base_precision", "fp16_fast"),
                                         "quantization": p.get("quantization", "fp8_e4m3fn_scaled"),
                                         "load_device": "offload_device", "attention_mode": p.get("attention", "sageattn")}},
            f"{name}_swap": {"class_type": "WanVideoSetBlockSwap",
                             "inputs": {"model": [f"{name}_model", 0], "block_swap_args": ["swap", 0]}},
            f"{name}_lora_pick": {"class_type": "WanVideoLoraSelect",
                                  "inputs": {"lora": LORA, "strength": float(lora_strength), "low_mem_load": False,
                                             "merge_loras": False}},
            f"{name}_lora": {"class_type": "WanVideoSetLoRAs",
                             "inputs": {"model": [f"{name}_swap", 0], "lora": [f"{name}_lora_pick", 0]}},
        }

    def sampler(model: str, start: int, end: int, cfg, samples=None) -> dict:
        inputs = {"model": [model, 0], "image_embeds": ["i2v", 0], "text_embeds": ["text", 0], "steps": steps,
                  "cfg": cfg, "shift": float(p.get("shift", 8.0)), "seed": int(seed) % (2**63), "force_offload": True,
                  "scheduler": p.get("scheduler", "dpm++_sde"), "riflex_freq_index": 0, "denoise_strength": 1.0,
                  "batched_cfg": False, "rope_function": "comfy", "start_step": start, "end_step": end,
                  "add_noise_to_samples": False}
        if samples:
            inputs["samples"] = samples
        return {"class_type": "WanVideoSampler", "inputs": inputs}

    return {
        **expert("high", HIGH, p.get("high_lora_strength", 3.0)),
        **expert("low", LOW, p.get("low_lora_strength", 1.0)),
        "swap": {"class_type": "WanVideoBlockSwap",
                 "inputs": {"blocks_to_swap": int(blocks_to_swap), "offload_img_emb": False, "offload_txt_emb": False,
                            "use_non_blocking": True, "vace_blocks_to_swap": 0, "prefetch_blocks": 1,
                            "block_swap_debug": False}},
        "vae": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": VAE, "precision": "bf16"}},
        "text": {"class_type": "WanVideoTextEncodeCached",
                 "inputs": {"model_name": TEXT_ENCODER, "precision": "bf16", "positive_prompt": prompt,
                            "negative_prompt": p.get("negative_prompt", NEGATIVE), "quantization": "disabled",
                            "use_disk_cache": False, "device": "gpu"}},
        "image": {"class_type": "LoadImage", "inputs": {"image": image}},
        "i2v": {"class_type": "WanVideoImageToVideoEncode",
                "inputs": {"vae": ["vae", 0], "start_image": ["image", 0], "width": int(p["width"]),
                           "height": int(p["height"]), "num_frames": int(p.get("frames", 81)),
                           "noise_aug_strength": 0.0, "start_latent_strength": 1.0, "end_latent_strength": 1.0,
                           "force_offload": True, "fun_or_fl2v_model": False, "tiled_vae": False}},
        "cfg_schedule": {"class_type": "CreateCFGScheduleFloatList",
                         "inputs": {"steps": steps, "cfg_scale_start": float(p.get("high_cfg", 2.0)),
                                    "cfg_scale_end": float(p.get("high_cfg", 2.0)), "interpolation": "linear",
                                    "start_percent": 0.0, "end_percent": float(p.get("cfg_end_percent", 0.01))}},
        "sample_high": sampler("high_lora", 0, split, ["cfg_schedule", 0]),
        "sample_low": sampler("low_lora", split, -1, 1.0, samples=["sample_high", 0]),
        "decode": {"class_type": "WanVideoDecode",
                   "inputs": {"vae": ["vae", 0], "samples": ["sample_low", 0], "enable_vae_tiling": False,
                              "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128,
                              "normalization": "default"}},
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "fps": float(p.get("fps", 16))}},
        "save": {"class_type": "SaveVideo",
                 "inputs": {"video": ["video", 0], "filename_prefix": "mw_clip", "format": "mp4", "codec": "h264"}},
    }


def execution_error(status: dict) -> str:
    """The useful part of a ComfyUI execution error: which node, what exception, and the end of its traceback,
    without the node's input values (tensors and models, which drown the message)."""
    for kind, m in status.get("messages") or []:
        if kind != "execution_error":
            continue
        tb = "".join(m.get("traceback") or [])[-1200:]
        return (f"node {m.get('node_id')} ({m.get('node_type')}): {m.get('exception_type')}: "
                f"{str(m.get('exception_message'))[:600]} | traceback: {tb}")
    return json.dumps(status)[:800]
