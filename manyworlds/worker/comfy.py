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
import math
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


def with_compile(g: dict, p: dict, loaders: list[str]) -> dict:
    """torch.compile on the transformer blocks when `p["compile"]` (WanVideoTorchCompileSettings into each model
    loader): slower first run on a machine, faster after; adopted only where the benchmark says so."""
    if not p.get("compile"):
        return g
    g["compile"] = {"class_type": "WanVideoTorchCompileSettings",
                    "inputs": {"backend": "inductor", "fullgraph": False, "mode": p.get("compile_mode", "default"),
                               "dynamic": False, "dynamo_cache_size_limit": 64, "compile_transformer_blocks_only": True}}
    for name in loaders:
        g[name]["inputs"]["compile_args"] = ["compile", 0]
    return g


def negative_text(params: dict, negative: str | None = None) -> str:
    """The negative prompt a graph encodes: the job's own (payload `negative_prompt`, compiled from the shot sheet by
    manyworlds.direction.compile_prompts), else params.negative_prompt, else Wan's standard NEGATIVE. A blank one
    counts as none, so an undirected job keeps the standard list."""
    for text in (negative, params.get("negative_prompt")):
        if isinstance(text, str) and text.strip():
            return text.strip()
    return NEGATIVE


def graph(*, image: str, audio: str, prompt: str, frames: int, seed: int, params: dict, blocks_to_swap: int,
          negative: str | None = None) -> dict:
    """The InfiniteTalk graph (see _talk_graph), with torch.compile when the params ask for it."""
    return with_compile(_talk_graph(image=image, audio=audio, prompt=prompt, frames=frames, seed=seed, params=params,
                                    blocks_to_swap=blocks_to_swap, negative=negative), params, ["model"])


def _talk_graph(*, image: str, audio: str, prompt: str, frames: int, seed: int, params: dict, blocks_to_swap: int,
                negative: str | None = None) -> dict:
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
                            "negative_prompt": negative_text(p, negative), "quantization": "disabled",
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


# --- walk-and-talk: InfiniteTalk video-to-video over a silent motion clip --------------------------------------------
# Kijai's example (example_workflows/wanvideo_2_1_14B_V2V_InfiniteTalk_example_02.json at the pinned commit): the clip's
# frames, resized, are VAE-encoded (WanVideoEncode) into the sampler's `samples`; its first frame is the start image of
# WanVideoImageToVideoMultiTalk (infinitetalk mode, 81-frame windows, 9 motion frames) and, optionally, CLIP-vision
# encoded; the sampler runs 4 steps from step 2 with noise added to the clip's latents (cfg 1, shift 11, dpm++_sde).
# Audio longer than the clip continues from its last frame (the wrapper pads the latents with it). The example's
# VideoHelperSuite and KJNodes nodes are not in our image: core LoadVideo + GetVideoComponents load the clip, core
# ImageScale (centre crop) stands in for ImageResizeKJv2 and ImageFromBatch for GetImageRangeFromBatch. The example's
# vocal separator is left out (our voice line is clean), as in the image-to-video graph. The clip must already be at
# TALK_FPS (the handler resamples it): the audio embeds are 25 per second, and nothing in core retimes frames.
TALK_FPS = 25
CLIP_VISION_WEIGHTS = {"clip_vision": [("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                                        "split_files/clip_vision/clip_vision_h.safetensors")]}
CLIP_VISION = CLIP_VISION_WEIGHTS["clip_vision"][0][1]


def v2v_start(params: dict) -> int:
    """The video-to-video start step, checked: from 0 to below params.steps."""
    steps, start = int(params["steps"]), int(params.get("start_step", 2))
    if not 0 <= start < steps:
        raise ValueError(f"start_step {start} must be from 0 to below steps ({steps})")
    return start


def graph_v2v(*, video: str, audio: str, prompt: str, frames: int, seed: int, params: dict,
              blocks_to_swap: int, negative: str | None = None) -> dict:
    """Lip sync onto an existing clip (`video`, a file name in ComfyUI's input folder, at TALK_FPS) in ComfyUI's API
    format. `frames` is the talk length at 25 fps (the audio's). params.start_step (of params.steps) is the denoise
    strength: how much of the clip is redrawn (lower redraws more; the example: 2 of 4). params.clip_vision adds the
    example's CLIP vision embeds of the first frame."""
    p = params
    start = v2v_start(p)
    g = _talk_graph(image="", audio=audio, prompt=prompt, frames=frames, seed=seed, params=params,
                    blocks_to_swap=blocks_to_swap, negative=negative)
    del g["image"]
    w, h = int(p["width"]), int(p["height"])
    g.update({
        "motion": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "motion_frames": {"class_type": "GetVideoComponents", "inputs": {"video": ["motion", 0]}},
        "motion_resized": {"class_type": "ImageScale",
                           "inputs": {"image": ["motion_frames", 0], "upscale_method": "lanczos", "width": w,
                                      "height": h, "crop": "center"}},
        "motion_first": {"class_type": "ImageFromBatch",
                         "inputs": {"image": ["motion_resized", 0], "batch_index": 0, "length": 1}},
        "motion_latents": {"class_type": "WanVideoEncode",
                           "inputs": {"vae": ["vae", 0], "image": ["motion_resized", 0], "enable_vae_tiling": False,
                                      "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128,
                                      "noise_aug_strength": 0.0, "latent_strength": 1.0}},
    })
    g["i2v"]["inputs"]["start_image"] = ["motion_first", 0]
    if p.get("clip_vision"):
        g["clip_vision"] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP_VISION}}
        g["clip_embeds"] = {"class_type": "WanVideoClipVisionEncode",
                            "inputs": {"clip_vision": ["clip_vision", 0], "image_1": ["motion_first", 0],
                                       "strength_1": 1.0, "strength_2": 1.0, "crop": "center",
                                       "combine_embeds": "average", "force_offload": True, "tiles": 0, "ratio": 0.5}}
        g["i2v"]["inputs"]["clip_embeds"] = ["clip_embeds", 0]
    g["sampler"]["inputs"].update(samples=["motion_latents", 0], start_step=start, add_noise_to_samples=True)
    g["save"]["inputs"]["filename_prefix"] = "mw_talk_v2v"
    return with_compile(g, params, ["model"])


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


def write_model_paths(models_dir: Path, extra: tuple[str, ...] | list[str] = ()) -> Path:
    """ComfyUI reads model folders from extra_model_paths.yaml: point them at the weights cache (a network volume
    where the pool has one), so weights are downloaded once. `extra`: a named graph's folders (comfy_graphs; a
    folder with a slash is a node pack's own directory, reached by link_models instead). ComfyUI reads the file only
    at start, so a running server that lacks a folder now listed is restarted by start()."""
    cfg = COMFY_DIR / "extra_model_paths.yaml"
    folders = dict.fromkeys([*WEIGHTS, *I2V_WEIGHTS, *(f for w in LONGCAT_WEIGHTS.values() for f in w),
                             *CLIP_VISION_WEIGHTS, *(f for f in extra if "/" not in f)])
    _server["folders"] = set(folders)
    lines = ["mw:", f"  base_path: {models_dir}"] + [f"  {folder}: {folder}" for folder in folders]
    cfg.write_text("\n".join(lines) + "\n")
    return cfg


def link_models(models_dir: Path, names: list[str] | tuple[str, ...]) -> list[Path]:
    """Node packs that read their weights from ComfyUI/models/<name> itself (ComfyUI_EchoMimic: models/echo_mimic)
    get that directory as a link into the weights cache. The pack makes an empty one when ComfyUI imports it: an empty
    directory is replaced, one holding files is left alone (refused: the weights would be split in two places)."""
    import shutil
    made = []
    for name in names:
        target, link = Path(models_dir) / name, COMFY_DIR / "models" / name
        target.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() == target.resolve():
                continue
            link.unlink()
        elif link.is_dir():
            if any(p.is_file() for p in link.rglob("*")):
                raise RuntimeError(f"{link} holds files; expected a link to {target}")
            shutil.rmtree(link)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
        made.append(link)
    return made


class VramPeak:
    """The most GPU memory in use while a render runs, sampled every `every_s` from ComfyUI's /system_stats (the
    card's own free figure, so other processes on it count too). For benchmarks: {peak_gb, total_gb, samples}."""

    def __init__(self, every_s: float, stats=None):
        import threading
        self.every_s, self._stats = float(every_s), stats or (lambda: _get("/system_stats", timeout=5))
        self._stop, self._thread = threading.Event(), None
        self.peak_gb, self.total_gb, self.samples = None, None, 0

    def sample(self) -> None:
        try:
            dev = (self._stats().get("devices") or [{}])[0]
            total, free = float(dev["vram_total"]), float(dev["vram_free"])
        except Exception:  # noqa: BLE001 - a missed sample only makes the peak a lower bound
            return
        used = round((total - free) / 1e9, 2)
        self.peak_gb = used if self.peak_gb is None else max(self.peak_gb, used)
        self.total_gb, self.samples = round(total / 1e9, 1), self.samples + 1

    def _loop(self) -> None:
        while not self._stop.wait(self.every_s):
            self.sample()

    def __enter__(self) -> "VramPeak":
        import threading
        self.sample()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="vram-peak")
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.every_s + 5)
        self.sample()

    def result(self) -> dict:
        return {"peak_gb": self.peak_gb, "total_gb": self.total_gb, "samples": self.samples}


def _remote_size(repo: str, name: str, token) -> int | None:
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        return get_hf_file_metadata(hf_hub_url(repo, name), token=token).size
    except Exception:  # noqa: BLE001 - unknown size: trust a file that is there
        return None


def _volume_lock(models_dir: Path, wait_s: float = 3600, stale_s: float = 7200):
    """One downloader per weights volume: two machines writing the same files on a shared network volume broke
    huggingface_hub's download records (2026-11-22: every story clip failed with UnboundLocalError 'metadata')."""
    from . import volume
    return volume.exclusive(models_dir / ".mw-download.lock", wait_s=wait_s, stale_s=stale_s, what="weights")


def prune_other_weights(models_dir: Path, *, partial: bool = True) -> list[str]:
    """Free the weights volume of this worker's own half-finished downloads, nothing else.

    It used to delete every sibling of `models_dir` (2026-11-22: the research script's ~82 GB filled the volume), but
    the siblings are other families' weights on a shared volume (MW_MODEL_DIR/svc, /longcat, /<org>--<repo> from
    worker.models): a talk job deleted the voice and image families' weights, and the next machine downloaded them
    again (owner 2026-11-25: families must never delete one another's weights). Stray paths are now named in policy
    (gpu_pools.<pool>.network_volume.prune) and removed at boot by worker.volume, which refuses the shared cache."""
    removed = []
    for p in models_dir.rglob("*.incomplete") if partial and models_dir.exists() else []:
        p.unlink(missing_ok=True)
        removed.append(p.name)
    return removed


def ensure_weights(models_dir: Path, weights: dict | None = None, download=None, remote_size=None,
                   prune: bool = False) -> None:
    """Every weight file in place, complete (its remote size), downloaded once per volume. A failed download
    clears huggingface_hub's records for that file and the partial file, then tries once more from scratch.

    Safe on a shared volume (owner 2026-11-26): one downloader at a time (volume.exclusive); before fetching, the bytes
    still missing plus the volume's floor must fit (volume.room, else the job is released as a machine fault rather
    than dying half-written); after fetching, a file whose size differs from the remote size is deleted and the
    download fails, so a truncated file is never left in place for the next machine to load. huggingface_hub writes
    to a temporary file and renames it into place, so a reader never sees a partial file under the real name."""
    from . import volume, weights_log
    token = os.environ.get("HF_TOKEN") or None
    if download is None:
        from huggingface_hub import hf_hub_download as download
    remote_size = remote_size or _remote_size
    todo = []
    for folder, files in (weights or WEIGHTS).items():
        for repo, name in files:
            target = models_dir / folder / name
            size = remote_size(repo, name, token)
            if not (target.is_file() and (size is None or target.stat().st_size == size)):
                todo.append((folder, repo, name, size))
            else:
                weights_log.note(folder, name, "reused", target.stat().st_size / 1024 ** 3)
    if not todo:
        return
    # A full volume is caught at boot (worker.volume: free space under the floor gives the machine up), before any
    # lock file needs writing.
    with _volume_lock(models_dir):
        if prune:
            removed = prune_other_weights(models_dir)
            if removed:
                print(f"pruned from the weights volume: {', '.join(removed[:20])}", flush=True)
        need = sum(max((s or 0) - ((models_dir / f / n).stat().st_size if (models_dir / f / n).is_file() else 0), 0)
                   for f, _r, n, s in todo)
        volume.room(models_dir, need)
        for folder, repo, name, _size in todo:
            local = models_dir / folder
            target = local / name
            size = remote_size(repo, name, token)
            if target.is_file() and (size is None or target.stat().st_size == size):
                weights_log.note(folder, name, "reused", target.stat().st_size / 1024 ** 3)
                continue   # another machine finished it while we waited
            t0 = time.monotonic()
            try:
                download(repo, name, local_dir=str(local), token=token)
            except Exception:  # noqa: BLE001 - stale or broken download records: start this file over
                records = local / ".cache" / "huggingface" / "download"
                for p in records.glob(f"{name}.*") if records.exists() else []:
                    p.unlink(missing_ok=True)
                target.unlink(missing_ok=True)
                download(repo, name, local_dir=str(local), token=token, force_download=True)
            got = target.stat().st_size if target.is_file() else None
            if got is None or (size is not None and got != size):
                target.unlink(missing_ok=True)
                raise RuntimeError(f"weights download incomplete: {folder}/{name} is {got} bytes, expected {size}")
            weights_log.note(folder, name, "downloaded",
                             target.stat().st_size / 1024 ** 3 if target.is_file() else None, time.monotonic() - t0)


def start(*, args: tuple[str, ...] = (), wait_s: float = 600) -> None:
    """Start ComfyUI once per worker and wait until it answers (again, when write_model_paths has listed a model
    folder since it started: it reads them only at start)."""
    proc = _server["proc"]
    if proc is not None and proc.poll() is None:
        if not set(_server.get("folders") or ()) - set(_server.get("started_with") or ()):
            return
        stop()
    _server["started_with"] = set(_server.get("folders") or ())
    log_path = Path(os.environ.get("MW_COMFY_LOG", "/tmp/comfyui.log"))
    log = open(log_path, "ab")
    _server["proc"] = subprocess.Popen(
        ["python", "main.py", "--listen", "127.0.0.1", "--port", str(PORT), "--disable-auto-launch", *args],
        cwd=str(COMFY_DIR), stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if _server["proc"].poll() is not None:
            tail = log_path.read_bytes()[-1500:].decode(errors="replace") if log_path.exists() else ""
            raise RuntimeError(f"ComfyUI exited at start ({_server['proc'].returncode}): {tail}")
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


def free() -> bool:
    """Ask the running ComfyUI server to unload its models and free GPU memory (before another engine on the same
    machine loads: the merged test machine runs voice, motion and lip sync in turn). False when no server runs."""
    if _server.get("proc") is None or _server["proc"].poll() is not None:
        return False
    req = urllib.request.Request(URL + "/free", data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60):
        return True


def _submit(g: dict) -> str:
    req = urllib.request.Request(URL + "/prompt", data=json.dumps({"prompt": g}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["prompt_id"]
    except urllib.error.HTTPError as e:   # 400: the graph was rejected (node_errors say why)
        raise RuntimeError(f"ComfyUI rejected the graph: {e.read().decode(errors='replace')[:1500]}") from None


def saved_videos(outputs: dict) -> list[Path]:
    """Every .mp4 a finished prompt saved, by file name (a graph that saves several names them in order)."""
    found = [COMFY_DIR / (f.get("type") or "output") / f.get("subfolder", "") / f["filename"]
             for out in (outputs or {}).values() for key in ("videos", "images", "gifs") for f in out.get(key) or []
             if str(f.get("filename", "")).endswith(".mp4")]
    return sorted(found, key=lambda p: p.name)


def run_all(g: dict, *, timeout_s: float, poll_s: float = 3) -> list[Path]:
    """Queue the graph, wait until it has finished, return every saved video (LongCat saves one per window)."""
    prompt_id = _submit(g)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        h = _get(f"/history/{prompt_id}").get(prompt_id)
        if h:
            status = h.get("status") or {}
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI failed: {execution_error(status)}")
            if status.get("completed"):
                videos = saved_videos(h.get("outputs"))
                if not videos:
                    raise RuntimeError(f"ComfyUI finished with no video: {json.dumps(h.get('outputs'))[:800]}")
                return videos
        if _server["proc"] is not None and _server["proc"].poll() is not None:
            raise RuntimeError("ComfyUI exited while rendering (out of memory?); see its log")
        time.sleep(poll_s)
    raise RuntimeError(f"ComfyUI render not done after {timeout_s:.0f} s")


def run(g: dict, *, timeout_s: float, poll_s: float = 3) -> Path:
    """Queue the graph, wait for it, return the saved video's path."""
    prompt_id = _submit(g)
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


def graph_i2v(*, image: str, prompt: str, seed: int, params: dict, blocks_to_swap: int,
              negative: str | None = None) -> dict:
    """A story clip from one approved photo, in ComfyUI's API format (torch.compile when the params ask)."""
    return with_compile(_i2v_graph(image=image, prompt=prompt, seed=seed, params=params, blocks_to_swap=blocks_to_swap,
                                   negative=negative), params, ["high_model", "low_model"])


def _i2v_graph(*, image: str, prompt: str, seed: int, params: dict, blocks_to_swap: int,
               negative: str | None = None) -> dict:
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
                            "negative_prompt": negative_text(p, negative), "quantization": "disabled",
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


# --- talking video, second model: LongCat-Video-Avatar (MIT) ---------------------------------------------------------
# Kijai's example (example_workflows/LongCatAvatar_audio_image_to_video_example_01.json at the pinned commit): the
# avatar model with the LongCat distillation LoRA (strength 0.9), bf16 base precision ("LongCat models only run with
# bf16 base precision"), longcat_distill_euler at 12 steps / shift 12 and cfg 1 via WanVideoSchedulerv2 +
# WanVideoSamplerv2, 93-frame windows overlapping by 13. The first window starts from the photo's latent (overlap 1,
# no reference latent); each later one continues from the previous window's latents, with the photo's latent as the
# reference (WanVideoLongCatAvatarExtendEmbeds, frames_processed = frames made so far). The example's note says
# SageAttention 1.0.6 (our image) does not work with LongCat, so attention is sdpa unless params say otherwise.
#
# Left out, as in the InfiniteTalk graph: the vocal separator (our audio is a clean voice line), audio trimming, and
# the editor-only nodes. The example's KJNodes / VideoHelperSuite nodes (constants, resize, frame-range and
# overlap-join helpers, VHS_VideoCombine) are not in our image: the photo is cover-cropped by the handler, frame counts
# are computed here, and each window's new frames (its first `overlap` frames dropped with core ImageFromBatch) are
# saved as their own clip by CreateVideo/SaveVideo, joined with the voice by the handler. Saving per window also keeps
# ComfyUI from holding every growing concatenation of a one-minute take in memory.
#
# Versions (params.longcat_version):
#   1.0  Kijai/LongCat-Video_comfy Avatar fp8-scaled (the example's model; wav2vec audio, MultiTalkWav2VecEmbeds at
#        32 audio frames/s, stride 2: video at 16 fps). The default.
#   1.5  Whisper-large-v3 audio (LongCatAvatarWhisperEmbeds, 25 fps, stride 1; ExtendEmbeds re-encodes the overlap
#        from the previous window's frames, the v1.5 path). No Kijai or Comfy-Org 1.5 checkpoint in the wrapper's key
#        format exists (2026-11-22: meituan's 1.5 is sharded original-format weights and its DMD LoRA uses keys the
#        wrapper does not convert); the only one found is a community GGUF conversion, and without 1.5's own
#        distillation LoRA it is untested. Benchmark before use (talk profile longcat15).
LONGCAT_WEIGHTS = {
    "1.0": {"diffusion_models": [("Kijai/LongCat-Video_comfy",
                                  "Avatar/LongCat-Avatar-single_fp8_e4m3fn_scaled_mixed_KJ.safetensors")],
            "loras": [("Kijai/LongCat-Video_comfy", "LongCat_distill_lora_alpha64_bf16.safetensors")],
            "vae": WEIGHTS["vae"], "text_encoders": WEIGHTS["text_encoders"]},
    "1.5": {"diffusion_models": [("vantagewithai/LongCat-Video-Avatar-1.5-GGUF-ComfyUI",
                                  "LongCat-Avatar-15_comfy-Q8_0.gguf")],
            "audio_encoders": [("Comfy-Org/LongCat-Video-Avatar", "audio_encoders/whisper_large_v3_encoder_fp16.safetensors")],
            "loras": [("Kijai/LongCat-Video_comfy", "LongCat_distill_lora_alpha64_bf16.safetensors")],
            "vae": WEIGHTS["vae"], "text_encoders": WEIGHTS["text_encoders"]},
}
WEIGHTS_LONGCAT = LONGCAT_WEIGHTS["1.0"]
LONGCAT_FPS = {"1.0": 16, "1.5": 25}          # the model's video rate (v1.0: audio stride 2 at 32 audio frames/s)
LONGCAT_QUANTIZATION = {"1.0": "fp8_e4m3fn_scaled", "1.5": "disabled"}   # a GGUF model must load with "disabled"


def longcat_version(params: dict) -> str:
    v = str(params.get("longcat_version", "1.0"))
    if v not in LONGCAT_WEIGHTS:
        raise ValueError(f"unknown LongCat version {v!r}; known: {sorted(LONGCAT_WEIGHTS)}")
    return v


def longcat_windows(frames: int, window: int, overlap: int) -> int:
    """How many windows make `frames` frames: the first gives `window`, each later one `window - overlap` more."""
    if frames <= window:
        return 1
    return 1 + math.ceil((frames - window) / (window - overlap))


def graph_longcat(*, image: str, audio: str, prompt: str, frames: int, seed: int, params: dict,
                  blocks_to_swap: int, negative: str | None = None) -> dict:
    """The LongCat-Video-Avatar graph in ComfyUI's API format (torch.compile when the params ask). `frames` is the
    clip length at the model's frame rate (LONGCAT_FPS); one SaveVideo per window, prefixed mw_longcat_wNN."""
    return with_compile(_longcat_graph(image=image, audio=audio, prompt=prompt, frames=frames, seed=seed,
                                       params=params, blocks_to_swap=blocks_to_swap, negative=negative),
                        params, ["model"])


def _longcat_graph(*, image: str, audio: str, prompt: str, frames: int, seed: int, params: dict,
                   blocks_to_swap: int, negative: str | None = None) -> dict:
    p, v = params, longcat_version(params)
    weights = LONGCAT_WEIGHTS[v]
    window, overlap = int(p.get("frame_window", 93)), int(p.get("overlap", 13))
    if not 0 < overlap < window:
        raise ValueError(f"overlap {overlap} must be between 0 and the window ({window})")
    fps = LONGCAT_FPS[v]
    n = longcat_windows(int(frames), window, overlap)
    tiles = {"enable_vae_tiling": False, "tile_x": 272, "tile_y": 272, "tile_stride_x": 144, "tile_stride_y": 128}
    # vae_file: another Wan 2.1 VAE in the vae folder (talk profile longcat_fast: lightx2v's 75%-pruned LightVAE,
    # which WanVideoVAELoader detects by its width); merge_loras: the distill LoRA merged into the weights;
    # text_device cpu: UMT5 encodes on the CPU, leaving the card to the DiT.
    g = {
        "vae": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": p.get("vae_file") or VAE,
                                                              "precision": "bf16"}},
        "lora": {"class_type": "WanVideoLoraSelect",
                 "inputs": {"lora": weights["loras"][0][1], "strength": float(p.get("lora_strength", 0.9)),
                            "low_mem_load": False, "merge_loras": bool(p.get("merge_loras", False))}},
        "swap": {"class_type": "WanVideoBlockSwap",
                 "inputs": {"blocks_to_swap": int(blocks_to_swap), "offload_img_emb": False, "offload_txt_emb": False,
                            "use_non_blocking": True, "vace_blocks_to_swap": 0, "prefetch_blocks": 1,
                            "block_swap_debug": False}},
        "model": {"class_type": "WanVideoModelLoader",
                  "inputs": {"model": weights["diffusion_models"][0][1], "base_precision": "bf16",
                             "quantization": p.get("longcat_quantization", LONGCAT_QUANTIZATION[v]),
                             "load_device": "offload_device", "attention_mode": p.get("attention", "sdpa"),
                             "block_swap_args": ["swap", 0], "lora": ["lora", 0]}},
        "text": {"class_type": "WanVideoTextEncodeCached",
                 "inputs": {"model_name": TEXT_ENCODER, "precision": "bf16", "positive_prompt": prompt,
                            "negative_prompt": negative_text(p, negative), "quantization": "disabled",
                            "use_disk_cache": False, "device": p.get("text_device", "gpu")}},
        "image": {"class_type": "LoadImage", "inputs": {"image": image}},
        "ref": {"class_type": "WanVideoEncode",
                "inputs": {"vae": ["vae", 0], "image": ["image", 0], **tiles, "noise_aug_strength": 0.0,
                           "latent_strength": 1.0}},
        "audio": {"class_type": "LoadAudio", "inputs": {"audio": audio}},
        "scheduler": {"class_type": "WanVideoSchedulerv2",
                      "inputs": {"scheduler": p.get("scheduler", "longcat_distill_euler"), "steps": int(p["steps"]),
                                 "shift": float(p.get("shift", 12.0)), "start_step": 0, "end_step": -1,
                                 "enhance_hf": False}},
    }
    embeds = {"audio_1": ["audio", 0], "normalize_loudness": True, "audio_scale": float(p.get("audio_scale", 1.0)),
              "audio_cfg_scale": float(p.get("audio_cfg", 1.0)), "multi_audio_type": "para"}
    if v == "1.0":
        g["wav2vec"] = {"class_type": "DownloadAndLoadWav2VecModel",
                        "inputs": {"model": WAV2VEC, "base_precision": "fp16", "load_device": "main_device"}}
        g["embeds"] = {"class_type": "MultiTalkWav2VecEmbeds",   # audio stride 2: two audio frames per video frame
                       "inputs": {"wav2vec_model": ["wav2vec", 0], **embeds, "num_frames": 2 * int(frames),
                                  "fps": float(2 * fps)}}
    else:
        g["whisper"] = {"class_type": "WhisperModelLoader",
                        "inputs": {"model": weights["audio_encoders"][0][1], "base_precision": "fp16",
                                   "load_device": "main_device"}}
        g["embeds"] = {"class_type": "LongCatAvatarWhisperEmbeds",
                       "inputs": {"whisper_model": ["whisper", 0], **embeds, "num_frames": int(frames),
                                  "fps": float(fps)}}
    for k in range(n):
        first = k == 0
        ext = {"prev_latents": ["ref", 0] if first else [f"sample{k - 1}", 0], "audio_embeds": ["embeds", 0],
               "num_frames": window, "overlap": 1 if first else overlap,
               "frames_processed": 0 if first else window + (window - overlap) * (k - 1),
               "if_not_enough_audio": "pad_with_start", "ref_frame_index": int(p.get("ref_frame_index", 10)),
               "ref_mask_frame_range": int(p.get("ref_mask_frame_range", 3))}
        if not first:
            ext["ref_latent"] = ["ref", 0]
            if v == "1.5":   # v1.5 re-encodes the overlap from the previous window's frames
                ext.update(prev_images=[f"decode{k - 1}", 0], vae=["vae", 0])
        g[f"ext{k}"] = {"class_type": "WanVideoLongCatAvatarExtendEmbeds", "inputs": ext}
        g[f"sample{k}"] = {"class_type": "WanVideoSamplerv2",
                           "inputs": {"model": ["model", 0], "image_embeds": [f"ext{k}", 0],
                                      "cfg": float(p.get("cfg", 1.0)), "seed": int(seed) % (2**63),
                                      "force_offload": True, "scheduler": ["scheduler", 0],
                                      "text_embeds": ["text", 0], "add_noise_to_samples": False}}
        g[f"decode{k}"] = {"class_type": "WanVideoDecode",
                           "inputs": {"vae": ["vae", 0], "samples": [f"sample{k}", 0], **tiles,
                                      "normalization": "default"}}
        frames_in = [f"decode{k}", 0]
        if not first:   # the first `overlap` frames repeat the end of the previous window
            g[f"new{k}"] = {"class_type": "ImageFromBatch",
                            "inputs": {"image": [f"decode{k}", 0], "batch_index": overlap, "length": window}}
            frames_in = [f"new{k}", 0]
        g[f"video{k}"] = {"class_type": "CreateVideo", "inputs": {"images": frames_in, "fps": float(fps)}}
        g[f"save{k}"] = {"class_type": "SaveVideo",
                         "inputs": {"video": [f"video{k}", 0], "filename_prefix": f"mw_longcat_w{k:02d}",
                                    "format": "mp4", "codec": "h264"}}
    return g


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
