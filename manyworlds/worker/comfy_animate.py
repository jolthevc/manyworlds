"""Dance and motion transfer: Wan-Animate-2 in ComfyUI CORE (not WanVideoWrapper). OFF (policy `dance`).

Plan docs/superpowers/plans/2026-11-22-dance-motion-transfer.md. Wan2.2-Animate-2-14B is Apache-2.0 (HF card). Its
ComfyUI nodes, `WanAnimate2ToVideo` and `WanAnimate2Cache`, came in with ComfyUI PR #15362 (merge a464ac3,
2026-08-07), which our pinned core v0.45.0 contains (GitHub compare a464ac3...v0.45.0: 598 ahead, 0 behind; read
2026-11-22). Node inputs below are as defined in comfy_extras/nodes_wan.py at v0.45.0.

The graph follows the official template `video_wan_animate2` (Comfy-Org/workflow_templates, read 2026-11-22),
flattened to ComfyUI's API format for ONE window (up to 81 frames): the template's loop, context windows, the
side-by-side preview stitch and its resize helper are left out, and core ImageScale (centre crop) stands in for
ResizeImageMaskNode. As in the template: the int8 checkpoint plus the lightx2v 4-step distill LoRA, the CLIP text
encoder on `wan`, CLIP vision of the reference and of the driving clip's first frame, an empty pose prompt, then
WanAnimate2Cache -> ModelSamplingSD3 (shift 5) -> SamplerCustom (lcm, cfg 1, simple schedule, 6 steps), the
reference latent trimmed (TrimVideoLatent by the node's trim_latent) before decoding. The template computes the
sigmas (BasicScheduler) from the cached model before the shift is applied; that is kept as found.

The output takes the driving clip's frame rate and NEVER its audio: a reference dance's sound is not ours to keep
(trend audio is added in-app from the platform's licensed library, if at all). Only clips that pass
manyworlds.dance.reference (ours or licensed, never a downloaded trend video) may be the driving video. `graph` is
pure (tested in CI); there is no handler yet (plan: worker image, benchmark).
"""

from __future__ import annotations

from .comfy import NEGATIVE   # Wan's standard negative prompt (the template has the same list in Chinese)

REPO = "Comfy-Org/Wan-Animate-2"    # files listed from the HF API, 2026-11-22
WEIGHTS = {
    "diffusion_models": [(REPO, "diffusion_models/wan_animate_2_int8_convrot.safetensors")],
    "loras": [(REPO, "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors")],
    "text_encoders": [(REPO, "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors")],
    "clip_vision": [(REPO, "clip_vision/clip_vision_h.safetensors")],
    "vae": [(REPO, "vae/Wan2_1_VAE_bf16.safetensors")],
}
MODEL = WEIGHTS["diffusion_models"][0][1]
LORA = WEIGHTS["loras"][0][1]
TEXT_ENCODER = WEIGHTS["text_encoders"][0][1]
CLIP_VISION = WEIGHTS["clip_vision"][0][1]
VAE = WEIGHTS["vae"][0][1]
MAX_WINDOW = 81


def check_length(frames: int) -> int:
    """WanAnimate2ToVideo's length steps by 4 from 1 (1, 5, ..., 81): one window here."""
    frames = int(frames)
    if not 1 <= frames <= MAX_WINDOW or (frames - 1) % 4:
        raise ValueError(f"length {frames} must be 4k+1 from 1 to {MAX_WINDOW}")
    return frames


def graph(*, image: str, video: str, prompt: str, frames: int, seed: int, params: dict) -> dict:
    """Animate the creator in `image` with the motion of `video` (file names in ComfyUI's input folder), in ComfyUI's
    API format. params (policy dance.generation): width, height, steps, shift, sampler, scheduler, cfg,
    lora_strength, pose_strength, pose_start_percent, pose_end_percent, reference_image_strength, pose_prompt,
    negative_prompt, cache {enabled, device, dtype}."""
    p = params
    w, h = int(p["width"]), int(p["height"])
    if w % 16 or h % 16:
        raise ValueError(f"width and height must be multiples of 16, got {w}x{h}")
    length = check_length(frames)
    cache = p.get("cache") or {}
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "lora": {"class_type": "LoraLoaderModelOnly",
                 "inputs": {"model": ["unet", 0], "lora_name": LORA,
                            "strength_model": float(p.get("lora_strength", 1.0))}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "wan", "device": "default"}},
        "clip_vision": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP_VISION}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "positive": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["clip", 0]}},
        "negative": {"class_type": "CLIPTextEncode",
                     "inputs": {"text": p.get("negative_prompt") or NEGATIVE, "clip": ["clip", 0]}},
        "pose_text": {"class_type": "CLIPTextEncode",
                      "inputs": {"text": str(p.get("pose_prompt", "")), "clip": ["clip", 0]}},
        "ref_image": {"class_type": "LoadImage", "inputs": {"image": image}},
        "ref": {"class_type": "ImageScale",
                "inputs": {"image": ["ref_image", 0], "upscale_method": "area", "width": w, "height": h,
                           "crop": "center"}},
        "drive": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "drive_frames": {"class_type": "GetVideoComponents", "inputs": {"video": ["drive", 0]}},
        "drive_scaled": {"class_type": "ImageScale",
                         "inputs": {"image": ["drive_frames", 0], "upscale_method": "area", "width": w, "height": h,
                                    "crop": "center"}},
        "drive_first": {"class_type": "ImageFromBatch",
                        "inputs": {"image": ["drive_scaled", 0], "batch_index": 0, "length": 1}},
        "ref_vision": {"class_type": "CLIPVisionEncode",
                       "inputs": {"clip_vision": ["clip_vision", 0], "image": ["ref", 0], "crop": "none"}},
        "pose_vision": {"class_type": "CLIPVisionEncode",
                        "inputs": {"clip_vision": ["clip_vision", 0], "image": ["drive_first", 0], "crop": "none"}},
        "animate": {"class_type": "WanAnimate2ToVideo",
                    "inputs": {"positive": ["positive", 0], "negative": ["negative", 0], "vae": ["vae", 0],
                               "width": w, "height": h, "length": length, "batch_size": 1,
                               "reference_image": ["ref", 0], "pose_video": ["drive_scaled", 0],
                               "clip_vision_output": ["ref_vision", 0], "positive_pose": ["pose_text", 0],
                               "clip_vision_output_pose": ["pose_vision", 0], "video_frame_offset": 0,
                               "pose_strength": float(p.get("pose_strength", 1.0)),
                               "pose_start_percent": float(p.get("pose_start_percent", 0.0)),
                               "pose_end_percent": float(p.get("pose_end_percent", 1.0)),
                               "reference_image_strength": float(p.get("reference_image_strength", 1.0))}},
    }
    model = ["lora", 0]
    if cache.get("enabled", True):
        g["cache"] = {"class_type": "WanAnimate2Cache",
                      "inputs": {"model": model, "device": str(cache.get("device", "cpu")),
                                 "dtype": str(cache.get("dtype", "int8"))}}
        model = ["cache", 0]
    g.update({
        "shift": {"class_type": "ModelSamplingSD3", "inputs": {"model": model, "shift": float(p.get("shift", 5.0))}},
        "sigmas": {"class_type": "BasicScheduler",
                   "inputs": {"model": model, "scheduler": str(p.get("scheduler", "simple")),
                              "steps": int(p.get("steps", 6)), "denoise": 1.0}},
        "sampler_pick": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": str(p.get("sampler", "lcm"))}},
        "sample": {"class_type": "SamplerCustom",
                   "inputs": {"model": ["shift", 0], "add_noise": True, "noise_seed": int(seed) % (2**63),
                              "cfg": float(p.get("cfg", 1.0)), "positive": ["animate", 0],
                              "negative": ["animate", 1], "sampler": ["sampler_pick", 0], "sigmas": ["sigmas", 0],
                              "latent_image": ["animate", 2]}},
        "trim": {"class_type": "TrimVideoLatent", "inputs": {"samples": ["sample", 0], "trim_amount": ["animate", 3]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["trim", 0], "vae": ["vae", 0]}},
        # the driving clip's frame rate, never its audio
        "video": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "fps": ["drive_frames", 2]}},
        "save": {"class_type": "SaveVideo",
                 "inputs": {"video": ["video", 0], "filename_prefix": "mw_dance", "format": "mp4", "codec": "h264"}},
    })
    return g
