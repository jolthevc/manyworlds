"""ACE-Step 1.5 songs in ComfyUI core (job kind music.song, handlers/music_song.py), on the ComfyUI worker image.

ComfyUI ships ACE-Step 1.5 natively (comfy_extras/nodes_ace.py: TextEncodeAceStepAudio1.5, EmptyAceStep1.5LatentAudio;
VAEDecodeAudio and SaveAudio in nodes_audio.py), so songs run on the same engine and weights volume as the talk and
story-clip graphs. `graph_song` follows Comfy-Org's split template (workflow_templates audio_ace_step_1_5_split.json,
read 2026-11-23) in ComfyUI's API format: UNETLoader -> ModelSamplingAuraFlow (shift 3) -> KSampler (8 steps, cfg 1,
euler, simple); DualCLIPLoader (type ace: the 0.6B text encoder and the 1.7B LM that plans the audio codes) ->
TextEncodeAceStepAudio1.5 (tags, lyrics, seed, bpm, duration, time signature, language, key); the negative is that
conditioning zeroed (ConditioningZeroOut); EmptyAceStep1.5LatentAudio (48 kHz latents); VAEDecodeAudio; SaveAudio
(FLAC). Every file name and setting comes from the job's `engine` (policy music.engines.ace_step_1_5). `graph_song`
is pure (tested in CI); the smoke test checks it against ComfyUI's own /object_info.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import comfy

AUDIO_SUFFIXES = (".flac", ".wav", ".mp3", ".opus", ".ogg")


def weights(engine: dict) -> dict:
    """The engine's files by ComfyUI model folder, for comfy.ensure_weights: {folder: [(repo, file)]}."""
    repo = str(engine["repo"])
    model = engine.get("diffusion_model")
    if not model:
        raise ValueError(f"music engine variant {engine.get('variant')!r} has no diffusion_model file")
    return {"diffusion_models": [(repo, str(model))],
            "text_encoders": [(repo, str(f)) for f in engine.get("text_encoders") or []],
            "vae": [(repo, str(engine["vae"]))]}


def graph_song(*, tags: str, lyrics: str, seconds: float, seed: int, bpm: int, key: str, time_signature: str,
               language: str, engine: dict, prefix: str = "mw_song") -> dict:
    """One song in ComfyUI's API format (see the module docstring)."""
    encoders = list(engine.get("text_encoders") or [])
    if len(encoders) != 2:
        raise ValueError("ACE-Step 1.5 needs two text encoders (the text encoder and the LM)")
    lm = engine.get("lm") or {}
    seed = int(seed) % (2**63)
    return {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": str(engine["diffusion_model"]), "weight_dtype": "default"}},
        "clip": {"class_type": "DualCLIPLoader",
                 "inputs": {"clip_name1": encoders[0], "clip_name2": encoders[1], "type": "ace"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": str(engine["vae"])}},
        "sampling": {"class_type": "ModelSamplingAuraFlow",
                     "inputs": {"model": ["unet", 0], "shift": float(engine.get("shift", 3.0))}},
        "text": {"class_type": "TextEncodeAceStepAudio1.5",
                 "inputs": {"clip": ["clip", 0], "tags": tags, "lyrics": lyrics, "seed": seed, "bpm": int(bpm),
                            "duration": float(seconds), "timesignature": str(time_signature),
                            "language": str(language), "keyscale": str(key),
                            "generate_audio_codes": bool(lm.get("generate_audio_codes", True)),
                            "cfg_scale": float(lm.get("cfg_scale", 2.0)),
                            "temperature": float(lm.get("temperature", 0.85)), "top_p": float(lm.get("top_p", 0.9)),
                            "top_k": int(lm.get("top_k", 0)), "min_p": float(lm.get("min_p", 0.0))}},
        "negative": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["text", 0]}},
        "latent": {"class_type": "EmptyAceStep1.5LatentAudio",
                   "inputs": {"seconds": float(seconds), "batch_size": 1}},
        "sampler": {"class_type": "KSampler",
                    "inputs": {"model": ["sampling", 0], "seed": seed, "steps": int(engine.get("steps", 8)),
                               "cfg": float(engine.get("cfg", 1.0)), "sampler_name": str(engine.get("sampler", "euler")),
                               "scheduler": str(engine.get("scheduler", "simple")), "positive": ["text", 0],
                               "negative": ["negative", 0], "latent_image": ["latent", 0], "denoise": 1.0}},
        "decode": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveAudio", "inputs": {"audio": ["decode", 0], "filename_prefix": f"audio/{prefix}"}},
    }


def saved_audio(outputs: dict) -> list[Path]:
    """Every audio file a finished prompt SAVED (type `output`), by name, whatever key the save node reports it under.
    A loader's preview of its input (type `input`) or a temp preview is never one (the 2026-11-26 video picker bug)."""
    found = [comfy.COMFY_DIR / "output" / f.get("subfolder", "") / f["filename"]
             for out in (outputs or {}).values() for files in out.values() if isinstance(files, list)
             for f in files if isinstance(f, dict) and str(f.get("filename", "")).lower().endswith(AUDIO_SUFFIXES)
             and (f.get("type") or "output") == "output"]
    return sorted(found, key=lambda p: p.name)


def run_audio(g: dict, *, timeout_s: float, poll_s: float = 2) -> Path:
    """Queue the graph, wait until it has COMPLETED, return the saved audio file, checked to be a new render
    (comfy.verified)."""
    prompt_id, h = comfy._finished(g, timeout_s=timeout_s, poll_s=poll_s)
    files = saved_audio(h.get("outputs"))
    if not files:
        raise RuntimeError(f"ComfyUI finished with no audio: {json.dumps(h.get('outputs'))[:800]}")
    return comfy.verified(files[:1], g, prompt_id, h.get("status") or {})[0]
