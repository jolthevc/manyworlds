"""Generated sound for the mix (manyworlds.sound): effects and room tone, and music beds. Both OFF (policy `sound`).

  sound.sfx    Stable Audio 3 Small-SFX (stabilityai/stable-audio-3-small-sfx, 433M, runs on CPU or GPU). Code MIT
               (github.com/Stability-AI/stable-audio-3); WEIGHTS under the Stability AI Community License: free
               commercial use only "with less than $1M in annual revenue" (stability.ai/license, read 2026-11-22),
               an Enterprise licence above that. The HF repo is gated (accept on HF) and its gate also binds the
               Gemma Terms of Use for a bundled component. So this handler refuses unless the policy baked into
               the worker says `sound.sfx.enabled: true` AND `sound.sfx.licence_accepted: true`, which only the
               owner sets after reading both licences. A payload can never switch it on.
               API (read from stable_audio_3/model.py at 790793b, 2026-11-22):
               StableAudioModel.from_pretrained("small-sfx").generate(prompt=, negative_prompt=, duration=, steps=,
               cfg_scale=, seed=, batch_size=) -> float32 tensor [batch, channels, samples] in [-1, 1], at
               model.model.sample_rate.
  sound.music  ACE-Step 1.5 (github.com/ace-step/ACE-Step-1.5, MIT; HF card ACE-Step/acestep-v15-base:
               "license: mit" and "You can strictly use the generated music for commercial purposes", read
               2026-11-22). Instrumental beds only. Refuses unless `sound.music.enabled: true`.
               API (docs/en/INFERENCE.md at ca1e85f, 2026-11-22): AceStepHandler().initialize_service(project_root=,
               config_path="acestep-v15-turbo", device=), LLMHandler(), generate_music(dit, llm, GenerationParams(
               caption=, instrumental=True, duration=, seed=, thinking=), GenerationConfig(batch_size=, audio_format=,
               use_random_seed=False, seeds=[...]), save_dir=) -> result.success, result.audios[i]["path"].
               UNVERIFIED (check on the first benchmark): that an uninitialised LLMHandler is accepted when
               thinking=False, and where initialize_service downloads checkpoints (the README says "Models are
               downloaded automatically on first run" for the Gradio app).

Neither model is in a worker image yet, and neither kind is in policy `job_kinds`, so nothing can claim these jobs.
Payload: prompt (sfx) or caption (music), seconds, seed, n; for music also `format` (a silent format: its caption
then defaults to `sound.music.bed_captions`), `formats` and `tag`. Outputs are 48 kHz-or-native WAV files ("audio");
each music output's meta carries its `library_entry` (manyworlds.sound catalogue schema, licence own-generated), so a
bed that passes a listen joins `sound.library.entries` with its provenance. Room tone needs no generator: it is
synthesised by ffmpeg on the control plane (manyworlds.sound.room_tone).
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Callable

from ...policy import Policy
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

_state: dict[str, Any] = {}
LICENCE_GATED = ("sfx",)      # revenue-capped weights: never on without the owner's explicit acceptance


def _policy(policy: Policy | None) -> Policy:
    if policy is not None:
        return policy
    from ... import policy as policy_mod
    return policy_mod.load()


def _gate(policy: Policy, name: str) -> dict:
    """The generator's policy block, or NonRetryable when it is off (or, for sfx, its licence is not accepted)."""
    c = (policy.get("sound", name, default={}) or {})
    if not c.get("enabled"):
        raise NonRetryable(f"sound.{name} is off (policy sound.{name}.enabled)")
    if (name in LICENCE_GATED or "licence_accepted" in c) and c.get("licence_accepted") is not True:
        raise NonRetryable(f"sound.{name}: its licence is not accepted (policy sound.{name}.licence_accepted); "
                           "the owner reads the licence and sets it")
    return c


def _seconds(payload: dict, c: dict) -> float:
    seconds = float(payload.get("seconds") or c.get("default_seconds", 10))
    if not 0 < seconds <= float(c.get("max_seconds", 30)):
        raise NonRetryable(f"seconds {seconds} outside (0, {c.get('max_seconds', 30)}]")
    return seconds


def write_wav(path: Path, audio, sample_rate: int) -> Path:
    """[channels, samples] (or [samples]) floats in [-1, 1] as 16-bit PCM WAV (stdlib wave: no soundfile needed)."""
    import numpy as np
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"expected [channels, samples], got shape {a.shape}")
    pcm = (np.clip(a, -1.0, 1.0) * 32767.0).round().astype("<i2").T.copy()   # interleave: [samples, channels]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(a.shape[0])
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return path


# --- sound.sfx: Stable Audio 3 Small-SFX ------------------------------------------------------------------------

def _stable_audio(model_id: str):
    if model_id not in _state:
        from stable_audio_3 import StableAudioModel
        _state[model_id] = StableAudioModel.from_pretrained(model_id)
    return _state[model_id]


def sfx(payload: dict, ctx: JobContext, *, policy: Policy | None = None,
        load_model: Callable[[str], Any] | None = None) -> HandlerResult:
    c = _gate(_policy(policy), "sfx")
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise NonRetryable("payload has no prompt")
    seconds = _seconds(payload, c)
    n = max(1, min(int(payload.get("n", 1)), int(c.get("max_batch", 4))))
    seed = int(payload.get("seed", 0)) % 2147483647
    model_id = str(c.get("model", "small-sfx"))
    model = (load_model or _stable_audio)(model_id)
    audio = model.generate(prompt=prompt, negative_prompt=c.get("negative_prompt") or None, duration=seconds,
                           steps=int(c.get("steps", 8)), cfg_scale=float(c.get("cfg_scale", 1.0)), seed=seed,
                           batch_size=n)
    sr = int(model.model.sample_rate)
    outs = []
    for i in range(n):
        path = write_wav(Path(ctx.workdir) / f"sfx_{i}.wav", audio[i], sr)
        outs.append(OutputFile(path, "audio", {"seconds": seconds, "index": i, "licence": c.get("licence_name"),
                                                "sound": "sfx"}))
    recipe = {"model": c.get("repo"), "model_id": model_id, "code_ref": c.get("code_ref"), "prompt": prompt,
              "seconds": seconds, "seed": seed, "steps": int(c.get("steps", 8)), "licence": c.get("licence_name")}
    return HandlerResult(outs, recipe=recipe, meta={"output_seconds": seconds * n})


# --- sound.music: ACE-Step 1.5 ----------------------------------------------------------------------------------

def _ace_engine(c: dict) -> Callable[..., list[Path]]:
    """generate(caption, seconds, seeds, save_dir) -> paths, via ACE-Step's documented Python API."""
    if "ace" not in _state:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        dit = AceStepHandler()
        dit.initialize_service(project_root=str(c.get("project_root", "/opt/acestep")),
                               config_path=str(c.get("config_path", "acestep-v15-turbo")), device="cuda")
        llm = LLMHandler()
        if c.get("thinking"):
            llm.initialize(checkpoint_dir=str(c.get("checkpoint_dir", "/opt/acestep/checkpoints")),
                           lm_model_path=str(c.get("lm_model", "acestep-5Hz-lm-0.6B")),
                           backend=str(c.get("lm_backend", "vllm")), device="cuda")
        _state["ace"] = (dit, llm)
    dit, llm = _state["ace"]

    def generate(caption: str, seconds: float, seeds: list[int], save_dir: Path) -> list[Path]:
        from acestep.inference import GenerationConfig, GenerationParams, generate_music
        params = GenerationParams(caption=caption, lyrics="[Instrumental]", instrumental=True, duration=seconds,
                                  thinking=bool(c.get("thinking", False)),
                                  inference_steps=int(c.get("steps", 8)))
        config = GenerationConfig(batch_size=len(seeds), use_random_seed=False, seeds=list(seeds),
                                  audio_format=str(c.get("audio_format", "wav")))
        result = generate_music(dit, llm, params, config, save_dir=str(save_dir))
        if not result.success:
            raise RuntimeError(f"ACE-Step failed: {result.error}")
        return [Path(a["path"]) for a in result.audios]
    return generate


def bed_caption(payload: dict, c: dict) -> str:
    """The payload's caption, else the policy's quiet-bed caption for its format (`sound.music.bed_captions`)."""
    caption = (payload.get("caption") or payload.get("prompt") or "").strip()
    if not caption and payload.get("format"):
        caption = str((c.get("bed_captions") or {}).get(str(payload["format"])) or "").strip()
    return caption


def library_entry(c: dict, *, caption: str, seed: int, formats: list[str], tag: str | None = None) -> dict:
    """The catalogue entry (manyworlds.sound library schema) a generated bed carries: our own generated output
    (`sound.music.library_licence`), its source the model, code ref and seed, so it can be traced and regenerated.
    The uri is filled in by whoever stores the output."""
    return {"tag": (tag or caption)[:80], "kind": "music", "formats": list(formats),
            "licence": str(c.get("library_licence") or "own-generated"),
            "source": f"ACE-Step 1.5 ({c.get('repo')}@{c.get('code_ref')}) seed {seed}: {caption}",
            "credit": ""}


def music(payload: dict, ctx: JobContext, *, policy: Policy | None = None,
          engine: Callable[..., list[Path]] | None = None) -> HandlerResult:
    c = _gate(_policy(policy), "music")
    caption = bed_caption(payload, c)
    if not caption:
        raise NonRetryable("payload has no caption (and no format with a sound.music.bed_captions entry)")
    seconds = _seconds(payload, c)
    n = max(1, min(int(payload.get("n", 1)), int(c.get("max_batch", 2))))
    seed = int(payload.get("seed", 0)) % 2147483647
    formats = [str(f) for f in (payload.get("formats") or ([payload["format"]] if payload.get("format") else []))]
    paths = (engine or _ace_engine(c))(caption, seconds, [seed + i for i in range(n)], Path(ctx.workdir))
    if not paths:
        raise RuntimeError("ACE-Step returned no audio")
    outs = [OutputFile(Path(p), "audio", {"seconds": seconds, "index": i, "licence": c.get("licence_name"),
                                          "sound": "music",
                                          "library_entry": library_entry(c, caption=caption, seed=seed + i,
                                                                         formats=formats, tag=payload.get("tag"))})
            for i, p in enumerate(paths)]
    recipe = {"model": c.get("repo"), "config_path": c.get("config_path"), "code_ref": c.get("code_ref"),
              "caption": caption, "seconds": seconds, "seed": seed, "licence": c.get("licence_name")}
    return HandlerResult(outs, recipe=recipe, meta={"output_seconds": seconds * len(outs)})


HANDLERS = {"sound.sfx": sfx, "sound.music": music}
