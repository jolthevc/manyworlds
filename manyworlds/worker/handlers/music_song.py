"""Original songs (job kind music.song, policy `music`): lyrics and style in, the full mix plus a vocal and an
instrumental stem out, on the ComfyUI worker image (family music).

Payload (manyworlds.music.song_payload builds it; the worker reads no policy):
  lyrics          section-tagged lyrics ("[Verse]\\n...\\n[Chorus]\\n..."), or "[Instrumental]"
  instrumental    no vocals: the mix is the instrumental, nothing is separated, there is no vocal stem
  tags            the style prompt: genre, mood, instruments, vocal ("dreamy indie pop, soft female vocal, warm synths")
  bpm, key, time_signature, language, duration_s, seed, title
  engine          the engine's settings (policy music.engines.<engine> with its variant resolved): `engine` picks the
                  adapter in ENGINES (ace_step_1_5: ComfyUI core, worker/comfy_music.py; heartmula: a slot, refused)
  separation      {engine: demucs | none, model, shifts, overlap}: the stems (SEPARATORS)
  check           (optional) {asr_model, max_wer, max_duration_error_share, lufs: [lo, hi]}: the objective check
  library_licence what an instrumental's catalogue entry says (own-generated)

Outputs, always in this order (so the stored keys, assets/<creator>/<job>/00.wav..., are the same on a rerun):
  00 mix           the song as made (WAV, the engine's rate: 48 kHz for ACE-Step 1.5)
  01 vocals        the vocal stem (Demucs `vocals`)            -- not for an instrumental song
  02 instrumental  everything but the vocals (Demucs: the other sources summed); an instrumental song's is the mix
Each output's meta: role, seconds, sample_rate, channels, instrumental; the instrumental of an instrumental song also
carries its `library_entry` (manyworlds.sound catalogue schema), so a bed that passes joins the sound library
(manyworlds.music.generated_beds). With `check`, the mix's meta has `check`: the vocal stem transcribed (Whisper, the
designed voices' transcriber, handlers/speech_audio._asr) against the sung words (manyworlds.speech.wer), the length
against the asked length, the mix's integrated loudness (pyloudnorm), and `passed`.

Idempotent: the seed is the payload's, the graph and file names are fixed per job, a rerun overwrites the same keys.
Weights download once into MW_MODEL_DIR/comfy (the pool's volume if it has one); Demucs's into MW_MODEL_DIR/comfy/torch
(inside it: the ComfyUI handlers prune the folder's siblings when they download).
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from ... import music as music_mod
from ...speech import wer
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

_state: dict[str, Any] = {"weights": set()}
MAX_SECONDS = 600     # a sanity bound on the payload; the policy's music.limits are tighter


def _models_dir() -> Path:
    return Path(os.environ.get("MW_MODEL_DIR", "/models/mw")) / "comfy"


def validate(payload: dict) -> dict:
    """The payload's song, or NonRetryable for one no retry could make."""
    engine = payload.get("engine") or {}
    if not engine.get("engine"):
        raise NonRetryable("payload has no engine (manyworlds.music.song_payload builds it)")
    tags = music_mod.tags_text(payload.get("tags"))
    if not tags:
        raise NonRetryable("payload has no style tags")
    instrumental = music_mod.is_instrumental(payload)
    lyrics = music_mod.INSTRUMENTAL if instrumental else str(payload.get("lyrics") or "").strip()
    try:
        seconds = float(payload.get("duration_s") or 0)
    except (TypeError, ValueError):
        seconds = 0.0
    if not 0 < seconds <= MAX_SECONDS:
        raise NonRetryable(f"duration_s {payload.get('duration_s')!r} outside (0, {MAX_SECONDS}]")
    if payload.get("key") not in music_mod.KEYS:
        raise NonRetryable(f"key {payload.get('key')!r} is not one of the engine's keys")
    if payload.get("seed") is None:
        raise NonRetryable("payload has no seed (a song must be remade the same on a rerun)")
    return {"tags": tags, "lyrics": lyrics, "instrumental": instrumental, "seconds": seconds,
            "seed": int(payload["seed"]), "bpm": int(payload.get("bpm") or 96), "key": payload["key"],
            "time_signature": str(payload.get("time_signature") or "4"),
            "language": str(payload.get("language") or "en"), "title": str(payload.get("title") or ""),
            "engine": engine}


# --- the engines (adapters: policy music.engine picks one) --------------------------------------------------------

def _ace_step(song: dict, ctx: JobContext) -> Path:
    """ACE-Step 1.5 in ComfyUI core (worker/comfy_music.py): the song's audio file in the job's folder."""
    from .. import comfy, comfy_music
    engine = song["engine"]
    try:
        files = comfy_music.weights(engine)
        g = comfy_music.graph_song(tags=song["tags"], lyrics=song["lyrics"], seconds=song["seconds"],
                                   seed=song["seed"], bpm=song["bpm"], key=song["key"],
                                   time_signature=song["time_signature"], language=song["language"], engine=engine,
                                   prefix=f"mw_song_{ctx.job.id}")
    except (KeyError, ValueError) as e:
        raise NonRetryable(f"the ACE-Step engine settings are incomplete: {e}") from None
    models = _models_dir()
    ready = (engine.get("variant"), engine.get("diffusion_model"))
    if ready not in _state["weights"]:
        comfy.write_model_paths(models)
        comfy.ensure_weights(models, files, prune=True)
        _state["weights"].add(ready)
    comfy.start()
    out = comfy_music.run_audio(g, timeout_s=float(engine.get("timeout_s", 900)))
    dest = Path(ctx.workdir) / f"song_raw{out.suffix}"
    shutil.move(str(out), dest)
    return dest


def _heartmula(song: dict, ctx: JobContext) -> Path:
    raise NonRetryable("the HeartMuLa adapter is a slot only (policy music.engines.heartmula.built: false)")


ENGINES: dict[str, Callable[[dict, JobContext], Path]] = {"ace_step_1_5": _ace_step, "heartmula": _heartmula}


# --- audio in and out -----------------------------------------------------------------------------------------------

def read_audio(path: Path):
    """(samples [n, channels] float32, sample_rate)."""
    import soundfile as sf
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return a, int(sr)


def write_audio(path: Path, a, sr: int) -> Path:
    import numpy as np
    import soundfile as sf
    sf.write(str(path), np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0), int(sr), subtype="PCM_16")
    return path


# --- stems (adapters: payload separation.engine) --------------------------------------------------------------------

def _demucs(mix, sr: int, sep: dict):
    """Demucs (MIT code and weights): (vocals, instrumental), both [n, channels] at `sr`. The instrumental is every
    other source summed, so vocals + instrumental is the separated mix."""
    import julius
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    # Inside the ComfyUI weights folder: comfy.ensure_weights(prune=True) clears that folder's siblings.
    os.environ.setdefault("TORCH_HOME", str(_models_dir() / "torch"))
    name = str(sep.get("model") or "htdemucs")
    if _state.get("demucs_name") != name:
        model = get_model(name)
        model.eval()
        _state["demucs"], _state["demucs_name"] = model, name
    model = _state["demucs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wav = torch.from_numpy(mix.T.copy()).float()             # [channels, n]
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    if sr != model.samplerate:
        wav = julius.resample_frac(wav, sr, model.samplerate)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        sources = apply_model(model, ((wav - mean) / std)[None], device=device, shifts=int(sep.get("shifts", 1)),
                              split=True, overlap=float(sep.get("overlap", 0.25)), progress=False)[0]
    sources = sources * std + mean
    vi = model.sources.index("vocals")
    vocals = sources[vi]
    rest = sources.sum(0) - vocals
    if sr != model.samplerate:
        vocals, rest = julius.resample_frac(vocals, model.samplerate, sr), julius.resample_frac(rest, model.samplerate, sr)
    n = mix.shape[0]
    return vocals.cpu().numpy().T[:n], rest.cpu().numpy().T[:n]


SEPARATORS: dict[str, Callable] = {"demucs": _demucs}


# --- the objective check ----------------------------------------------------------------------------------------------

def loudness_lufs(a, sr: int) -> float | None:
    """Integrated loudness (ITU-R BS.1770, pyloudnorm); None for a file too short to measure."""
    import pyloudnorm
    try:
        v = float(pyloudnorm.Meter(int(sr)).integrated_loudness(a))
    except ValueError:
        return None
    return None if v == float("-inf") else round(v, 2)


def transcribe(vocals, sr: int, check: dict) -> str:
    """The vocal stem as heard (Whisper, as the designed voices are scored: handlers/speech_audio._asr)."""
    from . import speech_audio
    mono = vocals.mean(axis=1) if getattr(vocals, "ndim", 1) == 2 else vocals
    return speech_audio._asr({"asr_model": check.get("asr_model", "openai/whisper-large-v3"),
                              "language": check.get("language", "en")})(mono, sr)


def judge(check: dict, *, asked_s: float, seconds: float, lufs: float | None, lyric_wer: float | None,
          coverage: float | None = None) -> dict:
    """The check's verdict from what was measured (pure): each limit that applies, and `passed`. With
    `min_line_coverage` in the check, the lyrics pass on the share of written lines sung (repeats are fine); WER is
    then reported only."""
    err = round(abs(seconds - asked_s) / asked_s, 3) if asked_s else None
    out = {"asked_s": asked_s, "seconds": seconds, "duration_error_share": err, "lufs": lufs, "wer": lyric_wer,
           "line_coverage": coverage}
    fails = []
    if err is not None and check.get("max_duration_error_share") is not None \
            and err > float(check["max_duration_error_share"]):
        fails.append("duration")
    lo_hi = check.get("lufs")
    if lo_hi and (lufs is None or not float(lo_hi[0]) <= lufs <= float(lo_hi[1])):
        fails.append("loudness")
    if check.get("min_line_coverage") is not None:
        if coverage is not None and coverage < float(check["min_line_coverage"]):
            fails.append("lyrics")
    elif lyric_wer is not None and check.get("max_wer") is not None and lyric_wer > float(check["max_wer"]):
        fails.append("lyrics")
    return {**out, "failed": fails, "passed": not fails}


def vocal_embedding(vocals, sr: int, settings: dict) -> dict:
    """The vocal stem's voice (payload `embed`: music.voice.similarity): its voiced windows' centroid (ECAPA,
    handlers/music_voice.embed_windows) as `embedding`, with the window count."""
    from ... import singing_voice
    from . import music_voice
    embs = music_voice.embed_windows(music_voice._mono(vocals), sr, settings)
    return {"embedding": singing_voice.centroid(embs), "windows": len(embs), "embedding_model": settings.get("model")}


# --- the job ------------------------------------------------------------------------------------------------------

def generate(payload: dict, ctx: JobContext) -> HandlerResult:
    song = validate(payload)
    name = str(song["engine"]["engine"])
    if name not in ENGINES:
        raise NonRetryable(f"no music engine adapter {name!r} (known: {sorted(ENGINES)})")
    work = Path(ctx.workdir)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    raw = ENGINES[name](song, ctx)
    render_s = round(time.monotonic() - t0, 1)
    mix, sr = read_audio(raw)
    seconds = round(mix.shape[0] / sr, 3)
    if seconds <= 0:
        raise RuntimeError("the engine made an empty song")
    base = {"sample_rate": sr, "channels": int(mix.shape[1]), "seconds": seconds,
            "instrumental": song["instrumental"], "engine": name, "variant": song["engine"].get("variant")}
    files: list[OutputFile] = [OutputFile(write_audio(work / "song_mix.wav", mix, sr), "audio", {**base, "role": "mix"})]
    vocals, sep_s = None, None
    sep = payload.get("separation") or {}
    if song["instrumental"]:
        shutil.copyfile(files[0].path, work / "song_instrumental.wav")
        entry = {"tag": (song["title"] or song["tags"])[:80], "kind": "music", "formats": [],
                 "licence": str(payload.get("library_licence") or "own-generated"),
                 "source": f"{name} {song['engine'].get('variant') or ''} seed {song['seed']}: {song['tags']}".strip(),
                 "credit": ""}
        files.append(OutputFile(work / "song_instrumental.wav", "audio",
                                {**base, "role": "instrumental", "library_entry": entry}))
    elif str(sep.get("engine") or "none") != "none":
        if sep["engine"] not in SEPARATORS:
            raise NonRetryable(f"no separation adapter {sep['engine']!r} (known: {sorted(SEPARATORS)})")
        t1 = time.monotonic()
        vocals, rest = SEPARATORS[sep["engine"]](mix, sr, sep)
        sep_s = round(time.monotonic() - t1, 1)
        vmeta = {**base, "role": "vocals"}
        if payload.get("embed"):   # a voice candidate or dataset song (creator/music_voice.py): the stem's voice
            vmeta.update(vocal_embedding(vocals, sr, payload["embed"]))
        files.append(OutputFile(write_audio(work / "song_vocals.wav", vocals, sr), "audio", vmeta))
        files.append(OutputFile(write_audio(work / "song_instrumental.wav", rest, sr), "audio",
                                {**base, "role": "instrumental"}))
    check = payload.get("check")
    if check:
        words = music_mod.sung_words(song["lyrics"]) if not song["instrumental"] else ""
        heard = transcribe(vocals, sr, check) if (vocals is not None and words) else None
        verdict = judge(check, asked_s=song["seconds"], seconds=seconds, lufs=loudness_lufs(mix, sr),
                        lyric_wer=wer(words, heard) if heard is not None else None,
                        coverage=(music_mod.line_coverage(song["lyrics"], heard,
                                                          min_line_ratio=float(check.get("min_line_ratio", 0.7)))
                                  if heard is not None else None))
        if heard is not None:
            verdict["transcript"] = heard[:2000]
        for f in files:   # every output knows whether its song passed (generated beds read the instrumental's)
            f.meta["check"] = verdict if f.meta["role"] == "mix" else {"passed": verdict["passed"]}
    recipe = {"engine": name, "variant": song["engine"].get("variant"), "licence": song["engine"].get("licence"),
              "model": song["engine"].get("diffusion_model"), "repo": song["engine"].get("repo"),
              "title": song["title"], "tags": song["tags"], "lyrics": song["lyrics"], "bpm": song["bpm"],
              "key": song["key"], "time_signature": song["time_signature"], "language": song["language"],
              "duration_s": song["seconds"], "seed": song["seed"], "separation": sep or None}
    meta = {"output_seconds": seconds, "render_seconds": render_s}
    if sep_s is not None:
        meta["separation_seconds"] = sep_s
    return HandlerResult(files, recipe=recipe, meta=meta)


def release() -> None:
    """Give the GPU back before another family's job (worker.runner._switch_to): ComfyUI's models and Demucs."""
    from .. import comfy
    _state.pop("demucs", None)
    _state.pop("demucs_name", None)
    comfy.free()


HANDLERS = {"music.song": generate}
