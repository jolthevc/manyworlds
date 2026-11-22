"""Designed voices (job kinds voice.design, voice.speak): VoxCPM2 (openbmb/VoxCPM2, Apache-2.0), on the talk worker
image (docker/worker-talkcomfy.Dockerfile). Owner 2026-11-22: every creator speaks in a voice of their own.

One model designs a voice from a description, clones it, and follows a delivery instruction on the clone (VoxCPM
README, 2.0.3): a description or delivery goes in parentheses at the start of `text`; cloning takes
`prompt_wav_path` + `prompt_text` (+ `reference_wav_path`); a delivery on a cloned voice takes `reference_wav_path`
alone. 2.0.3's `generate()` has no `seed` argument (the README on main shows one), so takes are seeded through torch.
Output varies run to run, so every line is generated `takes` times and the take whose transcript is closest to the
line wins. Voices are judged automatically, never by ear (owner 2026-11-21).

voice.design payload {description, text, n, seed, params}: n candidate clips of `text` in the described voice, each
with its word error rate, a speaker embedding and its pitch variance (creator/voice_lab.pick chooses one).
voice.speak payload {ref_uri, ref_text, lines: [{text, delivery, pause_after_s}], takes, seed, params}: one 48 kHz
wav of the lines in the cloned voice, joined with pauses, plus each line's timing.
params {model, cfg_value, inference_timesteps, optimize, asr_model, language, embedding, pitch_hz, pause_s}.

Transcripts: the speech gate's model (policy review.video.intelligibility.model, Whisper large-v3) run here on the
GPU, scored with the gate's own `speech.wer`. The hosted transcription API is not used from a rented machine: its key
would have to be handed to an untrusted host (CLAUDE.md, deploy/credentials.py).

Speaker embedding: VoxCPM exposes none, so SpeechBrain's ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb; model card
and toolkit Apache-2.0, checked 2026-11-22; trained on VoxCeleb 1+2). Only distances between our own creators'
voices are computed with it; nothing is identified.
"""

from __future__ import annotations

import os
import random
import wave
from pathlib import Path
from typing import Any

from ... import voices
from ...speech import wer
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

DEFAULT_MODEL = "openbmb/VoxCPM2"
ASR_RATE = 16000          # Whisper and ECAPA both take 16 kHz
_state: dict[str, Any] = {}


# --- the models (heavy imports only here, on the worker; tests replace these four) ---------------------------------
class _VoxCPM:
    def __init__(self, model_id: str, optimize: bool):
        from voxcpm import VoxCPM
        self.model = VoxCPM.from_pretrained(model_id, load_denoiser=False, optimize=optimize)
        self.sample_rate = int(self.model.tts_model.sample_rate)

    def generate(self, *, seed: int, **kwargs):
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return np.asarray(self.model.generate(**kwargs), dtype=np.float32).reshape(-1)


def _resample(wav, sr: int, target: int = ASR_RATE):
    import numpy as np
    if sr == target:
        return np.asarray(wav, dtype=np.float32)
    import librosa
    return librosa.resample(np.asarray(wav, dtype=np.float32), orig_sr=sr, target_sr=target)


class _Whisper:
    def __init__(self, model_id: str, language: str):
        import torch
        from transformers import pipeline
        cuda = torch.cuda.is_available()
        self.pipe = pipeline("automatic-speech-recognition", model=model_id,
                             torch_dtype=torch.float16 if cuda else torch.float32, device=0 if cuda else -1)
        self.language = language

    def __call__(self, wav, sr: int) -> str:
        out = self.pipe({"raw": _resample(wav, sr), "sampling_rate": ASR_RATE}, chunk_length_s=30,
                        generate_kwargs={"language": self.language, "task": "transcribe"})
        return str(out.get("text", "")).strip()


class _Ecapa:
    def __init__(self, source: str):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        save = Path(os.environ.get("HF_HOME", "/models/hf")) / "speechbrain" / source.replace("/", "--")
        self.clf = EncoderClassifier.from_hparams(source=source, savedir=str(save),
                                                  run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"})

    def __call__(self, wav, sr: int) -> list[float]:
        import torch
        x = torch.from_numpy(_resample(wav, sr)).float().unsqueeze(0)
        emb = self.clf.encode_batch(x).squeeze().detach().cpu().tolist()
        return [round(float(v), 6) for v in emb]


def _tts(params: dict):
    key = ("tts", params.get("model", DEFAULT_MODEL))
    if key not in _state:
        _state[key] = _VoxCPM(key[1], bool(params.get("optimize", False)))
    return _state[key]


def _asr(params: dict):
    key = ("asr", params.get("asr_model", "openai/whisper-large-v3"), params.get("language", "en"))
    if key not in _state:
        _state[key] = _Whisper(key[1], key[2])
    return _state[key]


def _embedder(params: dict):
    key = ("embed", params.get("embedding", "speechbrain/spkrec-ecapa-voxceleb"))
    if key not in _state:
        _state[key] = _Ecapa(key[1])
    return _state[key]


def _pitch_variance(wav, sr: int, params: dict) -> float:
    """How much the voice moves: the standard deviation of voiced pitch in semitones (low = monotone)."""
    import librosa
    import numpy as np
    lo, hi = params.get("pitch_hz", [65, 500])
    f0, voiced, _ = librosa.pyin(_resample(wav, sr), fmin=float(lo), fmax=float(hi), sr=ASR_RATE)
    f0 = f0[np.asarray(voiced, dtype=bool) & np.isfinite(f0)]
    if len(f0) < 2:
        return 0.0
    return round(float(np.std(12 * np.log2(f0 / np.median(f0)))), 4)


# --- pure parts --------------------------------------------------------------------------------------------------
def write_wav(path: Path, wav, sample_rate: int) -> Path:
    """Mono 16-bit PCM (the standard library's writer: no audio library needed)."""
    import numpy as np
    pcm = (np.clip(np.asarray(wav, dtype=np.float32).reshape(-1), -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return path


def _params(payload: dict) -> dict:
    p = payload.get("params") or {}
    if not isinstance(p, dict):
        raise NonRetryable("params must be an object")
    return p


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or int(value) < 1:
        raise NonRetryable(f"{name} must be a positive whole number, got {value!r}")
    return int(value)


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NonRetryable(f"payload needs {name}")
    return value.strip()


def _seed(payload: dict) -> int:
    seed = payload.get("seed")
    if seed is None:
        return random.randrange(1 << 31)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise NonRetryable(f"seed must be a whole number, got {seed!r}")
    return seed


def _generation(params: dict) -> dict:
    return {"cfg_value": float(params.get("cfg_value", 2.0)), "inference_timesteps": int(params.get("inference_timesteps", 10))}


def validate_design(payload: dict) -> dict:
    """The design payload, checked: {description, text, n, seed, params}."""
    description = _text(payload.get("description"), "description")
    text = voices.strip_delivery(_text(payload.get("text"), "text"))
    if not text:
        raise NonRetryable("text has no words to speak")
    return {"description": description, "text": text, "n": _positive_int(payload.get("n"), "n"),
            "seed": _seed(payload), "params": _params(payload)}


def validate_speak(payload: dict) -> dict:
    """The speak payload, checked: each line's spoken words with its delivery split off (an inline leading
    parenthetical counts as the delivery when none is given)."""
    ref_uri, ref_text = _text(payload.get("ref_uri"), "ref_uri"), _text(payload.get("ref_text"), "ref_text")
    params = _params(payload)
    lines = payload.get("lines")
    if not isinstance(lines, list) or not lines:
        raise NonRetryable("payload needs lines")
    out = []
    for i, line in enumerate(lines):
        if not isinstance(line, dict):
            raise NonRetryable(f"line {i} must be an object")
        inline, words = voices.split_instruction(_text(line.get("text"), f"lines[{i}].text"))
        if not words:
            raise NonRetryable(f"line {i} has no words to speak")
        delivery = line.get("delivery")
        if delivery is not None and not isinstance(delivery, str):
            raise NonRetryable(f"line {i} delivery must be text")
        pause = line.get("pause_after_s")
        if pause is None:
            pause = params.get("pause_s", 0.0)
        if isinstance(pause, bool) or not isinstance(pause, (int, float)) or pause < 0:
            raise NonRetryable(f"line {i} pause_after_s must be a number of seconds >= 0")
        out.append({"text": words, "delivery": (delivery or "").strip() or inline, "pause_after_s": float(pause)})
    takes = _positive_int(payload.get("takes", 1), "takes")
    return {"ref_uri": ref_uri, "ref_text": voices.strip_delivery(ref_text), "lines": out, "takes": takes,
            "seed": _seed(payload), "params": params}


def clone_args(ref: Path, ref_text: str, delivery: str) -> dict:
    """How to clone (VoxCPM README): with a delivery instruction, the reference clip alone, so the instruction is
    followed; without one, the clip and its transcript as well, for the closest likeness."""
    if delivery:
        return {"reference_wav_path": str(ref)}
    return {"prompt_wav_path": str(ref), "prompt_text": ref_text, "reference_wav_path": str(ref)}


# --- the handlers ------------------------------------------------------------------------------------------------
def design(payload: dict, ctx: JobContext) -> HandlerResult:
    p = validate_design(payload)
    params, text = p["params"], p["text"]
    tts, asr, embed = _tts(params), _asr(params), _embedder(params)
    sr = int(tts.sample_rate)
    work = Path(ctx.workdir)
    outputs, total = [], 0.0
    for i in range(p["n"]):
        seed = p["seed"] + i
        wav = tts.generate(seed=seed, text=voices.with_instruction(p["description"], text), **_generation(params))
        path = write_wav(work / f"candidate_{i:02d}.wav", wav, sr)
        heard = asr(wav, sr)
        seconds = round(len(wav) / sr, 2)
        total += seconds
        outputs.append(OutputFile(path, "audio", {
            "index": i, "seed": seed, "wer": wer(text, heard), "transcript": heard[:2000],
            "embedding": embed(wav, sr), "pitch_var": float(_pitch_variance(wav, sr, params)),
            "seconds": seconds, "sample_rate": sr}))
    recipe = {"model": params.get("model", DEFAULT_MODEL), "engine": "voxcpm", "description": p["description"],
              "text": text, "seed": p["seed"], "params": params}
    return HandlerResult(outputs, recipe=recipe, meta={"output_seconds": round(total, 2)})


def speak(payload: dict, ctx: JobContext) -> HandlerResult:
    import numpy as np

    p = validate_speak(payload)
    params = p["params"]
    tts, asr = _tts(params), _asr(params)
    sr = int(tts.sample_rate)
    ref = ctx.fetch(p["ref_uri"], "reference.wav")
    pieces, timings, at = [], [], 0
    for li, line in enumerate(p["lines"]):
        best = None
        for k in range(p["takes"]):
            wav = tts.generate(seed=p["seed"] + li * p["takes"] + k,
                               text=voices.with_instruction(line["delivery"], line["text"]),
                               **clone_args(ref, p["ref_text"], line["delivery"]), **_generation(params))
            heard = asr(wav, sr)
            score = wer(line["text"], heard)   # against the words only: the delivery is never spoken
            if best is None or score < best[0]:
                best = (score, k, np.asarray(wav, dtype=np.float32).reshape(-1), heard)
        score, take, wav, heard = best
        pieces.append(wav)
        timings.append({"index": li, "start_s": round(at / sr, 3), "end_s": round((at + len(wav)) / sr, 3),
                        "text": line["text"], "delivery": line["delivery"], "take": take, "wer": score,
                        "transcript": heard[:500]})
        at += len(wav)
        if li < len(p["lines"]) - 1 and line["pause_after_s"] > 0:
            gap = int(round(line["pause_after_s"] * sr))
            pieces.append(np.zeros(gap, dtype=np.float32))
            at += gap
    path = write_wav(Path(ctx.workdir) / "speech.wav", np.concatenate(pieces), sr)
    seconds = round(at / sr, 2)
    words = " ".join(t["text"] for t in timings)
    return HandlerResult([OutputFile(path, "audio", {"seconds": seconds, "sample_rate": sr, "lines": timings,
                                                     "wer": wer(words, " ".join(t["transcript"] for t in timings))})],
                         recipe={"model": params.get("model", DEFAULT_MODEL), "engine": "voxcpm", "ref_uri": p["ref_uri"],
                                 "lines": p["lines"], "takes": p["takes"], "seed": p["seed"], "params": params},
                         meta={"output_seconds": seconds})


HANDLERS = {"voice.design": design, "voice.speak": speak}
