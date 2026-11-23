"""Designed voices (job kinds voice.design, voice.speak): VoxCPM2 (openbmb/VoxCPM2, Apache-2.0), on the talk worker
image (docker/worker-talkcomfy.Dockerfile). Owner 2026-11-22: every creator speaks in a voice of their own.

One model designs a voice from a description, clones it, and follows a delivery instruction on the clone (VoxCPM
README, 2.0.3): a description or delivery goes in parentheses at the start of `text`; ultimate cloning takes
`prompt_wav_path` + `prompt_text` (+ `reference_wav_path`). Every line is cloned that way, the delivery staying the
instruction (params.clone `ultimate`, policy voice.clone): lines with a delivery used to get the reference clip
alone (controllable cloning, still available as `controllable`) and their timbre drifted from the lines without.
2.0.3's `generate()` has no `seed` argument (the README on main shows one), so takes are seeded through torch.
Output varies run to run, so every line is generated `takes` times and the take whose transcript is closest to the
line wins. Voices are judged automatically, never by ear (owner 2026-11-21).

voice.design payload {description, text, n, seed, params}: n candidate clips of `text` in the described voice, each
with its word error rate, a speaker embedding and its pitch variance (creator/voice_lab.pick chooses one).
voice.speak payload {ref_uri, ref_text, lines: [{text, delivery, pause_after_s}], takes, seed, params, compare}: one
48 kHz wav of the lines in the cloned voice, plus each line's and each clause's timing; with `compare`, the other
synthesis (whole line / clauses) as a second wav for the breadth test's comparison.
params {model, cfg_value, inference_timesteps, optimize, asr_model, language, embedding, pitch_hz, pause_s, clone,
synthesis} and the natural-speech keys (`speech_config`; policy voice, plan 2026-11-22-video-quality-autonomous
item 7).

Natural speech: the model's own pauses are not relied on. With synthesis `clauses`, each line is spoken clause by
clause (voices.segments) in the line's delivery; with `whole_line` (the default: one intonation across the
sentence), each line is one piece. Each piece's takes are scored against its words and the best
kept (a clause whose every take is worse than max_wer x clause_retry_wer_factor gets one more round of seeds); each
take is trimmed of its own leading and trailing silence and faded at its edges, then joined with silences of our
choosing: a clause gap or a sentence gap drawn from its policy range with a seeded RNG (reproducible, not
metronomic), the line's own pause_after_s after its last clause. A line whose assembled WER exceeds line_max_wer
fails the job (retried with new seeds). Experimental, off: a quiet synthetic breath before every n-th sentence, and
a rare filler ("um") before a clause, recorded in meta.fillers so captions leave it out.
Output meta: lines [{index, start_s, end_s, text, delivery, clauses, wer, transcript}], clauses [{index, line, text
(scripted words, no filler), start_s, end_s, kind, filler, breath_before, take, seed, retried, wer, transcript}],
fillers [{clause, line, word}], seconds, sample_rate, wer, synthesis (and compare, compare_of, failed_lines).

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
        pause = line.get("pause_after_s")   # None: the planned gap after the line (see `plan`)
        if pause is not None and (isinstance(pause, bool) or not isinstance(pause, (int, float)) or pause < 0):
            raise NonRetryable(f"line {i} pause_after_s must be a number of seconds >= 0")
        out.append({"text": words, "delivery": (delivery or "").strip() or inline,
                    "pause_after_s": None if pause is None else float(pause)})
    takes = _positive_int(payload.get("takes", 1), "takes")
    compare = payload.get("compare", False)
    if not isinstance(compare, bool):
        raise NonRetryable(f"compare must be true or false, got {compare!r}")
    clone_mode(params)                      # a bad mode fails now, not after the first take
    return {"ref_uri": ref_uri, "ref_text": voices.strip_delivery(ref_text), "lines": out, "takes": takes,
            "seed": _seed(payload), "params": params, "speech": speech_config(params), "compare": compare}


# --- natural speech (plan 2026-11-22-video-quality-autonomous item 7; policy voice) ------------------------------
def _number(value, name: str, *, lo: float | None = None, hi: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NonRetryable(f"{name} must be a number, got {value!r}")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise NonRetryable(f"{name} must be within [{lo}, {hi}], got {value!r}")
    return float(value)


def _span(value, name: str, fallback: float) -> tuple[float, float]:
    """A [lo, hi] range of seconds (a single number is a fixed length); absent, the fixed `fallback`."""
    if value is None:
        return (fallback, fallback)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = [value, value]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise NonRetryable(f"{name} must be [lo, hi] seconds, got {value!r}")
    lo, hi = _number(value[0], name, lo=0), _number(value[1], name, lo=0)
    if hi < lo:
        raise NonRetryable(f"{name} must be [lo, hi] with lo <= hi, got {value!r}")
    return (lo, hi)


SYNTHESIS = ("whole_line", "clauses")


def synthesis(params: dict) -> str:
    """How a line is generated (params.synthesis, policy voice.synthesis): `whole_line`, one generation per line
    (the model's own intonation across the sentence), or `clauses`, one per clause joined with our gaps. A payload
    from before the key keeps what its `split_clauses` said."""
    value = params.get("synthesis")
    if value is None:
        return "clauses" if params.get("split_clauses") else "whole_line"
    if value not in SYNTHESIS:
        raise NonRetryable(f"synthesis must be one of {list(SYNTHESIS)}, got {value!r}")
    return str(value)


def speech_config(params: dict) -> dict:
    """How lines become natural speech, from the worker params (creator/voice_lab.worker_params, policy `voice`).
    Every part is off when its key is absent, so a payload from before natural speech is spoken as it was: whole
    lines, joined with `pause_s`, untrimmed."""
    pause = _number(params.get("pause_s", 0.0), "pause_s", lo=0)
    cfg: dict[str, Any] = {
        "split": synthesis(params) == "clauses",
        "min_words": _positive_int(params.get("clause_min_words", 1), "clause_min_words"),
        "clause_gap": _span(params.get("clause_gap_s"), "clause_gap_s", pause),
        "sentence_gap": _span(params.get("sentence_gap_s"), "sentence_gap_s", pause),
        "fade_ms": _number(params.get("fade_ms", 0), "fade_ms", lo=0),
        "max_wer": None, "retry_factor": None, "line_max_wer": None, "trim": None, "breaths": None,
        "disfluency": None}
    if params.get("max_wer") is not None and params.get("clause_retry_wer_factor") is not None:
        cfg["max_wer"] = _number(params["max_wer"], "max_wer", lo=0)
        cfg["retry_factor"] = _number(params["clause_retry_wer_factor"], "clause_retry_wer_factor", lo=1)
    if params.get("line_max_wer") is not None:
        cfg["line_max_wer"] = _number(params["line_max_wer"], "line_max_wer", lo=0)
    trim = params.get("trim")
    if trim is not None:
        if not isinstance(trim, dict):
            raise NonRetryable("trim must be {threshold_db, frame_ms, pad_ms}")
        cfg["trim"] = {"threshold_db": _number(trim.get("threshold_db"), "trim.threshold_db", hi=0),
                       "frame_ms": _number(trim.get("frame_ms"), "trim.frame_ms", lo=0.1),
                       "pad_ms": _number(trim.get("pad_ms", 0), "trim.pad_ms", lo=0)}
    b = params.get("breaths") or {}
    if not isinstance(b, dict):
        raise NonRetryable("breaths must be an object")
    if b.get("enabled"):
        band = b.get("band_hz")
        if not isinstance(band, (list, tuple)) or len(band) != 2 or not 0 < _number(band[0], "breaths.band_hz") \
                < _number(band[1], "breaths.band_hz"):
            raise NonRetryable(f"breaths.band_hz must be [lo, hi] Hz, got {band!r}")
        cfg["breaths"] = {"every": _positive_int(b.get("every_n_sentences"), "breaths.every_n_sentences"),
                          "gain_db": _number(b.get("gain_db"), "breaths.gain_db", hi=0),
                          "seconds": _span(b.get("seconds"), "breaths.seconds", 0.0),
                          "band_hz": (float(band[0]), float(band[1])),
                          "lead_s": _number(b.get("lead_s", 0), "breaths.lead_s", lo=0)}
    d = params.get("disfluency") or {}
    if not isinstance(d, dict):
        raise NonRetryable("disfluency must be an object")
    if d.get("enabled"):
        fillers = d.get("fillers")
        if not isinstance(fillers, list) or not fillers or not all(isinstance(f, str) and f.strip() for f in fillers):
            raise NonRetryable("disfluency.fillers must be a list of words")
        cfg["disfluency"] = {"rate": _number(d.get("rate"), "disfluency.rate", lo=0, hi=1),
                             "max_per_line": _positive_int(d.get("max_per_line", 1), "disfluency.max_per_line"),
                             "fillers": [f.strip() for f in fillers]}
    return cfg


def plan(lines: list[dict], cfg: dict, seed: int) -> list[dict]:
    """What is spoken, piece by piece (pure): [{line, text, spoken, filler, delivery, kind, gap_after_s, breath}].
    `text` is the creator's scripted words (captions show these); `spoken` is what the model is asked to say:
    the same words, with a filler in front only when disfluency is enabled. A clause opening with its own
    parenthetical takes it as its delivery, added to the line's. Gaps: a clause or sentence gap drawn from its
    range with an RNG seeded by the job's seed, the line's explicit pause_after_s after its last clause (the
    sentence gap when it gives none), nothing after the last."""
    gaps, fill = random.Random(f"{seed}:gaps"), random.Random(f"{seed}:fillers")
    dis, breaths = cfg.get("disfluency"), cfg.get("breaths")
    units: list[dict] = []
    sentence_starts = 0
    for li, line in enumerate(lines):
        segs = voices.segments(line["text"], min_words=cfg["min_words"]) if cfg["split"] else []
        segs = [s for s in segs if voices.strip_delivery(s[0])] or [(line["text"], "end")]
        fillers_left = dis["max_per_line"] if dis else 0
        starts_sentence = True
        for si, (piece, kind) in enumerate(segs):
            inline, words = voices.split_instruction(piece)
            delivery = ", ".join(x for x in (line["delivery"], inline) if x)
            filler = None
            if dis and fillers_left and fill.random() < dis["rate"]:
                choice = fill.choice(dis["fillers"])
                first = (voices._WORD.findall(words) or [""])[0].lower()
                if first != choice.lower():
                    filler, fillers_left = choice, fillers_left - 1
            if si == len(segs) - 1 and li == len(lines) - 1:
                gap = 0.0
            elif si == len(segs) - 1 and line["pause_after_s"] is not None:
                gap = line["pause_after_s"]
            elif si == len(segs) - 1:
                gap = voices.gap_s(gaps, cfg["sentence_gap"])
            else:
                gap = voices.gap_s(gaps, cfg["clause_gap" if kind == "clause" else "sentence_gap"])
            breath = bool(breaths and starts_sentence and sentence_starts and sentence_starts % breaths["every"] == 0)
            if starts_sentence:
                sentence_starts += 1
            units.append({"line": li, "text": words, "spoken": f"{filler.capitalize()}, {words}" if filler else words,
                          "filler": filler, "delivery": delivery, "kind": kind, "gap_after_s": float(gap),
                          "breath": breath})
            starts_sentence = kind != "clause"
    return units


def trim(wav, sr: int, *, threshold_db: float, frame_ms: float, pad_ms: float = 0.0):
    """The clip without its leading and trailing silence: frames (`frame_ms`) whose RMS is more than
    `threshold_db` below the loudest frame are cut from both ends, keeping `pad_ms` around the speech so soft
    onsets and endings survive. A clip with nothing above the threshold is returned as it is."""
    import numpy as np
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    n = max(1, int(round(sr * frame_ms / 1000)))
    frames = -(-len(x) // n)
    if not frames:
        return x
    padded = np.zeros(frames * n, dtype=np.float32)
    padded[:len(x)] = x
    rms = np.sqrt(np.mean(padded.reshape(frames, n).astype(np.float64) ** 2, axis=1))
    peak = float(rms.max())
    if peak <= 0:
        return x
    loud = np.nonzero(rms >= peak * 10 ** (threshold_db / 20))[0]
    pad = int(round(sr * pad_ms / 1000))
    return x[max(0, int(loud[0]) * n - pad): min(len(x), (int(loud[-1]) + 1) * n + pad)]


def fade(wav, sr: int, ms: float):
    """Raised-cosine fade in and out over `ms` at both edges (length unchanged), so a clause meets the silence
    around it without a click. Joins are always through our own gap, so fading each edge is the crossfade; a zero
    gap meets two faded edges."""
    import numpy as np
    x = np.array(wav, dtype=np.float32).reshape(-1)
    n = min(int(round(sr * ms / 1000)), len(x) // 2)
    if n > 0:
        ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)).astype(np.float32)
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def breath(sr: int, seconds: float, *, gain_db: float, band_hz: tuple[float, float], seed: int):
    """EXPERIMENTAL (policy voice.breaths, off): a quiet synthetic in-breath, pink-shaped noise band-limited to
    `band_hz` under a Hann envelope, at `gain_db` dBFS RMS. Nothing is sampled from anyone: numpy noise only."""
    import numpy as np
    n = int(round(sr * seconds))
    if n < 4:
        return np.zeros(max(n, 0), dtype=np.float32)
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    shape = np.zeros_like(f)
    band = (f >= band_hz[0]) & (f <= band_hz[1])
    shape[band] = 1 / np.sqrt(f[band])
    x = np.fft.irfft(spectrum * shape, n) * np.hanning(n)
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms > 0:
        x *= 10 ** (gain_db / 20) / rms
    return x.astype(np.float32)


def assemble(pieces: list, gaps_s: list[float], sr: int, *, breaths: list | None = None, lead_s: float = 0.0):
    """Pieces joined by silences (gaps_s[i] after piece i; none after the last) -> (wav, [(start, end)] samples).
    breaths[i], when given, sits in the silence before piece i, ending `lead_s` before it; one longer than that
    silence is left out, so a gap is always exactly the planned length."""
    import numpy as np
    breaths = breaths or [None] * len(pieces)
    out, spans, at = [], [], 0
    for i, piece in enumerate(pieces):
        piece = np.asarray(piece, dtype=np.float32).reshape(-1)
        spans.append((at, at + len(piece)))
        out.append(piece)
        at += len(piece)
        if i < len(pieces) - 1:
            silence = np.zeros(int(round(gaps_s[i] * sr)), dtype=np.float32)
            b = breaths[i + 1]
            if b is not None and 0 < len(b) <= len(silence):
                end = len(silence) - min(int(round(lead_s * sr)), len(silence) - len(b))
                silence[end - len(b):end] = b
            out.append(silence)
            at += len(silence)
    return (np.concatenate(out) if out else np.zeros(0, dtype=np.float32)), spans


CLONE_MODES = ("ultimate", "controllable")


def clone_mode(params: dict) -> str:
    """How lines are cloned (params.clone, policy voice.clone): `ultimate` (the default) always gives the model the
    reference clip and its transcript, so every line continues from the same voice; `controllable` gives a line
    with a delivery the clip alone, as before 2026-11-23 (the instruction is followed more freely, but the timbre
    drifted between lines with and without one)."""
    mode = str(params.get("clone") or "ultimate")
    if mode not in CLONE_MODES:
        raise NonRetryable(f"clone must be one of {list(CLONE_MODES)}, got {mode!r}")
    return mode


def clone_args(ref: Path, ref_text: str, delivery: str, mode: str = "ultimate") -> dict:
    """The cloning arguments (VoxCPM README). Ultimate cloning: `prompt_wav_path` + `prompt_text` (the reference and
    exactly what it says) plus `reference_wav_path`, on every line, the delivery staying the instruction in the
    text's leading parenthetical (voices.with_instruction). Controllable cloning (mode `controllable`): a line with
    a delivery gets the reference clip alone."""
    if mode == "controllable" and delivery:
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


def _voice_lines(p: dict, cfg: dict, tts, asr, ref: Path, sr: int, attempt: int, *, strict: bool) -> dict:
    """Every line spoken as `cfg` says (whole lines or clauses), the best take of each piece kept, joined with the
    planned gaps: {audio, units, lines (timings), clauses, fillers, wer, failed}. `strict`: a line over the
    line ceiling fails the job (retryable); otherwise (the comparison variant) it is recorded in `failed`."""
    import numpy as np

    params, takes = p["params"], p["takes"]
    mode = clone_mode(params)
    units = plan(p["lines"], cfg, p["seed"])

    def take(unit: dict, k: int, seed: int):
        wav = tts.generate(seed=seed, text=voices.with_instruction(unit["delivery"], unit["spoken"]),
                           **clone_args(ref, p["ref_text"], unit["delivery"], mode), **_generation(params))
        heard = asr(wav, sr)
        # against the words asked for only: a delivery is never spoken, so it is never in `spoken`
        return (wer(unit["spoken"], heard), k, seed, np.asarray(wav, dtype=np.float32).reshape(-1), heard)

    # Seeds: a whole line spoken as one piece keeps the seeds it had before clauses (seed + line * takes + k);
    # retries of a clause use the block after them; a job's later attempt (after the line ceiling failed it) moves
    # past both, so it does not regenerate the same takes.
    base = p["seed"] + (attempt - 1) * 2 * len(units) * takes
    chosen = []
    for ui, unit in enumerate(units):
        tried = [take(unit, k, base + ui * takes + k) for k in range(takes)]
        retried = False
        if cfg["retry_factor"] and min(t[0] for t in tried) > cfg["max_wer"] * cfg["retry_factor"]:
            retried = True       # every take garbled the clause: once more with new seeds
            tried += [take(unit, takes + k, base + (len(units) + ui) * takes + k) for k in range(takes)]
        score, k, seed, wav, heard = min(tried, key=lambda t: (t[0], t[1]))
        if cfg["trim"]:
            wav = trim(wav, sr, **cfg["trim"])
        if cfg["fade_ms"]:
            wav = fade(wav, sr, cfg["fade_ms"])
        chosen.append({"wav": wav, "wer": score, "take": k, "seed": seed, "heard": heard, "retried": retried})

    b = cfg["breaths"]
    inserts = [breath(sr, voices.gap_s(random.Random(f"{p['seed']}:breath:{ui}"), b["seconds"]),
                      gain_db=b["gain_db"], band_hz=b["band_hz"], seed=(p["seed"] + ui) % (1 << 32))
               if b and u["breath"] else None for ui, u in enumerate(units)]
    audio, spans = assemble([c["wav"] for c in chosen], [u["gap_after_s"] for u in units], sr, breaths=inserts,
                            lead_s=b["lead_s"] if b else 0.0)

    clauses_meta, fillers = [], []
    for ui, (unit, c, (s, e)) in enumerate(zip(units, chosen, spans)):
        clauses_meta.append({"index": ui, "line": unit["line"], "text": unit["text"], "start_s": round(s / sr, 3),
                             "end_s": round(e / sr, 3), "kind": unit["kind"], "filler": unit["filler"],
                             "breath_before": bool(ui and inserts[ui] is not None and 0 < len(inserts[ui])
                                                   <= int(round(units[ui - 1]["gap_after_s"] * sr))),
                             "take": c["take"],
                             "seed": c["seed"], "retried": c["retried"], "wer": c["wer"],
                             "transcript": c["heard"][:500]})
        if unit["filler"]:
            fillers.append({"clause": ui, "line": unit["line"], "word": unit["filler"]})
    timings, failed = [], []
    for li, line in enumerate(p["lines"]):
        mine = [ui for ui, u in enumerate(units) if u["line"] == li]
        asked = " ".join(units[ui]["spoken"] for ui in mine)
        heard = " ".join(chosen[ui]["heard"] for ui in mine)
        timings.append({"index": li, "start_s": clauses_meta[mine[0]]["start_s"],
                        "end_s": clauses_meta[mine[-1]]["end_s"], "text": line["text"], "delivery": line["delivery"],
                        "clauses": mine, "wer": wer(asked, heard), "transcript": heard[:500]})
        if cfg["line_max_wer"] is not None and timings[-1]["wer"] > cfg["line_max_wer"]:
            if strict:   # retryable: the next attempt draws new seeds (see `base`)
                raise RuntimeError(f"line {li} came out with WER {timings[-1]['wer']} > {cfg['line_max_wer']} "
                                   f"after per-clause selection: {line['text'][:80]!r} heard as {heard[:80]!r}")
            failed.append(li)
    asked = " ".join(u["spoken"] for u in units)
    return {"audio": audio, "units": units, "lines": timings, "clauses": clauses_meta, "fillers": fillers,
            "wer": wer(asked, " ".join(c["heard"] for c in chosen)), "failed": failed}


def _audio_meta(v: dict, sr: int, variant: str) -> dict:
    return {"seconds": round(len(v["audio"]) / sr, 2), "sample_rate": sr, "synthesis": variant, "lines": v["lines"],
            "clauses": v["clauses"], "fillers": v["fillers"], "wer": v["wer"]}


def speak(payload: dict, ctx: JobContext) -> HandlerResult:
    """The lines in the cloned voice. With payload `compare: true` (the breadth test's whole-line vs clause
    comparison, policy voice.synthesis) the other synthesis is made too, from the same seeds, as a second audio
    output (meta.compare_of = 0); the first output is always the one production uses (assets are read in order),
    and its meta.compare summarises both. The comparison variant never fails the job: a line over the ceiling there
    is listed in its meta.failed_lines."""
    p = validate_speak(payload)
    params, cfg, takes = p["params"], p["speech"], p["takes"]
    tts, asr = _tts(params), _asr(params)
    sr = int(tts.sample_rate)
    ref = ctx.fetch(p["ref_uri"], "reference.wav")
    attempt = max(1, int(getattr(ctx.job, "attempts", 1) or 1))
    variant = "clauses" if cfg["split"] else "whole_line"
    main = _voice_lines(p, cfg, tts, asr, ref, sr, attempt, strict=True)
    work = Path(ctx.workdir)
    meta = _audio_meta(main, sr, variant)
    outputs = [OutputFile(write_wav(work / "speech.wav", main["audio"], sr), "audio", meta)]
    if p["compare"]:
        other = "whole_line" if cfg["split"] else "clauses"
        alt = _voice_lines(p, {**cfg, "split": not cfg["split"]}, tts, asr, ref, sr, attempt, strict=False)
        alt_meta = {**_audio_meta(alt, sr, other), "compare_of": 0, "failed_lines": alt["failed"]}
        outputs.append(OutputFile(write_wav(work / f"speech_{other}.wav", alt["audio"], sr), "audio", alt_meta))
        meta["compare"] = [{k: m[k] for k in ("synthesis", "seconds", "wer")} | {"output": i}
                           for i, m in enumerate((meta, alt_meta))]
    return HandlerResult(outputs,
        recipe={"model": params.get("model", DEFAULT_MODEL), "engine": "voxcpm", "ref_uri": p["ref_uri"],
                "lines": p["lines"], "takes": takes, "seed": p["seed"], "params": params, "synthesis": variant,
                "clone": clone_mode(params), "compare": p["compare"],
                "plan": [{k: u[k] for k in ("line", "spoken", "delivery", "gap_after_s", "breath")}
                         for u in main["units"]]},
        meta={"output_seconds": round(sum(o.meta["seconds"] for o in outputs), 2)})




HANDLERS = {"voice.design": design, "voice.speak": speak}
