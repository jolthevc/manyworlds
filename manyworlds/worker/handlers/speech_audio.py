"""Speech tooling shared by every designed-voice engine (handlers/tts_engines.py: voice.speak.qwen3,
voice.speak.indextts, voice.cast), the music handlers and the voice judges. No model of its own: an engine's
`generate` is passed in. VoxCPM2, which this tooling first served, was retired 2026-11-26 by the owner.

The speak payload {ref_uri, ref_text, lines: [{text, delivery, pause_after_s, emotion}], takes, seed, params, embed},
checked by `validate_speak`: one wav of the lines in the cloned voice, plus each line's and each clause's timing.
`embed`: the output's speaker embedding and pitch variance in its meta.

Natural speech: the model's own pauses are not relied on. With synthesis `clauses`, each line is spoken clause by
clause (voices.segments); with `whole_line` (the default: one intonation across the sentence), each line is one
piece. Each piece's takes are scored against its words and the best kept (a clause whose every take is worse than
max_wer x clause_retry_wer_factor gets one more round of seeds); each take is trimmed of its own leading and trailing
silence and faded at its edges, then joined with silences of our choosing: a clause gap or a sentence gap drawn from
its policy range with a seeded RNG (reproducible, not metronomic), the line's own pause_after_s after its last
clause. A line whose assembled WER exceeds line_max_wer fails the job (retried with new seeds). Experimental, off: a
quiet synthetic breath before every n-th sentence, and a rare filler ("um") before a clause, recorded in meta.fillers
so captions leave it out. The per-line check (policy voice.line_check, voice_judge.LineJudge) regenerates a piece
that fails it at the next emotion strength and keeps the better take.

Output meta (`_audio_meta`): lines [{index, start_s, end_s, text, delivery, clauses, wer, transcript, judge,
emotion}], clauses [{index, line, text (scripted words, no filler), start_s, end_s, kind, filler, breath_before,
take, seed, retried, wer, transcript, judge}], fillers [{clause, line, word}], seconds, sample_rate, wer, synthesis,
quality. production/voice.py, production/captioning.py and long_take.py read it.

Transcripts: the speech gate's model (policy review.video.intelligibility.model, Whisper large-v3) run here on the
GPU, scored with the gate's own `speech.wer`. The hosted transcription API is not used from a rented machine: its key
would have to be handed to an untrusted host (CLAUDE.md, deploy/credentials.py).

Speaker embedding: SpeechBrain's ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb; model card and toolkit Apache-2.0,
checked 2026-11-22; trained on VoxCeleb 1+2). Only distances between our own creators' voices are computed with it;
nothing is identified.
"""

from __future__ import annotations

import os
import random
import wave
from pathlib import Path
from typing import Any

from ... import voices
from ...speech import wer
from ..runner import NonRetryable

ASR_RATE = 16000          # Whisper and ECAPA both take 16 kHz
_state: dict[str, Any] = {}


# --- the transcriber and the embedder (heavy imports only here, on the worker; tests replace `_asr`, `_embedder`) ---
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
        from .. import hf_cache
        self.pipe = hf_cache.load(model_id, lambda: pipeline(
            "automatic-speech-recognition", model=model_id, torch_dtype=torch.float16 if cuda else torch.float32,
            device=0 if cuda else -1))
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
        from .. import hf_cache
        self.clf = hf_cache.load(source, lambda: EncoderClassifier.from_hparams(
            source=source, savedir=str(save), run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}))

    def __call__(self, wav, sr: int) -> list[float]:
        import torch
        x = torch.from_numpy(_resample(wav, sr)).float().unsqueeze(0)
        emb = self.clf.encode_batch(x).squeeze().detach().cpu().tolist()
        return [round(float(v), 6) for v in emb]


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


def read_wav(path: Path):
    """A 16-bit PCM wav (what write_wav writes) as (float32 mono samples, sample rate)."""
    import numpy as np
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise NonRetryable(f"{path.name} must be 16-bit PCM, got {8 * w.getsampwidth()}-bit")
        sr, channels, raw = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767
    return (x.reshape(-1, channels).mean(axis=1) if channels > 1 else x), int(sr)


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
        emotion = line.get("emotion")       # an engine's own delivery control (voice_lab.speak_payload): never text
        if emotion is not None:
            if not isinstance(emotion, dict) or not isinstance(emotion.get("name", ""), str):
                raise NonRetryable(f"line {i} emotion must be {{name, vector, alpha}}")
            vec = emotion.get("vector")
            if vec is not None and (not isinstance(vec, list) or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) for x in vec)):
                raise NonRetryable(f"line {i} emotion.vector must be a list of numbers")
        out.append({"text": words, "delivery": (delivery or "").strip() or inline,
                    "pause_after_s": None if pause is None else float(pause),
                    **({"emotion": emotion} if emotion is not None else {})})
    takes = _positive_int(payload.get("takes", 1), "takes")
    embed = payload.get("embed", False)
    if not isinstance(embed, bool):
        raise NonRetryable(f"embed must be true or false, got {embed!r}")
    return {"ref_uri": ref_uri, "ref_text": voices.strip_delivery(ref_text), "lines": out, "takes": takes,
            "seed": _seed(payload), "params": params, "speech": speech_config(params), "embed": embed}


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
    Every part is off when its key is absent, so a payload without these keys is spoken as whole lines, joined with
    `pause_s`, untrimmed."""
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


# --- speaking the lines --------------------------------------------------------------------------------------------
REPAIR_SEED_OFFSET = 1 << 20     # a line check's regenerations draw seeds from their own block


def _voice_lines(p: dict, cfg: dict, asr, sr: int, attempt: int, *, strict: bool, generate, judge=None) -> dict:
    """Every line spoken as `cfg` says (whole lines or clauses), the best take of each piece kept, joined with the
    planned gaps: {audio, units, lines (timings), clauses, fillers, wer, failed}. `strict`: a line over the
    line ceiling fails the job (retryable); otherwise it is recorded in `failed`.

    `generate(unit, line, seed, strength) -> wav`: the engine's voice (handlers/tts_engines.py), given only the words
    (a delivery is never spoken, so it is never in `spoken`). `judge` (voice_judge.LineJudge, policy
    voice.line_check): the chosen take of each piece is measured (clarity, naturalness, expressiveness, consistency,
    pacing); a piece that fails is regenerated with new seeds and the next emotion strength, at most `judge.repairs`
    times, and the best take kept (voice_eval.better). Its scores go into the clause's and the line's meta
    (`judge`)."""
    import numpy as np

    takes = p["takes"]
    units = plan(p["lines"], cfg, p["seed"])

    def take(unit: dict, k: int, seed: int, strength: float = 1.0):
        wav = generate(unit, p["lines"][unit["line"]], seed, strength)
        heard = asr(wav, sr)
        # against the words asked for only
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

        def finish(wav):
            if cfg["trim"]:
                wav = trim(wav, sr, **cfg["trim"])
            if cfg["fade_ms"]:
                wav = fade(wav, sr, cfg["fade_ms"])
            return wav

        wav = finish(wav)
        pick = {"wav": wav, "wer": score, "take": k, "seed": seed, "heard": heard, "retried": retried}
        if judge is not None:
            from ...voice_eval import better
            pick["judge"] = {**judge(unit["spoken"], wav, heard), "repairs": 0}
            for r in range(judge.repairs):
                if pick["judge"]["passed"]:
                    break
                strength = judge.strength(r)
                s2 = p["seed"] + REPAIR_SEED_OFFSET + ((attempt - 1) * len(units) + ui) * judge.repairs + r
                w2, k2, seed2, raw, heard2 = take(unit, 2 * takes + r, s2, strength)
                raw = finish(raw)
                v = {**judge(unit["spoken"], raw, heard2), "repairs": r + 1, "strength": strength}
                if better(v, pick["judge"]):
                    pick = {"wav": raw, "wer": w2, "take": k2, "seed": seed2, "heard": heard2, "retried": retried,
                            "judge": v}
                else:
                    pick["judge"]["repairs"] = r + 1
        chosen.append(pick)

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
        if c.get("judge") is not None:
            clauses_meta[-1]["judge"] = {k: v for k, v in c["judge"].items() if k != "transcript"}
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
        judged = [chosen[ui]["judge"] for ui in mine if chosen[ui].get("judge") is not None]
        if judged:
            timings[-1]["judge"] = {
                "passed": all(j["passed"] for j in judged),
                "composite": round(sum(j["composite"] for j in judged) / len(judged), 4),
                "reasons": [r for j in judged for r in j["reasons"]],
                "repairs": sum(int(j.get("repairs") or 0) for j in judged),
                **{k: judged[0].get(k) for k in ("mos", "f0_std_st", "f0_range_st", "energy_std_db", "wps",
                                                 "ref_distance") if len(judged) == 1}}
        if line.get("emotion"):
            timings[-1]["emotion"] = (line["emotion"] or {}).get("name")
        if cfg["line_max_wer"] is not None and timings[-1]["wer"] > cfg["line_max_wer"]:
            if strict:   # retryable: the next attempt draws new seeds (see `base`)
                raise RuntimeError(f"line {li} came out with WER {timings[-1]['wer']} > {cfg['line_max_wer']} "
                                   f"after per-clause selection: {line['text'][:80]!r} heard as {heard[:80]!r}")
            failed.append(li)
    asked = " ".join(u["spoken"] for u in units)
    return {"audio": audio, "units": units, "lines": timings, "clauses": clauses_meta, "fillers": fillers,
            "wer": wer(asked, " ".join(c["heard"] for c in chosen)), "failed": failed}


def line_judge(params: dict, ref: Path, sr: int):
    """The per-line check (params.line_check, policy voice.line_check) for a job, or None when it is off. Its
    consistency score compares every line with the reference clip's ECAPA embedding."""
    if not (params.get("line_check") or {}).get("enabled"):
        return None
    from . import voice_judge
    wav, rate = read_wav(ref)
    embed = _embedder(params)
    return voice_judge.LineJudge(params, sr, ref_embedding=embed(wav, rate), embed=embed)


def _audio_meta(v: dict, sr: int, variant: str) -> dict:
    out = {"seconds": round(len(v["audio"]) / sr, 2), "sample_rate": sr, "synthesis": variant, "lines": v["lines"],
           "clauses": v["clauses"], "fillers": v["fillers"], "wer": v["wer"]}
    judged = [line["judge"] for line in v["lines"] if line.get("judge")]
    if judged:   # for quality-per-accepted-minute stats (voice_lab.quality_stats)
        out["quality"] = {"lines": len(judged), "passed": sum(j["passed"] for j in judged),
                          "composite": round(sum(j["composite"] for j in judged) / len(judged), 4),
                          "repairs": sum(j["repairs"] for j in judged)}
    return out


def release() -> None:
    """Drop Whisper, the embedder and the line judges before another family's job, and hand their GPU memory back."""
    _state.clear()
    from . import voice_judge
    voice_judge.release()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - no torch or no GPU: nothing to free
        pass
