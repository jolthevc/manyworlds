"""The voice worker's own ears (manyworlds/voice_eval.py judges what they measure). Heavy imports only here.

Inside the voice job, on the same GPU (no extra job or machine), every spoken line is measured:
  clarity         Whisper large-v3 (the speech gate's model, handlers/tts_voxcpm._asr) -> WER over the words asked for
  naturalness     UTMOS22 strong (sarulab-speech/UTMOS22 via tarepan/SpeechMOS v1.2.0; code and weights MIT, checked
                  2026-11-25) -> a 1-5 MOS estimate
  expressiveness  pitch (librosa pYIN: std, 5-95% range and median, in semitones) and loudness movement (std of frame
                  RMS in dB over the voiced frames); CPU is fine
  consistency     ECAPA-TDNN (handlers/tts_voxcpm._embedder) -> cosine distance to the creator's reference clip
  pacing          scripted words per second of trimmed speech
Casting also asks who it sounds like (voice_lab.cast; never to identify anyone, only to match the bible):
  gender          alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech (Apache-2.0; trained on
                  LibriSpeech, CC-BY-4.0)
  age             empiriolabsai/speaker-age (Apache-2.0; new, unproven: a soft score only, policy voice.judge.fit)
Excluded on licence: audeering age/gender (CC-BY-NC-SA-4.0), torchaudio SQUIM_SUBJECTIVE (CC-BY-NC-4.0).
Tests replace `_mos`, `_gender`, `_age` and `_f0`.
"""

from __future__ import annotations

import os
from typing import Any

from ...speech import words as script_words
from ...speech import wer

RATE = 16000
_state: dict[str, Any] = {}


def _to16k(wav, sr: int):
    from .tts_voxcpm import _resample
    return _resample(wav, sr, RATE)


class _Utmos:
    def __init__(self, repo: str, name: str):
        import torch
        hub = os.environ.get("MW_TORCH_HUB")
        if hub:
            torch.hub.set_dir(hub)
        self.model = torch.hub.load(repo, name, trust_repo=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    def __call__(self, wav, sr: int) -> float:
        import torch
        x = torch.from_numpy(_to16k(wav, sr)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return round(float(self.model(x, RATE).reshape(-1)[0]), 4)


class _AudioClassifier:
    """A Hugging Face audio-classification model -> {label: probability} (lower-cased labels)."""

    def __init__(self, model_id: str):
        import torch
        from transformers import pipeline
        self.pipe = pipeline("audio-classification", model=model_id, device=0 if torch.cuda.is_available() else -1)

    def __call__(self, wav, sr: int) -> dict[str, float]:
        out = self.pipe({"raw": _to16k(wav, sr), "sampling_rate": RATE}, top_k=None)
        return {str(o["label"]).strip().lower(): round(float(o["score"]), 4) for o in out}


def _cached(key: tuple, make):
    if key not in _state:
        _state[key] = make()
    return _state[key]


def _mos(params: dict):
    m = (params.get("judge_models") or {}).get("mos") or {}
    repo, name = m.get("repo", "tarepan/SpeechMOS:v1.2.0"), m.get("name", "utmos22_strong")
    return _cached(("mos", repo, name), lambda: _Utmos(repo, name))


def _gender(params: dict):
    model = (params.get("judge_models") or {}).get("gender",
                                                    "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech")
    return _cached(("gender", model), lambda: _AudioClassifier(model))


def _age(params: dict):
    model = (params.get("judge_models") or {}).get("age", "empiriolabsai/speaker-age")
    return _cached(("age", model), lambda: _AudioClassifier(model))


def _f0(wav, sr: int, params: dict):
    """Voiced pitch in Hz (librosa pYIN at 16 kHz)."""
    import librosa
    import numpy as np
    lo, hi = params.get("pitch_hz", [65, 500])
    f0, voiced, _ = librosa.pyin(_to16k(wav, sr), fmin=float(lo), fmax=float(hi), sr=RATE)
    return f0[np.asarray(voiced, dtype=bool) & np.isfinite(f0)]


def energy_std_db(wav, sr: int, *, frame_ms: float = 25.0, floor_db: float = -45.0) -> float:
    """How much loudness moves: the std of frame RMS in dB over the frames within `floor_db` of the loudest."""
    import numpy as np
    x = np.asarray(wav, dtype=np.float64).reshape(-1)
    n = max(1, int(sr * frame_ms / 1000))
    frames = len(x) // n
    if frames < 2:
        return 0.0
    rms = np.sqrt(np.mean(x[: frames * n].reshape(frames, n) ** 2, axis=1)) + 1e-9
    db = 20 * np.log10(rms)
    db = db[db >= db.max() + floor_db]
    return round(float(np.std(db)), 4) if len(db) > 1 else 0.0


def pitch_stats(f0) -> dict:
    """{f0_std_st, f0_range_st (5-95%), f0_median_st (semitones above 55 Hz), f0_median_hz}; zeros when unvoiced."""
    import numpy as np
    f0 = np.asarray(f0, dtype=np.float64)
    if len(f0) < 2:
        return {"f0_std_st": 0.0, "f0_range_st": 0.0, "f0_median_st": None, "f0_median_hz": None}
    st = 12 * np.log2(f0 / np.median(f0))
    p5, p95 = np.percentile(st, [5, 95])
    med = float(np.median(f0))
    return {"f0_std_st": round(float(np.std(st)), 4), "f0_range_st": round(float(p95 - p5), 4),
            "f0_median_st": round(float(12 * np.log2(med / 55.0)), 4), "f0_median_hz": round(med, 2)}


def measure(wav, sr: int, text: str, heard: str, params: dict, *, ref_embedding: list[float] | None = None,
            embed=None) -> dict:
    """One line's numbers (see the module docstring): {wer, words, seconds, wps, mos, f0_*, energy_std_db,
    ref_distance, transcript}. `wav` is the trimmed take; `embed` the ECAPA embedder (only with a reference)."""
    from ... import voices
    seconds = len(wav) / float(sr) if sr else 0.0
    n = len(script_words(text))
    out = {"wer": wer(text, heard), "words": n, "seconds": round(seconds, 3),
           "wps": round(n / seconds, 3) if seconds > 0 else None, "transcript": (heard or "")[:300]}
    out["mos"] = _mos(params)(wav, sr)
    out.update(pitch_stats(_f0(wav, sr, params)))
    out["energy_std_db"] = energy_std_db(wav, sr)
    if ref_embedding and embed is not None:
        out["ref_distance"] = voices.cosine_distance(embed(wav, sr), ref_embedding)
    return out


def who(wav, sr: int, params: dict) -> dict:
    """Who a clip sounds like, for the bible match: {gender_p: {female, male}, age, age_label}."""
    from ...voice_eval import age_from_label
    # a model whose config has no label names (LABEL_0...) is read through policy judge_models.gender_labels
    names = {str(k).lower(): str(v).lower()
             for k, v in ((params.get("judge_models") or {}).get("gender_labels") or {}).items()}
    g = {names.get(k, k): p for k, p in _gender(params)(wav, sr).items()}
    probs = {"female": max((p for k, p in g.items() if k.startswith(("f", "wom"))), default=0.0),
             "male": max((p for k, p in g.items() if k.startswith(("m", "man"))), default=0.0)}
    if not any(probs.values()):
        raise RuntimeError(f"the gender model's labels {sorted(g)} name neither female nor male: set policy "
                           "voice.judge_models.gender_labels")
    ages = _age(params)(wav, sr)
    label = max(ages.items(), key=lambda kv: kv[1])[0] if ages else None
    weighted = [(age_from_label(k), p) for k, p in ages.items() if age_from_label(k) is not None]
    total = sum(p for _, p in weighted)
    age = round(sum(a * p for a, p in weighted) / total, 1) if total else None
    return {"gender_p": probs, "age": age, "age_label": label}


def release() -> None:
    _state.clear()


class LineJudge:
    """The per-line production check (policy voice.line_check, carried in params.line_check): each chosen take is
    measured and judged (voice_eval.line_verdict); `repairs` regenerations at most, each at the next of
    `strength_steps` (an engine's emotion strength; engines without one only change the seed)."""

    def __init__(self, params: dict, sr: int, *, ref_embedding: list[float] | None = None, embed=None):
        self.params, self.sr, self.cfg = params, int(sr), dict(params.get("line_check") or {})
        lo, hi = self.cfg.get("pace_wps") or [3.4, 4.6]
        self.band = (float(lo), float(hi))
        self.repairs = int(self.cfg.get("repairs", 0))
        self.steps = [float(x) for x in self.cfg.get("strength_steps") or [1.0]]
        self.ref_embedding, self.embed = ref_embedding, embed

    def strength(self, r: int) -> float:
        return self.steps[r % len(self.steps)]

    def __call__(self, text: str, wav, heard: str) -> dict:
        from ...voice_eval import line_verdict
        s = measure(wav, self.sr, text, heard, self.params, ref_embedding=self.ref_embedding, embed=self.embed)
        return {**s, **line_verdict(s, self.cfg, self.band)}
