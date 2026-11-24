"""A song's anatomy, measured on the CPU (control plane): sound, structure, vocal and lyric features.

    measure(mix_path, vocal_path, lyrics, policy=, title=)  -> {"features": {name: value}, "detail": {...}}
    lyric_features(lyrics, title)                            -> {name: value} from the text alone (no audio)

The feature names are the registry's (policy music.features: name, source, unit, kind); measure() computes every
one of them it can and leaves the rest out (no vocal stem: no vocal features). Every threshold, window and band
edge is in policy music.measure. Libraries: librosa (ISC), pyloudnorm (MIT), numpy and scipy (BSD), imported
lazily: the control-plane image has them (extra `music`), nothing else needs them.

Sound: duration, tempo and its stability (beat intervals), key and mode (Krumhansl-Schmuckler profiles over the
chroma, with the correlation margin as confidence), integrated loudness (BS.1770, pyloudnorm), true peak (4x
oversampled), loudness range (short-term loudness over music.measure.lra windows, gated, p95 - p10), crest factor,
clipping share, spectral balance (energy share per band in music.measure.spectral_bands), centroid, stereo width
(side over mid), L/R correlation and the level lost in a mono sum, onset density.
Structure: block features (chroma, MFCC, RMS) are segmented (librosa agglomerative clustering), segments with
alike mean features share a label, adjacent equal labels merge into sections; the chorus is the loudest label that
repeats. Repetition: librosa's recurrence matrix (share of blocks that recur at least a minimum lag away) and the
mean best self-similarity at that lag.
Vocal (from the vocal stem): first vocal entry, presence, clarity (vocal stem over the rest of the mix while the
voice sings, dB), singability (sung syllables per beat).
Lyrics: repetition ratio (repeated lines), rhyme density (orthographic rime of line endings, adjacent or alternate
lines in a section), syllables per line (vowel groups), title in the hook (a chorus or hook line holds the title).
"""

from __future__ import annotations

import math
import re
from typing import Any

from . import music
from .policy import Policy

VERSION = "m1"   # bump when a feature's measurement changes: stored features are versioned by it (song_features)

_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)   # Krumhansl-Kessler profiles
_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)
_PITCHES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


def cfg(policy: Policy | None) -> dict:
    return (policy.get("music", "measure", default={}) or {}) if policy is not None else {}


def registry(policy: Policy | None) -> list[dict]:
    """The feature registry (policy music.features): [{name, source, unit, kind}]."""
    return list((policy.get("music", "features", default=[]) or []) if policy is not None else [])


def numeric_names(policy: Policy | None, *, sources: tuple[str, ...] | None = None) -> list[str]:
    """Registry features that are numbers or flags (what analysis and similarity compare), optionally by source."""
    return [f["name"] for f in registry(policy) if f.get("kind", "number") in ("number", "flag")
            and (sources is None or f.get("source") in sources)]


def _r(x, nd: int = 3):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, nd)


# --- lyrics --------------------------------------------------------------------------------------------------------

_VOWELS = re.compile(r"[aeiouy]+")


def syllables(word: str) -> int:
    """Vowel groups, a silent final e dropped: an estimate, good enough to compare lines and songs."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    n = len(_VOWELS.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    return max(1, n)


def _rime(word: str) -> str:
    """The orthographic rime: from the last vowel group to the end ('night' -> 'ight', 'stars' -> 'ars')."""
    w = re.sub(r"[^a-z]", "", word.lower())
    w = re.sub(r"(?<=[^aeiou])s$", "", w) if len(w) > 3 else w   # plural s: 'stars' rhymes with 'ours'... nearly
    m = list(re.finditer(r"[aeiouy]+[^aeiouy]*$", w))
    return m[-1].group(0) if m else w


def _rhymes(a: str, b: str, min_chars: int) -> bool:
    if not a or not b or a == b:
        return False
    ra, rb = _rime(a), _rime(b)
    return ra == rb and len(ra) >= min_chars or (len(a) >= min_chars + 1 and a[-(min_chars + 1):] == b[-(min_chars + 1):])


def lyric_features(lyrics: str, title: str = "", *, policy: Policy | None = None) -> dict:
    """What the words show without listening: lines, repetition, rhyme, syllables, the title in the hook."""
    c = cfg(policy).get("lyrics") or {}
    min_chars = int(c.get("rhyme_min_chars", 2))
    hook_tags = tuple(c.get("hook_sections") or ("chorus", "hook", "post-chorus"))
    sections = music.parse(lyrics)
    lines = [(music.section_name(tag), re.sub(r"\([^)]*\)", " ", line)) for tag, ls in sections for line in ls]
    lines = [(t, line) for t, line in lines if music._plain(line)]
    if not lines:
        return {}
    plain = [music._plain(line) for _, line in lines]
    total = len(plain)
    seen: dict[str, int] = {}
    for p in plain:
        seen[p] = seen.get(p, 0) + 1
    repeated = sum(n for n in seen.values() if n > 1)
    pairs = hits = 0
    for _, ls in sections:
        ends = [music._plain(re.sub(r"\([^)]*\)", " ", x)).split()[-1:] for x in ls]
        ends = [e[0] for e in ends if e]
        for i in range(len(ends) - 1):
            pairs += 1
            hits += _rhymes(ends[i], ends[i + 1], min_chars) or (i + 2 < len(ends) and
                                                                _rhymes(ends[i], ends[i + 2], min_chars))
    syl = [sum(syllables(w) for w in p.split()) for p in plain]
    title_plain = music._plain(title)
    hook_lines = [music._plain(line) for t, line in lines if t in hook_tags]
    return {"line_count": total,
            "repetition_ratio": _r(repeated / total),
            "rhyme_density": _r(hits / pairs) if pairs else None,
            "syllables_per_line": _r(sum(syl) / total, 2),
            "sung_syllables": int(sum(syl)),
            "title_in_hook": (1 if title_plain and any(f" {title_plain} " in f" {h} " for h in hook_lines) else 0)
            if title_plain else None,
            "written_sections": sum(1 for tag, _ in sections if tag),
            "written_choruses": sum(1 for tag, _ in sections if music.section_name(tag) in hook_tags)}


# --- sound ---------------------------------------------------------------------------------------------------------

def _db(x: float, floor: float = 1e-10) -> float:
    return 20.0 * math.log10(max(float(x), floor))


def _key(chroma) -> tuple[str, str, float, list[dict]]:
    import numpy as np
    prof = np.asarray(chroma).mean(axis=1)
    scores = []
    for mode, p in (("major", _MAJOR), ("minor", _MINOR)):
        for i in range(12):
            r = float(np.corrcoef(prof, np.roll(np.asarray(p), i))[0, 1])
            scores.append((0.0 if math.isnan(r) else r, _PITCHES[i], mode))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    return f"{best[1]} {best[2]}", best[2], best[0] - second[0], [
        {"key": f"{s[1]} {s[2]}", "r": round(s[0], 3)} for s in scores[:3]]


def _loudness_range(data, sr: int, c: dict) -> float | None:
    """EBU-style loudness range: short-term loudness in `window_s` windows every `hop_s`, absolute and relative
    gates, then the spread between the low and high percentiles."""
    import numpy as np
    import pyloudnorm as pyln
    win, hop = int(sr * float(c.get("window_s", 3.0))), int(sr * float(c.get("hop_s", 1.0)))
    if len(data) < win:
        return None
    meter = pyln.Meter(sr)
    vals = []
    for start in range(0, len(data) - win + 1, hop):
        v = meter.integrated_loudness(data[start:start + win])
        if not math.isinf(v) and v > float(c.get("absolute_gate_lufs", -70)):
            vals.append(v)
    if len(vals) < 2:
        return None
    gate = 10 * math.log10(np.mean(np.power(10, np.asarray(vals) / 10))) + float(c.get("relative_gate_lu", -20))
    vals = [v for v in vals if v > gate]
    if len(vals) < 2:
        return None
    lo, hi = c.get("percentiles") or (10, 95)
    return float(np.percentile(vals, hi) - np.percentile(vals, lo))


def _sound(stereo, sr: int, c: dict) -> dict:
    import numpy as np
    import pyloudnorm as pyln
    from scipy.signal import resample_poly
    chans = stereo if stereo.ndim == 2 else stereo[None, :]
    data = chans.T.astype("float64")                       # (samples, channels): pyloudnorm's layout
    mono = chans.mean(axis=0)
    out: dict[str, Any] = {"duration_s": _r(chans.shape[1] / sr, 2)}
    lufs = pyln.Meter(sr).integrated_loudness(data) if chans.shape[1] >= int(sr * 0.4) else float("-inf")
    out["lufs_integrated"] = _r(lufs, 2)
    up = int(c.get("true_peak_oversample", 4))
    tp = max(float(np.max(np.abs(resample_poly(ch, up, 1)))) for ch in chans)
    peak = float(np.max(np.abs(chans)))
    rms = float(np.sqrt(np.mean(mono ** 2)))
    out["true_peak_dbtp"] = _r(_db(tp), 2)
    out["sample_peak_dbfs"] = _r(_db(peak), 2)
    out["crest_factor_db"] = _r(_db(peak) - _db(rms), 2) if rms > 0 else None
    out["clipping_share"] = _r(float(np.mean(np.abs(chans) >= float(c.get("clip_level", 0.999)))), 5)
    out["loudness_range_lu"] = _r(_loudness_range(data, sr, c.get("lra") or {}), 2)
    if chans.shape[0] >= 2:
        left, right = chans[0], chans[1]
        mid, side = (left + right) / 2, (left - right) / 2
        rm, rs = float(np.sqrt(np.mean(mid ** 2))), float(np.sqrt(np.mean(side ** 2)))
        both = math.sqrt((float(np.mean(left ** 2)) + float(np.mean(right ** 2))) / 2)
        out["stereo_width"] = _r(rs / rm if rm > 0 else 0.0)
        corr = np.corrcoef(left, right)[0, 1] if np.std(left) > 0 and np.std(right) > 0 else 1.0
        out["stereo_correlation"] = _r(corr)
        out["mono_loss_db"] = _r(_db(both) - _db(rm), 2) if rm > 0 else None
    else:
        out.update(stereo_width=0.0, stereo_correlation=1.0, mono_loss_db=0.0)
    return out


def _blocks(feat, block: int):
    """Mean of every `block` frames: (dims, frames) -> (dims, blocks)."""
    import numpy as np
    n = feat.shape[1] // block
    if n == 0:
        return feat.mean(axis=1, keepdims=True)
    return feat[:, :n * block].reshape(feat.shape[0], n, block).mean(axis=2)


def _structure(y, sr: int, hop: int, chroma, mfcc, rms, c: dict) -> tuple[dict, dict]:
    import librosa
    import numpy as np
    s = c.get("structure") or {}
    block = max(1, int(round(float(s.get("block_s", 0.5)) * sr / hop)))
    block_s = block * hop / sr
    stack = np.vstack([chroma, mfcc, librosa.amplitude_to_db(rms, ref=1.0) / 10.0])
    X = _blocks(stack, block)
    n = X.shape[1]
    Z = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-9)
    dur = len(y) / sr
    k = int(min(max(round(dur / float(s.get("section_s", 6.0))), int(s.get("min_segments", 2))),
                int(s.get("max_segments", 24)), n))
    if n < 4 or k < 2:
        return {"section_count": 1}, {"sections": []}
    starts = sorted(set(int(b) for b in librosa.segment.agglomerative(Z, k)) | {0})
    bounds = list(zip(starts, starts[1:] + [n]))
    means = [Z[:, a:b].mean(axis=1) for a, b in bounds]
    labels: list[int] = []
    protos: list = []
    thr = float(s.get("label_similarity", 0.8))
    for m in means:
        sims = [float(np.dot(m, p) / (np.linalg.norm(m) * np.linalg.norm(p) + 1e-9)) for p in protos]
        best = int(np.argmax(sims)) if sims else -1
        if best >= 0 and sims[best] >= thr:
            labels.append(best)
        else:
            protos.append(m)
            labels.append(len(protos) - 1)
    rms_db = librosa.amplitude_to_db(_blocks(rms, block)[0], ref=1.0)
    sections: list[dict] = []
    for (a, b), lab in zip(bounds, labels):
        if sections and sections[-1]["label"] == lab:
            sections[-1]["end_b"] = b
        else:
            sections.append({"label": lab, "start_b": a, "end_b": b})
    for sec in sections:
        sec["start_s"], sec["end_s"] = round(sec["start_b"] * block_s, 2), round(sec["end_b"] * block_s, 2)
        sec["rms_db"] = round(float(np.mean(rms_db[sec["start_b"]:sec["end_b"]])), 2)
    min_len = float(s.get("min_section_s", 2.0))
    counts: dict[int, int] = {}
    for sec in sections:
        if sec["end_s"] - sec["start_s"] >= min_len:
            counts[sec["label"]] = counts.get(sec["label"], 0) + 1
    repeated = [lab for lab, cnt in counts.items() if cnt >= 2]
    chorus = None
    if repeated:
        loud = {lab: float(np.mean([x["rms_db"] for x in sections if x["label"] == lab])) for lab in repeated}
        chorus = max(repeated, key=lambda lab: (loud[lab], counts[lab]))
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for sec in sections:
        sec["name"] = "chorus" if sec["label"] == chorus else letters[sec["label"] % 26]
        del sec["start_b"], sec["end_b"]
    # Repetition: blocks that recur at least min_lag away (librosa recurrence), and how alike their best match is.
    lag = max(1, int(round(float(s.get("min_lag_s", 4.0)) / block_s)))
    out: dict[str, Any] = {"section_count": len(sections)}
    if n > 2 * lag + 1:
        R = librosa.segment.recurrence_matrix(Z, width=lag, sym=True, mode="connectivity")
        out["repetition_share"] = _r(float(np.mean(np.asarray(R).sum(axis=1) > 0)))
        U = Z / (np.linalg.norm(Z, axis=0, keepdims=True) + 1e-9)
        S = U.T @ U
        idx = np.arange(n)
        S[np.abs(idx[:, None] - idx[None, :]) < lag] = -1.0
        out["self_similarity"] = _r(float(np.mean(np.clip(S.max(axis=1), 0, 1))))
    choruses = [x for x in sections if x["name"] == "chorus"]
    out["chorus_repeats"] = len([x for x in choruses if x["end_s"] - x["start_s"] >= min_len])
    if choruses:
        out["time_to_first_chorus_s"] = choruses[0]["start_s"]
        rest = [x["rms_db"] for x in sections if x["name"] != "chorus"]
        if rest:
            out["chorus_energy_lift_db"] = _r(float(np.mean([x["rms_db"] for x in choruses]) - np.mean(rest)), 2)
    if sections:
        out["energy_range_db"] = _r(max(x["rms_db"] for x in sections) - min(x["rms_db"] for x in sections), 2)
    return out, {"sections": sections}


def _vocal(vocal_path: str, mix_mono, sr: int, hop: int, c: dict) -> dict:
    import librosa
    import numpy as np
    v = c.get("vocal") or {}
    voc, _ = librosa.load(vocal_path, sr=sr, mono=True)
    n = min(len(voc), len(mix_mono))
    voc, mix = voc[:n], mix_mono[:n]
    frame = int(sr * float(v.get("frame_s", 0.05)))
    if n < frame * 4:
        return {}
    r = librosa.feature.rms(y=voc, frame_length=frame, hop_length=frame)[0]
    rest = librosa.feature.rms(y=mix - voc, frame_length=frame, hop_length=frame)[0]
    db = librosa.amplitude_to_db(r, ref=1.0)
    thr = max(float(np.max(db)) - float(v.get("active_below_peak_db", 30)), float(v.get("floor_dbfs", -50)))
    active = db > thr
    out: dict[str, Any] = {"vocal_presence_share": _r(float(np.mean(active)))}
    run = max(1, int(round(float(v.get("min_entry_s", 0.5)) / (frame / sr))))
    for i in range(len(active) - run + 1):
        if active[i:i + run].all():
            out["time_to_first_vocal_s"] = _r(i * frame / sr, 2)
            break
    if active.any():
        vr = float(np.sqrt(np.mean(r[active] ** 2)))
        ir = float(np.sqrt(np.mean(rest[: len(r)][active] ** 2)))
        out["vocal_clarity_db"] = _r(_db(vr) - _db(ir), 2) if ir > 0 else None
        out["vocal_seconds"] = _r(float(np.sum(active)) * frame / sr, 2)
    return out


def measure(mix_path: str, vocal_path: str | None = None, lyrics: str | None = None, *,
            policy: Policy | None = None, title: str = "") -> dict:
    """The song's features ({"features": {...}, "detail": {...}}): see the module docstring."""
    import librosa
    import numpy as np
    c = cfg(policy)
    stereo, native = librosa.load(mix_path, sr=None, mono=False)
    stereo = np.asarray(stereo, dtype="float32")
    feats: dict[str, Any] = _sound(stereo, int(native), c)
    sr, hop = int(c.get("sample_rate", 22050)), int(c.get("hop_length", 512))
    mono = librosa.to_mono(stereo) if stereo.ndim == 2 else stereo
    y = librosa.resample(mono, orig_sr=int(native), target_sr=sr) if int(native) != sr else mono
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    feats["tempo_bpm"] = _r(float(np.atleast_1d(tempo)[0]), 1)
    if len(beats) > 2:
        ibi = np.diff(librosa.frames_to_time(beats, sr=sr, hop_length=hop))
        feats["tempo_stability"] = _r(1.0 - float(np.std(ibi) / (np.mean(ibi) + 1e-9)))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
    key, mode, conf, candidates = _key(chroma)
    feats.update(key=key, mode=mode, key_confidence=_r(conf))
    spec = np.abs(librosa.stft(y, n_fft=int(c.get("n_fft", 2048)), hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=int(c.get("n_fft", 2048)))
    total = float(spec.sum()) or 1.0
    for name, (lo, hi) in (c.get("spectral_bands") or {}).items():
        feats[f"spectral_{name}_share"] = _r(float(spec[(freqs >= lo) & (freqs < hi)].sum()) / total)
    feats["spectral_centroid_hz"] = _r(float(np.mean(librosa.feature.spectral_centroid(S=np.sqrt(spec), sr=sr))), 1)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, units="time")
    feats["onset_density"] = _r(len(onsets) / max(len(y) / sr, 1e-9))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=int(c.get("n_mfcc", 13)))
    rms = librosa.feature.rms(y=y, hop_length=hop)
    n = min(chroma.shape[1], mfcc.shape[1], rms.shape[1])
    struct, detail = _structure(y, sr, hop, chroma[:, :n], mfcc[:, :n], rms[:, :n], c)
    feats.update(struct)
    detail["key_candidates"] = candidates
    if vocal_path:
        vocal = _vocal(vocal_path, y, sr, hop, c)
        feats.update({k: v for k, v in vocal.items() if k != "vocal_seconds"})
        detail["vocal_seconds"] = vocal.get("vocal_seconds")
    if lyrics and not music.is_instrumental({"lyrics": lyrics}):
        words = lyric_features(lyrics, title, policy=policy)
        sung = words.pop("sung_syllables", None)
        feats.update(words)
        secs, bpm = detail.get("vocal_seconds"), feats.get("tempo_bpm")
        if sung and secs and bpm:
            feats["syllables_per_beat"] = _r(sung / (secs * bpm / 60.0))
    return {"features": {k: v for k, v in feats.items() if v is not None}, "detail": detail, "version": VERSION}
