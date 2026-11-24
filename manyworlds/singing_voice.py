"""One singing voice per artist (policy `music.voice`, OFF): the conversion and training job payloads, the licence
rule for converter adapters, and the similarity maths. No heavy imports: the worker (worker/handlers/music_voice.py)
embeds and converts, the creator flow (creator/music_voice.py) builds and stores the voice asset, production
(production/song.py) follows a song with its conversion.

  kinds        music.convert_voice: a song's vocal stem (and instrumental) + the artist's voice asset -> the vocal in
               her voice (source pitch kept) and the new mix. music.train_voice: an RVC model from her sung clips.
  adapters     soulx_svc (SoulX-Singer-SVC, zero-shot from the asset's prompt clip) and rvc (a model trained per
               artist). Each carries its licence and `licence_verified`: an adapter whose code, weights or auxiliary
               weights (feature extractor, F0 model, base models) were not all read from a primary source and found
               fit for commercial use is refused here and on the worker, unless music.voice.allow_unverified_licences
               (the owner's call, never the code's).
  voice asset  {version, adapter, centroid, embedding_model, prompt_uri, model_uri, index_uri}: creators.anchors.
               music_voice, one row per version in music_voices.
  similarity   speaker embeddings (ECAPA-TDNN) of voiced windows of a vocal stem, each window's cosine similarity to
               the asset's centroid, the median (singing drifts: vibrato, range, stem bleed; a median shrugs off a
               few bad windows). `calibrate` compares same-artist with other-artist scores and proposes a threshold;
               `gate` is report-only until music.voice.gate.enforce (and a calibrated min_similarity).
"""

from __future__ import annotations

import math
import statistics
from typing import Iterable

from .policy import Policy

KINDS = {"convert": "music.convert_voice", "train": "music.train_voice"}
ADAPTERS = ("soulx_svc", "rvc")      # converters (worker/handlers/music_voice.CONVERTERS)
TRAINERS = ("rvc",)                  # adapters that need a model trained per artist
ROLES = ("vocals", "mix")            # a music.convert_voice job's outputs, in this order (deterministic keys)


class VoiceRefused(ValueError):
    """A conversion or training that may not run (adapter unknown, licence unverified, no voice asset)."""


def cfg(policy: Policy | None) -> dict:
    return (policy.get("music", "voice", default={}) or {}) if policy is not None else {}


def enabled(policy: Policy | None) -> bool:
    """Whether production follows a song with its conversion (music.voice.enabled, off as shipped)."""
    return bool(cfg(policy).get("enabled"))


def kind(policy: Policy | None, what: str) -> str:
    return str((cfg(policy).get("kinds") or {}).get(what) or KINDS[what])


# --- adapters and their licences ----------------------------------------------------------------------------------

def licence_allowed(settings: dict, *, allow_unverified: bool) -> bool:
    """Pure: an adapter runs only when its licences were verified, or the owner allowed unverified ones."""
    return bool(settings.get("licence_verified")) or bool(allow_unverified)


def adapter_settings(policy: Policy, name: str | None = None) -> dict:
    """What the worker needs of a converter or trainer (music.voice.adapters.<name>), with its name and the licence
    switch resolved into `allow_unverified`. Raises VoiceRefused for an unknown adapter or an unverified licence."""
    c = cfg(policy)
    name = str(name or c.get("adapter") or "soulx_svc")
    a = dict((c.get("adapters") or {}).get(name) or {})
    if name not in ADAPTERS or not a:
        raise VoiceRefused(f"no voice adapter {name!r} (music.voice.adapters; known: {list(ADAPTERS)})")
    allow = bool(c.get("allow_unverified_licences", False))
    if not licence_allowed(a, allow_unverified=allow):
        raise VoiceRefused(f"voice adapter {name!r} has licence_verified: false ({a.get('licence_notes') or 'unchecked'}); "
                           "the owner may set music.voice.allow_unverified_licences after reading the terms")
    return {"adapter": name, **a, "allow_unverified": allow}


def adapter_for(policy: Policy, asset: dict) -> str:
    """The converter for an asset: music.voice.adapter, where `auto` means rvc when the asset has a trained model
    and soulx_svc otherwise."""
    name = str(cfg(policy).get("adapter") or "auto")
    if name == "auto":
        return "rvc" if asset.get("model_uri") else "soulx_svc"
    return name


def similarity_settings(policy: Policy) -> dict:
    """The worker's embedding settings (music.voice.similarity): model, windows, the voiced-frame rule."""
    return dict(cfg(policy).get("similarity") or {})


def asset_fields(asset: dict) -> dict:
    """The parts of a voice asset a worker needs (no ids, no calibration)."""
    return {k: asset.get(k) for k in ("version", "adapter", "centroid", "embedding_model", "prompt_uri", "model_uri",
                                      "index_uri")}


def convert_payload(policy: Policy, asset: dict, *, seed: int, vocals_uri: str | None = None,
                    instrumental_uri: str | None = None, song: dict | None = None, adapter: str | None = None,
                    lyrics: str | None = None, check: bool = False) -> dict:
    """The music.convert_voice job: a song's stems into the artist's voice (checked: VoiceRefused when the adapter
    may not run, the asset lacks what the adapter needs, or a stem is missing). With `song` (a music.song payload,
    the voice bench) the job makes the song itself first, on the same machine, and needs no stems."""
    if not song and (not vocals_uri or not instrumental_uri):
        raise VoiceRefused("a conversion needs the song's vocal and instrumental stems (or a song to make)")
    if not asset or not asset.get("prompt_uri") or not asset.get("centroid"):
        raise VoiceRefused("the artist has no voice asset (prompt clip and centroid): creator/music_voice builds one")
    name = adapter or adapter_for(policy, asset)
    settings = adapter_settings(policy, name)
    if name in TRAINERS and not asset.get("model_uri"):
        raise VoiceRefused(f"voice adapter {name!r} needs a trained model; this asset (v{asset.get('version')}) has none")
    c = cfg(policy)
    payload = {"asset": asset_fields(asset), "adapter": settings, "seed": int(seed) % (2**31),
               "similarity": similarity_settings(policy), "remix": dict(c.get("remix") or {})}
    if song:
        payload["song"] = dict(song)
    else:
        payload.update(vocals_uri=vocals_uri, instrumental_uri=instrumental_uri)
    if check and lyrics:
        mc = policy.get("music", "check", default={}) or {}
        payload["check"] = {"lyrics": lyrics, "asr_model": mc.get("asr_model"),
                            "min_line_ratio": float(mc.get("min_line_ratio", 0.7))}
    return payload


def train_payload(policy: Policy, clips: list[str], *, seed: int, name: str) -> dict:
    """The music.train_voice job: an RVC model from the artist's sung clips (uris of converted, filtered vocals)."""
    c = cfg(policy)
    settings = adapter_settings(policy, str(c.get("trainer") or "rvc"))
    if settings["adapter"] not in TRAINERS:
        raise VoiceRefused(f"voice adapter {settings['adapter']!r} trains nothing")
    if len(clips) < int((c.get("creation") or {}).get("min_train_clips", 1)):
        raise VoiceRefused(f"{len(clips)} clips, fewer than music.voice.creation.min_train_clips")
    return {"clips": list(clips), "adapter": settings, "seed": int(seed) % (2**31), "name": str(name),
            "similarity": similarity_settings(policy)}


# --- the maths (pure; synthetic embeddings in the tests) -----------------------------------------------------------

def _norm(v: Iterable[float]) -> list[float]:
    v = [float(x) for x in v]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    """Cosine similarity (1 = same direction, 0 = unrelated); 0 for an empty or zero vector."""
    a, b = list(a), list(b)
    if not a or not b or len(a) != len(b):
        return 0.0
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def centroid(vectors: Iterable[Iterable[float]]) -> list[float]:
    """The artist's centroid: the mean of the unit-length embeddings, scaled back to unit length (each window counts
    the same however loud it was)."""
    vs = [_norm(v) for v in vectors if v]
    if not vs:
        return []
    dim = len(vs[0])
    vs = [v for v in vs if len(v) == dim]
    mean = [sum(v[i] for v in vs) / len(vs) for i in range(dim)]
    return [round(x, 6) for x in _norm(mean)]


def _quantile(values: list[float], q: float) -> float:
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def score_windows(window_embeddings: list[list[float]], center: list[float], *, min_windows: int = 1) -> dict:
    """A stem against a centroid: each voiced window's cosine similarity, the median (the score), p10 and p90.
    `similarity` is None when fewer than `min_windows` windows were voiced (nothing to judge)."""
    sims = [round(cosine(w, center), 4) for w in window_embeddings if w]
    if len(sims) < max(1, int(min_windows)) or not center:
        return {"similarity": None, "windows": len(sims), "p10": None, "p90": None, "each": sims}
    return {"similarity": round(statistics.median(sims), 4), "windows": len(sims),
            "p10": round(_quantile(sims, 0.1), 4), "p90": round(_quantile(sims, 0.9), 4), "each": sims}


def pairwise(embeddings: dict[str, list[float]]) -> list[dict]:
    """Every pair of songs' embeddings (each a stem's centroid): [{a, b, similarity}], in name order."""
    names = sorted(k for k, v in embeddings.items() if v)
    return [{"a": a, "b": b, "similarity": round(cosine(embeddings[a], embeddings[b]), 4)}
            for i, a in enumerate(names) for b in names[i + 1:]]


def calibrate(same: list[float], other: list[float], *, low_q: float = 0.1, high_q: float = 0.9) -> dict:
    """Same-artist scores (a voice's songs against its own centroid) vs other-artist scores (against another
    artist's or seed's centroid). `separable` when the same-artist low quantile sits above the other-artist high
    quantile; the proposed `threshold` is the midpoint between them (None when they overlap: the encoder cannot
    tell these voices apart on this data, so no gate should be enforced)."""
    same = [float(x) for x in same if x is not None]
    other = [float(x) for x in other if x is not None]
    if not same or not other:
        return {"same": len(same), "other": len(other), "separable": False, "threshold": None}
    lo, hi = _quantile(same, low_q), _quantile(other, high_q)
    out = {"same": len(same), "other": len(other), "same_median": round(statistics.median(same), 4),
           "same_low": round(lo, 4), "other_median": round(statistics.median(other), 4), "other_high": round(hi, 4),
           "gap": round(lo - hi, 4), "separable": lo > hi}
    out["threshold"] = round((lo + hi) / 2, 4) if lo > hi else None
    return out


def gate(similarity: float | None, policy: Policy) -> dict:
    """The voice gate's verdict: report-only (`enforced` false) until music.voice.gate.enforce and a calibrated
    min_similarity. `passed` is None when there is nothing to compare (no score, or no threshold yet)."""
    g = cfg(policy).get("gate") or {}
    threshold = g.get("min_similarity")
    enforced = bool(g.get("enforce")) and threshold is not None
    if similarity is None or threshold is None:
        return {"similarity": similarity, "threshold": threshold, "passed": None, "enforced": enforced}
    return {"similarity": similarity, "threshold": float(threshold), "passed": similarity >= float(threshold),
            "enforced": enforced}
