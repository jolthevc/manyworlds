"""Automatic voice judgement (owner 2026-11-25: no human selection gates). Pure: numbers in, verdicts out.

The voice worker (worker/handlers/voice_judge.py) measures every spoken line with cheap models on the same GPU:
Whisper (clarity: WER), UTMOS22 (naturalness: a MOS predictor), F0 and energy statistics (expressiveness), ECAPA
(consistency: distance to the creator's reference clip), and words per second over the trimmed speech (pacing).
Casting adds perceived gender and age (wav2vec2 classifiers) against the bible. This module turns those numbers into

    line_verdict(scores, cfg, band) -> {passed, reasons, composite}   one line: kept, or regenerated (voice.line_check)
    cast_summary(lines, cfg, band, fit) -> dict                       one candidate voice over the casting script
    rank_cast(candidates, cfg, others, min_distance) -> list           the candidates that qualify, best first
    bakeoff_winner(engines, cfg) -> dict                               the engine the bake-off picks

Every threshold and weight comes from policy `voice.judge` (passed in as `cfg`), so the worker and the control plane
judge alike. Distance from other creators is a floor only (voice.min_distance), never something to maximise: the
old pick rewarded oddness. A composite is a weighted mean of sub-scores in [0, 1]; a hard failure (gender mismatch,
unintelligible, robotic) is a reason, never just a low number.
"""

from __future__ import annotations

import math
import re
from statistics import mean, pstdev


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _ramp(x: float | None, lo: float, hi: float) -> float:
    """0 at lo, 1 at hi (either direction), clamped; None scores 0."""
    if x is None or hi == lo:
        return 0.0
    return _clamp((float(x) - lo) / (hi - lo))


def pace_band(policy, archetypes: list[str] | None) -> tuple[float, float]:
    """The creator's words-per-second band: her first archetype's in voice.judge.pace_wps, else its default
    (calibration 2026-11-24: real creators 3.7-4.3 w/s)."""
    bands = (policy.get("voice", "judge", "pace_wps", default={}) or {})
    for name in archetypes or []:
        if name in bands:
            lo, hi = bands[name]
            return float(lo), float(hi)
    lo, hi = bands.get("default", [3.4, 4.6])
    return float(lo), float(hi)


def sub_scores(s: dict, cfg: dict, band: tuple[float, float]) -> dict:
    """Each measured number as a [0, 1] sub-score: naturalness (MOS), expressiveness (pitch movement and loudness
    movement), clarity (WER), consistency (ECAPA distance to the reference), pacing (inside the band = 1, falling to 0
    over `pace_tolerance_wps` outside it)."""
    n = cfg.get("normalise") or {}
    mos_lo, mos_hi = n.get("mos", [2.5, 4.3])
    f0_lo, f0_hi = n.get("f0_std_st", [1.0, 4.0])
    en_lo, en_hi = n.get("energy_std_db", [3.0, 9.0])
    d_good, d_bad = n.get("ref_distance", [0.15, 0.45])
    out = {"naturalness": _ramp(s.get("mos"), mos_lo, mos_hi),
           "expressiveness": 0.5 * (_ramp(s.get("f0_std_st"), f0_lo, f0_hi)
                                    + _ramp(s.get("energy_std_db"), en_lo, en_hi)),
           "clarity": _clamp(1.0 - float(s["wer"]) / float(cfg.get("max_wer", 0.15))) if s.get("wer") is not None
           else 0.0,
           "consistency": 1.0 if s.get("ref_distance") is None and not cfg.get("require_reference")
           else _ramp(s.get("ref_distance"), d_bad, d_good)}
    wps, tol = s.get("wps"), float(cfg.get("pace_tolerance_wps", 0.8))
    if wps is None:
        out["pacing"] = 0.0
    else:
        lo, hi = band
        off = max(lo - float(wps), float(wps) - hi, 0.0)
        out["pacing"] = _clamp(1.0 - off / tol) if tol > 0 else float(off == 0)
    return {k: round(v, 4) for k, v in out.items()}


def composite(subs: dict, weights: dict) -> float:
    """The weighted mean of the sub-scores present in `weights`."""
    total = sum(float(w) for k, w in weights.items() if k in subs)
    if total <= 0:
        return 0.0
    return round(sum(float(w) * float(subs[k]) for k, w in weights.items() if k in subs) / total, 4)


def line_verdict(s: dict, cfg: dict, band: tuple[float, float]) -> dict:
    """One spoken line judged (policy voice.judge): unclear (WER), robotic (MOS under its floor), flat (pitch movement
    under its floor), off pace (outside the band by more than the tolerance), not her voice (ECAPA distance to the
    reference over its ceiling). -> {passed, reasons, composite, subs}."""
    reasons = []
    if s.get("wer") is None or float(s["wer"]) > float(cfg.get("max_wer", 0.15)):
        reasons.append(f"unclear: WER {s.get('wer')} over {cfg.get('max_wer', 0.15)}")
    if s.get("mos") is not None and float(s["mos"]) < float(cfg.get("min_mos", 3.0)):
        reasons.append(f"robotic: MOS {s['mos']} under {cfg.get('min_mos', 3.0)}")
    words = int(s.get("words") or 0)
    if s.get("f0_std_st") is not None and words >= int(cfg.get("flat_min_words", 4)) \
            and float(s["f0_std_st"]) < float(cfg.get("min_f0_std_st", 1.2)):
        reasons.append(f"flat: pitch moves {s['f0_std_st']} semitones, under {cfg.get('min_f0_std_st', 1.2)}")
    lo, hi = band
    tol = float(cfg.get("pace_tolerance_wps", 0.8))
    if s.get("wps") is not None and words >= int(cfg.get("pace_min_words", 5)) \
            and not (lo - tol <= float(s["wps"]) <= hi + tol):
        reasons.append(f"pacing: {s['wps']} words/s outside {lo}-{hi} (±{tol})")
    if s.get("ref_distance") is not None and float(s["ref_distance"]) > float(cfg.get("max_ref_distance", 0.4)):
        reasons.append(f"inconsistent: {s['ref_distance']} from her reference, over {cfg.get('max_ref_distance', 0.4)}")
    subs = sub_scores(s, cfg, band)
    return {"passed": not reasons, "reasons": reasons, "subs": subs,
            "composite": composite(subs, cfg.get("weights") or {})}


def better(a: dict, b: dict | None) -> bool:
    """Is take `a` (a line_verdict merged into its scores) better than `b`: a pass beats a fail, then the composite,
    then the lower WER."""
    if b is None:
        return True
    ka = (a.get("passed", False), a.get("composite", 0.0), -float(a.get("wer") or 0.0))
    kb = (b.get("passed", False), b.get("composite", 0.0), -float(b.get("wer") or 0.0))
    return ka > kb


# --- fit: perceived gender and age against the bible ---------------------------------------------------------------
def gender_of(text: str | None) -> str | None:
    """'female' / 'male' from a bible's gender presentation ('woman', 'feminine', 'f'...), else None."""
    t = (text or "").strip().lower()
    if not t:
        return None
    if re.match(r"^(f|fem|female|feminine|woman|women|girl|she)\b", t):
        return "female"
    if re.match(r"^(m|male|masculine|man|men|boy|he)\b", t):
        return "male"
    return None


def age_from_label(label: str | None) -> float | None:
    """An age-band label as its midpoint in years: 'teens' 16, '20s' 25, '30-39' 34.5, '70+' 75, a bare number
    as itself; None when it names no age."""
    t = (label or "").strip().lower()
    if not t:
        return None
    if "teen" in t:
        return 16.0
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)", t)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2
    m = re.search(r"(\d+)\s*s\b", t)
    if m:
        return float(m.group(1)) + 5
    m = re.search(r"(\d+)\s*\+", t)
    if m:
        return float(m.group(1)) + 5
    m = re.search(r"\d+(?:\.\d+)?", t)
    return float(m.group(0)) if m else None


def fit_scores(measured: dict, fit: dict, cfg: dict) -> dict:
    """{gender_ok, gender_p, age_gap, fit} for a candidate: `measured` {gender_p: {female, male}, age} from the
    classifiers, `fit` {gender, age} from the bible. Gender is a hard gate (probability of her gender at least
    `min_gender_p`); age is soft (1 inside `age_tolerance_years`, falling to 0 over as many again)."""
    fcfg = cfg.get("fit") or {}
    want = fit.get("gender")
    probs = measured.get("gender_p") or {}
    p = float(probs.get(want, 0.0)) if want else None
    ok = True if want is None else p is not None and p >= float(fcfg.get("min_gender_p", 0.7))
    age, target, tol = measured.get("age"), fit.get("age"), float(fcfg.get("age_tolerance_years", 8))
    gap = None if age is None or target is None else abs(float(age) - float(target))
    age_score = 0.5 if gap is None else _clamp(1.0 - max(0.0, gap - tol) / tol) if tol > 0 else float(gap == 0)
    gender_score = 1.0 if p is None else _clamp(p)
    w_age = float(fcfg.get("age_weight", 0.3))
    return {"gender_ok": bool(ok), "gender_p": None if p is None else round(p, 4),
            "age_gap": None if gap is None else round(gap, 1),
            "fit": round((1 - w_age) * gender_score + w_age * age_score, 4)}


# --- casting ------------------------------------------------------------------------------------------------------
def cast_summary(lines: list[dict], cfg: dict, band: tuple[float, float], fit: dict, measured: dict) -> dict:
    """One candidate voice over the casting script: each line's scores (`lines`, as the worker measured them), the
    fit of the voice (`measured`: gender and age of its reference clip) against the bible's `fit`. Expressiveness
    across lines: how far the lines' median pitch and speaking rate move between deliveries (neutral, warm,
    excited, quiet must not sound alike). -> {qualifies, reasons, composite, pass_share, means..., fit...}."""
    ccfg = cfg.get("casting") or {}
    verdicts = [line_verdict(s, cfg, band) for s in lines]
    reasons = []
    f = fit_scores(measured, fit, cfg)
    if not f["gender_ok"]:
        reasons.append(f"sounds {max((measured.get('gender_p') or {'?': 0}).items(), key=lambda kv: kv[1])[0]}, "
                       f"not {fit.get('gender')} (p {f['gender_p']} under "
                       f"{(cfg.get('fit') or {}).get('min_gender_p', 0.7)})")
    share = sum(v["passed"] for v in verdicts) / len(verdicts) if verdicts else 0.0
    if share < float(ccfg.get("min_pass_share", 0.7)):
        reasons.append(f"only {round(share, 2)} of the casting lines pass, under {ccfg.get('min_pass_share', 0.7)}")
    medians = [float(s["f0_median_st"]) for s in lines if s.get("f0_median_st") is not None]
    rates = [float(s["wps"]) for s in lines if s.get("wps") is not None]
    spread = round(pstdev(medians), 4) if len(medians) > 1 else 0.0
    rate_cv = round(pstdev(rates) / mean(rates), 4) if len(rates) > 1 and mean(rates) > 0 else 0.0
    if spread < float(ccfg.get("min_delivery_spread_st", 0.8)):
        reasons.append(f"every delivery sounds alike: median pitch moves {spread} semitones across lines, under "
                       f"{ccfg.get('min_delivery_spread_st', 0.8)}")
    mos = [float(s["mos"]) for s in lines if s.get("mos") is not None]
    if mos and mean(mos) < float(ccfg.get("min_mean_mos", 3.3)):
        reasons.append(f"mean MOS {round(mean(mos), 3)} under {ccfg.get('min_mean_mos', 3.3)}")
    across = {"delivery_spread": _ramp(spread, *(ccfg.get("spread_norm_st") or [0.5, 2.5])),
              "rate_variation": _ramp(rate_cv, *(ccfg.get("rate_cv_norm") or [0.03, 0.15]))}
    line_mean = mean(v["composite"] for v in verdicts) if verdicts else 0.0
    w = ccfg.get("weights") or {"lines": 0.6, "fit": 0.25, "delivery_spread": 0.1, "rate_variation": 0.05}
    score = composite({"lines": line_mean, "fit": f["fit"], **across}, w)
    return {"qualifies": not reasons, "reasons": reasons, "composite": score, "pass_share": round(share, 4),
            "line_composite": round(line_mean, 4), "delivery_spread_st": spread, "rate_cv": rate_cv,
            "mean_mos": round(mean(mos), 4) if mos else None,
            "mean_wer": round(mean(float(s["wer"]) for s in lines if s.get("wer") is not None), 4)
            if any(s.get("wer") is not None for s in lines) else None,
            **f, "lines": [{"passed": v["passed"], "reasons": v["reasons"], "composite": v["composite"]}
                           for v in verdicts]}


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return 1.0 if not na or not nb else round(1.0 - dot / (na * nb), 4)


def rank_cast(candidates: list[dict], others: list[list[float]], min_distance: float) -> list[dict]:
    """The candidates that qualify (their summary's `qualifies`, and at least `min_distance` from every other
    creator's voice: a floor, never maximised), best composite first, then the lower index. Adds `distance`."""
    out = []
    for c in candidates:
        d = min((cosine_distance(c.get("embedding") or [], o) for o in others if o), default=2.0)
        c = {**c, "distance": d}
        if (c.get("summary") or {}).get("qualifies") and d >= min_distance:
            out.append(c)
    return sorted(out, key=lambda c: (-float(c["summary"]["composite"]), int(c.get("index") or 0)))


# --- the bake-off: which engine becomes the default --------------------------------------------------------------
def bakeoff_winner(engines: dict[str, dict], cfg: dict) -> dict:
    """`engines` {name: {composite (mean over its accepted lines), accepted_minutes, usd, gender_ok, pass_share,
    licence_ok}}. An engine qualifies with its licence cleared, no gender miss and a pass share of at least
    `min_pass_share`. Value = composite / (usd per accepted minute) ** cost_exponent: the rule is quality per
    accepted-minute cost, and a small exponent keeps cost from outweighing a material quality gain (owner: a better
    model that costs modestly more wins). The incumbent keeps the job unless the winner's composite beats it by at
    least `min_gain`. -> {winner, ranked [{engine, value, ...}], reasons}."""
    exp, floor = float(cfg.get("cost_exponent", 0.25)), float(cfg.get("min_pass_share", 0.7))
    min_cost = float(cfg.get("min_usd_per_minute", 0.001))
    ranked, refused = [], {}
    for name, e in engines.items():
        why = []
        if not e.get("licence_ok", False):
            why.append("licence not cleared")
        if not e.get("gender_ok", True):
            why.append("missed the bible's gender")
        if float(e.get("pass_share") or 0) < floor:
            why.append(f"pass share {e.get('pass_share')} under {floor}")
        minutes = float(e.get("accepted_minutes") or 0)
        if minutes <= 0:
            why.append("no accepted audio")
        if why:
            refused[name] = why
            continue
        per_min = max(min_cost, float(e.get("usd") or 0) / minutes)
        ranked.append({"engine": name, "composite": float(e["composite"]), "usd_per_minute": round(per_min, 4),
                       "value": round(float(e["composite"]) / per_min ** exp, 4)})
    ranked.sort(key=lambda r: -r["value"])
    if not ranked:
        return {"winner": None, "ranked": [], "refused": refused}
    best, incumbent = ranked[0], cfg.get("incumbent")
    inc = next((r for r in ranked if r["engine"] == incumbent), None)
    if inc and best["engine"] != incumbent and best["composite"] - inc["composite"] < float(cfg.get("min_gain", 0.03)):
        best = inc
    return {"winner": best["engine"], "ranked": ranked, "refused": refused}
