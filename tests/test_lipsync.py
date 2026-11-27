"""The lip-sync score: a mouth that tracks the voice scores high, one that doesn't scores low (worker/lipsync.py)."""

from __future__ import annotations

import numpy as np

from manyworlds.worker import lipsync

FPS, RATE, SECONDS = 25, 16000, 6


def _voice(seed=0):
    """Speech-like audio: syllable bursts of noise with pauses, and its per-frame loudness."""
    rng = np.random.default_rng(seed)
    env = np.repeat((rng.random(FPS * SECONDS) > 0.45).astype(float), RATE // FPS)
    audio = rng.normal(0, 0.3, len(env)) * (0.05 + env)
    return audio, lipsync.loudness(audio, RATE, FPS, FPS * SECONDS)


def test_a_mouth_that_follows_the_voice_scores_high_at_its_lag():
    audio, loud = _voice()
    mouth = np.roll((loud - loud.min()) / np.ptp(loud), 2) * 0.4 + np.random.default_rng(1).normal(0, 0.02, len(loud))
    s = lipsync.sync_score(mouth, loud)
    assert s["corr"] > 0.8 and s["offset_frames"] == 2 and s["face_share"] == 1.0


def test_a_mouth_unrelated_to_the_voice_scores_low():
    _, loud = _voice(0)
    _, other = _voice(7)                      # a different line: same kind of motion, wrong timing
    assert lipsync.sync_score(other, loud)["corr"] < 0.35


def test_a_frozen_mouth_or_no_face_gives_no_score():
    _, loud = _voice()
    assert lipsync.sync_score(np.full(len(loud), 0.1), loud)["corr"] is None          # never moves
    s = lipsync.sync_score(np.full(len(loud), np.nan), loud)
    assert s["corr"] is None and s["face_share"] == 0.0


# The voiced span (perf_compare segment 8, 2026-11-25): a short line in an 81-frame InfiniteTalk window leaves
# frames with no voice that the join cuts away; the gate scores the span the voice covers when the whole clip fails.

def _short_line(seed=0, voiced=52, frames=81):
    rng = np.random.default_rng(seed)
    env = (rng.random(voiced) > 0.45).astype(float)
    audio = np.concatenate([(0.05 + np.repeat(env, RATE // FPS)) * rng.normal(0, 0.3, voiced * (RATE // FPS)),
                            np.zeros((frames - voiced) * (RATE // FPS))])
    return lipsync.loudness(audio, RATE, FPS, frames)


def test_a_synced_mouth_that_stays_parted_after_the_line_scores_well_on_the_voiced_span():
    loud = _short_line()
    mouth = np.concatenate([(loud[:52] - loud[:52].min()) / np.ptp(loud[:52]) * 0.15, np.full(29, 0.14)])
    mouth += np.random.default_rng(3).normal(0, 0.005, 81)
    assert lipsync.sync_score(mouth, loud)["corr"] < 0.3                     # the silent smile sinks the whole clip
    v = lipsync.voiced_score(mouth, loud, 52)
    assert v["corr"] > 0.8 and v["voiced_frames"] == 52 and v["open_std"] > 0.02


def test_an_unsynced_mouth_stays_low_on_the_voiced_span_too():
    loud, other = _short_line(0), _short_line(7)
    mouth = np.concatenate([(other[:52] - other[:52].min()) / np.ptp(other[:52]) * 0.15, np.full(29, 0.14)])
    v = lipsync.voiced_score(mouth, loud, 52)
    assert v["corr"] is None or v["corr"] < 0.3


def test_a_clip_without_frames_past_its_line_has_no_voiced_score():
    loud = _short_line(voiced=81, frames=81)
    assert lipsync.voiced_score(np.linspace(0, 1, 81), loud, 81) is None
    assert lipsync.voiced_score(np.linspace(0, 1, 81), loud, 70, tail_frames=20) is None


# --- 2026-11-26 audit: a slow mouth correlates with a slow voice by chance --------------------------------------------

def _slow(seed, n=FPS * SECONDS):
    """A slowly varying series (phrase-level loudness, a mouth that opens now and then): smooth noise."""
    raw = np.random.default_rng(seed).normal(0, 1, n + 30)
    return np.convolve(raw, np.ones(30) / 30, mode="valid")[:n]


def test_real_sync_beats_its_null_and_a_chance_match_does_not():
    _, loud = _voice()
    mouth = (loud - loud.min()) / np.ptp(loud)                         # follows the voice frame by frame
    s = lipsync.sync_score(mouth, loud)
    real = lipsync.null_margin(mouth, loud, s["corr"], min_lag=13, max_lag=40, percentile=95)
    assert real["margin"] > 0.5
    voice, lips = _slow(1), _slow(1) + _slow(2) * 0.5                  # both slow: they "match" at every lag
    s = lipsync.sync_score(lips, voice)
    chance = lipsync.null_margin(lips, voice, s["corr"], min_lag=13, max_lag=40, percentile=95)
    assert s["corr"] > 0.3 and chance["margin"] < real["margin"] - 0.3   # passes min_corr, but no better than chance


def test_a_clip_too_short_for_shifted_lags_has_no_null():
    _, loud = _voice()
    assert lipsync.null_margin(loud[:30], loud[:30], 0.9, min_lag=13, max_lag=40, percentile=95) is None
    assert lipsync.null_margin(loud, loud, None, min_lag=13, max_lag=40, percentile=95) is None
