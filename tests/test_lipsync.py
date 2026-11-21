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
