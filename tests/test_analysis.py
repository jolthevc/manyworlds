"""The analyst's statistics: effects of creative characteristics on normalised reach, with honest
uncertainty, so the Brain neither misses real patterns nor invents them from noise."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from manyworlds.brain import analysis, simulation

FORMATS = ("photo", "carousel", "story", "story_clip")
DIMS = ("personality", "aspiration", "novelty")


def _rows(policy, scenario, n, *, seed=0, formats=FORMATS, format_by_index=None):
    sc = simulation.load(scenario)
    rng = random.Random(f"rows:{scenario}:{seed}")
    rows, t0 = [], datetime(2026, 11, 1, tzinfo=timezone.utc)
    for i in range(n):
        fmt = format_by_index(i) if format_by_index else rng.choice(formats)
        scores = {d: rng.randint(2, 9) for d in DIMS}
        out = simulation.simulate(sc, {"format": fmt, "scores": scores, "attributes": {}}, index=i,
                                  rng_key=f"{seed}:{i}", policy=policy)
        rows.append({"post_id": f"p{i}", "published_at": t0 + timedelta(hours=8 * i), "format": fmt,
                     "scores": scores, "attributes": {}, "reach": out[-1]["metrics"]["reach"]})
    return rows


def _graded(result, grade="tentative"):
    return {(e["kind"], e["key"]): e for e in result["effects"] if e["grade"] == grade}


def test_few_posts_are_insufficient(policy):
    r = analysis.effects(analysis.normalise(_rows(policy, "personality_wins", 6), policy), policy, seed=1)
    assert r["insufficient"] and r["effects"] == []


def test_a_planted_effect_is_found_with_the_right_sign(policy):
    r = analysis.effects(analysis.normalise(_rows(policy, "personality_wins", 90), policy), policy, seed=1)
    t = _graded(r)
    assert t[("dimension", "personality")]["effect"] > 0
    assert t[("dimension", "personality")]["low"] > 0
    assert t[("format", "story_clip")]["effect"] < 0


def test_null_scenario_false_positive_rate(policy):
    runs = [analysis.effects(analysis.normalise(_rows(policy, "null", 60, seed=s), policy), policy, seed=s)
            for s in range(12)]
    with_any = sum(1 for r in runs if _graded(r))
    assert with_any / len(runs) <= float(policy.get("brain", "analysis", "alpha")) + 0.1


def test_trend_does_not_create_format_effects(policy):
    # A declining account that shifted from photos to story clips: raw reach blames story clips.
    rows = _rows(policy, "declining", 80, format_by_index=lambda i: "photo" if i < 40 else "story_clip")
    r = analysis.effects(analysis.normalise(rows, policy), policy, seed=1)
    assert ("format", "story_clip") not in _graded(r) and ("format", "photo") not in _graded(r)


def test_one_breakout_is_reported_not_graded(policy):
    rows = _rows(policy, "null", 40)
    rows[25]["reach"] *= 12                 # one post explodes
    r = analysis.effects(analysis.normalise(rows, policy), policy, seed=1)
    top = max(r["breakouts"], key=lambda b: b["lift"])
    assert top["post_id"] == "p25" and not _graded(r)       # (noise can push another early post past 3x)


def test_conflicting_signals_show_per_format(policy):
    rows = _rows(policy, "conflicting", 160, formats=("photo", "story_clip"))
    r = analysis.effects(analysis.normalise(rows, policy), policy, seed=1)
    by = {(e["kind"], e["key"]): e for e in r["effects"]}
    assert by[("format_dimension", "photo:aspiration")]["effect"] > 0
    assert by[("format_dimension", "story_clip:aspiration")]["effect"] < 0
    assert by[("dimension", "aspiration")]["grade"] != "tentative"
