"""Automatic voice judgement (manyworlds/voice_eval.py; owner 2026-11-25: no human selection gates): flat, robotic,
unclear, off-pace and inconsistent lines are named; a candidate that doesn't sound like the bible's gender is out
whatever it scores; distance from other creators is a floor, never maximised; the bake-off picks quality per
accepted-minute cost."""

from __future__ import annotations

import pytest

from manyworlds import voice_eval as ve

from voice_helpers import cast_meta, line, lines


@pytest.fixture()
def judge(policy):
    return policy.get("voice", "judge")


BAND = (3.6, 4.4)


def test_a_good_line_passes_with_a_high_composite(judge):
    v = ve.line_verdict(line(), judge, BAND)
    assert v["passed"] and v["reasons"] == [] and v["composite"] > 0.7


@pytest.mark.parametrize("over, word", [
    ({"wer": 0.4}, "unclear"),
    ({"mos": 2.1}, "robotic"),
    ({"f0_std_st": 0.4}, "flat"),
    ({"wps": 6.2}, "pacing"),
    ({"wps": 2.0}, "pacing"),
    ({"ref_distance": 0.7}, "inconsistent"),
])
def test_each_failure_is_named(judge, over, word):
    v = ve.line_verdict(line(**over), judge, BAND)
    assert not v["passed"] and any(r.startswith(word) for r in v["reasons"])
    assert v["composite"] < ve.line_verdict(line(), judge, BAND)["composite"]


def test_short_lines_are_not_judged_flat_or_off_pace(judge):
    """"Great. Love that for me." has too few words for a pitch range or a rate to mean anything."""
    v = ve.line_verdict(line(words=3, f0_std_st=0.3, wps=6.0), judge, BAND)
    assert v["passed"]


def test_a_repair_that_passes_beats_a_failing_take_with_a_higher_composite():
    assert ve.better({"passed": True, "composite": 0.5}, {"passed": False, "composite": 0.9})
    assert not ve.better({"passed": True, "composite": 0.5, "wer": 0.1}, {"passed": True, "composite": 0.6})


def test_bible_gender_and_age_labels():
    assert [ve.gender_of(x) for x in ("woman", "Feminine", "male", "man", "non-binary", "")] == \
        ["female", "female", "male", "male", None, None]
    assert [ve.age_from_label(x) for x in ("teens", "20s", "30-39", "70+", "41", "unknown")] == \
        [16.0, 25.0, 34.5, 75.0, 41.0, None]


def test_a_voice_of_the_wrong_gender_never_qualifies_however_well_it_speaks(judge):
    """Owner 2026-11-25 on the VoxCPM2 voice: "doesn't even sound female"."""
    fit = {"gender": "female", "age": 27.0}
    male = cast_meta(0, gender="male", p=0.9)
    s = ve.cast_summary(male["lines"], judge, BAND, fit, male["who"])
    assert not s["qualifies"] and not s["gender_ok"] and any("sounds male" in r for r in s["reasons"])
    good = cast_meta(1)
    assert ve.cast_summary(good["lines"], judge, BAND, fit, good["who"])["qualifies"]


def test_a_monotone_candidate_is_out_even_when_each_line_passes(judge):
    """Neutral, warm, excited and quiet lines that all sound alike are robotic casting."""
    same = [line(name=f"l{i}", f0_median_st=24.0) for i in range(5)]
    s = ve.cast_summary(same, judge, BAND, {"gender": "female"}, {"gender_p": {"female": 0.9}})
    assert all(x["passed"] for x in s["lines"]) and not s["qualifies"]
    assert any("sounds alike" in r for r in s["reasons"])


def test_age_is_soft(judge):
    fit = {"gender": "female", "age": 27.0}
    near, far = cast_meta(0, age=29.0), cast_meta(1, age=60.0)
    a = ve.cast_summary(near["lines"], judge, BAND, fit, near["who"])
    b = ve.cast_summary(far["lines"], judge, BAND, fit, far["who"])
    assert a["qualifies"] and b["qualifies"] and a["composite"] > b["composite"]


def test_distance_from_other_creators_is_a_floor_never_maximised(judge):
    """The old pick ranked by oddness; now a closer voice that still clears the floor wins on quality."""
    fit = {"gender": "female", "age": 27.0}
    other = [0.0, 1.0, 0.0]
    close_good = {"index": 0, "embedding": [0.6, 0.8, 0.0]}      # 0.2 from the other voice
    far_worse = {"index": 1, "embedding": [1.0, 0.0, 0.0]}       # 1.0 from it
    too_close = {"index": 2, "embedding": [0.05, 1.0, 0.0]}
    for c, mos in ((close_good, 4.2), (far_worse, 3.5), (too_close, 4.4)):
        m = cast_meta(c["index"], mos=mos)
        c["summary"] = ve.cast_summary(m["lines"], judge, BAND, fit, m["who"])
    ranked = ve.rank_cast([far_worse, close_good, too_close], [other], min_distance=0.15)
    assert [c["index"] for c in ranked] == [0, 1]


def test_pace_bands_come_from_her_archetype(policy):
    bands = policy.get("voice", "judge", "pace_wps")
    assert ve.pace_band(policy, ["lifestyle_slow"]) == tuple(bands["lifestyle_slow"])
    assert ve.pace_band(policy, ["no_such", "storyteller"]) == tuple(bands["storyteller"])
    assert ve.pace_band(policy, None) == tuple(bands["default"])


def test_the_bakeoff_picks_quality_per_accepted_minute_cost(policy):
    cfg = {**policy.get("voice", "bakeoff"), "incumbent": policy.get("voice", "engine_gate", "until_bakeoff")}
    assert cfg["incumbent"] == "qwen3"
    e = lambda c, usd, **kw: {"composite": c, "pass_share": 0.9, "accepted_minutes": 2.0, "usd": usd,  # noqa: E731
                              "gender_ok": True, "licence_ok": True, **kw}
    # better and modestly dearer wins: the cost exponent keeps price from outweighing a material gain
    out = ve.bakeoff_winner({"qwen3": e(0.60, 0.10), "indextts": e(0.75, 0.16)}, cfg)
    assert out["winner"] == "indextts" and [r["engine"] for r in out["ranked"]][0] == "indextts"
    # a gain under min_gain keeps the incumbent
    assert ve.bakeoff_winner({"qwen3": e(0.70, 0.10), "indextts": e(0.71, 0.10)}, cfg)["winner"] == "qwen3"
    # an uncleared licence, a gender miss or too few passing lines refuse an engine whatever it scores
    out = ve.bakeoff_winner({"qwen3": e(0.6, 0.1), "a": e(0.9, 0.1, licence_ok=False),
                             "b": e(0.9, 0.1, gender_ok=False), "c": e(0.9, 0.1, pass_share=0.2)}, cfg)
    assert out["winner"] == "qwen3" and set(out["refused"]) == {"a", "b", "c"}
    assert ve.bakeoff_winner({"a": e(0.9, 0.1, accepted_minutes=0)}, cfg)["winner"] is None


def test_every_casting_line_and_weight_is_in_policy(policy):
    judge = policy.get("voice", "judge")
    assert abs(sum(judge["weights"].values()) - 1) < 1e-9
    assert abs(sum(judge["casting"]["weights"].values()) - 1) < 1e-9
    script = policy.get("voice", "casting", "script")
    assert {s["name"] for s in script} >= {"neutral", "warm", "excited", "quiet"}
    assert any("Two am. I take it back." in x["text"] for x in policy.get("voice", "bakeoff", "lines"))
    assert len(lines()) == 5
