"""Cut-based talking edits (manyworlds.cuts): the grammar a talking post follows, which speaking shots are lip-synced,
the finish's punch-ins (cut only between words, the voice untouched), and scripts written to be said."""

from __future__ import annotations

import subprocess

import pytest

from manyworlds import cuts, production
from manyworlds import policy as policy_mod
from manyworlds.brain import planner


def _shot(*lines, **kw):
    return {"performance": {"lines": [{"text": t, "delivery": ""} for t in lines]}, **kw}


FACE = _shot("Okay, so the samosas are back.", framing={"shot_size": "close-up", "device": "selfie"})
HANDS = _shot("I bought four. For me.", framing={"shot_size": "close-up", "device": "pov"})
BEHIND = _shot("Walking home like nothing happened.", framing={"shot_size": "medium shot",
                                                                "angle": "over the shoulder"})
END = _shot("No regrets.", framing={"shot_size": "close-up", "device": "selfie"})


def _words(*spans):
    return [{"text": f"w{i}", "start": a, "end": b} for i, (a, b) in enumerate(spans)]


def _with_grammar(policy, **grammars):
    body = dict(policy.body)
    gen = dict(body["generation"])
    te = dict(gen["talk_edit"])
    te["grammars"] = {**te["grammars"], **grammars}
    gen["talk_edit"] = te
    body["generation"] = gen
    return policy_mod.from_dict(body)


# --- which grammar ---------------------------------------------------------------------------------------------------

def test_the_idea_then_the_archetype_then_the_default_picks_the_grammar(policy):
    names = cuts.grammar_names(policy)
    assert {"story_time", "commercial", "long_take"} <= set(names)
    assert cuts.choose(policy, {}) == policy.get("generation", "talk_edit", "default") == "story_time"
    assert cuts.choose(policy, {"edit_grammar": "long_take"}) == "long_take"
    assert cuts.choose(policy, {"edit_grammar": "nonsense"}) == "story_time"          # unknown: ignored
    assert cuts.choose(policy, {}, {"archetypes": ["food_home"]}) == "commercial"      # by her archetype
    assert cuts.choose(policy, {"edit_grammar": "story_time"}, {"archetypes": ["food_home"]}) == "story_time"


def test_the_long_single_take_is_no_longer_the_default(policy):
    assert policy.get("generation", "talk_edit", "default") != "long_take"
    assert not cuts.grammar(policy, "long_take").get("punch_in")


# --- lip sync only where her face speaks to camera ---------------------------------------------------------------

def test_on_camera_grammars_narrate_the_shots_that_dont_show_her_face(policy):
    assert not cuts.narrates(FACE, policy, "story_time")
    assert cuts.narrates(HANDS, policy, "story_time")          # her point of view: hands only
    assert cuts.narrates(BEHIND, policy, "commercial")         # over the shoulder
    assert not cuts.narrates(HANDS, policy, "long_take")       # lip_sync all: only the director's cutaways


def test_only_face_to_camera_takes_pay_for_lip_sync(policy):
    takes = production.shot_takes(policy, [FACE, HANDS, BEHIND, END],
                                  narrate=lambda s: cuts.narrates(s, policy, "commercial"))
    assert [(mode, i) for _, mode, i in takes] == [("talk", 0), ("narrated", 1), ("narrated", 2), ("talk", 3)]
    plain = production.shot_takes(policy, [FACE, HANDS, BEHIND, END])       # without a grammar: as before
    assert [mode for _, mode, _ in plain] == ["talk"] * 4


def test_when_only_narrated_shots_speak_the_lines_go_to_the_first_face_shot(policy):
    silent_face = _shot(framing={"shot_size": "close-up", "device": "selfie"})
    takes = production.shot_takes(policy, [HANDS, silent_face, BEHIND],
                                  narrate=lambda s: cuts.narrates(s, policy, "story_time"))
    assert takes == [("I bought four. For me. Walking home like nothing happened.", "talk", 1)]


# --- punch-ins ---------------------------------------------------------------------------------------------------

CFG = {"every_s": [2.0, 3.0], "scales": [1.0, 1.2], "min_shot_s": 1.0}
SPEECH = _words(*[(round(0.1 + 0.5 * i, 2), round(0.5 + 0.5 * i, 2)) for i in range(20)])   # 20 words over 10 s


def test_punch_ins_cut_only_between_words_and_cover_the_whole_video(policy):
    plan = cuts.punch_plan(SPEECH, [(0.0, 10.4)], 12.0, CFG, seed=7, fps=30)
    assert plan and plan[0]["start"] == 0 and plan[-1]["end"] == 12.0
    assert all(a["end"] == b["start"] for a, b in zip(plan, plan[1:]))                  # contiguous
    for s in plan[1:]:
        assert not any(w["start"] < s["start"] < w["end"] for w in SPEECH), s           # never mid-word
    inside = [s for s in plan if s["end"] <= 10.4]
    assert all(s["end"] - s["start"] >= 1.0 - 1 / 30 for s in inside)                   # min shot (frame-snapped)
    assert all(a["scale"] != b["scale"] for a, b in zip(plan, plan[1:]))                # alternating
    assert plan[-1]["scale"] == 1.0                                                     # outside the take: as made
    assert {s["scale"] for s in plan} == {1.0, 1.2}


def test_punch_ins_stay_inside_the_lip_synced_takes(policy):
    plan = cuts.punch_plan(SPEECH, [(0.0, 4.0), (7.0, 10.4)], 10.4, CFG, seed=1)
    for s in plan:
        if s["scale"] > 1.0:
            assert (0.0 <= s["start"] and s["end"] <= 4.0) or (7.0 <= s["start"] and s["end"] <= 10.4), s


def test_no_punch_ins_without_words_a_window_or_a_closer_scale(policy):
    assert cuts.punch_plan([], [(0, 10)], 10, CFG) == []
    assert cuts.punch_plan(SPEECH, [], 10, CFG) == []
    assert cuts.punch_plan(SPEECH, [(0, 10)], 10, {**CFG, "scales": [1.0]}) == []
    assert cuts.punch_plan(SPEECH, [(0, 1.5)], 10, CFG) == []                           # too short to cut


def test_the_plan_is_the_same_for_the_same_post(policy):
    assert cuts.punch_plan(SPEECH, [(0, 10.4)], 10.4, CFG, seed=3) == cuts.punch_plan(SPEECH, [(0, 10.4)], 10.4,
                                                                                         CFG, seed=3)


def test_the_crop_stays_in_frame_and_the_face_box_covers_every_framing(policy):
    w, h, x, y = cuts.crop_box((1080, 1920), 1.2, (0.95, 0.05))
    assert (w % 2, h % 2) == (0, 0) and 0 <= x <= 1080 - w and 0 <= y <= 1920 - h
    face = (0.4, 0.3, 0.6, 0.45)
    plan = [{"start": 0, "end": 2, "scale": 1.0}, {"start": 2, "end": 4, "scale": 1.2}]
    center = cuts.center_for(face, {})
    assert center == pytest.approx((0.5, 0.375))
    box = cuts.punched_face_box(face, plan, (1080, 1920), center)
    assert box[0] <= face[0] and box[1] <= face[1] and box[2] >= face[2] and box[3] >= face[3]
    assert cuts.center_for(None, {"center": [0.5, 0.4]}) == (0.5, 0.4)


def test_talk_windows_skip_the_cutaways(policy):
    pieces = [{"meta": {"seconds": 3.0}}, {"meta": {"seconds": 2.0}}, {"meta": {"seconds": 4.0}}]
    body = {"talk_modes": ["talk", "narrated", "talk"]}
    assert cuts.talk_windows(body, {}, pieces, 9.0) == [(0.0, 3.0), (5.0, 9.0)]
    assert cuts.talk_windows(body, {"segment_seconds": [2.5, 2.0, 4.0]}, pieces, 8.5) == [(0.0, 2.5), (4.5, 8.5)]
    assert cuts.talk_windows({}, {}, [{"meta": {}}], 6.0) == [(0.0, 6.0)]
    assert cuts.talk_windows({}, {}, [], None) == []


def _clip(path, seconds=4.0):
    from manyworlds import media
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc=size=64x112:rate=30:duration={seconds:g}", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={seconds:g}:sample_rate=48000",
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
                   check=True, capture_output=True)
    return path


def test_punch_ins_render_on_the_cpu_and_keep_the_voice(policy, tmp_path):
    from manyworlds import media
    src = _clip(tmp_path / "take.mp4")
    plan = [{"start": 0.0, "end": 1.5, "scale": 1.0}, {"start": 1.5, "end": 3.0, "scale": 1.2},
            {"start": 3.0, "end": 4.0, "scale": 1.0}]
    out = cuts.render_punch_ins(src, tmp_path / "punched.mp4", plan, (64, 112), (0.5, 0.4))
    a, b = media.probe(src), media.probe(out)
    assert (b["width"], b["height"]) == (64, 112)
    assert b["seconds"] == pytest.approx(a["seconds"], abs=0.1)
    assert "Audio" in subprocess.run([media.ffmpeg(), "-hide_banner", "-i", str(out)], capture_output=True,
                                     text=True).stderr


def test_the_finish_punches_in_a_talking_take_and_keeps_the_face_box(policy, tmp_path):
    from manyworlds.production import finish
    src = _clip(tmp_path / "take.mp4", seconds=6.0)
    words = _words(*[(round(0.1 + 0.5 * i, 2), round(0.5 + 0.5 * i, 2)) for i in range(11)])
    plan = {"post": {"id": "00000000-0000-0000-0000-000000000001", "format": "talking_video"}, "body": {},
            "pieces": [{"meta": {}}], "video_meta": {}, "keyframes": [],
            "punch_in": CFG, "cut_words": words}
    out, stretches, error, face = finish._punch_in(plan, policy, None, src, tmp_path)
    assert error is None and stretches and out != src and out.exists()
    assert "face" in face
    # Nothing to punch: the video as it was, no record.
    assert finish._punch_in({**plan, "punch_in": None}, policy, None, src, tmp_path)[:3] == (src, None, None)


def test_a_failed_punch_in_leaves_the_video_as_it_was(policy, tmp_path):
    from manyworlds.production import finish
    plan = {"post": {"id": None, "format": "talking_video"}, "body": {}, "pieces": [], "video_meta": {},
            "keyframes": [], "punch_in": CFG, "cut_words": SPEECH}
    missing = tmp_path / "missing.mp4"
    out, stretches, error, _ = finish._punch_in(plan, policy, None, missing, tmp_path)
    assert out == missing and stretches is None and error


# --- the director's brief ----------------------------------------------------------------------------------------

def test_the_grammar_reaches_the_directors_brief(policy):
    text = cuts.director_rules(policy, "commercial")
    assert text.startswith("Edit grammar for this talking post: commercial.")
    assert "angles" in text
    assert cuts.director_rules(policy, "nonsense") == ""


# --- scripts written to be said ----------------------------------------------------------------------------------

def test_spoken_rules_reach_the_writers_and_long_sentences_are_caught(policy):
    rules = planner.voice_rules(policy)
    assert "written to be said" in rules and "Contractions" in rules
    limit = policy.get("captions", "spoken", "max_words_per_sentence")
    assert planner.spoken_issues(policy, "Two am. I take it back. I was not ready for anything.") == []
    long = " ".join(["word"] * (limit + 1)) + "."
    assert planner.spoken_issues(policy, long) == [f"a {limit + 1}-word sentence (spoken lines stay under {limit})"]
    assert planner.spoken_issues(policy, "") == []
