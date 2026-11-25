"""Repairs of a talking take that failed lip sync (breadth run 2026-11-23 `default-0229fe`): a V2V take (walk-and-talk)
redraws more on each repair (params.start_step one lower, down to the policy floor) instead of only reseeding, and a
knob whose repair scored the same as the attempt before is not tried again; with every knob spent the post drops
with the reason."""

from __future__ import annotations

import pytest

from manyworlds import production
from manyworlds.production.steps import DropPost, Step, advance_step

from test_production import creator

SCRIPT = "Walking to the station, honestly the best part of my day."
MOTION = {"id": "m1", "uri": "s3://b/motion.mp4"}


def _take(conn):
    cid = creator(conn)
    post = conn.execute("INSERT INTO posts (creator_id, format) VALUES (%s, 'talking_video') RETURNING *",
                        (cid,)).fetchone()
    body = {"scenes": ["on the platform"], "script": SCRIPT, "talk_jobs": []}
    return dict(post), body


def _make(conn, policy, post, body, motion=MOTION):
    job = production._talk_job(conn, policy, post, body, [{"uri": "s3://b/a.png"}], len(body["talk_jobs"]),
                               motion=motion)
    body["talk_jobs"].append(str(job))
    return conn.execute("SELECT payload FROM jobs WHERE id = %s", (job,)).fetchone()["payload"]


def _judged(conn, job_id, corr, passed=False):
    a = conn.execute("INSERT INTO assets (job_id, kind, uri, sha256, bytes) VALUES (%s, 'video', %s, 'x', 1) "
                     "RETURNING id", (job_id, f"s3://b/{job_id}.mp4")).fetchone()["id"]
    conn.execute("INSERT INTO reviews (asset_id, gate, passed, score) VALUES (%s, 'lip_sync', %s, %s)",
                 (a, passed, corr))


def test_a_v2v_take_that_fails_lip_sync_redraws_more_on_each_repair(conn, policy):
    start = policy.get("generation", "talk_motion", "v2v", "start_step")
    cfg = policy.get("generation", "talk_motion", "v2v_repair")
    assert start - cfg["start_step_step"] >= cfg["start_step_floor"]
    post, body = _take(conn)
    first = _make(conn, policy, post, body)
    assert first["params"]["start_step"] == start and "repair" not in first
    seen = [start]
    for corr in (0.2, -0.1, 0.15, -0.2):                                    # each a different (bad) score
        _judged(conn, body["talk_jobs"][-1], corr)
        p = _make(conn, policy, post, body)
        seen.append(p["params"]["start_step"])
        assert p["repair"]["cause"] == "lip_sync"
    floor = cfg["start_step_floor"]
    expected = [max(floor, start - k * cfg["start_step_step"]) for k in range(len(seen))]
    assert seen == expected and seen[-1] == floor
    assert p["repair"]["knob"] == "seed"                                    # at the floor only the seed is left


def test_a_repair_that_scores_the_same_stops_that_knob_then_drops_the_post(conn, policy):
    post, body = _take(conn)
    start = _make(conn, policy, post, body)["params"]["start_step"]
    _judged(conn, body["talk_jobs"][-1], 0.2)
    p = _make(conn, policy, post, body)
    assert p["repair"]["knob"] == "start_step" and p["params"]["start_step"] == start - 1
    _judged(conn, body["talk_jobs"][-1], 0.2)                                # redrawing more changed nothing
    p = _make(conn, policy, post, body)
    assert p["repair"]["knob"] == "seed" and p["params"]["start_step"] == start - 1   # kept, not lowered again
    assert "start_step" in p["repair"]["dead_knobs"] and "0.200" in p["repair"]["dead_knobs"]["start_step"]
    _judged(conn, body["talk_jobs"][-1], 0.2)                                # a new seed changed nothing either
    p = _make(conn, policy, post, body)                                      # so the audio guidance goes up
    guidance = policy.get("review", "video", "talk_repair", "audio_guidance")
    assert p["repair"]["knob"] == "audio_guidance" and all(p["params"][k] == v for k, v in guidance.items())
    _judged(conn, body["talk_jobs"][-1], 0.2)                                # no cutaway for a one-take video
    with pytest.raises(DropPost, match="lip sync failed and no repair changes it"):
        _make(conn, policy, post, body)


def test_the_dropped_reason_reaches_the_post(conn, policy):
    post, body = _take(conn)
    _make(conn, policy, post, body)
    for _ in range(3):
        _judged(conn, body["talk_jobs"][-1], 0.2)
        _make(conn, policy, post, body)
    _judged(conn, body["talk_jobs"][-1], 0.2)
    jobs = list(body["talk_jobs"])
    step = Step(jobs, lambda j: ("rejected", None),
                lambda attempt: production._talk_job(conn, policy, post, {**body, "talk_jobs": jobs},
                                                     [{"uri": "s3://b/a.png"}], attempt, motion=MOTION),
                ("clip_repairs", 9), ("follow_up_failures", 9), "clip")
    assert advance_step(body, step) == ("dropped", None)
    assert "no repair changes it" in body["drop_reason"] and "seed" in body["drop_reason"]
    assert "audio_guidance" in body["drop_reason"]


def test_a_take_rejected_for_something_else_keeps_its_redraw_and_reseeds(conn, policy):
    post, body = _take(conn)
    start = _make(conn, policy, post, body)["params"]["start_step"]
    _judged(conn, body["talk_jobs"][-1], 0.1)
    assert _make(conn, policy, post, body)["params"]["start_step"] == start - 1
    _judged(conn, body["talk_jobs"][-1], 0.6, passed=True)                   # in sync; the judge rejected it
    p = _make(conn, policy, post, body)
    assert p["params"]["start_step"] == start - 1 and p["repair"] == {**p["repair"], "cause": None, "knob": "seed"}


def test_a_photo_take_only_reseeds_and_stops_when_the_seed_changes_nothing(conn, policy):
    post, body = _take(conn)
    assert "start_step" not in _make(conn, policy, post, body, motion=None)["params"]
    _judged(conn, body["talk_jobs"][-1], 0.1)
    p = _make(conn, policy, post, body, motion=None)
    assert p["repair"]["knob"] == "seed" and "start_step" not in p["params"]
    _judged(conn, body["talk_jobs"][-1], 0.4)                                # a different score: keep trying
    assert _make(conn, policy, post, body, motion=None)["repair"]["dead_knobs"] == {}
    _judged(conn, body["talk_jobs"][-1], 0.4)
    p = _make(conn, policy, post, body, motion=None)                         # the seed changed nothing: escalate
    assert p["repair"]["knob"] == "audio_guidance" and "seed" in p["repair"]["dead_knobs"]
    _judged(conn, body["talk_jobs"][-1], 0.45)
    with pytest.raises(DropPost):
        _make(conn, policy, post, body, motion=None)


# Escalation (perf_compare 2026-11-25: segment 8 failed lip sync at -0.46, -0.34, -0.43, each repair a new seed).

LINES = ["Okay, story time.", "He walks in and sits down.", "He points at the scratch.", "Best night of the week."]


def _story(conn, *, cutaway_shot=True):
    """A directed story_time talking video: four on-camera takes, one per shot, plus a faceless shot."""
    cid = creator(conn)
    post = conn.execute("INSERT INTO posts (creator_id, format) VALUES (%s, 'talking_video') RETURNING *",
                        (cid,)).fetchone()
    shots = [{"framing": {"shot_size": "medium shot"}} for _ in LINES]
    if cutaway_shot:
        shots.append({"cutaway": True, "framing": {"shot_size": "close-up"}})
    body = {"scenes": [f"behind the bar, shot {k}" for k in range(len(shots))], "shots": shots,
            "edit_grammar": "story_time", "talk_segments": list(LINES), "talk_shots": list(range(len(LINES))),
            "talk_modes": ["talk"] * len(LINES), "talk_segment_jobs": [[] for _ in LINES]}
    chosen = [{"uri": f"s3://b/shot{k}.png"} for k in range(len(shots))]
    return dict(post), body, chosen


def _seg(conn, policy, post, body, chosen, segment):
    jobs = body["talk_segment_jobs"][segment]
    job = production._talk_job(conn, policy, post, body, chosen, len(jobs), segment=segment)
    jobs.append(str(job))
    return conn.execute("SELECT kind, payload FROM jobs WHERE id = %s", (job,)).fetchone()


def test_a_middle_line_that_keeps_failing_lip_sync_escalates_then_goes_under_a_cutaway(conn, policy):
    post, body, chosen = _story(conn)
    _seg(conn, policy, post, body, chosen, 2)
    for corr in (-0.46, -0.34):                                              # within the tolerance of each other
        _judged(conn, body["talk_segment_jobs"][2][-1], corr)
        j = _seg(conn, policy, post, body, chosen, 2)
    assert j["payload"]["repair"]["knob"] == "audio_guidance" and j["payload"]["params"]["cfg"] > 1
    _judged(conn, body["talk_segment_jobs"][2][-1], -0.43)
    j = _seg(conn, policy, post, body, chosen, 2)
    assert j["kind"] == "image.narrated" and j["payload"]["images"] == ["s3://b/shot4.png"]   # the faceless shot
    assert body["talk_modes"][2] == "narrated" and body["rerouted"]["2"]["image_uri"] == "s3://b/shot4.png"
    assert body["talk_modes"].count("talk") == 3


def test_the_hook_and_the_sign_off_stay_on_camera(conn, policy):
    post, body, chosen = _story(conn)
    last = len(LINES) - 1
    _seg(conn, policy, post, body, chosen, last)
    for corr in (-0.46, -0.34):
        _judged(conn, body["talk_segment_jobs"][last][-1], corr)
        _seg(conn, policy, post, body, chosen, last)
    _judged(conn, body["talk_segment_jobs"][last][-1], -0.43)
    with pytest.raises(DropPost, match="lip sync failed and no repair changes it"):
        _seg(conn, policy, post, body, chosen, last)
    assert body["talk_modes"][last] == "talk"


def test_without_a_faceless_shot_there_is_no_cutaway(conn, policy):
    post, body, chosen = _story(conn, cutaway_shot=False)
    _seg(conn, policy, post, body, chosen, 1)
    for corr in (0.1, 0.12):
        _judged(conn, body["talk_segment_jobs"][1][-1], corr)
        _seg(conn, policy, post, body, chosen, 1)
    _judged(conn, body["talk_segment_jobs"][1][-1], 0.11)
    with pytest.raises(DropPost):
        _seg(conn, policy, post, body, chosen, 1)


def _gate(conn, job_id, gate, score):
    a = conn.execute("INSERT INTO assets (job_id, kind, uri, sha256, bytes) VALUES (%s, 'video', %s, 'x', 1) "
                     "RETURNING id", (job_id, f"s3://b/{job_id}.mp4")).fetchone()["id"]
    conn.execute("INSERT INTO reviews (asset_id, gate, passed, score) VALUES (%s, 'lip_sync', true, 0.6)", (a,))
    conn.execute("INSERT INTO reviews (asset_id, gate, passed, score) VALUES (%s, %s, false, %s)", (a, gate, score))


def test_a_take_that_keeps_failing_identity_starts_from_another_accepted_photo(conn, policy):
    post, body, chosen = _story(conn)
    scene_job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('image.zimage', 'done', %s, "
                             "'{}') RETURNING id", (post["creator_id"],)).fetchone()["id"]
    for name, ok in (("shot1.png", True), ("alt.png", True), ("bad.png", False)):
        a = conn.execute("INSERT INTO assets (job_id, kind, uri, sha256, bytes) VALUES (%s, 'image', %s, 'x', 1) "
                         "RETURNING id", (scene_job, f"s3://b/{name}")).fetchone()["id"]
        conn.execute("INSERT INTO reviews (asset_id, gate, passed, score) VALUES (%s, 'identity', %s, 8)", (a, ok))
    body["scene_jobs"] = [[] for _ in body["shots"]]
    body["scene_jobs"][1] = [str(scene_job)]
    first = _seg(conn, policy, post, body, chosen, 1)
    assert first["payload"]["image_uri"] == "s3://b/shot1.png"
    for score in (3, 3):
        _gate(conn, body["talk_segment_jobs"][1][-1], "identity", score)
        j = _seg(conn, policy, post, body, chosen, 1)
    assert j["payload"]["repair"]["cause"] == "identity" and j["payload"]["repair"]["knob"] == "next_photo"
    assert j["payload"]["image_uri"] == "s3://b/alt.png"
    _gate(conn, body["talk_segment_jobs"][1][-1], "identity", 6)             # better: the new photo is kept
    j = _seg(conn, policy, post, body, chosen, 1)
    assert j["payload"]["image_uri"] == "s3://b/alt.png"
