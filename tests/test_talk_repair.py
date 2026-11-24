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
    for corr in (0.2, -0.1, 0.15, 0.05):                                    # each a different (bad) score
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
    with pytest.raises(DropPost, match="lip sync failed and no repair changes it"):
        _make(conn, policy, post, body)


def test_the_dropped_reason_reaches_the_post(conn, policy):
    post, body = _take(conn)
    _make(conn, policy, post, body)
    for _ in range(2):
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
    with pytest.raises(DropPost):
        _make(conn, policy, post, body, motion=None)
