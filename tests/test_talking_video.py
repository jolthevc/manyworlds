"""Talking video posts: the creator says `script` to camera, lip-synced. The start photo is made and
reviewed like any scene; then the talk job turns it into a lip-synced clip in the creator's own voice,
which must pass the video gate like a story clip."""

from __future__ import annotations

import dataclasses

from psycopg.types.json import Jsonb

from manyworlds import production
from manyworlds.review import video

from test_production import NOW, START, PlanStub, creator, finish, post, real_refs
from test_review import JudgeStub, ok
from test_video_review import ClipStub, clip, rendered

SCRIPT = "Okay, honest review of the new corner shop: the samosas are elite, the coffee is a crime."


def _talking(policy):
    prod = {**policy.body["production"], "formats": [*policy.body["production"]["formats"], "talking_video"]}
    return dataclasses.replace(policy, body={**policy.body, "production": prod})


def _talk_post(conn, policy, store, tmp_path, **kw):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    conn.execute("UPDATE creators SET anchors = anchors || %s WHERE id = %s",
                 (Jsonb({"voice": {"voice": "bf_emma", "speed": 1.05}}), cid))
    from manyworlds.brain import planner
    planner.plan_period(conn, policy, PlanStub([post(0, "talking_video", ["outside the corner shop at night"],
                                                     script=SCRIPT, **kw)]), cid, start=START)
    pid = conn.execute("SELECT id FROM posts WHERE creator_id = %s", (cid,)).fetchone()["id"]
    assert production.advance(conn, policy, now=NOW)["started"] == 1
    finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok()] * 4), pid)   # the start photo is chosen
    assert production.advance(conn, policy, now=NOW)["producing"] == 1
    return pid


def _talk_jobs(conn, pid):
    return conn.execute("SELECT id, payload FROM jobs WHERE post_id = %s AND kind = 'talk.infinitetalk' "
                        "ORDER BY created_at", (pid,)).fetchall()


def test_a_talking_video_says_the_script_in_their_voice_from_the_chosen_photo(conn, policy, store, tmp_path):
    policy = _talking(policy)
    pid = _talk_post(conn, policy, store, tmp_path)
    [job] = _talk_jobs(conn, pid)
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    p, talk = job["payload"], policy.get("generation", "talk")
    assert p["image_uri"] == body["images"][0]["uri"] and p["script"] == SCRIPT
    assert (p["voice"], p["speed"]) == ("bf_emma", 1.05)
    assert "thin gold wire glasses" in p["prompt"] and "to the camera" in p["prompt"]
    assert p["params"]["steps"] == talk["steps"] and "voice" not in p["params"]
    assert p["scene"].startswith("outside the corner shop at night") and SCRIPT in p["scene"]
    assert p["keyframes"] == policy.get("review", "video", "keyframes")
    rendered(conn, store, tmp_path, job["id"])
    video.run_pending(conn, policy, ClipStub(clip()), store, tmp_path)
    assert production.advance(conn, policy, now=NOW)["ready"] == 1
    row = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()
    assert row["body"]["video"]["uri"].endswith(".mp4")


def test_a_rejected_talking_clip_is_remade_with_a_new_seed(conn, policy, store, tmp_path):
    policy = _talking(policy)
    pid = _talk_post(conn, policy, store, tmp_path)
    rendered(conn, store, tmp_path, _talk_jobs(conn, pid)[0]["id"])
    video.run_pending(conn, policy, ClipStub(clip(identity=2)), store, tmp_path)
    production.advance(conn, policy, now=NOW)
    jobs = _talk_jobs(conn, pid)
    assert len(jobs) == 2 and jobs[0]["payload"]["seed"] != jobs[1]["payload"]["seed"]


def test_talking_video_waits_until_production_makes_it(conn, policy, store, tmp_path):
    # Not in production.formats until a lip-sync model passes the gate on the benchmark (M10).
    assert "talking_video" not in policy.get("production", "formats")
    assert "talking_video" in policy.get("planner", "video_formats")
