"""The talking-video route comparison (breadth plan perf_compare, owner 2026-11-24): one directed story-time script made
two ways (InfiniteTalk, LongCat 1.6; the third, borrowed motion, was dropped 2026-11-25 with its stock clips) that
share their scene photos, voice line and seeds through body.shared_key, so only the talking route differs."""

from __future__ import annotations

import uuid

from psycopg.types.json import Jsonb

from manyworlds import breadth, direction, production
from manyworlds.production import common

from test_breadth import _creator, _paused

DESIGNED = {"voice": {"engine": "voxcpm", "ref_uri": "s3://b/voice.wav", "ref_text": "Okay so this happened."}}
ITEMS = ["perf_v1_infinitetalk", "perf_v2_longcat"]


def _post(key=None):
    return {"id": str(uuid.uuid4()), "body": {"shared_key": key} if key else {}}


def test_a_shared_key_shares_piece_keys_and_seeds_and_nothing_else_does():
    a, b, own = _post("perf-compare:r1"), _post("perf-compare:r1"), _post()
    assert common.piece_key(a) == common.piece_key(b) == "shared:perf-compare:r1"
    assert common._seed(a, 800, 1) == common._seed(b, 800, 1)
    assert common._seed(a, 800, 1) != common._seed(a, 800, 2)            # a retry still asks for a new output
    assert common.piece_key(own) == f"post:{own['id']}"
    assert common._seed(own, 0, 0) == int(own["id"].replace("-", "")[:7], 16)   # unchanged without a key
    assert common._seed(_post("perf-compare:r2"), 800, 1) != common._seed(a, 800, 1)   # another run: its own


def test_the_plan_is_one_script_two_routes_with_short_talking_lines(raw_policy):
    pol = breadth.test_policy(raw_policy)
    plan, items = breadth.load_plan(pol, "perf_compare")
    assert [i.id for i in items] == ITEMS and float(plan["max_usd"]) <= 6
    assert all("runpod_secure_l40s_vol" not in pools for pools in plan["pools"].values())   # the volume is full
    v1, v2 = items
    lines = [[direction.spoken(s) for s in i.body["shots"]] for i in items]
    assert lines[0] == lines[1] and v1.body["scenes"] == v2.body["scenes"]
    assert {i.body["shared_key"] for i in items} == {"perf-compare"}
    assert {i.body["edit_grammar"] for i in items} == {"story_time"}
    words = sum(len(t.split()) for t in lines[0])
    assert 80 <= words <= 120   # ~30-45 s at the voice's pace
    talking = [s for s in v1.body["shots"] if not s.get("cutaway")]
    assert len(talking) >= 6 and all(len(direction.spoken(s).split()) <= 12 for s in talking)   # <= one motion window
    assert v2.options == {"talk_profile": "longcat16"}
    assert pol.get("generation", "talk_profiles", "longcat16", "model") == "longcat_native"
    assert not any(production.dance_reference(pol, s) for s in v1.body["shots"] + v2.body["shots"])
    ids = {r["id"] for r in pol.get("dance", "references")}
    assert not any(i.startswith("perf-") for i in ids)   # the stock clips that leaked their performers: gone


def test_the_estimate_counts_shared_pieces_once(conn, policy):
    handle = _creator(conn, anchors=DESIGNED)
    out = breadth.run(conn, _paused(policy), "perf_compare", creator=handle, dry_run=True)
    jobs = {}
    for i in out["items"]:   # one entry per take: summed per kind
        jobs[i["id"]] = {}
        for j in i["estimate"]["jobs"]:
            jobs[i["id"]][j["kind"]] = jobs[i["id"]].get(j["kind"], 0) + j["n"]
    v1, v2 = (jobs[i] for i in ITEMS)
    assert v1["image.zimage"] == 9 and "voice.speak" in v1
    assert "image.zimage" not in v2 and "voice.speak" not in v2
    assert not any("comfy.wan_animate2" in j for j in (v1, v2))
    assert out["estimated_usd"] > 0


def test_the_posts_share_one_set_of_photos_and_one_voice_line(conn, policy, store):
    pol = _paused(policy)
    handle = _creator(conn, anchors=DESIGNED)
    out = breadth.run(conn, pol, "perf_compare", creator=handle, store=store, once=True)
    posts = conn.execute("SELECT * FROM posts WHERE body->>'breadth_run' = %s ORDER BY body->>'breadth_item'",
                         (out["run"],)).fetchall()
    assert len(posts) == len(ITEMS)
    keys = {p["body"]["shared_key"] for p in posts}
    assert keys == {f"perf-compare:{out['run']}"}                            # scoped to the run
    assert len({str(p["body"]["scene_jobs"]) for p in posts}) == 1           # the same scene jobs
    # one place for all nine shots (sets.shared_set): the leader is made first, the rest wait for its photo
    assert conn.execute("SELECT count(*) AS n FROM jobs WHERE kind = 'image.zimage' AND post_id IS NOT NULL").fetchone()["n"] == 1
    assert [len(j) for j in posts[0]["body"]["scene_jobs"]] == [1] + [0] * 8
    tpol = breadth.test_policy(pol)
    voice = DESIGNED["voice"]
    ids = {production.voice._voice_job(conn, tpol, dict(p), voice, [{"text": "Okay, story time."}], 0, segment=0)
           for p in posts}
    assert len(ids) == 1                                                     # one voice line for all


def test_a_borrowed_motion_take_is_animated_from_its_reference_then_lip_synced(conn, policy):
    """The walk-and-talk path for a shot with a listed motion reference (test overlay only), on the plan's own
    shots with one of the remaining licensed clips."""
    pol = breadth.test_policy(_paused(policy))
    _, items = breadth.load_plan(pol, "perf_compare")
    v1 = items[0]
    ref = pol.get("dance", "references")[0]["id"]
    cid = conn.execute("INSERT INTO creators (handle, bible, anchors) VALUES (%s, '{}', %s) RETURNING id",
                       (f"c{uuid.uuid4().hex[:6]}", Jsonb(DESIGNED))).fetchone()["id"]
    shots = [{**s, "motion_reference": ref} if k == 0 else s for k, s in enumerate(v1.body["shots"])]
    body = {**v1.body, "shots": shots, "talk_shots": [0, 1, 2, 4, 6, 7, 8]}
    post = {"id": uuid.uuid4(), "creator_id": cid, "format": "talking_video", "body": body}
    assert production.talk_v2v(pol)
    assert production.motion._needs_motion(pol, post, body, 0)               # shot 0 has a motion reference
    plain = {**v1.body, "talk_shots": [0]}
    assert not production.motion._needs_motion(pol, {**post, "body": plain}, plain, 0)
