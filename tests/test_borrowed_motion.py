"""Borrowed motion never leaks a real identity (perf_compare 2026-11-25: seven Wan-Animate-2 clips of licensed stock
footage came back with the stock performers, identity 0-1 of 10, and were lip-synced unreviewed; the two dance tests
of 2026-11-24 did the same). Off in production, on in the test overlay; a clip that runs is judged on her identity
before any lip sync, and one under the floor is deleted."""

from __future__ import annotations

import uuid
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from psycopg.types.json import Jsonb

from manyworlds import breadth, policy as policy_mod, production
from manyworlds.review import video

from test_production import creator, real_refs
from test_video_review import ClipStub, _put, clip


def _with(policy, **blocks):
    body = dict(policy.body)
    for name, over in blocks.items():
        body[name] = {**(body.get(name) or {}), **over}
    return policy_mod.from_dict(body)


def _ref(policy):
    return policy.get("dance", "references")[0]["id"]


def test_borrowed_motion_is_off_in_production_and_on_in_the_test_overlay(raw_policy):
    assert raw_policy.get("generation", "borrowed_motion") is False
    shot = {"motion_reference": _ref(raw_policy)}
    assert production.dance_reference(raw_policy, shot) is None                 # even with dance on
    assert breadth.test_policy(raw_policy).get("generation", "borrowed_motion") is True
    on = breadth.test_policy(raw_policy)
    assert production.dance_reference(on, shot) == _ref(raw_policy)


def _animate_job(conn, store, tmp_path, cid, *, intermediate=True):
    payload = {"image_uri": _put(store, tmp_path, b"\x89PNG start"), "reference": "walk-suit",
               "scene": "behind the bar", "intermediate": intermediate}
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('comfy.wan_animate2', 'done', %s, %s) "
                       "RETURNING id", (cid, Jsonb(payload))).fetchone()["id"]
    conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'video', %s, 'x', 1)",
                 (job, cid, _put(store, tmp_path, b"mp4", ".mp4")))
    for k in range(8):
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'keyframe', %s, "
                     "'x', 1)", (job, cid, _put(store, tmp_path, b"jpg %d" % k, ".jpg")))
    return job


def test_an_intermediate_animate_clip_is_reviewed_and_one_showing_someone_else_is_deleted(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = _animate_job(conn, store, tmp_path, cid)
    assert job in video.pending_jobs(conn, policy)                                # judged before any lip sync
    uris = [r["uri"] for r in conn.execute("SELECT uri FROM assets WHERE job_id = %s", (job,)).fetchall()]
    [r] = video.review_job(conn, policy, ClipStub(clip(identity=1, identity_per_frame=[1] * 8)), store, job, tmp_path)
    assert r["dance_identity"] is False and r["purged"] is True and not r["accepted"]
    left = conn.execute("SELECT count(*) n FROM assets WHERE job_id = %s AND deleted_at IS NULL", (job,)).fetchone()["n"]
    assert left == 0 and not any(Path(url2pathname(urlparse(u).path)).exists() for u in uris)
    assert "identity guard" in conn.execute("SELECT error FROM jobs WHERE id = %s", (job,)).fetchone()["error"]
    state, out = production.motion._borrowed_state(conn, [str(job)])
    assert state == "rejected" and out is None                                   # never passed to lip sync


def test_a_clip_that_holds_her_identity_is_kept_and_lip_synced(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = _animate_job(conn, store, tmp_path, cid)
    assert production.motion._borrowed_state(conn, [str(job)]) == ("waiting", None)   # not reviewed yet
    [r] = video.review_job(conn, policy, ClipStub(clip(identity=8, identity_per_frame=[8] * 8), clip()), store, job,
                           tmp_path)
    assert r["dance_identity"] is True and "purged" not in r
    state, out = production.motion._borrowed_state(conn, [str(job)])
    assert state == "chosen" and out["uri"].endswith(".mp4")


def test_a_near_miss_on_the_mean_is_rejected_but_not_deleted(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = _animate_job(conn, store, tmp_path, cid)
    floor = policy.get("dance", "identity", "min_score")
    per = [floor] * 7 + [floor + 1]                                              # every frame at the floor, mean low
    [r] = video.review_job(conn, policy, ClipStub(clip(identity=floor, identity_per_frame=per)), store, job, tmp_path)
    assert r["dance_identity"] is False and r.get("purged") is False
    assert conn.execute("SELECT count(*) n FROM assets WHERE job_id = %s AND deleted_at IS NULL",
                        (job,)).fetchone()["n"] > 0
    assert production.motion._borrowed_state(conn, [str(job)])[0] == "rejected"


def test_a_walk_and_talk_clip_without_a_reference_is_still_not_reviewed(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('comfy.wan_i2v', 'done', %s, %s) "
                       "RETURNING id", (cid, Jsonb({"image_uri": "s3://b/a.png", "intermediate": True}))).fetchone()["id"]
    for kind in ("video", "keyframe"):
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, %s, %s, 'x', 1)",
                     (job, cid, kind, f"s3://b/{uuid.uuid4().hex}"))
    assert job not in video.pending_jobs(conn, policy)
