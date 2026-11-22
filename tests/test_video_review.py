"""Story clips are judged from keyframes before a post can use them: a clip whose face drifts
or whose hands melt is rejected and re-rendered with a new seed, and a post whose clips keep
failing is dropped instead of shipping one."""

from __future__ import annotations

import uuid

from psycopg.types.json import Jsonb

from manyworlds import production
from manyworlds.review import video
from manyworlds.worker.handlers.video_wan import keyframe_indices

from test_production import NOW, START, PlanStub, creator, finish, post, real_refs
from test_review import JudgeStub, ok


def clip(**kw):
    return {"identity": 9, "motion": 8, "realism": 8, "defects": [], "looks_under_min_age": False,
            "sexualized": False, "real_brand_marks": False, "notes": "fine", **kw}


class ClipStub:
    """Stands in for Claude: returns the given clip reviews in call order, records calls."""

    def __init__(self, *reviews):
        self.reviews, self.calls = list(reviews), []

    def model_for(self, role):
        return f"stub-{role}"

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        self.calls.append({"role": role, "content": content})
        return schema(**self.reviews[min(len(self.calls), len(self.reviews)) - 1])


def _put(store, tmp_path, data: bytes, suffix=".png"):
    p = tmp_path / f"{uuid.uuid4().hex}{suffix}"
    p.write_bytes(data)
    return store.put_file(p, f"v/{p.name}").uri


def rendered(conn, store, tmp_path, job_id, keyframes=4):
    """A worker finished this clip job: one video plus its keyframes."""
    j = conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s RETURNING creator_id", (job_id,)).fetchone()
    conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'video', %s, 'x', 1)",
                 (job_id, j["creator_id"], _put(store, tmp_path, b"mp4", ".mp4")))
    for k in range(keyframes):
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'keyframe', %s, 'x', 1)",
                     (job_id, j["creator_id"], _put(store, tmp_path, b"jpg %d" % k, ".jpg")))


def clip_job(conn, store, tmp_path, cid, *, kind="video.wan_i2v", scene="pouring tea; steam rises"):
    start = _put(store, tmp_path, b"\x89PNG start")
    return conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES (%s, 'queued', %s, %s) RETURNING id",
                        (kind, cid, Jsonb({"image_uri": start, "scene": scene}))).fetchone()["id"]


def test_keyframes_spread_over_the_clip_and_end_on_its_last_frame():
    assert keyframe_indices(121, 4) == [30, 60, 90, 120]
    assert keyframe_indices(3, 4) == [1, 2]          # never more keyframes than frames after the start


def test_a_clip_whose_face_drifts_is_rejected_without_paying_for_a_confirm(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = clip_job(conn, store, tmp_path, cid)
    rendered(conn, store, tmp_path, job)
    llm = ClipStub(clip(identity=3, drift_from=3, defects=["keyframe 3: different nose and jaw"]))
    [r] = video.review_job(conn, policy, llm, store, job, tmp_path)
    assert not r["accepted"] and not r["identity"] and r["motion"] and len(llm.calls) == 1
    rows = conn.execute("SELECT gate, passed FROM reviews WHERE asset_id = %s ORDER BY gate", (r["asset_id"],)).fetchall()
    assert {x["gate"]: x["passed"] for x in rows} == {"identity": False, "motion": True, "realism": True,
                                                      "safety": True, "scene": True}
    # The judge saw the references, the start photo, every keyframe (as JPEG) and the intended shot.
    texts = [b.get("text", "") for b in llm.calls[0]["content"]]
    assert sum(t.startswith("REFERENCE") for t in texts) == 3 and any(t.startswith("START PHOTO") for t in texts)
    assert sum(t.startswith("KEYFRAME") for t in texts) == 4 and any("steam rises" in t for t in texts)
    assert [b["source"]["media_type"] for b in llm.calls[0]["content"] if b.get("type") == "image"][-1] == "image/jpeg"


def test_a_good_clip_passes_both_stages_and_melting_hands_fail_motion(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    good, bad = clip_job(conn, store, tmp_path, cid), clip_job(conn, store, tmp_path, cid)
    rendered(conn, store, tmp_path, good)
    rendered(conn, store, tmp_path, bad)
    llm = ClipStub(clip(), clip(identity=8))
    [r] = video.review_job(conn, policy, llm, store, good, tmp_path)
    assert r["accepted"] and r["confirmed"] and [s["stage"] for s in r["stages"]] == ["screen", "confirm"]
    assert [c["role"] for c in llm.calls] == [policy.get("review", "video", "role"),
                                             policy.get("review", "video", "confirm_role")]
    [r] = video.review_job(conn, policy, ClipStub(clip(motion=3)), store, bad, tmp_path)
    assert not r["accepted"] and not r["motion"]


def test_only_finished_clips_with_keyframes_await_review(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    ready, no_keys, mirror = (clip_job(conn, store, tmp_path, cid), clip_job(conn, store, tmp_path, cid),
                              clip_job(conn, store, tmp_path, cid, kind="video.mirror"))
    rendered(conn, store, tmp_path, ready)
    rendered(conn, store, tmp_path, no_keys, keyframes=0)
    rendered(conn, store, tmp_path, mirror)
    assert video.pending_jobs(conn, policy) == [ready]
    out = video.run_pending(conn, policy, ClipStub(clip()), store, tmp_path)
    assert out["accepted"] == 1 and video.pending_jobs(conn, policy) == []


def _story_post(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    planner_post = post(0, "story_clip", ["pouring tea by a rainy window"], script="steam rises from the cup",
                        series="slow mornings")
    from manyworlds.brain import planner
    planner.plan_period(conn, policy, PlanStub([planner_post]), cid, start=START)
    pid = conn.execute("SELECT id FROM posts WHERE creator_id = %s", (cid,)).fetchone()["id"]
    assert production.advance(conn, policy, now=NOW)["started"] == 1
    finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok()] * 4), pid)   # the start photo is chosen
    assert production.advance(conn, policy, now=NOW)["producing"] == 1
    return pid


def _clip_jobs(conn, pid):
    return conn.execute("SELECT id, payload, status FROM jobs WHERE post_id = %s AND kind = 'video.wan_i2v' "
                        "ORDER BY created_at", (pid,)).fetchall()


def test_a_story_clip_post_is_ready_once_its_clip_passes_review(conn, policy, store, tmp_path):
    pid = _story_post(conn, policy, store, tmp_path)
    [job] = _clip_jobs(conn, pid)
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    p = job["payload"]
    assert p["image_uri"] == body["images"][0]["uri"]                       # animated from the chosen photo
    assert "steam rises" in p["prompt"] and "thin gold wire glasses" in p["prompt"]
    assert p["scene"] == "pouring tea by a rainy window Movement: steam rises from the cup"
    assert p["keyframes"] == policy.get("review", "video", "keyframes")
    assert p["frames"] == policy.get("generation", "video", "frames")
    rendered(conn, store, tmp_path, job["id"])
    assert production.advance(conn, policy, now=NOW)["producing"] == 1     # rendered, not yet reviewed
    video.run_pending(conn, policy, ClipStub(clip()), store, tmp_path)
    assert production.advance(conn, policy, now=NOW)["ready"] == 1
    row = conn.execute("SELECT body, asset_ids FROM posts WHERE id = %s", (pid,)).fetchone()
    assert row["body"]["video"]["uri"].endswith(".mp4") and len(row["asset_ids"]) == 2


def test_rejected_clips_get_new_seeds_then_the_post_is_dropped(conn, policy, store, tmp_path):
    import dataclasses
    prod = {**policy.body["production"], "reroute_failed_video": {}}     # rerouting off (tested in test_qa)
    policy = dataclasses.replace(policy, body={**policy.body, "production": prod})
    pid = _story_post(conn, policy, store, tmp_path)
    limit = policy.get("review", "video", "max_repairs_per_post")
    for _ in range(limit + 1):
        job = _clip_jobs(conn, pid)[-1]
        rendered(conn, store, tmp_path, job["id"])
        video.run_pending(conn, policy, ClipStub(clip(identity=2)), store, tmp_path)
        state = production.advance(conn, policy, now=NOW)
    assert state["dropped"] == 1
    jobs = _clip_jobs(conn, pid)
    assert len(jobs) == limit + 1 and len({j["payload"]["seed"] for j in jobs}) == limit + 1
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    assert "no clip passed review" in body["drop_reason"]


def test_a_clip_job_that_fails_is_retried_without_using_a_repair(conn, policy, store, tmp_path):
    pid = _story_post(conn, policy, store, tmp_path)
    [job] = _clip_jobs(conn, pid)
    conn.execute("UPDATE jobs SET status = 'failed' WHERE id = %s", (job["id"],))
    production.advance(conn, policy, now=NOW)
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    assert len(_clip_jobs(conn, pid)) == 2 and body.get("clip_repairs", 0) == 0 and body["follow_up_failures"] == 1


def test_talking_clips_go_through_the_same_gate(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    talk = clip_job(conn, store, tmp_path, cid, kind="talk.infinitetalk")
    rendered(conn, store, tmp_path, talk)
    assert video.pending_jobs(conn, policy) == [talk]


def _with_sync(policy, **lip_sync):
    import dataclasses
    review = {**policy.body["review"], "video": {**policy.body["review"]["video"],
              "lip_sync": {**policy.body["review"]["video"]["lip_sync"], **lip_sync}}}
    return dataclasses.replace(policy, body={**policy.body, "review": review})


def test_a_talking_clip_out_of_sync_is_rejected_before_any_paid_review(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    off, on = (clip_job(conn, store, tmp_path, cid, kind="talk.infinitetalk") for _ in range(2))
    for job, corr in ((off, 0.05), (on, 0.6)):
        rendered(conn, store, tmp_path, job)
        conn.execute("UPDATE assets SET meta = %s WHERE job_id = %s AND kind = 'video'",
                     (Jsonb({"sync": {"corr": corr, "offset_frames": 1, "face_share": 1.0, "frames": 150}}), job))
    pol = _with_sync(policy, enforce=True)
    llm = ClipStub(clip(), clip())
    [r] = video.review_job(conn, pol, llm, store, off, tmp_path)
    assert not r["accepted"] and r["lip_sync"] is False and llm.calls == []          # no judge paid for it
    [r] = video.review_job(conn, pol, llm, store, on, tmp_path)
    assert r["accepted"] and r["lip_sync"] is True and len(llm.calls) == 2
    gates = {x["gate"] for x in conn.execute("SELECT gate FROM reviews WHERE asset_id = %s", (r["asset_id"],))}
    assert "lip_sync" in gates


def test_until_calibrated_the_sync_score_is_recorded_but_never_rejects(conn, policy, store, tmp_path):
    assert policy.get("review", "video", "lip_sync", "enforce") is False
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = clip_job(conn, store, tmp_path, cid, kind="talk.infinitetalk")
    rendered(conn, store, tmp_path, job)
    conn.execute("UPDATE assets SET meta = %s WHERE job_id = %s AND kind = 'video'",
                 (Jsonb({"sync": {"corr": 0.01, "face_share": 1.0}}), job))
    [r] = video.review_job(conn, policy, ClipStub(clip(), clip()), store, job, tmp_path)
    assert r["accepted"] and r["sync"]["corr"] == 0.01 and "lip_sync" not in r


def _talk_with_voice(conn, store, tmp_path, cid, script):
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('talk.infinitetalk', 'queued', %s, %s) "
                       "RETURNING id", (cid, Jsonb({"image_uri": _put(store, tmp_path, b"\x89PNG start"), "script": script,
                                                    "scene": "at the podium"}))).fetchone()["id"]
    rendered(conn, store, tmp_path, job)
    conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) VALUES (%s, %s, 'audio', %s, 'x', 1, %s)",
                 (job, cid, _put(store, tmp_path, b"RIFF wav", ".wav"), Jsonb({"seconds": 6.0})))
    return job


def test_a_talking_clip_whose_voice_doesnt_say_the_script_fails_before_paid_review(conn, policy, store, tmp_path,
                                                                                  monkeypatch):
    from manyworlds import speech
    monkeypatch.setenv("TOGETHER_API_KEY", "k-test")
    heard = {"text": ""}
    monkeypatch.setattr(speech, "transcribe", lambda path, cfg, post=None: heard["text"])
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    script = "No comment on the trade. We played hard, the fans were great."
    garbled, clear = (_talk_with_voice(conn, store, tmp_path, cid, script) for _ in range(2))
    llm = ClipStub(clip(), clip())
    heard["text"] = "no common on the train we pay hard"
    [r] = video.review_job(conn, policy, llm, store, garbled, tmp_path)
    assert not r["accepted"] and r["intelligibility"] is False and llm.calls == []
    heard["text"] = "No comment on the trade. We played hard; the fans were great!"
    [r] = video.review_job(conn, policy, llm, store, clear, tmp_path)
    assert r["accepted"] and r["intelligibility"] is True
    usd = conn.execute("SELECT sum(usd) u FROM cost_entries WHERE category = 'hosted_api'").fetchone()["u"]
    assert float(usd) > 0                                                   # every transcription is in the ledger


def test_word_error_rate_ignores_case_and_punctuation():
    from manyworlds import speech
    assert speech.wer("it\u2019s fine honestly", "it's fine honestly") == 0.0
    assert speech.wer("one two three four", "one two four") == 0.25
    assert speech.wer("", "") == 0.0
