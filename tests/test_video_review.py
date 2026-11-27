"""Story clips are judged from keyframes before a post can use them: a clip whose face drifts
or whose hands melt is rejected and re-rendered with a new seed, and a post whose clips keep
failing is dropped instead of shipping one."""

from __future__ import annotations

import uuid

from psycopg.types.json import Jsonb

from manyworlds import production
from manyworlds.review import video
from manyworlds.worker.keyframes import keyframe_indices

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


def clip_job(conn, store, tmp_path, cid, *, kind="comfy.wan_i2v", scene="pouring tea; steam rises"):
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
    # just over every bar, so the screen alone doesn't clear it (clear_pass_margin) and the confirm runs
    lo = {k: policy.get("review", "video", f"min_{k}") for k in ("identity", "motion", "realism")}
    llm = ClipStub(clip(**lo), clip(identity=8))
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
                              clip_job(conn, store, tmp_path, cid, kind="image.mirror"))
    rendered(conn, store, tmp_path, ready)
    rendered(conn, store, tmp_path, no_keys, keyframes=0)
    rendered(conn, store, tmp_path, mirror)
    assert video.pending_jobs(conn, policy) == [ready]
    out = video.run_pending(conn, policy, ClipStub(clip()), store, tmp_path)
    assert out["accepted"] == 1 and video.pending_jobs(conn, policy) == []


def test_an_intermediate_motion_clip_is_not_reviewed(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    story = clip_job(conn, store, tmp_path, cid, kind="comfy.wan_i2v")
    motion = clip_job(conn, store, tmp_path, cid, kind="comfy.wan_i2v")
    conn.execute("UPDATE jobs SET payload = payload || %s WHERE id = %s", (Jsonb({"intermediate": True}), motion))
    for job in (story, motion):
        rendered(conn, store, tmp_path, job, keyframes=1)
    assert video.pending_jobs(conn, policy) == [story]   # walk-and-talk's source clip: the take is judged instead


def _story_post(conn, policy, store, tmp_path, scenes=("pouring tea by a rainy window",)):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    planner_post = post(0, "story_clip", list(scenes), script="steam rises from the cup",
                        series="slow mornings")
    from manyworlds.brain import planner
    planner.plan_period(conn, policy, PlanStub([planner_post]), cid, start=START)
    pid = conn.execute("SELECT id FROM posts WHERE creator_id = %s", (cid,)).fetchone()["id"]
    assert production.advance(conn, policy, now=NOW)["started"] == 1
    finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok()] * 4), pid)   # the start photo is chosen
    assert production.advance(conn, policy, now=NOW)["producing"] == 1
    return pid


def _clip_jobs(conn, pid):
    return conn.execute("SELECT id, payload, status FROM jobs WHERE post_id = %s AND kind = 'comfy.wan_i2v' "
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


def test_a_story_clip_is_graded_with_no_captions_and_gets_a_cover_from_its_keyframes(conn, policy, store, tmp_path,
                                                                                      monkeypatch):
    import dataclasses

    from manyworlds import captions, finishing
    calls = []

    def fake_finish(video, out, *, look, grain=None, hook_ass=None, ass=None, has_voice=True, policy, timeout=None, **_):
        calls.append(ass or hook_ass)
        out.write_bytes(b"graded")
        return out
    monkeypatch.setattr(finishing, "finish", fake_finish)
    monkeypatch.setattr(captions, "burn", lambda *a, **k: calls.append("burn"))
    shipped = dataclasses.replace(policy, body={
        **policy.body, "finishing": {**policy.body["finishing"], "enabled": True},
        "burned_captions": {**policy.body["burned_captions"], "enabled": True}})   # as in policy/default.yaml
    pid = _story_post(conn, shipped, store, tmp_path)
    [job] = _clip_jobs(conn, pid)
    rendered(conn, store, tmp_path, job["id"])
    video.run_pending(conn, shipped, ClipStub(clip()), store, tmp_path)
    assert production.advance(conn, shipped, now=NOW, store=store)["ready"] == 1
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    raw = conn.execute("SELECT uri FROM assets WHERE job_id = %s AND kind = 'video'", (job["id"],)).fetchone()["uri"]
    assert calls == [None]                                       # graded once; a story clip says nothing
    assert body["video"]["uri"].endswith("final.mp4") and body["video_raw"]["uri"] == raw
    assert body["finish"]["captions"] is None and body["finish"]["passes"] == 1
    keys = [r["uri"] for r in conn.execute("SELECT uri FROM assets WHERE job_id = %s AND kind = 'keyframe'",
                                           (job["id"],)).fetchall()]
    assert body["cover"]["uri"] in keys


def test_a_clip_edit_makes_one_clip_per_scene_and_joins_them(conn, policy, store, tmp_path, monkeypatch):
    from manyworlds import media
    joined = []

    def fake_concat(paths, out, **kw):
        joined.append(len(paths))
        out.write_bytes(b"joined")
        return out
    monkeypatch.setattr(media, "concat", fake_concat)
    monkeypatch.setattr(media, "duration", lambda p, **k: 5.0)
    pid = _story_post(conn, policy, store, tmp_path, scenes=("a fitting room mirror", "a street at golden hour"))
    jobs = _clip_jobs(conn, pid)
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    assert len(jobs) == 2 and [j["payload"]["image_uri"] for j in jobs] == [i["uri"] for i in body["images"]]
    assert len({j["payload"]["seed"] for j in jobs}) == 2
    for j in jobs:
        rendered(conn, store, tmp_path, j["id"])
    video.run_pending(conn, policy, ClipStub(clip()), store, tmp_path)
    assert production.advance(conn, policy, now=NOW, store=store)["ready"] == 1
    row = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    assert joined == [2] and row["video"]["uri"].endswith(".mp4")


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
    assert r["accepted"] and r["lip_sync"] is True and llm.calls        # in sync: the judge reviews it
    gates = {x["gate"] for x in conn.execute("SELECT gate FROM reviews WHERE asset_id = %s", (r["asset_id"],))}
    assert "lip_sync" in gates


def test_when_not_enforced_the_sync_score_is_recorded_but_never_rejects(conn, policy, store, tmp_path):
    assert policy.get("review", "video", "lip_sync", "enforce") is True   # enforced since 2026-11-22
    policy = _with_sync(policy, enforce=False)
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


def test_cached_word_timestamps_are_reused_instead_of_a_second_transcription(conn, policy, store, tmp_path,
                                                                            monkeypatch):
    from manyworlds import speech

    def never(*a, **kw):
        raise AssertionError("transcribed again although meta.words was cached")

    monkeypatch.setattr(speech, "transcribe", never)
    monkeypatch.setattr(speech, "_request", never)
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    script = "No comment on the trade. We played hard, the fans were great."
    good, bad = (_talk_with_voice(conn, store, tmp_path, cid, script) for _ in range(2))
    for job, said in ((good, "No comment on the trade. We played hard; the fans were great!"),
                      (bad, "no common on the train we pay hard")):
        words = [{"word": w, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, w in enumerate(said.split())]
        conn.execute("UPDATE assets SET meta = %s WHERE job_id = %s AND kind = 'audio'",
                     (Jsonb({"seconds": 6.0, "words": words}), job))
    llm = ClipStub(clip(), clip())
    [r] = video.review_job(conn, policy, llm, store, bad, tmp_path)
    assert not r["accepted"] and r["intelligibility"] is False and llm.calls == []
    [r] = video.review_job(conn, policy, llm, store, good, tmp_path)
    assert r["accepted"] and r["intelligibility"] is True
    paid = conn.execute("SELECT count(*) n FROM cost_entries WHERE category = 'hosted_api'").fetchone()["n"]
    assert paid == 0                                                         # nothing paid twice


def test_word_error_rate_ignores_case_and_punctuation():
    from manyworlds import speech
    assert speech.wer("it\u2019s fine honestly", "it's fine honestly") == 0.0
    assert speech.wer("one two three four", "one two four") == 0.25
    assert speech.wer("", "") == 0.0


def test_a_clip_the_screen_clears_by_the_margin_skips_the_paid_confirm(conn, policy, store, tmp_path):
    m = policy.get("review", "video", "clear_pass_margin")
    lo = {k: policy.get("review", "video", f"min_{k}") for k in ("identity", "motion", "realism")}
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    clear, close = clip_job(conn, store, tmp_path, cid), clip_job(conn, store, tmp_path, cid)
    rendered(conn, store, tmp_path, clear)
    rendered(conn, store, tmp_path, close)
    llm = ClipStub(clip(**{k: v + m for k, v in lo.items()}))
    [r] = video.review_job(conn, policy, llm, store, clear, tmp_path)
    assert r["accepted"] and r["confirmed"] and r.get("clear_pass") and len(llm.calls) == 1
    llm = ClipStub(clip(**lo), clip(**lo))                                   # just over the bars: confirm it
    [r] = video.review_job(conn, policy, llm, store, close, tmp_path)
    assert r["accepted"] and r["confirmed"] and not r.get("clear_pass") and len(llm.calls) == 2


# --- direction: a directed clip or photo is also scored on how well it follows its shot sheet -------------
SHOT = {"set": {"location": "a rainy bus stop"}, "framing": {"shot_size": "medium shot", "camera_move": "push_in"},
        "blocking": [{"at_s": 0, "action": "checks the timetable", "eyeline": "prop"},
                     {"at_s": 2.5, "action": "turns to the lens and shrugs", "eyeline": "lens"}]}


def _directed_clip(conn, store, tmp_path, cid):
    payload = {"image_uri": _put(store, tmp_path, b"\x89PNG start"), "scene": "at a rainy bus stop", "shot": SHOT}
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('comfy.wan_i2v', 'queued', %s, %s) "
                       "RETURNING id", (cid, Jsonb(payload))).fetchone()["id"]
    rendered(conn, store, tmp_path, job)
    return job


def test_a_directed_clip_is_judged_on_its_beats_and_rejected_below_the_direction_minimum(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    low, good = _directed_clip(conn, store, tmp_path, cid), _directed_clip(conn, store, tmp_path, cid)
    minimum = policy.get("review", "direction", "min_score")
    llm = ClipStub(clip(direction=minimum - 2))
    [r] = video.review_job(conn, policy, llm, store, low, tmp_path)
    assert not r["accepted"] and r["direction"] is False and r["identity"] and len(llm.calls) == 1
    texts = [b.get("text", "") for b in llm.calls[0]["content"]]
    brief = next(t for t in texts if t.startswith("DIRECTED SHOT"))
    assert "checks the timetable" in brief and "at 2.5 s: turns to the lens and shrugs" in brief
    assert "camera push in" in brief and "RUBRIC direction v1" in brief
    rows = {x["gate"]: (x["passed"], x["score"]) for x in conn.execute(
        "SELECT gate, passed, score FROM reviews WHERE asset_id = %s", (r["asset_id"],))}
    assert rows["direction"] == (False, minimum - 2) and rows["identity"][0] is True
    [r] = video.review_job(conn, policy, ClipStub(clip(direction=minimum + 2)), store, good, tmp_path)
    assert r["accepted"] and r["direction"] is True


def test_a_clip_without_a_shot_has_no_direction_gate(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = clip_job(conn, store, tmp_path, cid)
    rendered(conn, store, tmp_path, job)
    llm = ClipStub(clip(direction=1))                       # ignored: nothing was directed
    [r] = video.review_job(conn, policy, llm, store, job, tmp_path)
    assert r["accepted"] and "direction" not in r
    assert not any(str(b.get("text", "")).startswith("DIRECTED SHOT") for b in llm.calls[0]["content"])
    gates = {x["gate"] for x in conn.execute("SELECT gate FROM reviews WHERE asset_id = %s", (r["asset_id"],))}
    assert "direction" not in gates


def test_a_directed_photo_is_rejected_below_the_direction_minimum(conn, policy, store, tmp_path):
    from manyworlds.review import gates
    from test_review import creator_with_job
    _, job = creator_with_job(conn, store, tmp_path, n_images=2)
    conn.execute("UPDATE jobs SET payload = %s WHERE id = %s", (Jsonb({"scene": "at a rainy bus stop", "shot": SHOT}), job))
    minimum = policy.get("review", "direction", "min_score")
    judge = JudgeStub([ok(direction=minimum - 1), ok(direction=minimum + 3)])
    results = gates.review_job(conn, policy, judge, store, job, tmp_path)
    assert [r["accepted"] for r in results] == [False, True] and [r["direction"] for r in results] == [False, True]
    brief = next(b["text"] for b in judge.calls[0]["content"] if str(b.get("text", "")).startswith("DIRECTED SHOT"))
    assert "checks the timetable" in brief and "shrugs" not in brief                # a photo is its first beat


# --- the breadth test's batches are judged under the test overlay ------------------------------------------------
def test_a_clip_in_a_breadth_batch_is_judged_under_the_test_overlay(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    test, prod = _directed_clip(conn, store, tmp_path, cid), _directed_clip(conn, store, tmp_path, cid)
    conn.execute("UPDATE jobs SET batch = %s WHERE id = %s", (f"{policy.get('operations', 'test_overlay', 'batches')[0]}r1", test))
    conn.execute("UPDATE jobs SET batch = 'post-1' WHERE id = %s", (prod,))
    low = clip(direction=1)
    [r] = video.review_job(conn, policy, ClipStub(low), store, test, tmp_path)
    assert r["accepted"] and r["direction"] is True          # direction is report-only in the overlay
    [r] = video.review_job(conn, policy, ClipStub(low), store, prod, tmp_path)
    assert not r["accepted"] and r["direction"] is False     # production keeps its bar


# --- motion transfer: every keyframe must still be her ----------------------------------------------------------
def _dance_job(conn, store, tmp_path, cid, *, kind="comfy.wan_animate2", video_file=None):
    job = clip_job(conn, store, tmp_path, cid, kind=kind, scene=None)
    conn.execute("UPDATE jobs SET payload = payload || %s WHERE id = %s", (Jsonb({"reference": "dance-casual"}), job))
    rendered(conn, store, tmp_path, job)
    if video_file is not None:
        conn.execute("UPDATE assets SET uri = %s WHERE job_id = %s AND kind = 'video'",
                     (store.put_file(video_file, f"v/{uuid.uuid4().hex}.mp4").uri, job))
    return job


def test_a_motion_transfer_clip_fails_when_any_keyframe_loses_her_face(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    drifts, holds = _dance_job(conn, store, tmp_path, cid), _dance_job(conn, store, tmp_path, cid)
    assert set(video.pending_jobs(conn, policy)) == {drifts, holds}      # animate-2 clips are reviewed
    ic = policy.get("dance", "identity")
    llm = ClipStub(clip(identity_per_frame=[9, 9, ic["min_score"] - 1, 9]))
    [r] = video.review_job(conn, policy, llm, store, drifts, tmp_path)
    assert not r["accepted"] and r["dance_identity"] is False and r["identity"] is True
    assert any(str(b.get("text", "")).startswith("PER-KEYFRAME IDENTITY") for b in llm.calls[0]["content"])
    row = conn.execute("SELECT passed, score, details FROM reviews WHERE asset_id = %s AND gate = 'dance_identity'",
                       (r["asset_id"],)).fetchone()
    assert row["passed"] is False and float(row["score"]) == ic["min_score"] - 1
    assert row["details"]["dance_identity"]["failed_frames"] == [2]
    [r] = video.review_job(conn, policy, ClipStub(clip(identity_per_frame=[9] * 4)), store, holds, tmp_path)
    assert r["accepted"] and r["dance_identity"] is True


def test_a_motion_transfer_review_fails_closed_without_per_keyframe_scores(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = _dance_job(conn, store, tmp_path, cid, kind="comfy.wan_i2v")   # any clip driven by a reference motion
    [r] = video.review_job(conn, policy, ClipStub(clip()), store, job, tmp_path)
    assert not r["accepted"] and r["dance_identity"] is False
    [r2] = video.review_job(conn, policy, ClipStub(clip(identity_per_frame=[9, 9])), store,
                            _dance_job(conn, store, tmp_path, cid), tmp_path)
    assert not r2["accepted"]                                         # 2 scores for 4 keyframes


def test_a_plain_clip_has_no_dance_identity_gate(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = clip_job(conn, store, tmp_path, cid)
    rendered(conn, store, tmp_path, job)
    llm = ClipStub(clip(identity_per_frame=[1, 1, 1, 1]))   # ignored: not motion transfer
    [r] = video.review_job(conn, policy, llm, store, job, tmp_path)
    assert r["accepted"] and "dance_identity" not in r
    assert not any(str(b.get("text", "")).startswith("PER-KEYFRAME") for b in llm.calls[0]["content"])


def test_the_dance_gate_cuts_its_keyframes_from_the_clip_when_the_worker_stored_fewer(conn, policy, store, tmp_path):
    from test_media import _clip
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = _dance_job(conn, store, tmp_path, cid, video_file=_clip(tmp_path / "dance.mp4", seconds=1.0, audio=False))
    n = int(policy.get("dance", "identity", "keyframes"))
    llm = ClipStub(clip(identity_per_frame=[9] * n))
    [r] = video.review_job(conn, policy, llm, store, job, tmp_path)
    texts = [str(b.get("text", "")) for b in llm.calls[0]["content"]]
    assert sum(t.startswith("KEYFRAME") for t in texts) == n > 4 and r["accepted"]


def _short_line_wav(path, voiced_frames=52, seed=0, fps=25, rate=16000):
    """A speech-like line of `voiced_frames` video frames (syllable bursts), as a 16-bit WAV; its per-frame
    loudness envelope (1 = loud, 0 = pause)."""
    import wave

    import numpy as np
    rng = np.random.default_rng(seed)
    env = (rng.random(voiced_frames) > 0.45).astype(float)
    samples = np.repeat(env, rate // fps) * rng.normal(0, 0.3, voiced_frames * (rate // fps)) + 0.001
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return env


def _short_take(conn, store, tmp_path, cid, mouth, lead_s=0.0):
    """A talking clip of len(mouth) frames whose voice line (its audio output) covers 52 of them, after `lead_s` of
    silence (the take's lead-in, meta.lead_s)."""
    job = clip_job(conn, store, tmp_path, cid, kind="talk.infinitetalk")
    rendered(conn, store, tmp_path, job)
    wav = tmp_path / f"{uuid.uuid4().hex}.wav"
    _short_line_wav(wav)
    conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) VALUES (%s, %s, 'audio', %s, 'x', 1, %s)",
                 (job, cid, store.put_file(wav, f"v/{wav.name}").uri, Jsonb({"seconds": 52 / 25})))
    conn.execute("UPDATE assets SET meta = %s WHERE job_id = %s AND kind = 'video'",
                 (Jsonb({"lead_s": lead_s, "sync": {"corr": -0.46, "offset_frames": -5, "face_share": 1.0, "frames": len(mouth),
                                  "fps": 25.0, "openness": [round(float(x), 4) for x in mouth]}}), job))
    return job


def test_a_short_line_whose_mouth_stays_parted_after_it_passes_on_its_voiced_frames(conn, policy, store, tmp_path):
    """perf_compare segment 8 (2026-11-25): an 81-frame InfiniteTalk window for a 52-frame line; the mouth follows the
    voice, then stays parted in a smile over the 29 silent frames the join cuts away."""
    import numpy as np
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    env = _short_line_wav(tmp_path / "probe.wav")
    mouth = np.concatenate([0.02 + env * 0.12, np.full(29, 0.14)])
    job = _short_take(conn, store, tmp_path, cid, mouth)
    llm = ClipStub(clip(), clip())
    [r] = video.review_job(conn, policy, llm, store, job, tmp_path)
    assert r["lip_sync"] is True and r["sync"]["voiced"]["corr"] >= 0.3 and r["sync"]["voiced"]["voiced_frames"] == 52
    row = conn.execute("SELECT score, details FROM reviews WHERE asset_id = %s AND gate = 'lip_sync'",
                       (r["asset_id"],)).fetchone()
    assert float(row["score"]) == r["sync"]["voiced"]["corr"] and "openness" not in row["details"]


def test_an_unsynced_or_frozen_mouth_still_fails_on_its_voiced_frames(conn, policy, store, tmp_path):
    import numpy as np
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    other = (np.random.default_rng(7).random(52) > 0.45).astype(float)      # another line's timing
    unsynced = np.concatenate([0.02 + other * 0.12, np.full(29, 0.14)])
    env = _short_line_wav(tmp_path / "probe.wav")
    frozen = np.concatenate([0.002 + env * 0.002, np.full(29, 0.002)])     # in step, but the mouth never opens
    for mouth in (unsynced, frozen):
        job = _short_take(conn, store, tmp_path, cid, mouth)
        llm = ClipStub(clip(), clip())
        [r] = video.review_job(conn, policy, llm, store, job, tmp_path)
        assert r["lip_sync"] is False and not r["accepted"] and llm.calls == []


def test_the_voiced_span_can_be_turned_off(conn, policy, store, tmp_path):
    import numpy as np
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    env = _short_line_wav(tmp_path / "probe.wav")
    job = _short_take(conn, store, tmp_path, cid, np.concatenate([0.02 + env * 0.12, np.full(29, 0.14)]))
    pol = _with_sync(policy, voiced_span={**policy.get("review", "video", "lip_sync", "voiced_span"), "enabled": False})
    [r] = video.review_job(conn, pol, ClipStub(clip(), clip()), store, job, tmp_path)
    assert r["lip_sync"] is False and "voiced" not in r["sync"]


def test_a_rolled_take_is_scored_against_its_line_after_the_lead_in(conn, policy, store, tmp_path):
    """Review 2026-11-26: a take opens on 0.3 s of silence (talk_comfy.rolls), but its stored voice line starts at 0;
    unshifted, the line sat 8 frames early (past the 5-frame lag search) and the voiced span could not pass."""
    import numpy as np
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    env = _short_line_wav(tmp_path / "probe.wav")
    mouth = np.concatenate([np.full(8, 0.02), 0.02 + env * 0.12, np.full(29, 0.14)])   # 0.3 s lead-in at 25 fps
    job = _short_take(conn, store, tmp_path, cid, mouth, lead_s=0.32)
    [r] = video.review_job(conn, policy, ClipStub(clip(), clip()), store, job, tmp_path)
    assert r["lip_sync"] is True and abs(r["sync"]["voiced"]["offset_frames"]) <= 1
    assert r["sync"]["voiced"]["voiced_frames"] == 60                                  # 8 lead-in + 52 voiced
    assert r["sync"]["null"]["margin"] is not None                                     # recorded, report-only


def test_a_clip_no_better_than_chance_is_rejected_once_a_null_margin_is_set(conn, policy, store, tmp_path):
    import numpy as np
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    env = _short_line_wav(tmp_path / "probe.wav")
    job = _short_take(conn, store, tmp_path, cid, np.concatenate([0.02 + env * 0.12, np.full(29, 0.14)]))
    conn.execute("UPDATE assets SET meta = jsonb_set(meta, '{sync,corr}', '0.5') WHERE job_id = %s AND kind = 'video'",
                 (job,))
    null = {**policy.get("review", "video", "lip_sync", "null"), "min_margin": 2.0}   # unreachable: must reject
    [r] = video.review_job(conn, _with_sync(policy, null=null), ClipStub(clip(), clip()), store, job, tmp_path)
    assert r["lip_sync"] is False and r["sync"]["null"]["margin"] < 2.0
