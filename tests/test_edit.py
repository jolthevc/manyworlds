"""Narrated edits: once a post's photos pass review, one CPU job voices the script over them;
the post is ready when that video exists. Captions follow the speech chunk by chunk."""

from __future__ import annotations

import dataclasses

import pytest
from psycopg.types.json import Jsonb

from manyworlds import production
from manyworlds.brain import planner
from manyworlds.worker.handlers import edit_narrated as edit

from test_production import NOW, START, PlanStub, creator, finish, post, real_refs
from test_review import JudgeStub, ok


def test_script_chunks_follow_sentences_and_long_pauses():
    assert edit.chunks("thursday soup. week one. ... it's... soup.") == [
        ("thursday soup.", 0.3), ("week one.", 0.7), ("it's", 0.7), ("soup.", 0.3)]
    assert edit.wrap("a b c " * 10, 10).count("\n") >= 2


def test_a_narrated_post_is_ready_once_its_edit_exists(conn, policy, store, tmp_path):
    from brain_helpers import with_narration
    policy = with_narration(policy)   # off in policy until talking video exists; the pipeline is still tested
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    conn.execute("UPDATE creators SET anchors = anchors || %s WHERE id = %s",
                 (Jsonb({"voice": {"engine": "kokoro", "voice": "bf_emma"}}), cid))
    planner.plan_period(conn, policy, PlanStub([post(0, "narrated_edit", ["a", "b", "c"], script="soup. ... soup.",
                                                     on_screen_text="thursday soup")]), cid, start=START)
    pid = conn.execute("SELECT id FROM posts WHERE creator_id = %s", (cid,)).fetchone()["id"]
    production.advance(conn, policy, now=NOW)
    finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok()] * 4), pid)
    assert production.advance(conn, policy, now=NOW)["producing"] == 1          # edit queued
    job = conn.execute("SELECT id, payload FROM jobs WHERE post_id = %s AND kind = 'image.narrated'", (pid,)).fetchone()
    p = job["payload"]
    assert len(p["images"]) == 3 and p["voice"] == "bf_emma" and p["script"] == "soup. ... soup."
    assert p["on_screen_text"] == "thursday soup" and p["width"] == policy.get("production", "edit", "width")
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job["id"],))
    conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'video', 's3://b/e.mp4', 'x', 1)",
                 (job["id"], cid))
    assert production.advance(conn, policy, now=NOW)["ready"] == 1
    row = conn.execute("SELECT asset_ids, body FROM posts WHERE id = %s", (pid,)).fetchone()
    assert row["body"]["video"]["uri"] == "s3://b/e.mp4" and len(row["asset_ids"]) == 4


def test_failed_jobs_do_not_use_up_review_repairs(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    planner.plan_period(conn, policy, PlanStub([post(0)]), cid, start=START)
    pid = conn.execute("SELECT id FROM posts WHERE creator_id = %s", (cid,)).fetchone()["id"]
    production.advance(conn, policy, now=NOW)
    for _ in range(policy.get("review", "max_repairs_per_post") + 1):
        conn.execute("UPDATE jobs SET status = 'failed' WHERE post_id = %s AND status = 'queued'", (pid,))
        assert production.advance(conn, policy, now=NOW)["producing"] == 1
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    assert body.get("repairs", 0) == 0 and body["job_failures"] == policy.get("review", "max_repairs_per_post") + 1


def test_a_designed_voice_reads_the_edit_script_first(conn, policy, store, tmp_path):
    from brain_helpers import with_narration
    policy = with_narration(policy)
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    conn.execute("UPDATE creators SET anchors = anchors || %s WHERE id = %s",
                 (Jsonb({"voice": {"engine": "voxcpm", "ref_uri": "s3://b/voice/ref.wav", "ref_text": "Okay.",
                                   "voice": "bf_emma"}}), cid))
    planner.plan_period(conn, policy, PlanStub([post(0, "narrated_edit", ["a", "b", "c"], script="Soup. More soup!",
                                                     on_screen_text="soup")]),
                        cid, start=START)
    pid = conn.execute("SELECT id FROM posts WHERE creator_id = %s", (cid,)).fetchone()["id"]
    production.advance(conn, policy, now=NOW)
    finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok()] * 4), pid)
    production.advance(conn, policy, now=NOW)
    assert conn.execute("SELECT 1 FROM jobs WHERE post_id = %s AND kind = 'image.narrated'", (pid,)).fetchone() is None
    [voice] = conn.execute("SELECT id, payload, idempotency_key FROM jobs WHERE post_id = %s AND kind = 'voice.speak'",
                           (pid,)).fetchall()
    assert voice["idempotency_key"] == f"post:{pid}:voice:0"
    assert voice["payload"]["lines"] == [{"text": "Soup."}, {"text": "More soup!"}]
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (voice["id"],))
    lines = [{"text": "Soup.", "start_s": 0.0, "end_s": 0.8}, {"text": "More soup!", "start_s": 1.0, "end_s": 2.0}]
    conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                 "VALUES (%s, %s, 'audio', 's3://b/voice/edit.wav', 'x', 1, %s)", (voice["id"], cid, Jsonb({"lines": lines})))
    production.advance(conn, policy, now=NOW)
    [job] = conn.execute("SELECT payload FROM jobs WHERE post_id = %s AND kind = 'image.narrated'", (pid,)).fetchall()
    assert job["payload"]["audio_uri"] == "s3://b/voice/edit.wav" and job["payload"]["cues"] == lines
    assert job["payload"]["script"] == "Soup. More soup!"


def _finishing(policy, *, finish=True, captions=True):
    """The finish stage as shipped (policy/default.yaml turns both on; tests run with them off)."""
    return dataclasses.replace(policy, body={
        **policy.body, "finishing": {**policy.body["finishing"], "enabled": finish},
        "burned_captions": {**policy.body["burned_captions"], "enabled": captions}})


SOUP = "Thursday *soup*. More soup!"


def _edit_post(conn, policy, store, tmp_path, script=SOUP):
    """A narrated post whose photos are chosen and whose (Kokoro) edit job is queued."""
    from brain_helpers import with_narration
    policy = with_narration(policy)
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    conn.execute("UPDATE creators SET anchors = anchors || %s WHERE id = %s",
                 (Jsonb({"voice": {"engine": "kokoro", "voice": "bf_emma"}}), cid))
    planner.plan_period(conn, policy, PlanStub([post(0, "narrated_edit", ["a", "b", "c"], script=script,
                                                     on_screen_text="thursday soup")]), cid, start=START)
    pid = conn.execute("SELECT id FROM posts WHERE creator_id = %s", (cid,)).fetchone()["id"]
    production.advance(conn, policy, now=NOW, store=store)
    finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok()] * 4), pid)
    production.advance(conn, policy, now=NOW, store=store)
    job = conn.execute("SELECT id, payload FROM jobs WHERE post_id = %s AND kind = 'image.narrated'", (pid,)).fetchone()
    return policy, pid, cid, job


def _edited(conn, store, tmp_path, cid, job, path=None, meta=None):
    """The edit worker finished: its video (`path`, else a stand-in), with the meta the worker writes."""
    if path is None:
        path = tmp_path / f"{job['id']}.mp4"
        path.write_bytes(b"mp4")
    uri = store.put_file(path, f"edit/{job['id']}.mp4").uri
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job["id"],))
    meta = meta if meta is not None else {"seconds": 3.0, "lead_s": edit.LEAD_S, "tail_s": edit.TAIL_S, "cues": [
        {"text": "Thursday soup.", "start_s": 0.4, "end_s": 1.4}, {"text": "More soup!", "start_s": 1.7, "end_s": 2.4}]}
    conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                 "VALUES (%s, %s, 'video', %s, 'x', 1, %s)", (job["id"], cid, uri, Jsonb(meta)))
    return uri


def test_the_edit_draws_no_captions_of_its_own_when_they_are_burned_in(conn, policy, store, tmp_path):
    _, _, _, job = _edit_post(conn, _finishing(policy), store, tmp_path)
    assert job["payload"]["captions"] is False and job["payload"]["script"] == "Thursday soup. More soup!"
    _, _, _, job = _edit_post(conn, policy, store, tmp_path)                       # burned captions off
    assert "captions" not in job["payload"]


@pytest.fixture()
def fake_ffmpeg(monkeypatch):
    """finishing.finish and captions.burn stand-ins that record their calls; the frame size from probe."""
    from manyworlds import captions, finishing, media
    calls = []

    def fake_finish(video, out, *, look, grain=None, hook_ass=None, policy, timeout=None):
        calls.append(("finish", {"video": video, "out": out, "look": look, "policy": policy,
                                 "ass": hook_ass and open(hook_ass, encoding="utf-8").read()}))
        out.write_bytes(b"finished")
        return out

    def fake_burn(video, plan, out, *, workdir, policy=None, use_ass=None, timeout=900):
        calls.append(("burn", {"video": video, "plan": plan}))
        out.write_bytes(b"captioned")
        return out
    monkeypatch.setattr(finishing, "finish", fake_finish)
    monkeypatch.setattr(captions, "burn", fake_burn)
    monkeypatch.setattr(media, "probe", lambda p, **k: {"seconds": 3.0, "width": 1080, "height": 1920, "audio": True})
    return calls


def test_a_narrated_edit_is_finished_with_its_captions_in_one_pass(conn, policy, store, tmp_path, fake_ffmpeg,
                                                                    monkeypatch):
    from manyworlds import media
    monkeypatch.setattr(media, "has_libass", lambda **k: True)
    policy, pid, cid, job = _edit_post(conn, _finishing(policy), store, tmp_path)
    raw = _edited(conn, store, tmp_path, cid, job)
    assert production.advance(conn, policy, now=NOW, store=store)["ready"] == 1
    row = conn.execute("SELECT asset_ids, body FROM posts WHERE id = %s", (pid,)).fetchone()
    body = row["body"]
    assert body["video"]["uri"].endswith(f"post-{pid}/final.mp4") and body["video_raw"]["uri"] == raw
    assert str(row["asset_ids"][0]) == body["video"]["asset_id"]
    assert body["finish"] == {"finished": True, "look": policy.get("finishing", "default_look"), "captions": "clean",
                              "placement": "lower", "passes": 1}
    assert body["cover"]["uri"] in [im["uri"] for im in body["images"]]                # no keyframes: a photo
    [(name, call)] = fake_ffmpeg                                                        # grade, grain and captions
    assert name == "finish" and "Dialogue:" in call["ass"] and "Thursday" in call["ass"]
    assert "\\1c" in call["ass"]                                                        # *soup* in the accent colour
    assert call["policy"].get("finishing", "fonts_dir")                                 # the captions' own fonts
    # A rerun reuses the finished file.
    p = conn.execute("SELECT * FROM posts WHERE id = %s", (pid,)).fetchone()
    video = {"id": body["video_raw"]["asset_id"], "uri": raw}
    assert str(production.finish_video(conn, policy, p, body, video, store)["id"]) == body["video"]["asset_id"]
    assert len(fake_ffmpeg) == 1


def test_without_libass_the_captions_are_a_second_pass_over_the_graded_video(conn, policy, store, tmp_path,
                                                                             fake_ffmpeg, monkeypatch):
    from manyworlds import media
    monkeypatch.setattr(media, "has_libass", lambda **k: False)
    policy, pid, cid, job = _edit_post(conn, _finishing(policy), store, tmp_path)
    _edited(conn, store, tmp_path, cid, job)
    assert production.advance(conn, policy, now=NOW, store=store)["ready"] == 1
    (f, fc), (b, bc) = fake_ffmpeg
    assert (f, b) == ("finish", "burn") and fc["ass"] is None and bc["video"] == fc["out"]
    assert [w.text for w in bc["plan"].chunks[0].words][:2] == ["Thursday", "soup."]
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    assert body["finish"]["passes"] == 2


def test_captions_over_a_small_video_are_planned_at_the_finished_size(conn, policy, store, tmp_path, fake_ffmpeg,
                                                                      monkeypatch):
    from manyworlds import media
    monkeypatch.setattr(media, "has_libass", lambda **k: False)
    monkeypatch.setattr(media, "probe", lambda p, **k: {"seconds": 3.0, "width": 480, "height": 832, "audio": True})
    policy, pid, cid, job = _edit_post(conn, _finishing(policy), store, tmp_path)
    _edited(conn, store, tmp_path, cid, job)
    assert production.advance(conn, policy, now=NOW, store=store)["ready"] == 1
    (_, _), (_, bc) = fake_ffmpeg
    size = policy.get("finishing", "output")                     # the finish pass upscales 480x832 to it first
    assert (bc["plan"].width, bc["plan"].height) == (size["width"], size["height"])
    fake_ffmpeg.clear()
    captions_only, pid, cid, job = _edit_post(conn, _finishing(policy, finish=False), store, tmp_path)
    _edited(conn, store, tmp_path, cid, job)
    assert production.advance(conn, captions_only, now=NOW, store=store)["ready"] == 1
    [(_, bc)] = fake_ffmpeg
    assert (bc["plan"].width, bc["plan"].height) == (480, 832)    # no finish pass: no upscale


def test_each_switch_alone_and_both_off(conn, policy, store, tmp_path, fake_ffmpeg):
    captions_only, pid, cid, job = _edit_post(conn, _finishing(policy, finish=False), store, tmp_path)
    _edited(conn, store, tmp_path, cid, job)
    assert production.advance(conn, captions_only, now=NOW, store=store)["ready"] == 1
    assert [name for name, _ in fake_ffmpeg] == ["burn"]
    fake_ffmpeg.clear()
    grade_only, pid, cid, job = _edit_post(conn, _finishing(policy, captions=False), store, tmp_path)
    assert "captions" not in job["payload"]                                             # the edit draws its own
    _edited(conn, store, tmp_path, cid, job)
    assert production.advance(conn, grade_only, now=NOW, store=store)["ready"] == 1
    assert [name for name, _ in fake_ffmpeg] == ["finish"] and fake_ffmpeg[0][1]["ass"] is None
    fake_ffmpeg.clear()
    both_off, pid, cid, job = _edit_post(conn, policy, store, tmp_path)               # as tests run
    raw = _edited(conn, store, tmp_path, cid, job)
    assert production.advance(conn, both_off, now=NOW, store=store)["ready"] == 1
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    assert fake_ffmpeg == [] and body["video"]["uri"] == raw and "video_raw" not in body


def test_a_finish_that_keeps_failing_ships_the_unfinished_video_and_tells_the_owner(conn, policy, store, tmp_path,
                                                                                    monkeypatch):
    from manyworlds import finishing
    def broken(*a, **k):
        raise RuntimeError("ffmpeg could not finish raw.mp4")
    monkeypatch.setattr(finishing, "finish", broken)
    policy, pid, cid, job = _edit_post(conn, _finishing(policy, captions=False), store, tmp_path)
    raw = _edited(conn, store, tmp_path, cid, job)
    limit = policy.get("finishing", "max_failures")
    for _ in range(limit - 1):
        assert production.advance(conn, policy, now=NOW, store=store)["producing"] == 1
    assert production.advance(conn, policy, now=NOW, store=store)["ready"] == 1
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    assert body["video"]["uri"] == raw and body["finish_failures"] == limit and "could not finish" in body["finish_skipped"]
    event = conn.execute("SELECT type, routed_to FROM events WHERE creator_id = %s", (cid,)).fetchone()
    assert event == {"type": "finish_failed", "routed_to": "owner"}


def test_the_finish_stage_really_grades_and_burns_captions(conn, policy, store, tmp_path):
    """ffmpeg from the imageio-ffmpeg wheel: grade, grain, two-pass loudness and the captions (libass when this
    ffmpeg has it, else the Pillow overlay) on a real 3 s clip."""
    import subprocess

    from manyworlds import media
    src = tmp_path / "edit.mp4"
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=360x640:rate=30:duration=3",
                    "-f", "lavfi", "-i", "sine=frequency=330:duration=3", "-shortest", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
    policy, pid, cid, job = _edit_post(conn, _finishing(policy), store, tmp_path)
    raw = _edited(conn, store, tmp_path, cid, job, path=src)
    assert production.advance(conn, policy, now=NOW, store=store)["ready"] == 1
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    assert body["video_raw"]["uri"] == raw and body["finish"]["captions"] == "clean"
    out = store.get_file(body["video"]["uri"], tmp_path / "final.mp4")
    info = media.probe(out)
    size = policy.get("finishing", "output")                         # upscaled from 360x640 in the same pass
    assert (info["width"], info["height"]) == (size["width"], size["height"])
    assert info["audio"] and abs(info["seconds"] - 3.0) < 0.2
    assert out.read_bytes() != src.read_bytes()


def test_captions_over_a_supplied_voice_follow_its_cues_or_share_it_by_words():
    cues = edit.supplied_cues({"cues": [{"text": "Soup.", "start_s": 0.0, "end_s": 0.8}]}, "Soup.", 0.4, 1.0)
    assert cues == [("Soup.", 0.4, pytest.approx(1.2))]
    shared = edit.supplied_cues({}, "one two three. four.", 0.4, 4.0)
    assert [t for t, _, _ in shared] == ["one two three.", "four."]
    assert shared[0][1:] == (0.4, pytest.approx(3.4)) and shared[1][2] == pytest.approx(4.4)
