"""Narrated edits: once a post's photos pass review, one CPU job voices the script over them;
the post is ready when that video exists. Captions follow the speech chunk by chunk."""

from __future__ import annotations

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


def test_captions_over_a_supplied_voice_follow_its_cues_or_share_it_by_words():
    cues = edit.supplied_cues({"cues": [{"text": "Soup.", "start_s": 0.0, "end_s": 0.8}]}, "Soup.", 0.4, 1.0)
    assert cues == [("Soup.", 0.4, pytest.approx(1.2))]
    shared = edit.supplied_cues({}, "one two three. four.", 0.4, 4.0)
    assert [t for t, _, _ in shared] == ["one two three.", "four."]
    assert shared[0][1:] == (0.4, pytest.approx(3.4)) and shared[1][2] == pytest.approx(4.4)
