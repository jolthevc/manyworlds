"""Long talking videos: split at sentences, one judged talk job per segment, joined into one video."""

from __future__ import annotations

import subprocess

import pytest
from psycopg.types.json import Jsonb

from manyworlds import media, policy as policy_mod, production
from manyworlds.review import video
from manyworlds.talk_segments import estimate_seconds, split_script

from test_production import NOW, START, PlanStub, creator, finish, post, real_refs
from test_review import JudgeStub, ok
from test_video_review import ClipStub, clip, rendered

SENTENCE = ("So here is the thing about the corner shop on my street that nobody wants to talk about at all, "
            "and I have been thinking about it for weeks now.")   # 30 words
LONG = " ".join([SENTENCE] * 5)                                     # 150 words, ~60 s at 2.5 words/s


def test_a_short_script_stays_one_piece():
    assert split_script("Hi there. Bye now.", max_seconds=25, words_per_second=2.5) == ["Hi there. Bye now."]
    assert split_script("   ", max_seconds=25, words_per_second=2.5) == []


def test_a_long_script_splits_between_sentences_under_the_limit():
    segs = split_script(LONG, max_seconds=25, words_per_second=2.5)          # at most 62 words each
    assert len(segs) == 3 and all(estimate_seconds(s, 2.5) <= 25 for s in segs)
    assert all(s.endswith("now.") for s in segs) and " ".join(segs) == LONG


def test_a_run_on_sentence_is_split_at_commas_then_words():
    run_on = ", ".join(["one two three four five six seven eight nine ten"] * 12)   # 120 words, no full stop
    segs = split_script(run_on, max_seconds=10, words_per_second=2.0)               # at most 20 words each
    assert all(len(s.split()) <= 20 for s in segs)
    assert " ".join(segs).split() == run_on.split()
    words = split_script(" ".join(["word"] * 50), max_seconds=4, words_per_second=2.5)
    assert [len(s.split()) for s in words] == [10, 10, 10, 10, 10]


def _talking(policy, join="cut"):
    body = {**policy.body,
            "production": {**policy.body["production"],
                           "formats": [*policy.body["production"]["formats"], "talking_video"]},
            "generation": {**policy.body["generation"],
                           # 25 s segments keep the test scripts short (production takes run to 60 s)
                           "talk_long": {**policy.body["generation"]["talk_long"], "join": join,
                                         "max_segment_seconds": 25}}}
    return policy_mod.from_dict(body)


def _long_post(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    conn.execute("UPDATE creators SET anchors = anchors || %s WHERE id = %s",
                 (Jsonb({"voice": {"voice": "bf_emma"}}), cid))
    from manyworlds.brain import planner
    planner.plan_period(conn, policy, PlanStub([post(0, "talking_video", ["outside the corner shop at night"],
                                                     script=LONG)]), cid, start=START)
    pid = conn.execute("SELECT id FROM posts WHERE creator_id = %s", (cid,)).fetchone()["id"]
    production.advance(conn, policy, now=NOW, store=store)
    finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok()] * 4), pid)   # the start photo is chosen
    production.advance(conn, policy, now=NOW, store=store)
    return pid


def _segment_jobs(conn, pid):
    return conn.execute("SELECT id, status, payload FROM jobs WHERE post_id = %s AND kind = 'talk.infinitetalk' "
                        "ORDER BY (payload->'segment'->>'index')::int, created_at", (pid,)).fetchall()


@pytest.fixture()
def joined(monkeypatch):
    calls = []

    def fake_concat(paths, out, timeout=600):
        calls.append(list(paths))
        out.write_bytes(b"".join(p.read_bytes() for p in paths))
        return out
    monkeypatch.setattr(media, "concat", fake_concat)
    return calls


def test_cut_renders_every_segment_from_the_photo_then_joins_them(conn, policy, store, tmp_path, joined):
    policy = _talking(policy)
    pid = _long_post(conn, policy, store, tmp_path)
    jobs = _segment_jobs(conn, pid)
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    assert len(jobs) == 3 == len(body["talk_segments"])
    assert [j["payload"]["script"] for j in jobs] == body["talk_segments"]
    assert {j["payload"]["image_uri"] for j in jobs} == {body["images"][0]["uri"]}        # all from the photo
    assert len({j["payload"]["seed"] for j in jobs}) == 3
    assert [j["payload"]["segment"] for j in jobs] == [{"index": i, "of": 3} for i in range(3)]
    for j in jobs:
        rendered(conn, store, tmp_path, j["id"])
    video.run_pending(conn, policy, ClipStub(clip()), store, tmp_path)
    assert production.advance(conn, policy, now=NOW, store=store)["ready"] == 1
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    assert body["video"]["uri"].endswith("talk-joined.mp4") and len(joined[0]) == 3
    asset = conn.execute("SELECT meta FROM assets WHERE uri = %s", (body["video"]["uri"],)).fetchone()
    assert len(asset["meta"]["segments"]) == 3 and asset["meta"]["lip_sync"] is True


def test_a_rejected_segment_is_remade_alone(conn, policy, store, tmp_path, joined):
    policy = _talking(policy)
    pid = _long_post(conn, policy, store, tmp_path)
    first, second, third = _segment_jobs(conn, pid)
    for j in (first, third):
        rendered(conn, store, tmp_path, j["id"])
    video.run_pending(conn, policy, ClipStub(clip()), store, tmp_path)
    rendered(conn, store, tmp_path, second["id"])
    video.run_pending(conn, policy, ClipStub(clip(identity=2)), store, tmp_path)     # the face drifted
    production.advance(conn, policy, now=NOW, store=store)
    jobs = _segment_jobs(conn, pid)
    assert len(jobs) == 4
    redo = [j for j in jobs if j["payload"]["segment"]["index"] == 1 and j["id"] != second["id"]]
    assert len(redo) == 1 and redo[0]["payload"]["seed"] != second["payload"]["seed"]
    assert conn.execute("SELECT status FROM posts WHERE id = %s", (pid,)).fetchone()["status"] == "producing"


def test_chain_starts_each_segment_from_the_last_frame_before(conn, policy, store, tmp_path, joined):
    policy = _talking(policy, join="chain")
    pid = _long_post(conn, policy, store, tmp_path)
    [first] = _segment_jobs(conn, pid)                       # one at a time
    rendered(conn, store, tmp_path, first["id"])
    video.run_pending(conn, policy, ClipStub(clip()), store, tmp_path)
    production.advance(conn, policy, now=NOW, store=store)
    first_, second = _segment_jobs(conn, pid)
    last = conn.execute("SELECT uri FROM assets WHERE job_id = %s AND kind = 'keyframe' ORDER BY uri DESC LIMIT 1",
                        (first["id"],)).fetchone()["uri"]
    assert second["payload"]["image_uri"] == last and second["payload"]["segment"]["index"] == 1


def test_joining_twice_reuses_the_joined_video(conn, policy, store, tmp_path, joined):
    policy = _talking(policy)
    pid = _long_post(conn, policy, store, tmp_path)
    for j in _segment_jobs(conn, pid):
        rendered(conn, store, tmp_path, j["id"])
    video.run_pending(conn, policy, ClipStub(clip()), store, tmp_path)
    p = conn.execute("SELECT * FROM posts WHERE id = %s", (pid,)).fetchone()
    vids = conn.execute("SELECT a.id, a.uri FROM assets a JOIN jobs j ON j.id = a.job_id WHERE j.post_id = %s "
                        "AND a.kind = 'video'", (pid,)).fetchall()
    a = production.join_talk(conn, policy, p, vids, store)
    b = production.join_talk(conn, policy, p, vids, store)
    assert a["id"] == b["id"] and len(joined) == 1


def test_concat_really_joins_video_and_audio(tmp_path):   # ffmpeg from the imageio-ffmpeg wheel (base deps)
    import imageio_ffmpeg
    exe = media.ffmpeg()
    parts = []
    for i, secs in enumerate((1, 2)):
        p = tmp_path / f"p{i}.mp4"
        subprocess.run([exe, "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=25:duration={secs}",
                        "-f", "lavfi", "-i", f"sine=frequency=440:duration={secs}", "-shortest",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(p)], check=True)
        parts.append(p)
    out = media.concat(parts, tmp_path / "joined.mp4")
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(out))
    assert frames == 75 and abs(seconds - 3.0) < 0.2
