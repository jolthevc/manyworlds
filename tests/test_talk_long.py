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


# --- hybrid: some segments on camera, the rest cutaways over the other photos ----------------------------
def test_the_hook_and_sign_off_stay_on_camera_and_enough_of_the_middle_joins_them():
    segs = ["a b c d", "a b", "a b c d e f", "a", "a b c"]          # 16 words
    modes = production.hybrid_modes(segs, 0.5)
    assert modes[0] == modes[-1] == "talk"
    on = sum(len(s.split()) for s, m in zip(segs, modes) if m == "talk")
    assert on / 16 >= 0.5 and modes.count("narrated") >= 1
    assert production.hybrid_modes(["a", "b"], 0.4) == ["talk", "talk"]


def test_a_hybrid_talking_video_renders_only_its_on_camera_part_and_joins_with_cutaways(conn, policy, store, tmp_path,
                                                                                     joined, monkeypatch):
    normalised = []
    def fake_normalize(p, out, **k):
        normalised.append(k)
        out.write_bytes(p.read_bytes())
        return out
    monkeypatch.setattr(media, "normalize", fake_normalize)
    policy = _talking(policy)
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    from manyworlds.brain import planner
    planner.plan_period(conn, policy, PlanStub([post(0, "talking_video", ["facing camera outside the corner shop",
                                                                          "the shop's window at night",
                                                                          "her hands around a paper cup"],
                                                     script=LONG)]), cid, start=START)
    pid = conn.execute("SELECT id FROM posts WHERE creator_id = %s", (cid,)).fetchone()["id"]
    production.advance(conn, policy, now=NOW, store=store)
    finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok()] * 4), pid)       # three photos chosen
    production.advance(conn, policy, now=NOW, store=store)
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    modes = body["talk_modes"]
    assert len(modes) == len(body["talk_segments"]) > 2 and modes[0] == modes[-1] == "talk" and "narrated" in modes
    talks = _segment_jobs(conn, pid)
    cuts = conn.execute("SELECT id, payload FROM jobs WHERE post_id = %s AND kind = 'image.narrated'", (pid,)).fetchall()
    assert len(talks) == modes.count("talk") and len(cuts) == modes.count("narrated")
    assert all(len(c["payload"]["images"]) == 2 and c["payload"]["width"] == 720 for c in cuts)   # the cutaway photos
    for j in talks:
        rendered(conn, store, tmp_path, j["id"])
    video.run_pending(conn, policy, ClipStub(clip()), store, tmp_path)
    for c in cuts:
        conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (c["id"],))
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'video', %s, 'x', 1)",
                     (c["id"], cid, store.put_file(_tmp(tmp_path, c["id"]), f"cut/{c['id']}.mp4").uri))
    assert production.advance(conn, policy, now=NOW, store=store)["ready"] == 1
    assert len(joined[0]) == len(modes) and len(normalised) == len(modes) and normalised[0]["width"] == 720


def _tmp(tmp_path, name):
    p = tmp_path / f"{name}.mp4"
    p.write_bytes(b"cut")
    return p


# --- directed: one take per shot, cut only at scene changes ----------------------------------------------
def _says(*lines, **shot):
    return {**shot, "performance": {"lines": [{"text": t, "delivery": "dry, quick"} for t in lines]}}


SHOP = _says("Right, quick one from outside the corner shop.", "The samosas are back, and I am not okay.",
             set={"location": "outside the corner shop at night"}, framing={"shot_size": "medium close-up",
                                                                            "device": "selfie"},
             blocking=[{"at_s": 0, "action": "talks to the phone", "eyeline": "lens"}])
WINDOW = _says("They still haven't cleaned that window, by the way.", cutaway=True,
               set={"location": "the shop's fogged window"}, framing={"shot_size": "close-up"})
DOORWAY = _says("Anyway. Going in. Wish me luck.",
                set={"location": "the shop doorway"}, framing={"shot_size": "medium shot", "device": "selfie"},
                blocking=[{"at_s": 0, "action": "turns back to the phone and grins", "eyeline": "lens"}])


def test_overlong_take_splits_at_sentence(policy):
    long_take = _says(*[SENTENCE] * 7)                                      # 210 words, ~84 s at 2.5 words/s
    takes = production.shot_takes(policy, [long_take])
    limit = policy.get("generation", "talk", "max_take_seconds") * policy.get("generation", "talk_long",
                                                                              "words_per_second")
    assert len(takes) == 2 and all(len(text.split()) <= limit for text, _, _ in takes)
    assert all(text.endswith("now.") for text, _, _ in takes)               # cut at a sentence end
    assert [(mode, i) for _, mode, i in takes] == [("talk", 0), ("talk", 0)]
    assert " ".join(text for text, _, _ in takes) == " ".join([SENTENCE] * 7)
    assert "dry, quick" not in takes[0][0]                                  # deliveries are never spoken


def test_cutaway_shot_is_narrated(policy):
    takes = production.shot_takes(policy, [SHOP, WINDOW, DOORWAY])
    assert [(mode, i) for _, mode, i in takes] == [("talk", 0), ("narrated", 1), ("talk", 2)]
    assert takes[1][0] == "They still haven't cleaned that window, by the way."
    # Without cutaways (hybrid off, or one photo) the cutaway's lines join the take before it.
    merged = production.shot_takes(policy, [SHOP, WINDOW, DOORWAY], cutaways=False)
    assert [(mode, i) for _, mode, i in merged] == [("talk", 0), ("talk", 2)]
    assert merged[0][0].endswith("by the way.")
    # A silent shot has no take; when only a cutaway speaks, the lines go to the on-camera shot.
    silent = {**SHOP, "performance": {}}
    assert production.shot_takes(policy, [silent, WINDOW]) == [(WINDOW["performance"]["lines"][0]["text"], "talk", 0)]
    assert production.shot_takes(policy, [silent]) == []


def _directed_post(conn, policy, store, tmp_path, shots):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    from manyworlds.brain import planner
    from manyworlds.direction import summary
    planner.plan_period(conn, policy, PlanStub([post(0, "talking_video", [summary(s) for s in shots])]), cid,
                        start=START)
    pid = conn.execute("SELECT id FROM posts WHERE creator_id = %s", (cid,)).fetchone()["id"]
    # The director (manyworlds.brain.director) writes the shots onto the scheduled post.
    conn.execute("UPDATE posts SET body = body || %s WHERE id = %s", (Jsonb({"shots": shots}), pid))
    production.advance(conn, policy, now=NOW, store=store)
    scene_jobs = conn.execute("SELECT payload FROM jobs WHERE post_id = %s ORDER BY idempotency_key", (pid,)).fetchall()
    assert [j["payload"]["shot"]["set"]["location"] for j in scene_jobs] == [s["set"]["location"] for s in shots]
    finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok()] * 4), pid)       # one photo per shot chosen
    production.advance(conn, policy, now=NOW, store=store)
    return pid, conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]


def test_one_take_per_shot(conn, policy, store, tmp_path, joined):
    policy = _talking(policy)
    pid, body = _directed_post(conn, policy, store, tmp_path, [SHOP, DOORWAY])
    assert body["talk_shots"] == [0, 1] and "talk_modes" not in body                # both on camera
    jobs = _segment_jobs(conn, pid)
    assert len(jobs) == 2
    for i, (j, shot) in enumerate(zip(jobs, (SHOP, DOORWAY))):
        p = j["payload"]
        assert p["script"] == " ".join(line["text"] for line in shot["performance"]["lines"])
        assert p["image_uri"] == body["images"][i]["uri"]                            # this shot's own photo
        assert p["shot"] == body["shots"][i] and p["prompt"].startswith("Camera: smartphone video")
        assert shot["set"]["location"] in p["prompt"] and "dry, quick" not in p["script"]
        assert "max_take_seconds" not in p["params"]
    for j in jobs:
        rendered(conn, store, tmp_path, j["id"])
    video.run_pending(conn, policy, ClipStub(clip()), store, tmp_path)
    assert production.advance(conn, policy, now=NOW, store=store)["ready"] == 1
    assert len(joined[0]) == 2


def test_a_directed_cutaway_is_narrated_over_its_own_photo(conn, policy, store, tmp_path, joined):
    policy = _talking(policy)
    pid, body = _directed_post(conn, policy, store, tmp_path, [SHOP, WINDOW, DOORWAY])
    assert body["talk_modes"] == ["talk", "narrated", "talk"] and body["talk_shots"] == [0, 1, 2]
    talks = _segment_jobs(conn, pid)
    [cut] = conn.execute("SELECT payload FROM jobs WHERE post_id = %s AND kind = 'image.narrated'", (pid,)).fetchall()
    assert [j["payload"]["image_uri"] for j in talks] == [body["images"][0]["uri"], body["images"][2]["uri"]]
    assert cut["payload"]["images"] == [body["images"][1]["uri"]]
    assert cut["payload"]["script"] == "They still haven't cleaned that window, by the way."
