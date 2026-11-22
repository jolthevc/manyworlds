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


def _directed_post(conn, policy, store, tmp_path, shots, voice=None):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    if voice is not None:
        conn.execute("UPDATE creators SET anchors = anchors || %s WHERE id = %s", (Jsonb({"voice": voice}), cid))
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


# --- the designed voice: every spoken piece is a voice.speak job first -----------------------------------
DESIGNED = {"engine": "voxcpm", "ref_uri": "s3://b/voice/ref.wav", "ref_text": "Okay, so this morning.", "voice": "bf_emma"}


def _voice_jobs(conn, pid, status=None):
    rows = conn.execute("SELECT id, status, payload, idempotency_key FROM jobs WHERE post_id = %s "
                        "AND kind = 'voice.speak' ORDER BY idempotency_key", (pid,)).fetchall()
    return [r for r in rows if status is None or r["status"] == status]


def _spoken(conn, job):
    """The voice worker finished: one wav and each line's timing."""
    j = conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s RETURNING creator_id", (job["id"],)).fetchone()
    lines = [{"index": i, "text": line["text"], "start_s": 2.0 * i, "end_s": 2.0 * i + 1.5}
             for i, line in enumerate(job["payload"]["lines"])]
    uri = f"s3://b/voice/{job['id']}.wav"
    conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                 "VALUES (%s, %s, 'audio', %s, 'x', 1, %s)", (job["id"], j["creator_id"], uri, Jsonb({"lines": lines})))
    return uri, lines


def test_a_designed_voice_speaks_each_take_before_its_talk_job(conn, policy, store, tmp_path, joined):
    policy = _talking(policy)
    pid, body = _directed_post(conn, policy, store, tmp_path, [SHOP, DOORWAY], voice=DESIGNED)
    assert _segment_jobs(conn, pid) == []                                            # no talk job before its voice
    first, second = _voice_jobs(conn, pid)
    assert [first["idempotency_key"], second["idempotency_key"]] == [f"post:{pid}:voice:0:0", f"post:{pid}:voice:1:0"]
    for job, shot in zip((first, second), (SHOP, DOORWAY)):
        p = job["payload"]
        assert p["ref_uri"] == DESIGNED["ref_uri"] and p["ref_text"] == DESIGNED["ref_text"]
        assert p["lines"] == [{"text": line["text"], "delivery": "dry, quick"} for line in shot["performance"]["lines"]]
        assert p["takes"] == policy.get("voice", "takes")
    # A failed voice job is retried like any other step, with a new seed.
    conn.execute("UPDATE jobs SET status = 'failed' WHERE id = %s", (first["id"],))
    production.advance(conn, policy, now=NOW, store=store)
    retry = [j for j in _voice_jobs(conn, pid) if j["idempotency_key"] == f"post:{pid}:voice:0:1"]
    assert len(retry) == 1 and retry[0]["payload"]["seed"] != first["payload"]["seed"]
    assert conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]["follow_up_failures"] == 1
    assert _segment_jobs(conn, pid) == []
    uris = [_spoken(conn, j)[0] for j in _voice_jobs(conn, pid, "queued")]          # segment 0's retry and segment 1
    production.advance(conn, policy, now=NOW, store=store)
    talks = _segment_jobs(conn, pid)
    assert len(talks) == 2
    assert sorted(t["payload"]["audio_uri"] for t in talks) == sorted(uris)
    by_key = {j["idempotency_key"]: f"s3://b/voice/{j['id']}.wav" for j in _voice_jobs(conn, pid, "done")}
    assert talks[0]["payload"]["audio_uri"] == by_key[f"post:{pid}:voice:0:1"]
    assert talks[1]["payload"]["audio_uri"] == by_key[f"post:{pid}:voice:1:0"]
    assert talks[0]["payload"]["script"] == " ".join(line["text"] for line in SHOP["performance"]["lines"])
    production.advance(conn, policy, now=NOW, store=store)                           # idempotent: nothing new
    assert len(_voice_jobs(conn, pid)) == 3 and len(_segment_jobs(conn, pid)) == 2


def test_a_cutaway_uses_the_designed_voice_too(conn, policy, store, tmp_path, joined):
    policy = _talking(policy)
    pid, body = _directed_post(conn, policy, store, tmp_path, [SHOP, WINDOW, DOORWAY], voice=DESIGNED)
    assert body["talk_modes"] == ["talk", "narrated", "talk"]
    assert conn.execute("SELECT count(*) n FROM jobs WHERE post_id = %s AND kind = 'image.narrated'",
                        (pid,)).fetchone()["n"] == 0
    jobs = _voice_jobs(conn, pid)
    assert len(jobs) == 3
    assert jobs[1]["payload"]["lines"] == [{"text": "They still haven't cleaned that window, by the way.",
                                            "delivery": "dry, quick"}]
    spoken = [_spoken(conn, j) for j in jobs]
    production.advance(conn, policy, now=NOW, store=store)
    [cut] = conn.execute("SELECT payload FROM jobs WHERE post_id = %s AND kind = 'image.narrated'", (pid,)).fetchall()
    uri, lines = spoken[1]
    assert cut["payload"]["audio_uri"] == uri
    assert cut["payload"]["cues"] == [{k: line[k] for k in ("text", "start_s", "end_s")} for line in lines]
    assert [t["payload"]["audio_uri"] for t in _segment_jobs(conn, pid)] == [spoken[0][0], spoken[2][0]]


def test_a_kokoro_creator_gets_no_voice_job(conn, policy, store, tmp_path, joined):
    policy = _talking(policy)
    # No reference clip yet: still Kokoro (voices.engine), so production is exactly as before.
    pid, body = _directed_post(conn, policy, store, tmp_path, [SHOP, DOORWAY],
                               voice={"engine": "voxcpm", "voice": "bf_emma"})
    assert _voice_jobs(conn, pid) == []
    talks = _segment_jobs(conn, pid)
    assert len(talks) == 2 and all("audio_uri" not in t["payload"] for t in talks)
    assert talks[0]["payload"]["voice"] == "bf_emma"
    assert "voice_segment_jobs" not in body


def test_voice_lines_follow_the_shots_and_a_split_take_gets_only_its_part(policy):
    long_take = _says(*[SENTENCE] * 7)
    takes = production.shot_takes(policy, [long_take])
    body = {"shots": [long_take], "talk_shots": [i for _, _, i in takes], "talk_segments": [t for t, _, _ in takes]}
    parts = [production.voice_lines(body, k) for k in range(len(takes))]
    assert [len(p) for p in parts] == [len(t.split()) // 30 for t, _, _ in takes]   # whole lines, one per sentence
    assert all(line == {"text": SENTENCE, "delivery": "dry, quick", "pause_after_s": None}
               for p in parts for line in p)
    # One take on the shot: its lines; without shots, one line per sentence and no delivery.
    single = {"shots": [SHOP], "talk_shots": [0], "talk_segments": [], "talk_script": production.shot_takes(
        policy, [SHOP])[0][0]}
    assert [line["text"] for line in production.voice_lines(single)] == [
        line["text"] for line in SHOP["performance"]["lines"]]
    assert production.voice_lines({"script": "Hi there. Bye now."}) == [{"text": "Hi there."}, {"text": "Bye now."}]


# --- walk-and-talk: a moving take is a silent motion clip first, then lip-synced video-to-video --------------------
WALK = _says("Okay, walking you to the bakery.",
             set={"location": "a quiet street at dawn"}, framing={"shot_size": "medium shot", "device": "selfie"},
             blocking=[{"at_s": 0, "action": "walks toward the camera along the street", "eyeline": "lens"}])


def _motion_jobs(conn, pid, policy):
    return conn.execute("SELECT id, status, payload, idempotency_key FROM jobs WHERE post_id = %s AND kind = %s "
                        "ORDER BY idempotency_key", (pid, policy.get("generation", "talk_motion", "kind"))).fetchall()


def _without_motion(policy):
    gen = policy.body["generation"]
    return policy_mod.from_dict({**policy.body, "generation": {**gen, "talk_motion": {**gen["talk_motion"],
                                                                                       "enabled": False}}})


def test_which_shots_move(policy):
    assert production.moves(policy, WALK) and not production.moves(policy, SHOP)
    assert not production.moves(policy, DOORWAY)                 # turning back to the phone is not walking
    stands = {"blocking": [{"action": "puts the mug down"}, {"at_s": 2, "action": "stands up from the sofa"}]}
    assert production.moves(policy, stands)
    assert production.moves(policy, {"framing": {"camera_move": "push in"}})        # a camera move
    assert not production.moves(policy, {"framing": {"camera_move": "static"}})
    assert not production.moves(policy, {"framing": {"camera_move": "handheld"}})   # a held phone's sway is not one
    assert production.moves(policy, {"blocking": [{"action": "steps back and laughs"}]})     # whole words, inflected
    assert not production.moves(policy, {"blocking": [{"action": "eats a walkable snack"}]})
    assert not production.moves(policy, None) and not production.moves(_without_motion(policy), WALK)


def test_a_walking_take_is_a_motion_clip_first_then_lip_synced_onto_it(conn, policy, store, tmp_path, joined):
    policy = _talking(policy)
    pid, body = _directed_post(conn, policy, store, tmp_path, [WALK, DOORWAY])
    assert body["talk_shots"] == [0, 1]
    [motion] = _motion_jobs(conn, pid, policy)
    p = motion["payload"]
    assert motion["idempotency_key"] == f"post:{pid}:motion:0:0"
    assert p["image_uri"] == body["images"][0]["uri"] and p["shot"] == body["shots"][0]
    assert "walks toward the camera along the street" in p["prompt"] and "a quiet street at dawn" in p["prompt"]
    assert p["intermediate"] is True and p["keyframes"] == policy.get("generation", "talk_motion", "keyframes")
    fps = policy.get("generation", "comfy_video", "fps")
    seconds = len(WALK["performance"]["lines"][0]["text"].split()) / policy.get("generation", "talk_long",
                                                                                   "words_per_second")
    assert p["params"]["frames"] == p["frames"] == production._wan_frames(seconds, fps)
    assert (p["frames"] - 1) % 4 == 0 and p["frames"] >= seconds * fps
    # The static take is lip-synced from its photo as before; the walking one waits for its motion clip.
    [doorway] = _segment_jobs(conn, pid)
    assert doorway["payload"]["segment"]["index"] == 1 and "video_uri" not in doorway["payload"]
    rendered(conn, store, tmp_path, motion["id"], keyframes=0)     # the motion clip is made, and not reviewed
    production.advance(conn, policy, now=NOW, store=store)
    talks = _segment_jobs(conn, pid)
    walk = next(t for t in talks if t["payload"]["segment"]["index"] == 0)
    clip_uri = conn.execute("SELECT uri FROM assets WHERE job_id = %s AND kind = 'video'", (motion["id"],)).fetchone()["uri"]
    assert walk["payload"]["video_uri"] == clip_uri and walk["payload"]["image_uri"] == body["images"][0]["uri"]
    v2v = policy.get("generation", "talk_motion", "v2v")
    assert all(walk["payload"]["params"][k] == v for k, v in v2v.items())
    production.advance(conn, policy, now=NOW, store=store)                           # idempotent: nothing new
    assert len(_motion_jobs(conn, pid, policy)) == 1 and len(_segment_jobs(conn, pid)) == 2
    for t in talks:
        rendered(conn, store, tmp_path, t["id"])
    video.run_pending(conn, policy, ClipStub(clip()), store, tmp_path)
    assert production.advance(conn, policy, now=NOW, store=store)["ready"] == 1
    assert len(joined[0]) == 2


def test_a_static_take_gets_no_motion_clip(conn, policy, store, tmp_path, joined):
    policy = _talking(policy)
    pid, body = _directed_post(conn, policy, store, tmp_path, [SHOP, DOORWAY])
    assert _motion_jobs(conn, pid, policy) == [] and "motion_segment_jobs" not in body
    assert all("video_uri" not in j["payload"] for j in _segment_jobs(conn, pid))
    # Switched off, a walking take is lip-synced from its photo too.
    off = _without_motion(policy)
    pid, body = _directed_post(conn, off, store, tmp_path, [WALK])
    [talk] = _segment_jobs(conn, pid)
    assert _motion_jobs(conn, pid, off) == [] and "video_uri" not in talk["payload"]


def test_a_single_walking_take_is_voiced_and_moved_at_once_and_lip_synced_once_both_are_done(conn, policy, store,
                                                                                            tmp_path, joined):
    policy = _talking(policy)
    pid, body = _directed_post(conn, policy, store, tmp_path, [WALK], voice=DESIGNED)
    assert body["talk_segments"] == []                                               # one take: one clip
    [voice_job] = _voice_jobs(conn, pid)
    [motion] = _motion_jobs(conn, pid, policy)                                       # in parallel with the voice
    assert motion["idempotency_key"] == f"post:{pid}:motion:0"
    assert _segment_jobs(conn, pid) == []
    uri, _ = _spoken(conn, voice_job)
    production.advance(conn, policy, now=NOW, store=store)
    assert _segment_jobs(conn, pid) == []                                            # still waits for the clip
    conn.execute("UPDATE jobs SET status = 'failed' WHERE id = %s", (motion["id"],))  # a machine vanished
    production.advance(conn, policy, now=NOW, store=store)
    first, retry = _motion_jobs(conn, pid, policy)
    assert retry["idempotency_key"] == f"post:{pid}:motion:1" and retry["payload"]["seed"] != motion["payload"]["seed"]
    rendered(conn, store, tmp_path, retry["id"], keyframes=0)
    production.advance(conn, policy, now=NOW, store=store)
    [talk] = _segment_jobs(conn, pid)
    clip_uri = conn.execute("SELECT uri FROM assets WHERE job_id = %s AND kind = 'video'", (retry["id"],)).fetchone()["uri"]
    assert talk["payload"]["video_uri"] == clip_uri and talk["payload"]["audio_uri"] == uri
    assert talk["payload"]["params"]["steps"] == policy.get("generation", "talk_motion", "v2v", "steps")
