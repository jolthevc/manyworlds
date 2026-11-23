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


class _Joins(list):
    """The joins made (each a list of paths), and `trims`: the frames trimmed off each motion clip before a join."""

    def __init__(self):
        super().__init__()
        self.trims: list[int] = []


@pytest.fixture()
def joined(monkeypatch):
    calls = _Joins()

    def fake_concat(paths, out, timeout=600):
        calls.append(list(paths))
        out.write_bytes(b"".join(p.read_bytes() for p in paths))
        return out

    def fake_trim(path, out, *, frames=1, timeout=600):   # motion-chain joins drop each later clip's first frame
        calls.trims.append(frames)
        out.write_bytes(path.read_bytes())
        return out
    monkeypatch.setattr(media, "concat", fake_concat)
    monkeypatch.setattr(media, "trim_start", fake_trim)
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


def test_trim_start_really_drops_the_first_frame(tmp_path):   # motion-chain joins (production.join_motion)
    import imageio_ffmpeg
    src = tmp_path / "clip.mp4"
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    cut = media.trim_start(src, tmp_path / "cut.mp4", frames=1)
    whole = media.trim_start(src, tmp_path / "whole.mp4", frames=0)
    assert imageio_ffmpeg.count_frames_and_secs(str(cut))[0] == 24
    assert imageio_ffmpeg.count_frames_and_secs(str(whole))[0] == 25
    joined = media.concat([whole, cut], tmp_path / "joined.mp4")
    assert imageio_ffmpeg.count_frames_and_secs(str(joined))[0] == 49


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
    out_w = policy.get("generation", "talk_hybrid", "output", "width")   # the delivery width (finishing.output)
    assert out_w == policy.get("finishing", "output", "width")
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
    assert all(len(c["payload"]["images"]) == 2 and c["payload"]["width"] == out_w for c in cuts)   # delivery size
    for j in talks:
        rendered(conn, store, tmp_path, j["id"])
    video.run_pending(conn, policy, ClipStub(clip()), store, tmp_path)
    for c in cuts:
        conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (c["id"],))
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'video', %s, 'x', 1)",
                     (c["id"], cid, store.put_file(_tmp(tmp_path, c["id"]), f"cut/{c['id']}.mp4").uri))
    assert production.advance(conn, policy, now=NOW, store=store)["ready"] == 1
    assert len(joined[0]) == len(modes) and len(normalised) == len(modes) and normalised[0]["width"] == out_w
    assert normalised[0]["fps"] == policy.get("finishing", "fps")      # one frame rate for the whole join


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


# --- walk-and-talk: chained motion clips, and only on the ComfyUI talk engine -----------------------------------
STROLL = "Okay, walking you to the bakery, the long way round because the light is good."   # 15 words
LONG_WALK = _says(STROLL, STROLL,                                                             # 12 s at 2.5 words/s
                  set={"location": "a quiet street at dawn"}, framing={"shot_size": "medium shot", "device": "selfie"},
                  blocking=[{"at_s": 0, "action": "walks toward the camera along the street", "eyeline": "lens"}])


def _motion(policy, **cfg):
    gen = policy.body["generation"]
    return policy_mod.from_dict({**policy.body, "generation": {**gen, "talk_motion": {**gen["talk_motion"], **cfg}}})


def _key_of(conn, job_id):
    return conn.execute("SELECT uri FROM assets WHERE job_id = %s AND kind = 'keyframe' ORDER BY uri DESC LIMIT 1",
                        (job_id,)).fetchone()["uri"]


def test_motion_lengths_chain_up_to_the_cap_and_skip_a_tiny_remainder(policy):
    p = _motion(policy, max_motion_seconds=5, max_motion_chain=3, chain_min_seconds=1.5)
    assert production.motion_lengths(p, 3.0) == [3.0]
    assert production.motion_lengths(p, 12.0) == [5.0, 5.0, 2.0]
    assert production.motion_lengths(p, 10.5) == [5.0, 5.0]            # 0.5 s left: a hold, not a clip
    assert production.motion_lengths(p, 30.0) == [5.0, 5.0, 5.0]       # capped: the rest ends on a hold
    assert production.motion_lengths(_motion(p, max_motion_chain=1), 12.0) == [5.0]
    assert production.motion_lengths(p, 0.0) == [0.0]


def test_a_long_walking_take_chains_motion_clips_from_each_last_frame_then_joins_them(conn, policy, store, tmp_path,
                                                                                        joined):
    policy = _motion(_talking(policy), max_motion_seconds=5, max_motion_chain=3, chain_min_seconds=1.5)
    pid, body = _directed_post(conn, policy, store, tmp_path, [LONG_WALK])
    assert body["talk_segments"] == []                                               # one take
    fps = policy.get("generation", "comfy_video", "fps")
    [first] = _motion_jobs(conn, pid, policy)
    assert first["idempotency_key"] == f"post:{pid}:motion:0"
    assert first["payload"]["frames"] == production._wan_frames(5.0, fps) and "chain" not in first["payload"]
    rendered(conn, store, tmp_path, first["id"], keyframes=1)
    production.advance(conn, policy, now=NOW, store=store)
    second = next(j for j in _motion_jobs(conn, pid, policy) if j["id"] != first["id"])
    assert second["idempotency_key"] == f"post:{pid}:motion:c1:0" and second["payload"]["chain"] == {"index": 1}
    assert second["payload"]["image_uri"] == _key_of(conn, first["id"])             # continues from its last frame
    assert second["payload"]["seed"] != first["payload"]["seed"] and second["payload"]["intermediate"] is True
    assert _segment_jobs(conn, pid) == []                                            # the take waits for its chain
    rendered(conn, store, tmp_path, second["id"], keyframes=1)
    production.advance(conn, policy, now=NOW, store=store)
    third = next(j for j in _motion_jobs(conn, pid, policy) if j["id"] not in (first["id"], second["id"]))
    assert third["idempotency_key"] == f"post:{pid}:motion:c2:0"
    assert third["payload"]["frames"] == production._wan_frames(2.0, fps)
    assert third["payload"]["image_uri"] == _key_of(conn, second["id"])
    rendered(conn, store, tmp_path, third["id"], keyframes=1)
    production.advance(conn, policy, now=NOW, store=store)
    [talk] = _segment_jobs(conn, pid)
    joined_uri = conn.execute("SELECT uri FROM assets WHERE meta ? 'motion_chain'").fetchone()["uri"]
    assert talk["payload"]["video_uri"] == joined_uri and len(joined) == 1 and len(joined[0]) == 3
    assert joined.trims == [0, 1, 1]                 # each later clip's first frame repeats the last one: trimmed
    production.advance(conn, policy, now=NOW, store=store)                           # idempotent: nothing new
    assert len(_motion_jobs(conn, pid, policy)) == 3 and len(_segment_jobs(conn, pid)) == 1 and len(joined) == 1
    assert joined.trims == [0, 1, 1]


def test_a_capped_chain_ends_the_take_on_a_hold(conn, policy, store, tmp_path, joined):
    policy = _motion(_talking(policy), max_motion_chain=1)
    pid, body = _directed_post(conn, policy, store, tmp_path, [LONG_WALK])
    [first] = _motion_jobs(conn, pid, policy)
    rendered(conn, store, tmp_path, first["id"], keyframes=1)
    production.advance(conn, policy, now=NOW, store=store)
    [talk] = _segment_jobs(conn, pid)
    clip_uri = conn.execute("SELECT uri FROM assets WHERE job_id = %s AND kind = 'video'", (first["id"],)).fetchone()["uri"]
    assert talk["payload"]["video_uri"] == clip_uri and len(_motion_jobs(conn, pid, policy)) == 1 and joined == []


def _talk_image(policy, image, env=None):
    images = policy.body["worker_images"]
    talk = {**images["talk"], "image": image}
    if env:
        talk["env"] = {**(images["talk"].get("env") or {}), **env}
    return policy_mod.from_dict({**policy.body, "worker_images": {**images, "talk": talk}})


def test_only_the_comfyui_talk_engine_lip_syncs_a_motion_clip(conn, policy, store, tmp_path, joined):
    assert production.talk_v2v(policy)                                               # mw-worker-talkcomfy
    script_engine = _talk_image(_talking(policy), "mw-worker-talk")                  # the research-script image
    assert not production.talk_v2v(script_engine)
    assert production.talk_v2v(_talk_image(policy, "mw-worker-talk", {"MW_TALK_ENGINE": "comfy"}))
    assert not production.talk_v2v(_talk_image(policy, "mw-worker-talkcomfy", {"MW_TALK_ENGINE": "script"}))
    pid, body = _directed_post(conn, script_engine, store, tmp_path, [WALK])
    [talk] = _segment_jobs(conn, pid)                                                # lip-synced from its photo
    assert _motion_jobs(conn, pid, script_engine) == []
    assert "video_uri" not in talk["payload"] and talk["payload"]["image_uri"] == body["images"][0]["uri"]


def test_emphasis_marks_are_never_spoken(conn, policy, store, tmp_path, joined):
    policy = _talking(policy)
    marked = _says("The samosas are *back*, and I am *not okay*.",
                   set={"location": "outside the corner shop at night"},
                   framing={"shot_size": "medium close-up", "device": "selfie"})
    assert production.unmarked("*so* good") == "so good"
    pid, body = _directed_post(conn, policy, store, tmp_path, [marked], voice=DESIGNED)
    [voice_job] = _voice_jobs(conn, pid)
    assert voice_job["payload"]["lines"] == [{"text": "The samosas are back, and I am not okay.",
                                              "delivery": "dry, quick"}]
    _spoken(conn, voice_job)
    production.advance(conn, policy, now=NOW, store=store)
    [talk] = _segment_jobs(conn, pid)
    assert "*" not in talk["payload"]["script"] and "*" not in talk["payload"]["scene"]
    assert body["talk_script"] == "The samosas are *back*, and I am *not okay*."     # kept for the captions


# --- captions timed across the joined video ---------------------------------------------------------------------
def _piece(conn, cid, kind="talk.infinitetalk", *, seconds=None, voice=None, own_audio=None):
    """A made piece of a talking video: its job (with its designed voice line, or its own Kokoro audio) and video."""
    payload = {}
    if voice is not None:
        vj = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('voice.speak', 'done', %s, %s) "
                          "RETURNING id", (cid, Jsonb({}))).fetchone()["id"]
        payload["audio_uri"] = f"s3://b/voice/{vj}.wav"
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                     "VALUES (%s, %s, 'audio', %s, 'x', 1, %s)", (vj, cid, payload["audio_uri"], Jsonb(voice)))
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES (%s, 'done', %s, %s) RETURNING id",
                       (kind, cid, Jsonb(payload))).fetchone()["id"]
    if own_audio is not None:
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                     "VALUES (%s, %s, 'audio', %s, 'x', 1, %s)", (job, cid, f"s3://b/{job}.wav",
                                                                Jsonb({"seconds": own_audio})))
    meta = {"seconds": seconds} if seconds is not None else {}
    return conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                        "VALUES (%s, %s, 'video', %s, 'x', 1, %s) RETURNING id, uri",
                        (job, cid, f"s3://b/{job}.mp4", Jsonb(meta))).fetchone()


def _joined_of(conn, cid, pieces, measured):
    return conn.execute("INSERT INTO assets (creator_id, kind, uri, sha256, bytes, meta) "
                        "VALUES (%s, 'video', %s, 'x', 1, %s) RETURNING id, uri",
                        (cid, f"s3://b/joined-{pieces[0]['id']}-{len(measured)}-{measured[0]}.mp4",
                         Jsonb({"segments": [str(p["id"]) for p in pieces], "segment_seconds": measured}))).fetchone()


def test_a_cutaway_draws_no_captions_when_the_joined_video_gets_them(conn, policy, store, tmp_path, joined):
    talking = _talking(policy)
    burned = policy_mod.from_dict({**talking.body, "burned_captions": {**talking.body["burned_captions"],
                                                                       "enabled": True}})
    pid, body = _directed_post(conn, burned, store, tmp_path, [SHOP, WINDOW, DOORWAY])
    [cut] = conn.execute("SELECT payload FROM jobs WHERE post_id = %s AND kind = 'image.narrated'", (pid,)).fetchall()
    assert cut["payload"]["captions"] is False
    pid, body = _directed_post(conn, talking, store, tmp_path, [SHOP, WINDOW, DOORWAY])   # burned captions off
    [cut] = conn.execute("SELECT payload FROM jobs WHERE post_id = %s AND kind = 'image.narrated'", (pid,)).fetchall()
    assert "captions" not in cut["payload"]


HOOK = "Right, quick one from *outside* the corner shop."
SIGN_OFF = "Anyway. Going in. Wish me luck."


def test_captions_follow_each_take_in_the_joined_video_with_clauses_and_no_fillers(conn, policy):
    cid = creator(conn)
    first = _piece(conn, cid, seconds=4.0, voice={"seconds": 3.5, "lines": [
        {"index": 0, "text": production.unmarked(HOOK), "start_s": 0.2, "end_s": 3.4}]})
    # The second line was spoken clause by clause, with a filler ("um") before the second clause.
    second = _piece(conn, cid, seconds=3.2, voice={"seconds": 3.0, "lines": [
        {"index": 0, "text": SIGN_OFF, "start_s": 0.0, "end_s": 3.0, "clauses": [0, 1, 2]}], "clauses": [
        {"index": 0, "line": 0, "text": "Anyway.", "start_s": 0.0, "end_s": 0.8, "filler": None},
        {"index": 1, "line": 0, "text": "Going in.", "start_s": 1.0, "end_s": 2.0, "filler": "um"},
        {"index": 2, "line": 0, "text": "Wish me luck.", "start_s": 2.3, "end_s": 3.0, "filler": None}],
        "fillers": [{"clause": 1, "line": 0, "word": "um"}]})
    body = {"scenes": ["a", "b"], "talk_segments": [HOOK, SIGN_OFF]}
    words = production.caption_words(conn, policy, body, _joined_of(conn, cid, [first, second], [None, None]))
    n = len(HOOK.split())
    assert [w.text for w in words] == production.unmarked(f"{HOOK} {SIGN_OFF}").split()   # no "um"
    assert words[0].start == pytest.approx(0.2) and words[n - 1].end == pytest.approx(3.4)
    assert [w.text for w in words if w.emphasis] == ["outside"]                           # *emphasis* from the script
    later = words[n:]
    assert later[0].start == pytest.approx(4.0)                                           # after the first take's 4 s
    assert later[1].text == "Going" and later[1].start > 5.0                              # the filler kept its time
    assert later[2].end <= 6.0 + 1e-6 and later[-1].end == pytest.approx(7.0)
    # The join's measured lengths win over each take's own.
    words = production.caption_words(conn, policy, body, _joined_of(conn, cid, [first, second], [4.5, None]))
    assert words[n].start == pytest.approx(4.5)


def test_a_cutaway_is_captioned_after_its_lead_and_a_kokoro_take_over_its_own_audio(conn, policy):
    from manyworlds.worker.handlers import edit_narrated
    cid = creator(conn)
    talk = _piece(conn, cid, seconds=2.5, own_audio=2.0)                                   # Kokoro on the talk worker
    cut = _piece(conn, cid, "image.narrated", seconds=3.0, voice={"seconds": 2.0, "lines": [
        {"index": 0, "text": "They still haven't cleaned that window.", "start_s": 0.0, "end_s": 2.0}]})
    body = {"scenes": ["a", "b"], "talk_segments": ["Hi there. Bye now.", "They still haven't cleaned that window."],
            "talk_modes": ["talk", "narrated"]}
    words = production.caption_words(conn, policy, body, _joined_of(conn, cid, [talk, cut], [None, None]))
    assert words[0].start == 0 and words[3].end == pytest.approx(2.0)                      # the script over its audio
    assert words[4].text == "They" and words[4].start == pytest.approx(2.5 + edit_narrated.LEAD_S)
    # A piece whose length is unknown: later captions can't be placed.
    blind = _piece(conn, cid)
    assert production.caption_words(conn, policy, body, _joined_of(conn, cid, [blind, cut], [None, None])) is None


# --- captions timed from word timestamps (burned_captions.word_timestamps) ----------------------------------------
class _AudioStore:
    """Hands back a stand-in file for any audio URI."""
    def get_file(self, uri, local):
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(b"RIFF")
        return local


def _word_timed(policy, on=True):
    return policy_mod.from_dict({**policy.body, "burned_captions": {**policy.body["burned_captions"],
                                                                    "word_timestamps": on}})


def test_word_timestamps_come_from_the_hosted_whisper_as_verbose_json(tmp_path, monkeypatch):
    from manyworlds import speech
    monkeypatch.setenv("TOGETHER_API_KEY", "test")
    wav = tmp_path / "v.wav"
    wav.write_bytes(b"RIFF")
    sent = {}

    def post(url, body, ctype, key):
        sent["body"] = body
        return {"text": "hi there", "words": [{"word": "hi", "start": 0.1, "end": 0.3},
                                              {"word": "there", "start": 0.35, "end": 0.7}, {"word": "x"}]}
    words = speech.transcribe_words(wav, {"host": "together", "model": "openai/whisper-large-v3"}, post=post)
    assert words == [{"word": "hi", "start": 0.1, "end": 0.3}, {"word": "there", "start": 0.35, "end": 0.7}]
    assert b'name="response_format"\r\n\r\nverbose_json' in sent["body"]
    assert b'name="timestamp_granularities"\r\n\r\nword' in sent["body"]
    with pytest.raises(RuntimeError):
        speech.transcribe_words(wav, {}, post=lambda *a: {"text": "hi"})          # no word timestamps
    assert speech.transcribe(wav, {}, post=lambda *a: {"text": "hi"}) == "hi"


def test_captions_use_word_timestamps_once_and_fall_back_to_estimates(conn, policy):
    cid = creator(conn)
    voice = {"seconds": 3.0, "lines": [{"index": 0, "text": "Going in. Wish me luck.", "start_s": 0.0, "end_s": 3.0}],
             "fillers": [{"clause": 1, "line": 0, "word": "um"}]}
    piece = _piece(conn, cid, seconds=3.2, voice=voice)
    body = {"scenes": ["a"], "talk_script": "Going in. Wish me *luck*."}
    heard = [{"word": "going", "start": 0.4, "end": 0.6}, {"word": "in", "start": 0.6, "end": 0.8},
             {"word": "um", "start": 1.0, "end": 1.2}, {"word": "wish", "start": 1.9, "end": 2.1},
             {"word": "me", "start": 2.1, "end": 2.2}, {"word": "luck", "start": 2.25, "end": 2.6}]
    calls = []

    def transcriber(conn_, policy_, path, seconds, *, job_id=None, creator_id=None):
        calls.append((path.exists(), seconds, creator_id))
        return heard
    on = _word_timed(policy)
    words = production.caption_words(conn, on, body, piece, store=_AudioStore(), transcriber=transcriber)
    assert [w.text for w in words] == ["Going", "in.", "Wish", "me", "luck."]          # the script's spelling, no "um"
    assert words[0].start == pytest.approx(0.4) and words[-1].end == pytest.approx(2.6)
    assert [w.text for w in words if w.emphasis] == ["luck."]
    assert calls == [(True, 3.0, cid)]
    meta = conn.execute("SELECT meta FROM assets WHERE kind = 'audio' AND creator_id = %s", (cid,)).fetchone()["meta"]
    assert meta["words"] == heard                                                       # cached on the voice asset
    production.caption_words(conn, on, body, piece, store=_AudioStore(), transcriber=transcriber)
    assert len(calls) == 1                                                              # paid once

    # Off, or a failed transcription: the estimates.
    estimated = production.caption_words(conn, policy, body, piece)
    assert estimated[0].start == pytest.approx(0.0)
    other = _piece(conn, cid, seconds=3.2, voice=voice)

    def broken(*a, **k):
        raise RuntimeError("host down")
    words = production.caption_words(conn, on, body, other, store=_AudioStore(), transcriber=broken)
    assert [w.text for w in words] == ["Going", "in.", "Wish", "me", "luck."] and words[0].start == pytest.approx(0.0)
