"""Wave 2: production composes the capabilities the director asks for: 9:16 scene photos for every video format,
negatives in the video payloads, Wan-Animate-2 for a shot with a motion reference, transitions at shot changes,
the sound mix on every finished video, one text script (captions and on-screen text) in the finish pass, the AI
upscaler hook."""

from __future__ import annotations

from pathlib import Path

import pytest
from psycopg.types.json import Jsonb

from manyworlds import finishing, media, production, sound
from manyworlds import policy as policy_mod
from manyworlds.creator import lab

from test_production import creator


def _with(policy, **blocks):
    """The policy with these top-level blocks merged over the shipped ones (one level deep)."""
    body = dict(policy.body)
    for name, over in blocks.items():
        body[name] = {**(body.get(name) or {}), **over}
    return policy_mod.from_dict(body)


def _post(conn, fmt):
    cid = creator(conn)
    return dict(conn.execute("INSERT INTO posts (creator_id, format) VALUES (%s, %s) RETURNING *",
                             (cid, fmt)).fetchone())


def _bible(conn, post):
    return conn.execute("SELECT bible FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["bible"]


def _payload(conn, job_id):
    return conn.execute("SELECT kind, payload FROM jobs WHERE id = %s", (job_id,)).fetchone()


CAFE = {"set": {"location": "a small cafe by the window"},
        "blocking": [{"at_s": 0, "action": "lifts the cup and smiles", "eyeline": "lens"}]}


# --- scene photos ----------------------------------------------------------------------------------------------
def test_every_video_format_photographs_9x16(policy):
    video = policy.get("generation", "image_by_format", "video")
    for fmt in policy.get("planner", "video_formats"):
        size = production.scene_size(policy, fmt)
        assert (size["width"], size["height"]) == (video["width"], video["height"])
    # A video format the image table doesn't name (a new archetype) falls back to the video entry.
    grwm = _with(policy, planner={"video_formats": [*policy.get("planner", "video_formats"), "grwm"]})
    size = production.scene_size(grwm, "grwm")
    assert (size["width"], size["height"]) == (video["width"], video["height"]) and size["height"] > size["width"]
    photo = production.scene_size(policy, "photo")
    assert (photo["width"], photo["height"]) == (policy.get("generation", "image", "width"),
                                                policy.get("generation", "image", "height"))


# --- negatives --------------------------------------------------------------------------------------------------
def test_clip_talk_and_motion_payloads_carry_the_compiled_negative(conn, policy):
    post = _post(conn, "story_clip")
    body = {"scenes": ["a cafe"], "script": "steam rises", "shots": [CAFE]}
    p = _payload(conn, production._clip_job(conn, policy, post, body, [{"uri": "s3://b/a.png"}], 0))["payload"]
    want = lab.prompts("i2v", _bible(conn, post), "a cafe", motion="steam rises", shot=CAFE, policy=policy)
    assert want["negative"] and p["negative_prompt"] == want["negative"] and p["prompt"] == want["prompt"]
    assert want["negative"] not in p["prompt"]                                          # never in the positive

    talk = _post(conn, "talking_video")
    tbody = {"scenes": ["a cafe"], "script": "Hi.", "shots": [CAFE]}
    p = _payload(conn, production._talk_job(conn, policy, talk, tbody, [{"uri": "s3://b/a.png"}], 0))["payload"]
    assert p["negative_prompt"] == lab.prompts("talk", _bible(conn, talk), "a cafe", shot=CAFE,
                                               policy=policy)["negative"]
    p = _payload(conn, production._motion_job(conn, policy, talk, tbody, [{"uri": "s3://b/a.png"}], 0))["payload"]
    assert p["negative_prompt"] and p["intermediate"] is True

    # Undirected: no negative from us (the worker keeps its own default).
    plain = _payload(conn, production._clip_job(conn, policy, _post(conn, "story_clip"), {"scenes": ["a cafe"]},
                                                [{"uri": "s3://b/a.png"}], 0))["payload"]
    assert "negative_prompt" not in plain


# --- dance / motion transfer ------------------------------------------------------------------------------------
def _ref(policy):
    return policy.get("dance", "references")[0]["id"]


def test_a_shot_with_a_motion_reference_is_animated_only_while_dance_is_on(conn, policy):
    on = _with(policy, dance={"enabled": True})
    shot = {**CAFE, "motion_reference": _ref(on)}
    post = _post(conn, "story_clip")
    body = {"scenes": ["a cafe"], "script": "", "shots": [shot]}
    row = _payload(conn, production._clip_job(conn, on, post, body, [{"uri": "s3://b/a.png"}], 0))
    assert row["kind"] == on.get("production", "dance_kind")
    p = row["payload"]
    assert p["reference"] == _ref(on) and p["image_uri"] == "s3://b/a.png"
    assert p["keyframes"] == on.get("dance", "identity", "keyframes")                   # the dance identity row's
    assert {k: v for k, v in p["params"].items() if k != "negative_prompt"} == on.get("dance", "generation")
    assert p["shot"]["motion_reference"] == _ref(on)
    off = _with(policy, dance={"enabled": False})
    row = _payload(conn, production._clip_job(conn, off, _post(conn, "story_clip"), body, [{"uri": "s3://b/a.png"}], 0))
    assert row["kind"] == off.get("production", "video_kind") and "reference" not in row["payload"]
    # An id that is not in dance.references is never sent.
    unknown = {"scenes": ["a cafe"], "shots": [{**CAFE, "motion_reference": "a-trend-video"}]}
    row = _payload(conn, production._clip_job(conn, on, _post(conn, "story_clip"), unknown,
                                              [{"uri": "s3://b/a.png"}], 0))
    assert "reference" not in row["payload"]


def test_a_talking_take_with_a_motion_reference_lip_syncs_onto_its_animate_clip(conn, policy):
    on = _with(policy, dance={"enabled": True})
    shot = {**CAFE, "motion_reference": _ref(on), "performance": {"lines": [{"text": "Quick one, watch this."}]}}
    post = _post(conn, "talking_video")
    body = {"scenes": ["a cafe"], "shots": [shot], "talk_shots": [0], "talk_segments": [],
            "talk_script": "Quick one, watch this."}
    if not production.talk_v2v(on):
        pytest.skip("the shipped talk engine has no video-to-video lip sync")
    assert production._needs_motion(on, post, body, None)
    assert not production._needs_motion(_with(policy, dance={"enabled": False}), post, body, None)
    row = _payload(conn, production._motion_job(conn, on, post, body, [{"uri": "s3://b/a.png"}], 0))
    assert row["kind"] == on.get("production", "dance_kind")
    assert row["payload"]["reference"] == _ref(on) and row["payload"]["intermediate"] is True


# --- joins --------------------------------------------------------------------------------------------------------
def test_each_shot_change_gets_its_transition_and_a_split_take_a_cut(policy):
    shots = [{"set": {"location": "a"}}, {"set": {"location": "b"}, "edit": {"transition_in": "whip pan to her"}},
             {"set": {"location": "c"}}]
    body = {"shots": shots, "talk_shots": [0, 1, 1, 2]}                 # shot 1's take was split for length
    assert production.join_transitions(policy, body, 4) == ["whip", "cut", "cut"]
    assert production.join_transitions(policy, {"shots": shots[:1] + shots[2:], "talk_shots": [0, 1]}, 2) is None
    assert production.join_transitions(policy, {"talk_segments": ["a", "b"]}, 2) is None   # undirected: plain join


# --- the finish stage ---------------------------------------------------------------------------------------------
class _Sounds:
    """The object store, plus stand-in files for library sounds."""
    def __init__(self, store):
        self.store = store

    def __getattr__(self, name):
        return getattr(self.store, name)

    def get_file(self, uri, local):
        if uri.startswith("s3://b/sfx/"):
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(b"RIFF")
            return local
        return self.store.get_file(uri, local)


@pytest.fixture()
def fake_finish(monkeypatch):
    calls = []

    def fake(video, out, *, look, grain=None, ass=None, hook_ass=None, has_voice=True, policy, timeout=None):
        script = ass or hook_ass
        calls.append(("finish", {"ass": script and Path(script).read_text(encoding="utf-8"), "has_voice": has_voice}))
        Path(out).write_bytes(b"finished")
        return out

    def fake_mix(video, out, **kw):
        calls.append(("mix", kw))
        Path(out).write_bytes(b"mixed")
        return Path(out)

    def no_burn(*a, **k):
        raise AssertionError("a second text pass")
    monkeypatch.setattr(finishing, "finish", fake)
    monkeypatch.setattr(sound, "mix", fake_mix)
    monkeypatch.setattr(media, "has_libass", lambda **k: True)
    monkeypatch.setattr(media, "probe", lambda p, **k: {"seconds": 3.0, "width": 1080, "height": 1920, "audio": True})
    from manyworlds import captions, overlays
    monkeypatch.setattr(captions, "burn", no_burn)
    monkeypatch.setattr(overlays, "burn", no_burn)
    return calls


def _finishing(policy, **sound_blocks):
    return _with(policy, finishing={"enabled": True}, burned_captions={"enabled": True},
                 sound={"enabled": True, **sound_blocks})


def _made(conn, store, tmp_path, fmt, kind, body):
    cid = creator(conn)
    src = tmp_path / f"raw-{fmt}.mp4"
    src.write_bytes(b"mp4")
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES (%s, 'done', %s, %s) RETURNING id",
                       (kind, cid, Jsonb({}))).fetchone()["id"]
    video = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                         "VALUES (%s, %s, 'video', %s, 'x', 1, %s) RETURNING id, uri",
                         (job, cid, store.put_file(src, f"v/{job}.mp4").uri, Jsonb({"seconds": 3.0}))).fetchone()
    pid = conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, %s, 'producing', %s) "
                       "RETURNING id", (cid, fmt, Jsonb(body))).fetchone()["id"]
    return {"id": pid, "creator_id": cid, "format": fmt, "body": body}, video


LABELLED = {"edit": {"hook": "she lifts the lid", "on_screen_text": [{"kind": "hook", "text": "Wait for the end"},
                                                                     {"kind": "label", "text": "Soup day"}]}}


def test_captions_labels_and_the_hook_ride_in_one_script_in_one_pass(conn, policy, store, tmp_path, fake_finish):
    p = _finishing(policy)
    body = {"scenes": ["the kitchen"], "shots": [LABELLED], "script": "Thursday soup is back."}
    post, video = _made(conn, store, tmp_path, "narrated_edit", "image.narrated", body)
    final = production.finish_video(conn, p, post, body, video, store)
    finishes = [c for name, c in fake_finish if name == "finish"]
    assert len(finishes) == 1                                                   # one pass, one script
    ass = finishes[0]["ass"]
    assert "Thursday" in ass and "Soup day" in ass and "Wait for the end" in ass
    assert "she lifts the lid" not in ass                                # edit.hook is a note, never drawn
    assert finishes[0]["has_voice"] is True
    assert {o["kind"] for o in final["meta"]["on_screen_text"]} == {"label", "hook"}
    assert final["meta"]["captions"] == p.get("burned_captions", "default_style")


def test_every_finished_video_is_mixed_with_room_tone_and_a_silent_one_gets_its_bed(conn, policy, store, tmp_path,
                                                                                   fake_finish):
    bed = {"tag": "quiet bed", "uri": "s3://b/sfx/bed.wav", "licence": "CC0-1.0", "source": "us", "kind": "music",
           "formats": ["story_clip"]}
    p = _finishing(policy, library={"entries": [bed]}, silent_beds={"formats": ["story_clip"], "music": True})
    # A talking format with no library cues: the synthesised room tone, the polish room, no music.
    body = {"scenes": ["the kitchen"], "shots": [{"set": {"location": "her kitchen"}}], "script": "Kettle's on."}
    post, video = _made(conn, store, tmp_path, "narrated_edit", "image.narrated", body)
    production.finish_video(conn, p, post, body, video, _Sounds(store))
    [mix] = [kw for name, kw in fake_finish if name == "mix"]
    want = sound.design(body["shots"], seed_key=str(post["id"]), policy=p)
    assert mix["room_tone_synth"] == want.room_tone and mix["room_tone_synth"] is not None
    assert mix["room_size"] == want.room_size and mix["room_tone"] is None and mix["music"] is None
    # A silent story clip: the same bed, plus the format's music bed (no voice: levelled as a bed).
    fake_finish.clear()
    body = {"scenes": ["a rooftop"], "shots": [{"set": {"location": "a rooftop"}}]}
    post, video = _made(conn, store, tmp_path, "story_clip", "comfy.wan_i2v", body)
    final = production.finish_video(conn, p, post, body, video, _Sounds(store))
    [mix] = [kw for name, kw in fake_finish if name == "mix"]
    assert mix["music"] is not None and mix["room_tone_synth"] is not None
    assert [c["has_voice"] for name, c in fake_finish if name == "finish"] == [False]
    assert [s["tag"] for s in final["meta"]["sound"]] == ["quiet bed"]


def test_a_mix_that_fails_ships_the_video_unmixed_and_says_so(conn, policy, store, tmp_path, fake_finish,
                                                             monkeypatch):
    def broken(video, out, **kw):
        raise RuntimeError("ffmpeg could not mix")
    monkeypatch.setattr(sound, "mix", broken)
    body = {"scenes": ["the kitchen"], "script": "Kettle's on."}
    post, video = _made(conn, store, tmp_path, "narrated_edit", "image.narrated", body)
    final = production.finish_video(conn, _finishing(policy), post, body, video, store)
    assert "could not mix" in final["meta"]["sound_error"] and final["meta"]["passes"] == 1


def test_word_timestamps_follow_the_format(policy):
    from manyworlds import captions
    shipped = policy.get("burned_captions", "word_timestamps_formats")
    for fmt in shipped:
        assert captions.word_timestamps_for(policy, fmt)
    assert not captions.word_timestamps_for(_with(policy, burned_captions={"word_timestamps": False,
                                                                           "word_timestamps_formats": []}),
                                            "talking_video")


def test_the_final_review_is_asked_once_with_the_joins_and_the_text(conn, policy, store, tmp_path, fake_finish,
                                                                   monkeypatch):
    from manyworlds.review import final as final_review
    seen = {}

    def fake_review(conn_, policy_, llm, post, path, *, joins=None, text_spans=None):
        seen.update(llm=llm, exists=Path(path).exists(), joins=joins, spans=text_spans)
        post["body"]["final_review"] = {"passed": True}
        return {"passed": True}
    monkeypatch.setattr(final_review, "review_final", fake_review)
    body = {"scenes": ["the kitchen"], "shots": [LABELLED], "script": "Thursday soup is back."}
    post, video = _made(conn, store, tmp_path, "narrated_edit", "image.narrated", body)
    production.finish_video(conn, _finishing(policy), post, body, video, store, llm="judge")
    assert seen["llm"] == "judge" and seen["exists"] and seen["joins"] == []
    kinds = {s["kind"] for s in seen["spans"]}
    assert {"caption", "label", "hook"} <= kinds and body["final_review"] == {"passed": True}


# --- the AI upscaler hook -----------------------------------------------------------------------------------------
def test_the_upscaler_runs_before_the_finish_only_when_it_is_on(conn, policy):
    post = _post(conn, "story_clip")
    video = {"id": None, "uri": "s3://b/raw.mp4"}
    p = _with(policy, finishing={"enabled": True})
    assert production._advance_upscale(conn, p, post, {}, video) == ("chosen", None)          # shipped: off
    on = _with(policy, finishing={"enabled": True, "upscaler": {"enabled": True, "kind": "video.upscale",
                                                                 "model": "m", "scale": 2}})
    body: dict = {}
    assert production._advance_upscale(conn, on, post, body, video) == ("waiting", None)
    [job] = body["upscale_jobs"]
    row = _payload(conn, job)
    assert row["kind"] == "video.upscale" and row["payload"]["video_uri"] == "s3://b/raw.mp4"
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job,))
    up = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'video', "
                      "'s3://b/up.mp4', 'x', 1) RETURNING id, uri", (job, post["creator_id"])).fetchone()
    state, got = production._advance_upscale(conn, on, post, body, video)
    assert state == "chosen" and got["uri"] == up["uri"]
