"""Original songs (policy `music`, OFF): the song spec and its checks (manyworlds.music), the ACE-Step 1.5 ComfyUI graph
(worker/comfy_music.py), the music.song handler's contract with its engine and separator stood in for, the lyricist
(brain/lyrics.py), the musician playbooks, generated beds in the sound library, the song under a music video's mix,
and production's music_video wiring (off by default)."""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from psycopg.types.json import Jsonb

from manyworlds import bench, direction, music, production, sound
from manyworlds import policy as policy_mod
from manyworlds.brain import archetypes, lyrics
from manyworlds.production import followups, mixing
from manyworlds.production import song as song_mod
from manyworlds.worker import comfy_music
from manyworlds.worker.handlers import music_song as ms
from manyworlds.worker.runner import NonRetryable, handlers_for

from test_production import creator

SPEC = {"title": "Window Seat", "tags": ["dreamy indie pop", "soft female vocal", "warm synths"], "bpm": 100,
        "key": "a minor", "time_signature": "4", "duration_s": 25,
        "lyrics": "[Verse]\nRain on the bus window, I draw a little sun\n[Chorus]\nWindow seat, window seat\n"
                  "(ooh) the city hums along with me"}


def _with(policy, **blocks):
    body = dict(policy.body)
    for name, over in blocks.items():
        body[name] = {**(body.get(name) or {}), **over}
    return policy_mod.from_dict(body)


# --- the shipped policy -------------------------------------------------------------------------------------------
def test_music_ships_off_with_its_kind_family_and_capabilities(raw_policy):
    assert music.enabled(raw_policy) is False
    assert "music_video" not in raw_policy.get("production", "formats")          # no music post is made
    assert raw_policy.get("sound", "library", "generated", "enabled") is False   # no generated bed is mixed
    k = raw_policy.get("job_kinds", "music.song")
    assert k["min_vram_gb"] > 0 and k["pools"] and raw_policy.family("music.song") == "music"
    assert raw_policy.get("worker_images", "music", "image") == "mw-worker-talkcomfy"   # the ComfyUI image
    assert raw_policy.get("autoscale", "scale", "default_job_seconds", "music") > 0
    ids = set(direction.capability_ids(raw_policy))
    assert {"music.original_song", "music.instrumental_bed", "actions.sing_to_camera"} <= ids
    now = set(direction.capability_ids(raw_policy, available_only=True))
    assert not {"music.original_song", "music.instrumental_bed", "actions.sing_to_camera"} & now   # off
    on = _with(raw_policy, music={"enabled": True},
               production={"formats": [*raw_policy.get("production", "formats"), "music_video"]})
    assert {"music.original_song", "actions.sing_to_camera"} <= set(direction.capability_ids(on, available_only=True))
    assert raw_policy.get("llm", "roles", "lyricist")
    # the second adapter is a slot, never built; the excluded models are not engines
    engines = raw_policy.get("music", "engines")
    assert set(engines) == {"ace_step_1_5", "heartmula"} and engines["heartmula"]["built"] is False


def test_the_handler_loads_through_the_runner():
    assert handlers_for(["music.song"])["music.song"] is ms.generate


# --- the spec and its checks ----------------------------------------------------------------------------------------
def test_a_clean_song_passes_and_normalises(policy):
    assert music.issues(SPEC, policy) == []
    s = music.normalise(SPEC, policy, max_seconds=20)
    assert s.key == "A minor" and s.duration_s == 20 and s.tags == "dreamy indie pop, soft female vocal, warm synths"
    assert music.sung_words(SPEC["lyrics"]).startswith("Rain on the bus window")
    assert "ooh" not in music.sung_words(SPEC["lyrics"]) and "[" not in music.sung_words(SPEC["lyrics"])
    p = music.song_payload(policy, SPEC, seed=7, check=True)
    assert p["engine"]["engine"] == "ace_step_1_5" and p["engine"]["diffusion_model"].endswith(".safetensors")
    assert p["check"]["max_wer"] == policy.get("music", "check", "max_wer") and p["seed"] == 7


@pytest.mark.parametrize("change, why", [
    ({"tags": ["pop in the style of Taylor Swift"]}, "real artist"),
    ({"tags": ["sounds like a famous band"]}, "someone else's"),
    ({"title": "Drake type beat"}, "real artist"),
    ({"lyrics": SPEC["lyrics"] + "\n[Bridge]\nShe said \"the quiet river keeps my name\""}, "quote"),
    ({"lyrics": SPEC["lyrics"] + "\n[Outro]\nI'm a real human, not AI"}, "deny being AI"),
    ({"lyrics": "Just one line with no tags"}, "no section tags"),
    ({"lyrics": "[Solo Rap Battle]\nla la"}, "unknown section tags"),
    ({"lyrics": SPEC["lyrics"] + "\n[Bridge]\nWe talk about the elections tonight"}, "excluded topic"),
])
def test_lyric_checks_refuse_what_is_not_original(policy, change, why):
    problems = music.issues({**SPEC, **change}, policy)
    assert any(why in p for p in problems), problems
    with pytest.raises(music.SongRefused):
        music.song_payload(policy, {**SPEC, **change}, seed=1)


def test_a_line_the_owner_listed_is_caught(policy):
    listed = _with(policy, music={"lyrics": {**policy.get("music", "lyrics"), "known_lines": ["Window seat, WINDOW seat!"]}})
    assert any("real song" in p for p in music.issues(SPEC, listed))


def test_normalise_refuses_what_the_engine_cannot_take(policy):
    for bad in ({"key": "H major"}, {"bpm": 400}, {"time_signature": "7/8"}, {"language": "xx"}, {"tags": []}):
        with pytest.raises(music.SongRefused):
            music.normalise({**SPEC, **bad}, policy)
    s = music.normalise({"tags": "lofi", "instrumental": True}, policy)
    assert s.instrumental and s.lyrics == music.INSTRUMENTAL and s.duration_s == policy.get("music", "defaults", "duration_s")
    with pytest.raises(music.SongRefused, match="variant"):
        music.engine_settings(policy, variant="nope")


# --- the graph ----------------------------------------------------------------------------------------------------
def test_the_song_graph_follows_the_comfy_template(policy):
    engine = music.engine_settings(policy)
    g = comfy_music.graph_song(tags="pop", lyrics="[Verse]\nla", seconds=25, seed=3, bpm=100, key="A minor",
                               time_signature="4", language="en", engine=engine, prefix="mw_song_j1")
    assert g["unet"]["inputs"]["unet_name"] == engine["diffusion_model"]
    assert g["clip"]["inputs"]["type"] == "ace" and g["clip"]["inputs"]["clip_name2"].endswith("qwen_1.7b_ace15.safetensors")
    text = g["text"]["inputs"]
    assert g["text"]["class_type"] == "TextEncodeAceStepAudio1.5" and text["keyscale"] == "A minor"
    assert text["duration"] == 25 and g["latent"]["inputs"]["seconds"] == 25 and text["seed"] == g["sampler"]["inputs"]["seed"] == 3
    assert g["negative"]["class_type"] == "ConditioningZeroOut" and g["sampler"]["inputs"]["negative"] == ["negative", 0]
    s = g["sampler"]["inputs"]
    assert (s["steps"], s["cfg"], s["sampler_name"], s["scheduler"]) == (8, 1.0, "euler", "simple")
    assert g["sampling"]["inputs"]["shift"] == 3.0 and g["save"]["inputs"]["filename_prefix"] == "audio/mw_song_j1"
    w = comfy_music.weights(engine)
    assert w["diffusion_models"][0] == (engine["repo"], engine["diffusion_model"]) and len(w["text_encoders"]) == 2
    with pytest.raises(ValueError):
        comfy_music.weights(music.engine_settings(policy, variant="xl"))   # no confirmed file: refused, not guessed


def test_saved_audio_finds_the_song_whatever_the_key(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy_music.comfy, "COMFY_DIR", tmp_path)
    out = comfy_music.saved_audio({"9": {"audio": [{"filename": "mw_song_00001_.flac", "subfolder": "audio",
                                                    "type": "output"}], "text": ["x"]}})
    assert out == [tmp_path / "output" / "audio" / "mw_song_00001_.flac"]


# --- the handler --------------------------------------------------------------------------------------------------
SR = 4000   # small stand-in files


@pytest.fixture
def stubbed(monkeypatch):
    """The engine, the separator, the transcriber and the audio files stood in for (no GPU, no soundfile)."""
    seen = {"engine": [], "sep": [], "asr": []}
    song = np.full((SR * 25, 2), 0.1, dtype=np.float32)

    def engine(s, ctx):
        seen["engine"].append(s)
        p = Path(ctx.workdir) / "song_raw.flac"
        p.write_bytes(b"flac")
        return p
    monkeypatch.setitem(ms.ENGINES, "ace_step_1_5", engine)
    monkeypatch.setitem(ms.SEPARATORS, "demucs", lambda mix, sr, sep: seen["sep"].append(sep) or (mix * 0.6, mix * 0.4))
    monkeypatch.setattr(ms, "read_audio", lambda path: (song, SR))
    monkeypatch.setattr(ms, "write_audio", lambda path, a, sr: Path(path).write_bytes(np.asarray(a).tobytes()) and Path(path))
    monkeypatch.setattr(ms, "loudness_lufs", lambda a, sr: -12.5)
    monkeypatch.setattr(ms, "transcribe", lambda v, sr, check: seen["asr"].append(check) or
                        "rain on the bus window i draw a little sun window seat window seat the city hums along with me")
    return seen


def _ctx(tmp_path, name="w"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(job=SimpleNamespace(id="job-1"), workdir=d, store=None)


def test_a_song_job_returns_the_mix_and_both_stems_in_a_fixed_order_with_its_check(stubbed, tmp_path, policy):
    payload = music.song_payload(policy, SPEC, seed=11, check=True)
    res = ms.generate(payload, _ctx(tmp_path))
    assert [o.meta["role"] for o in res.outputs] == list(music.ROLES)
    assert [o.path.name for o in res.outputs] == ["song_mix.wav", "song_vocals.wav", "song_instrumental.wav"]
    assert all(o.kind == "audio" and o.meta["sample_rate"] == SR and o.meta["seconds"] == 25 for o in res.outputs)
    check = res.outputs[0].meta["check"]
    assert check["passed"] and check["wer"] == 0.0 and check["lufs"] == -12.5 and check["duration_error_share"] == 0
    assert res.outputs[1].meta["check"] == {"passed": True}
    assert stubbed["engine"][0]["seed"] == 11 and stubbed["engine"][0]["key"] == "A minor"
    assert res.recipe["licence"] == "MIT" and res.recipe["lyrics"] == SPEC["lyrics"]
    assert res.meta["output_seconds"] == 25 and "separation_seconds" in res.meta


def test_a_rerun_makes_the_same_files_and_recipe(stubbed, tmp_path, policy):
    payload = music.song_payload(policy, SPEC, seed=11, check=True)
    a, b = ms.generate(payload, _ctx(tmp_path, "a")), ms.generate(payload, _ctx(tmp_path, "b"))
    assert [o.path.name for o in a.outputs] == [o.path.name for o in b.outputs]
    assert [o.path.read_bytes() for o in a.outputs] == [o.path.read_bytes() for o in b.outputs]
    assert a.recipe == b.recipe and stubbed["engine"][0] == stubbed["engine"][1]


def test_an_instrumental_is_its_own_bed_with_a_catalogue_entry_and_no_vocal_stem(stubbed, tmp_path, policy):
    payload = music.song_payload(policy, {"title": "Quiet Desk", "tags": "lo-fi, felt piano, no vocals",
                                          "instrumental": True, "duration_s": 25}, seed=2, check=True)
    res = ms.generate(payload, _ctx(tmp_path))
    assert [o.meta["role"] for o in res.outputs] == ["mix", "instrumental"] and not stubbed["sep"]
    entry = res.outputs[1].meta["library_entry"]
    assert entry["licence"] == "own-generated" and entry["kind"] == "music" and "seed 2" in entry["source"]
    assert res.outputs[0].meta["check"]["wer"] is None and not stubbed["asr"]
    assert sound.SoundLibrary([{**entry, "uri": "s3://b/bed.wav"}],
                              allowed_licences=policy.get("sound", "library", "allowed_licences")).tags()


def test_a_song_that_misses_its_limits_fails_its_check(stubbed, tmp_path, policy, monkeypatch):
    monkeypatch.setattr(ms, "transcribe", lambda v, sr, check: "something else entirely")
    monkeypatch.setattr(ms, "loudness_lufs", lambda a, sr: -30.0)
    payload = music.song_payload(policy, {**SPEC, "duration_s": 60}, seed=1, check=True)
    check = ms.generate(payload, _ctx(tmp_path)).outputs[0].meta["check"]
    assert not check["passed"] and set(check["failed"]) == {"lyrics", "loudness", "duration"}


def test_bad_payloads_and_unbuilt_engines_never_retry(stubbed, tmp_path, policy):
    good = music.song_payload(policy, SPEC, seed=1)
    for bad in ({"engine": {}}, {"tags": ""}, {"duration_s": 0}, {"key": "H major"}, {"seed": None},
                {"engine": {**good["engine"], "engine": "nope"}}, {"engine": {**good["engine"], "engine": "heartmula"}},
                {"separation": {"engine": "spleeter"}}):
        with pytest.raises(NonRetryable):
            ms.generate({**good, **bad}, _ctx(tmp_path))


def test_the_ace_engine_downloads_its_files_once_and_runs_the_graph(tmp_path, policy, monkeypatch):
    from manyworlds.worker import comfy
    calls = []
    monkeypatch.setattr(comfy, "write_model_paths", lambda d: calls.append("paths"))
    monkeypatch.setattr(comfy, "ensure_weights", lambda d, w, prune=False: calls.append(("weights", w)))
    monkeypatch.setattr(comfy, "start", lambda: calls.append("start"))
    monkeypatch.setattr(ms, "_state", {"weights": set()})
    out = tmp_path / "comfy_out.flac"

    def run(g, timeout_s):
        calls.append(("run", g["text"]["inputs"]["tags"], g["save"]["inputs"]["filename_prefix"]))
        out.write_bytes(b"flac")
        return out
    monkeypatch.setattr(comfy_music, "run_audio", run)
    song = ms.validate(music.song_payload(policy, SPEC, seed=4))
    for _ in range(2):
        path = ms._ace_step(song, _ctx(tmp_path))
        assert path.name == "song_raw.flac" and path.exists()
    assert [c for c in calls if c == "paths"] == ["paths"]                   # weights checked once per worker
    assert calls.count("start") == 2 and calls[-1] == ("run", song["tags"], "audio/mw_song_job-1")


# --- lyrics -------------------------------------------------------------------------------------------------------
class LyricistStub:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    @contextlib.contextmanager
    def tagged(self, agent, run_id, before_call=None):
        yield self

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        self.calls.append({"role": role, "system": system, "content": content})
        return schema(**self.answers.pop(0))


DRAFT = {"title": "Window Seat", "lyrics": SPEC["lyrics"], "tags": SPEC["tags"], "bpm": 100, "key": "A minor",
         "time_signature": "4", "duration_s": 25}


def test_the_lyricist_rewrites_a_refused_draft_once_with_its_problems(conn, policy):
    cid = creator(conn)
    llm = LyricistStub([{**DRAFT, "tags": ["pop like Billie Eilish"]}, DRAFT])
    spec = lyrics.write_song(conn, policy, llm, creator_id=cid, idea={"concept": "a rainy commute"})
    assert spec["title"] == "Window Seat" and spec["key"] == "A minor" and len(llm.calls) == 2
    assert all(c["role"] == "lyricist" for c in llm.calls)
    assert "billie eilish" in llm.calls[1]["content"].lower() and "refused" in llm.calls[1]["content"]
    assert "never name a real artist" in llm.calls[0]["system"].lower()
    bad = LyricistStub([{**DRAFT, "title": "in the style of someone"}] * 2)
    with pytest.raises(music.SongRefused):
        lyrics.write_song(conn, policy, bad, creator_id=cid, idea={})


def test_songs_are_written_only_for_sung_posts_while_music_is_on(conn, policy):
    cid = creator(conn)
    pid = conn.execute("INSERT INTO posts (creator_id, format, status) VALUES (%s, 'music_video', 'planned') "
                       "RETURNING id", (cid,)).fetchone()["id"]
    llm = LyricistStub([DRAFT])
    posts = [(pid, "music_video", {"concept": "rain"}), (pid, "photo", {})]
    assert lyrics.write_songs(conn, policy, llm, cid, posts) == 0 and not llm.calls        # off as shipped
    on = _with(policy, music={"enabled": True})
    assert lyrics.write_songs(conn, on, llm, cid, posts) == 1 and len(llm.calls) == 1
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    assert body["song"]["title"] == "Window Seat" and body["song"]["lyrics"] == SPEC["lyrics"]


# --- the playbook -------------------------------------------------------------------------------------------------
def test_the_musician_playbooks_load_beside_the_main_library(policy):
    names = archetypes.names(policy)
    assert names[:2] == ["fashion_style", "beauty_grwm"]                   # the main file first, as before
    assert {"singer_songwriter", "bedroom_producer"} <= set(names)
    play = archetypes.playbook(policy, "singer_songwriter")
    assert "music.original_song" in play["lean_on"] and any("weekly song drop" == s["name"]
                                                            for s in play["series_patterns"])
    assert abs(sum(play["mix"].values()) - 1) < 1e-6
    assert archetypes.for_creator(policy, {"niche": "singer-songwriter, original songs"})[:1] == ["singer_songwriter"]
    assert archetypes.unknown_capabilities(policy) == []
    assert "+musician-v1" in archetypes.library(policy)["version"]


# --- generated beds -----------------------------------------------------------------------------------------------
def _bed(conn, cid, *, passed, instrumental=True):
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id) VALUES ('music.song', 'done', %s) RETURNING id",
                       (cid,)).fetchone()["id"]
    entry = {"tag": "quiet desk", "kind": "music", "formats": [], "licence": "own-generated",
             "source": "ace_step_1_5 turbo seed 2: lo-fi", "credit": ""}
    conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                 "VALUES (%s, %s, 'audio', %s, 'x', 1, %s)",
                 (job, cid, f"s3://b/{job}/01.wav", Jsonb({"role": "instrumental", "instrumental": instrumental,
                                                           "library_entry": entry, "check": {"passed": passed}})))
    return job


def test_generated_beds_join_the_library_only_when_switched_on_and_checked(conn, policy):
    cid = creator(conn)
    good = _bed(conn, cid, passed=True)
    _bed(conn, cid, passed=False)
    _bed(conn, cid, passed=True, instrumental=False)        # a sung song's instrumental stem is not a bed
    assert music.generated_beds(conn, policy) == [] and music.with_generated_beds(conn, policy) is policy
    on = _with(policy, sound={"library": {**policy.get("sound", "library"),
                                          "generated": {**policy.get("sound", "library", "generated"), "enabled": True}}})
    beds = music.generated_beds(conn, on)
    assert [b["uri"] for b in beds] == [f"s3://b/{good}/01.wav"]
    lib = sound.library(music.with_generated_beds(conn, on))
    assert lib.lookup("quiet desk").licence == "own-generated" and not lib.refused


# --- the mix -----------------------------------------------------------------------------------------------------
def test_a_music_video_ships_the_full_mix_in_place_of_its_vocal_stem(policy):
    body = {"song": SPEC, "song_media": {"mix": {"asset_id": "a1", "uri": "s3://b/song/00.wav", "seconds": 25},
                                         "vocals": {"asset_id": "a2", "uri": "s3://b/song/01.wav", "seconds": 25}}}
    plan = mixing._sound_plan_from(policy, {"id": "p1", "format": "music_video"}, body, lambda: ({}, []))
    assert plan["song"].uri == "s3://b/song/00.wav" and plan["song"].licence == "own-generated"
    assert plan["room_tone_synth"] is None and plan["music"] is None and not plan["effects"]
    off = _with(policy, sound={"enabled": False})        # with sound off a vocal stem alone still never ships
    assert mixing._sound_plan_from(off, {"id": "p1", "format": "music_video"}, body, lambda: ({}, []))["song"]
    talk = mixing._sound_plan_from(policy, {"id": "p1", "format": "talking_video"}, body, lambda: ({}, []))
    assert "song" not in talk                            # only sung formats
    p = sound.plan(has_voice=True, duration_s=24.5, song="song.wav", policy=policy)
    assert p.inputs == [["-i", "song.wav"]]
    assert p.graph == f"[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,atrim=end=24.5[mix]"


# --- production wiring -------------------------------------------------------------------------------------------
def _post(conn, fmt):
    cid = creator(conn)
    return dict(conn.execute("INSERT INTO posts (creator_id, format) VALUES (%s, %s) RETURNING *",
                             (cid, fmt)).fetchone())


def test_music_video_is_a_follow_up_that_production_does_not_make_yet(policy):
    assert "music_video" in production.FOLLOW_UPS and "music_video" in production.FOLLOW_UP_FORMATS
    assert "music_video" not in policy.get("production", "formats")
    size = production.scene_size(policy, "music_video")
    assert size["height"] > size["width"]                                   # photographed 9:16 like any video


def test_a_music_video_makes_its_song_then_sings_its_vocal_stem(conn, policy):
    post = _post(conn, "music_video")
    body = {"scenes": ["her bedroom, a guitar on the bed"], "song": SPEC}
    chosen = [{"id": "x", "uri": "s3://b/photo.png"}]
    state, _ = followups._advance_follow_up(conn, policy, post, body, chosen)
    assert state == "waiting" and len(body["song_jobs"]) == 1 and not body.get("talk_jobs")
    job = conn.execute("SELECT kind, payload, idempotency_key FROM jobs WHERE id = %s", (body["song_jobs"][0],)).fetchone()
    assert job["kind"] == "music.song" and job["payload"]["check"] and job["payload"]["lyrics"] == SPEC["lyrics"]
    assert job["payload"]["duration_s"] == min(SPEC["duration_s"], policy.get("music", "video", "max_seconds"))
    assert job["idempotency_key"] == f"post:{post['id']}:song:0"
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (body["song_jobs"][0],))
    for i, role in enumerate(music.ROLES):
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                     "VALUES (%s, %s, 'audio', %s, 'x', 1, %s)",
                     (body["song_jobs"][0], post["creator_id"], f"s3://b/song/{i:02d}.wav",
                      Jsonb({"role": role, "seconds": 25, "check": {"passed": True}})))
    state, _ = followups._advance_follow_up(conn, policy, post, body, chosen)
    assert state == "waiting" and body["song_media"]["vocals"]["uri"] == "s3://b/song/01.wav"
    talk = conn.execute("SELECT kind, payload FROM jobs WHERE id = %s", (body["talk_jobs"][0],)).fetchone()
    assert talk["kind"] == policy.get("production", "talk_kind")
    assert talk["payload"]["audio_uri"] == "s3://b/song/01.wav"            # lip-synced to the vocal stem
    assert "Singing to camera" in talk["payload"]["scene"] and talk["payload"]["script"].startswith("Rain on the bus")


def test_a_song_brief_that_breaks_the_rules_drops_the_post_before_any_gpu(conn, policy):
    post = _post(conn, "music_video")
    body = {"scenes": ["a stage"], "song": {**SPEC, "tags": ["in the style of a famous singer"]}}
    state, _ = followups._advance_follow_up(conn, policy, post, body, [{"id": "x", "uri": "s3://b/p.png"}])
    assert state == "dropped" and "can't be made" in body["drop_reason"]
    assert conn.execute("SELECT count(*) n FROM jobs WHERE kind = 'music.song'").fetchone()["n"] == 0
    nosong = {"scenes": ["a stage"]}
    assert song_mod.advance_song(conn, policy, post, nosong)[0] == "dropped"


def test_an_enforced_check_remakes_a_failed_song(conn, policy):
    strict = _with(policy, music={"check": {**policy.get("music", "check"), "enforce": True}})
    post = _post(conn, "music_video")
    body = {"scenes": ["a stage"], "song": SPEC}
    song_mod.advance_song(conn, strict, post, body)
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (body["song_jobs"][0],))
    conn.execute("INSERT INTO assets (job_id, kind, uri, sha256, bytes, meta) VALUES (%s, 'audio', 's3://b/m.wav', "
                 "'x', 1, %s)", (body["song_jobs"][0], Jsonb({"role": "mix", "check": {"passed": False}})))
    state, _ = song_mod.advance_song(conn, strict, post, body)
    assert state == "waiting" and len(body["song_jobs"]) == 2 and body["song_repairs"] == 1
    seeds = [conn.execute("SELECT payload->>'seed' s FROM jobs WHERE id = %s", (j,)).fetchone()["s"]
             for j in body["song_jobs"]]
    assert seeds[0] != seeds[1]


# --- the bench ----------------------------------------------------------------------------------------------------
def test_the_bench_queues_original_songs_with_their_check(conn, policy):
    batch = bench.enqueue(conn, policy, "music.song", 3, extra={"duration_s": 20})
    rows = conn.execute("SELECT payload FROM jobs WHERE batch = %s ORDER BY payload->>'seed'", (batch,)).fetchall()
    assert len(rows) == 3 and all(r["payload"]["check"] and r["payload"]["duration_s"] == 20 for r in rows)
    assert [r["payload"]["instrumental"] for r in rows] == [False, False, True]
    assert all(r["payload"]["engine"]["engine"] == "ace_step_1_5" for r in rows)
    for s in bench.SONGS:                                   # the benchmark's own songs pass the lyric checks
        assert music.issues(s, policy) == []


def test_the_bench_report_counts_songs_and_summarises_their_checks(conn, policy):
    batch = bench.enqueue(conn, policy, "music.song", 1)
    job = conn.execute("SELECT id FROM jobs WHERE batch = %s", (batch,)).fetchone()["id"]
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job,))
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, gpu_type, gpu_seconds, outcome) "
                 "VALUES (%s, 'w', 'vast_4090', 'rtx4090', 60, 'done')", (job,))
    for i, role in enumerate(music.ROLES):
        meta = {"role": role, "seconds": 30}
        if role == "mix":
            meta["check"] = {"passed": True, "wer": 0.2, "duration_error_share": 0.01, "lufs": -11.0, "failed": []}
        conn.execute("INSERT INTO assets (job_id, kind, uri, sha256, bytes, meta) VALUES (%s, 'audio', %s, 'x', 1, %s)",
                     (job, f"s3://b/{i}.wav", Jsonb(meta)))
    r = bench.report(conn, policy, batch)
    assert r["measured"][0]["outputs"] == 1 and r["measured"][0]["sec_per_output_median"] == 60
    mc = r["music_checks"]
    assert mc["songs"] == 1 and mc["passed"] == 1 and mc["wer_median"] == 0.2 and mc["lufs_median"] == -11.0


LYRICS = ("[Verse]\nThe kettle sings before I do\nSteam on the window, writing you\n"
          "[Chorus]\nStay up, stay up, the night is ours\nCounting the streetlights instead of stars\n"
          "[Verse]\nSocks on the tiles, a borrowed sweater\nEvery small hour feels a little better")


def test_a_song_that_sings_every_line_and_repeats_its_chorus_covers_its_lyrics():
    # 2026-11-23, the first real song: every line sung, the chorus repeated; WER scored it 6.0
    heard = ("The kettle sing before I do. Steam on the window. You. Stay up, stay up. The night is size. Counting the "
             "streetlights instead of stars Socks on the tiles, a borrowed sweater Every small hour feels a little "
             "better. Stay up, stay up, the night is ours. Counting the streetlights instead of stars. Stay up, stay up.")
    assert music.line_coverage(LYRICS, heard, min_line_ratio=0.7) == 1.0
    assert music.line_coverage(LYRICS, "la la la, oh oh, yeah", min_line_ratio=0.7) == 0.0
    assert music.line_coverage("[Instrumental]", heard, min_line_ratio=0.7) is None


def test_with_a_coverage_floor_the_lyrics_verdict_ignores_wer():
    from manyworlds.worker.handlers import music_song
    check = {"max_wer": 0.35, "min_line_coverage": 0.7}
    ok = music_song.judge(check, asked_s=120, seconds=120, lufs=None, lyric_wer=6.0, coverage=1.0)
    bad = music_song.judge(check, asked_s=120, seconds=120, lufs=None, lyric_wer=0.1, coverage=0.4)
    assert ok["passed"] and ok["line_coverage"] == 1.0 and bad["failed"] == ["lyrics"]
    assert music_song.judge({"max_wer": 0.35}, asked_s=1, seconds=1, lufs=None, lyric_wer=6.0)["failed"] == ["lyrics"]
