"""One singing voice per artist (policy music.voice, OFF): the payloads and the licence rule (manyworlds.singing_voice),
the similarity maths on synthetic embeddings, the music.convert_voice / music.train_voice handlers' contracts with
their converters, embedder and audio files stood in for, the voice creation flow (creator/music_voice.py), the
production wiring (off by default) and the voice-consistency bench."""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from psycopg.types.json import Jsonb

from manyworlds import bench, music
from manyworlds import policy as policy_mod
from manyworlds import singing_voice as sv
from manyworlds.creator import music_voice as lab
from manyworlds.production import followups
from manyworlds.production import song as song_mod
from manyworlds.worker.handlers import music_voice as mv
from manyworlds.worker.runner import NonRetryable, OutputFile, handlers_for

from test_music import SPEC
from test_production import creator

SR = 4000


def _with_voice(policy, **over):
    body = dict(policy.body)
    m = dict(body["music"])
    m["voice"] = {**m["voice"], **over}
    body["music"] = m
    return policy_mod.from_dict(body)


ASSET = {"version": 1, "adapter": "soulx_svc", "centroid": [1.0, 0.0, 0.0], "embedding_model": "ecapa",
         "prompt_uri": "s3://b/prompt.wav", "model_uri": None, "index_uri": None}


# --- the shipped policy ---------------------------------------------------------------------------------------------

def test_the_voice_ships_off_report_only_with_rvc_refused(raw_policy):
    v = raw_policy.get("music", "voice")
    assert v["enabled"] is False and v["allow_unverified_licences"] is False
    assert v["gate"]["enforce"] is False and v["gate"]["min_similarity"] is None
    assert v["adapters"]["soulx_svc"]["licence_verified"] is True
    assert v["adapters"]["rvc"]["licence_verified"] is False          # auxiliary weights' licences unverified
    for a in v["adapters"].values():                                  # pinned to a commit
        assert len(a["commit"]) == 40
    assert v["adapters"]["soulx_svc"]["auto_shift"] is False and v["adapters"]["soulx_svc"]["pitch_shift"] == 0
    assert v["adapters"]["rvc"]["infer"]["pitch"] == 0                # the source pitch is kept
    assert v["similarity"]["wavlm"]["enabled"] is False
    for kind in ("music.convert_voice", "music.train_voice"):
        assert raw_policy.family(kind) == "music" and raw_policy.get("job_kinds", kind, "pools")
    assert raw_policy.get("gpu_runs", "max_attempt_minutes_by_kind", "music.train_voice") > 20


def test_the_handlers_load_through_the_runner():
    found = handlers_for(["music.convert_voice", "music.train_voice", "music.song"])
    assert found["music.convert_voice"] is mv.convert and found["music.train_voice"] is mv.train


# --- licences ---------------------------------------------------------------------------------------------------------

def test_an_unverified_adapter_is_refused_unless_the_owner_allows_it(policy):
    assert sv.adapter_settings(policy, "soulx_svc")["adapter"] == "soulx_svc"
    with pytest.raises(sv.VoiceRefused, match="licence_verified"):
        sv.adapter_settings(policy, "rvc")
    with pytest.raises(sv.VoiceRefused, match="no voice adapter"):
        sv.adapter_settings(policy, "seed_vc")
    allowed = _with_voice(policy, allow_unverified_licences=True)
    assert sv.adapter_settings(allowed, "rvc")["allow_unverified"] is True
    trained = {**ASSET, "model_uri": "s3://b/m.pth"}
    assert sv.adapter_for(policy, trained) == "rvc" and sv.adapter_for(policy, ASSET) == "soulx_svc"
    with pytest.raises(sv.VoiceRefused):                              # auto picks rvc for a trained asset: refused
        sv.convert_payload(policy, trained, vocals_uri="s3://b/v.wav", instrumental_uri="s3://b/i.wav", seed=1)
    with pytest.raises(sv.VoiceRefused):
        sv.train_payload(policy, ["s3://b/c.wav"] * 20, seed=1, name="x")
    assert sv.train_payload(allowed, ["s3://b/c.wav"] * 20, seed=1, name="x")["adapter"]["adapter"] == "rvc"


def test_the_worker_refuses_an_unverified_adapter_before_touching_anything(policy, tmp_path):
    rvc = {**policy.get("music", "voice", "adapters", "rvc"), "adapter": "rvc", "allow_unverified": False}
    ctx = SimpleNamespace(job=SimpleNamespace(id="j"), workdir=tmp_path, store=None, fetch=None)
    with pytest.raises(NonRetryable, match="not licence-verified"):
        mv.convert({"adapter": rvc, "asset": ASSET, "vocals_uri": "a", "instrumental_uri": "b", "seed": 1}, ctx)
    with pytest.raises(NonRetryable, match="not licence-verified"):
        mv.train({"adapter": rvc, "clips": ["a"], "seed": 1}, ctx)
    with pytest.raises(NonRetryable, match="no voice adapter"):
        mv.convert({"adapter": {"adapter": "seed_vc", "licence_verified": True}, "asset": ASSET, "seed": 1}, ctx)


def test_a_conversion_needs_stems_or_a_song_and_an_asset(policy):
    with pytest.raises(sv.VoiceRefused):
        sv.convert_payload(policy, ASSET, vocals_uri="s3://b/v.wav", instrumental_uri=None, seed=1)
    with pytest.raises(sv.VoiceRefused):
        sv.convert_payload(policy, {**ASSET, "centroid": []}, vocals_uri="a", instrumental_uri="b", seed=1)
    p = sv.convert_payload(policy, ASSET, vocals_uri="s3://b/v.wav", instrumental_uri="s3://b/i.wav", seed=2**31 + 5,
                           lyrics=SPEC["lyrics"], check=True)
    assert p["seed"] == 5 and p["adapter"]["adapter"] == "soulx_svc" and p["check"]["lyrics"] == SPEC["lyrics"]
    assert p["similarity"]["model"] == "speechbrain/spkrec-ecapa-voxceleb" and "song" not in p
    s = sv.convert_payload(policy, ASSET, seed=1, song={"title": "x"})
    assert s["song"] == {"title": "x"} and "vocals_uri" not in s


# --- the maths --------------------------------------------------------------------------------------------------------

def test_cosine_and_centroid_weigh_every_window_the_same():
    assert sv.cosine([1, 0], [2, 0]) == 1.0 and sv.cosine([1, 0], [0, 3]) == 0.0
    assert sv.cosine([], [1]) == 0.0 and sv.cosine([0, 0], [1, 1]) == 0.0 and sv.cosine([1, 0], [1, 0, 0]) == 0.0
    c = sv.centroid([[10, 0], [0, 1]])                               # a loud window counts no more than a quiet one
    assert c == pytest.approx([0.707107, 0.707107], abs=1e-6)
    assert sum(x * x for x in c) == pytest.approx(1.0, abs=1e-5) and sv.centroid([]) == []


def test_the_score_is_the_median_so_a_few_bad_windows_do_not_move_it():
    center = [1.0, 0.0]
    good = [[1.0, 0.05]] * 7
    bad = [[0.0, 1.0]] * 2                                           # stem bleed, a held vibrato
    out = sv.score_windows(good + bad, center, min_windows=3)
    assert out["similarity"] == pytest.approx(0.9988, abs=1e-4) and out["windows"] == 9 and out["p10"] < 0.5
    assert sv.score_windows(good[:2], center, min_windows=3)["similarity"] is None
    assert sv.score_windows(good, [], min_windows=1)["similarity"] is None


def test_pairwise_and_calibration_separate_same_from_other_artists():
    pairs = sv.pairwise({"b": [1, 0], "a": [1, 0.1], "c": [0, 1], "none": []})
    assert [(p["a"], p["b"]) for p in pairs] == [("a", "b"), ("a", "c"), ("b", "c")]
    cal = sv.calibrate([0.82, 0.85, 0.8, 0.9, 0.88], [0.3, 0.42, 0.35, 0.5])
    assert cal["separable"] and cal["other_high"] < cal["threshold"] < cal["same_low"]
    overlap = sv.calibrate([0.5, 0.6, 0.7], [0.55, 0.65, 0.75])
    assert overlap["separable"] is False and overlap["threshold"] is None
    assert sv.calibrate([], [0.1])["threshold"] is None


def test_the_gate_is_report_only_until_enforced_with_a_calibrated_threshold(policy):
    assert sv.gate(0.4, policy) == {"similarity": 0.4, "threshold": None, "passed": None, "enforced": False}
    calibrated = _with_voice(policy, gate={"enforce": False, "min_similarity": 0.6})
    assert sv.gate(0.4, calibrated)["passed"] is False and sv.gate(0.4, calibrated)["enforced"] is False
    enforced = _with_voice(policy, gate={"enforce": True, "min_similarity": 0.6})
    assert sv.gate(0.7, enforced) == {"similarity": 0.7, "threshold": 0.6, "passed": True, "enforced": True}
    assert sv.gate(None, enforced)["passed"] is None
    assert sv.gate(0.1, _with_voice(policy, gate={"enforce": True, "min_similarity": None}))["enforced"] is False


def test_spread_and_keeping_the_most_similar_clips():
    same, other = lab.spread([("a", [1, 0]), ("b", [0, 1])], {"a": [1, 0], "b": [0, 1]})
    assert same == [1.0, 1.0] and other == [0.0, 0.0]
    clips = [{"uri": f"u{i}", "similarity": s} for i, s in enumerate([0.9, 0.5, None, 0.7, 0.8, 0.6])]
    kept = lab.keep_for_training(clips, 0.8)
    assert [c["uri"] for c in kept] == ["u0", "u4", "u3", "u5"] and lab.keep_for_training([], 0.8) == []


# --- the worker: windows, remix, commands -----------------------------------------------------------------------------

def _tone(seconds, sr=SR, amp=0.3, f=220.0):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_only_voiced_windows_are_embedded():
    mono = np.concatenate([np.zeros(SR * 6, dtype=np.float32), _tone(10), np.zeros(SR * 6, dtype=np.float32)])
    s = {"window_s": 4, "hop_s": 2, "frame_s": 0.05, "voiced_db": -30, "min_voiced_share": 0.7}
    spans = mv.voiced_windows(mono, SR, s)
    assert spans and all(SR * 6 - SR <= a and b <= SR * 16 + SR for a, b in spans)   # inside the sung stretch
    assert mv.voiced_windows(np.zeros(SR * 10, dtype=np.float32), SR, s) == []
    many = mv.voiced_windows(_tone(60), SR, {**s, "max_windows": 5})
    assert len(many) == 5 and many[0][0] == 0
    seen = []
    sim = mv.voice_similarity(mono, SR, ASSET, {**s, "min_windows": 1},
                              embed=lambda w, sr: seen.append(len(w)) or [1.0, 0.0, 0.0])
    assert sim["similarity"] == 1.0 and sim["windows"] == len(spans) and set(seen) == {SR * 4}
    assert sim["embedding"] == [1.0, 0.0, 0.0]


def test_the_remix_keeps_the_instrumental_length_the_stem_loudness_and_never_clips():
    inst = np.stack([_tone(5, amp=0.6, f=110)] * 2, axis=1)
    source = np.stack([_tone(5, amp=0.2)] * 2, axis=1)
    conv = _tone(4, amp=0.9)[:, None]                               # shorter and louder than the stem
    vocal, mix = mv.remix(conv, SR, inst, SR, source, {"match_loudness": True, "peak_ceiling": 0.98})
    assert vocal.shape == mix.shape == inst.shape
    assert np.abs(mix).max() <= 0.98 + 1e-6
    unscaled = mv.remix(conv, SR, inst * 0.1, SR, source, {"match_loudness": True})[0]
    rms = lambda a: float(np.sqrt(np.mean(a[: SR * 4] ** 2)))      # noqa: E731
    # matched over the whole stem length, 4 s of it sung: the sung part is sqrt(5/4) louder than the stem's average
    assert rms(unscaled[:, 0]) == pytest.approx(float(np.sqrt(np.mean(source[:, 0] ** 2))) * (5 / 4) ** 0.5, rel=0.05)


def test_the_converters_keep_the_source_pitch_and_skip_the_gpl_separator(policy):
    s = {**policy.get("music", "voice", "adapters", "soulx_svc"), "adapter": "soulx_svc"}
    cmds = mv.soulx_commands("/py", 7, s, prompt=Path("p.wav"), target=Path("t.wav"), work=Path("w"), device="cuda")
    assert len(cmds) == 3 and all(c[:2] == ["/py", "-c"] and c[3] == "7" for c in cmds)
    for pre in cmds[:2]:
        assert pre[4] == "preprocess.pipeline" and pre[pre.index("--vocal_sep") + 1] == "False"
        assert pre[pre.index("--midi_transcribe") + 1] == "False"
    infer = cmds[2]
    assert infer[4] == "cli.inference_svc" and infer[infer.index("--pitch_shift") + 1] == "0"
    assert "--auto_shift" not in infer and "--fp16" in infer
    assert infer[infer.index("--model_path") + 1].endswith("model-svc.pt")
    r = {**policy.get("music", "voice", "adapters", "rvc"), "adapter": "rvc"}
    cmd = mv.rvc_infer_command("/py", 3, r, model_name="voice.pth", index=Path("v.index"), source=Path("s.wav"),
                               out=Path("o.wav"))
    assert cmd[4] == "infer/cli.py" and cmd[cmd.index("--pitch") + 1] == "0"
    assert cmd[cmd.index("--f0-method") + 1] == "rmvpe" and cmd[cmd.index("--index") + 1].endswith("v.index")
    steps = mv.rvc_train_commands("/py", 3, r, exp="mw_x", dataset=Path("d"), n_cpu=4, device="cuda")
    assert list(steps) == ["preprocess", "f0", "features", "train", "index"]
    assert steps["train"][1] == "train/train.py" and steps["train"][steps["train"].index("-te") + 1] == "200"
    assert steps["preprocess"][3] == "40000" and steps["f0"][2:4] == ["cuda", "1"]


def test_the_training_filelist_is_the_webuis_shuffled_by_the_seed(tmp_path):
    exp = tmp_path / "logs" / "mw_x"
    for sub, suffix in (("0_gt_wavs", ".wav"), ("3_feature768", ".npy"), ("2a_f0", ".wav.npy"),
                        ("2b-f0nsf", ".wav.npy")):
        (exp / sub).mkdir(parents=True)
        for n in ("0_0", "0_1", "1_0"):
            if not (sub == "2a_f0" and n == "1_0"):                 # one clip lacks its F0: left out
                (exp / sub / f"{n}{suffix}").write_bytes(b"x")
    n = mv.write_filelist(exp, "40k", seed=5, mute_dir=tmp_path / "mute")
    lines = (exp / "filelist.txt").read_text().splitlines()
    assert n == 2 and len(lines) == 4 and sum("mute40k.wav" in x for x in lines) == 2
    assert all(x.endswith("|0") and x.count("|") == 4 for x in lines)
    mv.write_filelist(exp, "40k", seed=5, mute_dir=tmp_path / "mute")
    assert (exp / "filelist.txt").read_text().splitlines() == lines


# --- the conversion job -------------------------------------------------------------------------------------------------

@pytest.fixture
def stubbed(monkeypatch):
    """Audio files as arrays in memory, the converter and the embedder stood in for (no GPU, no soundfile)."""
    files: dict[str, tuple] = {}
    seen = {"convert": []}
    vocals = np.stack([np.concatenate([_tone(8), np.zeros(SR * 2, dtype=np.float32)])] * 2, axis=1)
    inst = np.stack([_tone(10, amp=0.2, f=110)] * 2, axis=1)

    def write(path, a, sr):
        files[str(path)] = (np.asarray(a, dtype=np.float32), sr)
        Path(path).write_bytes(np.asarray(a, dtype=np.float32).tobytes())
        return Path(path)

    def read(path):
        return files[str(path)]

    def fetch(ctx):
        def get(uri, name):
            p = Path(ctx.workdir) / "inputs" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            return write(p, vocals if "vocal" in uri else inst, SR)
        return get

    def converter(src, ctx, settings, asset, seed):
        seen["convert"].append({"seed": seed, "settings": settings, "asset": asset})
        a, sr = files[str(src)]
        return write(Path(ctx.workdir) / "conv.wav", a.mean(axis=1, keepdims=True) * 2, sr), SR

    monkeypatch.setattr(mv, "write_audio", write)
    monkeypatch.setattr(mv, "read_audio", read)
    monkeypatch.setitem(mv.CONVERTERS, "soulx_svc", converter)
    monkeypatch.setattr(mv, "_embedder", lambda model: (lambda w, sr: [1.0, 0.1, 0.0]))
    monkeypatch.setattr(mv, "transcribe", lambda v, sr, check: SPEC["lyrics"].replace("[Verse]", "").replace(
        "[Chorus]", "").replace("(ooh)", ""))
    seen["fetch"], seen["files"] = fetch, files
    return seen


def _ctx(tmp_path, stubbed, name="w"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    ctx = SimpleNamespace(job=SimpleNamespace(id="job-1"), workdir=d, store=None, charges=[])
    ctx.fetch = stubbed["fetch"](ctx)
    return ctx


def test_a_conversion_returns_her_vocal_and_the_new_mix_with_its_similarity(stubbed, tmp_path, policy):
    payload = sv.convert_payload(policy, ASSET, vocals_uri="s3://b/vocals.wav", instrumental_uri="s3://b/inst.wav",
                                 seed=9, lyrics=SPEC["lyrics"], check=True)
    payload["similarity"]["min_windows"] = 1
    res = mv.convert(payload, _ctx(tmp_path, stubbed))
    assert [o.meta["role"] for o in res.outputs] == list(sv.ROLES)
    assert [o.path.name for o in res.outputs] == ["voice_vocals.wav", "voice_mix.wav"]
    v = res.outputs[0].meta
    assert v["converted"] and v["similarity"] == pytest.approx(0.995, abs=1e-3) and v["windows"] >= 1
    assert v["source_similarity"] == v["similarity"] and v["embedding"] and v["line_coverage"] == 1.0
    assert v["seconds"] == 10 and v["sample_rate"] == SR and v["voice_version"] == 1
    assert stubbed["convert"][0]["seed"] == 9 and stubbed["convert"][0]["settings"]["voiced"]["window_s"] == 4.0
    assert res.recipe["commit"] == policy.get("music", "voice", "adapters", "soulx_svc", "commit")


def test_a_rerun_makes_the_same_files(stubbed, tmp_path, policy):
    payload = sv.convert_payload(policy, ASSET, vocals_uri="s3://b/vocals.wav", instrumental_uri="s3://b/inst.wav",
                                 seed=9)
    a, b = mv.convert(payload, _ctx(tmp_path, stubbed, "a")), mv.convert(payload, _ctx(tmp_path, stubbed, "b"))
    assert [o.path.name for o in a.outputs] == [o.path.name for o in b.outputs]
    assert [o.path.read_bytes() for o in a.outputs] == [o.path.read_bytes() for o in b.outputs]
    assert a.recipe == b.recipe and a.outputs[0].meta == b.outputs[0].meta


def test_song_mode_makes_the_song_then_converts_it_keeping_the_original_last(stubbed, tmp_path, policy, monkeypatch):
    def make(song, ctx):
        get = stubbed["fetch"](ctx)        # stand-in song files: the vocal and the instrumental as the stub makes them
        return {role: OutputFile(get(f"s3://b/{'vocals' if role == 'vocals' else 'inst'}", f"song_{role}.wav"),
                                 "audio", {"role": role, "seconds": 10}) for role in music.ROLES}
    monkeypatch.setattr(mv, "_make_song", make)
    payload = sv.convert_payload(policy, ASSET, seed=3, song={"title": "x", "seed": 1})
    res = mv.convert(payload, _ctx(tmp_path, stubbed))
    assert [o.meta["role"] for o in res.outputs] == ["vocals", "mix", "source_mix", "source_vocals", "instrumental"]
    assert all(o.meta["converted"] is False for o in res.outputs[2:]) and res.recipe["song"]["title"] == "x"


# --- the creator flow ---------------------------------------------------------------------------------------------------

def _finish(conn, job_id, roles_meta: dict):
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job_id,))
    for i, (role, meta) in enumerate(roles_meta.items()):
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                     "SELECT %s, creator_id, 'audio', %s, 'x', 1, %s FROM jobs WHERE id = %s",
                     (job_id, f"s3://b/{job_id}/{i:02d}.wav", Jsonb({"role": role, **meta}), job_id))


def _artist(conn, centroid=None):
    cid = creator(conn, lora=False)
    conn.execute("UPDATE creators SET bible = bible || %s WHERE id = %s",
                 (Jsonb({"music": {"vocal_tags": ["warm indie pop", "airy female vocal"]}}), cid))
    if centroid:
        conn.execute("UPDATE creators SET anchors = anchors || %s WHERE id = %s",
                     (Jsonb({"music_voice": {**ASSET, "centroid": centroid}}), cid))
    return cid


def test_the_voice_flow_seeds_picks_far_from_other_artists_and_builds_a_dataset(conn, policy):
    _artist(conn, centroid=[1.0, 0.0, 0.0])                         # another artist already sings like [1, 0, 0]
    cid = _artist(conn)
    first = lab.advance(conn, policy, cid)
    n = policy.get("music", "voice", "creation", "seed_candidates")
    assert first["stage"] == "seed" and len(first["jobs"]) == n
    again = lab.advance(conn, policy, cid)
    assert again == {"stage": "seed", "waiting": n}
    jobs = conn.execute("SELECT payload FROM jobs WHERE creator_id = %s AND kind = 'music.song'", (cid,)).fetchall()
    assert len(jobs) == n and all(j["payload"]["embed"]["model"] for j in jobs)
    assert all("airy female vocal" in j["payload"]["tags"] for j in jobs)
    seeds = lab._lab(conn, cid)["seed_jobs"]
    for i, job in enumerate(seeds):
        emb = [1.0, 0.0, 0.0] if i == 0 else [0.0, 1.0, 0.1 * i]   # candidate 0 sounds like the other artist
        _finish(conn, job, {"mix": {"check": {"passed": True, "line_coverage": 1.0 if i in (0, 3) else 0.8}},
                            "vocals": {"embedding": emb}, "instrumental": {}})
    picked = lab.advance(conn, policy, cid)
    assert picked["stage"] == "pick" and picked["picked"]["index"] == 3 and picked["version"] == 1
    asset = lab.current(conn, cid)
    assert asset["adapter"] == "soulx_svc" and asset["prompt_uri"].startswith(f"s3://b/{seeds[3]}/")
    row = conn.execute("SELECT version, source FROM music_voices WHERE creator_id = %s", (cid,)).fetchone()
    assert row["version"] == 1 and row["source"]["candidate"] == 3
    d = conn.execute("SELECT step, rationale FROM decisions WHERE creator_id = %s AND kind = 'music_voice_pick'",
                     (cid,)).fetchone()
    assert d["step"] == "music_voice_pick" and "coverage" in d["rationale"]
    ds = lab.advance(conn, policy, cid)
    c = policy.get("music", "voice", "creation")
    assert ds == {"stage": "dataset", "jobs": len(c["registers"]) * c["songs_per_register"]}
    keys = {r["payload"]["key"] for r in conn.execute(
        "SELECT payload FROM jobs WHERE creator_id = %s AND kind = 'music.song' AND payload->>'title' LIKE 'voice %%'",
        (cid,)).fetchall()}
    assert {"E minor", "G major", "B minor"} <= keys                 # every register
    for job in lab._lab(conn, cid)["dataset_jobs"]:
        _finish(conn, job, {"mix": {}, "vocals": {}, "instrumental": {}})
    conv = lab.advance(conn, policy, cid)
    assert conv["stage"] == "convert" and conv["jobs"] == ds["jobs"]
    cj = conn.execute("SELECT payload, idempotency_key FROM jobs WHERE kind = 'music.convert_voice' AND creator_id = %s",
                      (cid,)).fetchall()
    assert all(j["payload"]["adapter"]["adapter"] == "soulx_svc" and ":v1:convert:" in j["idempotency_key"] for j in cj)
    for i, job in enumerate(lab._lab(conn, cid)["convert_jobs"]):
        _finish(conn, job, {"vocals": {"similarity": 0.5 + i / 100, "seconds": 60, "embedding": [0, 1, 0]}, "mix": {}})
    refused = lab.advance(conn, policy, cid)                         # rvc's licences unverified: no training
    assert refused["stage"] == "train" and "licence_verified" in refused["refused"]
    assert refused["clips"] == round(ds["jobs"] * c["dataset_keep_share"])
    assert conn.execute("SELECT count(*) n FROM jobs WHERE kind = 'music.train_voice'").fetchone()["n"] == 0


def test_the_owner_may_pick_a_candidate_but_never_one_like_another_artist(conn, policy):
    _artist(conn, centroid=[1.0, 0.0, 0.0])
    cid = _artist(conn)
    lab.advance(conn, policy, cid)
    seeds = lab._lab(conn, cid)["seed_jobs"]
    for i, job in enumerate(seeds):
        _finish(conn, job, {"mix": {"check": {"passed": True}}, "vocals": {"embedding": [1.0, 0.0, 0.0] if i == 0
                                                                           else [0.0, 1.0, 0.0]}, "instrumental": {}})
    out = lab.pick(conn, policy, cid, seeds, candidate=0)
    assert out["picked"] is None and lab.current(conn, cid) is None
    out = lab.pick(conn, policy, cid, seeds, candidate=2)
    assert out["picked"]["index"] == 2 and out["override"] and lab.current(conn, cid)["version"] == 1


def test_a_trained_model_becomes_the_next_version(conn, policy):
    cid = _artist(conn)
    lab.store(conn, policy, cid, adapter="soulx_svc", centroid=[0, 1, 0], prompt_uri="s3://b/p.wav")
    asset = lab.current(conn, cid)
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('music.train_voice', 'done', %s, "
                       "'{}') RETURNING id", (cid,)).fetchone()["id"]
    conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) VALUES "
                 "(%s, %s, 'voice_model', 's3://b/m.pth', 'x', 1, %s), (%s, %s, 'voice_index', 's3://b/m.index', 'x', 1, '{}')",
                 (job, cid, Jsonb({"centroid": [0, 0.9, 0.1]}), job, cid))
    new = lab.store_trained(conn, policy, cid, asset, job, [{"uri": "u", "similarity": 0.8}])
    assert new["version"] == 2 and new["adapter"] == "rvc" and new["model_uri"] == "s3://b/m.pth"
    assert new["prompt_uri"] == "s3://b/p.wav" and lab.current(conn, cid)["version"] == 2
    assert lab.advance(conn, policy, cid)["stage"] == "ready"
    assert [r["version"] for r in conn.execute("SELECT version FROM music_voices WHERE creator_id = %s ORDER BY 1",
                                               (cid,)).fetchall()] == [1, 2]


# --- production -----------------------------------------------------------------------------------------------------------

def _music_post(conn, *, asset=True):
    cid = _artist(conn, centroid=[1.0, 0.0, 0.0] if asset else None)
    return dict(conn.execute("INSERT INTO posts (creator_id, format) VALUES (%s, 'music_video') RETURNING *",
                             (cid,)).fetchone())


def _song_done(conn, body, post):
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (body["song_jobs"][0],))
    for i, role in enumerate(music.ROLES):
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                     "VALUES (%s, %s, 'audio', %s, 'x', 1, %s)",
                     (body["song_jobs"][0], post["creator_id"], f"s3://b/song/{i:02d}.wav",
                      Jsonb({"role": role, "seconds": 25, "check": {"passed": True}})))


CHOSEN = [{"id": "x", "uri": "s3://b/photo.png"}]


def test_production_does_not_convert_while_the_switch_is_off(conn, policy):
    post = _music_post(conn)
    body = {"scenes": ["her bedroom"], "song": SPEC}
    followups._advance_follow_up(conn, policy, post, body, CHOSEN)
    _song_done(conn, body, post)
    followups._advance_follow_up(conn, policy, post, body, CHOSEN)
    assert not body.get("song_voice_jobs") and body["song_media"]["vocals"]["uri"] == "s3://b/song/01.wav"
    assert conn.execute("SELECT count(*) n FROM jobs WHERE kind = 'music.convert_voice'").fetchone()["n"] == 0


def test_with_the_switch_on_the_song_is_sung_in_her_voice(conn, policy):
    on = _with_voice(policy, enabled=True)
    post = _music_post(conn)
    body = {"scenes": ["her bedroom"], "song": SPEC}
    followups._advance_follow_up(conn, on, post, body, CHOSEN)
    _song_done(conn, body, post)
    state, _ = followups._advance_follow_up(conn, on, post, body, CHOSEN)
    assert state == "waiting" and len(body["song_voice_jobs"]) == 1 and not body.get("talk_jobs")
    job = conn.execute("SELECT kind, payload, idempotency_key FROM jobs WHERE id = %s",
                       (body["song_voice_jobs"][0],)).fetchone()
    assert job["kind"] == "music.convert_voice" and job["payload"]["vocals_uri"] == "s3://b/song/01.wav"
    assert job["payload"]["instrumental_uri"] == "s3://b/song/02.wav" and job["payload"]["check"]["lyrics"]
    assert job["idempotency_key"] == f"post:{post['id']}:song_voice:v1:0"
    _finish(conn, body["song_voice_jobs"][0], {"vocals": {"similarity": 0.7, "seconds": 25}, "mix": {"seconds": 25}})
    state, _ = followups._advance_follow_up(conn, on, post, body, CHOSEN)
    media = body["song_media"]
    assert media["vocals"]["uri"].endswith("/00.wav") and media["mix"]["uri"].endswith("/01.wav")
    assert media["source_vocals"]["uri"] == "s3://b/song/01.wav"
    talk = conn.execute("SELECT payload FROM jobs WHERE id = %s", (body["talk_jobs"][0],)).fetchone()
    assert talk["payload"]["audio_uri"] == media["vocals"]["uri"]   # lip-synced to her converted voice


def test_a_refused_conversion_keeps_the_songs_voice_and_says_why(conn, policy):
    on = _with_voice(policy, enabled=True)
    post = _music_post(conn)
    conn.execute("UPDATE creators SET anchors = jsonb_set(anchors, '{music_voice,model_uri}', '\"s3://b/m.pth\"') "
                 "WHERE id = %s", (post["creator_id"],))             # trained: auto picks rvc, licence unverified
    body = {"scenes": ["a stage"], "song": SPEC}
    song_mod.advance_song(conn, on, post, body)
    _song_done(conn, body, post)
    state, media = song_mod.advance_song(conn, on, post, body)
    assert state == "chosen" and "licence_verified" in body["song_voice_skipped"] and not body.get("song_voice_jobs")
    assert body["song_media"]["vocals"]["uri"] == "s3://b/song/01.wav"


def test_an_enforced_voice_gate_remakes_a_conversion_under_it(conn, policy):
    strict = _with_voice(policy, enabled=True, gate={"enforce": True, "min_similarity": 0.6})
    post = _music_post(conn)
    body = {"scenes": ["a stage"], "song": SPEC}
    song_mod.advance_song(conn, strict, post, body)
    _song_done(conn, body, post)
    song_mod.advance_song(conn, strict, post, body)
    _finish(conn, body["song_voice_jobs"][0], {"vocals": {"similarity": 0.3}, "mix": {}})
    state, _ = song_mod.advance_song(conn, strict, post, body)
    assert state == "waiting" and len(body["song_voice_jobs"]) == 2 and body["song_voice_repairs"] == 1
    no_asset = _music_post(conn, asset=False)
    body2 = {"scenes": ["a stage"], "song": SPEC}
    song_mod.advance_song(conn, strict, no_asset, body2)
    _song_done(conn, body2, no_asset)
    assert song_mod.advance_song(conn, strict, no_asset, body2)[0] == "chosen" and not body2.get("song_voice_jobs")


# --- the bench ------------------------------------------------------------------------------------------------------------

def test_the_voice_bench_needs_a_test_artist_with_a_voice(conn, policy):
    with pytest.raises(ValueError, match="voice asset"):
        bench.enqueue(conn, policy, "music.convert_voice", 4)
    with pytest.raises(ValueError, match="voice asset"):
        bench.enqueue(conn, policy, "music.convert_voice", 4, extra={"creator_id": str(_artist(conn))})


def test_the_voice_bench_makes_and_converts_four_songs_and_reports_their_consistency(conn, policy):
    cid = _artist(conn, centroid=[1.0, 0.0, 0.0])
    batch = bench.enqueue(conn, policy, "music.convert_voice", 4, extra={"creator_id": str(cid)})
    rows = conn.execute("SELECT id, payload FROM jobs WHERE batch = %s ORDER BY payload->>'seed'", (batch,)).fetchall()
    assert len(rows) == 4 and len({r["payload"]["song"]["title"] for r in rows}) == 4
    for r in rows:
        p = r["payload"]
        assert p["song"]["embed"] and p["song"]["check"] and p["check"]["lyrics"] == next(
            s["lyrics"] for s in bench.voice_songs() if s["title"] == p["song"]["title"])
        assert p["adapter"]["adapter"] == "soulx_svc" and p["asset"]["centroid"] == [1.0, 0.0, 0.0]
    for s in bench.voice_songs():                                    # the test's own songs pass the lyric checks
        assert music.issues(s, policy) == []
    for i, r in enumerate(rows):
        conn.execute("INSERT INTO attempts (job_id, worker_id, pool, gpu_type, gpu_seconds, outcome) "
                     "VALUES (%s, 'w', 'vast_4090', 'rtx4090', 240, 'done')", (r["id"],))
        _finish(conn, r["id"], {"vocals": {"similarity": 0.8 + i / 100, "source_similarity": 0.3, "line_coverage": 1.0,
                                           "embedding": [1.0, 0.1 * i, 0.0], "source_embedding": [0.0, 1.0, i]},
                                "mix": {}, "source_mix": {}, "source_vocals": {}, "instrumental": {}})
    rep = bench.report(conn, policy, batch)
    assert rep["measured"][0]["outputs"] == 4 and rep["measured"][0]["sec_per_output_median"] == 240
    vc = rep["voice_consistency"]
    assert vc["songs"] == 4 and vc["similarity_median"] == pytest.approx(0.815) and vc["line_coverage_median"] == 1.0
    assert len(vc["pairwise"]) == 6 and vc["pairwise_min"] > vc["source_pairwise_median"]
    assert all(e["gate"]["enforced"] is False for e in vc["each"])


def test_calibration_reads_converted_vocals_against_every_artist(conn, policy):
    a = _artist(conn, centroid=[1.0, 0.0, 0.0])
    b = _artist(conn, centroid=[0.0, 1.0, 0.0])
    for cid, emb in ((a, [1.0, 0.1, 0.0]), (b, [0.1, 1.0, 0.0])):
        job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload, batch) VALUES "
                           "('music.convert_voice', 'queued', %s, '{}', %s) RETURNING id",
                           (cid, f"cal-{uuid.uuid4().hex[:6]}")).fetchone()["id"]
        _finish(conn, job, {"vocals": {"embedding": emb}, "mix": {}})
    cal = lab.calibration(conn, policy)
    assert cal["same"] >= 2 and cal["other"] >= 2
