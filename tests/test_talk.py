"""Lip-synced talking video (InfiniteTalk): one approved photo + the creator's voice → a talking clip,
with keyframes for the video gate. The model runs as a subprocess from its own repo; these tests pin
what we hand it and what we store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from manyworlds.worker.handlers import talk_infinitetalk as talk
from manyworlds.worker.runner import JobContext, NonRetryable, OutputFile


def test_the_command_points_at_the_weights_and_the_input(tmp_path):
    cmd = talk.build_command(
        base="/w/wan", wav2vec="/w/wav2vec", infinitetalk="/w/it", input_json=tmp_path / "in.json",
        out_stem=tmp_path / "clip", size="infinitetalk-480", steps=8, audio_cfg=4.0, text_cfg=5.0, seed=7,
        extra=["--quant", "fp8"], lora=None, motion_frame=9)
    s = " ".join(cmd)
    assert "--ckpt_dir /w/wan" in s and "--wav2vec_dir /w/wav2vec" in s
    assert "--infinitetalk_dir /w/it/single/infinitetalk.safetensors" in s
    assert f"--input_json {tmp_path / 'in.json'}" in s and "--size infinitetalk-480" in s
    assert "--sample_steps 8" in s and "--sample_audio_guide_scale 4.0" in s and "--base_seed 7" in s
    assert cmd[-2:] == ["--quant", "fp8"]


def test_a_lora_adds_its_path_and_scale():
    cmd = talk.build_command(base="b", wav2vec="w", infinitetalk="i", input_json=Path("j"), out_stem=Path("o"),
                             size="infinitetalk-480", steps=8, audio_cfg=2.0, text_cfg=1.0, seed=1, extra=[],
                             lora={"path": "/w/lora.safetensors", "scale": 1.0}, motion_frame=9)
    s = " ".join(cmd)
    assert "--lora_dir /w/lora.safetensors" in s and "--lora_scale 1.0" in s


def test_the_input_file_names_the_photo_the_voice_and_the_prompt(tmp_path):
    p = talk.write_input(tmp_path, image=tmp_path / "start.png", audio=tmp_path / "voice.wav",
                         prompt="a woman talks to camera at a podium")
    body = json.loads(p.read_text())
    assert body == {"prompt": "a woman talks to camera at a podium", "cond_video": str(tmp_path / "start.png"),
                    "cond_audio": {"person1": str(tmp_path / "voice.wav")}}


PARAMS = {"size": "infinitetalk-480", "steps": 8, "audio_cfg": 2.0, "text_cfg": 1.0, "motion_frame": 9,
          "timeout_s": 600}


def test_settings_come_from_the_payload_not_handler_defaults():
    ctx = JobContext(job=None, workdir=Path("."), store=None)
    with pytest.raises(NonRetryable, match="params missing"):
        talk.generate({"image_uri": "s3://x/a.png", "script": "hi", "keyframes": 4,
                       "params": {k: v for k, v in PARAMS.items() if k != "steps"}}, ctx)


def test_a_job_needs_a_photo_and_either_audio_or_a_script():
    ctx = JobContext(job=None, workdir=Path("."), store=None)
    with pytest.raises(NonRetryable, match="image_uri"):
        talk.generate({"script": "hi"}, ctx)
    with pytest.raises(NonRetryable, match="audio_uri or script"):
        talk.generate({"image_uri": "s3://x/a.png"}, ctx)


def test_a_talking_clip_is_stored_with_its_keyframes_and_recipe(tmp_path, store, monkeypatch):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG")
    wav = tmp_path / "v.wav"
    wav.write_bytes(b"RIFF")
    image_uri, audio_uri = store.put_file(img, "t/a.png").uri, store.put_file(wav, "t/v.wav").uri
    monkeypatch.setattr(talk, "weights", lambda store, workdir, **kw: {"base": "b", "wav2vec": "w", "infinitetalk": "i"})
    monkeypatch.setattr(talk, "audio_seconds", lambda path: 6.5)
    ran = {}

    def fake_run(cmd, cwd, timeout=None):
        ran["timeout"] = timeout
        ran["cmd"] = cmd
        Path(cmd[cmd.index("--save_file") + 1] + ".mp4").write_bytes(b"mp4")
    monkeypatch.setattr(talk, "run", fake_run)
    monkeypatch.setattr(talk, "extract_keyframes",
                        lambda path, workdir, n: [OutputFile(workdir / f"k{i}.jpg", "keyframe", {"index": i})
                                                  for i in range(n)])
    work = tmp_path / "work"
    work.mkdir()
    ctx = JobContext(job=None, workdir=work, store=store)
    out = talk.generate({"image_uri": image_uri, "audio_uri": audio_uri, "prompt": "talks", "keyframes": 4,
                         "seed": 3, "params": PARAMS}, ctx)
    assert [o.kind for o in out.outputs] == ["video", "keyframe", "keyframe", "keyframe", "keyframe"]
    assert out.outputs[0].meta["seconds"] == 6.5 and out.meta["output_seconds"] == 6.5
    assert ran["timeout"] == 600 and "--motion_frame" in ran["cmd"]
    assert out.recipe["model"] == talk.MODEL and out.recipe["seed"] == 3 and out.recipe["image_uri"] == image_uri


def test_only_the_adapter_files_a_run_names_are_downloaded():
    assert talk.adapter_files([]) == ["single/*"]
    assert talk.adapter_files(["--quant", "fp8", "--quant_dir",
                               "{infinitetalk}/quant_models/infinitetalk_single_fp8.safetensors"]) == [
        "single/*", "quant_models/infinitetalk_single_fp8*", "quant_models/t5_fp8*", "quant_models/t5_map_fp8*"]


def test_a_lora_given_as_a_path_is_used_as_is(tmp_path):
    assert talk.lora_path({"path": "/w/l.safetensors", "scale": 0.8}, tmp_path) == {"path": "/w/l.safetensors", "scale": 0.8}
    assert talk.lora_path(None, tmp_path) is None


def test_a_talk_benchmark_uses_policy_settings_and_the_given_photo(conn, policy):
    from manyworlds import bench
    from brain_helpers import creator
    cid = creator(conn)
    batch = bench.enqueue(conn, policy, "talk.infinitetalk", 2,
                          extra={"image_uri": "s3://b/maren.png", "creator_id": str(cid)}, pools=["vast_4090"])
    jobs = conn.execute("SELECT payload, creator_id FROM jobs WHERE batch = %s", (batch,)).fetchall()
    assert all(j["creator_id"] == cid and "creator_id" not in j["payload"] for j in jobs)   # the gate judges against her
    talk_cfg = policy.get("generation", "talk")
    assert len(jobs) == 2 and all(j["payload"]["image_uri"] == "s3://b/maren.png" for j in jobs)
    p = jobs[0]["payload"]
    assert p["script"] and p["voice"] == talk_cfg["voice"] and p["params"]["steps"] == talk_cfg["steps"]
    assert "voice" not in p["params"]


def test_a_talk_benchmark_without_a_start_photo_is_refused_before_any_machine_starts(conn, policy):
    from manyworlds import bench
    with pytest.raises(ValueError, match="image_uri"):
        bench.enqueue(conn, policy, "talk.infinitetalk", 1)
    assert conn.execute("SELECT count(*) n FROM jobs").fetchone()["n"] == 0


def test_a_quantised_run_skips_the_full_precision_base_weights(tmp_path, monkeypatch):
    calls = {}

    def fake_local(repo, store, workdir, **kw):
        calls[repo] = kw
        d = tmp_path / repo.replace("/", "--")
        d.mkdir(exist_ok=True)
        (d / "model.safetensors").write_bytes(b"x")
        return str(d)
    monkeypatch.setattr(talk, "_local", fake_local)
    talk.weights(None, tmp_path, extra=["--quant", "fp8", "--quant_dir", "{infinitetalk}/quant_models/x.safetensors"])
    assert calls[talk.BASE]["ignore_patterns"] == talk.QUANT_SKIPS_BASE
    talk.weights(None, tmp_path, extra=[])
    assert calls[talk.BASE] == {}                                     # full precision needs everything
