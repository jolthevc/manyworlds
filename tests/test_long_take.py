"""One continuous take against our segment chain (manyworlds.long_take, worker/handlers/talk_longcat.py): the native
LongCat 1.6 runner's command and segments, the shared voice line cut between sentences for a chain, the chain
continuing from each segment's last frame, the finished take measured (colour drift, identity) and the breadth plan
that runs both. No GPU: the LongCat subprocess, the lip-sync score and the judge are stubbed."""

from __future__ import annotations

import subprocess
import uuid
import wave
from pathlib import Path

import pytest
from PIL import Image
from psycopg.types.json import Jsonb

from manyworlds import breadth, long_take, media, policy as policy_mod, queue
from manyworlds.worker import runner
from manyworlds.worker.handlers import talk_comfy, talk_longcat
from manyworlds.worker.runner import JobContext, NonRetryable, OutputFile

from test_breadth import _paused
from test_review import BIBLE

VOICE = {"engine": "qwen3", "ref_uri": "s3://b/ref.wav", "ref_text": "Hi, it's me."}


def _creator(conn, *, designed=True):
    anchors = {"reference_images": [{"uri": "s3://b/r0.png"}]}
    if designed:
        anchors["voice"] = VOICE
    return conn.execute("INSERT INTO creators (handle, bible, anchors) VALUES (%s, %s, %s) "
                        "RETURNING id, handle, anchors", (f"c{uuid.uuid4().hex[:6]}", Jsonb(BIBLE),
                                                           Jsonb(anchors))).fetchone()


def _profile(raw_policy):
    return {**raw_policy.get("generation", "talk"), **raw_policy.get("generation", "talk_profiles", "longcat16")}


# ------------------------------------------------------------------------------------------- the native runner

def test_a_90_second_line_is_one_job_of_28_segments(raw_policy):
    p = _profile(raw_policy)
    seg = dict(fps=p["fps"], frames_per_segment=p["frames_per_segment"], cond_frames=p["cond_frames"])
    assert talk_longcat.segments_for(90, **seg) == 28          # 93 + 27 * 80 frames = 2253 >= 2250
    assert talk_longcat.segments_for(3.0, **seg) == 1
    assert talk_longcat.segments_for(3.8, **seg) == 2
    assert p["max_seconds"] >= 90 and p["model"] == talk_comfy.NATIVE_LONGCAT


def test_quantization_follows_the_card_and_policy(raw_policy):
    p = _profile(raw_policy)
    assert talk_longcat.resolve_quantization(p, 47.7) == "int8"      # an L40S
    assert talk_longcat.resolve_quantization(p, 85.0) == "bf16"      # an H100
    assert talk_longcat.resolve_quantization(p, None) == "int8"
    assert talk_longcat.resolve_quantization({**p, "quantization": "bf16"}, 47.7) == "bf16"
    with pytest.raises(NonRetryable):
        talk_longcat.resolve_quantization({**p, "quantization": "int4"}, 47.7)


def test_only_the_files_the_script_reads_are_downloaded():
    int8 = talk_longcat.avatar_patterns("int8")
    assert "base_model_int8/*" in int8 and "base_model/*" not in int8
    assert "loraA/*" in int8 and "loraB/*" in int8 and "whisper-large-v3/model.safetensors" in int8
    assert not any("flax" in f or "fp32" in f or f.endswith(".bin") or f.startswith("assets") for f in int8)
    assert "base_model/*" in talk_longcat.avatar_patterns("bf16")
    assert {"tokenizer/*", "text_encoder/*", "vae/*"} <= set(talk_longcat.BASE_FILES)


def test_the_command_runs_the_1_6_recipe_on_one_gpu(raw_policy, tmp_path):
    p = _profile(raw_policy)
    cmd = talk_longcat.build_command(checkpoint=tmp_path / "LongCat-Video-Avatar-1.6", input_json=tmp_path / "i.json",
                                     out_dir=tmp_path / "out", segments=28, quantization="int8", params=p,
                                     python="py", repo=Path("/opt/LongCat-Video"))
    assert cmd[:5] == ["py", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1"]
    for flag in ("--num_segments=28", "--model_type=avatar-v1.6", "--use_distill", "--use_int8", "--stage_1=ai2v",
                 f"--mix_steps={p['mix_steps']}", f"--ref_img_index={p['ref_img_index']}",
                 "--context_parallel_size=1"):
        assert flag in cmd, flag
    assert cmd[cmd.index("--lora_path") + 1] == p["lora_path"]
    bf16 = talk_longcat.build_command(checkpoint=tmp_path, input_json=tmp_path, out_dir=tmp_path, segments=1,
                                      quantization="bf16", params=p)
    assert "--use_int8" not in bf16 and "--use_fp8" not in bf16


def test_the_weights_refuse_a_full_disk_as_a_machine_fault(tmp_path, monkeypatch):
    monkeypatch.setattr(talk_longcat.shutil, "disk_usage", lambda p: type("U", (), {"free": 5e9})())
    with pytest.raises(RuntimeError) as e:
        talk_longcat.ensure_weights(tmp_path, "int8")
    assert runner.is_worker_fault(e.value)


def _png(path: Path) -> Path:
    Image.new("RGB", (864, 1536), (180, 120, 90)).save(path)
    return path


def test_one_job_renders_the_whole_line_and_records_the_take(raw_policy, tmp_path, store, monkeypatch):
    p = _profile(raw_policy)
    image_uri = store.put_file(_png(tmp_path / "s.png"), "t/s.png").uri
    audio = tmp_path / "v.wav"
    audio.write_bytes(b"RIFF")
    ran = {}

    def fake_run(cmd, cwd, timeout):
        ran.update(cmd=cmd, timeout=timeout)
        out = Path(next(a.split("=", 1)[1] for a in cmd if a.startswith("--output_dir=")))
        out.mkdir(parents=True)
        (out / "video_continue_28.mp4").write_bytes(b"mp4")
    monkeypatch.setattr(talk_longcat, "run", fake_run)
    monkeypatch.setattr(talk_longcat, "ensure_weights", lambda root, q, **kw: (root / "a", root / "b"))
    monkeypatch.setattr(talk_longcat, "_vram_gb", lambda: 47.7)
    monkeypatch.setattr(talk_longcat, "_free_comfy", lambda: ran.setdefault("freed", True))
    monkeypatch.setattr(talk_comfy, "voice_line", lambda payload, ctx, work: (audio, 90.0, []))
    monkeypatch.setattr(talk_comfy, "join_windows", lambda clips, a, s, dest: dest.write_bytes(b"mp4") and dest)
    monkeypatch.setattr(talk_longcat.lipsync, "score_in_venv", lambda clip, a, **kw: {"corr": 0.6})
    monkeypatch.setattr(talk_longcat.crops, "crops_in_venv", lambda frames, work, spec: [])
    monkeypatch.setattr(talk_longcat, "extract_keyframes",
                        lambda path, workdir, n: [OutputFile(workdir / f"k{i}.jpg", "keyframe", {"index": i})
                                                  for i in range(n)])
    work = tmp_path / "work"
    work.mkdir()
    payload = {"image_uri": image_uri, "audio_uri": "s3://b/v.wav", "prompt": "she talks", "keyframes": 4,
               "params": p}
    out = talk_comfy.generate(payload, JobContext(job=None, workdir=work, store=store))   # dispatched by model
    assert "--num_segments=28" in ran["cmd"] and "--use_int8" in ran["cmd"] and ran["freed"]
    assert ran["timeout"] == p["timeout_s"]
    video = out.outputs[0]
    assert video.kind == "video" and video.meta["single_take"] and video.meta["segments"] == 28
    assert video.meta["sync"] == {"corr": 0.6} and video.meta["quantization"] == "int8"
    assert [o.kind for o in out.outputs[1:]] == ["keyframe"] * 4
    assert out.recipe["model"] == talk_longcat.AVATAR and out.meta["output_seconds"] == 90.0
    assert Image.open(work / "longcat_start.png").size == (p["width"], p["height"])


def test_a_line_longer_than_one_take_is_refused(raw_policy, tmp_path, store, monkeypatch):
    p = _profile(raw_policy)
    monkeypatch.setattr(talk_comfy, "voice_line", lambda payload, ctx, work: (tmp_path / "v.wav", 200.0, []))
    with pytest.raises(NonRetryable, match="at most"):
        talk_longcat.generate({"image_uri": "s3://b/i.png", "audio_uri": "s3://b/v.wav", "keyframes": 4,
                               "params": p}, JobContext(job=None, workdir=tmp_path, store=store))
    with pytest.raises(NonRetryable, match="params missing"):
        talk_longcat.generate({"image_uri": "s3://b/i.png", "audio_uri": "s3://b/v.wav", "keyframes": 4,
                               "params": {"model": "longcat_native"}}, JobContext(job=None, workdir=tmp_path,
                                                                                 store=store))


def test_the_talk_worker_serves_the_long_take_kind(raw_policy):
    assert "talk.longcat" in runner.handlers_for(["talk.longcat"])
    assert raw_policy.family("talk.longcat") == "talk"
    kinds = raw_policy.get("job_kinds", "talk.longcat")
    assert kinds["min_vram_gb"] >= 40
    runs = raw_policy.get("gpu_runs")
    minutes = runs["max_attempt_minutes_by_kind"]["talk.longcat"]
    assert minutes * 60 > _profile(raw_policy)["timeout_s"] > runs["max_attempt_minutes_by_family"]["talk"] * 60
    text = (Path(__file__).resolve().parents[1] / "docker" / "worker-talkcomfy.Dockerfile").read_text()
    assert "talk.longcat" in text and "LONGCAT_REF=" in text and "constraints-longcat.txt" in text


def _wav(path: Path, seconds: float, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return path


def test_a_window_of_the_shared_line_is_cut_out(tmp_path):
    out = talk_comfy.cut_audio(_wav(tmp_path / "a.wav", 5.0), tmp_path / "w.wav", [1.0, 3.5])
    with wave.open(str(out)) as w:
        assert abs(w.getnframes() / w.getframerate() - 2.5) < 0.05
    with pytest.raises(NonRetryable):
        talk_comfy.cut_audio(tmp_path / "a.wav", tmp_path / "x.wav", [3, 1])
    with pytest.raises(NonRetryable):
        talk_comfy.cut_audio(tmp_path / "a.wav", tmp_path / "x.wav", "all")


# --------------------------------------------------------------------------------------------- the comparison

LINES = [{"start_s": 0.0, "end_s": 20.0, "text": "One."}, {"start_s": 20.5, "end_s": 28.0, "text": "Two."},
         {"start_s": 28.6, "end_s": 45.0, "text": "Three."}, {"start_s": 45.4, "end_s": 69.0, "text": "Four."}]


def test_a_chain_is_cut_between_sentences_and_a_single_take_is_the_whole_line():
    parts = long_take.windows(LINES, 70.0, 30)
    assert [p["text"] for p in parts] == ["One. Two.", "Three.", "Four."]
    assert parts[0]["start_s"] == 0 and parts[0]["end_s"] == pytest.approx(28.3)   # halfway through the pause
    assert parts[1]["start_s"] == parts[0]["end_s"] and parts[1]["end_s"] == pytest.approx(45.2)
    assert parts[-1]["end_s"] == 70.0
    one = long_take.windows(LINES, 70.0, None)
    assert len(one) == 1 and one[0]["text"] == "One. Two. Three. Four." and one[0]["end_s"] == 70.0


def test_colour_difference_matches_the_ciede2000_reference():
    # Sharma, Wu and Dalal (2005), test pairs 1 and 7
    assert long_take.ciede2000((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485)) == pytest.approx(2.0425, abs=1e-4)
    assert long_take.ciede2000((50.0, 0.0, 0.0), (50.0, -1.0, 2.0)) == pytest.approx(2.3669, abs=1e-4)
    white = long_take.srgb_to_lab((255, 255, 255))
    assert white[0] == pytest.approx(100, abs=0.01) and abs(white[1]) < 0.01 and abs(white[2]) < 0.01


def test_colour_drift_is_zero_for_the_same_frame_and_grows_with_a_shift(tmp_path):
    import io

    def png(rgb):
        b = io.BytesIO()
        Image.new("RGB", (480, 832), rgb).save(b, "PNG")
        return b.getvalue()
    drift = long_take.colour_drift([png((180, 120, 90)), png((180, 120, 90)), png((190, 120, 80))])
    assert drift[0] == 0 and drift[1] == 0 and drift[2] > 2


def test_a_take_without_a_designed_voice_is_refused_with_the_reason(conn, policy):
    pol = breadth.test_policy(_paused(policy))
    plan, items = breadth.load_plan(pol, "longtake")
    assert {i.id for i in items} == {"talk_longcat16_90s", "talk_chain_90s"}
    kokoro = breadth._creator(conn, _creator(conn, designed=False)["handle"])
    reason = breadth.skip_reason(pol, items[0], kokoro)
    assert "designed voice" in reason and "mw voice cast" in reason
    designed = breadth._creator(conn, _creator(conn)["handle"])
    assert all(breadth.skip_reason(pol, i, designed) is None for i in items)
    longcat = next(i for i in items if i.take["kind"] == "talk.longcat")
    chain = next(i for i in items if i.take.get("max_segment_seconds"))
    assert longcat.take["script"] == chain.take["script"] and longcat.take["image"] == chain.take["image"]
    with pytest.raises(ValueError):
        breadth.parse_item({"id": "x", "exercises": ["talking"], "take": {"kind": "talk.longcat"}})   # no script


def _finish(conn, job_id, kind: str, uri: str, meta: dict | None = None, frame: int | None = None):
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job_id,))
    conn.execute("INSERT INTO assets (job_id, kind, uri, sha256, bytes, meta) VALUES (%s, %s, %s, 'x', 1, %s)",
                 (job_id, kind, uri, Jsonb(meta or {})))
    if frame is not None:
        conn.execute("INSERT INTO assets (job_id, kind, uri, sha256, bytes, meta) VALUES (%s, 'keyframe', %s, "
                     "'x', 1, %s)", (job_id, uri + ".last.jpg", Jsonb({"frame": frame})))


def _args(pol, run, item, creator, **kw):
    return dict(run=run, batch=f"test-breadth-{run}", item_id=item.id, take=item.take, creator=creator,
                item_batch=f"test-breadth-{run}-{item.id}", anchors_uri="s3://b/r0.png",
                pins={"talk": ["runpod_secure_l40s"], "voice": ["runpod_secure_l40s"]},
                meta={"run": run, "plan": "longtake"}, **kw)


def test_the_chain_shares_one_voice_line_and_continues_from_each_last_frame(conn, policy, monkeypatch):
    pol = breadth.test_policy(_paused(policy))
    _, items = breadth.load_plan(pol, "longtake")
    chain = next(i for i in items if i.take.get("max_segment_seconds"))
    single = next(i for i in items if i.take["kind"] == "talk.longcat")
    cr = breadth._creator(conn, _creator(conn)["handle"])
    assert long_take.advance(conn, pol, **_args(pol, "r9", chain, cr))["state"] == "waiting"
    assert long_take.advance(conn, pol, **_args(pol, "r9", single, cr))["on"] == "voice"
    voices = conn.execute("SELECT id, batch, payload, pools FROM jobs WHERE kind = 'voice.speak.qwen3'").fetchall()
    assert len(voices) == 1 and voices[0]["batch"] == "test-breadth-r9-take-voice"   # one line for both takes
    assert voices[0]["payload"]["ref_uri"] == VOICE["ref_uri"] and len(voices[0]["payload"]["lines"]) > 5
    _finish(conn, voices[0]["id"], "audio", "s3://b/line.wav", {"seconds": 70.0, "lines": LINES})

    made = []
    monkeypatch.setattr(long_take, "finish", lambda conn, policy, **kw: made.append(kw) or {"id": "take"})
    for k in range(3):
        assert long_take.advance(conn, pol, **_args(pol, "r9", chain, cr))["state"] == "waiting"
        seg = conn.execute("SELECT id, kind, batch, pools, payload FROM jobs WHERE idempotency_key = %s",
                           (long_take.segment_key("r9", chain.id, k),)).fetchone()
        assert seg["kind"] == "talk.infinitetalk" and seg["batch"] == f"test-breadth-r9-{chain.id}"
        assert seg["pools"] == ["runpod_secure_l40s"] and seg["payload"]["audio_uri"] == "s3://b/line.wav"
        assert seg["payload"]["audio_window"] == [long_take.windows(LINES, 70.0, 30)[k][x] for x in ("start_s", "end_s")]
        assert seg["payload"]["image_uri"] == ("s3://b/r0.png" if k == 0 else f"s3://b/seg{k - 1}.mp4.last.jpg")
        assert conn.execute("SELECT count(*) n FROM jobs WHERE idempotency_key LIKE %s",
                            (f"breadth:r9:{chain.id}:seg%",)).fetchone()["n"] == k + 1   # one at a time
        _finish(conn, seg["id"], "video", f"s3://b/seg{k}.mp4", {"sync": {"corr": 0.5}}, frame=100)
    assert long_take.advance(conn, pol, **_args(pol, "r9", chain, cr, store=object()))["state"] == "done"
    assert [v["uri"] for v in made[0]["videos"]] == [f"s3://b/seg{k}.mp4" for k in range(3)]

    assert long_take.advance(conn, pol, **_args(pol, "r9", single, cr))["on"].startswith("segment 0")
    one = conn.execute("SELECT kind, payload FROM jobs WHERE idempotency_key = %s",
                       (long_take.segment_key("r9", single.id, 0),)).fetchone()
    assert one["kind"] == "talk.longcat" and "audio_window" not in one["payload"]
    assert one["payload"]["params"]["model"] == "longcat_native"
    assert one["payload"]["script"] == "One. Two. Three. Four."


def test_a_failed_voice_line_fails_the_take(conn, policy):
    pol = breadth.test_policy(_paused(policy))
    _, items = breadth.load_plan(pol, "longtake")
    cr = breadth._creator(conn, _creator(conn)["handle"])
    long_take.advance(conn, pol, **_args(pol, "r8", items[0], cr))
    conn.execute("UPDATE jobs SET status = 'failed', error = 'boom' WHERE kind = 'voice.speak.qwen3'")
    out = long_take.advance(conn, pol, **_args(pol, "r8", items[0], cr))
    assert out["state"] == "failed" and "boom" in out["reason"]


def _clip(path: Path, colour: str, seconds: float = 1.0) -> Path:
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"color=c={colour}:s=160x288:r=25:d={seconds}", "-f", "lavfi", "-i",
                    "anullsrc=r=24000:cl=mono", "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(path)], check=True, capture_output=True)
    return path


def test_the_finished_chain_is_joined_measured_and_reported(conn, policy, store, tmp_path, monkeypatch):
    pol = policy_mod.from_dict(policy_mod.deep_merge(breadth.test_policy(_paused(policy)).body, {
        "breadth": {"takes": {"checkpoints_s": [0, 1, 30]}}}))
    _, items = breadth.load_plan(pol, "longtake")
    chain = next(i for i in items if i.take.get("max_segment_seconds"))
    cr = breadth._creator(conn, _creator(conn)["handle"])
    jobs = [queue.enqueue(conn, "talk.infinitetalk", {}, batch="b") for _ in range(2)]
    videos = [{"uri": store.put_file(_clip(tmp_path / f"{c}.mp4", c), f"t/{c}.mp4").uri, "job_id": j,
               "meta": {"sync": {"corr": corr}}} for c, j, corr in (("red", jobs[0], 0.6), ("blue", jobs[1], 0.4))]
    parts = [{"start_s": 0, "end_s": 1}, {"start_s": 1, "end_s": 2}]
    monkeypatch.setattr(long_take, "identity_at", lambda *a, **kw: {"per_checkpoint": [9, 9, 8], "worst": 8})

    class LLM:
        pass
    row = long_take.finish(conn, pol, run="r7", item_id=chain.id, take=chain.take, creator=cr, videos=videos,
                           parts=parts, audio={"uri": "s3://b/line.wav", "meta": {"seconds": 2.0}}, store=store,
                           llm=LLM(), meta={"run": "r7"})
    asset = conn.execute("SELECT uri, meta FROM assets WHERE id = %s", (row["id"],)).fetchone()
    m = asset["meta"]["take"]
    assert asset["meta"]["role"] == long_take.ROLE and asset["uri"].endswith(f"{chain.id}-take.mp4")
    assert m["segments"] == 2 and m["seconds"] == pytest.approx(2.0, abs=0.2)
    assert m["checkpoints_s"][0] == 0 and m["checkpoints_s"][-1] < m["seconds"]       # 30 s: the last frame
    assert m["delta_e_vs_first"][0] == 0 and m["delta_e_vs_first"][-1] > 20             # red to blue
    assert m["lip_sync_corr"] == [0.6, 0.4] and m["lip_sync_corr_min"] == 0.4
    assert m["identity"]["worst"] == 8
    assert long_take.take_asset(conn, "r7", chain.id)["id"] == row["id"]
    assert long_take.report(conn, "r7", store)[chain.id]["segments"] == 2


def test_the_breadth_run_starts_with_the_shared_voice_line_only(conn, policy, store):
    handle = _creator(conn)["handle"]
    pol = _paused(policy)
    dry = breadth.run(conn, pol, "longtake", creator=handle, dry_run=True)
    assert not dry["skipped"] and _jobs(conn) == 0
    est = {i["id"]: i["estimate"]["jobs"] for i in dry["items"]}
    assert [j["kind"] for j in est["talk_longcat16_90s"]] == ["voice.speak.qwen3", "talk.longcat"]
    assert [j["kind"] for j in est["talk_chain_90s"]].count("talk.infinitetalk") >= 3
    out = breadth.run(conn, pol, "longtake", creator=handle, store=store, once=True, sleep=lambda s: None)
    assert out["takes"] == {"talk_longcat16_90s": {"state": "waiting", "on": "voice"},
                            "talk_chain_90s": {"state": "waiting", "on": "voice"}}
    rows = conn.execute("SELECT kind, batch, priority FROM jobs").fetchall()
    assert [r["kind"] for r in rows] == ["voice.speak.qwen3"]
    assert rows[0]["batch"] == f"{out['batch']}-take-voice" and rows[0]["priority"] == breadth.urgent_priority(
        breadth.test_policy(pol))
    assert not out["done"]


def _jobs(conn):
    return conn.execute("SELECT count(*) AS n FROM jobs").fetchone()["n"]


def test_a_longcat_failure_names_the_childs_exception_not_torchruns_banner():
    # 2026-11-25: every recorded LongCat error was torchrun's ChildFailedError banner, the cause cut off.
    from manyworlds.worker.handlers.talk_longcat import child_error
    out = ("Traceback (most recent call last):\n  File \"run.py\", line 9\n"
           "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n"
           "[rank0]: something\n" + "x\n" * 200 +
           "torch.distributed.elastic.multiprocessing.errors.ChildFailedError: \n  error_file: <N/A>\n")
    msg = child_error(out)
    assert "CUDA out of memory" in msg and "ChildFailedError" not in msg
    assert child_error("nothing useful") == "no exception line in the output"
