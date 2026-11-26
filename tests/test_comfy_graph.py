"""Named ComfyUI graphs (job kind comfy.graph, worker/handlers/comfy_graph.py, policy comfy_graphs) and the candidate
benchmark that runs them (breadth plan `candidates`, manyworlds.long_take). ComfyUI, the GPU and the weights are
stood in for: what is tested is that every shipped graph fills from a job, that a job's values land where its sidecar
says, the benchmark numbers come back, unknown graphs and wrong machines are refused, and the plan reuses
perf_compare's voice lines, pins each candidate to its card and prices itself."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from psycopg.types.json import Jsonb

from manyworlds import breadth, long_take, queue
from manyworlds.worker import comfy, runner, smoke
from manyworlds.worker.handlers import comfy_graph, talk_comfy
from manyworlds.worker.runner import NonRetryable

from test_breadth import _paused
from test_review import BIBLE

GRAPHS = ("echomimic_v3_flash", "ltx25_a2v", "ltx25_i2v_av")


def _spec(raw_policy, name):
    return raw_policy.get("comfy_graphs", name)


# ------------------------------------------------------------------------------------------------ the graphs

def test_every_shipped_graph_has_its_files_and_fills_from_a_job(raw_policy):
    assert set(GRAPHS) <= set(raw_policy.get("comfy_graphs"))
    assert "named graphs fill" in smoke._check_named_graphs(None, raw_policy.get("comfy_graphs"))
    for name in GRAPHS:
        graph, side = comfy_graph.load(name)
        assert side["source"]["repo"].startswith("https://github.com/") and side["source"]["commit"]
        assert any(n["class_type"] == "SaveVideo" for n in graph.values())
        for target in (t for ts in side["inputs"].values() for t in ts):   # every target is a real node input
            node, field = target.split(".", 1)
            assert field in graph[node]["inputs"], (name, target)


def test_a_job_s_values_land_where_the_sidecar_says(raw_policy):
    params = {**_spec(raw_policy, "ltx25_a2v")["params"]}
    graph, side = comfy_graph.load("ltx25_a2v")
    values = comfy_graph.values_for(image="i.png", audio="a.wav", prompt="she talks", negative="blurry", seed=7,
                                    seconds=3.2, params=params)
    g = comfy_graph.fill(graph, side["inputs"], values)
    assert values["frames"] == 81 and (values["frames"] - 1) % 8 == 0          # 3.2 s at 24 fps: 8k + 1
    assert values["seconds"] == pytest.approx(80 / 24, abs=0.001)               # the padded line covers the clip
    assert g["empty_video"]["inputs"]["width"] == params["width"] // 2 and g["empty_video"]["inputs"]["length"] == 81
    assert g["positive"]["inputs"]["text"] == "she talks" and g["negative"]["inputs"]["text"] == "blurry"
    assert g["audio"]["inputs"]["audio"] == "a.wav" and g["trim"]["inputs"]["duration"] == values["seconds"]
    assert g["concat_1"]["inputs"]["audio_latent"] == ["audio_latent", 0]       # our line, kept fixed
    assert g["keep_audio"]["inputs"]["value"] == 0.0
    assert graph["positive"]["inputs"]["text"] != "she talks"                   # the shipped graph is untouched
    with pytest.raises(NonRetryable, match="no value"):
        comfy_graph.fill(graph, side["inputs"], {**values, "seconds": None})
    with pytest.raises(NonRetryable, match="no input"):
        comfy_graph.fill(graph, {"prompt": ["positive.nope"]}, values)


def test_frames_follow_each_graph_s_rule(raw_policy):
    echo = _spec(raw_policy, "echomimic_v3_flash")["params"]
    assert comfy_graph.frames_for(3.0, echo) == 75
    assert comfy_graph.frames_for(3.01, echo) == 76
    ltx = _spec(raw_policy, "ltx25_a2v")["params"]
    assert comfy_graph.frames_for(5.0, ltx) == 121 and comfy_graph.frames_for(4.99, ltx) == 121


def test_unknown_graphs_and_wrong_machines_are_refused(raw_policy, monkeypatch):
    with pytest.raises(NonRetryable, match="unknown graph"):
        comfy_graph.spec_for({"graph": "no_such_graph"}, {"params": {}})
    with pytest.raises(NonRetryable, match="name like"):
        comfy_graph.spec_for({"graph": "../etc/passwd"})
    monkeypatch.setattr(comfy_graph, "_policy_spec", lambda name: None)
    with pytest.raises(NonRetryable, match="not in policy"):
        comfy_graph.spec_for({"graph": "ltx25_a2v"})
    name, spec = comfy_graph.spec_for({"graph": "ltx25_a2v", "graph_spec": {"params": {"fps": 1}}})
    assert spec == {"params": {"fps": 1}}                                       # the job's spec wins
    ltx = _spec(raw_policy, "ltx25_a2v")
    with pytest.raises(NonRetryable, match="needs 44 GB"):
        comfy_graph.check_machine(ltx, 25.3, env={"HF_TOKEN": "t"})
    with pytest.raises(NonRetryable, match="HF_TOKEN"):
        comfy_graph.check_machine(ltx, 47.7, env={})
    comfy_graph.check_machine(ltx, 47.7, env={"HF_TOKEN": "t"})
    assert "comfy.graph" in runner.handlers_for(["comfy.graph"])
    assert raw_policy.family("comfy.graph") == "comfy"


# ------------------------------------------------------------------------------------------------ the handler

class Ctx:
    def __init__(self, work: Path):
        self.job = SimpleNamespace(id="job-1")
        self.workdir = work

    def fetch(self, uri: str, name: str) -> Path:
        p = self.workdir / "inputs" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"input")
        return p


@pytest.fixture
def comfy_stub(tmp_path, monkeypatch):
    seen: dict = {"joined": [], "padded": []}
    monkeypatch.setattr(comfy, "COMFY_DIR", tmp_path / "comfy")
    monkeypatch.setenv("MW_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(comfy, "write_model_paths", lambda models, extra=(): seen.setdefault("folders", list(extra)))
    monkeypatch.setattr(comfy, "ensure_weights", lambda models, weights, **k: seen.setdefault("weights", weights))
    monkeypatch.setattr(comfy, "link_models", lambda models, names: seen.setdefault("links", list(names)))
    monkeypatch.setattr(comfy, "start", lambda **k: None)
    monkeypatch.setattr(comfy, "_get", lambda path, timeout=10: {"devices": [{"vram_total": 48e9,
                                                                              "vram_free": 20e9}]})

    def run_all(g, timeout_s):
        seen["graph"] = g
        p = tmp_path / "comfy" / "output" / "mw_graph_00001.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"video")
        return [p]
    monkeypatch.setattr(comfy, "run_all", run_all)
    monkeypatch.setattr(comfy_graph, "_vram_total_gb", lambda: 47.7)
    monkeypatch.setattr(comfy_graph, "_ready", {})
    monkeypatch.setattr(comfy_graph, "_rendered", set())
    monkeypatch.setattr(talk_comfy, "_prepare_image", lambda src, dest, w, h, **k: dest.write_bytes(b"png"))
    monkeypatch.setattr(talk_comfy, "join_windows",
                        lambda clips, audio, seconds, dest: seen["joined"].append(seconds) or dest.write_bytes(b"v"))
    monkeypatch.setattr("manyworlds.worker.handlers.talk_infinitetalk.audio_seconds", lambda p: 3.2)
    monkeypatch.setattr(comfy_graph, "pad_audio",
                        lambda src, dest, s: seen["padded"].append(s) or dest.write_bytes(b"wav") or dest)
    monkeypatch.setattr(comfy_graph, "speech_of", lambda clip, dest: dest.write_bytes(b"wav") or dest)
    monkeypatch.setattr(comfy_graph, "extract_keyframes", lambda clip, work, n: [])
    monkeypatch.setattr(comfy_graph.crops, "crops_in_venv", lambda *a, **k: [])
    monkeypatch.setattr(comfy_graph.lipsync, "score_in_venv", lambda clip, audio, **k: {"corr": 0.55})
    monkeypatch.setenv("HF_TOKEN", "test-token")
    return seen


def test_an_audio_graph_renders_with_our_line_and_reports_its_numbers(raw_policy, comfy_stub, tmp_path):
    spec = _spec(raw_policy, "ltx25_a2v")
    payload = {"graph": "ltx25_a2v", "graph_spec": spec, "image_uri": "s3://b/r0.png", "audio_uri": "s3://b/l.wav",
               "prompt": "she talks", "seed": 5, "keyframes": 4}
    out = comfy_graph.generate(payload, Ctx(tmp_path / "w"))
    g = comfy_stub["graph"]
    assert g["image"]["inputs"]["image"] == "mw_job-1.png" and g["audio"]["inputs"]["audio"] == "mw_job-1.wav"
    assert g["noise_1"]["inputs"]["noise_seed"] == 5
    assert g["negative"]["inputs"]["text"] == spec["params"]["negative_prompt"]   # the graph's own default
    assert comfy_stub["padded"] == [pytest.approx(80 / 24, abs=0.001)]           # padded to the clip
    assert comfy_stub["joined"] == [3.2]                                         # our line, cut to its length
    assert set(comfy_stub["folders"]) == set(spec["weights"]) and comfy_stub["links"] == []
    video = out.outputs[0]
    assert video.kind == "video" and video.meta["sync"] == {"corr": 0.55} and video.meta["graph"] == "ltx25_a2v"
    bench = video.meta["bench"]
    assert bench["vram"]["peak_gb"] == 28.0 and bench["vram"]["total_gb"] == 48.0 and bench["cold"] is True
    assert bench["frames"] == 81 and out.meta["bench"] == bench and out.recipe["source"]["commit"]
    again = comfy_graph.generate(payload, Ctx(tmp_path / "w2"))
    assert again.meta["bench"]["cold"] is False                                  # models loaded: warm


def test_a_graph_with_its_own_sound_needs_a_length_and_keeps_its_audio(raw_policy, comfy_stub, tmp_path):
    spec = _spec(raw_policy, "ltx25_i2v_av")
    payload = {"graph": "ltx25_i2v_av", "graph_spec": spec, "image_uri": "s3://b/r0.png", "prompt": "she says hi",
               "keyframes": 4}
    with pytest.raises(NonRetryable, match="params.seconds"):
        comfy_graph.generate(payload, Ctx(tmp_path / "w"))
    out = comfy_graph.generate({**payload, "params": {"seconds": 2.4}}, Ctx(tmp_path / "w"))
    assert comfy_stub["joined"] == [] and comfy_stub["padded"] == []
    assert comfy_stub["graph"]["empty_audio"]["inputs"]["frames_number"] == 65   # 2.4 s at 24 fps -> 8k + 1
    assert out.outputs[0].meta["native_audio"] and out.outputs[0].meta["sync"] == {"corr": 0.55}


def test_echomimic_reaches_its_pack_s_own_model_folder(raw_policy, comfy_stub, tmp_path):
    spec = _spec(raw_policy, "echomimic_v3_flash")
    out = comfy_graph.generate({"graph": "echomimic_v3_flash", "graph_spec": spec, "image_uri": "s3://b/r0.png",
                                "audio_uri": "s3://b/l.wav", "prompt": "she talks", "keyframes": 4},
                               Ctx(tmp_path / "w"))
    g = comfy_stub["graph"]
    assert comfy_stub["links"] == ["echo_mimic"] and g["model"]["inputs"]["version"] == "V3_flash"
    assert (g["predata"]["inputs"]["width"], g["predata"]["inputs"]["height"]) == (576, 1024)   # 9:16
    assert g["predata"]["inputs"]["length"] == 80 and g["sampler"]["inputs"]["steps"] == 8
    assert comfy_stub["padded"] == [] and comfy_stub["joined"] == [3.2]
    assert out.outputs[0].meta["width"] == 576


def test_vram_peak_keeps_the_most_in_use():
    seq = iter([10e9, 30e9, 5e9, 40e9])
    v = comfy.VramPeak(60, stats=lambda: {"devices": [{"vram_total": 48e9, "vram_free": next(seq)}]})
    for _ in range(4):
        v.sample()
    assert v.result() == {"peak_gb": 43.0, "total_gb": 48.0, "samples": 4}
    broken = comfy.VramPeak(60, stats=lambda: {})
    broken.sample()
    assert broken.result()["peak_gb"] is None


def test_a_pack_s_model_folder_becomes_a_link_into_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy, "COMFY_DIR", tmp_path / "comfy")
    (tmp_path / "comfy" / "models" / "echo_mimic" / "transformer").mkdir(parents=True)   # made by the pack's import
    comfy.link_models(tmp_path / "cache", ["echo_mimic"])
    link = tmp_path / "comfy" / "models" / "echo_mimic"
    assert link.is_symlink() and link.resolve() == (tmp_path / "cache" / "echo_mimic").resolve()
    assert comfy.link_models(tmp_path / "cache", ["echo_mimic"]) == []           # already linked
    link.unlink()
    (link / "x").mkdir(parents=True)
    (link / "x" / "w.bin").write_bytes(b"w")
    with pytest.raises(RuntimeError, match="holds files"):
        comfy.link_models(tmp_path / "cache", ["echo_mimic"])


def test_a_new_model_folder_restarts_a_running_server(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy, "COMFY_DIR", tmp_path)
    stops = []
    monkeypatch.setattr(comfy, "stop", lambda: stops.append(1) or comfy._server.update(proc=None))

    class Proc:
        def poll(self):
            return None
    monkeypatch.setattr(comfy, "_server", {"proc": Proc(), "started_with": set()})
    comfy.write_model_paths(tmp_path / "m")
    comfy._server["started_with"] = set(comfy._server["folders"])
    comfy.start()                                                                # same folders: still running
    assert stops == []
    comfy.write_model_paths(tmp_path / "m", ["latent_upscale_models", "echo_mimic/transformer"])
    assert "latent_upscale_models" in (tmp_path / "extra_model_paths.yaml").read_text()
    assert "echo_mimic/transformer" not in (tmp_path / "extra_model_paths.yaml").read_text()
    monkeypatch.setattr(comfy.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("started")))
    monkeypatch.setattr(comfy, "_server", {**comfy._server, "proc": Proc()})
    with pytest.raises(RuntimeError, match="started"):
        comfy.start()
    assert stops == [1]


# ------------------------------------------------------------------------------------------------ longcat_fast

def test_longcat_fast_applies_the_portable_speed_ups_to_the_wrapper_graph(raw_policy):
    p = {**raw_policy.get("generation", "talk"), **raw_policy.get("generation", "talk_profiles", "longcat_fast"),
         "vae_file": "lightvaew2_1.safetensors"}
    g = comfy.graph_longcat(image="i.png", audio="a.wav", prompt="p", frames=93, seed=1, params=p, blocks_to_swap=0)
    assert g["scheduler"]["inputs"]["steps"] == 6
    assert g["model"]["inputs"]["quantization"] == "fp8_e4m3fn_scaled_fast"
    assert g["text"]["inputs"]["device"] == "cpu" and g["vae"]["inputs"]["model_name"] == "lightvaew2_1.safetensors"
    assert "compile" in g and g["lora"]["inputs"]["merge_loras"] is False
    stock = comfy.graph_longcat(image="i.png", audio="a.wav", prompt="p", frames=93, seed=1,
                                params={**p, "vae_file": None, "text_device": None, "compile": False},
                                blocks_to_swap=0)
    assert stock["vae"]["inputs"]["model_name"] == comfy.VAE


def test_longcat_fast_downloads_its_lightvae_beside_the_wan_vae(raw_policy, tmp_path, monkeypatch):
    from test_comfy_handlers import TALK, Ctx as TalkCtx
    got = {}
    monkeypatch.setattr(comfy, "COMFY_DIR", tmp_path / "comfy")
    monkeypatch.setenv("MW_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(comfy, "write_model_paths", lambda *a, **k: None)
    monkeypatch.setattr(comfy, "ensure_weights", lambda models, weights, **k: got.setdefault("weights", weights))
    monkeypatch.setattr(talk_comfy, "_ready", {})

    class Stop(Exception):
        pass
    monkeypatch.setattr(comfy, "start", lambda **k: (_ for _ in ()).throw(Stop()))
    params = {**TALK["params"], **raw_policy.get("generation", "talk_profiles", "longcat_fast")}
    with pytest.raises(Stop):
        talk_comfy.generate({**TALK, "params": params}, TalkCtx(tmp_path))
    assert ("lightx2v/Autoencoders", "lightvaew2_1.safetensors") in got["weights"]["vae"]
    assert comfy.WEIGHTS["vae"][0] in got["weights"]["vae"]


# ------------------------------------------------------------------------------------------------ the plan

def _creator(conn):
    return conn.execute("INSERT INTO creators (handle, bible, anchors) VALUES (%s, %s, %s) RETURNING id, handle, "
                        "anchors", (f"c{uuid.uuid4().hex[:6]}", Jsonb(BIBLE),
                                    Jsonb({"reference_images": [{"uri": "s3://b/r0.png"}]}))).fetchone()


def _perf_voice(conn, text: str, uri: str, key: str = "shared:perf-compare:voice:0:0"):
    j = queue.enqueue(conn, "voice.speak.qwen3", {"lines": [{"text": t} for t in text.split("|")]}, batch="post-x",
                      idempotency_key=key)
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (j,))
    conn.execute("INSERT INTO assets (job_id, kind, uri, sha256, bytes, meta) VALUES (%s, 'audio', %s, 'x', 1, %s)",
                 (j, uri, Jsonb({"seconds": 3.2})))


def test_the_candidates_plan_pins_each_model_to_its_card(policy, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    pol = breadth.test_policy(_paused(policy))
    _, items = breadth.load_plan(pol, "candidates")
    by = {i.id: i.take for i in items}
    assert {t.get("graph") or t.get("profile") for t in by.values()} == {*GRAPHS, "longcat_fast"}
    assert by["echo_natural"]["pools"][0].endswith("4090") and by["ltx_a2v_natural"]["pools"] == ["runpod_secure_l40s"]
    assert by["longcat_fast_natural"]["pools"][0].endswith("5090")
    for t in by.values():
        assert all(p in pol.get("gpu_pools") for p in t["pools"])
    scripts = {t["script"] for t in by.values()}
    perf = [ln["text"] for s in pol.get("breadth", "plans", "perf_compare", "items")[0]["body"]["shots"]
            for ln in s["performance"]["lines"]]
    assert len(scripts) == 3 and scripts <= set(perf)                        # perf_compare's own lines
    assert by["ltx_i2v_av_lifestyle"]["audio"] == "none" and by["ltx_i2v_av_lifestyle"]["script"] in \
        by["ltx_i2v_av_lifestyle"]["prompt"]
    assert all(t["audio"] == "@shared:perf-compare" for k, t in by.items() if k != "ltx_i2v_av_lifestyle")


def test_a_gated_graph_is_skipped_without_its_token(policy):
    pol = breadth.test_policy(_paused(policy))
    _, items = breadth.load_plan(pol, "candidates")
    ltx = next(i for i in items if i.take.get("graph") == "ltx25_a2v")
    assert "HF_TOKEN" in long_take.skip_reason(pol, ltx.take, {"handle": "m", "anchors": {}}, env={})
    assert long_take.skip_reason(pol, ltx.take, {"handle": "m", "anchors": {}}, env={"HF_TOKEN": "t"}) is None
    with pytest.raises(ValueError, match="take.audio"):
        long_take.validate("x", {**ltx.take, "audio": "perf-compare"})
    with pytest.raises(ValueError, match="take.graph"):
        long_take.validate("x", {**ltx.take, "graph": None})


def test_a_reused_line_queues_the_graph_on_its_pool_and_voices_nothing(conn, policy, monkeypatch):
    pol = breadth.test_policy(_paused(policy))
    _, items = breadth.load_plan(pol, "candidates")
    echo = next(i for i in items if i.id == "echo_natural")
    cr = _creator(conn)
    args = dict(run="c1", batch="test-breadth-c1", item_id=echo.id, take=echo.take, creator=cr,
                item_batch=f"test-breadth-c1-{echo.id}", anchors_uri="s3://b/r0.png",
                pins={"comfy": ["runpod_secure_l40s"]}, meta={"run": "c1", "plan": "candidates"})
    out = long_take.advance(conn, pol, **args)
    assert out["state"] == "failed" and "perf-compare" in out["reason"]       # perf_compare hasn't voiced it
    _perf_voice(conn, "Some other line.", "s3://b/other.wav", key="shared:perf-compare:voice:1:0")
    _perf_voice(conn, "Okay, story time.|Five to two, I'm closing up.", "s3://b/line0.wav")
    assert long_take.advance(conn, pol, **args)["state"] == "waiting"
    job = conn.execute("SELECT kind, pools, payload FROM jobs WHERE idempotency_key = %s",
                       (long_take.segment_key("c1", echo.id, 0),)).fetchone()
    assert job["kind"] == "comfy.graph" and job["pools"] == echo.take["pools"]
    p = job["payload"]
    assert p["graph"] == "echomimic_v3_flash" and p["graph_spec"] == pol.get("comfy_graphs", "echomimic_v3_flash")
    assert p["audio_uri"] == "s3://b/line0.wav" and p["params"]["width"] == 576
    assert conn.execute("SELECT count(*) n FROM jobs WHERE kind LIKE 'voice.speak%' AND batch LIKE 'test-%'"
                        ).fetchone()["n"] == 0


def test_a_take_with_the_model_s_own_sound_gets_a_length_not_a_line(conn, policy):
    pol = breadth.test_policy(_paused(policy))
    _, items = breadth.load_plan(pol, "candidates")
    av = next(i for i in items if i.id == "ltx_i2v_av_lifestyle")
    cr = _creator(conn)
    long_take.advance(conn, pol, run="c2", batch="test-breadth-c2", item_id=av.id, take=av.take, creator=cr,
                      item_batch=f"test-breadth-c2-{av.id}", anchors_uri="s3://b/r0.png", pins={},
                      meta={"run": "c2"})
    p = conn.execute("SELECT payload FROM jobs WHERE idempotency_key = %s",
                     (long_take.segment_key("c2", av.id, 0),)).fetchone()["payload"]
    assert "audio_uri" not in p and p["params"]["seconds"] == long_take.script_seconds(pol, av.take)
    assert p["params"]["native_audio"] is True


def test_the_dry_run_prices_each_candidate_on_its_own_machine(conn, policy, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    handle = _creator(conn)["handle"]
    dry = breadth.run(conn, _paused(policy), "candidates", creator=handle, dry_run=True)
    assert not dry["skipped"], dry["skipped"]
    items = {i["id"]: i for i in dry["items"]}
    assert items["echo_natural"]["audio"].startswith("MISSING")
    echo = items["echo_natural"]["estimate"]["jobs"]
    assert [j["kind"] for j in echo] == ["comfy.graph"] and echo[0]["pool"] == "vast_4090"
    assert echo[0]["source"].startswith("policy (comfy.graph:echomimic_v3_flash")
    pools = {m["pool"]: m for m in dry["machines"]}
    assert set(pools) == {"vast_4090", "runpod_secure_l40s", "vast_5090"}
    assert pools["runpod_secure_l40s"]["weights"] == ["ltx25"] and pools["runpod_secure_l40s"]["download_gb"] == 39.7
    assert pools["vast_5090"]["download_gb"] == 30.2
    assert 0 < dry["estimated_usd"] < pol_cap(policy)
    assert conn.execute("SELECT count(*) n FROM jobs").fetchone()["n"] == 0


def pol_cap(policy) -> float:
    return float(breadth.test_policy(policy).get("breadth", "plans", "candidates", "max_usd"))


def test_the_graphs_ship_in_the_package_and_the_image():
    root = Path(__file__).resolve().parents[1]
    assert "worker/graphs/*.json" in (root / "pyproject.toml").read_text()
    dockerfile = root / "docker" / "worker-talkcomfy.Dockerfile"
    if not dockerfile.exists():   # the local test container mounts the package, not docker/ (CI has the checkout)
        pytest.skip("docker/ not in this checkout")
    docker = dockerfile.read_text()
    assert "ComfyUI_EchoMimic" in docker and "ECHOMIMIC_REF=" in docker
    commit = docker.split("ECHOMIMIC_REF=", 1)[1].split()[0]
    assert commit == comfy_graph.load("echomimic_v3_flash")[1]["source"]["commit"]
    for name in GRAPHS:
        json.loads((comfy_graph.GRAPHS_DIR / f"{name}.json").read_text())


def test_a_reused_line_is_found_under_the_run_scoped_shared_key(conn):
    # 2026-11-25: the breadth driver keys a run's shared voice lines shared:<key>:<run>:voice:..., so the
    # benchmark found none of the comparison's nine finished lines.
    _perf_voice(conn, "Honestly? Best night of the week.", "s3://b/l8.wav",
                key="shared:perf-compare:perf-11242157:voice:8:0")
    found = long_take.shared_audio(conn, "perf-compare", "Honestly? Best night of the week.")
    assert found and found["uri"] == "s3://b/l8.wav"
