"""The ComfyUI InfiniteTalk graph (worker/comfy.py): wired as Kijai's example, valid against ComfyUI's catalogue.
The real catalogue check runs in the talk image's smoke test; here the graph's shape and the checker itself."""

from __future__ import annotations

from PIL import Image

from manyworlds.worker import comfy
from manyworlds.worker.handlers import talk_comfy


def _g(**p):
    return comfy.graph(image="a.png", audio="a.wav", prompt="she talks", frames=150, seed=7,
                       params={"width": 480, "height": 832, "steps": 6, **p}, blocks_to_swap=0)


def test_the_graph_wires_the_fast_stack_together():
    g = _g()
    model = g["model"]["inputs"]
    assert model["quantization"] == "fp8_e4m3fn_scaled" and model["attention_mode"] == "sageattn"
    assert model["lora"] == ["lora", 0] and model["multitalk_model"] == ["adapter", 0]
    assert g["lora"]["inputs"]["lora"].endswith("cfg_step_distill_rank64_bf16.safetensors")
    s = g["sampler"]["inputs"]
    assert (s["steps"], s["cfg"], s["scheduler"]) == (6, 1.0, "dpm++_sde")      # Kijai's example settings
    assert s["multitalk_embeds"] == ["embeds", 0] and g["embeds"]["inputs"]["num_frames"] == 150
    assert g["video"]["inputs"]["audio"] == ["audio", 0]                      # the voice is muxed into the clip
    # every link points at a node in the graph
    for node in g.values():
        for v in node["inputs"].values():
            if isinstance(v, list):
                assert v[0] in g


def test_the_checker_finds_unknown_nodes_and_missing_or_unknown_inputs():
    g = {"a": {"class_type": "Known", "inputs": {"x": 1, "typo": 2}}, "b": {"class_type": "Nope", "inputs": {}}}
    info = {"Known": {"input": {"required": {"x": [], "y": []}, "optional": {"z": []}}}}
    problems = comfy.missing_inputs(g, info)
    assert any("no node Nope" in p for p in problems)
    assert any("missing required input y" in p for p in problems)
    assert any("unknown input typo" in p for p in problems)


def test_the_start_photo_is_cover_cropped_to_the_talking_shot(tmp_path):
    src, dest = tmp_path / "s.png", tmp_path / "d.png"
    Image.new("RGB", (896, 1152), (200, 100, 50)).save(src)          # a 4:5 photo
    talk_comfy._prepare_image(src, dest, 480, 832)                      # into a 9:16 shot
    assert Image.open(dest).size == (480, 832)


def test_model_folders_point_at_the_weights_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy, "COMFY_DIR", tmp_path)
    cfg = comfy.write_model_paths(tmp_path / "cache").read_text()
    assert f"base_path: {tmp_path / 'cache'}" in cfg and "diffusion_models: diffusion_models" in cfg


def test_the_story_clip_graph_runs_the_two_experts_in_turn():
    g = comfy.graph_i2v(image="a.png", prompt="steam rises", seed=3, params={"width": 480, "height": 832},
                        blocks_to_swap=0)
    hi, lo = g["sample_high"]["inputs"], g["sample_low"]["inputs"]
    assert hi["model"] == ["high_lora", 0] and lo["model"] == ["low_lora", 0]
    assert (hi["start_step"], hi["end_step"], lo["start_step"], lo["end_step"]) == (0, 3, 3, -1)
    assert lo["samples"] == ["sample_high", 0] and hi["cfg"] == ["cfg_schedule", 0] and lo["cfg"] == 1.0
    assert g["high_lora_pick"]["inputs"]["strength"] == 3.0 and g["low_lora_pick"]["inputs"]["strength"] == 1.0
    assert g["high_model"]["inputs"]["model"].endswith("HIGH_fp8_e4m3fn_scaled_KJ.safetensors")
    for node in g.values():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g


def test_story_clip_settings_reach_a_comfy_job(conn, policy):
    from manyworlds import bench
    batch = bench.enqueue(conn, policy, "comfy.wan_i2v", 1, extra={"image_uri": "s3://b/x.png"})
    p = conn.execute("SELECT payload FROM jobs WHERE batch = %s", (batch,)).fetchone()["payload"]
    assert p["params"]["width"] == policy.get("generation", "comfy_video", "width") and p["keyframes"] == 4


def test_an_execution_error_names_the_node_and_exception_without_the_inputs():
    status = {"status_str": "error", "messages": [["execution_start", {}], ["execution_error", {
        "node_id": "sampler", "node_type": "WanVideoSampler", "exception_type": "torch.OutOfMemoryError",
        "exception_message": "CUDA out of memory", "traceback": ["  File x\n", "OutOfMemoryError\n"],
        "current_inputs": {"model": "x" * 50000}}]]}
    msg = comfy.execution_error(status)
    assert "WanVideoSampler" in msg and "OutOfMemoryError: CUDA out of memory" in msg and len(msg) < 2500


def test_weights_already_complete_are_not_downloaded_again(tmp_path):
    from manyworlds.worker import comfy
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae" / "v.safetensors").write_bytes(b"x" * 10)
    calls = []
    comfy.ensure_weights(tmp_path, {"vae": [("r", "v.safetensors")]}, download=lambda *a, **k: calls.append(a),
                         remote_size=lambda repo, name, token: 10)
    assert calls == [] and not (tmp_path / ".mw-download.lock").exists()


def test_a_broken_download_record_is_cleared_and_the_file_fetched_again(tmp_path):
    """2026-11-22: two machines on one volume left records huggingface_hub crashed on (UnboundLocalError)."""
    from manyworlds.worker import comfy
    records = tmp_path / "diffusion_models" / ".cache" / "huggingface" / "download" / "I2V"
    records.mkdir(parents=True)
    (records / "h.safetensors.metadata").write_text("broken")
    (tmp_path / "diffusion_models" / "I2V").mkdir()
    (tmp_path / "diffusion_models" / "I2V" / "h.safetensors").write_bytes(b"part")
    calls = []

    def download(repo, name, local_dir, token, force_download=False):
        calls.append(force_download)
        if not force_download:
            raise UnboundLocalError("cannot access local variable 'metadata'")
        (tmp_path / "diffusion_models" / name).write_bytes(b"x" * 10)

    comfy.ensure_weights(tmp_path, {"diffusion_models": [("r", "I2V/h.safetensors")]}, download=download,
                         remote_size=lambda repo, name, token: 10)
    assert calls == [False, True] and not (records / "h.safetensors.metadata").exists()
    assert (tmp_path / "diffusion_models" / "I2V" / "h.safetensors").stat().st_size == 10
    assert not (tmp_path / ".mw-download.lock").exists()


def test_a_full_volume_is_pruned_of_weights_the_worker_no_longer_uses(tmp_path):
    """2026-11-22: the research script's weights filled the 150 GB volume (Disk quota exceeded)."""
    from manyworlds.worker import comfy
    root = tmp_path / "mw"
    (root / "Wan-AI" / "Wan2.1-I2V-14B-480P").mkdir(parents=True)
    (root / "comfy" / "vae").mkdir(parents=True)
    (root / "comfy" / "vae" / "x.safetensors.incomplete").write_bytes(b"part")
    (root / "comfy" / "vae" / "keep.safetensors").write_bytes(b"x" * 10)
    comfy.ensure_weights(root / "comfy", {"vae": [("r", "new.safetensors")]}, prune=True,
                         download=lambda repo, name, local_dir, token, **k: (root / "comfy" / "vae" / name).write_bytes(b"x"),
                         remote_size=lambda repo, name, token: None)
    assert sorted(p.name for p in root.iterdir()) == ["comfy"]
    assert sorted(p.name for p in (root / "comfy" / "vae").iterdir()) == ["keep.safetensors", "new.safetensors"]


def test_a_full_volume_is_pruned_before_the_lock_and_a_lock_that_cannot_be_written_is_not_left_behind(tmp_path, monkeypatch):
    """2026-11-22, image v7: on the full volume even the lock file could not be written (Disk quota exceeded), so
    pruning inside the lock never ran, and the half-written lock would have blocked every machine for two hours."""
    import errno
    import pytest
    root = tmp_path / "mw"
    (root / "old-research").mkdir(parents=True)
    (root / "comfy").mkdir()
    real_write = comfy.os.write
    lock_fds = []
    real_open = comfy.os.open
    def opened(path, *a, **k):
        fd = real_open(path, *a, **k)
        if str(path).endswith(".mw-download.lock"):
            lock_fds.append(fd)
        return fd
    def full(fd, data):
        if fd in lock_fds:
            comfy.os.close(fd)
            raise OSError(errno.EDQUOT, "Disk quota exceeded")
        return real_write(fd, data)
    monkeypatch.setattr(comfy.os, "open", opened)
    monkeypatch.setattr(comfy.os, "write", full)
    with pytest.raises(OSError):
        comfy.ensure_weights(root / "comfy", {"vae": [("r", "new.safetensors")]}, prune=True,
                             download=lambda *a, **k: None, remote_size=lambda repo, name, token: None)
    assert not (root / "old-research").exists()
    assert not (root / "comfy" / ".mw-download.lock").exists()
    monkeypatch.setattr(comfy.os, "write", real_write)


def test_comfyui_exiting_at_start_reports_the_end_of_its_log(tmp_path, monkeypatch):
    import pytest
    log = tmp_path / "comfy.log"
    monkeypatch.setenv("MW_COMFY_LOG", str(log))
    class Dead:
        returncode = 1
        def __init__(self, *a, stdout=None, **k):
            stdout.write(b"RuntimeError: CUDA driver version is insufficient\n"); stdout.flush()
        def poll(self):
            return 1
    monkeypatch.setattr(comfy.subprocess, "Popen", Dead)
    monkeypatch.setitem(comfy._server, "proc", None)
    with pytest.raises(RuntimeError, match="CUDA driver version is insufficient"):
        comfy.start(wait_s=5)
    comfy._server["proc"] = None


def test_torch_compile_is_wired_into_every_model_loader_only_when_asked():
    from manyworlds.worker import comfy
    plain = comfy.graph_i2v(image="a.png", prompt="p", seed=1, params={"width": 480, "height": 832}, blocks_to_swap=0)
    assert "compile" not in plain
    g = comfy.graph_i2v(image="a.png", prompt="p", seed=1, params={"width": 480, "height": 832, "compile": True}, blocks_to_swap=0)
    assert g["high_model"]["inputs"]["compile_args"] == ["compile", 0] == g["low_model"]["inputs"]["compile_args"]
    t = comfy.graph(image="a.png", audio="a.wav", prompt="p", frames=81, seed=1,
                    params={"width": 480, "height": 832, "steps": 6, "compile": True}, blocks_to_swap=0)
    assert t["compile"]["class_type"] == "WanVideoTorchCompileSettings" and t["model"]["inputs"]["compile_args"]
