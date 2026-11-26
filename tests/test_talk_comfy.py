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


def test_a_photo_already_at_the_frames_aspect_is_scaled_whole_not_cropped(tmp_path):
    # the policy's 9:16 scene photo (generation.image_by_format.video) into the hd720 talk frame: same aspect
    assert talk_comfy.fit_box(864, 1536, 720, 1280, 0.01) is None
    # into Wan's 480x832 bucket (2.6% off 9:16): beyond a 1% stretch, so cover-scaled; only ~2.5% of the height goes
    assert talk_comfy.fit_box(864, 1536, 480, 832, 0.01) == (480, 854)
    assert talk_comfy.fit_box(864, 1536, 480, 832, 0.03) is None      # within a looser tolerance: scaled whole
    assert talk_comfy.fit_box(896, 1152, 480, 832, 0.01) == (648, 832)  # a 4:5 photo loses ~26% of its width
    src, dest = tmp_path / "s.png", tmp_path / "d.png"
    im = Image.new("RGB", (864, 1536), (200, 100, 50))
    for y in range(1536):                                               # a red edge column on the far left
        for x in range(8):
            im.putpixel((x, y), (255, 0, 0))
    im.save(src)
    talk_comfy._prepare_image(src, dest, 720, 1280, stretch_tolerance=0.01)
    out = Image.open(dest).convert("RGB")
    assert out.size == (720, 1280) and out.getpixel((1, 640))[0] > 200 and out.getpixel((1, 640))[1] < 60


def test_the_handlers_take_the_fit_tolerance_from_policy(raw_policy):
    assert raw_policy.get("generation", "talk", "fit_stretch_tolerance") > 0
    assert raw_policy.get("generation", "comfy_video", "fit_stretch_tolerance") > 0
    photo = raw_policy.get("generation", "image_by_format", "talking_video")
    assert photo == raw_policy.get("generation", "image_by_format", "video")
    assert photo["width"] * 16 == photo["height"] * 9 and photo["width"] % 16 == photo["height"] % 16 == 0


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


def test_pruning_clears_only_its_own_partial_downloads_never_another_familys_weights(tmp_path):
    """Owner 2026-11-25: a comfy job deleted the voice and image families' weights on the shared volume (every sibling
    of its folder); strays are named in policy and pruned at boot by worker.volume instead."""
    from manyworlds.worker import comfy
    root = tmp_path / "mw"
    (root / "Wan-AI" / "Wan2.1-I2V-14B-480P").mkdir(parents=True)
    (root / "comfy" / "vae").mkdir(parents=True)
    (root / "comfy" / "vae" / "x.safetensors.incomplete").write_bytes(b"part")
    (root / "comfy" / "vae" / "keep.safetensors").write_bytes(b"x" * 10)
    comfy.ensure_weights(root / "comfy", {"vae": [("r", "new.safetensors")]}, prune=True,
                         download=lambda repo, name, local_dir, token, **k: (root / "comfy" / "vae" / name).write_bytes(b"x"),
                         remote_size=lambda repo, name, token: None)
    assert sorted(p.name for p in root.iterdir()) == ["Wan-AI", "comfy"]          # another family's weights stay
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
    assert (root / "old-research").exists()                                     # siblings are never touched
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


# LongCat-Video-Avatar (comfy.graph_longcat): node classes of Kijai's example workflow
# (example_workflows/LongCatAvatar_audio_image_to_video_example_01.json at the pinned commit), minus the vocal
# separator, audio trim, KJNodes and VideoHelperSuite helpers our image doesn't have; core nodes stand in for those.
LONGCAT_EXAMPLE_NODES = {"LoadImage", "LoadAudio", "WanVideoVAELoader", "WanVideoTextEncodeCached", "WanVideoEncode",
                         "WanVideoDecode", "WanVideoSamplerv2", "WanVideoSchedulerv2",
                         "WanVideoLongCatAvatarExtendEmbeds", "DownloadAndLoadWav2VecModel", "MultiTalkWav2VecEmbeds",
                         "WanVideoBlockSwap", "WanVideoModelLoader", "WanVideoLoraSelect", "WanVideoTorchCompileSettings"}
CORE_STAND_INS = {"ImageFromBatch", "CreateVideo", "SaveVideo"}          # for GetImageRangeFromBatch, VHS_VideoCombine
LONGCAT_15_AUDIO = {"WhisperModelLoader", "LongCatAvatarWhisperEmbeds"}  # the wrapper's v1.5 audio nodes


def _lc(frames=250, **p):
    return comfy.graph_longcat(image="a.png", audio="a.wav", prompt="she talks", frames=frames, seed=5,
                               params={"width": 480, "height": 832, "steps": 10, "frame_window": 93, "overlap": 13, **p},
                               blocks_to_swap=0)


def _links_resolve(g):
    for node in g.values():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in g, v


def test_the_longcat_graph_uses_the_examples_nodes_and_wires_image_audio_embeds_and_sampler():
    g = _lc()                                   # 250 frames: 93, then 80 new per window -> 3 windows
    assert {n["class_type"] for n in g.values()} <= LONGCAT_EXAMPLE_NODES | CORE_STAND_INS
    assert sorted(k for k in g if k.startswith("sample")) == ["sample0", "sample1", "sample2"]
    assert g["ref"]["inputs"]["image"] == ["image", 0] and g["ref"]["inputs"]["vae"] == ["vae", 0]
    assert g["embeds"]["class_type"] == "MultiTalkWav2VecEmbeds" and g["embeds"]["inputs"]["audio_1"] == ["audio", 0]
    assert g["embeds"]["inputs"]["num_frames"] == 500 and g["embeds"]["inputs"]["fps"] == 32.0   # audio stride 2
    for k in range(3):
        assert g[f"ext{k}"]["inputs"]["audio_embeds"] == ["embeds", 0]
        s = g[f"sample{k}"]["inputs"]
        assert (s["model"], s["image_embeds"], s["scheduler"], s["text_embeds"]) == (
            ["model", 0], [f"ext{k}", 0], ["scheduler", 0], ["text", 0])
        assert g[f"save{k}"]["inputs"]["filename_prefix"] == f"mw_longcat_w{k:02d}"
    e0, e1, e2 = (g[f"ext{k}"]["inputs"] for k in range(3))
    assert e0["prev_latents"] == ["ref", 0] and e0["overlap"] == 1 and "ref_latent" not in e0   # starts from the photo
    assert e1["prev_latents"] == ["sample0", 0] and e1["ref_latent"] == ["ref", 0] and e1["overlap"] == 13
    assert (e0["frames_processed"], e1["frames_processed"], e2["frames_processed"]) == (0, 93, 173)
    assert "new0" not in g and g["new1"]["inputs"]["batch_index"] == 13   # each later window drops its overlap
    assert g["video0"]["inputs"]["images"] == ["decode0", 0] and g["video1"]["inputs"]["images"] == ["new1", 0]
    m = g["model"]["inputs"]
    assert (m["base_precision"], m["attention_mode"], m["lora"]) == ("bf16", "sdpa", ["lora", 0])   # example's note
    assert g["scheduler"]["inputs"]["scheduler"] == "longcat_distill_euler" and g["video0"]["inputs"]["fps"] == 16.0
    _links_resolve(g)


def test_the_longcat_15_graph_takes_whisper_audio_and_re_encodes_the_overlap():
    g = _lc(frames=100, longcat_version="1.5")
    assert {n["class_type"] for n in g.values()} <= LONGCAT_EXAMPLE_NODES | CORE_STAND_INS | LONGCAT_15_AUDIO
    assert g["embeds"]["class_type"] == "LongCatAvatarWhisperEmbeds" and g["embeds"]["inputs"]["whisper_model"] == ["whisper", 0]
    assert g["embeds"]["inputs"]["num_frames"] == 100 and g["video0"]["inputs"]["fps"] == 25.0
    assert g["ext1"]["inputs"]["prev_images"] == ["decode0", 0] and g["ext1"]["inputs"]["vae"] == ["vae", 0]
    assert g["model"]["inputs"]["quantization"] == "disabled"           # a GGUF model loads unquantized
    _links_resolve(g)


def test_longcat_window_counts():
    assert [comfy.longcat_windows(f, 93, 13) for f in (1, 93, 94, 173, 174)] == [1, 1, 2, 2, 3]


def test_longcat_weights_are_their_own_and_the_talk_model_is_chosen_by_params(tmp_path, monkeypatch):
    import pytest
    from manyworlds.worker.runner import NonRetryable
    assert comfy.WEIGHTS_LONGCAT["diffusion_models"] != comfy.WEIGHTS["diffusion_models"]
    assert not any("LongCat" in f for files in comfy.WEIGHTS.values() for _, f in files)   # InfiniteTalk never downloads them
    assert talk_comfy.model_of({}) == "infinitetalk" and talk_comfy.model_of({"model": "longcat"}) == "longcat"
    with pytest.raises(NonRetryable):
        talk_comfy.model_of({"model": "flashtalk"})
    monkeypatch.setattr(comfy, "COMFY_DIR", tmp_path)
    assert "audio_encoders: audio_encoders" in comfy.write_model_paths(tmp_path / "cache").read_text()


def test_the_longcat_talk_profile_reaches_a_bench_job(conn, policy):
    from manyworlds import bench
    batch = bench.enqueue(conn, policy, "talk.infinitetalk", 1, extra={"image_uri": "s3://b/x.png"}, profile="longcat")
    p = conn.execute("SELECT payload FROM jobs WHERE batch = %s", (batch,)).fetchone()["payload"]["params"]
    assert (p["model"], p["frame_window"], p["overlap"], p["scheduler"], p["attention"]) == (
        "longcat", 93, 13, "longcat_distill_euler", "sdpa")


# Walk-and-talk (comfy.graph_v2v): InfiniteTalk video-to-video as Kijai's example
# (example_workflows/wanvideo_2_1_14B_V2V_InfiniteTalk_example_02.json at the pinned commit), with core ComfyUI nodes in
# place of its VideoHelperSuite / KJNodes ones (VHS_LoadVideo, ImageResizeKJv2, GetImageRangeFromBatch, VHS_VideoCombine).
V2V_EXAMPLE_NODES = {"WanVideoVAELoader", "WanVideoLoraSelect", "WanVideoBlockSwap", "MultiTalkModelLoader",
                     "WanVideoModelLoader", "DownloadAndLoadWav2VecModel", "LoadAudio", "MultiTalkWav2VecEmbeds",
                     "WanVideoTextEncodeCached", "WanVideoImageToVideoMultiTalk", "WanVideoSampler",
                     "WanVideoEncode", "WanVideoClipVisionEncode", "CLIPVisionLoader", "WanVideoTorchCompileSettings"}
V2V_CORE = {"LoadVideo", "GetVideoComponents", "ImageScale", "ImageFromBatch", "WanVideoPassImagesFromSamples",
            "CreateVideo", "SaveVideo"}
# The inputs of the nodes the V2V graph adds or rewires, as declared at the pins (wrapper a7439db, ComfyUI core):
# what the real catalogue check in the talkcomfy smoke test sees for them.
V2V_CATALOGUE = {
    "LoadVideo": {"input": {"required": {"file": []}}},
    "GetVideoComponents": {"input": {"required": {"video": []}}},
    "ImageScale": {"input": {"required": {"image": [], "upscale_method": [], "width": [], "height": [], "crop": []}}},
    "ImageFromBatch": {"input": {"required": {"image": [], "batch_index": [], "length": []}}},
    "WanVideoEncode": {"input": {"required": {k: [] for k in ("vae", "image", "enable_vae_tiling", "tile_x", "tile_y",
                                                              "tile_stride_x", "tile_stride_y")},
                                 "optional": {"noise_aug_strength": [], "latent_strength": [], "mask": []}}},
    "CLIPVisionLoader": {"input": {"required": {"clip_name": []}}},
    "WanVideoClipVisionEncode": {"input": {"required": {k: [] for k in ("clip_vision", "image_1", "strength_1",
                                                                        "strength_2", "crop", "combine_embeds",
                                                                        "force_offload")},
                                           "optional": {"image_2": [], "negative_image": [], "tiles": [], "ratio": []}}},
    "WanVideoImageToVideoMultiTalk": {"input": {"required": {k: [] for k in ("vae", "width", "height",
                                                                             "frame_window_size", "motion_frame",
                                                                             "force_offload", "colormatch")},
                                                "optional": {k: [] for k in ("start_image", "tiled_vae", "clip_embeds",
                                                                             "mode", "output_path")}}},
    "WanVideoSampler": {"input": {"required": {k: [] for k in ("model", "image_embeds", "steps", "cfg", "shift", "seed",
                                                               "force_offload", "scheduler", "riflex_freq_index")},
                                  "optional": {k: [] for k in ("text_embeds", "samples", "denoise_strength",
                                                               "feta_args", "context_options", "cache_args",
                                                               "flowedit_args", "batched_cfg", "slg_args",
                                                               "rope_function", "loop_args", "experimental_args",
                                                               "sigmas", "unianimate_poses", "fantasytalking_embeds",
                                                               "uni3c_embeds", "multitalk_embeds", "freeinit_args",
                                                               "start_step", "end_step", "add_noise_to_samples")}}},
}


def _v2v(**p):
    return comfy.graph_v2v(video="m.mp4", audio="a.wav", prompt="she walks and talks", frames=150, seed=7,
                           params={"width": 480, "height": 832, "steps": 4, "start_step": 2, **p}, blocks_to_swap=0)


def test_the_v2v_graph_encodes_the_motion_clip_into_the_samplers_samples():
    g = _v2v()
    assert {n["class_type"] for n in g.values()} <= V2V_EXAMPLE_NODES | V2V_CORE
    assert "LoadImage" not in {n["class_type"] for n in g.values()}             # the clip, not the photo
    assert g["motion"]["inputs"]["file"] == "m.mp4"
    assert g["motion_frames"]["inputs"]["video"] == ["motion", 0]
    r = g["motion_resized"]["inputs"]
    assert (r["image"], r["width"], r["height"], r["crop"]) == (["motion_frames", 0], 480, 832, "center")
    enc = g["motion_latents"]["inputs"]
    assert enc["image"] == ["motion_resized", 0] and enc["vae"] == ["vae", 0]
    first = g["motion_first"]["inputs"]
    assert (first["image"], first["batch_index"], first["length"]) == (["motion_resized", 0], 0, 1)
    i2v = g["i2v"]["inputs"]
    assert i2v["start_image"] == ["motion_first", 0] and i2v["mode"] == "infinitetalk"
    assert (i2v["frame_window_size"], i2v["motion_frame"]) == (81, 9) and "clip_embeds" not in i2v
    s = g["sampler"]["inputs"]
    assert s["samples"] == ["motion_latents", 0] and s["image_embeds"] == ["i2v", 0]
    assert (s["steps"], s["start_step"], s["add_noise_to_samples"], s["denoise_strength"]) == (4, 2, True, 1.0)
    assert (s["cfg"], s["shift"], s["scheduler"]) == (1.0, 11.0, "dpm++_sde")   # the example's settings
    assert s["multitalk_embeds"] == ["embeds", 0] and g["embeds"]["inputs"]["num_frames"] == 150
    assert g["video"]["inputs"]["audio"] == ["audio", 0] and g["video"]["inputs"]["fps"] == float(comfy.TALK_FPS)
    _links_resolve(g)
    # The nodes it adds or rewires take exactly the inputs their classes declare.
    changed = {k: g[k] for k in ("motion", "motion_frames", "motion_resized", "motion_first", "motion_latents",
                                 "i2v", "sampler")}
    assert comfy.missing_inputs(changed, V2V_CATALOGUE) == []


def test_the_v2v_graph_adds_clip_vision_of_the_first_frame_only_when_asked():
    g = _v2v(clip_vision=True)
    assert g["clip_vision"]["inputs"]["clip_name"] == comfy.CLIP_VISION
    assert g["clip_embeds"]["inputs"]["image_1"] == ["motion_first", 0]
    assert g["i2v"]["inputs"]["clip_embeds"] == ["clip_embeds", 0]
    assert comfy.missing_inputs({k: g[k] for k in ("clip_vision", "clip_embeds", "i2v")}, V2V_CATALOGUE) == []
    _links_resolve(g)
    assert "clip_vision" not in comfy.WEIGHTS                                     # only a CLIP-vision job fetches it


def test_the_v2v_start_step_must_leave_steps_to_sample():
    import pytest
    for bad in (4, 5, -1):
        with pytest.raises(ValueError):
            _v2v(start_step=bad)
    assert _v2v(start_step=0)["sampler"]["inputs"]["start_step"] == 0


def test_a_v2v_talk_job_needs_infinitetalk_and_a_valid_start_step():
    import pytest
    from manyworlds.worker.runner import NonRetryable
    base = {"image_uri": "s3://b/p.png", "audio_uri": "s3://b/a.wav", "keyframes": 4, "video_uri": "s3://b/m.mp4"}
    params = {"width": 480, "height": 832, "steps": 4, "timeout_s": 60}
    with pytest.raises(NonRetryable, match="InfiniteTalk only"):
        talk_comfy.generate({**base, "params": {**params, "model": "longcat"}}, None)
    with pytest.raises(NonRetryable, match="start_step"):
        talk_comfy.generate({**base, "params": {**params, "start_step": 4}}, None)


def test_the_motion_clip_is_resampled_to_the_talk_frame_rate(tmp_path):
    import subprocess

    import imageio_ffmpeg

    from manyworlds import media
    src = tmp_path / "m16.mp4"
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x288:rate=16:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    out = talk_comfy.at_fps(src, tmp_path / "m25.mp4", comfy.TALK_FPS)
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(out))
    assert abs(frames - 2 * comfy.TALK_FPS) <= 4 and abs(seconds - 2.0) < 0.15   # minterpolate may add edge frames      # same length, 25 fps


def test_walk_and_talk_interpolates_the_motion_clip_instead_of_repeating_frames(tmp_path, raw_policy, monkeypatch):
    import subprocess

    import imageio_ffmpeg

    from manyworlds import media
    src = tmp_path / "m16.mp4"
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x288:rate=16:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    interp = raw_policy.get("generation", "talk", "motion_interpolate")
    assert interp["enabled"] and interp["method"] == "minterpolate"
    seen = []
    real_run = subprocess.run

    def spy(cmd, *a, **k):
        seen.append(cmd)
        return real_run(cmd, *a, **k)
    monkeypatch.setattr(subprocess, "run", spy)
    out = talk_comfy.at_fps(src, tmp_path / "m25.mp4", comfy.TALK_FPS, interpolate=interp)
    vf = next(c[c.index("-vf") + 1] for c in seen if "-vf" in c)
    assert vf.startswith(f"minterpolate=fps={comfy.TALK_FPS}:mi_mode=mci")
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(out))
    assert abs(frames - 2 * comfy.TALK_FPS) <= 4 and abs(seconds - 2.0) < 0.15   # minterpolate may add edge frames
    seen.clear()
    talk_comfy.at_fps(src, tmp_path / "dup.mp4", comfy.TALK_FPS, interpolate={**interp, "enabled": False})
    assert next(c[c.index("-vf") + 1] for c in seen if "-vf" in c) == f"fps={comfy.TALK_FPS}"   # switched off
