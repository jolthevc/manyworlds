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
