"""The ComfyUI handlers pass a job's own negative prompt (payload `negative_prompt`, compiled from the shot sheet by
manyworlds.direction.compile_prompts) into every graph they build: story clips (comfy_video, graph_i2v) and talk
(talk_comfy: graph, graph_v2v, graph_longcat). ComfyUI, the GPU and the weights are stood in for; the graph
builders themselves are tested in test_comfy_negative.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from manyworlds.worker import comfy
from manyworlds.worker.handlers import comfy_video, talk_comfy

NEG = "waxy plastic skin, logos, extra people"


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
def stubbed(tmp_path, monkeypatch):
    """ComfyUI stood in for: the graph builders record their `negative`, `run` / `run_all` hand back a file."""
    seen: dict[str, object] = {}
    monkeypatch.setattr(comfy, "COMFY_DIR", tmp_path / "comfy")
    monkeypatch.setenv("MW_MODEL_DIR", str(tmp_path / "models"))
    for name in ("write_model_paths", "ensure_weights", "start"):
        monkeypatch.setattr(comfy, name, lambda *a, **k: None)

    def out(name: str) -> Path:
        p = tmp_path / "comfy" / "output" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"video")
        return p
    monkeypatch.setattr(comfy, "run", lambda g, timeout_s: out("clip.mp4"))
    monkeypatch.setattr(comfy, "run_all", lambda g, timeout_s: [out("w0.mp4"), out("w1.mp4")])
    for name in ("graph", "graph_v2v", "graph_longcat", "graph_i2v"):
        def spy(*, _name=name, negative=None, **kw):
            seen[_name] = negative
            return {"graph": _name}
        monkeypatch.setattr(comfy, name, spy)
    for mod in (comfy_video, talk_comfy):
        monkeypatch.setattr(mod, "_blocks_to_swap", lambda params: 0)
        monkeypatch.setattr(mod, "_prepare_image", lambda *a, **k: None)
        monkeypatch.setattr(mod, "extract_keyframes", lambda clip, work, n: [])
    monkeypatch.setattr(comfy_video.crops, "crops_in_venv", lambda *a, **k: [])
    monkeypatch.setattr(talk_comfy, "audio_seconds", lambda path: 2.0)
    monkeypatch.setattr(talk_comfy, "at_fps", lambda src, dest, fps, **k: dest)
    monkeypatch.setattr(talk_comfy, "join_windows", lambda clips, audio, seconds, dest: dest.write_bytes(b"v") and dest)
    monkeypatch.setattr(talk_comfy.lipsync, "score_in_venv", lambda *a, **k: {"frames": 50})
    return seen


TALK = {"image_uri": "s3://b/a.png", "audio_uri": "s3://b/a.wav", "keyframes": 2, "prompt": "she talks",
        "params": {"width": 480, "height": 832, "steps": 10, "timeout_s": 60}}


def test_a_story_clip_job_passes_its_negative_to_the_i2v_graph(stubbed, tmp_path):
    payload = {"image_uri": "s3://b/a.png", "prompt": "steam rises", "negative_prompt": NEG, "seed": 3,
               "params": {"width": 480, "height": 832, "timeout_s": 60}}
    res = comfy_video.generate(payload, Ctx(tmp_path / "w"))
    assert stubbed["graph_i2v"] == NEG and res.recipe["negative_prompt"] == NEG
    stubbed.clear()
    res = comfy_video.generate({k: v for k, v in payload.items() if k != "negative_prompt"}, Ctx(tmp_path / "w2"))
    assert stubbed["graph_i2v"] is None and res.recipe["negative_prompt"] == comfy.NEGATIVE   # the graph's default


@pytest.mark.parametrize("variant, graph", [
    ({}, "graph"),
    ({"video_uri": "s3://b/motion.mp4", "params": {**TALK["params"], "start_step": 2}}, "graph_v2v"),
    ({"params": {**TALK["params"], "model": "longcat", "longcat_version": "1.0"}}, "graph_longcat"),
])
def test_every_talk_graph_gets_the_jobs_negative(stubbed, tmp_path, variant, graph):
    res = talk_comfy.generate({**TALK, **variant, "negative_prompt": NEG}, Ctx(tmp_path / "w"))
    assert stubbed == {graph: NEG}
    assert res.recipe["negative_prompt"] == NEG
    stubbed.clear()
    talk_comfy.generate({**TALK, **variant}, Ctx(tmp_path / "w2"))
    assert stubbed == {graph: None}                       # no negative of its own: the graph's standard list
