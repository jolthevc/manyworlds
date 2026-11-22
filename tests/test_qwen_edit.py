"""The image.qwen_edit handler (Qwen-Image-Edit-2511 + Lightning): payload checks, and a run with a stand-in pipeline
and torch (no GPU libraries in the test image): the input images in order, the Lightning sampler and LoRA per job,
one output per candidate with its own seed."""

from __future__ import annotations

import contextlib
import sys
import types

import pytest
from PIL import Image

from manyworlds import queue
from manyworlds.worker.handlers import image_qwen_edit as edit
from manyworlds.worker.runner import JobContext, NonRetryable, handlers_for

GOOD = {"prompt": "Picture 1 is the person. She makes toast.", "images": ["s3://b/a.png", "s3://b/plate.png"],
        "n": 2, "seed": 100, "width": 896, "height": 1152}


def test_settings_have_the_lightning_defaults():
    s = edit.settings(GOOD)
    assert s["steps"] == 4 and s["true_cfg"] == 1.0 and s["negative_prompt"] == " " and s["lightning"] is True
    assert s["images"] == GOOD["images"] and s["n"] == 2 and s["seed"] == 100


@pytest.mark.parametrize("bad", [
    {"prompt": ""},
    {"images": []},
    {"images": "s3://b/a.png"},
    {"images": ["s3://b/1", "s3://b/2", "s3://b/3", "s3://b/4"]},   # anchor, plate, outfit at most
    {"images": ["s3://b/a.png", ""]},
    {"width": 900},                                                # not a multiple of 16
    {"height": 128},
    {"n": 0},
    {"params": {"true_cfg": 0.5}},
    {"params": {"steps": "many"}},
])
def test_unusable_payloads_are_not_retried(bad):
    with pytest.raises(NonRetryable):
        edit.settings({**GOOD, **bad})


def test_the_kind_is_registered_on_the_image_worker():
    assert handlers_for(["image.qwen_edit"])["image.qwen_edit"] is edit.generate


def test_the_kind_runs_only_on_large_cards(policy):
    pools = policy.get("job_kinds", "image.qwen_edit", "pools")
    gpus = {policy.pool(p)["gpu"] for p in pools}
    assert pools and gpus <= {"l40s", "h100", "a40"}
    assert policy.get("sets", "reference_conditioning") is False     # off until benchmarked


class FakePipe:
    def __init__(self):
        self.calls, self.scheduler, self.lora = [], "BASE", True

    def enable_lora(self):
        self.lora = True

    def disable_lora(self):
        self.lora = False

    def __call__(self, **kw):
        self.calls.append({**kw, "scheduler": self.scheduler, "lora": self.lora})
        return types.SimpleNamespace(images=[Image.new("RGB", (kw["width"], kw["height"]))])


class FakeGenerator:
    def __init__(self, device):
        self.device, self.seed = device, None

    def manual_seed(self, seed):
        self.seed = seed
        return self


@pytest.fixture()
def fake_gpu(monkeypatch):
    oom = type("OutOfMemoryError", (RuntimeError,), {})
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        Generator=FakeGenerator, inference_mode=contextlib.nullcontext,
        cuda=types.SimpleNamespace(OutOfMemoryError=oom)))
    pipe = FakePipe()
    monkeypatch.setattr(edit, "_pipe", lambda ctx: pipe)
    monkeypatch.setattr(edit, "lightning_scheduler", lambda: "LIGHTNING")
    for k, v in {"lightning": None, "base_scheduler": "BASE", "offload": False, "fallback": False}.items():
        monkeypatch.setitem(edit._state, k, v)
    return pipe


def _ctx(store, tmp_path, payload):
    uris = []
    for i, colour in enumerate(["red", "green", "blue"][:len(payload["images"])]):
        p = tmp_path / f"in{i}.png"
        Image.new("RGB", (32, 48), colour).save(p)
        uris.append(store.put_file(p, f"in/{i}.png").uri)
    job = queue.Job(id="j1", kind="image.qwen_edit", payload=payload, attempts=1, max_attempts=3, creator_id=None,
                    batch=None)
    (tmp_path / "work").mkdir()
    return JobContext(job=job, workdir=tmp_path / "work", store=store), uris


def test_generate_feeds_the_images_in_order_and_makes_n_candidates(fake_gpu, store, tmp_path):
    ctx, uris = _ctx(store, tmp_path, GOOD)
    out = edit.generate({**GOOD, "images": uris}, ctx)
    assert len(out.outputs) == 2 and all(o.kind == "image" for o in out.outputs)
    assert [o.meta["seed"] for o in out.outputs] == [100, 101]
    first = fake_gpu.calls[0]
    assert [im.getpixel((0, 0)) for im in first["image"]] == [(255, 0, 0), (0, 128, 0)]   # anchor, then plate
    assert first["prompt"] == GOOD["prompt"] and first["num_inference_steps"] == 4 and first["true_cfg_scale"] == 1.0
    assert first["width"] == 896 and first["height"] == 1152 and first["num_images_per_prompt"] == 1
    assert [c["generator"].seed for c in fake_gpu.calls] == [100, 101]
    assert first["scheduler"] == "LIGHTNING" and first["lora"] is True
    assert out.recipe["model"] == edit.MODEL_ID and edit.LORA_FILE in out.recipe["lora"]
    assert out.recipe["images"] == uris


def test_without_lightning_the_base_sampler_runs_without_the_lora(fake_gpu, store, tmp_path):
    payload = {**GOOD, "n": 1, "params": {"lightning": False, "steps": 40, "true_cfg": 4.0}}
    ctx, uris = _ctx(store, tmp_path, payload)
    out = edit.generate({**payload, "images": uris}, ctx)
    call = fake_gpu.calls[0]
    assert call["scheduler"] == "BASE" and call["lora"] is False
    assert call["num_inference_steps"] == 40 and call["true_cfg_scale"] == 4.0 and out.recipe["lora"] is None
