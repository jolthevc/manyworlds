"""Per-creator identity LoRA, trained with ostris/ai-toolkit on a 24 GB card.

Payload: creator_id, trigger, images[{uri, caption}], steps, rank, learning_rate, resolution.
Output: one .safetensors asset (kind "lora"); the orchestrator attaches it to the creator.

ai-toolkit's config keys for Z-Image Turbo change between releases (it trains Turbo
through a de-distillation adapter), so the model block comes from MW_LORA_MODEL_BLOCK
(JSON) when set. Not yet verified on a GPU: run one training job on day 2 and check
the result on a held-out prompt before trusting it.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml

from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

TOOLKIT = Path(os.environ.get("MW_AI_TOOLKIT", "/ai-toolkit"))
DEFAULT_MODEL_BLOCK = {
    "name_or_path": os.environ.get("MW_IMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo"),
    "arch": "zimage",
    "quantize": True,
}


def train(payload: dict, ctx: JobContext) -> HandlerResult:
    images = payload.get("images") or []
    if len(images) < 8:
        raise NonRetryable("need at least 8 training images")
    data = ctx.workdir / "dataset"
    data.mkdir(parents=True)
    for i, img in enumerate(images):
        ctx.store.get_file(img["uri"], data / f"{i:03d}.png")
        (data / f"{i:03d}.txt").write_text(img["caption"], encoding="utf-8")

    name = f"lora_{payload['trigger']}"
    out_dir = ctx.workdir / "output"
    model_block = json.loads(os.environ["MW_LORA_MODEL_BLOCK"]) if os.environ.get("MW_LORA_MODEL_BLOCK") else DEFAULT_MODEL_BLOCK
    rank = int(payload.get("rank", 16))
    config = {"job": "extension", "config": {"name": name, "process": [{
        "type": "sd_trainer", "training_folder": str(out_dir), "device": "cuda:0",
        "trigger_word": payload["trigger"],
        "network": {"type": "lora", "linear": rank, "linear_alpha": rank},
        "save": {"dtype": "float16", "save_every": 100000, "max_step_saves_to_keep": 1},
        "datasets": [{"folder_path": str(data), "caption_ext": "txt", "caption_dropout_rate": 0.05,
                      "resolution": [int(payload.get("resolution", 1024))]}],
        "train": {"batch_size": 1, "steps": int(payload.get("steps", 2000)),
                  "lr": float(payload.get("learning_rate", 1e-4)), "optimizer": "adamw8bit",
                  "gradient_checkpointing": True, "noise_scheduler": "flowmatch", "dtype": "bf16"},
        "model": model_block,
        "sample": {"sample_every": 100000, "prompts": [], "sample_steps": 8},
    }]}}
    cfg_path = ctx.workdir / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    proc = subprocess.run(["python", "run.py", str(cfg_path)], cwd=TOOLKIT, capture_output=True, text=True)
    weights = sorted(out_dir.rglob("*.safetensors"))
    if proc.returncode != 0 or not weights:
        raise RuntimeError(f"ai-toolkit exited {proc.returncode}: {proc.stderr[-3000:]}")
    final = weights[-1]
    return HandlerResult([OutputFile(final, "lora", {"trigger": payload["trigger"], "rank": rank})],
                         recipe={"trainer": "ostris/ai-toolkit", "model": model_block, "steps": payload.get("steps"),
                                 "images": len(images), "trigger": payload["trigger"]},
                         meta={"trigger": payload["trigger"]})


HANDLERS = {"train.lora": train}
