"""Per-creator identity LoRA, trained with ostris/ai-toolkit on a 24 GB card.

Payload: creator_id, trigger, images[{uri, caption}], steps, rank, learning_rate, resolution.
Output: one .safetensors asset (kind "lora"); the orchestrator attaches it to the creator.

Model and train settings follow ai-toolkit's own "zimage:turbo" preset
(extensions_built_in/diffusion_models/ui.tsx, checked against commit 4a7f285, Nov 2026):
Turbo is trained through ostris's training adapter, which is merged in for training and
cancelled out at inference, so the LoRA learns identity without undoing the 8-step
distillation. 8-bit quantisation plus low_vram keeps it inside a 24 GB consumer card.
MW_LORA_MODEL_BLOCK (JSON) overrides the model block if a later ai-toolkit changes keys.
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
    "arch": "zimage:turbo",
    "assistant_lora_path": "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors",
    "quantize": True,
    "quantize_te": True,
    "qtype": "qfloat8",
    "low_vram": True,
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
        "train": {"batch_size": 1, "steps": int(payload.get("steps", 1200)),
                  "lr": float(payload.get("learning_rate", 1e-4)), "optimizer": "adamw8bit",
                  "gradient_checkpointing": True, "noise_scheduler": "flowmatch",
                  "timestep_type": "weighted", "unload_text_encoder": False, "dtype": "bf16"},
        "model": model_block,
        # Sampling is off during training (we judge the LoRA ourselves afterwards).
        "sample": {"sampler": "flowmatch", "sample_every": 10_000_000, "sample_steps": 9,
                   "guidance_scale": 1, "width": 896, "height": 1152,
                   "prompts": [f"photo of {payload['trigger']}"]},
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
