"""Checks a worker image before it is pushed, with no GPU and nothing rented.

    python -m manyworlds.worker.smoke image      # run inside the built image

Handlers import their heavy libraries lazily, so a missing package only shows up on a rented
machine, after its boot and model download. Between 2026-11-12 and 2026-11-14 that cost about
$1.46 of machines: missing ftfy and imageio, a keyframe reader that ran out of memory, and more.
This imports what each kind needs and runs the CPU parts for real. scripts/build-worker.sh
pushes an image only when it passes.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
from pathlib import Path


def _imports(names: list[str]) -> None:
    for name in names:
        importlib.import_module(name)


def check_image() -> list[str]:
    done = []
    _imports(["torch", "diffusers", "transformers", "accelerate", "peft", "safetensors", "PIL", "soundfile",
              "kokoro", "ftfy", "imageio", "imageio_ffmpeg"])
    from diffusers import AutoencoderKLWan, DiffusionPipeline, WanImageToVideoPipeline  # noqa: F401
    done.append("imports")

    from .runner import handlers_for
    from .. import policy
    kinds = [k for k in policy.load().get("job_kinds", default={}) if k.split(".")[0] in ("image", "video")]
    handlers_for(kinds)
    done.append(f"handlers: {', '.join(kinds)}")

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg missing (narrated edits need it)")
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean   # uses ftfy, as the Wan pipeline does
    prompt_clean("a woman pours tea &amp; smiles")
    done.append("ffmpeg, prompt cleaning")

    import numpy as np
    from diffusers.utils import export_to_video
    from .handlers.video_wan import extract_keyframes, keyframe_indices
    d = Path(tempfile.mkdtemp())
    frames = [np.full((64, 48, 3), i / 121, dtype=np.float32) for i in range(121)]
    export_to_video(frames, str(d / "clip.mp4"), fps=24)
    outs = extract_keyframes(d / "clip.mp4", d, 4)
    if [o.meta["frame"] for o in outs] != keyframe_indices(121, 4):
        raise RuntimeError(f"keyframes came out wrong: {[o.meta for o in outs]}")
    done.append("clip export and keyframes")
    return done


def check_train() -> list[str]:
    _imports(["torch", "yaml"])
    from .runner import handlers_for
    handlers_for(["train.lora"])
    return ["imports", "handlers: train.lora"]


def check_talk() -> list[str]:
    import subprocess

    _imports(["torch", "xformers", "flash_attn", "librosa", "misaki", "kokoro", "soundfile", "pyloudnorm",
              "decord", "optimum.quanto", "imageio", "imageio_ffmpeg", "huggingface_hub"])
    from .runner import handlers_for
    from .handlers import talk_infinitetalk as talk
    handlers_for(["talk.infinitetalk"])
    done = ["imports", "handlers: talk.infinitetalk"]
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg missing (InfiniteTalk muxes the audio with it)")
    # --help imports the whole script (wan, xfuser, its attention kernels). wan/modules/t5.py asks for the
    # current CUDA device in a default argument at import time, so that one call is stubbed here.
    run_help = ("import runpy, sys, torch; torch.cuda.current_device = lambda: 0; "
                "sys.argv = ['generate_infinitetalk.py', '--help']; "
                "runpy.run_path('generate_infinitetalk.py', run_name='__main__')")
    proc = subprocess.run([sys.executable, "-c", run_help], cwd=str(talk.REPO_DIR), capture_output=True, text=True)
    if proc.returncode != 0 or "--infinitetalk_dir" not in proc.stdout:
        raise RuntimeError(f"generate_infinitetalk.py --help failed: {(proc.stderr or proc.stdout)[-1500:]}")
    done.append("InfiniteTalk script imports")

    # Its wav2vec wrapper, loaded as generate_infinitetalk.py loads it, on a tiny random model: the first
    # benchmark (2026-11-20) died here after a 20-minute weight download.
    probe = """
import sys, tempfile, torch
sys.path.insert(0, '.')
from transformers import Wav2Vec2Config
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
assert 'attn_implementation="eager"' in open('generate_infinitetalk.py').read(), 'wav2vec not loaded with eager attention'
d = tempfile.mkdtemp()
cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                     conv_dim=(32, 32), conv_stride=(5, 2), conv_kernel=(10, 3), num_feat_extract_layers=2)
Wav2Vec2Model(cfg).save_pretrained(d)
m = Wav2Vec2Model.from_pretrained(d, local_files_only=True, attn_implementation='eager')
out = m(torch.zeros(1, 16000), seq_len=25, output_hidden_states=True)
assert len(out.hidden_states) == 2, len(out.hidden_states)
"""
    proc = subprocess.run([sys.executable, "-c", probe], cwd=str(talk.REPO_DIR), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"InfiniteTalk's wav2vec encoder failed: {(proc.stderr or proc.stdout)[-1500:]}")
    done.append("wav2vec encoder forward pass")

    # The fp8 path's Marlin kernel must be prebuilt, with the same flags it will be loaded with.
    import glob
    import optimum.quanto
    build = os.path.join(os.path.dirname(optimum.quanto.__file__), "library", "extensions", "cuda", "build")
    if not glob.glob(os.path.join(build, "*.so")) or os.environ.get("QUANTO_CUDA_ARCH") != "800":
        raise RuntimeError(f"quanto CUDA kernels are not prebuilt in {build} (or QUANTO_CUDA_ARCH unset)")
    done.append("quanto fp8 kernels prebuilt")

    import numpy as np
    import soundfile as sf
    d = Path(tempfile.mkdtemp())
    sf.write(str(d / "a.wav"), np.zeros(24000 * 2, dtype=np.float32), 24000)
    if talk.audio_seconds(d / "a.wav") != 2.0:
        raise RuntimeError("audio length came out wrong")
    done.append("audio length")
    return done


CHECKS = {"image": check_image, "train": check_train, "talk": check_talk}


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    family = args[0] if args else "image"
    try:
        for step in CHECKS[family]():
            print(f"ok  {step}")
    except Exception as e:  # noqa: BLE001 - report any failure and exit non-zero
        print(f"FAIL {type(e).__name__}: {e}")
        return 1
    print(f"smoke test passed for the {family} worker image")
    return 0


if __name__ == "__main__":
    sys.exit(main())
