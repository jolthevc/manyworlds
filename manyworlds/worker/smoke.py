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

    # The DMD sampler (FastWan): a model that predicts the clean latents exactly must land on them in 3 steps.
    import torch

    from .dmd import DMDScheduler
    clean = torch.randn(1, 4, 2, 3, 3)
    s = DMDScheduler([1000, 757, 522], seed=1)
    s.set_timesteps(3)
    x = torch.randn_like(clean)
    for t in s.timesteps:
        sigma = s.sigma(t)
        v = (x - clean) / sigma          # the flow prediction a perfect model would give at this noise level
        x = s.step(v, t, x)[0]
    if not torch.allclose(x, clean, atol=1e-4):
        raise RuntimeError("DMD sampler did not recover the clean latents")
    done.append("DMD sampler")
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

    # The lip-sync score in its own environment, on a blank clip: mediapipe and its model load, and find no face.
    import imageio.v3 as iio

    from . import lipsync
    iio.imwrite(str(d / "blank.mp4"), np.zeros((25, 64, 64, 3), dtype=np.uint8), fps=25)
    s = lipsync.score_in_venv(d / "blank.mp4", d / "a.wav")
    if "error" in s or s.get("face_share") != 0.0 or not s.get("frames"):
        raise RuntimeError(f"lip-sync score failed: {s}")
    done.append("lip-sync score")
    return done


def check_talkcomfy() -> list[str]:
    """The ComfyUI talk worker: its imports, the handler, ComfyUI starting (on the CPU here) with WanVideoWrapper
    loaded, and the InfiniteTalk graph valid against ComfyUI's own node catalogue, so a wiring mistake fails the
    build instead of a rented machine."""
    import numpy as np
    import soundfile as sf

    _imports(["torch", "kokoro", "misaki", "soundfile", "sageattention", "huggingface_hub", "imageio", "PIL"])
    from .runner import handlers_for
    from . import comfy
    os.environ["MW_TALK_ENGINE"] = "comfy"
    handlers_for(["talk.infinitetalk"])
    done = ["imports", "handlers: talk.infinitetalk (comfy)"]
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg missing")
    if not shutil.which("gcc"):   # Triton (SageAttention) compiles at run time; 2026-11-22 it failed without one
        raise RuntimeError("no C compiler: Triton can't build SageAttention's kernels")
    comfy.write_model_paths(Path(tempfile.mkdtemp()))
    comfy.start(args=("--cpu",), wait_s=300)
    try:
        info = comfy._get("/object_info", timeout=60)
        g = comfy.graph(image="x.png", audio="x.wav", prompt="a person talks", frames=125, seed=1,
                        params={"width": 480, "height": 832, "steps": 6}, blocks_to_swap=20)
        problems = comfy.missing_inputs(g, info)
        if problems:
            raise RuntimeError("InfiniteTalk graph doesn't match ComfyUI's nodes: " + "; ".join(problems))
        clip = comfy.graph_i2v(image="x.png", prompt="she smiles", seed=1,
                               params={"width": 480, "height": 832}, blocks_to_swap=20)
        problems = comfy.missing_inputs(clip, info)
        if problems:
            raise RuntimeError("Wan 2.2 story-clip graph doesn't match ComfyUI's nodes: " + "; ".join(problems))
        fast = {"width": 480, "height": 832, "steps": 6, "compile": True, "quantization": "fp8_e4m3fn_scaled_fast"}
        problems = comfy.missing_inputs(comfy.graph(image="x.png", audio="x.wav", prompt="a person talks", frames=125,
                                                    seed=1, params=fast, blocks_to_swap=20), info)
        problems += comfy.missing_inputs(comfy.graph_i2v(image="x.png", prompt="she smiles", seed=1, params=fast,
                                                         blocks_to_swap=20), info)
        if problems:
            raise RuntimeError("the compiled / fp8-fast graphs don't match ComfyUI's nodes: " + "; ".join(problems))
        # LongCat-Video-Avatar, both audio paths (v1.0 wav2vec, v1.5 Whisper), over three windows so the
        # continuation nodes are checked too.
        for version in comfy.LONGCAT_WEIGHTS:
            lc = {"width": 480, "height": 832, "steps": 10, "frame_window": 93, "overlap": 13,
                  "longcat_version": version}
            g = comfy.graph_longcat(image="x.png", audio="x.wav", prompt="a person talks", frames=250, seed=1,
                                    params=lc, blocks_to_swap=25)
            problems = comfy.missing_inputs(g, info)
            if problems:
                raise RuntimeError(f"LongCat {version} graph doesn't match ComfyUI's nodes: " + "; ".join(problems))
    finally:
        comfy.stop()
    handlers_for(["comfy.wan_i2v"])
    done.append("ComfyUI + WanVideoWrapper nodes match the talk (InfiniteTalk, LongCat) and story-clip graphs")

    # The wrapper's own wav2vec encoder, as MultiTalkWav2VecEmbeds calls it, on a tiny random model: it must return
    # hidden states (with transformers 5 it returned None, after the weights had downloaded; 2026-11-22).
    import subprocess
    probe = """
import sys, tempfile, torch
sys.path.insert(0, 'custom_nodes/ComfyUI-WanVideoWrapper')
from transformers import Wav2Vec2Config
from multitalk.wav2vec2 import Wav2Vec2Model
d = tempfile.mkdtemp()
cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                     conv_dim=(32, 32), conv_stride=(5, 2), conv_kernel=(10, 3), num_feat_extract_layers=2)
Wav2Vec2Model(cfg).save_pretrained(d)
m = Wav2Vec2Model.from_pretrained(d, local_files_only=True)
out = m(torch.zeros(1, 16000), seq_len=25, output_hidden_states=True)
assert out.hidden_states is not None and len(out.hidden_states) == 3, out.hidden_states
"""
    proc = subprocess.run([sys.executable, "-c", probe], cwd=os.environ.get("MW_COMFY_DIR", "/opt/ComfyUI"),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"the wrapper's wav2vec encoder returned no hidden states: {(proc.stderr or proc.stdout)[-1500:]}")
    done.append("wav2vec encoder returns hidden states")

    from . import lipsync
    import imageio.v3 as iio
    d = Path(tempfile.mkdtemp())
    sf.write(str(d / "a.wav"), np.zeros(24000, dtype=np.float32), 24000)
    iio.imwrite(str(d / "blank.mp4"), np.zeros((25, 64, 64, 3), dtype=np.uint8), fps=25)
    s = lipsync.score_in_venv(d / "blank.mp4", d / "a.wav")
    if "error" in s or s.get("face_share") != 0.0:
        raise RuntimeError(f"lip-sync score failed: {s}")
    done.append("lip-sync score")
    return done + check_voice()


def check_voice() -> list[str]:
    """Designed voices on the talk image (handlers/tts_voxcpm.py): VoxCPM2 and its documented arguments, the
    transcriber and speaker embedding importable, transformers still on the wrapper's pin, the pitch tracker and the
    line joining run for real on the CPU. A real design run needs a GPU and the model download: on a build machine
    without one it is skipped and said so."""
    import inspect

    import numpy as np
    import transformers

    _imports(["voxcpm", "speechbrain.inference.speaker", "librosa", "einops"])
    from voxcpm import VoxCPM
    from .runner import JobContext, handlers_for
    from .handlers import tts_voxcpm as tts
    handlers_for(["voice.design", "voice.speak"])
    done = ["imports, handlers: voice.design, voice.speak"]
    if transformers.__version__ != "4.57.6":
        raise RuntimeError(f"voxcpm moved transformers to {transformers.__version__} (the wrapper needs 4.57.6)")
    args = set(inspect.signature(VoxCPM._generate).parameters)
    need = {"text", "prompt_wav_path", "prompt_text", "reference_wav_path", "cfg_value", "inference_timesteps"}
    if need - args:
        raise RuntimeError(f"VoxCPM.generate lacks {sorted(need - args)}")
    done.append("VoxCPM generate() takes the documented arguments")

    sr = 48000
    t = np.arange(sr) / sr
    tone = (0.3 * np.sin(2 * np.pi * (180 + 40 * np.sin(2 * np.pi * 2 * t)) * t)).astype(np.float32)
    if not tts._pitch_variance(tone, sr, {"pitch_hz": [65, 500]}) > 0:
        raise RuntimeError("pitch tracker found no pitch movement in a vibrato tone")
    done.append("pitch variance")

    class Fake:   # the model's place, so joining, timings and the 48 kHz file are checked without weights
        sample_rate = sr

        def generate(self, *, seed, text, **kw):
            return tone[: sr // 2]

    d = Path(tempfile.mkdtemp())
    ref = tts.write_wav(d / "ref.wav", tone, sr)

    class Store:
        def get_file(self, uri, local):
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ref, local)
            return local

    saved = (tts._tts, tts._asr)
    tts._tts, tts._asr = (lambda params: Fake()), (lambda params: (lambda wav, rate: "hello there"))
    try:
        res = tts.speak({"ref_uri": "x://ref.wav", "ref_text": "hello there", "takes": 2, "seed": 1,
                         "lines": [{"text": "hello there", "delivery": "softly", "pause_after_s": 0.5},
                                   {"text": "hello there"}]}, JobContext(job=None, workdir=d, store=Store()))
    finally:
        tts._tts, tts._asr = saved
    meta = res.outputs[0].meta
    if meta["sample_rate"] != sr or meta["seconds"] != 1.5 or [line["wer"] for line in meta["lines"]] != [0.0, 0.0]:
        raise RuntimeError(f"voice lines joined wrongly: {meta}")
    done.append("voice lines joined with pauses at 48 kHz")

    import torch
    if not torch.cuda.is_available():
        done.append("VoxCPM2 design run skipped: no GPU here (the first voice.design job is the real test)")
        return done
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False)
    wav = model.generate(text="(A calm adult voice)Hi.", cfg_value=2.0, inference_timesteps=4)
    if len(wav) < model.tts_model.sample_rate // 10:
        raise RuntimeError("VoxCPM2 made no audio")
    done.append(f"VoxCPM2 design run ({len(wav) / model.tts_model.sample_rate:.1f} s)")
    return done


CHECKS = {"image": check_image, "train": check_train, "talk": check_talk, "talkcomfy": check_talkcomfy,
          "voice": check_voice}


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
