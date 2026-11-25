"""Any ComfyUI graph by name (job kind comfy.graph): a new model slots in as a workflow file plus a policy entry, with
no handler of its own. Runs on the ComfyUI worker image (mw-worker-talkcomfy), family comfy.

A graph is `manyworlds/worker/graphs/<name>.json` (ComfyUI's API format, written from the node pack's own published
example and runnable as it stands) and its sidecar `<name>.yaml`: where the example came from (`source`), and
`inputs`, which payload value goes into which node input (`value: [node.input, ...]`). What the graph needs to
run is policy `comfy_graphs.<name>` (or the job's `graph_spec`, so a worker whose baked policy predates a graph still
runs it by the same rules): `weights` (ComfyUI model folder -> [[Hugging Face repo, file]]; a folder with a slash is a
node pack's own directory under `model_links`), `model_links`, `min_vram_gb`, `requires_env` (a gated repo's token),
`mux_audio` (our voice line replaces the graph's audio track), `pad_audio` (the line padded to the clip's length
before the graph reads it) and `params` (defaults; the payload's params override them).

Values a sidecar can map: image and audio (the uploaded file names), prompt, negative_prompt, seed, width, height,
half_width, half_height, fps, frames (the clip's length at params.fps, rounded up to frame_multiple plus frame_offset,
from the voice line, or params.seconds without one), seconds (the clip's length in seconds), and any params key.

Payload: graph, image_uri, audio_uri (when the graph reads audio), prompt, negative_prompt, seed, keyframes, crops,
params, graph_spec. Output: the video (with the graph's audio, or our line when mux_audio), keyframes for the video
gate, the lip-sync score where there is speech, and the benchmark's numbers in the video's meta.bench: weights
download, server start and render seconds, peak VRAM (sampled from ComfyUI while it renders), and whether the
graph's models were cold (the first render of this graph on this worker loads them inside the render).
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .. import comfy, crops, lipsync
from ..keyframes import extract_keyframes
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

KIND = "comfy.graph"
GRAPHS_DIR = Path(__file__).resolve().parents[1] / "graphs"
NAME = re.compile(r"^[a-z0-9_]+$")
REQUIRED = ("width", "height", "fps", "timeout_s", "vram_sample_s")
_ready: dict = {}        # graph name -> its weights are in place on this worker
_rendered: set = set()   # graphs rendered once here: their models are loaded (warm)


def _policy_spec(name: str) -> dict | None:
    from ... import policy as policy_mod
    return policy_mod.load().get("comfy_graphs", name, default=None)


def spec_for(payload: dict, policy_spec=None) -> tuple[str, dict]:
    """(name, spec) for the job's graph; NonRetryable for a name with no workflow file in this image or no entry in
    comfy_graphs (unknown graphs are never guessed at)."""
    name = str(payload.get("graph") or "")
    if not NAME.match(name):
        raise NonRetryable(f"payload needs graph, a name like ltx25_a2v (not {name!r})")
    if not (GRAPHS_DIR / f"{name}.json").is_file() or not (GRAPHS_DIR / f"{name}.yaml").is_file():
        raise NonRetryable(f"unknown graph {name!r}: no manyworlds/worker/graphs/{name}.json and .yaml in this image")
    spec = payload.get("graph_spec") or (policy_spec if policy_spec is not None else _policy_spec(name))
    if not spec:
        raise NonRetryable(f"unknown graph {name!r}: not in policy comfy_graphs")
    return name, spec


def load(name: str) -> tuple[dict, dict]:
    """The workflow (API format) and its sidecar."""
    import yaml
    graph = json.loads((GRAPHS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    side = yaml.safe_load((GRAPHS_DIR / f"{name}.yaml").read_text(encoding="utf-8")) or {}
    return graph, side


def frames_for(seconds: float, params: dict) -> int:
    """The clip's frame count for `seconds` at params.fps: rounded up so that frames - frame_offset is a multiple
    of frame_multiple (LTX: 8k + 1), never under min_frames."""
    fps, mult = float(params["fps"]), int(params.get("frame_multiple") or 1)
    off = int(params.get("frame_offset") or 0)
    n = math.ceil(round(max(float(seconds) * fps - off, 0), 6) / mult) * mult + off
    return max(n, int(params.get("min_frames") or 1))


def values_for(*, image: str | None, audio: str | None, prompt: str, negative: str | None, seed: int,
               seconds: float, params: dict) -> dict:
    """Everything a sidecar may map, by name."""
    frames = frames_for(seconds, params)
    w, h = int(params["width"]), int(params["height"])
    return {**params, "image": image, "audio": audio, "prompt": prompt, "negative_prompt": negative, "seed": seed,
            "width": w, "height": h, "half_width": w // 2, "half_height": h // 2, "frames": frames,
            "seconds": round((frames - int(params.get("frame_offset") or 0)) / float(params["fps"]), 3)}


def fill(graph: dict, inputs: dict, values: dict) -> dict:
    """A copy of the graph with each mapped value written into its node inputs. A mapped value that is missing, or a
    target the graph doesn't have, is refused (a broken sidecar fails here, not inside ComfyUI)."""
    g = json.loads(json.dumps(graph))
    for name, targets in (inputs or {}).items():
        if values.get(name) is None:
            raise NonRetryable(f"graph input {name!r} has no value (payload or params)")
        for target in targets:
            node, _, field = str(target).partition(".")
            if node not in g or field not in (g[node].get("inputs") or {}):
                raise NonRetryable(f"graph has no input {target!r} for {name!r}")
            g[node]["inputs"][field] = values[name]
    return g


def weights_of(spec: dict) -> dict:
    return {str(folder): [tuple(f) for f in files] for folder, files in (spec.get("weights") or {}).items()}


def _vram_total_gb() -> float | None:
    try:
        import torch
        return round(torch.cuda.mem_get_info()[1] / 1e9, 1) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - no CUDA here: the floor can't be checked
        return None


def check_machine(spec: dict, vram_gb: float | None, env=os.environ) -> None:
    """A card under the graph's floor, or a gated repo's token missing, can't be fixed by a retry: refused."""
    floor = float(spec.get("min_vram_gb") or 0)
    if vram_gb is not None and floor and vram_gb < floor:
        raise NonRetryable(f"this card has {vram_gb} GB; the graph needs {floor:g} GB (pin it to a bigger pool)")
    missing = [k for k in spec.get("requires_env") or [] if not env.get(k)]
    if missing:
        raise NonRetryable(f"the graph's weights need {', '.join(missing)} on the worker (a gated repo)")


def _ffmpeg(args: list[str], what: str, timeout: float = 600) -> None:
    from ...media import ffmpeg
    proc = subprocess.run([ffmpeg(), "-y", "-loglevel", "error", *args], capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed: {proc.stderr[-800:]}")


def pad_audio(src: Path, dest: Path, seconds: float) -> Path:
    """The line with silence after it, `seconds` long in all (the graph encodes audio as long as the clip)."""
    _ffmpeg(["-i", str(src), "-af", f"apad=whole_dur={seconds:.3f}", "-c:a", "pcm_s16le", str(dest)], "padding the line")
    return dest


def speech_of(clip: Path, dest: Path) -> Path:
    """The clip's own audio track as 16 kHz mono (a graph that makes its own sound), for the lip-sync score."""
    _ffmpeg(["-i", str(clip), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)], "reading the clip's audio")
    return dest


def generate(payload: dict, ctx: JobContext, *, policy_spec: dict | None = None) -> HandlerResult:
    from .talk_comfy import _models_dir, _prepare_image, join_windows
    from .talk_infinitetalk import audio_seconds
    name, spec = spec_for(payload, policy_spec)
    graph, side = load(name)
    inputs = side.get("inputs") or {}
    params: dict[str, Any] = {**(spec.get("params") or {}), **(payload.get("params") or {})}
    missing = [k for k in REQUIRED if params.get(k) is None]
    if missing:
        raise NonRetryable(f"params missing {missing}")
    reads_audio, reads_image = "audio" in inputs, "image" in inputs
    if reads_image and not payload.get("image_uri"):
        raise NonRetryable(f"graph {name} needs image_uri")
    if reads_audio and not payload.get("audio_uri"):
        raise NonRetryable(f"graph {name} needs audio_uri")
    if not reads_audio and not params.get("seconds"):
        raise NonRetryable(f"graph {name} reads no audio: give params.seconds")
    if not payload.get("keyframes"):
        raise NonRetryable("payload needs keyframes (the video gate judges them)")
    check_machine(spec, _vram_total_gb())
    seed = payload.get("seed")
    seed = int(seed) if seed is not None else random.randrange(1 << 31)
    work = Path(ctx.workdir)
    models = _models_dir()
    weights = weights_of(spec)

    t0 = time.monotonic()
    if not _ready.get(name):
        comfy.write_model_paths(models, list(weights))
        comfy.ensure_weights(models, weights, prune=True)
        comfy.link_models(models, spec.get("model_links") or [])
        _ready[name] = True
    download_s = round(time.monotonic() - t0, 1)
    t0 = time.monotonic()
    comfy.start()
    start_s = round(time.monotonic() - t0, 1)

    tag = ctx.job.id
    inp = comfy.COMFY_DIR / "input"
    inp.mkdir(parents=True, exist_ok=True)
    image_name = audio_name = None
    staged: list[Path] = []
    if reads_image:
        image_name = f"mw_{tag}.png"
        _prepare_image(ctx.fetch(payload["image_uri"], "start_frame.png"), inp / image_name, int(params["width"]),
                       int(params["height"]), stretch_tolerance=float(params.get("fit_stretch_tolerance") or 0))
        staged.append(inp / image_name)
    audio = None
    if reads_audio:
        audio = ctx.fetch(payload["audio_uri"], "voice" + (Path(str(payload["audio_uri"])).suffix or ".wav"))
        seconds = audio_seconds(audio)
    else:
        seconds = float(params["seconds"])
    values = values_for(image=image_name, audio=None, prompt=payload.get("prompt") or str(params.get("prompt") or ""),
                        negative=payload.get("negative_prompt") or params.get("negative_prompt"), seed=seed,
                        seconds=seconds, params=params)
    if audio is not None:
        audio_name = f"mw_{tag}.wav"
        if spec.get("pad_audio"):
            pad_audio(audio, inp / audio_name, values["seconds"])
        else:
            shutil.copy(audio, inp / audio_name)
        staged.append(inp / audio_name)
        values["audio"] = audio_name
    g = fill(graph, inputs, values)

    cold = name not in _rendered
    t0 = time.monotonic()
    with comfy.VramPeak(float(params["vram_sample_s"])) as vram:
        videos = comfy.run_all(g, timeout_s=float(params["timeout_s"]))
    render_s = round(time.monotonic() - t0, 1)
    _rendered.add(name)
    clip = work / "clip.mp4"
    if audio is not None and spec.get("mux_audio"):
        join_windows(videos[:1], audio, seconds, clip)   # our line, cut to its length (the graph pads past it)
    else:
        shutil.move(str(videos[0]), clip)
    for f in [*videos, *staged]:
        f.unlink(missing_ok=True)

    speech = audio if audio is not None else (speech_of(clip, work / "clip_audio.wav")
                                              if params.get("native_audio") else None)
    sync = lipsync.score_in_venv(clip, speech, max_offset_frames=int(params.get("sync_max_offset_frames", 5))) \
        if speech is not None else None
    bench = {"graph": name, "download_seconds": download_s, "server_start_seconds": start_s,
             "render_seconds": render_s, "cold": cold, "vram": vram.result(), "frames": values["frames"],
             "seconds_per_output_second": round(render_s / max(seconds, 0.1), 1)}
    meta = {"seconds": seconds, "width": params["width"], "height": params["height"], "engine": "comfy",
            "model": name, "graph": name, "bench": bench, "native_audio": bool(params.get("native_audio"))}
    if sync is not None:
        meta.update(lip_sync=True, sync=sync)
    outputs = [OutputFile(clip, "video", meta), *extract_keyframes(clip, work, int(payload["keyframes"]))]
    outputs += crops.crops_in_venv([o for o in outputs if o.kind == "keyframe"], work, payload.get("crops"))
    recipe = {"engine": "comfy", "graph": name, "source": side.get("source"), "weights": spec.get("weights"),
              "prompt": payload.get("prompt"), "negative_prompt": values["negative_prompt"],
              "image_uri": payload.get("image_uri"), "audio_uri": payload.get("audio_uri"),
              "script": payload.get("script"), "seed": seed, "params": params}
    return HandlerResult(outputs, recipe=recipe, meta={"output_seconds": seconds, "render_seconds": render_s,
                                                        "seconds_per_output_second": bench["seconds_per_output_second"],
                                                        "bench": bench})


HANDLERS = {KIND: generate}


def release() -> None:
    """Give the GPU back before another family's job (manyworlds.worker.runner._switch_to)."""
    comfy.free()
