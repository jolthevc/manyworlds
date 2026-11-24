"""One singing voice per artist (policy music.voice, OFF; manyworlds.singing_voice builds the payloads): job kinds
music.convert_voice and music.train_voice, family music, on the ComfyUI worker image beside music.song.

music.convert_voice payload:
  vocals_uri, instrumental_uri   a music.song job's stems (roles vocals, instrumental)
  asset                          the artist's voice asset {version, adapter, centroid, embedding_model, prompt_uri,
                                 model_uri, index_uri}
  adapter                        the converter's settings (music.voice.adapters.<name>, with `adapter` and
                                 `allow_unverified`): soulx_svc | rvc (CONVERTERS)
  similarity, remix, seed, check (optional {lyrics, asr_model, min_line_ratio}: line coverage of the converted vocal)
Outputs, always in this order (assets/<creator>/<job>/00.wav, 01.wav on every rerun):
  00 vocals   the converted vocal, at the instrumental's rate, the stem's length and loudness (meta converted: true)
  01 mix      the converted vocal over the instrumental
The vocal's meta has `similarity` (voice_similarity against the asset's centroid: median over voiced windows),
`source_similarity` (the stem before conversion, the same way: what the conversion bought), the stem's own
`embedding` (its windows' centroid; the bench compares songs pairwise with it) and, with `check`, `line_coverage`.

music.train_voice payload: {clips: [uris of sung clips], adapter (rvc settings), name, seed, similarity}. Outputs:
00 the RVC model (kind voice_model, .pth), 01 its retrieval index (kind voice_index); meta: the clips' centroid
(the asset's centroid when it is stored), total seconds, epochs.

Licences: an adapter with licence_verified false is refused here too (NonRetryable) unless the payload says the
owner allowed unverified licences (music.voice.allow_unverified_licences). Original voices only: the prompt clip and
the training clips are our own ACE-Step vocals (creator/music_voice.py), never a real singer.

Both upstream stacks pin torch and transformers below the image's (torch 2.8, transformers 4.57.6), so each runs in
its own virtualenv over the image's packages (docker/worker-talkcomfy.Dockerfile, docker/constraints-svc.txt), as a
subprocess from its repo root (their scripts read relative paths): MW_SOULX_PYTHON / MW_SOULX_DIR, MW_RVC_PYTHON /
MW_RVC_DIR. Seeds: each subprocess is started through a small launcher that seeds python, numpy and torch before
running the upstream module, so a rerun redraws the same audio as far as the GPU's kernels allow.

Idempotent: inputs fetched into the job's folder, file names fixed per job, a rerun overwrites the same keys.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from ... import music as music_mod
from ... import singing_voice as sv
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

_state: dict[str, Any] = {"weights": set()}
EMBED_RATE = 16000     # ECAPA takes 16 kHz


def _models_dir() -> Path:
    return Path(os.environ.get("MW_MODEL_DIR", "/models/mw")) / "svc"


# --- audio ------------------------------------------------------------------------------------------------------

def read_audio(path: Path):
    """(samples [n, channels] float32, sample_rate)."""
    import soundfile as sf
    a, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return a, int(sr)


def write_audio(path: Path, a, sr: int) -> Path:
    import numpy as np
    import soundfile as sf
    sf.write(str(path), np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0), int(sr), subtype="PCM_16")
    return path


def resample(a, sr: int, target: int):
    """[n, channels] at `target` (julius, Demucs's resampler, already on the image)."""
    import numpy as np
    if int(sr) == int(target):
        return np.asarray(a, dtype=np.float32)
    import julius
    import torch
    x = torch.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=np.float32).T))
    return julius.resample_frac(x, int(sr), int(target)).numpy().T


def _mono(a):
    import numpy as np
    a = np.asarray(a, dtype=np.float32)
    return a.mean(axis=1) if a.ndim == 2 else a


def _rms(a) -> float:
    import numpy as np
    a = np.asarray(a, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a))) if a.size else 0.0


def remix(converted, conv_sr: int, instrumental, inst_sr: int, source_vocals, settings: dict):
    """(vocal [n, ch], mix [n, ch]) at the instrumental's rate and length: the converted vocal resampled, matched in
    length, RMS-matched to the source stem (`match_loudness`), `vocal_gain_db` added, laid over the instrumental and
    scaled down as a whole when the sum's peak would pass `peak_ceiling` (deterministic, no limiter)."""
    import numpy as np
    inst = np.asarray(instrumental, dtype=np.float32)
    if inst.ndim == 1:
        inst = inst[:, None]
    n, ch = inst.shape
    v = resample(np.asarray(converted, dtype=np.float32).reshape(len(converted), -1), conv_sr, inst_sr)
    v = _mono(v)
    v = v[:n] if len(v) >= n else np.concatenate([v, np.zeros(n - len(v), dtype=np.float32)])
    if settings.get("match_loudness", True):
        want, have = _rms(_mono(source_vocals)), _rms(v)
        if want > 0 and have > 0:
            v = v * (want / have)
    v = v * float(10 ** (float(settings.get("vocal_gain_db", 0.0)) / 20))
    vocal = np.repeat(v[:, None], ch, axis=1).astype(np.float32)
    mix = inst + vocal
    peak, ceiling = float(np.max(np.abs(mix))) if mix.size else 0.0, float(settings.get("peak_ceiling", 0.98))
    if peak > ceiling > 0:
        scale = ceiling / peak
        mix, vocal = mix * scale, vocal * scale
    return vocal.astype(np.float32), mix.astype(np.float32)


# --- voice similarity --------------------------------------------------------------------------------------------

def voiced_windows(mono, sr: int, s: dict) -> list[tuple[int, int]]:
    """[(start, end)] sample ranges of `window_s` windows every `hop_s` in which at least `min_voiced_share` of the
    `frame_s` frames are within `voiced_db` of the loudest frame (singing, not breaths, bleed or silence); at most
    `max_windows`, evenly spread over the song."""
    import numpy as np
    x = np.asarray(mono, dtype=np.float64)
    frame = max(1, int(round(float(s.get("frame_s", 0.05)) * sr)))
    nf = len(x) // frame
    if nf == 0:
        return []
    energy = np.sqrt(np.mean(x[: nf * frame].reshape(nf, frame) ** 2, axis=1))
    top = float(energy.max())
    if top <= 0:
        return []
    voiced = 20 * np.log10(np.maximum(energy, 1e-12) / top) >= float(s.get("voiced_db", -30))
    win = int(round(float(s.get("window_s", 4.0)) * sr))
    hop = max(1, int(round(float(s.get("hop_s", 2.0)) * sr)))
    out = []
    for start in range(0, len(x) - win + 1, hop):
        f0, f1 = start // frame, (start + win) // frame
        if f1 > f0 and voiced[f0:f1].mean() >= float(s.get("min_voiced_share", 0.7)):
            out.append((start, start + win))
    cap = int(s.get("max_windows", 60))
    if len(out) > cap > 0:
        idx = [int(round(i * (len(out) - 1) / (cap - 1))) for i in range(cap)] if cap > 1 else [0]
        out = [out[i] for i in sorted(set(idx))]
    return out


def _embedder(model: str):
    """SpeechBrain ECAPA-TDNN (the designed voices' embedder, handlers/tts_voxcpm._Ecapa): wav at any rate -> list."""
    from . import tts_voxcpm
    return tts_voxcpm._embedder({"embedding": model})


def embed_windows(mono, sr: int, s: dict, embed: Callable | None = None) -> list[list[float]]:
    """One speaker embedding per voiced window (voiced_windows)."""
    embed = embed or _embedder(str(s.get("model") or "speechbrain/spkrec-ecapa-voxceleb"))
    return [embed(mono[a:b], sr) for a, b in voiced_windows(mono, sr, s)]


def voice_similarity(stem, sr: int, asset: dict, s: dict, embed: Callable | None = None) -> dict:
    """A vocal stem against an artist's voice asset: each voiced window's cosine similarity to the asset's centroid
    and their median (`similarity`, None with fewer than `min_windows` voiced windows), plus the stem's own centroid
    (`embedding`). Runs on the CPU as well (ECAPA is small)."""
    mono = _mono(stem)
    embs = embed_windows(mono, sr, s, embed)
    out = sv.score_windows(embs, list(asset.get("centroid") or []), min_windows=int(s.get("min_windows", 3)))
    out.pop("each", None)
    return {**out, "embedding": sv.centroid(embs), "model": s.get("model")}


def prompt_clip(src: Path, dest: Path, seconds: float, s: dict) -> Path:
    """The prompt the zero-shot converter hears: the loudest fully voiced `seconds` stretch of the asset's prompt
    audio (the seed's vocal stem), mono; the whole file when it is shorter or nothing is voiced enough."""
    import numpy as np
    a, sr = read_audio(src)
    mono = _mono(a)
    n = int(round(seconds * sr))
    if len(mono) <= n:
        return write_audio(dest, mono, sr)
    spans = voiced_windows(mono, sr, {**s, "window_s": seconds, "hop_s": float(s.get("prompt_hop_s", 0.5)),
                                      "max_windows": 0})
    if spans:
        start, end = max(spans, key=lambda ab: (_rms(mono[ab[0]:ab[1]]), -ab[0]))
    else:
        start = int(np.argmax([_rms(mono[i:i + n]) for i in range(0, len(mono) - n + 1, max(1, sr // 2))])) * max(1, sr // 2)
        end = start + n
    return write_audio(dest, mono[start:end], sr)


# --- subprocesses ----------------------------------------------------------------------------------------------

LAUNCH = ("import random, runpy, sys\n"
          "seed = int(sys.argv[1]); module = sys.argv[2]; sys.argv = [module] + sys.argv[3:]\n"
          "random.seed(seed)\n"
          "try:\n    import numpy; numpy.random.seed(seed)\nexcept Exception:\n    pass\n"
          "try:\n    import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)\nexcept Exception:\n    pass\n"
          "if module.endswith('.py'):\n    import os; sys.path[:0] = [os.getcwd(), os.path.dirname(os.path.abspath(module))]\n"
          "    runpy.run_path(module, run_name='__main__')\n"
          "else:\n    runpy.run_module(module, run_name='__main__', alter_sys=True)\n")


def launch(python: str, seed: int, module: str, args: list[str]) -> list[str]:
    """The command that runs an upstream module or script seeded (LAUNCH)."""
    return [python, "-c", LAUNCH, str(int(seed)), module, *[str(a) for a in args]]


def run(cmd: list[str], *, cwd: Path, timeout_s: float, env: dict | None = None) -> str:
    """Run an upstream step; its last lines in the error when it fails (a missing module reads as a code fault)."""
    full = {**os.environ, "PYTHONPATH": str(cwd), **(env or {})}
    p = subprocess.run(cmd, cwd=str(cwd), env=full, capture_output=True, text=True, timeout=float(timeout_s))
    if p.returncode != 0:
        tail = "\n".join((p.stderr or p.stdout or "").strip().splitlines()[-15:])
        step = cmd[4] if len(cmd) > 4 and cmd[1] == "-c" else cmd[0]   # the upstream module (launch) or the program
        raise RuntimeError(f"{step} failed ({p.returncode}): {tail}")
    return p.stdout


def _free_song_engine() -> None:
    """The converters run in their own processes: ComfyUI's ACE-Step models and Demucs (music.song, same family)
    give the card back first, or a 16-24 GB card would hold both (a no-op when no ComfyUI server runs)."""
    from . import music_song
    music_song.release()


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _python(var: str, default: str) -> str:
    return os.environ.get(var, default)


def _repo(var: str, default: str) -> Path:
    return Path(os.environ.get(var, default))


def _link(link: Path, target: Path) -> None:
    """`link` -> `target` (the upstream repo reads its weights from fixed relative paths)."""
    target.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and Path(os.readlink(link)) == target:
        return
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def _download(repo: str, files: list[str], local: Path, revision: str | None) -> None:
    """Hugging Face files into `local`, once per volume (huggingface_hub skips a complete file)."""
    from huggingface_hub import hf_hub_download
    for name in files:
        if not (local / name).is_file():
            hf_hub_download(repo, name, local_dir=str(local), revision=revision or None,
                            token=os.environ.get("HF_TOKEN") or None)


# --- converters (adapters: payload adapter.adapter) ---------------------------------------------------------------

def soulx_commands(python: str, seed: int, settings: dict, *, prompt: Path, target: Path, work: Path,
                   device: str) -> list[list[str]]:
    """The SoulX-Singer-SVC steps (pure): F0 for the prompt and the target with the repo's preprocess pipeline
    (RMVPE; vocal separation and MIDI transcription off: our stems come from Demucs), then cli.inference_svc with the
    source pitch kept (auto_shift off, pitch_shift 0 unless the policy says otherwise)."""
    weights = settings.get("weights") or {}
    pre = []
    for name, wav in (("prompt", prompt), ("target", target)):
        pre.append(launch(python, seed, "preprocess.pipeline",
                          ["--audio_path", wav, "--save_dir", work / name, "--language", settings.get("language", "English"),
                           "--device", device, "--vocal_sep", "False", "--midi_transcribe", "False"]))
    args = ["--device", device, "--model_path", Path("pretrained_models/SoulX-Singer") / weights.get("file", "model-svc.pt"),
            "--config", settings.get("config", "soulxsinger/config/soulxsinger.yaml"),
            "--prompt_wav_path", work / "prompt" / "vocal.wav", "--target_wav_path", work / "target" / "vocal.wav",
            "--prompt_f0_path", work / "prompt" / "vocal_f0.npy", "--target_f0_path", work / "target" / "vocal_f0.npy",
            "--save_dir", work / "out", "--n_steps", int(settings.get("n_steps", 32)),
            "--cfg", float(settings.get("cfg", 3.0)), "--pitch_shift", int(settings.get("pitch_shift", 0))]
    if settings.get("fp16"):
        args.append("--fp16")
    if settings.get("auto_shift"):
        args.append("--auto_shift")
    return [*pre, launch(python, seed, "cli.inference_svc", args)]


def _soulx(vocals: Path, ctx: JobContext, settings: dict, asset: dict, seed: int) -> tuple[Path, int]:
    """SoulX-Singer-SVC (Apache-2.0): the stem in the voice of the asset's prompt clip, zero-shot."""
    repo = _repo("MW_SOULX_DIR", "/opt/soulx-singer")
    models = _models_dir() / "soulx"
    weights, pre = settings.get("weights") or {}, settings.get("preprocess") or {}
    ready = ("soulx", weights.get("repo"), weights.get("file"))
    if ready not in _state["weights"]:
        _download(weights["repo"], [weights.get("file", "model-svc.pt"), weights.get("config_file", "config.yaml")],
                  models / "SoulX-Singer", weights.get("revision"))
        _download(pre["repo"], list(pre.get("files") or ["rmvpe/rmvpe.pt"]), models / "SoulX-Singer-Preprocess",
                  pre.get("revision"))
        _link(repo / "pretrained_models", models)
        _state["weights"].add(ready)
    work = Path(ctx.workdir) / "soulx"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    prompt = prompt_clip(ctx.fetch(asset["prompt_uri"], "prompt_full.wav"), work / "prompt_clip.wav",
                         float(settings.get("prompt_seconds", 8.0)), settings.get("voiced") or {})
    _free_song_engine()
    device = _device()
    python = _python("MW_SOULX_PYTHON", "/opt/svc-soulx/bin/python")
    for cmd in soulx_commands(python, seed, settings, prompt=prompt, target=vocals, work=work, device=device):
        run(cmd, cwd=repo, timeout_s=float(settings.get("timeout_s", 900)), env={"HF_HOME": os.environ.get("HF_HOME", "")})
    out = work / "out" / "generated.wav"
    if not out.is_file():
        raise RuntimeError("SoulX-Singer-SVC wrote no generated.wav")
    return out, int(settings.get("sample_rate", 24000))


def rvc_infer_command(python: str, seed: int, settings: dict, *, model_name: str, index: Path | None, source: Path,
                      out: Path) -> list[str]:
    """RVC's headless inference (infer/cli.py): the source pitch kept (pitch 0), RMVPE F0."""
    i = settings.get("infer") or {}
    args = ["--model", model_name, "--input", source, "--output", out, "--f0-method", i.get("f0_method", "rmvpe"),
            "--pitch", int(i.get("pitch", 0)), "--index-rate", float(i.get("index_rate", 0.75)),
            "--protect", float(i.get("protect", 0.33)), "--rms-mix-rate", float(i.get("rms_mix_rate", 1.0)),
            "--resample-sr", 0]
    if index is not None:
        args += ["--index", index]
    return launch(python, seed, "infer/cli.py", args)


def _rvc_assets(repo: Path, settings: dict) -> None:
    """RVC's shared assets (hubert_base, rmvpe, pretrained_v2, mute) into the volume, linked where its scripts look."""
    a = settings.get("assets") or {}
    models = _models_dir() / "rvc"
    ready = ("rvc", a.get("repo"), a.get("revision"))
    if ready in _state["weights"]:
        return
    _download(a["repo"], list(a.get("files") or []), models, a.get("revision"))
    (models / "rmvpe").mkdir(exist_ok=True)
    if (models / "rmvpe.pt").is_file() and not (models / "rmvpe" / "rmvpe.pt").is_file():
        shutil.copyfile(models / "rmvpe.pt", models / "rmvpe" / "rmvpe.pt")
    if (models / "mute.zip").is_file() and not (models / "mute").is_dir():
        shutil.unpack_archive(str(models / "mute.zip"), str(models))
    for sub in ("hubert_base", "rmvpe", "pretrained_v2"):
        _link(repo / "assets" / sub, models / sub)
    if (models / "mute").is_dir():
        _link(repo / "logs" / "mute", models / "mute")
    _state["weights"].add(ready)


def _rvc(vocals: Path, ctx: JobContext, settings: dict, asset: dict, seed: int) -> tuple[Path, int]:
    """RVC (MIT code): the stem through the artist's trained model and its index."""
    repo = _repo("MW_RVC_DIR", "/opt/rvc")
    _rvc_assets(repo, settings)
    if not asset.get("model_uri"):
        raise NonRetryable("the rvc adapter needs the asset's trained model (model_uri)")
    weights = Path(ctx.workdir) / "rvc_weights"
    weights.mkdir(parents=True, exist_ok=True)
    model = ctx.fetch(asset["model_uri"], "voice.pth")
    shutil.copyfile(model, weights / "voice.pth")
    index = ctx.fetch(asset["index_uri"], "voice.index") if asset.get("index_uri") else None
    out = Path(ctx.workdir) / "rvc_out.wav"
    out.unlink(missing_ok=True)
    python = _python("MW_RVC_PYTHON", "/opt/svc-rvc/bin/python")
    run(rvc_infer_command(python, seed, settings, model_name="voice.pth", index=index, source=vocals, out=out),
        cwd=repo, timeout_s=float(settings.get("timeout_s", 900)),
        env={"weight_root": str(weights), "rmvpe_root": str(repo / "assets" / "rmvpe")})
    if not out.is_file():
        raise RuntimeError("RVC wrote no output")
    _, sr = read_audio(out)
    return out, sr


CONVERTERS: dict[str, Callable[[Path, JobContext, dict, dict, int], tuple[Path, int]]] = {
    "soulx_svc": _soulx, "rvc": _rvc}


def _check_adapter(settings: dict, known) -> str:
    name = str(settings.get("adapter") or "")
    if name not in known:
        raise NonRetryable(f"no voice adapter {name!r} (known: {sorted(known)})")
    if not sv.licence_allowed(settings, allow_unverified=bool(settings.get("allow_unverified"))):
        raise NonRetryable(f"voice adapter {name!r} is not licence-verified ({settings.get('licence_notes') or 'unchecked'}) "
                           "and music.voice.allow_unverified_licences is off")
    return name


def transcribe(vocals, sr: int, check: dict) -> str:
    """The converted vocal as heard (Whisper, music.song's transcriber)."""
    from . import music_song
    return music_song.transcribe(vocals, sr, {"asr_model": check.get("asr_model") or "openai/whisper-large-v3",
                                              "language": check.get("language", "en")})


# --- the jobs ----------------------------------------------------------------------------------------------------

def convert(payload: dict, ctx: JobContext) -> HandlerResult:
    settings = payload.get("adapter") or {}
    name = _check_adapter(settings, CONVERTERS)
    asset = payload.get("asset") or {}
    if not payload.get("vocals_uri") or not payload.get("instrumental_uri"):
        raise NonRetryable("payload needs vocals_uri and instrumental_uri (a music.song job's stems)")
    if not asset.get("prompt_uri") or not asset.get("centroid"):
        raise NonRetryable("payload has no voice asset (prompt clip and centroid)")
    if payload.get("seed") is None:
        raise NonRetryable("payload has no seed")
    seed = int(payload["seed"])
    work = Path(ctx.workdir)
    work.mkdir(parents=True, exist_ok=True)
    src = ctx.fetch(payload["vocals_uri"], "vocals.wav")
    inst_path = ctx.fetch(payload["instrumental_uri"], "instrumental.wav")
    stem, stem_sr = read_audio(src)
    inst, inst_sr = read_audio(inst_path)
    random.seed(seed)
    t0 = time.monotonic()
    conv_path, conv_sr = CONVERTERS[name](src, ctx, {**settings, "voiced": payload.get("similarity") or {}}, asset, seed)
    convert_s = round(time.monotonic() - t0, 1)
    conv, conv_rate = read_audio(conv_path)
    conv_sr = conv_rate or conv_sr
    stem_at_inst = resample(stem, stem_sr, inst_sr)
    vocal, mix = remix(conv, conv_sr, inst, inst_sr, stem_at_inst, payload.get("remix") or {})
    seconds = round(mix.shape[0] / inst_sr, 3)
    s = payload.get("similarity") or {}
    t1 = time.monotonic()
    sim = voice_similarity(vocal, inst_sr, asset, s)
    before = voice_similarity(stem, stem_sr, asset, s)
    sim_s = round(time.monotonic() - t1, 1)
    base = {"sample_rate": inst_sr, "channels": int(mix.shape[1]), "seconds": seconds, "converted": True,
            "adapter": name, "voice_version": asset.get("version")}
    vmeta = {**base, "role": "vocals", "similarity": sim["similarity"], "windows": sim["windows"], "p10": sim["p10"],
             "p90": sim["p90"], "embedding": sim["embedding"], "source_similarity": before["similarity"],
             "embedding_model": sim["model"]}
    check = payload.get("check")
    if check and check.get("lyrics"):
        heard = transcribe(vocal, inst_sr, check)
        vmeta["line_coverage"] = music_mod.line_coverage(check["lyrics"], heard,
                                                         min_line_ratio=float(check.get("min_line_ratio", 0.7)))
        vmeta["transcript"] = heard[:2000]
    files = [OutputFile(write_audio(work / "voice_vocals.wav", vocal, inst_sr), "audio", vmeta),
             OutputFile(write_audio(work / "voice_mix.wav", mix, inst_sr), "audio",
                        {**base, "role": "mix", "similarity": sim["similarity"]})]
    recipe = {"adapter": name, "licence": settings.get("licence"), "repo": settings.get("repo"),
              "commit": settings.get("commit"), "voice_version": asset.get("version"), "seed": seed,
              "vocals_uri": payload["vocals_uri"], "instrumental_uri": payload["instrumental_uri"],
              "remix": payload.get("remix") or {}}
    return HandlerResult(files, recipe=recipe, meta={"output_seconds": seconds, "convert_seconds": convert_s,
                                                     "similarity_seconds": sim_s})


def _script(python: str, script: str, args: list) -> list[str]:
    return [python, script, *[str(a) for a in args]]


def rvc_train_commands(python: str, seed: int, settings: dict, *, exp: str, dataset: Path, n_cpu: int,
                       device: str) -> dict[str, list[str]]:
    """RVC v2's training steps from the WebUI (pure; run from the repo root, experiment folder logs/<exp>): slice
    and resample, F0 with RMVPE, ContentVec/HuBERT features, train (the filelist and config are written between
    features and train: write_filelist), and the retrieval index. Plain script runs, not the seeding launcher: the
    trainer spawns worker processes that re-import their main file; its seed is the config's train.seed."""
    t = settings.get("train") or {}
    sr = str(t.get("sample_rate", "40k"))
    sr_int = int(sr.rstrip("k")) * 1000
    exp_dir = f"logs/{exp}"
    version = str(t.get("version", "v2"))
    half = "True" if device == "cuda" else "False"
    f0 = ["cuda", 1, 0, 0, exp_dir, half.lower()] if device == "cuda" else ["cpu", exp_dir, n_cpu, "rmvpe"]
    feats = [device, 1, 0, 0, exp_dir, version, half] if device == "cuda" else [device, 1, 0, exp_dir, version, half]
    return {
        "preprocess": _script(python, "train/preprocess.py",
                              [dataset, sr_int, n_cpu, exp_dir, "False", float(t.get("slice_db", 3.7))]),
        "f0": _script(python, "train/dataset/extract_f0.py", f0),
        "features": _script(python, "train/dataset/extract_hubert_feature.py", feats),
        "train": _script(python, "train/train.py",
                         ["-e", exp, "-sr", sr, "-f0", 1, "-bs", int(t.get("batch_size", 8)), "-g", 0,
                          "-te", int(t.get("epochs", 200)), "-se", int(t.get("save_every", 50)),
                          "-pg", Path("assets") / t.get("pretrained_g", "pretrained_v2/f0G40k.pth"),
                          "-pd", Path("assets") / t.get("pretrained_d", "pretrained_v2/f0D40k.pth"),
                          "-l", 1, "-c", 1 if t.get("cache_in_gpu") else 0, "-sw", 0, "-v", version]),
        "index": _script(python, "train/train_index.py", [exp, version, "assets/indices", n_cpu, "auto"]),
    }


def write_filelist(exp_dir: Path, sr: str, *, seed: int, mute_dir: Path) -> int:
    """logs/<exp>/filelist.txt as the WebUI writes it (one line per sliced clip that has its features and F0, plus
    two mute lines), shuffled with the job's seed; returns the clip count."""
    gt, feat = exp_dir / "0_gt_wavs", exp_dir / "3_feature768"
    f0, nsf = exp_dir / "2a_f0", exp_dir / "2b-f0nsf"
    names = sorted(p.stem for p in gt.glob("*.wav"))
    names = [n for n in names if (feat / f"{n}.npy").is_file() and (f0 / f"{n}.wav.npy").is_file()
             and (nsf / f"{n}.wav.npy").is_file()]
    lines = [f"{gt / (n + '.wav')}|{feat / (n + '.npy')}|{f0 / (n + '.wav.npy')}|{nsf / (n + '.wav.npy')}|0"
             for n in names]
    mute = (f"{mute_dir / '0_gt_wavs' / f'mute{sr}.wav'}|{mute_dir / '3_feature768' / 'mute.npy'}|"
            f"{mute_dir / '2a_f0' / 'mute.wav.npy'}|{mute_dir / '2b-f0nsf' / 'mute.wav.npy'}|0")
    lines += [mute, mute]
    random.Random(seed).shuffle(lines)
    (exp_dir / "filelist.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(names)


def train(payload: dict, ctx: JobContext) -> HandlerResult:
    settings = payload.get("adapter") or {}
    name = _check_adapter(settings, sv.TRAINERS)
    clips = list(payload.get("clips") or [])
    if not clips:
        raise NonRetryable("payload has no clips to train on")
    if payload.get("seed") is None:
        raise NonRetryable("payload has no seed")
    seed, t = int(payload["seed"]), settings.get("train") or {}
    work = Path(ctx.workdir)
    dataset = work / "dataset"
    shutil.rmtree(dataset, ignore_errors=True)
    dataset.mkdir(parents=True)
    s = payload.get("similarity") or {}
    embs, total = [], 0.0
    for i, uri in enumerate(clips):
        p = ctx.fetch(uri, f"clip_{i:03d}.wav")
        a, sr = read_audio(p)
        total += a.shape[0] / sr
        shutil.copyfile(p, dataset / f"clip_{i:03d}.wav")
        embs += embed_windows(_mono(a), sr, s)
    if total < float(t.get("min_seconds", 0)):
        raise NonRetryable(f"{round(total)} s of singing, under the adapter's train.min_seconds {t.get('min_seconds')}")
    if total > float(t.get("max_seconds", math.inf)):
        raise NonRetryable(f"{round(total)} s of singing, over the adapter's train.max_seconds {t.get('max_seconds')}")
    repo = _repo("MW_RVC_DIR", "/opt/rvc")
    _rvc_assets(repo, settings)
    exp = f"mw_{payload.get('name') or 'voice'}"
    exp_dir = repo / "logs" / exp
    shutil.rmtree(exp_dir, ignore_errors=True)     # a rerun starts clean: same steps, same seed
    exp_dir.mkdir(parents=True)
    sr_name = str(t.get("sample_rate", "40k"))
    version = str(t.get("version", "v2"))
    cfg_src = repo / "configs" / ("v1" if sr_name == "40k" else version) / f"{sr_name}.json"
    conf = json.loads(cfg_src.read_text(encoding="utf-8"))
    conf.setdefault("train", {})["seed"] = seed
    (exp_dir / "config.json").write_text(json.dumps(conf), encoding="utf-8")
    _free_song_engine()
    device = _device()
    python = _python("MW_RVC_PYTHON", "/opt/svc-rvc/bin/python")
    cmds = rvc_train_commands(python, seed, settings, exp=exp, dataset=dataset, n_cpu=max(1, (os.cpu_count() or 2) // 2),
                              device=device)
    t0 = time.monotonic()
    timeout = float(settings.get("train_timeout_s", 8400))
    for step in ("preprocess", "f0", "features"):
        run(cmds[step], cwd=repo, timeout_s=timeout)
    n = write_filelist(exp_dir, sr_name, seed=seed, mute_dir=repo / "logs" / "mute")
    if n == 0:
        raise RuntimeError("RVC preprocessing left no training clips")
    (repo / "assets" / "weights").mkdir(parents=True, exist_ok=True)
    run(cmds["train"], cwd=repo, timeout_s=timeout)
    run(cmds["index"], cwd=repo, timeout_s=timeout)
    model = repo / "assets" / "weights" / f"{exp}.pth"
    indices = sorted(exp_dir.glob("added_*.index"))
    if not model.is_file() or not indices:
        raise RuntimeError(f"RVC training wrote no model ({model.name}) or index")
    shutil.copyfile(model, work / "voice_model.pth")
    shutil.copyfile(indices[-1], work / "voice_model.index")
    train_s = round(time.monotonic() - t0, 1)
    center = sv.centroid(embs)
    meta = {"adapter": name, "clips": len(clips), "train_clips": n, "seconds": round(total, 1),
            "epochs": int(t.get("epochs", 200)), "centroid": center, "windows": len(embs),
            "embedding_model": s.get("model")}
    files = [OutputFile(work / "voice_model.pth", "voice_model", {**meta, "role": "model"}),
             OutputFile(work / "voice_model.index", "voice_index", {"role": "index", "adapter": name})]
    recipe = {"adapter": name, "licence": settings.get("licence"), "repo": settings.get("repo"),
              "commit": settings.get("commit"), "assets": settings.get("assets"), "train": t, "seed": seed,
              "clips": clips}
    return HandlerResult(files, recipe=recipe, meta={"train_seconds": train_s, "output_seconds": round(total, 1)})


def release() -> None:
    """Give the GPU back before another family's job: the embedder (the upstream models ran in subprocesses)."""
    from . import tts_voxcpm
    for key in [k for k in tts_voxcpm._state if isinstance(k, tuple) and k[0] == "embed"]:
        tts_voxcpm._state.pop(key, None)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


HANDLERS = {"music.convert_voice": convert, "music.train_voice": train}
