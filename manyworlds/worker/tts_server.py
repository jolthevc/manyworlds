"""A TTS model in its own environment, served over stdin/stdout (one JSON request per line, one JSON reply per line).

Run by worker/handlers/tts_engines.py with the engine's venv python (the engines pin transformers apart from the
image's 4.57.6: IndexTTS-2.5 4.52.1, qwen-tts 4.57.3; docker/worker-talkcomfy.Dockerfile). Standalone on purpose:
it imports only the standard library, numpy, torch and the engine's own package, never `manyworlds`.

    python tts_server.py indextts --model-dir /root/indextts-2.5 [--bf16]
    python tts_server.py qwen3 --base Qwen/Qwen3-TTS-12Hz-1.7B-Base --design Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign

Requests: {"op": "speak", "text", "ref", "ref_text", "lang", "seed", "out", "emo_vector", "emo_alpha"} (indextts,
qwen3 Base clone), {"op": "design", "text", "instruct", "lang", "seed", "out"} (qwen3 VoiceDesign), {"op": "ping"},
{"op": "quit"}. Replies: {"ok": true, "out", "sample_rate", "seconds"} or {"ok": false, "error"}. Output: 16-bit mono
wav at the model's own rate. Engines print freely: their stdout goes to stderr; the protocol keeps its own pipe.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
import wave

PROTO = None


def _protocol():
    """The reply pipe: the process's original stdout. Afterwards anything the engine prints lands on stderr."""
    global PROTO
    PROTO = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return PROTO

# IndexTTS-2.5's sub-models, pre-fetched into <model_dir>/hf_cache so its downloader never runs: that downloader also
# fetches amphion/MaskGCT's semantic codec (CC-BY-NC-4.0), which 2.5 does not use (licence check 2026-11-25).
INDEXTTS_CACHE = {
    "w2v-bert-2.0": ("facebook/w2v-bert-2.0", None),                                    # MIT
    "campplus_cn_common.bin": ("funasr/campplus", "campplus_cn_common.bin"),            # Apache-2.0
    "bigvgan": ("nvidia/bigvgan_v2_22khz_80band_256x", ("config.json", "bigvgan_generator.pt")),   # MIT
}


def _seed(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write(path: str, wav, sr: int) -> float:
    import numpy as np
    x = np.asarray(wav)
    if x.dtype == np.int16:
        pcm = x.reshape(-1)
    else:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        if peak > 1.0:             # some engines return int-scaled floats
            x = x / 32768.0
        pcm = (np.clip(x, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.astype("<i2").tobytes())
    return round(len(pcm) / float(sr), 3)


def fetch_indextts(model_dir: str, repo: str) -> None:
    """The IndexTTS-2.5 checkpoint and its sub-models, once (idempotent: present files are kept)."""
    from huggingface_hub import hf_hub_download, snapshot_download
    if not os.path.isfile(os.path.join(model_dir, "config.yaml")):
        snapshot_download(repo, local_dir=model_dir)
    cache = os.path.join(model_dir, "hf_cache")
    os.makedirs(cache, exist_ok=True)
    for name, (src, files) in INDEXTTS_CACHE.items():
        target = os.path.join(cache, name)
        if files is None:
            if not os.path.isdir(target) or not os.listdir(target):
                snapshot_download(src, local_dir=target)
        elif isinstance(files, str):
            if not os.path.isfile(target):
                hf_hub_download(src, files, local_dir=cache)
        else:
            os.makedirs(target, exist_ok=True)
            for f in files:
                if not os.path.isfile(os.path.join(target, f)):
                    hf_hub_download(src, f, local_dir=target)


class IndexTTS:
    def __init__(self, args):
        fetch_indextts(args.model_dir, args.repo)
        from indextts.infer_v2_5 import IndexTTS2
        self.tts = IndexTTS2(cfg_path=os.path.join(args.model_dir, "config.yaml"), model_dir=args.model_dir,
                             use_bf16=args.bf16, use_cuda_kernel=False)

    def speak(self, r: dict) -> dict:
        _seed(int(r.get("seed", 0)))
        kw = {"spk_audio_prompt": r["ref"], "text": r["text"], "output_path": None, "lang": r.get("lang", "en"),
              "verbose": False}
        if r.get("emo_vector") is not None:
            kw.update(emo_vector=[float(x) for x in r["emo_vector"]], emo_alpha=float(r.get("emo_alpha", 1.0)))
        if r.get("emo_ref"):
            kw.update(emo_audio_prompt=r["emo_ref"], emo_alpha=float(r.get("emo_alpha", 1.0)))
        out = self.tts.infer(**kw)
        if out is None:
            raise RuntimeError("IndexTTS made no audio")
        sr, wav = out
        return {"out": r["out"], "sample_rate": int(sr), "seconds": _write(r["out"], wav, int(sr))}


class Qwen3:
    def __init__(self, args):
        import torch
        from qwen_tts import Qwen3TTSModel
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.models = {}
        for role, model_id in (("base", args.base), ("design", args.design)):
            if model_id:
                self.models[role] = Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=dtype,
                                                                  attn_implementation=args.attn)

    def _model(self, role: str):
        if role not in self.models:
            raise RuntimeError(f"this server has no {role} model")
        return self.models[role]

    def speak(self, r: dict) -> dict:
        _seed(int(r.get("seed", 0)))
        wavs, sr = self._model("base").generate_voice_clone(text=r["text"], language=r.get("lang", "English"),
                                                            ref_audio=r["ref"], ref_text=r["ref_text"])
        return {"out": r["out"], "sample_rate": int(sr), "seconds": _write(r["out"], wavs[0], int(sr))}

    def design(self, r: dict) -> dict:
        _seed(int(r.get("seed", 0)))
        wavs, sr = self._model("design").generate_voice_design(text=r["text"], language=r.get("lang", "English"),
                                                               instruct=r["instruct"])
        return {"out": r["out"], "sample_rate": int(sr), "seconds": _write(r["out"], wavs[0], int(sr))}


ENGINES = {"indextts": IndexTTS, "qwen3": Qwen3}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=sorted(ENGINES))
    ap.add_argument("--model-dir", default=os.environ.get("MW_INDEXTTS_DIR", "/root/indextts-2.5"))
    ap.add_argument("--repo", default="IndexTeam/IndexTTS-2.5")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--base", default="")
    ap.add_argument("--design", default="")
    ap.add_argument("--attn", default="sdpa")
    args = ap.parse_args(argv)
    _protocol()
    try:
        engine = ENGINES[args.engine](args)
    except Exception as e:  # noqa: BLE001 - reported to the client, which fails the job
        PROTO.write(json.dumps({"ok": False, "error": f"load: {type(e).__name__}: {e}",
                                "trace": traceback.format_exc()[-2000:]}) + "\n")
        return 1
    PROTO.write(json.dumps({"ok": True, "ready": args.engine}) + "\n")
    for raw in sys.stdin:
        try:
            r = json.loads(raw)
            op = r.get("op")
            if op == "quit":
                break
            if op == "ping":
                reply = {"ok": True}
            elif op in ("speak", "design") and hasattr(engine, op):
                reply = {"ok": True, **getattr(engine, op)(r)}
            else:
                reply = {"ok": False, "error": f"unknown op {op!r} for {args.engine}"}
        except Exception as e:  # noqa: BLE001 - one bad request never kills the loaded model
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
        PROTO.write(json.dumps(reply) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
