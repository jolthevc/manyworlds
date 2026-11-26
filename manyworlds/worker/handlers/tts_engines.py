"""Expressive voice engines and automatic casting (owner 2026-11-25: the best commercially usable self-hosted TTS,
cast and judged automatically, no human selection gates). On the talk worker image, family `voice`.

    voice.speak.indextts   IndexTTS-2.5 (bilibili Model Use License: commercial use below 100M MAU / RMB 1bn revenue;
                           its outputs must never train or improve another model; licence check 2026-11-25 in
                           docs/research/2026-11-24-expressive-voice.md §6). Timbre from the creator's reference clip,
                           delivery from its separate emotion control: an 8-value emotion vector (happy, angry, sad,
                           afraid, disgusted, melancholic, surprised, calm) per line, chosen on the control plane
                           (creator/voice_lab.speak_payload, policy voice.engines.indextts.emotions). No delivery word
                           ever reaches the spoken text.
    voice.speak.qwen3      Qwen3-TTS 1.7B Base (Apache-2.0 code and weights): clones the reference clip; it takes no
                           instruction, so delivery comes from the script and the line check's regenerations.
    voice.cast             N candidate voices for a creator: each reference designed from the bible's description
                           (Qwen3-TTS VoiceDesign, Apache-2.0, or VoxCPM2), then the casting script (neutral, warm,
                           excited, quiet...) spoken in it by the creator's engine and every line measured
                           (voice_judge); creator/voice_lab.pick_cast chooses, never a person.

voice.speak.* payloads and outputs are voice.speak's (handlers/tts_voxcpm.py): one wav of the lines, line and clause
timings, per-line WER and transcript, `embedding` with `embed`, plus each line's judge scores (policy
voice.line_check) and `quality` for quality-per-accepted-minute stats. Only `generate` differs per engine; planning,
takes, trimming, joining and the line check are tts_voxcpm's.

Each engine runs in its own venv (its transformers pin clashes with the image's 4.57.6) as a small server
(worker/tts_server.py) kept loaded for the job and released on a family switch.
"""

from __future__ import annotations

import json
import os
import queue as queue_mod
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from ... import voices
from ...speech import wer
from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile
from . import tts_voxcpm as vox

SERVER = Path(__file__).resolve().parents[1] / "tts_server.py"
_servers: dict[tuple, "_Server"] = {}


class _Server:
    """worker/tts_server.py in an engine's venv: one JSON line per request and per reply, with timeouts."""

    def __init__(self, cmd: list[str], *, load_timeout_s: float, env: dict | None = None):
        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
                                     env={**os.environ, **(env or {})})
        self.lines: queue_mod.Queue = queue_mod.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        ready = self._read(load_timeout_s)
        if not ready.get("ok"):
            self.close()
            raise RuntimeError(f"{cmd[2] if len(cmd) > 2 else 'engine'} failed to load: {ready.get('error')}\n"
                               f"{ready.get('trace', '')}")

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def _read(self, timeout_s: float) -> dict:
        try:
            line = self.lines.get(timeout=timeout_s)
        except queue_mod.Empty:
            self.close()
            raise RuntimeError(f"engine server gave no answer in {timeout_s} s") from None
        if line is None:
            raise RuntimeError(f"engine server exited ({self.proc.poll()})")
        return json.loads(line)

    def call(self, request: dict, timeout_s: float) -> dict:
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        reply = self._read(timeout_s)
        if not reply.get("ok"):
            raise RuntimeError(f"{request.get('op')} failed: {reply.get('error')}\n{reply.get('trace', '')}")
        return reply

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - it is being thrown away
            pass
        if self.proc.poll() is None:
            self.proc.kill()


def _settings(params: dict, name: str) -> dict:
    s = (params.get("engines") or {}).get(name)
    if not isinstance(s, dict):
        raise NonRetryable(f"params.engines.{name} is missing (policy voice.engines.{name})")
    return s


def _python(s: dict) -> str:
    return os.environ.get(str(s.get("python_env", "")), "") or str(s.get("python") or sys.executable)


def _server(name: str, s: dict, args: list[str]) -> _Server:
    key = (name, tuple(args))
    if key not in _servers:
        _servers[key] = _Server([_python(s), str(SERVER), name, *args],
                                load_timeout_s=float(s.get("load_timeout_s", 1800)))
    return _servers[key]


def _drop(name: str) -> None:
    for key in [k for k in _servers if k[0] == name]:
        _servers.pop(key).close()


def _read(path: Path, target_sr: int):
    wav, sr = vox.read_wav(path)
    return vox._resample(wav, sr, target_sr) if sr != target_sr else wav


# --- engines: speak(text, ref, ref_text, emotion, strength, seed, out) -> float32 wav at the engine's sample_rate ---
class IndexTTS:
    def __init__(self, params: dict):
        s = _settings(params, "indextts")
        self.s, self.sample_rate = s, int(s.get("sample_rate", 22050))
        args = ["--model-dir", os.environ.get(str(s.get("model_dir_env", "")), "") or str(s.get("model_dir")),
                "--repo", str(s["repo"])] + (["--bf16"] if s.get("bf16") else [])
        self.server = _server("indextts", s, args)

    def speak(self, *, text: str, ref: Path, ref_text: str, emotion: dict | None, strength: float, seed: int,
              out: Path):
        req = {"op": "speak", "text": text, "ref": str(ref), "lang": self.s.get("lang", "en"), "seed": int(seed),
               "out": str(out)}
        vec = (emotion or {}).get("vector")
        if vec:
            cap = float(self.s.get("max_emotion", 1.0))
            req["emo_vector"] = [round(min(cap, max(0.0, float(x) * strength)), 4) for x in vec]
            req["emo_alpha"] = float((emotion or {}).get("alpha", self.s.get("emo_alpha", 1.0)))
        self.server.call(req, float(self.s.get("timeout_s", 300)))
        return _read(out, self.sample_rate)


class Qwen3:
    def __init__(self, params: dict, *, design: bool = False, speak: bool = True):
        s = _settings(params, "qwen3")
        self.s, self.sample_rate = s, int(s.get("sample_rate", 24000))
        args = (["--base", str(s["base"])] if speak else []) + (["--design", str(s["design"])] if design else []) \
            + ["--attn", str(s.get("attn", "sdpa"))]
        self.server = _server("qwen3", s, args)

    def speak(self, *, text: str, ref: Path, ref_text: str, emotion: dict | None, strength: float, seed: int,
              out: Path):
        self.server.call({"op": "speak", "text": text, "ref": str(ref), "ref_text": ref_text,
                          "lang": self.s.get("language", "English"), "seed": int(seed), "out": str(out)},
                         float(self.s.get("timeout_s", 300)))
        return _read(out, self.sample_rate)

    def design(self, *, description: str, text: str, seed: int, out: Path) -> Path:
        self.server.call({"op": "design", "text": text, "instruct": description,
                          "lang": self.s.get("language", "English"), "seed": int(seed), "out": str(out)},
                         float(self.s.get("timeout_s", 300)))
        return out


class VoxCPM:
    """The baseline in the same shape (casting with VoxCPM2: design from the description, ultimate cloning)."""

    def __init__(self, params: dict):
        self.params, self.model = params, vox._tts(params)
        self.sample_rate = int(self.model.sample_rate)

    def speak(self, *, text: str, ref: Path, ref_text: str, emotion: dict | None, strength: float, seed: int,
              out: Path):
        return self.model.generate(seed=seed, text=text, **vox.clone_args(ref, ref_text, "", "ultimate"),
                                   **vox._generation(self.params))

    def design(self, *, description: str, text: str, seed: int, out: Path) -> Path:
        wav = self.model.generate(seed=seed, text=voices.with_instruction(description, text),
                                  **vox._generation(self.params))
        return vox.write_wav(out, wav, self.sample_rate)


ENGINES = {"indextts": IndexTTS, "qwen3": Qwen3, "voxcpm": VoxCPM}
DESIGNERS = {"qwen3_design": lambda params: Qwen3(params, design=True, speak=False), "voxcpm": VoxCPM}


def _engine(name: str, params: dict):
    if name not in ENGINES:
        raise NonRetryable(f"unknown voice engine {name!r}; known: {sorted(ENGINES)}")
    return ENGINES[name](params)


# --- voice.speak.<engine> ------------------------------------------------------------------------------------------
def _speak(engine_name: str, payload: dict, ctx: JobContext) -> HandlerResult:
    p = vox.validate_speak(payload)
    params, cfg = p["params"], p["speech"]
    engine = _engine(engine_name, params)
    sr, asr = int(engine.sample_rate), vox._asr(params)
    ref = ctx.fetch(p["ref_uri"], "reference.wav")
    work = Path(ctx.workdir)
    attempt = max(1, int(getattr(ctx.job, "attempts", 1) or 1))
    counter = iter(range(1 << 30))

    def generate(unit: dict, line: dict, seed: int, strength: float):
        return engine.speak(text=unit["spoken"], ref=ref, ref_text=p["ref_text"], emotion=line.get("emotion"),
                            strength=strength, seed=seed, out=work / f"take_{next(counter):04d}.wav")

    main = vox._voice_lines(p, cfg, None, asr, ref, sr, attempt, strict=True, generate=generate,
                            judge=vox.line_judge(params, ref, sr))
    meta = {**vox._audio_meta(main, sr, "clauses" if cfg["split"] else "whole_line"), "engine": engine_name}
    if p["embed"]:
        meta.update(embedding=vox._embedder(params)(main["audio"], sr),
                    pitch_var=float(vox._pitch_variance(main["audio"], sr, params)))
    out = OutputFile(vox.write_wav(work / "speech.wav", main["audio"], sr), "audio", meta)
    return HandlerResult([out], recipe={"engine": engine_name, "settings": _settings(params, engine_name)
                                        if engine_name != "voxcpm" else {"model": params.get("model")},
                                        "ref_uri": p["ref_uri"], "lines": p["lines"], "takes": p["takes"],
                                        "seed": p["seed"]},
                         meta={"output_seconds": meta["seconds"]})


def speak_indextts(payload: dict, ctx: JobContext) -> HandlerResult:
    return _speak("indextts", payload, ctx)


def speak_qwen3(payload: dict, ctx: JobContext) -> HandlerResult:
    return _speak("qwen3", payload, ctx)


# --- voice.cast ----------------------------------------------------------------------------------------------------
def validate_cast(payload: dict) -> dict:
    """{engine, designer, description, anchor_text, n, seed, script [{text, emotion}], params}, checked."""
    engine, designer = payload.get("engine"), payload.get("designer")
    if engine not in ENGINES:
        raise NonRetryable(f"engine must be one of {sorted(ENGINES)}, got {engine!r}")
    if designer not in DESIGNERS:
        raise NonRetryable(f"designer must be one of {sorted(DESIGNERS)}, got {designer!r}")
    description = vox._text(payload.get("description"), "description")
    anchor = voices.strip_delivery(vox._text(payload.get("anchor_text"), "anchor_text"))
    script = payload.get("script")
    if not isinstance(script, list) or not script:
        raise NonRetryable("payload needs a casting script")
    lines = []
    for i, line in enumerate(script):
        if not isinstance(line, dict):
            raise NonRetryable(f"script[{i}] must be an object")
        words = voices.strip_delivery(vox._text(line.get("text"), f"script[{i}].text"))
        emotion = line.get("emotion")
        if emotion is not None and not isinstance(emotion, dict):
            raise NonRetryable(f"script[{i}].emotion must be an object")
        lines.append({"text": words, "name": str(line.get("name") or f"line{i}"), "emotion": emotion})
    return {"engine": engine, "designer": designer, "description": description, "anchor_text": anchor,
            "n": vox._positive_int(payload.get("n"), "n"), "seed": vox._seed(payload), "script": lines,
            "params": vox._params(payload)}


def cast(payload: dict, ctx: JobContext) -> HandlerResult:
    """Each candidate: a reference designed from the description reading the anchor text (checked by WER), who it
    sounds like (gender, age), then every casting line spoken in it by the engine, trimmed and measured. One audio
    output per candidate (the reference: what becomes the creator's voice), its meta holding every score.

    After the references, each candidate's casting lines as the engine spoke them, joined with params.listen_gap_s of
    silence (kind `audio_lines`: candidate selection reads kind `audio` and never sees them). The owner can then hear
    what each engine made of the voice, not only the designer's reference (2026-11-26: every bake-off clip heard so
    far was the Qwen3 VoiceDesign reference, never an engine's own speech)."""
    from . import voice_judge
    p = validate_cast(payload)
    params, work = p["params"], Path(ctx.workdir)
    trim_cfg = vox.speech_config(params)["trim"]
    asr = vox._asr(params)
    designer = DESIGNERS[p["designer"]](params)
    refs = []
    for i in range(p["n"]):               # all references first, then the designer's memory is handed back
        seed = p["seed"] + i * 1000
        path = designer.design(description=p["description"], text=p["anchor_text"], seed=seed,
                               out=work / f"candidate_{i:02d}.wav")
        refs.append((i, seed, path))
    if p["designer"] == "qwen3_design" and p["engine"] != "qwen3":
        _drop("qwen3")
    elif p["designer"] == "voxcpm" and p["engine"] != "voxcpm":
        for key in [k for k in vox._state if k[0] == "tts"]:
            vox._state.pop(key)
    engine = _engine(p["engine"], params)
    sr, embed = int(engine.sample_rate), vox._embedder(params)
    outputs, spoken, total = [], [], 0.0
    gap = float(params.get("listen_gap_s", 0.6))
    for i, seed, path in refs:
        ref_wav, ref_sr = vox.read_wav(path)
        heard = asr(ref_wav, ref_sr)
        ref_emb = embed(ref_wav, ref_sr)
        judge = voice_judge.LineJudge(params, sr, ref_embedding=ref_emb, embed=embed)
        lines, pieces = [], []
        for li, line in enumerate(p["script"]):
            wav = engine.speak(text=line["text"], ref=path, ref_text=p["anchor_text"], emotion=line["emotion"],
                               strength=1.0, seed=seed + 1 + li, out=work / f"cand{i:02d}_line{li:02d}.wav")
            if trim_cfg:
                wav = vox.trim(wav, sr, **trim_cfg)
            pieces.append(wav)
            said = asr(wav, sr)
            scores = judge(line["text"], wav, said)
            lines.append({"name": line["name"], "text": line["text"],
                          **{k: v for k, v in scores.items() if k not in ("subs",)}})
            total += len(wav) / sr
        seconds = round(len(ref_wav) / ref_sr, 2)
        outputs.append(OutputFile(path, "audio", {
            "index": i, "seed": seed, "role": "cast_reference", "engine": p["engine"], "designer": p["designer"],
            "description": p["description"], "ref_text": p["anchor_text"], "ref_wer": wer(p["anchor_text"], heard),
            "transcript": heard[:1000], "embedding": ref_emb, "who": voice_judge.who(ref_wav, ref_sr, params),
            "lines": lines, "seconds": seconds, "sample_rate": ref_sr}))
        total += seconds
        joined, spans = vox.assemble(pieces, [gap] * max(len(pieces) - 1, 0), sr)
        spoken.append(OutputFile(vox.write_wav(work / f"cand{i:02d}_lines.wav", joined, sr), "audio_lines", {
            "index": i, "role": "cast_lines", "engine": p["engine"], "designer": p["designer"],
            "lines": [{"name": x["name"], "start_s": round(a / sr, 2), "end_s": round(b / sr, 2)}
                      for x, (a, b) in zip(lines, spans)], "seconds": round(len(joined) / sr, 2)}))
    return HandlerResult(outputs + spoken, recipe={"engine": p["engine"], "designer": p["designer"],
                                          "description": p["description"], "seed": p["seed"], "n": p["n"],
                                          "script": p["script"]},
                         meta={"output_seconds": round(total, 2)})


HANDLERS = {"voice.speak.indextts": speak_indextts, "voice.speak.qwen3": speak_qwen3, "voice.cast": cast}


def release() -> None:
    """Stop the engine servers and drop the transcriber, embedder and judges before another family's job."""
    for key in list(_servers):
        _servers.pop(key).close()
    from . import voice_judge
    voice_judge.release()
    vox.release()
