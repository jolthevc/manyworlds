"""Dependency-free stand-in handlers. They exercise the whole pipeline (queue, leases,
storage, ledger, provenance) on a laptop with no GPU, and are what the tests use."""

from __future__ import annotations

import math
import random
import struct
import time
import wave
import zlib
from pathlib import Path

from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile


def _png(path: Path, width: int, height: int, rgb: tuple[int, int, int]) -> None:
    row = b"\x00" + bytes(rgb) * width
    raw = zlib.compress(row * height)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", raw) + chunk(b"IEND", b""))


def _maybe_fail(payload: dict) -> None:
    if payload.get("fail") == "always":
        raise RuntimeError("fake handler asked to fail")
    if payload.get("fail") == "fatal":
        raise NonRetryable("fake handler asked to fail permanently")
    if payload.get("fail") == "gpu":
        raise RuntimeError("CUDA error: out of memory (simulated)")
    time.sleep(float(payload.get("sleep", 0)))


def fake_image(payload: dict, ctx: JobContext) -> HandlerResult:
    _maybe_fail(payload)
    n = int(payload.get("n", 4))
    seed = payload.get("seed") or random.randrange(1 << 30)
    rng = random.Random(seed)
    outs = []
    for i in range(n):
        p = ctx.workdir / f"img_{i}.png"
        _png(p, 64, 80, (rng.randrange(256), rng.randrange(256), rng.randrange(256)))
        outs.append(OutputFile(p, "image", {"width": 64, "height": 80, "seed": seed + i}))
    return HandlerResult(outs, recipe={"model": "fake", "prompt": payload.get("prompt"), "seed": seed})


def fake_video(payload: dict, ctx: JobContext) -> HandlerResult:
    _maybe_fail(payload)
    p = ctx.workdir / "clip.mp4"
    p.write_bytes(b"fake-video:" + str(payload.get("prompt")).encode())
    return HandlerResult([OutputFile(p, "video", {"frames": payload.get("frames", 81)})],
                         recipe={"model": "fake", "prompt": payload.get("prompt")})


def fake_tts(payload: dict, ctx: JobContext) -> HandlerResult:
    _maybe_fail(payload)
    text = str(payload.get("text", ""))
    seconds = max(0.5, len(text) / 15)
    rate = 24000
    p = ctx.workdir / "speech.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(struct.pack("<h", int(3000 * math.sin(2 * math.pi * 220 * t / rate)))
                          for t in range(int(seconds * rate)))
        w.writeframes(frames)
    return HandlerResult([OutputFile(p, "audio", {"seconds": round(seconds, 2)})],
                         recipe={"model": "fake", "voice": payload.get("voice"), "text": text})


HANDLERS = {"fake.image": fake_image, "fake.video": fake_video, "fake.tts": fake_tts}
