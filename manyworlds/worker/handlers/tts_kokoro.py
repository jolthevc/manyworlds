"""Voice lines: Kokoro-82M (Apache 2.0), far faster than real time on any GPU.

Kind image.voice rides along on image machines (their image has Kokoro for narrated edits), so a
creator's voice sample never needs a machine of its own. A separate TTS worker image ran 31
benchmark jobs (2026-11-12) and was retired: one more image to build for seconds of work.
Payload: text, voice (e.g. af_heart), speed, lang (a = American English)."""

from __future__ import annotations

from typing import Any

from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

SAMPLE_RATE = 24000
_state: dict[str, Any] = {}


def speak(payload: dict, ctx: JobContext) -> HandlerResult:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    text = (payload.get("text") or "").strip()
    if not text:
        raise NonRetryable("payload has no text")
    lang = payload.get("lang", "a")
    voice = payload.get("voice", "af_heart")
    speed = float(payload.get("speed", 1.0))
    if lang not in _state:
        _state[lang] = KPipeline(lang_code=lang)
    chunks = [audio for _, _, audio in _state[lang](text, voice=voice, speed=speed)]
    audio = np.concatenate([np.asarray(c) for c in chunks])
    path = ctx.workdir / "speech.wav"
    sf.write(str(path), audio, SAMPLE_RATE)
    seconds = round(len(audio) / SAMPLE_RATE, 2)
    return HandlerResult([OutputFile(path, "audio", {"seconds": seconds})],
                         recipe={"model": "hexgrad/Kokoro-82M", "voice": voice, "speed": speed, "text": text},
                         meta={"output_seconds": seconds})


HANDLERS = {"image.voice": speak}
