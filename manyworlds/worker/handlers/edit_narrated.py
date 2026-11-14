"""Narrated edits (kind image.narrated): a creator's approved photos, their voice reading a short script, and
burned-in captions, as a vertical video. CPU work (Kokoro TTS + ffmpeg) that rides along on the
image machines already rented for the photos, so it needs no machine of its own.

Payload: images [uri], script, voice (Kokoro id; bf_/bm_ = British), speed, on_screen_text,
width, height, fps. The script is spoken in chunks (sentences, and "..." for a longer
pause) so each caption appears exactly while it is said. Photos get a slow push-in or
pull-out each and hard cuts, the way phone edits look.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

SAMPLE_RATE = 24000
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_state: dict[str, Any] = {}


def chunks(script: str) -> list[tuple[str, float]]:
    """(text, pause after in seconds). '...' is a longer, deliberate pause."""
    out: list[tuple[str, float]] = []
    for part in re.split(r"(\.\.\.|…)", script):
        if part in ("...", "…"):
            if out:
                out[-1] = (out[-1][0], 0.7)
            continue
        for s in re.split(r"(?<=[.!?])\s+", part.strip()):
            s = s.strip()
            if s:
                out.append((s, 0.3))
    return out


def wrap(text: str, width: int = 22) -> str:
    lines, line = [], ""
    for w in text.split():
        if line and len(line) + 1 + len(w) > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    return "\n".join(lines + [line] if line else lines)


def _speak(text: str, voice: str, speed: float):
    import numpy as np
    from kokoro import KPipeline

    lang = voice[0] if voice[:1] in ("a", "b") else "a"
    if lang not in _state:
        _state[lang] = KPipeline(lang_code=lang)
    parts = [np.asarray(a) for _, _, a in _state[lang](text, voice=voice, speed=speed)]
    return np.concatenate(parts) if parts else np.zeros(0, dtype="float32")


def _drawtext(textfile: Path, start: float, end: float, y: str, size: int, box: bool) -> str:
    style = (f"fontfile={FONT}:textfile='{textfile.as_posix()}':fontsize={size}:fontcolor=white:"
             f"line_spacing=10:x=(w-text_w)/2:y={y}:enable='between(t,{start:.2f},{end:.2f})'")
    style += ":box=1:boxcolor=black@0.45:boxborderw=24" if box else ":borderw=5:bordercolor=black"
    return f"drawtext={style}"


def render(payload: dict, ctx: JobContext) -> HandlerResult:
    import numpy as np
    import soundfile as sf

    images, script = payload.get("images") or [], (payload.get("script") or "").strip()
    if not images or not script:
        raise NonRetryable("payload needs images and script")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not installed on this worker")
    voice, speed = payload.get("voice", "af_heart"), float(payload.get("speed", 0.95))
    w, h, fps = int(payload.get("width", 1080)), int(payload.get("height", 1920)), int(payload.get("fps", 30))
    work = ctx.workdir

    # Voice, chunk by chunk, so captions line up with speech.
    lead = 0.4
    audio, cues, t = [np.zeros(int(lead * SAMPLE_RATE), dtype="float32")], [], lead
    for text, pause in chunks(script):
        a = _speak(text, voice, speed).astype("float32")
        cues.append((text, t, t + len(a) / SAMPLE_RATE))
        audio += [a, np.zeros(int(pause * SAMPLE_RATE), dtype="float32")]
        t += len(a) / SAMPLE_RATE + pause
    total = t + 0.6
    audio.append(np.zeros(int(0.6 * SAMPLE_RATE), dtype="float32"))
    wav = work / "voice.wav"
    sf.write(str(wav), np.concatenate(audio), SAMPLE_RATE)

    # Photos: equal share of the voice's length each, slow zoom alternating in and out.
    paths = [ctx.fetch(u, f"img_{i}.png") for i, u in enumerate(images)]
    per = total / len(paths)
    frames = max(int(per * fps), 1)
    inputs, filters = [], []
    for i, p in enumerate(paths):
        inputs += ["-loop", "1", "-t", f"{per:.3f}", "-i", str(p)]
        z = f"1+0.07*on/{frames}" if i % 2 == 0 else f"1.07-0.07*on/{frames}"
        filters.append(f"[{i}:v]scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,crop={w * 2}:{h * 2},"
                       f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={fps},"
                       f"setsar=1[v{i}]")
    filters.append("".join(f"[v{i}]" for i in range(len(paths))) + f"concat=n={len(paths)}:v=1:a=0[base]")

    # Captions while each chunk is spoken; the on-screen hook at the top for the first seconds.
    draws = []
    for i, (text, a, b) in enumerate(cues):
        f = work / f"cap_{i}.txt"
        f.write_text(wrap(text))
        draws.append(_drawtext(f, a, b, "h*0.68", int(h * 0.032), box=False))
    hook = (payload.get("on_screen_text") or "").strip()
    if hook:
        f = work / "hook.txt"
        f.write_text(wrap(hook, 26))
        draws.append(_drawtext(f, 0, min(3.5, total), "h*0.11", int(h * 0.028), box=True))
    filters.append("[base]" + (",".join(draws) if draws else "null") + "[out]")

    out = work / "edit.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-i", str(wav), "-filter_complex", ";".join(filters),
           "-map", "[out]", "-map", f"{len(paths)}:a", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"ffmpeg exited {proc.returncode}: {proc.stderr[-2000:]}")
    seconds = round(total, 2)
    return HandlerResult([OutputFile(out, "video", {"seconds": seconds, "width": w, "height": h})],
                         recipe={"kind": "narrated_edit", "voice": voice, "speed": speed, "script": script,
                                 "images": images, "on_screen_text": hook, "tts": "hexgrad/Kokoro-82M"},
                         meta={"output_seconds": seconds, "captions": len(cues)})


HANDLERS = {"image.narrated": render}
