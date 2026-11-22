"""Narrated edits (kind image.narrated): a creator's approved photos, their voice reading a short script, and
burned-in captions, as a vertical video. CPU work (Kokoro TTS + ffmpeg) that rides along on the
image machines already rented for the photos, so it needs no machine of its own.

Payload: images [uri], script, voice (Kokoro id; bf_/bm_ = British), speed, on_screen_text,
width, height, fps. The script is spoken in chunks (sentences, and "..." for a longer
pause) so each caption appears exactly while it is said. Photos get a slow push-in or
pull-out each and hard cuts, the way phone edits look.

With `audio_uri` (a creator's designed voice, already spoken by voice.speak), that wav is the voice instead of
Kokoro, and `cues` [{text, start_s, end_s}] (the voice job's line timings) time the captions; without cues the
script's chunks share the line's length by word count.

With `captions: false` (production sets it when the post's captions are burned in at the finish stage,
manyworlds.captions), no captions are drawn here; the on-screen hook still is. Either way the video's meta keeps the
caption cues ([{text, start_s, end_s}] in the video's time) and the voice's lead and tail, so the finish stage can time
its captions. *Emphasis* asterisks in the script are never spoken or drawn.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..runner import HandlerResult, JobContext, NonRetryable, OutputFile

SAMPLE_RATE = 24000
LEAD_S, TAIL_S = 0.4, 0.6   # silence before and after the voice (production's finish stage reads them too)
FONT ="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
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
    from ... import voices   # a creator's own blend of stock voices, or one stock voice
    parts = [np.asarray(a) for _, _, a in _state[lang](text, voice=voices.voice_arg(_state[lang], voice), speed=speed)]
    return np.concatenate(parts) if parts else np.zeros(0, dtype="float32")


def _drawtext(textfile: Path, start: float, end: float, y: str, size: int, box: bool) -> str:
    style = (f"fontfile={FONT}:textfile='{textfile.as_posix()}':fontsize={size}:fontcolor=white:"
             f"line_spacing=10:x=(w-text_w)/2:y={y}:enable='between(t,{start:.2f},{end:.2f})'")
    style += ":box=1:boxcolor=black@0.45:boxborderw=24" if box else ":borderw=5:bordercolor=black"
    return f"drawtext={style}"


def supplied_cues(payload: dict, script: str, lead: float, seconds: float) -> list[tuple[str, float, float]]:
    """Caption timings over a supplied voice line of `seconds` that starts at `lead`: the payload's cues when given,
    else the script's chunks sharing the line by word count."""
    given = [(str(c.get("text") or "").strip(), c.get("start_s"), c.get("end_s")) for c in payload.get("cues") or []
             if isinstance(c, dict)]
    given = [(t, lead + float(a), lead + float(b)) for t, a, b in given if t and a is not None and b is not None]
    if given:
        return given
    parts = [text for text, _ in chunks(script)]
    words = [max(len(p.split()), 1) for p in parts]
    out, t = [], lead
    for p, n in zip(parts, words):
        d = seconds * n / sum(words)
        out.append((p, t, t + d))
        t += d
    return out


def render(payload: dict, ctx: JobContext) -> HandlerResult:
    import numpy as np
    import soundfile as sf

    images = payload.get("images") or []
    script = (payload.get("script") or "").replace("*", "").strip()   # *emphasis* is for burned captions only
    audio_uri = payload.get("audio_uri")
    draw_captions = payload.get("captions", True) is not False
    if not images or not (script or audio_uri):
        raise NonRetryable("payload needs images and script (or audio_uri)")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not installed on this worker")
    voice, speed = payload.get("voice", "af_heart"), float(payload.get("speed", 0.95))
    w, h, fps = int(payload.get("width", 1080)), int(payload.get("height", 1920)), int(payload.get("fps", 30))
    work = ctx.workdir

    lead, tail = LEAD_S, TAIL_S
    wav = work / "voice.wav"
    if audio_uri:
        # The designed voice's finished line: padded like a Kokoro read, captions on its line timings.
        line, rate = sf.read(str(ctx.fetch(audio_uri, "line.wav")), dtype="float32", always_2d=False)
        line = line.mean(axis=1) if line.ndim > 1 else line
        seconds = len(line) / rate
        cues = supplied_cues(payload, script, lead, seconds)
        total = lead + seconds + tail
        sf.write(str(wav), np.concatenate([np.zeros(int(lead * rate), dtype="float32"), line,
                                           np.zeros(int(tail * rate), dtype="float32")]), rate)
    else:
        # Voice, chunk by chunk, so captions line up with speech.
        audio, cues, t = [np.zeros(int(lead * SAMPLE_RATE), dtype="float32")], [], lead
        for text, pause in chunks(script):
            a = _speak(text, voice, speed).astype("float32")
            cues.append((text, t, t + len(a) / SAMPLE_RATE))
            audio += [a, np.zeros(int(pause * SAMPLE_RATE), dtype="float32")]
            t += len(a) / SAMPLE_RATE + pause
        total = t + tail
        audio.append(np.zeros(int(tail * SAMPLE_RATE), dtype="float32"))
        sf.write(str(wav), np.concatenate(audio), SAMPLE_RATE)

    # Photos: equal share of the voice's length each, slow zoom alternating in and out.
    paths = [ctx.fetch(u, f"img_{i}.png") for i, u in enumerate(images)]
    per = total / len(paths)
    frames = max(int(per * fps), 1)
    inputs, filters = [], []
    for i, p in enumerate(paths):
        # One frame in: zoompan makes exactly `frames` frames from it. (A looped input would make
        # `frames` frames from every input frame, so the first photo filled the whole video.)
        inputs += ["-i", str(p)]
        z = f"1+0.07*on/{frames}" if i % 2 == 0 else f"1.07-0.07*on/{frames}"
        filters.append(f"[{i}:v]scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,crop={w * 2}:{h * 2},"
                       f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps={fps},"
                       f"setsar=1[v{i}]")
    filters.append("".join(f"[v{i}]" for i in range(len(paths))) + f"concat=n={len(paths)}:v=1:a=0[base]")

    # Captions while each chunk is spoken (unless burned in later); the on-screen hook at the top for the first seconds.
    draws = []
    for i, (text, a, b) in enumerate(cues if draw_captions else []):
        f = work / f"cap_{i}.txt"
        f.write_text(wrap(text))
        draws.append(_drawtext(f, a, b, "h*0.68", int(h * 0.032), box=False))
    hook = (payload.get("on_screen_text") or "").strip()
    if hook:
        f = work / "hook.txt"
        f.write_text(wrap(hook, 26))
        draws.append(_drawtext(f, 0, min(3.5, total), "h*0.11", int(h * 0.028), box=True))
    filters.append("[base]" + (",".join(draws) if draws else "null") + "[out]")
    # Voice at short-form loudness (about -14 LUFS), so every edit plays at the same level.
    filters.append(f"[{len(paths)}:a]loudnorm=I={payload.get('loudness_lufs', -14)}:TP=-1.5:LRA=11,aresample=48000,"
                   "aformat=sample_fmts=fltp:channel_layouts=stereo[aout]")

    out = work / "edit.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-i", str(wav), "-filter_complex", ";".join(filters),
           "-map", "[out]", "-map", "[aout]", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"ffmpeg exited {proc.returncode}: {proc.stderr[-2000:]}")
    seconds = round(total, 2)
    timed = [{"text": text, "start_s": round(a, 3), "end_s": round(b, 3)} for text, a, b in cues]
    return HandlerResult([OutputFile(out, "video", {"seconds": seconds, "width": w, "height": h, "lead_s": lead,
                                                    "tail_s": tail, "cues": timed, "captions": draw_captions})],
                         recipe={"kind": "narrated_edit", "voice": voice, "speed": speed, "script": script,
                                 "images": images, "on_screen_text": hook,
                                 **({"audio_uri": audio_uri} if audio_uri else {"tts": "hexgrad/Kokoro-82M"})},
                         meta={"output_seconds": seconds, "captions": len(cues) if draw_captions else 0})


HANDLERS = {"image.narrated": render}
