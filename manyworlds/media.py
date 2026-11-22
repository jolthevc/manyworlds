"""Small media edits on the control plane (no GPU): joining rendered video segments.

ffmpeg comes from the imageio-ffmpeg wheel (a static build), so the control plane image needs no system
package. Segments rendered with the same settings share their codecs and are joined without re-encoding;
anything else is re-encoded once.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        found = shutil.which("ffmpeg")
        if not found:
            raise RuntimeError("no ffmpeg: install imageio-ffmpeg") from None
        return found


def concat(paths: list[Path], out: Path, *, timeout: float = 600) -> Path:
    """Join videos (with their audio) end to end into `out`."""
    if not paths:
        raise ValueError("nothing to join")
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{Path(p).resolve().as_posix()}'\n" for p in paths), encoding="utf-8")
    base = [ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing)]
    tries = (["-c", "copy"],
             ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac"])
    error = ""
    for codec in tries:
        r = subprocess.run(base + codec + ["-movflags", "+faststart", str(out)], capture_output=True, text=True,
                           timeout=timeout)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out
        error = (r.stderr or "")[-500:]
    raise RuntimeError(f"ffmpeg could not join {len(paths)} segments: {error}")


def normalize(path: Path, out: Path, *, width: int, height: int, fps: int, lufs: float = -14.0,
              sample_rate: int = 48000, timeout: float = 600) -> Path:
    """One segment at the joined video's size, frame rate, audio format and loudness, so segments made by
    different engines (a lip-synced talk at 480p, a narrated photo edit at 1080p) cut together cleanly."""
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},"
          f"fps={fps},format=yuv420p,setsar=1")
    cmd = [ffmpeg(), "-y", "-v", "error", "-i", str(path), "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=stereo",
           "-filter_complex", f"[0:v]{vf}[v];[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0,"
                              f"loudnorm=I={lufs}:TP=-1.5:LRA=11,aresample={sample_rate}[a]",
           "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
           "-c:a", "aac", "-b:a", "160k", "-ac", "2", "-shortest", "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg could not normalise {Path(path).name}: {(r.stderr or '')[-500:]}")
    return out


def frame_count(path: Path) -> int:
    import imageio_ffmpeg
    return int(imageio_ffmpeg.count_frames_and_secs(str(path))[0])


def keyframes(video: Path, workdir: Path, n: int, *, timeout: float = 120) -> list[tuple[Path, int, int]]:
    """`n` frames spread over the clip, ending on its last (the video gate's keyframes), as JPEGs:
    [(path, frame index, total frames)]. Plain ffmpeg, so it runs where there is no GPU stack (hosted video)."""
    from .worker.handlers.video_wan import keyframe_indices
    total = frame_count(video)
    out = []
    for i, idx in enumerate(keyframe_indices(total, n)):
        p = Path(workdir) / f"keyframe_{i}.jpg"
        r = subprocess.run([ffmpeg(), "-y", "-v", "error", "-i", str(video), "-vf", f"select=eq(n\,{idx})",
                            "-fps_mode", "passthrough", "-frames:v", "1", "-q:v", "3", str(p)], capture_output=True, text=True,
                           timeout=timeout)
        if r.returncode != 0 or not p.exists():
            raise RuntimeError(f"keyframe {idx} of {video.name}: {r.stderr[-300:]}")
        out.append((p, idx, total))
    return out


def burn_captions(video: Path, ass: Path, out: Path, fontsdir: Path | None = None, *, overlay=None,
                  use_ass: bool | None = None, crf: int = 18, preset: str = "veryfast",
                  timeout: float = 900) -> Path:
    """Burn captions (manyworlds.captions) into `video` -> `out`, audio copied. With libass in this ffmpeg (its
    `ass` filter; `use_ass=None` asks ffmpeg), the ASS script is rendered with the fonts in `fontsdir`. Without
    it, `overlay` (an ffconcat listing of transparent PNGs, or a function that makes one) is laid over the video.
    Paths go into the filter graph quoted, with ':' escaped; a path with a quote in it is refused."""
    import re

    exe = ffmpeg()
    if use_ass is None:
        r = subprocess.run([exe, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=60)
        use_ass = bool(re.search(r"^\s*\S+\s+ass\s+V->V", r.stdout or "", re.M))

    def arg(p: Path) -> str:
        s = Path(p).resolve().as_posix()
        if "'" in s:
            raise ValueError(f"caption path has a quote: {s}")
        return "'" + s.replace(":", "\:") + "'"

    encode = ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "copy",
              "-movflags", "+faststart", str(out)]
    if use_ass:
        vf = f"ass=filename={arg(ass)}" + (f":fontsdir={arg(fontsdir)}" if fontsdir else "")
        cmd = [exe, "-y", "-v", "error", "-i", str(video), "-vf", vf, "-map", "0:v:0", "-map", "0:a?"] + encode
    else:
        if overlay is None:
            raise RuntimeError("this ffmpeg has no libass (no `ass` filter) and no caption overlay was given")
        listing = overlay() if callable(overlay) else Path(overlay)
        cmd = [exe, "-y", "-v", "error", "-i", str(video), "-f", "concat", "-safe", "0", "-i", str(listing),
               "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto:eof_action=pass,format=yuv420p[v]",
               "-map", "[v]", "-map", "0:a?"] + encode
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not Path(out).exists() or Path(out).stat().st_size == 0:
        how = "libass" if use_ass else "overlay"
        raise RuntimeError(f"ffmpeg could not burn captions ({how}) into {Path(video).name}: {(r.stderr or '')[-500:]}")
    return Path(out)
