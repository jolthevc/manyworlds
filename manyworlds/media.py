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
