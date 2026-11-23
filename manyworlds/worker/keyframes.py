"""Keyframes beside every clip (the video review gate judges identity drift and motion from them). Shared by every
clip handler: story clips (comfy.wan_i2v), lip sync (talk.*), dance (comfy.wan_animate2)."""

from __future__ import annotations

from .runner import OutputFile


def keyframe_indices(frames: int, n: int) -> list[int]:
    """n frames spread over the clip, ending on the last (where drift is worst); frame 0 is the
    approved start photo itself, so it is left out."""
    n = max(1, min(n, frames - 1))
    return sorted({round((i + 1) * (frames - 1) / n) for i in range(n)})


def extract_keyframes(video_path, workdir, n: int) -> list[OutputFile]:
    import imageio.v2 as iio
    from PIL import Image

    # The reader's length is unknown (inf) until counted: list(reader) raises MemoryError.
    reader = iio.get_reader(str(video_path), "ffmpeg")
    total = reader.count_frames()
    wanted = keyframe_indices(total, n)
    outs = []
    for idx, frame in enumerate(reader.iter_data()):
        if idx in wanted:
            p = workdir / f"keyframe_{len(outs)}.jpg"
            Image.fromarray(frame).convert("RGB").save(p, quality=90)
            outs.append(OutputFile(p, "keyframe", {"index": len(outs), "frame": idx, "of_frames": total}))
    reader.close()
    return outs
