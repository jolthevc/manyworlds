"""Small media edits on the control plane (no GPU): joining rendered video segments (hard cuts, or the motion
transitions below), bringing a segment to one size and frame rate, reading a file's length and size, burning captions.

ffmpeg comes from the imageio-ffmpeg wheel (a static build), so the control plane image needs no system
package. Segments rendered with the same settings share their codecs and are joined without re-encoding;
segments that differ (size, frame rate, codec, audio) are normalised first, so a copy-join never mixes them.

Frame rate: published video has one rate (policy `finishing.fps`). A slower source (Wan 2.2 clips at 16 fps,
InfiniteTalk at 25) is brought up by motion interpolation (`finishing.interpolate`: ffmpeg minterpolate, or a blend
for very short clips), never by repeating frames, unless interpolation is switched off. `fps_filter` builds that
step; the finish stage, `normalize`, the transition joiner and the talk worker's walk-and-talk resample all use it.

Transitions (`concat(..., transitions=[...])`, one per join, policy `finishing.transitions`): cut (the default),
match (a cut that first trims each side to a given time, snapped to the nearest frame: the motion peak), whip (a
directional blur and slide), blur (both sides blur through a dissolve) and flash (a short dip to white), each a few
frames at the output rate. Any transition but a cut re-encodes the join through one filter graph (xfade).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

TRANSITIONS = ("cut", "match", "whip", "blur", "flash")
# minterpolate's own option names (ffmpeg's vocabulary, not tunables): the policy block may set any of them.
_MINTERPOLATE_OPTIONS = ("mi_mode", "mc_mode", "me_mode", "me", "mb_size", "search_param", "vsbmc", "scd",
                         "scd_threshold")


def ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        found = shutil.which("ffmpeg")
        if not found:
            raise RuntimeError("no ffmpeg: install imageio-ffmpeg") from None
        return found


def _fmt(v: float) -> str:
    """A filter number without float noise: 30.0 -> '30', 0.2 -> '0.2'."""
    s = f"{float(v):.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _finishing(policy) -> dict:
    """The finishing block of `policy`; without one, the shipped policy's (callers that hold a policy pass it), or {}
    when even that can't be read (then frame rates are matched by repeating frames, as before interpolation)."""
    if policy is None:
        try:
            from .policy import load
            policy = load()
        except Exception:  # noqa: BLE001 - no policy here: plain behaviour
            return {}
    return policy.get("finishing", default={}) or {}


def _encode_args(fin: dict) -> list[str]:
    enc = fin.get("encode") or {}
    return ["-c:v", "libx264", "-preset", str(enc.get("preset", "veryfast")), "-crf", str(enc.get("crf", 18)),
            "-pix_fmt", "yuv420p"]


# --- frame rate -------------------------------------------------------------------------------------------------

def fps_filter(src_fps: float | None, target: float | None, interpolate: dict | None = None, *,
               seconds: float | None = None) -> str | None:
    """The step that brings a video from `src_fps` to `target`: None when it is already there (or no target);
    plain `fps=` when the source is faster (frames are dropped), its rate is unknown, or interpolation is off;
    otherwise the `interpolate` method: minterpolate (motion-compensated new frames, the policy's options), blend
    (ffmpeg `framerate`, a cross-blend; also any clip shorter than `blend_below_seconds`, where motion estimation
    has too little to work with) or duplicate (repeat frames)."""
    if not target:
        return None
    target, src = float(target), float(src_fps or 0)
    if src and abs(src - target) < 0.01:
        return None
    t = _fmt(target)
    c = interpolate or {}
    if not src or src > target or not c.get("enabled"):
        return f"fps={t}"
    method = str(c.get("method") or "minterpolate")
    short = c.get("blend_below_seconds")
    if method == "blend" or (method == "minterpolate" and seconds is not None and short and seconds < float(short)):
        return f"framerate=fps={t}"
    if method == "minterpolate":
        return "minterpolate=" + ":".join([f"fps={t}"] + [f"{k}={c[k]}" for k in _MINTERPOLATE_OPTIONS
                                                          if c.get(k) is not None])
    if method == "duplicate":
        return f"fps={t}"
    raise ValueError(f"unknown interpolation method {method!r}: minterpolate, blend or duplicate")


def fit_filter(width: int, height: int, flags: str | None = None) -> str:
    """Scale to cover width x height keeping the aspect, then centre-crop to it (a no-op crop when the aspect
    already matches)."""
    f = f":flags={flags}" if flags else ""
    return (f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=increase{f},"
            f"crop={int(width)}:{int(height)}")


# --- joining ----------------------------------------------------------------------------------------------------

def _same(infos: list[dict]) -> bool:
    """Whether segments can be copy-joined: same frame size, frame rate, video codec and audio presence."""
    def key(i: dict):
        fps = i.get("fps")
        return (i.get("width"), i.get("height"), round(float(fps), 2) if fps else None, i.get("vcodec"),
                bool(i.get("audio")))
    return len({key(i) for i in infos}) <= 1


def _target(infos: list[dict], width: int | None, height: int | None, fps: float | None,
            fin: dict) -> tuple[int, int, float]:
    """The size and rate a mixed join is brought to: the given ones, else the largest segment's size, and the
    segments' own rate when they share one (silent motion clips keep Wan's 16 fps for the talk worker), else the
    policy's delivery rate (`finishing.fps`), else the fastest segment's."""
    sized = [i for i in infos if i.get("width") and i.get("height")]
    if not (width and height):
        if not sized:
            raise RuntimeError("can't read the frame size of the segments to join")
        big = max(sized, key=lambda i: i["width"] * i["height"])
        width, height = big["width"], big["height"]
    rates = {round(float(i["fps"]), 2) for i in infos if i.get("fps")}
    shared = next(iter(rates)) if len(rates) == 1 and all(i.get("fps") for i in infos) else None
    rate = fps or shared or fin.get("fps") or max(rates, default=0)
    if not rate:
        raise RuntimeError("can't read the frame rate of the segments to join")
    return int(width), int(height), float(rate)


def concat(paths: list[Path], out: Path, *, timeout: float = 600, transitions: list | None = None,
           policy=None, width: int | None = None, height: int | None = None, fps: float | None = None) -> Path:
    """Join videos (with their audio) end to end into `out`. `transitions`: one per join (a name from TRANSITIONS
    or a dict {type, frames, direction, out_at, in_at}); None or all cuts is a plain join. Segments that differ in
    size, frame rate, codec or audio are first normalised to one (`width`/`height`/`fps`, else the largest size and
    the policy rate), so a copy-join never mixes them."""
    if not paths:
        raise ValueError("nothing to join")
    paths, out = [Path(p) for p in paths], Path(out)
    fin = _finishing(policy) if transitions else {}
    joins = joins_for(transitions, len(paths), fin)
    if any(j["type"] != "cut" for j in joins):
        return join_with_transitions(paths, out, joins, timeout=timeout, policy=policy, width=width,
                                     height=height, fps=fps)
    if len(paths) > 1:
        infos = [probe(p, timeout=timeout) for p in paths]
        if not _same(infos):
            fin = fin or _finishing(policy)
            w, h, rate = _target(infos, width, height, fps, fin)
            paths = [normalize(p, out.with_name(f"{out.stem}_norm{i:02d}.mp4"), width=w, height=h, fps=rate,
                               timeout=timeout, policy=policy) for i, p in enumerate(paths)]
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


def joins_for(transitions: list | None, n: int, fin: dict | None = None) -> list[dict]:
    """One join spec per gap between `n` segments: {type, ...}. None = all cuts; names are TRANSITIONS; a
    transition without its policy block (`finishing.transitions.<type>`) is refused (cut and match need none)."""
    if not transitions:
        return [{"type": "cut"} for _ in range(max(0, n - 1))]
    if len(transitions) != n - 1:
        raise ValueError(f"{len(transitions)} transitions for {n} segments (need {n - 1})")
    blocks = (fin or {}).get("transitions") or {}
    out = []
    for t in transitions:
        spec = {"type": t} if isinstance(t, str) or t is None else dict(t)
        spec["type"] = str(spec.get("type") or "cut")
        if spec["type"] not in TRANSITIONS:
            raise ValueError(f"unknown transition {spec['type']!r}; known: {list(TRANSITIONS)}")
        if spec["type"] not in ("cut", "match") and spec["type"] not in blocks:
            raise ValueError(f"no finishing.transitions.{spec['type']} in the policy")
        out.append(spec)
    return out


def _frames(spec: dict, block: dict, fin: dict) -> int:
    """A transition's length in frames: the join's own, else its block's, clamped to `transitions.frames_range`."""
    n = int(spec.get("frames") or block.get("frames") or 0)
    lo_hi = (fin.get("transitions") or {}).get("frames_range")
    if lo_hi:
        n = min(int(lo_hi[1]), max(int(lo_hi[0]), n))
    if n <= 0:
        raise ValueError(f"transition {spec['type']!r} has no length in frames")
    return n


def _snap(t, rate: float) -> float | None:
    """A time snapped to the nearest frame at `rate` (the match cut's 'frame closest to the motion peak')."""
    if t is None:
        return None
    return round(float(t) * rate) / rate


def transition_graph(segments: list[dict], joins: list[dict], *, width: int, height: int, fps: float,
                     fin: dict) -> tuple[str, str, str | None]:
    """The filter graph that joins `segments` ({seconds, fps, audio} per input, in order) with `joins` (joins_for),
    at width x height and `fps`: (graph, video label, audio label or None when no segment has audio).

    Each input is trimmed (match cuts), brought to the rate (fps_filter with `finishing.interpolate`), fitted to the
    size, and given one time base so xfade accepts it; a silent segment among voiced ones gets silence. Cuts and
    match cuts concatenate; whip, blur and flash cross over `frames` frames (xfade, the audio acrossfade), never
    more than half of either side."""
    rate, interp = float(fps), fin.get("interpolate")
    blocks = fin.get("transitions") or {}
    sr = int((fin.get("loudness") or {}).get("sample_rate", 48000))
    flags = (fin.get("output") or {}).get("flags")
    audio = any(s.get("audio") for s in segments)
    lines, lengths = [], []
    for i, s in enumerate(segments):
        src_rate = float(s.get("fps") or rate)
        into = joins[i - 1] if i > 0 and joins[i - 1]["type"] == "match" else {}
        outof = joins[i] if i < len(joins) and joins[i]["type"] == "match" else {}
        start, end = _snap(into.get("in_at"), src_rate), _snap(outof.get("out_at"), src_rate)
        total = float(s.get("seconds") or 0)
        length = (min(end, total) if end is not None and total else end if end is not None else total) - (start or 0)
        if length <= 0:
            raise ValueError(f"segment {i} has no length to join (trimmed to {start}-{end} of {total} s)")
        lengths.append(length)
        trim = ""
        if start or end is not None:
            trim = "trim=" + ":".join(([f"start={_fmt(start)}"] if start else [])
                                      + ([f"end={_fmt(end)}"] if end is not None else [])) + ",setpts=PTS-STARTPTS,"
        step = fps_filter(s.get("fps"), rate, interp, seconds=length) or f"fps={_fmt(rate)}"
        lines.append(f"[{i}:v]setpts=PTS-STARTPTS,{trim}{step},{fit_filter(width, height, flags)},setsar=1,"
                     f"format=yuv420p,settb=AVTB[v{i}]")
        if audio:
            if s.get("audio"):
                atrim = ""
                if start or end is not None:
                    atrim = "atrim=" + ":".join(([f"start={_fmt(start)}"] if start else [])
                                                + ([f"end={_fmt(end)}"] if end is not None else [])) + ","
                # cut or padded to the picture's length, so sound never drifts from picture across joins
                lines.append(f"[{i}:a]asetpts=PTS-STARTPTS,{atrim}asetpts=PTS-STARTPTS,aresample={sr},"
                             f"atrim=duration={_fmt(length)},apad=whole_dur={_fmt(length)},"
                             f"aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]")
            else:
                lines.append(f"anullsrc=r={sr}:cl=stereo,atrim=duration={_fmt(length)},"
                             f"aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]")
    v, a, cur = "v0", "a0" if audio else None, lengths[0]
    for i in range(1, len(segments)):
        j, nv, na = joins[i - 1], f"j{i}", f"k{i}"
        if j["type"] in ("cut", "match"):
            lines.append(f"[{v}][v{i}]concat=n=2:v=1:a=0[{nv}]")
            if audio:
                lines.append(f"[{a}][a{i}]concat=n=2:v=0:a=1[{na}]")
            cur += lengths[i]
        else:
            block = blocks[j["type"]] or {}
            d = min(_frames(j, block, fin) / rate, cur / 2, lengths[i] / 2)
            off = cur - d
            left, right = v, f"v{i}"
            if j["type"] == "blur":
                sigma = _fmt(float(j.get("sigma") or block["sigma"]))
                lines.append(f"[{v}]gblur=sigma={sigma}:enable='gte(t,{_fmt(off)})'[{nv}b]")
                lines.append(f"[v{i}]gblur=sigma={sigma}:enable='lt(t,{_fmt(d)})'[v{i}b]")
                left, right, xf = f"{nv}b", f"v{i}b", str(block["xfade"])
            elif j["type"] == "whip":
                direction = str(j.get("direction") or block.get("direction") or "left")
                xf = (block.get("xfade") or {}).get(direction)
                if not xf:
                    raise ValueError(f"whip direction {direction!r} has no xfade in finishing.transitions.whip")
                px = int(j.get("blur_px") or block["blur_px"])
                sx, sy = (px, 1) if direction in ("left", "right") else (1, px)
                lines.append(f"[{v}]avgblur=sizeX={sx}:sizeY={sy}:enable='gte(t,{_fmt(off)})'[{nv}b]")
                lines.append(f"[v{i}]avgblur=sizeX={sx}:sizeY={sy}:enable='lt(t,{_fmt(d)})'[v{i}b]")
                left, right = f"{nv}b", f"v{i}b"
            else:   # flash
                xf = str(block["xfade"])
            lines.append(f"[{left}][{right}]xfade=transition={xf}:duration={_fmt(d)}:offset={_fmt(off)}[{nv}]")
            if audio:
                lines.append(f"[{a}][a{i}]acrossfade=d={_fmt(d)}[{na}]")
            cur += lengths[i] - d
        v, a = nv, (na if audio else None)
    return ";".join(lines), v, a


def _with_video_seconds(info: dict, path: Path) -> dict:
    """The probe with `seconds` as the video stream's own length (a decode count): xfade offsets fall inside the
    picture even when a segment's audio runs a little longer than it."""
    try:
        import imageio_ffmpeg
        secs = float(imageio_ffmpeg.count_frames_and_secs(str(path))[1])
    except Exception:  # noqa: BLE001 - keep the container length
        return info
    return {**info, "seconds": secs} if secs > 0 else info


def join_with_transitions(paths: list[Path], out: Path, joins: list[dict], *, timeout: float = 600, policy=None,
                          width: int | None = None, height: int | None = None, fps: float | None = None) -> Path:
    """The segments joined through one filter graph (transition_graph), re-encoded once (`finishing.encode`)."""
    fin = _finishing(policy)
    infos = [_with_video_seconds(probe(p, timeout=timeout), p) for p in paths]
    if any(not i.get("seconds") for i in infos):
        raise RuntimeError("can't read the length of every segment to join")
    w, h, rate = _target(infos, width, height, fps, fin)
    graph, v, a = transition_graph(infos, joins, width=w, height=h, fps=rate, fin=fin)
    cmd = [ffmpeg(), "-y", "-v", "error"]
    for p in paths:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", graph, "-map", f"[{v}]"] + _encode_args(fin)
    if a:
        cmd += ["-map", f"[{a}]", "-c:a", "aac", "-b:a", str((fin.get("encode") or {}).get("audio_bitrate", "160k")),
                "-ac", "2"]
    cmd += ["-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg could not join {len(paths)} segments with transitions: {(r.stderr or '')[-500:]}")
    return out


def trim_start(path: Path, out: Path, *, frames: int = 1, timeout: float = 600) -> Path:
    """A silent clip without its first `frames` frames, re-encoded (H.264, yuv420p) into `out`; 0 re-encodes it
    whole. A chained motion clip starts on the frame the one before it ended on, so a join drops that repeat; every
    clip of a join goes through here, so the joined clips share one encoding. Audio is not kept (motion clips have
    none: the talk job adds the voice)."""
    vf = f"trim=start_frame={max(0, int(frames))},setpts=PTS-STARTPTS,format=yuv420p"
    cmd = [ffmpeg(), "-y", "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf", vf, "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg could not trim {Path(path).name}: {(r.stderr or '')[-500:]}")
    return out


def normalize_filter(width: int, height: int, fps: float, *, src_fps: float | None = None,
                     interpolate: dict | None = None, seconds: float | None = None, flags: str | None = None) -> str:
    """normalize's video chain: the frame rate first (interpolating at the source size is cheaper), then the fit to
    width x height, yuv420p, square pixels. Always ends on the rate (`fps=` when it is already there: constant)."""
    step = fps_filter(src_fps, fps, interpolate, seconds=seconds) or f"fps={_fmt(fps)}"
    return f"{step},{fit_filter(width, height, flags)},format=yuv420p,setsar=1"


def normalize(path: Path, out: Path, *, width: int, height: int, fps: float, lufs: float = -14.0,
              sample_rate: int = 48000, timeout: float = 600, policy=None, interpolate: dict | None = None) -> Path:
    """One segment at the joined video's size, frame rate, audio format and loudness, so segments made by
    different engines (a lip-synced talk at 480p and 25 fps, a narrated photo edit at 1080p and 30) cut together
    cleanly. A slower segment is motion-interpolated (`interpolate`, else the policy's `finishing.interpolate`), not
    frame-repeated. A segment without audio gets silence, so every normalised segment carries a track."""
    info = probe(path, timeout=timeout)
    fin = _finishing(policy)
    interp = interpolate if interpolate is not None else fin.get("interpolate")
    vf = normalize_filter(width, height, fps, src_fps=info.get("fps"), interpolate=interp, seconds=info.get("seconds"),
                          flags=(fin.get("output") or {}).get("flags"))
    cmd = [ffmpeg(), "-y", "-v", "error", "-i", str(path), "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=stereo"]
    if info.get("audio"):
        cmd += ["-filter_complex", f"[0:v]{vf}[v];[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0,"
                                   f"loudnorm=I={lufs}:TP=-1.5:LRA=11,aresample={sample_rate}[a]",
                "-map", "[v]", "-map", "[a]"]
    else:
        cmd += ["-filter_complex", f"[0:v]{vf}[v]", "-map", "[v]", "-map", "1:a"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "160k", "-ac", "2", "-shortest", "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg could not normalise {Path(path).name}: {(r.stderr or '')[-500:]}")
    return out


def probe(path: Path, *, timeout: float = 60) -> dict:
    """{seconds, width, height, fps, vcodec, audio} of a media file, read from ffmpeg's banner (no decode); a field
    it can't read is None (audio: False)."""
    r = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, timeout=timeout)
    err = r.stderr or ""
    out: dict = {"seconds": None, "width": None, "height": None, "fps": None, "vcodec": None,
                 "audio": bool(re.search(r"Stream #.*Audio:", err))}
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if m:
        out["seconds"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    line = re.search(r"Stream #.*Video:.*", err)
    if line:
        v = line.group(0)
        v = v[v.index("Video:"):]   # not the stream id ("#0:0[0x1]")
        m = re.search(r"Video:\s*([A-Za-z0-9_]+)", v)
        out["vcodec"] = m.group(1) if m else None
        m = re.search(r"\b(\d{2,5})x(\d{2,5})\b", v)
        if m:
            out["width"], out["height"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"(\d+(?:\.\d+)?)\s*fps\b", v) or re.search(r"(\d+(?:\.\d+)?)\s*tbr\b", v)
        if m:
            out["fps"] = float(m.group(1))
    return out


def duration(path: Path, *, timeout: float = 60) -> float | None:
    """A media file's length in seconds, or None when ffmpeg can't read it."""
    try:
        return probe(path, timeout=timeout)["seconds"]
    except (OSError, subprocess.SubprocessError, RuntimeError):
        return None


_LIBASS: dict = {}


def has_libass(*, timeout: float = 60) -> bool:
    """Whether this ffmpeg has the `ass` filter (libass), asked once per process."""
    if "ass" not in _LIBASS:
        import re
        r = subprocess.run([ffmpeg(), "-hide_banner", "-filters"], capture_output=True, text=True, timeout=timeout)
        _LIBASS["ass"] = bool(re.search(r"^\s*\S+\s+ass\s+V->V", r.stdout or "", re.M))
    return _LIBASS["ass"]


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
        r = subprocess.run([ffmpeg(), "-y", "-v", "error", "-i", str(video), "-vf", rf"select=eq(n\,{idx})",
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
    exe = ffmpeg()
    if use_ass is None:
        use_ass = has_libass()

    def arg(p: Path) -> str:
        s = Path(p).resolve().as_posix()
        if "'" in s:
            raise ValueError(f"caption path has a quote: {s}")
        return "'" + s.replace(":", r"\:") + "'"

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
