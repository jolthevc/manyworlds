"""Joining and normalising video on the control plane (manyworlds.media): one frame rate reached by motion
interpolation instead of repeated frames, mixed segments normalised before a copy-join, and motion transitions
(whip, blur, flash, match) in the joiner. ffmpeg from the imageio-ffmpeg wheel (base deps)."""

from __future__ import annotations

import subprocess

import pytest

from manyworlds import media

MCI = "minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff:scd_threshold=10"


def _clip(path, *, size="160x288", rate=30, seconds=1.0, audio=True):
    cmd = [media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate={rate}:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-shortest", "-c:a", "aac"]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def _frames(path):
    import imageio_ffmpeg
    return imageio_ffmpeg.count_frames_and_secs(str(path))


# --- frame rate ------------------------------------------------------------------------------------------------
def test_a_slower_source_is_interpolated_never_a_faster_one(raw_policy):
    interp = raw_policy.get("finishing", "interpolate")
    assert media.fps_filter(16, 30, interp, seconds=5) == MCI
    assert media.fps_filter(25, 30, interp, seconds=5) == MCI
    assert media.fps_filter(30, 30, interp) is None                     # already there
    assert media.fps_filter(60, 30, interp) == "fps=30"                 # faster: frames dropped
    assert media.fps_filter(None, 30, interp) == "fps=30"               # unknown rate: plain
    assert media.fps_filter(16, None, interp) is None                   # no target: keep it
    assert media.fps_filter(16, 30, interp, seconds=0.4) == "framerate=fps=30"   # too short to estimate motion
    assert media.fps_filter(16, 30, {**interp, "enabled": False}) == "fps=30"
    assert media.fps_filter(16, 30, {**interp, "method": "duplicate"}) == "fps=30"
    assert media.fps_filter(16, 25, {**interp, "method": "blend"}) == "framerate=fps=25"
    with pytest.raises(ValueError, match="interpolation method"):
        media.fps_filter(16, 30, {**interp, "method": "magic"})


def test_normalize_interpolates_before_it_scales(raw_policy):
    interp = raw_policy.get("finishing", "interpolate")
    vf = media.normalize_filter(1080, 1920, 30, src_fps=25, interpolate=interp, seconds=8, flags="lanczos")
    assert vf.startswith(MCI + ",scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920")
    assert vf.endswith("format=yuv420p,setsar=1")
    assert media.normalize_filter(1080, 1920, 30, src_fps=30).startswith("fps=30,scale=")   # constant rate kept


def test_probe_reads_the_frame_rate_and_codec(tmp_path):
    info = media.probe(_clip(tmp_path / "a.mp4", rate=16, audio=False))
    assert (info["width"], info["height"], info["fps"], info["vcodec"], info["audio"]) == (160, 288, 16.0, "h264", False)


def test_normalize_really_interpolates_to_the_rate_and_gives_a_silent_clip_sound(tmp_path, raw_policy):
    src = _clip(tmp_path / "m16.mp4", rate=16, seconds=2, audio=False)
    out = media.normalize(src, tmp_path / "n.mp4", width=180, height=320, fps=30, policy=raw_policy)
    info = media.probe(out)
    frames, seconds = _frames(out)
    assert (info["width"], info["height"], info["audio"]) == (180, 320, True)
    assert abs(frames - 60) <= 4 and abs(seconds - 2.0) < 0.15   # minterpolate may add an edge frame or two


# --- joins -----------------------------------------------------------------------------------------------------
def test_segments_that_differ_are_normalised_before_the_copy_join(tmp_path, raw_policy, monkeypatch):
    same = [_clip(tmp_path / f"s{i}.mp4") for i in range(2)]
    calls = []
    real = media.normalize

    def spy(p, out, **k):
        calls.append(k)
        return real(p, out, **k)
    monkeypatch.setattr(media, "normalize", spy)
    media.concat(same, tmp_path / "same.mp4", policy=raw_policy)
    assert calls == []                                                   # alike: copied as they are
    mixed = [_clip(tmp_path / "small.mp4", size="160x288", rate=25), _clip(tmp_path / "big.mp4", size="320x576")]
    out = media.concat(mixed, tmp_path / "mixed.mp4", policy=raw_policy)
    assert len(calls) == 2 and {(k["width"], k["height"], k["fps"]) for k in calls} == {(320, 576, 30)}
    info = media.probe(out)
    frames, seconds = _frames(out)
    assert (info["width"], info["height"]) == (320, 576) and abs(frames - 60) <= 2 and abs(seconds - 2.0) < 0.15


def test_joins_default_to_cuts_and_refuse_unknown_transitions(raw_policy):
    fin = raw_policy.get("finishing")
    assert media.joins_for(None, 3, fin) == [{"type": "cut"}, {"type": "cut"}]
    assert media.joins_for(["whip", {"type": "match", "out_at": 1.2}], 3, fin)[1]["out_at"] == 1.2
    with pytest.raises(ValueError, match="need 2"):
        media.joins_for(["whip"], 3, fin)
    with pytest.raises(ValueError, match="unknown transition"):
        media.joins_for(["spin"], 2, fin)
    with pytest.raises(ValueError, match="no finishing.transitions.whip"):
        media.joins_for(["whip"], 2, {})


def _graph(raw_policy, joins, segments=None, fps=30):
    fin = raw_policy.get("finishing")
    segments = segments or [{"seconds": 2.0, "fps": 30.0, "audio": True} for _ in range(len(joins) + 1)]
    return media.transition_graph(segments, media.joins_for(joins, len(segments), fin), width=1080, height=1920,
                                  fps=fps, fin=fin)


def test_whip_flash_and_blur_cross_over_a_few_frames_with_the_audio(raw_policy):
    graph, v, a = _graph(raw_policy, ["whip", "flash"])
    blocks = raw_policy.get("finishing", "transitions")
    px, whip_d = blocks["whip"]["blur_px"], blocks["whip"]["frames"] / 30
    assert f"avgblur=sizeX={px}:sizeY=1:enable='gte(t,{media._fmt(2 - whip_d)})'" in graph   # A's tail blurs
    assert f"avgblur=sizeX={px}:sizeY=1:enable='lt(t,{media._fmt(whip_d)})'" in graph        # B's head blurs
    assert f"xfade=transition=slideleft:duration={media._fmt(whip_d)}:offset={media._fmt(2 - whip_d)}" in graph
    flash_d = blocks["flash"]["frames"] / 30
    off = 2 + 2 - whip_d - flash_d                                       # after the first crossover, minus this one
    assert f"xfade=transition=fadewhite:duration={media._fmt(flash_d)}:offset={media._fmt(off)}" in graph
    assert f"acrossfade=d={media._fmt(whip_d)}" in graph and (v, a) == ("j2", "k2")
    graph, _, _ = _graph(raw_policy, ["blur"])
    assert f"gblur=sigma={blocks['blur']['sigma']}" in graph and "xfade=transition=fade:" in graph
    graph, _, _ = _graph(raw_policy, [{"type": "whip", "direction": "up"}])
    assert f"avgblur=sizeX=1:sizeY={px}" in graph and "transition=slideup" in graph
    lo, hi = blocks["frames_range"]
    graph, _, _ = _graph(raw_policy, [{"type": "flash", "frames": 99}])
    assert f"duration={media._fmt(hi / 30)}:" in graph                    # clamped to 4-8 frames
    graph, _, _ = _graph(raw_policy, ["flash"], segments=[{"seconds": 0.1, "fps": 30, "audio": True}] * 2)
    assert "duration=0.05:" in graph                                     # never more than half of a side


def test_cut_is_a_plain_concat_and_match_trims_to_the_nearest_frame(raw_policy):
    graph, _, _ = _graph(raw_policy, ["cut"])
    assert "xfade" not in graph and "[v0][v1]concat=n=2:v=1:a=0[j1]" in graph and "[a0][a1]concat=n=2:v=0:a=1[k1]" in graph
    graph, _, _ = _graph(raw_policy, [{"type": "match", "out_at": 1.49, "in_at": 0.51}])
    first, second = graph.split(";")[0], graph.split(";")[2]
    assert "trim=end=1.5," in first and "trim=start=0.5," in second       # snapped to frames 45 and 15 at 30 fps
    assert "xfade" not in graph and "concat=n=2:v=1:a=0" in graph


def test_every_input_reaches_one_size_and_rate_and_a_silent_one_gets_silence(raw_policy):
    segments = [{"seconds": 2.0, "fps": 25.0, "audio": True}, {"seconds": 2.0, "fps": 30.0, "audio": False}]
    graph, _, a = _graph(raw_policy, ["flash"], segments=segments)
    first, second = graph.split(";")[0], graph.split(";")[2]           # each input's video, then its audio
    assert first.startswith("[0:v]setpts=PTS-STARTPTS," + MCI + ",scale=1080:1920")   # 25 fps interpolated
    assert second.startswith("[1:v]setpts=PTS-STARTPTS,fps=30,scale=1080:1920") and "settb=AVTB[v1]" in second
    assert "anullsrc=r=48000:cl=stereo,atrim=duration=2," in graph and a == "k1"
    graph, _, a = _graph(raw_policy, ["cut"], segments=[{"seconds": 1.0, "fps": 16.0, "audio": False}] * 2)
    assert a is None and "[0:a]" not in graph and "anullsrc" not in graph   # all silent: video only


def test_transitions_really_render(tmp_path, raw_policy):
    parts = [_clip(tmp_path / f"t{i}.mp4", size="160x288", seconds=1) for i in range(4)]
    out = media.concat(parts, tmp_path / "t.mp4", transitions=["whip", "blur", "flash"], policy=raw_policy)
    blocks = raw_policy.get("finishing", "transitions")
    overlap = sum(blocks[k]["frames"] for k in ("whip", "blur", "flash"))
    frames, _ = _frames(out)
    assert abs(frames - (4 * 30 - overlap)) <= 3 and media.probe(out)["audio"]
    mixed = [_clip(tmp_path / "m0.mp4", rate=25, seconds=2), _clip(tmp_path / "m1.mp4", size="320x576", audio=False)]
    out = media.concat(mixed, tmp_path / "m.mp4", transitions=[{"type": "match", "out_at": 1.5}], policy=raw_policy)
    info = media.probe(out)
    frames, _ = _frames(out)
    assert (info["width"], info["height"], info["audio"]) == (320, 576, True) and abs(frames - 75) <= 3
