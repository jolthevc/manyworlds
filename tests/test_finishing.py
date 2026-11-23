"""Finishing: a gentle grade that can't oversaturate, subtle grain, the hook text on top, loudness at -14 LUFS and
-1 dBTP in two passes, and a cover frame that is sharp with the eyes open and the mouth closed (manyworlds.finishing)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from manyworlds import finishing
from manyworlds import policy as policy_mod


def _with(policy, **finishing_cfg):
    return policy_mod.from_dict({**policy.body, "finishing": {**policy.body["finishing"], **finishing_cfg}})


def test_the_default_chain_grades_then_grains_with_no_sharpen_or_vignette(policy):
    vf = finishing.filters("neutral", policy=policy)
    parts = vf.split(",")
    assert parts[0].startswith("eq=contrast=1.02:") and parts[-1] == "format=yuv420p"
    strength = policy.get("finishing", "grain", "strength")
    assert f"noise=alls={strength}:allf=t+u" in parts                      # temporal, uniform grain
    assert not any(p.startswith(("unsharp", "vignette", "ass=")) for p in parts)   # off by default
    assert parts.index(f"noise=alls={strength}:allf=t+u") > 0               # grain after the grade


def test_a_look_gets_its_colour_balance_and_curves(policy):
    vf = finishing.filters("warm_night", policy=policy)
    assert "colorbalance=" in vf and "rm=0.04" in vf
    assert "curves=master='0/0.02 0.25/0.23 0.75/0.76 1/0.98'" in vf
    assert vf.index("eq=") < vf.index("colorbalance=") < vf.index("curves=") < vf.index("noise=")


def test_no_look_can_oversaturate_or_push_balance_far(policy):
    sat_max = policy.get("finishing", "limits", "saturation")[1]
    bal = policy.get("finishing", "limits", "balance")
    vf = finishing.filters({"saturation": 2.5, "contrast": 3, "balance": {"rh": 0.9, "zz": 1}}, policy=policy)
    eq = next(p for p in vf.split(",") if p.startswith("eq="))
    assert f"saturation={finishing._num(sat_max)}" in eq
    assert f"contrast={finishing._num(policy.get('finishing', 'limits', 'contrast')[1])}" in eq
    assert f"colorbalance=rh={finishing._num(bal)}" in vf and "zz" not in vf


def test_an_unknown_look_falls_back_to_the_default_and_a_flat_look_adds_nothing(policy):
    assert finishing.filters("disco", policy=policy) == finishing.filters(policy.get("finishing", "default_look"), policy=policy)
    assert finishing.filters({}, policy=policy, grain=False) == "format=yuv420p"


def test_grain_sharpen_and_vignette_follow_their_arguments(policy):
    assert "noise=" not in finishing.filters("neutral", policy=policy, grain=False)
    assert "noise=alls=6:allf=t+u" in finishing.filters("neutral", policy=policy, grain=6)
    cap = policy.get("finishing", "grain", "max_strength")
    assert f"noise=alls={cap}:" in finishing.filters("neutral", policy=policy, grain=cap + 20)   # never heavy
    vf = finishing.filters("neutral", policy=policy, sharpen=True, vignette=True)
    assert "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.3" in vf and "vignette=angle=0.35" in vf


def test_the_text_script_is_burned_in_last_so_it_gets_no_grain(policy, tmp_path):
    ass = tmp_path / "text.ass"
    ass.write_text("[Script Info]\n", encoding="utf-8")
    parts = finishing.filters("neutral", policy=policy, ass=ass).split(",")
    hook = next(i for i, p in enumerate(parts) if p.startswith("ass=filename='"))
    noise = next(i for i, p in enumerate(parts) if p.startswith("noise="))
    assert noise < hook == len(parts) - 2 and ass.resolve().as_posix() in parts[hook]
    # hook_ass is the deprecated name of the same one script: same chain; two different scripts are refused
    assert finishing.filters("neutral", policy=policy, hook_ass=ass) == ",".join(parts)
    assert finishing.filters("neutral", policy=policy, ass=ass, hook_ass=ass) == ",".join(parts)
    other = tmp_path / "other.ass"
    try:
        finishing.filters("neutral", policy=policy, ass=ass, hook_ass=other)
    except ValueError:
        pass
    else:
        raise AssertionError("two scripts must be refused: captions and on-screen texts are one script")


def test_the_hook_defers_to_the_on_screen_text_hook(raw_policy):
    assert raw_policy.get("finishing", "hook") == {"defer_to": "on_screen_text.kinds.hook"}
    hook = raw_policy.get("on_screen_text", "kinds", "hook")
    assert hook["style"] and hook["duration_s"] <= hook["max_duration_s"]       # timed and styled there
    import inspect
    params = inspect.signature(finishing.finish).parameters
    assert {"ass", "hook_ass", "has_voice"} <= set(params) and params["has_voice"].default is True


def test_a_looks_own_grain_and_sharpen_replace_the_global_blocks(policy):
    looks = policy.get("finishing", "looks")
    native = finishing.filters("phone_native", policy=policy)
    assert looks["phone_native"]["grain"] == 0 and "noise=" not in native                 # this look has none
    assert f"luma_amount={finishing._num(looks['phone_native']['sharpen'])}" in native
    night = finishing.filters("phone_night_indoor", policy=policy)
    assert f"noise=alls={looks['phone_night_indoor']['grain']}:" in night
    assert f"luma_amount={finishing._num(looks['phone_night_indoor']['sharpen'])}" in night
    assert night.index("curves=") < night.index("unsharp=") < night.index("noise=")       # after the grade
    neutral = finishing.filters("neutral", policy=policy)                                 # no own keys: global
    assert f"noise=alls={policy.get('finishing', 'grain', 'strength')}:" in neutral and "unsharp" not in neutral
    # an explicit argument still wins over the look; no look can sharpen past limits.sharpen
    assert "noise=" not in finishing.filters("phone_night_indoor", policy=policy, grain=False)
    assert "unsharp" not in finishing.filters("phone_native", policy=policy, sharpen=False)
    top = policy.get("finishing", "limits", "sharpen")[1]
    assert f"luma_amount={finishing._num(top)}" in finishing.filters({"sharpen": 5}, policy=policy)
    cap = policy.get("finishing", "grain", "max_strength")
    assert f"noise=alls={cap}:" in finishing.filters({"grain": cap * 3}, policy=policy)


def test_a_smaller_video_is_upscaled_to_the_output_size_first(policy):
    size = policy.get("finishing", "output")
    assert (size["width"], size["height"]) == (1080, 1920) and size["flags"] == "lanczos"    # native 9:16 delivery
    # 480x832 (talk and story clips): covers 1080x1920 keeping its aspect, then the centre is cropped
    assert finishing.scale_filter(480, 832, policy) == "scale=1108:1920:flags=lanczos,crop=1080:1920"
    assert finishing.scale_filter(720, 1280, policy) == "scale=1080:1920:flags=lanczos"        # same aspect: no crop
    assert finishing.scale_filter(704, 1280, policy) == "scale=1080:1964:flags=lanczos,crop=1080:1920"
    assert finishing.scale_filter(1080, 1920, policy) is None                                  # already that size
    assert finishing.scale_filter(1440, 2560, policy) is None                                  # never scaled down
    assert finishing.scale_filter(None, None, policy) is None
    assert finishing.finished_size(480, 832, policy) == (1080, 1920)
    assert finishing.finished_size(1080, 1920, policy) == (1080, 1920)
    vf = finishing.filters("warm_night", policy=policy, size=(480, 832))
    assert vf.startswith("scale=1108:1920:flags=lanczos,crop=1080:1920,eq=")                 # before the grade
    assert finishing.filters("neutral", policy=policy, size=(1080, 1920)) == finishing.filters("neutral", policy=policy)
    off = _with(policy, output=None)
    assert finishing.scale_filter(480, 832, off) is None and finishing.finished_size(480, 832, off) == (480, 832)
    lanczos_hq = _with(policy, output={**size, "flags": "lanczos+accurate_rnd"})                # the chain is policy
    assert finishing.scale_filter(720, 1280, lanczos_hq) == "scale=1080:1920:flags=lanczos+accurate_rnd"


MCI = "minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff:scd_threshold=10"


def test_published_video_has_one_frame_rate_reached_by_motion_interpolation(policy):
    assert finishing.target_fps(policy) == 30
    # story clips (16 fps) and talk (25) are interpolated first, at their own small size, then upscaled and graded
    parts = finishing.filters("neutral", policy=policy, size=(480, 832), fps=16.0, seconds=5).split(",")
    assert parts[0] == MCI.split(",")[0] and parts[1].startswith("scale=1108:1920")
    assert finishing.filters("neutral", policy=policy, fps=25.0, seconds=8).startswith(MCI + ",eq=")
    # already 30 (narrated edits, hybrid joins): nothing to do; faster or unknown: plain fps= (frames dropped)
    assert finishing.filters("neutral", policy=policy, fps=30.0) == finishing.filters("neutral", policy=policy)
    assert finishing.filters("neutral", policy=policy, fps=60.0).startswith("fps=30,")
    assert finishing.filters("neutral", policy=policy, fps=0.0).startswith("fps=30,")
    # a very short clip blends (too few frames for motion estimation); the method and the switch are policy
    assert finishing.rate_filter(16, policy, seconds=0.5) == "framerate=fps=30"
    interp = policy.get("finishing", "interpolate")
    assert finishing.rate_filter(16, _with(policy, interpolate={**interp, "method": "blend"})) == "framerate=fps=30"
    assert finishing.rate_filter(16, _with(policy, interpolate={**interp, "enabled": False})) == "fps=30"
    assert finishing.rate_filter(16, _with(policy, fps=None)) is None                         # keep the native rate
    try:
        finishing.rate_filter(16, _with(policy, interpolate={**interp, "method": "dain"}))
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown interpolation method must be refused")


def test_the_ai_upscaler_hook_is_off_until_benchmarked(policy):
    assert finishing.upscaler(policy) is None
    on = _with(policy, upscaler={"enabled": True, "kind": "upscale.seedvr2", "model": "x", "scale": 2})
    assert finishing.upscaler(on)["kind"] == "upscale.seedvr2"
    assert finishing.upscaler(_with(policy, upscaler={"enabled": True, "kind": None})) is None


def test_filter_paths_escape_a_drive_colon_and_refuse_quotes():
    assert finishing._filter_path("C:/work/hook.ass").startswith("'") and "\\:" in finishing._filter_path("C:/work/hook.ass")
    try:
        finishing._filter_path("/tmp/it's.ass")
    except ValueError:
        pass
    else:
        raise AssertionError("a quote in a filter path must be refused")


def test_looks_resolve_by_set_then_creator_then_default(policy):
    p = _with(policy, look_by_set={"bodega": "warm_night"}, look_by_creator={"maren": "daylight"})
    assert finishing.look_for(p, creator="maren", set_id="bodega") == "warm_night"
    assert finishing.look_for(p, creator="maren", set_id="kitchen") == "daylight"
    assert finishing.look_for(p, creator="someone") == policy.get("finishing", "default_look")


STDERR = """[Parsed_loudnorm_0 @ 0x55]
{
	"input_i" : "-23.54",
	"input_tp" : "-7.12",
	"input_lra" : "5.40",
	"input_thresh" : "-33.80",
	"output_i" : "-14.02",
	"output_tp" : "-1.00",
	"output_lra" : "4.90",
	"output_thresh" : "-24.20",
	"normalization_type" : "dynamic",
	"target_offset" : "0.02"
}
"""


def test_loudness_is_two_pass_at_minus_14_lufs_and_minus_1_true_peak(policy):
    one = finishing.loudnorm_filter(policy)
    assert one == "loudnorm=I=-14:TP=-1:LRA=11"
    measured = finishing.parse_loudnorm(STDERR)
    assert measured == {"input_i": "-23.54", "input_tp": "-7.12", "input_lra": "5.40", "input_thresh": "-33.80",
                        "target_offset": "0.02"}
    af = finishing.audio_filter(policy, measured)
    assert af.startswith(one + ":measured_I=-23.54:measured_TP=-7.12:measured_LRA=5.40:measured_thresh=-33.80")
    assert ":offset=0.02:linear=true," in af and af.endswith("aresample=48000")


def test_a_silent_or_missing_measurement_falls_back_to_one_pass(policy):
    silent = STDERR.replace('"-23.54"', '"-inf"')
    assert finishing.parse_loudnorm(silent) is None
    assert finishing.parse_loudnorm("no json here") is None
    assert finishing.audio_filter(policy, None) == "loudnorm=I=-14:TP=-1:LRA=11,aresample=48000"


def test_a_bed_only_track_is_never_pulled_up_to_the_voice_target(policy):
    bed = policy.get("finishing", "loudness", "bed_only")
    assert bed["mode"] == "cap" and bed["lufs"] < policy.get("finishing", "loudness", "lufs") - 10
    quiet = {**finishing.parse_loudnorm(STDERR), "input_i": "-46.0"}          # a -44 dB room tone, measured
    up = finishing.bed_gain_db(policy, quiet)
    assert up == bed["max_gain_db"] < bed["lufs"] + 46                          # capped, not the full lift
    assert finishing.bed_audio_filter(policy, quiet) == f"volume={finishing._num(up)}dB,aresample=48000"
    assert "loudnorm" not in finishing.bed_audio_filter(policy, quiet)
    loud = {**quiet, "input_i": "-20.0"}                                        # a loud music bed is turned down
    assert finishing.bed_gain_db(policy, loud) == bed["lufs"] + 20
    assert finishing.bed_audio_filter(policy, None) == "aresample=48000"        # silence: left alone
    skip = _with(policy, loudness={**policy.get("finishing", "loudness"), "bed_only": {**bed, "mode": "skip"}})
    assert finishing.bed_audio_filter(skip, quiet) == "aresample=48000"
    odd = _with(policy, loudness={**policy.get("finishing", "loudness"), "bed_only": {**bed, "mode": "boost"}})
    try:
        finishing.bed_audio_filter(odd, quiet)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown bed_only mode must be refused")


def _clip(path, *, amplitude: float, seconds: float = 2.0):
    import subprocess

    from manyworlds import media
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc=size=64x112:rate=30:duration={seconds:g}", "-f", "lavfi", "-i",
                    f"anoisesrc=colour=pink:amplitude={amplitude}:duration={seconds:g}:sample_rate=48000",
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
                   check=True, capture_output=True)
    return path


def test_finish_keeps_a_bed_only_video_quiet_and_loudnorms_a_voiced_one(policy, tmp_path):
    p = _with(policy, output=None, fps=None)                                     # small and fast: level only
    src = _clip(tmp_path / "room.mp4", amplitude=0.005)
    voiced = finishing.finish(src, tmp_path / "voiced.mp4", look="neutral", policy=p)
    bed = finishing.finish(src, tmp_path / "bed.mp4", look="neutral", has_voice=False, policy=p)
    level = {name: float(finishing.measure(f, p)["input_i"]) for name, f in (("voiced", voiced), ("bed", bed))}
    assert abs(level["voiced"] - policy.get("finishing", "loudness", "lufs")) < 2.5   # the old behaviour, unchanged
    assert level["bed"] <= policy.get("finishing", "loudness", "bed_only", "lufs") + 1.5
    assert level["bed"] < level["voiced"] - 10


# --- cover frame -------------------------------------------------------------------------------------------------

def _frame(path, blur=0.0, seed=0):
    rng = np.random.default_rng(seed)
    img = Image.fromarray((rng.random((96, 64)) > 0.5).astype(np.uint8) * 255).convert("RGB").resize((128, 192))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    img.save(path)
    return path


def test_without_a_face_reader_the_sharpest_frame_is_the_cover(tmp_path):
    sharp = _frame(tmp_path / "a.png")
    soft, softer = _frame(tmp_path / "b.png", blur=2), _frame(tmp_path / "c.png", blur=5)
    assert finishing.pick_cover([softer, sharp, soft], face=lambda p: None) == sharp


def test_eyes_open_and_mouth_closed_beat_a_sharper_frame(policy, tmp_path):
    blink, good, talking = (_frame(tmp_path / "blink.png"), _frame(tmp_path / "good.png", blur=1.5),
                            _frame(tmp_path / "talk.png", blur=0.5))
    eyes_min = policy.get("finishing", "cover", "eyes_open_min")
    mouth_max = policy.get("finishing", "cover", "mouth_closed_max")
    readings = {"blink.png": {"eyes_open": eyes_min / 3, "mouth_open": 0.01},
                "good.png": {"eyes_open": eyes_min * 1.5, "mouth_open": mouth_max / 2},
                "talk.png": {"eyes_open": eyes_min * 1.5, "mouth_open": mouth_max * 3}}
    assert finishing.pick_cover([blink, good, talking], policy=policy, face=lambda p: readings[p.name]) == good


def test_a_face_beats_no_face_and_an_unreadable_frame_is_never_the_cover(policy, tmp_path):
    empty, face = _frame(tmp_path / "empty.png"), _frame(tmp_path / "face.png", blur=3)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    reader = {"face.png": {"eyes_open": 0.01, "mouth_open": 0.5, "box": (0.2, 0.1, 0.8, 0.6)}}
    assert finishing.pick_cover([broken, empty, face], policy=policy, face=lambda p: reader.get(p.name)) == face
    assert finishing.pick_cover([broken, empty], policy=policy, face=lambda p: None) == empty


def test_laplacian_sharpness_ranks_blur():
    rng = np.random.default_rng(3)
    crisp = rng.random((40, 40))
    flat = np.full((40, 40), 0.5)
    assert finishing.sharpness(crisp) > finishing.sharpness(flat) == 0.0
    assert finishing.sharpness(np.zeros((2, 2))) == 0.0


# --- cover from the finished video ---------------------------------------------------------------------------------

def test_cover_times_sample_the_video_inside_its_edges(policy):
    assert finishing.cover_times(2.0, policy) == [0.2, 0.7, 1.2, 1.7]
    long = finishing.cover_times(60.0, policy)
    most = policy.get("finishing", "cover", "max_samples")
    assert len(long) == most and long[0] == 0.2 and long[-1] == 59.8              # spread evenly, capped
    assert finishing.cover_times(0.1, policy) == [0.05] and finishing.cover_times(None, policy) == [0.0]


def _half_blurred(path):
    """2 s at 10 fps: blurred for the first second, sharp after it."""
    import subprocess

    from manyworlds import media
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc=size=128x224:rate=10:duration=2,boxblur=8:enable='lt(t,1)'",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True, capture_output=True)
    return path


def test_the_cover_comes_from_the_finished_video_and_avoids_text_on_screen(policy, tmp_path):
    video = _half_blurred(tmp_path / "final.mp4")
    none = lambda p: None   # noqa: E731 - no face reader here: sharpness decides
    best = finishing.cover_from_video(video, workdir=tmp_path / "a", policy=policy, face=none)
    assert best["t"] >= 1.0 and best["frame"].exists()                           # the sharp second
    # a caption over the sharp second: a clear (if softer) frame wins
    clear = finishing.cover_from_video(video, workdir=tmp_path / "b", policy=policy, face=none, avoid=[(1.0, 2.0)])
    assert clear["t"] < 1.0 - policy.get("finishing", "cover", "avoid_margin_s")
    # text on screen the whole time: still a cover, the sharpest
    always = finishing.cover_from_video(video, workdir=tmp_path / "c", policy=policy, face=none, avoid=[(0.0, 2.0)])
    assert always["t"] >= 1.0
    # pick_cover takes the video too (and still a list of frames, as before)
    frame = finishing.pick_cover(video, policy=policy, face=none, workdir=tmp_path / "d", avoid=[(1.0, 2.0)])
    assert frame.parent == tmp_path / "d" and frame.suffix == ".jpg"
    assert finishing.pick_cover([frame], policy=policy, face=none) == frame


def test_a_good_face_wins_among_the_frames_clear_of_text(policy, tmp_path):
    video = _half_blurred(tmp_path / "final.mp4")
    eyes_min = policy.get("finishing", "cover", "eyes_open_min")
    mouth_max = policy.get("finishing", "cover", "mouth_closed_max")
    good = {"eyes_open": eyes_min * 2, "mouth_open": mouth_max / 2}
    talking = {"eyes_open": eyes_min * 2, "mouth_open": mouth_max * 4}
    # the softer first frame has the good face; every other frame is mid-word
    reader = lambda p: good if p.name == "cover_000.jpg" else talking   # noqa: E731
    best = finishing.cover_from_video(video, workdir=tmp_path / "w", policy=policy, face=reader)
    assert best["frame"].name == "cover_000.jpg"
