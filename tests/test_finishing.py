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


def test_the_hook_text_is_burned_in_last_so_it_gets_no_grain(policy, tmp_path):
    ass = tmp_path / "hook.ass"
    ass.write_text("[Script Info]\n", encoding="utf-8")
    parts = finishing.filters("neutral", policy=policy, hook_ass=ass).split(",")
    hook = next(i for i, p in enumerate(parts) if p.startswith("ass=filename='"))
    noise = next(i for i, p in enumerate(parts) if p.startswith("noise="))
    assert noise < hook == len(parts) - 2 and ass.resolve().as_posix() in parts[hook]


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
