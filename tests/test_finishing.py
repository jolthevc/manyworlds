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
    assert finishing.filters("disco", policy=policy) == finishing.filters("neutral", policy=policy)
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
