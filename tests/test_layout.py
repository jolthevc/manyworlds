"""Structural control for photos (manyworlds.layout): a pose skeleton drawn from the shot sheet, sized by shot size,
bent by camera height, placed by the blocking; no skeleton where there is no body to place."""

from __future__ import annotations

import pytest

from manyworlds import layout
from manyworlds.worker.handlers import image_zimage
from manyworlds.worker.runner import NonRetryable


def shot(size, angle="eye level", action="standing", device="held"):
    return {"framing": {"shot_size": size, "angle": angle, "device": device}, "blocking": [{"action": action}]}


def _height(pts):
    ys = [p[1] for p in pts if p]
    return max(ys) - min(ys)


def test_shot_size_sets_how_big_she_is_and_close_ups_get_no_skeleton(policy):
    full = layout.keypoints(layout.plan(shot("full body"), policy), policy, 864, 1536)
    tiny = layout.keypoints(layout.plan(shot("wide", action="a tiny figure walking the ridge"), policy), policy, 864, 1536)
    assert _height(full) > 3 * _height(tiny)
    assert layout.plan(shot("wide", action="a tiny figure walking"), policy)["scale"] == 0.12
    assert layout.plan(shot("close-up"), policy) is None
    assert layout.plan(shot("medium shot", device="pov"), policy) is None
    assert layout.plan(shot("medium close-up"), policy)["scale"] > layout.plan(shot("medium shot"), policy)["scale"]


def test_poses_follow_the_blocking_and_her_back_has_no_face(policy):
    p = lambda a: layout.plan(shot("full body", action=a), policy)["pose"]
    assert p("walking away from the camera down the street") == "back_walk"
    assert p("sitting on the floor against the sofa") == "sit_floor"
    assert p("lying on her stomach reading") == "lie_stomach"
    assert p("crouching to smell the tulips") == "crouch"
    assert p("walking away and looking back over her shoulder") == "over_shoulder"
    assert p("looks back over her shoulder") == "over_shoulder"
    assert p("just being there") == "stand"
    back = layout.keypoints(layout.plan(shot("full body", action="walking away"), policy), policy, 864, 1536)
    assert back[0] is None and back[14] is None and back[15] is None


def test_a_low_camera_bends_the_body_and_placement_moves_her(policy):
    eye = layout.keypoints(layout.plan(shot("full body"), policy), policy, 864, 1536)
    low = layout.keypoints(layout.plan(shot("full body", angle="low angle"), policy), policy, 864, 1536)
    leg = lambda k: k[10][1] - k[9][1]          # knee to ankle, pixels
    torso = lambda k: k[8][1] - k[1][1]         # neck to hip
    assert leg(low) / torso(low) > leg(eye) / torso(eye)
    right = layout.keypoints(layout.plan(shot("medium shot", action="sits gazing out, small in the right third"),
                                         policy), policy, 864, 1536)
    assert right[1][0] > 864 * 0.6


def test_the_map_is_drawn_and_the_payload_round_trips(policy):
    img = layout.pose_map(shot("full body", action="weight on one hip"), policy, 864, 1536)
    assert img.size == (864, 1536) and img.getbbox() is not None
    assert layout.pose_map(shot("close-up"), policy, 864, 1536) is None
    block = layout.control_payload(policy, "s3://b/pose.png", {"pose": "stand"})
    assert image_zimage.control_spec({"control": block})["file"].endswith(".safetensors")
    assert image_zimage.control_spec({}) is None
    with pytest.raises(NonRetryable):
        image_zimage.control_spec({"control": {"kind": "depth", "uri": "x"}})
