"""Dance and motion transfer (off): the Wan-Animate-2 core graph, the reference-clip rule (ours or licensed, never a
downloaded platform video) and the identity check."""

from __future__ import annotations

import pytest

from manyworlds import dance, policy as policy_mod
from manyworlds.worker import comfy_animate


def _refs(policy, refs):
    return policy_mod.from_dict({**policy.body, "dance": {**policy.body["dance"], "references": refs}})


def test_the_shipped_policy_keeps_dance_off_with_no_references(raw_policy):
    c = raw_policy.get("dance")
    assert c["enabled"] is False and c["references"] == []
    assert not dance.enabled(raw_policy)
    with pytest.raises(dance.RefusedReference, match="not in dance.references"):
        dance.reference("any", raw_policy)


def test_only_our_own_or_licensed_clips_drive_a_dance(policy):
    p = _refs(policy, [
        {"id": "ours-1", "uri": "s3://dance/ours-1.mp4", "owner": "ours", "licence": "own-recording",
         "source": "shot by us 2026-11"},
        {"id": "stock-1", "uri": "s3://dance/stock-1.mp4", "owner": "licensed", "licence": "stock licence #123",
         "source": "https://stock.example.com/clip/1"},
        {"id": "trend", "uri": "s3://dance/trend.mp4", "owner": "licensed", "licence": "?",
         "source": "https://www.tiktok.com/@someone/video/1"},
        {"id": "yt", "uri": "s3://dance/yt.mp4", "owner": "ours", "licence": "x", "source": "m.youtube.com/watch"},
        {"id": "nolicence", "uri": "s3://dance/n.mp4", "owner": "ours", "source": "us"},
        {"id": "stranger", "uri": "s3://dance/s.mp4", "owner": "someone", "licence": "x", "source": "us"},
        {"id": "nouri", "owner": "ours", "licence": "x", "source": "us"},
    ])
    assert dance.reference("ours-1", p)["uri"] == "s3://dance/ours-1.mp4"
    assert dance.reference("stock-1", p)["owner"] == "licensed"
    for ref_id, why in (("trend", "tiktok.com"), ("yt", "m.youtube.com"), ("nolicence", "no licence"),
                        ("stranger", "owner"), ("nouri", "no uri")):
        with pytest.raises(dance.RefusedReference, match=why):
            dance.reference(ref_id, p)


def test_identity_holds_only_when_every_keyframe_and_the_mean_pass(policy):
    ic = policy.get("dance", "identity")
    ok = dance.identity_check([ic["min_mean"], ic["min_mean"] + 1, ic["min_mean"]], policy)
    assert ok["passed"] and ok["failed_frames"] == []
    drift = dance.identity_check([9, 9, ic["min_score"] - 1, 9], policy)
    assert not drift["passed"] and drift["failed_frames"] == [2]
    low_mean = dance.identity_check([ic["min_score"]] * 4, policy)
    assert low_mean["passed"] is (ic["min_score"] >= ic["min_mean"])
    assert dance.identity_check([], policy)["passed"] is False


def _graph(policy, **over):
    params = {**policy.get("dance", "generation"), **over}
    return comfy_animate.graph(image="ref.png", video="drive.mp4", prompt="she dances in the kitchen", frames=81,
                               seed=11, params=params)


def test_the_animate2_graph_follows_the_core_template(policy):
    g = _graph(policy)
    gen = policy.get("dance", "generation")
    a = g["animate"]
    assert a["class_type"] == "WanAnimate2ToVideo"
    assert a["inputs"]["reference_image"] == ["ref", 0] and a["inputs"]["pose_video"] == ["drive_scaled", 0]
    assert a["inputs"]["clip_vision_output"] == ["ref_vision", 0]
    assert a["inputs"]["clip_vision_output_pose"] == ["pose_vision", 0]
    assert a["inputs"]["positive_pose"] == ["pose_text", 0] and g["pose_text"]["inputs"]["text"] == ""
    assert (a["inputs"]["width"], a["inputs"]["height"], a["inputs"]["length"]) == (gen["width"], gen["height"], 81)
    assert g["cache"]["class_type"] == "WanAnimate2Cache" and g["cache"]["inputs"]["model"] == ["lora", 0]
    assert g["shift"]["inputs"] == {"model": ["cache", 0], "shift": 5.0}
    assert g["sigmas"]["inputs"]["steps"] == gen["steps"] and g["sigmas"]["inputs"]["model"] == ["cache", 0]
    s = g["sample"]["inputs"]
    assert s["model"] == ["shift", 0] and s["positive"] == ["animate", 0] and s["negative"] == ["animate", 1]
    assert s["latent_image"] == ["animate", 2] and s["cfg"] == 1.0 and s["noise_seed"] == 11
    assert g["sampler_pick"]["inputs"]["sampler_name"] == "lcm"
    assert g["trim"]["inputs"] == {"samples": ["sample", 0], "trim_amount": ["animate", 3]}
    assert g["clip"]["inputs"]["type"] == "wan"
    assert g["unet"]["inputs"]["unet_name"] == comfy_animate.MODEL
    # the driving clip's frame rate, never its audio
    assert g["video"]["inputs"] == {"images": ["decode", 0], "fps": ["drive_frames", 2]}
    for node_id, node in g.items():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int):
                assert v[0] in g, f"{node_id} links to a missing node {v[0]}"


def test_the_cache_can_be_left_out_and_bad_sizes_are_refused(policy):
    g = _graph(policy, cache={"enabled": False})
    assert "cache" not in g and g["shift"]["inputs"]["model"] == ["lora", 0]
    with pytest.raises(ValueError):
        _graph(policy, width=470)
    for frames in (0, 80, 85):
        with pytest.raises(ValueError):
            comfy_animate.check_length(frames)
    assert comfy_animate.check_length(77) == 77


def test_every_weight_is_from_the_comfy_org_repackage():
    assert {repo for files in comfy_animate.WEIGHTS.values() for repo, _ in files} == {"Comfy-Org/Wan-Animate-2"}
