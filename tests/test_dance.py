"""Dance and motion transfer (off): the Wan-Animate-2 core graph, the reference-clip rule (ours or licensed, never a
downloaded platform video) and the identity check."""

from __future__ import annotations

import pytest

from manyworlds import dance, policy as policy_mod
from manyworlds.worker import comfy_animate


def _refs(policy, refs, enabled=False):
    return policy_mod.from_dict({**policy.body, "dance": {**policy.body["dance"], "references": refs, "enabled": enabled}})


def test_every_shipped_reference_is_licensed_and_passes_the_rule(raw_policy):
    c = raw_policy.get("dance")
    assert c["references"], "the shipped catalogue has the Pexels clips"
    for e in c["references"]:
        assert e["owner"] == "licensed" and e["licence"] and e["source"].startswith("https://www.pexels.com/")
        assert dance.reference(e["id"], raw_policy)["uri"].startswith("s3://manyworlds/references/motion/")
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


# --- the worker handler (comfy.wan_animate2, off) and the benchmark path -----------------------------------------
OURS = {"id": "ours-1", "uri": "s3://dance/ours-1.mp4", "owner": "ours", "licence": "own-recording",
        "source": "shot by us 2026-11"}
TREND = {"id": "trend", "uri": "s3://dance/trend.mp4", "owner": "licensed", "licence": "?",
         "source": "https://www.tiktok.com/@someone/video/1"}


def _on(policy, refs=(OURS, TREND)):
    p = _refs(policy, list(refs))
    return policy_mod.from_dict({**p.body, "dance": {**p.body["dance"], "enabled": True}})


def test_the_dance_handler_refuses_while_dance_is_off_or_the_clip_is_not_ours(policy):
    from manyworlds.worker.handlers import dance_animate
    from manyworlds.worker.runner import NonRetryable
    job = {"image_uri": "s3://b/photo.png", "reference": "ours-1"}
    with pytest.raises(NonRetryable, match="dance is off"):
        dance_animate.generate(job, None, policy=_refs(policy, [OURS]))     # refused before anything is fetched
    on = _on(policy)
    ref, params, frames = dance_animate.check(job, on)
    assert ref["uri"] == OURS["uri"] and frames == on.get("dance", "generation", "frames")
    assert params["timeout_s"] == on.get("dance", "generation", "timeout_s")
    for bad, why in (({**job, "reference": "trend"}, "tiktok.com"), ({"image_uri": "x"}, "reference"),
                     ({**job, "params": {"frames": 80}}, "4k\+1"), ({**job, "params": {"width": 470}}, "16")):
        with pytest.raises(NonRetryable, match=why):
            dance_animate.check(bad, on)


def test_dance_runs_on_its_own_family_of_the_comfyui_image(policy):
    from manyworlds.worker.handlers import dance_animate
    from manyworlds.worker.runner import handlers_for
    assert policy.family("comfy.wan_animate2") == "dance"
    img = policy.get("worker_images", "dance")
    assert img["image"] == policy.get("worker_images", "comfy", "image")       # mw-worker-talkcomfy
    assert set(policy.get("job_kinds", "comfy.wan_animate2", "pools")) <= set(policy.get("gpu_pools"))
    assert handlers_for(["comfy.wan_animate2"])["comfy.wan_animate2"] is dance_animate.generate
    assert "comfy.wan_animate2" not in (policy.get("production", "formats") or [])


def test_the_dance_benchmark_needs_dance_on_and_a_cleared_reference(conn, policy):
    from manyworlds import bench
    extra = {"image_uri": "s3://b/photo.png", "reference": "ours-1"}
    with pytest.raises(ValueError, match="dance is off"):
        bench.enqueue(conn, _refs(policy, [OURS]), "comfy.wan_animate2", 1, extra=extra)
    on = _on(policy)
    for bad, why in (({**extra, "reference": "trend"}, "tiktok.com"), ({"image_uri": "s3://b/p.png"}, "reference")):
        with pytest.raises(ValueError, match=why):
            bench.enqueue(conn, on, "comfy.wan_animate2", 1, extra=bad)
    assert conn.execute("SELECT count(*) AS n FROM jobs").fetchone()["n"] == 0
    batch = bench.enqueue(conn, on, "comfy.wan_animate2", 2, extra=extra)
    jobs = conn.execute("SELECT kind, payload FROM jobs WHERE batch = %s", (batch,)).fetchall()
    assert len(jobs) == 2 and {j["kind"] for j in jobs} == {"comfy.wan_animate2"}
    p = jobs[0]["payload"]
    assert p["reference"] == "ours-1" and p["image_uri"] == "s3://b/photo.png" and p["keyframes"] == on.get("dance", "identity", "keyframes")
    assert p["params"]["frames"] == on.get("dance", "generation", "frames")


def test_a_worker_whose_baked_policy_predates_a_clip_takes_the_entry_the_job_carries(policy):
    # 2026-11-24: seven new licensed clips failed on a worker image built before they were listed.
    from manyworlds.worker.handlers import dance_animate
    from manyworlds.worker.runner import NonRetryable
    old = _on(policy, refs=())                       # the worker's baked policy lists no clips
    job = {"image_uri": "s3://b/photo.png", "reference": "ours-1", "reference_entry": OURS}
    ref, _, _ = dance_animate.check(job, old)
    assert ref["uri"] == OURS["uri"]
    for bad in ({**job, "reference_entry": {**TREND, "id": "ours-1"}},          # same rules: a platform source
                {**job, "reference_entry": {**OURS, "owner": "someone"}},       # not ours or licensed
                {**job, "reference_entry": {**OURS, "id": "other"}}):           # an entry for another id
        with pytest.raises(NonRetryable):
            dance_animate.check(bad, old)
