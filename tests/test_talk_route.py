"""Talk routes (policy generation.talk_route / talk_routes, production.talk): a talking take renders with the first
route that can make it. LTX-2.5 audio-to-video (route ltx25: comfy.graph, graph ltx25_a2v) is selectable per post
(body.talk_route) with InfiniteTalk as the fallback; the default stays InfiniteTalk. The job is built as the breadth
takes build theirs (long_take.talk_params, graph_fields), goes through the same video gates and the per-gate repair
escalation with the route's own knobs, and a route whose repairs run out hands the take to the next one."""

from __future__ import annotations

import uuid

import pytest
from psycopg.types.json import Jsonb

from manyworlds import breadth, long_take, production
from manyworlds.production.steps import DropPost

from test_review import BIBLE
from test_talk_repair import LINES, _gate, _judged, _story

ROUTE = ["ltx25", "infinitetalk"]


def _audio(seconds=3.2, k=0):
    return {"uri": f"s3://b/line{k}.wav", "meta": {"seconds": seconds}}


def _seg(conn, policy, post, body, chosen, segment, audio=None):
    jobs = body["talk_segment_jobs"][segment]
    job = production._talk_job(conn, policy, post, body, chosen, len(jobs), segment=segment,
                               audio=_audio() if audio is None else audio)
    jobs.append(str(job))
    return conn.execute("SELECT kind, pools, payload FROM jobs WHERE id = %s", (job,)).fetchone()


@pytest.fixture()
def hf(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")   # the graph's weights are gated (comfy_graphs requires_env)


def test_the_default_route_is_infinitetalk(conn, policy, hf):
    assert policy.get("generation", "talk_route") == ["infinitetalk"]
    post, body, chosen = _story(conn)
    j = _seg(conn, policy, post, body, chosen, 1)
    assert j["kind"] == policy.get("production", "talk_kind") and j["payload"]["talk_route"] == "infinitetalk"
    assert "graph" not in j["payload"] and j["payload"]["params"]["steps"] == policy.get("generation", "talk", "steps")


def test_a_post_routed_to_ltx_gets_a_comfy_graph_job_built_like_the_breadth_takes(conn, policy, hf):
    post, body, chosen = _story(conn)
    body["talk_route"] = ROUTE
    j = _seg(conn, policy, post, body, chosen, 1)
    route = policy.get("generation", "talk_routes", "ltx25")
    p = j["payload"]
    assert j["kind"] == "comfy.graph" and p["talk_route"] == "ltx25" and j["pools"] == route["pools"]
    assert p["graph"] == "ltx25_a2v" and p["graph_spec"] == policy.get("comfy_graphs", "ltx25_a2v")
    assert p["params"] == long_take.talk_params(policy, {"graph": "ltx25_a2v"})        # the graph's, as the takes
    assert (p["params"]["width"], p["params"]["height"]) == (576, 1024) and "steps" not in p["params"]
    assert p["image_uri"] == "s3://b/shot1.png" and p["audio_uri"] == "s3://b/line0.wav"
    assert p["script"] == LINES[1] and p["prompt"] and p["keyframes"] and isinstance(p["seed"], int)
    assert "repair" not in p


def test_ltx_is_skipped_for_what_it_cannot_make(conn, policy, hf, monkeypatch):
    post, body, chosen = _story(conn)
    body["talk_route"] = ROUTE
    longest = float(policy.get("generation", "talk_routes", "ltx25", "max_seconds"))
    assert _seg(conn, policy, post, body, chosen, 0, audio=_audio(longest + 1))["kind"] == "talk.infinitetalk"
    assert _seg(conn, policy, post, body, chosen, 1, audio={})["kind"] == "talk.infinitetalk"   # no designed voice
    monkeypatch.delenv("HF_TOKEN")
    assert _seg(conn, policy, post, body, chosen, 2)["kind"] == "talk.infinitetalk"   # the gated weights' token
    seconds, why = 3.0, production.route_unavailable
    assert "motion clip" in why(policy, "ltx25", fmt="talking_video", seconds=seconds, voiced=True, moving=True,
                                env={"HF_TOKEN": "x"})
    assert why(policy, "ltx25", fmt="music_video", seconds=seconds, voiced=True, moving=False, env={"HF_TOKEN": "x"})
    assert why(policy, "ltx25", fmt="talking_video", seconds=seconds, voiced=True, moving=False,
               env={"HF_TOKEN": "x"}) is None


def test_ltx_lip_sync_escalates_seed_then_cutaway_without_audio_guidance(conn, policy, hf):
    post, body, chosen = _story(conn)
    body["talk_route"] = ROUTE
    _seg(conn, policy, post, body, chosen, 2)
    for corr in (-0.46, -0.34):                                  # within the tolerance: the seed is dead
        _judged(conn, body["talk_segment_jobs"][2][-1], corr)
        j = _seg(conn, policy, post, body, chosen, 2)
    assert j["kind"] == "image.narrated"                         # no guidance knob on this graph: the cutaway
    assert body["talk_modes"][2] == "narrated"


def test_a_take_whose_ltx_repairs_run_out_goes_to_infinitetalk_and_stays(conn, policy, hf):
    post, body, chosen = _story(conn)
    body["talk_route"] = ROUTE
    last = len(LINES) - 1                                        # the sign-off stays on camera: no cutaway
    first = _seg(conn, policy, post, body, chosen, last)
    assert first["kind"] == "comfy.graph"
    for corr in (-0.46, -0.34):
        _judged(conn, body["talk_segment_jobs"][last][-1], corr)
        j = _seg(conn, policy, post, body, chosen, last)
    assert j["kind"] == "talk.infinitetalk" and "repair" not in j["payload"]   # a fresh first take on the next route
    assert j["payload"]["route_fallback"]["from"] == "ltx25"
    assert "lip sync" in j["payload"]["route_fallback"]["reason"]
    _judged(conn, body["talk_segment_jobs"][last][-1], 0.1)
    j = _seg(conn, policy, post, body, chosen, last)
    assert j["kind"] == "talk.infinitetalk" and j["payload"]["repair"]["knob"] == "seed"   # never back to ltx
    _judged(conn, body["talk_segment_jobs"][last][-1], 0.12)
    j = _seg(conn, policy, post, body, chosen, last)
    assert j["payload"]["repair"]["knob"] == "audio_guidance"     # InfiniteTalk's own escalation
    _judged(conn, body["talk_segment_jobs"][last][-1], 0.11)
    with pytest.raises(DropPost):                                 # the last route's repairs ran out: dropped
        _seg(conn, policy, post, body, chosen, last)


def test_ltx_identity_starts_from_another_accepted_photo(conn, policy, hf):
    post, body, chosen = _story(conn)
    body["talk_route"] = ROUTE
    scene_job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('image.zimage', 'done', %s, "
                             "'{}') RETURNING id", (post["creator_id"],)).fetchone()["id"]
    for name in ("shot1.png", "alt.png"):
        a = conn.execute("INSERT INTO assets (job_id, kind, uri, sha256, bytes) VALUES (%s, 'image', %s, 'x', 1) "
                         "RETURNING id", (scene_job, f"s3://b/{name}")).fetchone()["id"]
        conn.execute("INSERT INTO reviews (asset_id, gate, passed, score) VALUES (%s, 'identity', true, 8)", (a,))
    body["scene_jobs"] = [[] for _ in body["shots"]]
    body["scene_jobs"][1] = [str(scene_job)]
    _seg(conn, policy, post, body, chosen, 1)
    for score in (3, 3):
        _gate(conn, body["talk_segment_jobs"][1][-1], "identity", score)
        j = _seg(conn, policy, post, body, chosen, 1)
    assert j["kind"] == "comfy.graph" and j["payload"]["repair"]["knob"] == "next_photo"
    assert j["payload"]["image_uri"] == "s3://b/alt.png"


def test_the_route_repair_config_overrides_only_its_own_gates(policy):
    cfg = production.route_repair_cfg(policy, policy.get("generation", "talk_routes", "ltx25"))
    shared = policy.get("review", "video", "talk_repair")
    assert cfg["gates"]["lip_sync"]["steps"] == ["seed", "cutaway"] and cfg["audio_guidance"] == {}
    assert cfg["gates"]["identity"] == shared["gates"]["identity"] and cfg["cutaway"] == shared["cutaway"]
    assert production.route_repair_cfg(policy, {}) == {**shared, "gates": shared["gates"]}


def test_an_unknown_route_falls_back_to_the_policy_order(policy):
    assert production.route_order(policy, {"talk_route": ["nope"]}) == policy.get("generation", "talk_route")
    assert production.route_order(policy, {"talk_route": "ltx25"}) == ["ltx25"]


# ------------------------------------------------------------------------------------------- the maren_final plan

def _maren(conn):
    voice = {"engine": "indextts", "ref_uri": "s3://b/ref.wav", "ref_text": "hello there"}
    handle = f"c{uuid.uuid4().hex[:6]}"
    conn.execute("INSERT INTO creators (handle, bible, anchors) VALUES (%s, %s, %s)",
                 (handle, Jsonb({**BIBLE, "display_name": "M"}),
                  Jsonb({"reference_images": [{"uri": f"s3://b/r{i}"} for i in range(3)], "voice": voice})))
    return handle


def test_the_maren_final_plan_is_one_story_time_video_on_ltx_with_infinitetalk_behind(raw_policy):
    pol = breadth.test_policy(raw_policy)
    plan, items = breadth.load_plan(pol, "maren_final")
    assert len(items) == 1 and float(plan["max_usd"]) == 4 and plan["creator"] == "maren.afterhours"
    item = items[0]
    assert item.format == "talking_video" and item.body["edit_grammar"] == "story_time"
    assert item.body["talk_route"] == ROUTE and item.body["shared_key"] == "maren-final"
    assert "voice_engine" not in item.options                     # whatever the voice gate resolves to
    assert all(v == ["runpod_secure_l40s"] for v in plan["pools"].values())   # L40S only
    words = sum(len(ln["text"].split()) for s in item.body["shots"] for ln in s["performance"]["lines"])
    seconds = words / float(pol.get("generation", "talk_long", "words_per_second"))
    assert 34 <= seconds <= 45
    assert any(s.get("cutaway") for s in item.body["shots"])


def test_the_maren_final_dry_run_prices_ltx_takes(conn, raw_policy, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    handle = _maren(conn)
    dry = breadth.run(conn, raw_policy, "maren_final", creator=handle, dry_run=True)
    assert not dry["skipped"], dry["skipped"]
    jobs = dry["items"][0]["estimate"]["jobs"]
    talk = [j for j in jobs if j.get("route")]
    assert talk and all(j["kind"] == "comfy.graph" and j["pool"] == "runpod_secure_l40s" for j in talk)
    assert all(j["source"].startswith("policy (comfy.graph:ltx25_a2v") for j in talk)
    assert any(j["kind"] == "image.narrated" for j in jobs)        # the cutaways
    assert 0 < dry["estimated_usd"] < 4
    assert conn.execute("SELECT count(*) n FROM jobs").fetchone()["n"] == 0


def test_each_route_gets_its_own_repairs_under_the_takes_cap(policy):
    # 2026-11-25 maren-final-3: three LTX takes per on-camera line failed lip sync, then the take's step dropped the
    # post at max_repairs_per_post (2) before InfiniteTalk was ever tried. The step's cap now covers every route.
    per = policy.get("review", "video", "max_repairs_per_post")
    assert production.take_repair_limit(policy, {"talk_route": ROUTE}) == (per + 1) * 2 - 1
    assert production.take_repair_limit(policy, {"talk_route": "infinitetalk"}) == per
