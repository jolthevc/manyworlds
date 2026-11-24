"""A shot with no face of hers in it (her point of view, or a cutaway): its prompt describes her hands, not her face
or eyes, and no gate judges it on identity (review 2026-11-22: POV B-roll either drew her face or failed identity)."""

from __future__ import annotations

from manyworlds import direction
from manyworlds.review import gates, video

from test_review import BIBLE

POV = {"framing": {"device": "pov", "shot_size": "close-up"},
       "blocking": [{"at_s": 0, "action": "pours tea into a mug", "eyeline": "prop"}]}
FACE = {"framing": {"device": "propped", "shot_size": "medium shot"},
        "blocking": [{"at_s": 0, "action": "pours tea into a mug", "eyeline": "lens"}]}


def test_faceless_shots_are_pov_and_cutaways():
    assert direction.faceless(POV) and direction.faceless({**FACE, "cutaway": True})
    assert not direction.faceless(FACE) and not direction.faceless(None)


def test_a_pov_prompt_names_her_hands_not_her_face(policy):
    for target in ("photo", "i2v"):
        prompt = direction.compile_prompt(POV, BIBLE, target, policy)
        assert "hands and forearms" in prompt and "olive skin" in prompt
        assert "oval face" not in prompt and "brown eyes" not in prompt
    face = direction.compile_prompt(FACE, BIBLE, "photo", policy)
    assert "oval face" in face


def test_no_identity_gate_for_a_faceless_shot():
    assert "identity" not in gates.gates_for("a kitchen", POV)
    assert "identity" in gates.gates_for("a kitchen", FACE)
    assert "identity" not in video.gates_for("a kitchen", POV)
    assert "identity" in video.gates_for("a kitchen", FACE)


def test_props_beyond_the_feasible_number_are_cut_from_the_prompt(policy):
    cap = policy.get("direction", "feasibility", "max_props")
    props = ["a mug", "a kettle", "a cat", "a scarf", "a teapot"]
    shot = {**FACE, "set": {"location": "a small kitchen", "props": ", ".join(props)}}
    prompt = direction.compile_prompt(shot, BIBLE, "photo", policy)
    assert all(p in prompt for p in props[:cap]) and not any(p in prompt for p in props[cap:])


def test_both_clear_passes_ignore_identity_on_a_faceless_shot(policy):
    """A POV or cutaway clip's identity score judges nothing: a low one must not send it to the paid confirm."""
    from manyworlds.review import common
    assert not common.judges_identity(POV) and common.judges_identity(FACE)
    assert not common.judges_identity(None, plate=True)
    cfg = policy.get("review", "video")
    m = int(cfg["clear_pass_margin"])
    bars = {k: int(cfg[f"min_{k}"]) + m for k in ("motion", "realism")}
    r = video.ClipReview(identity=1, defects=[], looks_under_min_age=False, sexualized=False, real_brand_marks=False,
                         notes="fine", **bars)
    assert video.clear_pass(r, cfg, POV) and video.clear_pass(r, cfg, {**FACE, "cutaway": True})
    assert not video.clear_pass(r, cfg, FACE) and not video.clear_pass(r, cfg)


def test_a_faceless_clip_with_a_low_identity_score_is_cleared_without_a_confirm(conn, policy, store, tmp_path):
    from psycopg.types.json import Jsonb
    from test_production import creator, real_refs
    from test_video_review import ClipStub, _put, clip, rendered
    cfg = policy.get("review", "video")
    m = int(cfg["clear_pass_margin"])
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    payload = {"image_uri": _put(store, tmp_path, b"\x89PNG start"), "scene": "pouring tea", "shot": POV}
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('comfy.wan_i2v', 'queued', %s, %s) "
                       "RETURNING id", (cid, Jsonb(payload))).fetchone()["id"]
    rendered(conn, store, tmp_path, job)
    llm = ClipStub(clip(identity=1, motion=int(cfg["min_motion"]) + m, realism=int(cfg["min_realism"]) + m))
    [r] = video.review_job(conn, policy, llm, store, job, tmp_path)
    assert r["accepted"] and r["confirmed"] and r.get("clear_pass") and len(llm.calls) == 1


def test_a_faceless_photo_with_a_low_identity_score_is_cleared_without_a_confirm(conn, raw_policy, store, tmp_path):
    from psycopg.types.json import Jsonb
    from manyworlds import policy as policy_mod
    from test_review import JudgeStub, creator_with_job, ok
    pol = policy_mod.from_dict({**raw_policy.body, "operations": {**raw_policy.body["operations"], "mode": "live"},
                                "llm": {**raw_policy.body["llm"], "shadow": {}}})
    cfg = pol.get("review", "image")
    m = int(cfg["clear_pass_margin"])
    _, job = creator_with_job(conn, store, tmp_path, n_images=1)
    conn.execute("UPDATE jobs SET payload = %s WHERE id = %s", (Jsonb({"scene": "a kitchen", "shot": POV}), job))
    judge = JudgeStub([ok(identity=1, realism=int(cfg["min_realism"]) + m)])
    [r] = gates.review_job(conn, pol, judge, store, job, tmp_path)
    assert r["accepted"] and r.get("clear_pass") and r["confirmed"] and len(judge.calls) == 1
