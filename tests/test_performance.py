"""Performance direction (manyworlds.performance; docs/2026-11-26-performance-direction.md): one control language,
compiled into one coherent performance per model, observed by the judges we already call (rank only), stored per
asset, and calibrated only from reliable observations."""

from __future__ import annotations

import contextlib
import dataclasses
import uuid

import pytest
from psycopg.types.json import Jsonb

from manyworlds import direction, performance
from manyworlds.direction import Beat, ControlLevel, FramingSpec, PerformanceSpec, SetSpec, ShotSheet

BIBLE = {"appearance": {"apparent_age": 29, "gender_presentation": "woman", "ethnicity_and_skin": "olive skin",
                        "face": "oval face", "eyes": "brown eyes", "hair": "black bob with a clip",
                        "build": "slim", "signature_details": "thin gold wire glasses"}}


def _with(policy, **sections):
    """The policy with performance sub-sections replaced, e.g. calibration={"enabled": True}."""
    perf = {**policy.body["performance"], **{k: {**policy.body["performance"][k], **v} for k, v in sections.items()}}
    return dataclasses.replace(policy, body={**policy.body, "performance": perf})


def phrase(control, level, channel, policy=None):
    return performance.library(policy)["controls"][control]["levels"][level][channel]


# --- the library ----------------------------------------------------------------------------------------------------

def test_the_shipped_library_is_sound_and_names_the_static_controls(policy):
    lib = performance.library(policy)
    assert lib["version"] == "performance-v1"
    assert performance.problems(lib) == []                       # names, levels, channels, positive phrases
    assert tuple(lib["controls"]) == performance.CONTROL_NAMES   # the strict schema's enum is the library's
    assert set(lib["targets"]) == {"photo", "i2v", "talk", "voice"}
    assert policy.get("performance", "file") == "performance-v1.yaml"


def test_a_negation_or_unknown_name_is_a_problem():
    bad = {"controls": {"sass": {"bipolar": False, "levels": {1: {"face": "no smile"}, 5: {"nose": "x"}},
                                 "leaks": ["sparkle"]}},
           "contexts": {"gala": {"sparkle": 1}}, "compose": {"conflicts": [{"pair": ["sass", "glow"]}]}}
    found = " ".join(performance.problems(bad))
    for what in ("negation", "level 5", "unknown channel 'nose'", "unknown control 'sparkle'", "'glow'"):
        assert what in found, what


def test_a_context_shifts_her_baseline_within_the_levels(policy):
    base = {"elegance": 3, "energy": 0, "polish": 2}
    out = performance.apply_context(base, "luxury", policy)
    assert out["elegance"] == 4 and out["energy"] == 0 and out["polish"] == 4      # clamped to 0-4
    assert out["gesture"] == 1                                                    # from neutral 2, -1
    hers = {"contexts": {"luxury": {"elegance": 1}}}
    assert performance.apply_context(base, "luxury", policy, profile=hers)["elegance"] == 4
    assert performance.apply_context(base, {"sass": 9}, policy)["sass"] == 4       # offsets capped at max_context_offset


# --- compose: one coherent performance, not a list ------------------------------------------------------------------

def test_compose_writes_the_intent_first_then_a_few_salient_phrases_per_channel(policy):
    out = performance.compose("Dry, amused confidence that warms at the end.",
                              {"sass": 3, "warmth": 2, "confidence": 3, "energy": 2, "playfulness": 1,
                               "candidness": 2},
                              "photo", policy, baseline={"warmth": 2, "energy": 2, "confidence": 1, "candidness": 2,
                                                         "playfulness": 1})
    assert out["text"].startswith("Dry, amused confidence that warms at the end; ")
    kept = [u["control"] for u in out["used"]]
    assert kept == ["confidence", "sass", "playfulness"]    # furthest from her baseline first, then from neutral
    assert "warmth" not in kept and "energy" not in kept    # at her baseline and at neutral: nothing to say
    assert len(kept) <= performance.library(policy)["compose"]["max_controls"]
    assert all(len(v) <= performance.library(policy)["compose"]["per_channel"] for v in out["phrases"].values())
    assert phrase("confidence", 3, "face") in out["text"] and phrase("sass", 3, "face") in out["text"]
    assert phrase("playfulness", 1, "face") not in out["text"]   # two phrases a channel: the face is full
    # channels in the target's order after the intent
    labels = performance.library(policy)["compose"]["channel_labels"]
    assert out["text"].index(f"{labels['face']}:") < out["text"].index(f"{labels['gaze']}:")
    assert "walk" not in out["phrases"] and "voice" not in out["phrases"]   # a photo has no walk or voice


def test_conflicts_resolve_by_context_else_the_stronger_request(policy):
    both = {"elegance": 3, "flamboyance": 3}
    plain = performance.compose("", both, "photo", policy)
    assert any(n.startswith("flamboyance 3 -> 2") for n in plain["notes"])        # tie: the first of the pair wins
    party = performance.compose("", both, "photo", policy, context="outfit_check")
    assert any(n.startswith("elegance 3 -> 2") and "outfit_check context favours flamboyance" in n
               for n in party["notes"])                                         # the context's favourite wins
    louder = performance.compose("", {"elegance": 3, "flamboyance": 4}, "photo", policy, extreme_reason="a reveal")
    assert any(n.startswith("elegance 3 -> 2") for n in louder["notes"])         # else the stronger request
    assert [u["control"] for u in louder["used"]] == ["flamboyance"]             # elegance at neutral says nothing


def test_level_four_needs_a_reason_and_her_range_holds(policy):
    out = performance.compose("", {"playfulness": 4}, "photo", policy)
    assert out["used"][0]["level"] == 3 and "needs a stated reason" in out["notes"][0]
    ok = performance.compose("", {"playfulness": 4}, "photo", policy, extreme_reason="comedy: a mock catwalk")
    assert ok["used"][0]["level"] == 4
    hers = performance.compose("", {"playfulness": 4}, "photo", policy, extreme_reason="comedy",
                               profile={"natural_range": {"playfulness": [0, 2]}, "baseline": {"playfulness": 1}})
    assert hers["used"][0]["level"] == 2 and "natural range" in " ".join(hers["notes"])


def test_calibration_offsets_change_the_phrase_not_the_request(policy):
    prof = {"calibration": {"photo": {"sass": {"2": 1}}}, "baseline": {"sass": 1}}
    out = performance.compose("", {"sass": 2}, "photo", policy, profile=prof)
    assert out["used"] == [{"control": "sass", "requested": 2, "level": 2, "compiled": 3}]
    assert phrase("sass", 3, "face") in out["text"] and phrase("sass", 2, "face") not in out["text"]


def test_edges_come_from_her_profile_else_the_library(policy):
    lib = performance.library(policy)["edges"]
    assert performance.edges(None, policy, "talk") == lib["talk"]
    assert performance.edges({"opening": "already laughing at something off camera"}, policy, "i2v") == {
        "opening": "already laughing at something off camera", "closing": lib["i2v"]["closing"]}


# --- voice ------------------------------------------------------------------------------------------------------------

def test_voice_params_follow_expressiveness_energy_and_pace(policy):
    v = performance.library(policy)["voice"]
    p = performance.voice_params({"vocal_expressiveness": 3, "energy": 4, "pace": 0}, "indextts", policy)
    assert p["alpha"] == v["indextts"]["emo_alpha_by_level"][3]
    assert p["boost"]["happy"] == pytest.approx(v["indextts"]["energy_boost"]["happy"] * 2)
    assert p["pause_scale"] == v["pause_scale_by_pace"][0]
    assert phrase("pace", 0, "voice") in p["instruct"]                       # the most salient voice phrases
    q = performance.voice_params({"energy": 4}, "qwen3", policy)
    assert q["alpha"] is None and q["boost"] == {}                           # Qwen3 takes no emotion vector
    excited = {"name": "excited", "vector": [0.65, 0, 0, 0, 0, 0, 0.3, 0], "alpha": 1.0}
    out = performance.apply_voice(excited, p, "indextts", policy)
    assert out["alpha"] == p["alpha"] and out["vector"][0] > 0.65 and out["vector"][6] > 0.3
    assert performance.apply_voice(None, performance.voice_params({"energy": 1}, "indextts", policy),
                                   "indextts", policy) is None


def test_the_speak_payload_carries_a_lines_performance(policy):
    from manyworlds.creator import voice_lab
    voice = {"engine": "indextts", "ref_uri": "s3://b/v.wav", "ref_text": "Hi."}
    calm = voice_lab.speak_payload(policy, voice, [{"text": "Hi.", "delivery": "excited", "pause_after_s": 0.5}])
    fast = voice_lab.speak_payload(policy, voice, [{"text": "Hi.", "delivery": "excited", "pause_after_s": 0.5,
                                                    "controls": {"vocal_expressiveness": 1, "pace": 4}}])
    assert calm["lines"][0]["emotion"]["alpha"] == policy.get("voice", "engines", "indextts", "emo_alpha")
    assert fast["lines"][0]["emotion"]["alpha"] == performance.library(policy)["voice"]["indextts"][
        "emo_alpha_by_level"][1]
    assert fast["lines"][0]["pause_after_s"] < 0.5 and "instruct" not in fast["lines"][0]
    q = voice_lab.speak_payload(policy, {**voice, "engine": "qwen3"},
                                [{"text": "Hi.", "controls": {"vocal_expressiveness": 4}}])
    assert "instruct" not in q["lines"][0]       # the hook stays off: Qwen3 Base takes no instruction


# --- closeness and leakage --------------------------------------------------------------------------------------------

def test_closeness_and_leakage(policy):
    assert performance.closeness({"sass": 2, "warmth": 2}, {"sass": 2, "warmth": 2}) == 1.0
    assert performance.closeness({"sass": 4}, {"sass": 0}) == 0.0
    assert performance.closeness({"sass": 4}, {"warmth": 0}) is None
    lib = performance.library(policy)
    out = performance.leakage({"sass": 4, "warmth": 2, "polish": 2, "confidence": 2},
                              {"sass": 3, "warmth": 0, "polish": 2, "confidence": 3}, lib, moved=["sass"])
    assert [x["control"] for x in out["leaks"]] == ["warmth"]          # warmth moved without being expected to
    assert [x["control"] for x in out["tolerated"]] == ["confidence"]  # sass leaks into confidence by design
    assert out["held"] == 3 and out["score"] == pytest.approx(1 / 3, abs=1e-3)


# --- the shot sheet and the compiler ----------------------------------------------------------------------------------

def directed(**perf) -> ShotSheet:
    return ShotSheet(set=SetSpec(location="her galley kitchen", light_quality="soft"),
                     framing=FramingSpec(shot_size="medium shot"),
                     blocking=[Beat(at_s=0, action="leans on the counter", eyeline="lens"),
                               Beat(at_s=3, action="sips her tea", eyeline="prop")],
                     performance=PerformanceSpec(**perf))


def test_a_directed_photo_says_the_moment_as_one_performance(policy):
    shot = directed(intent="Dry, amused confidence", controls=[ControlLevel(control="sass", level=3)],
                    energy="high", body_language="hip against the counter")
    p = direction.compile_prompt(shot, BIBLE, "photo", policy)
    assert "Performance: in this moment, dry, amused confidence; " in p
    assert phrase("sass", 3, "face") in p
    assert "high energy" not in p                               # no adjective line once directed
    assert "hip against the counter" in p                       # the body language stays with the blocking
    assert p.index("Action:") < p.index("Performance:") < p.index("Set:")
    assert "opening:" not in p                                  # a photo is one moment


def test_talk_and_i2v_get_natural_edges_and_a_faceless_shot_no_face(policy):
    lib = performance.library(policy)["edges"]
    shot = directed(intent="warm and a little conspiratorial", controls=[ControlLevel(control="warmth", level=3)])
    talk = direction.compile_prompt(shot, BIBLE, "talk", policy)
    assert f"opening: {lib['talk']['opening']}" in talk and f"closing: {lib['talk']['closing']}" in talk
    shot.performance.closing = "she laughs and looks down at her mug"
    assert "closing: she laughs and looks down at her mug" in direction.compile_prompt(shot, BIBLE, "i2v", policy)
    hers = {"opening": "already mid-laugh"}
    assert "opening: already mid-laugh" in direction.compile_prompt(shot, BIBLE, "talk", policy, profile=hers)
    away = directed(intent="calm", controls=[ControlLevel(control="warmth", level=3),
                                             ControlLevel(control="gesture", level=3)])
    away.cutaway = True
    q = direction.compile_prompt(away, BIBLE, "i2v", policy)
    assert phrase("warmth", 3, "face") not in q and "opening:" not in q and phrase("gesture", 3, "body") in q


def test_controls_keep_the_legacy_energy_and_idle_line(policy):
    idle = policy.get("direction", "idle", "by_energy")
    shot = directed(intent="sleepy", controls=[ControlLevel(control="energy", level=0)])
    assert idle["low"] in direction.compile_prompt(shot, BIBLE, "talk", policy)
    shot = directed(intent="up for it", controls=[ControlLevel(control="sass", level=2)], energy="high")
    assert direction.shot_controls(shot) == {"sass": 2, "energy": 3}
    assert direction.shot_controls(directed(energy="high")) == {}          # no controls: the old sheet as it was


def test_a_long_directed_prompt_stays_in_budget(policy):
    shot = directed(intent="a long, rich, dry, amused, confident, warm performance that turns over twice " * 4,
                    controls=[ControlLevel(control=c, level=4) for c in ("sass", "warmth", "candidness", "gesture")],
                    extreme_reason="a dare")
    for target in ("photo", "i2v", "talk"):
        p = direction.compile_prompt(shot, BIBLE, target, policy)
        assert direction.estimate_tokens(p) <= policy.get("direction", "prompt_budget", target)


def test_the_strict_shots_schema_leaves_performance_to_the_edit_pass():
    from manyworlds.llm import strict_json_schema
    shot = strict_json_schema(direction.DirectedShots)["properties"]["shots"]["items"]
    perf = shot["properties"]["performance"]["properties"]
    for k in (*direction.PERFORMANCE_FIELDS, "profile"):
        assert k not in perf
    assert "lines" in perf and "energy" in perf
    edit = strict_json_schema(direction.DirectedEdits)["properties"]["performances"]["items"]
    control = edit["properties"]["controls"]["items"]["properties"]["control"]
    assert control["enum"] == [*performance.CONTROL_NAMES, ""]
    back = direction.DirectedEdits.model_validate({"edits": [{}], "performances": [
        {"intent": "x", "context": "", "controls": [{"control": "", "level": ""}, {"control": "sass", "level": 3}],
         "extreme_reason": "", "opening": "", "closing": ""}]})
    assert performance.levels_of(back.performances[0].controls) == {"sass": 3}


def test_spoken_lines_carry_the_shots_levels():
    shot = directed(controls=[ControlLevel(control="pace", level=4)])
    shot.performance.lines = [direction.Line(text="Okay.", delivery="fast")]
    assert direction.spoken_lines(shot)[0]["controls"] == {"pace": 4}
    plain = directed()
    plain.performance.lines = [direction.Line(text="Okay.")]
    assert "controls" not in direction.spoken_lines(plain)[0]


# --- the dossier section ----------------------------------------------------------------------------------------------

PROFILE_OUT = {
    "baseline": [{"control": "warmth", "level": 2}, {"control": "sass", "level": 1}, {"control": "elegance", "level": 7},
                 {"control": "sparkle", "level": 2}],
    "natural_range": [{"control": "playfulness", "low": 3, "high": 0}, {"control": "sass", "low": 2, "high": 3}],
    "signature": {"resting_expression": "a calm, attentive face", "smile": "lopsided, arrives late",
                  "gaze_habits": "looks away to think", "posture": "upright", "walk": "unhurried",
                  "hands": "around a mug", "vocal_cadence": "a dry pause before the punchline",
                  "laugh": "a short exhale", "mannerisms": ["tucks her hair", "says 'right' to start"]},
    "opening": "already mid-thought", "closing": "a small nod and a glance away",
    "context_shifts": [{"context": "Quiet Morning", "offsets": [{"control": "energy", "level": -5},
                                                                {"control": "sparkle", "level": 1},
                                                                {"control": "warmth", "level": 0}]}],
}


def test_the_performance_section_is_checked_and_reaches_the_planners(policy):
    from manyworlds.creator import dossier
    assert "performance" in dossier.SECTIONS
    saved = dossier.clean(policy, "performance", dossier.PerformanceProfile.model_validate(PROFILE_OUT), {})["performance"]
    assert saved["baseline"] == {"warmth": 2, "sass": 1, "elegance": 4}          # unknown dropped, level clamped
    assert saved["natural_range"] == {"playfulness": [0, 3], "sass": [1, 3]}      # ordered, holds her baseline
    assert saved["contexts"] == {"quiet_morning": {"energy": -2}}                # capped, zero and unknown dropped
    view = dossier.planning_view({"performance": saved})
    assert view["performance"]["signature"]["smile"] == "lopsided, arrives late"
    # a stored profile goes through the same checks again when the talent manager changes one field
    again = dossier.checked_change(policy, {"performance": saved}, "performance.opening", "already laughing")
    assert again["performance"]["opening"] == "already laughing" and again["performance"]["baseline"] == saved["baseline"]
    with pytest.raises(ValueError):
        dossier.clean_performance(policy, {"baseline": [{"control": "sparkle", "level": 2}]})


def test_an_existing_creator_without_the_section_is_onboarded_on_the_next_review(conn, policy):
    from manyworlds.brain.agents import talent_manager
    from manyworlds.creator import dossier, profile
    cid = conn.execute("INSERT INTO creators (handle, status, bible) VALUES ('maren_p', 'active', '{}') RETURNING id"
                       ).fetchone()["id"]
    body = {s: {"x": 1} for s in dossier.SECTIONS if s != "performance"}
    profile.save(conn, policy, cid, {**body, "archetypes": ["lifestyle_slow"]}, author="test", reason="seed")
    assert dossier.missing(profile.current(conn, cid)[1]) == ["performance"]
    [job] = talent_manager.ensure_reviews(conn, policy)
    assert conn.execute("SELECT payload->>'mode' m FROM jobs WHERE id = %s", (job,)).fetchone()["m"] == "onboard"


# --- the director -----------------------------------------------------------------------------------------------------

def test_the_director_writes_intent_and_controls_in_the_edit_pass(conn, policy):
    from test_director import DirectorLLM, text
    from test_production import creator
    from manyworlds.brain import director
    from manyworlds.creator import profile

    cid = creator(conn)
    profile.save(conn, policy, cid, {"performance": {"baseline": {"warmth": 2, "sass": 1}, "opening": "mid-thought"}},
                 author="test", reason="seed")
    conn.execute("INSERT INTO performance_calibration (model, target, control, level, level_offset) "
                 "VALUES ('image.zimage', 'photo', 'sass', 3, 1)")

    class PerfLLM(DirectorLLM):
        def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
            if schema.__name__ == "DirectedEdits":
                super().structured(role, system=system, content=content, schema=schema)
                return schema.model_validate({"edits": [{}], "performances": [
                    {"intent": "dry amused confidence", "context": "outfit_check",
                     "controls": [{"control": "sass", "level": 3}, {"control": "sass", "level": 2}],
                     "extreme_reason": "", "opening": "", "closing": ""}]})
            return super().structured(role, system=system, content=content, schema=schema)

    llm = PerfLLM({"shots": [{"set": {"location": "a mirror in the hallway"}}]}, conn=conn)
    [shot] = director.direct(conn, policy, llm, {"scenes": ["outfit check in the hallway"]}, cid, fmt="photo")
    perf = shot["performance"]
    assert perf["intent"] == "dry amused confidence" and perf["context"] == "outfit_check"
    assert perf["controls"] == [{"control": "sass", "level": 2}]                 # one per name, the last wins
    assert perf["profile"]["baseline"] == {"warmth": 2, "sass": 1}
    assert perf["profile"]["calibration"] == {"photo": {"sass": {"3": 1}}}
    brief = text(llm.calls[0]["content"])
    assert "performance vocabulary" in brief and "Her performance profile" in brief and "mid-thought" in brief
    assert "Level 4 only with a reason" in text(llm.calls[1]["content"])
    assert "Performance: in this moment, dry amused confidence" in direction.compile_prompt(shot, BIBLE, "photo",
                                                                                           policy)


def test_an_unknown_context_is_cleared(policy):
    from manyworlds.brain import director
    out = director.directed_performance(policy, {"intent": "x", "context": "moon_landing", "controls": []}, {"grwm"})
    assert out["context"] == ""


# --- the judges: observed, ranked, stored, never rejecting ------------------------------------------------------------

DIRECTED_SHOT = directed(intent="dry amused confidence",
                         controls=[ControlLevel(control="sass", level=3), ControlLevel(control="warmth", level=1)]
                         ).model_dump(mode="json")


def _image_job(conn, store, tmp_path, n=2, *, shot=None, post_body=None):
    from test_review import creator_with_job
    cid, job = creator_with_job(conn, store, tmp_path, n_images=n)
    post = None
    if post_body is not None:
        post = conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, 'carousel', 'producing', "
                            "%s) RETURNING id", (cid, Jsonb(post_body))).fetchone()["id"]
    conn.execute("UPDATE jobs SET payload = %s, post_id = %s WHERE id = %s",
                 (Jsonb({"shot": shot} if shot else {}), post, job))
    return cid, job, post


def _uris(conn, job):
    return [r["id"] for r in conn.execute("SELECT id FROM assets WHERE job_id = %s ORDER BY uri", (job,)).fetchall()]


def test_the_image_judge_observes_the_shot_and_ranks_by_appeal_without_rejecting(conn, policy, store, tmp_path):
    from test_review import JudgeStub, ok
    from manyworlds.production.steps import _best_confirmed
    from manyworlds.review import gates
    _, job, _ = _image_job(conn, store, tmp_path, shot=DIRECTED_SHOT)
    perf = {"observed_levels": [{"control": "sass", "level": 3}, {"control": "warmth", "level": 1},
                                {"control": "polish", "level": 4}], "pose_expression": "hip on the counter, smirk"}
    judge = JudgeStub([ok(), ok()], confirm=[ok(appeal=3, charisma=3, expression=3, naturalness=1, **perf),
                                             ok(appeal=9, charisma=8, expression=8, naturalness=9, **perf)])
    results = gates.review_job(conn, policy, judge, store, job, tmp_path)
    assert all(r["accepted"] for r in results)                       # naturalness 1 ranks low, never rejects
    ask = " ".join(str(b.get("text", "")) for b in judge.calls[0]["content"])
    assert "PERFORMANCE (rank only" in ask and "sass:" in ask and "warmth:" in ask
    assert "dry amused confidence" in ask and "naturalness" in ask.lower()
    assert "sass 3" not in ask                                       # blind to the requested levels
    rows = conn.execute("SELECT passed, score FROM reviews WHERE gate = 'performance' ORDER BY score").fetchall()
    assert [r["passed"] for r in rows] == [True, True] and float(rows[0]["score"]) < float(rows[1]["score"])
    obs = conn.execute("SELECT * FROM performance_observations ORDER BY points").fetchall()
    assert len(obs) == 2 and obs[0]["requested"] == {"sass": 3, "warmth": 1}
    assert obs[0]["observed"] == {"sass": 3, "warmth": 1}               # polish was not asked: left out
    assert float(obs[0]["closeness"]) == 1.0 and obs[0]["model"] == "image.zimage" and obs[0]["stage"] == "confirm"
    assert obs[0]["descriptor"] == "hip on the counter, smirk" and obs[0]["target"] == "photo"
    best = _best_confirmed(conn, [str(job)])
    assert best["id"] == _uris(conn, job)[1]                            # the more appealing, natural one


def test_without_observation_the_judge_is_asked_nothing_more(conn, policy, store, tmp_path):
    from test_review import JudgeStub, ok
    from manyworlds.review import gates
    _, job, _ = _image_job(conn, store, tmp_path, shot=DIRECTED_SHOT)
    off = _with(policy, observe={"enabled": False})
    judge = JudgeStub([ok(), ok()])
    gates.review_job(conn, off, judge, store, job, tmp_path)
    assert "PERFORMANCE" not in " ".join(str(b.get("text", "")) for b in judge.calls[0]["content"])
    assert conn.execute("SELECT count(*) n FROM performance_observations").fetchone()["n"] == 0


def test_variety_prefers_a_frame_that_does_not_repeat_an_earlier_scene(conn, policy, store, tmp_path):
    from test_review import JudgeStub, ok, creator_with_job
    from manyworlds.production.steps import _best_confirmed
    from manyworlds.review import gates
    cid, first = creator_with_job(conn, store, tmp_path, n_images=1)
    second = conn.execute("INSERT INTO jobs (kind, status, creator_id) VALUES ('image.zimage', 'done', %s) RETURNING id",
                          (cid,)).fetchone()["id"]
    for i in range(2):
        p = tmp_path / f"v{uuid.uuid4().hex}.png"
        p.write_bytes(b"\x89PNG v%d" % i)
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'image', %s, "
                     "'x', 1)", (second, cid, store.put_file(p, f"t/{p.name}").uri))
    post = conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, 'carousel', 'producing', %s) "
                        "RETURNING id", (cid, Jsonb({"scene_jobs": [[str(first)], [str(second)]]}))).fetchone()["id"]
    conn.execute("UPDATE jobs SET post_id = %s WHERE id = ANY(%s)", (post, [first, second]))
    same = "leaning on the counter, half-smile, eyes down"
    gates.review_job(conn, policy, JudgeStub([ok()], confirm=[ok(appeal=7, pose_expression=same)]), store, first,
                     tmp_path)
    a, b = _uris(conn, second)
    gates.review_job(conn, policy, JudgeStub([ok(), ok()], confirm=[ok(appeal=8, pose_expression=same),
                                                                     ok(appeal=7, pose_expression="spinning, laughing, "
                                                                                                  "hair across her face")]),
                     store, second, tmp_path)
    assert _best_confirmed(conn, [str(second)])["id"] == b               # appeal 8 repeats scene 1; 7 is new
    no_penalty = conn.execute("SELECT id FROM assets WHERE id = %s", (a,)).fetchone()
    assert no_penalty and _best_confirmed(conn, [str(first)])["id"] == _uris(conn, first)[0]


def test_the_clip_judge_observes_too(conn, policy, store, tmp_path):
    from test_production import creator, real_refs
    from test_video_review import ClipStub, clip, clip_job, rendered
    from manyworlds.review import video
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = clip_job(conn, store, tmp_path, cid, kind="talk.infinitetalk")
    conn.execute("UPDATE jobs SET payload = payload || %s WHERE id = %s", (Jsonb({"shot": DIRECTED_SHOT}), job))
    rendered(conn, store, tmp_path, job)
    llm = ClipStub(clip(appeal=7, naturalness=8, observed_levels=[{"control": "sass", "level": 2}]))
    [r] = video.review_job(conn, policy, llm, store, job, tmp_path)
    assert r["accepted"] is not None
    assert "PERFORMANCE (rank only" in " ".join(str(b.get("text", "")) for b in llm.calls[0]["content"])
    o = conn.execute("SELECT target, observed, closeness FROM performance_observations").fetchone()
    assert o["target"] == "talk" and o["observed"] == {"sass": 2} and float(o["closeness"]) == 0.75


# --- calibration ------------------------------------------------------------------------------------------------------

def _observe(conn, cid, *, judge, requested, observed, n, model="image.zimage", target="photo", posts=None):
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id) VALUES (%s, 'done', %s) RETURNING id",
                       (model, cid)).fetchone()["id"]
    assets = []
    for i in range(n):
        post = None
        if posts:
            post = conn.execute("INSERT INTO posts (creator_id, format) VALUES (%s, 'photo') RETURNING id",
                                (cid,)).fetchone()["id"]
        aid = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'image', "
                           "%s, 'x', 1) RETURNING id", (job, cid, f"s3://b/{uuid.uuid4().hex}.png")).fetchone()["id"]
        assets.append(aid)
        performance.record_observations(conn, [{"asset_id": aid, "post_id": post, "creator_id": cid, "job_id": job,
                                                "model": model, "judge": judge, "stage": "confirm", "target": target,
                                                "requested": requested, "observed": observed}])
    return assets


def _second_judge(conn, cid, assets, observed):
    for aid in assets:
        performance.record_observations(conn, [{"asset_id": aid, "creator_id": cid, "model": "image.zimage",
                                                "judge": "qwen4-max", "stage": "second", "target": "photo",
                                                "requested": {}, "observed": observed}])


def test_calibration_waits_for_its_switch_and_for_measured_agreement(conn, policy):
    from test_production import creator
    cid = creator(conn)
    _observe(conn, cid, judge="claude-sonnet-5-5", requested={"sass": 3}, observed={"sass": 2}, n=14, posts=True)
    assert performance.refresh_calibration(conn, policy)["reason"].startswith("performance.calibration.enabled")
    on = _with(policy, calibration={"enabled": True})
    out = performance.refresh_calibration(conn, on)
    assert out["changed"] == [] and "agreement" in out["reason"]
    rep = performance.report(conn, on)
    assert "waiting" in rep["calibration"] and rep["cells"][0]["n"] == 14
    assert conn.execute("SELECT count(*) n FROM performance_calibration").fetchone()["n"] == 0


def test_calibration_steps_once_from_reliable_observations(conn, policy):
    from test_production import creator
    cid = creator(conn)
    on = _with(policy, calibration={"enabled": True})
    assets = _observe(conn, cid, judge="claude-sonnet-5-5", requested={"sass": 3}, observed={"sass": 2}, n=24,
                      posts=True)
    _second_judge(conn, cid, assets, {"sass": 2})                     # another family agrees: agreement measured
    out = performance.refresh_calibration(conn, on)
    assert out["judge"] == "claude-sonnet-5-5" and len(out["changed"]) == 1
    row = conn.execute("SELECT level_offset FROM performance_calibration WHERE model = 'image.zimage' "
                       "AND target = 'photo' AND control = 'sass' AND level = 3").fetchone()
    assert row["level_offset"] == 1                                    # undershoot: asked one higher
    assert conn.execute("SELECT count(*) n FROM decisions WHERE kind = 'performance_calibration'").fetchone()["n"] == 1
    again = performance.refresh_calibration(conn, on)                  # capped at max_offset: no second step
    assert again["changed"] == [] and conn.execute("SELECT count(*) n FROM decisions WHERE kind = "
                                                   "'performance_calibration'").fetchone()["n"] == 1
    assert performance.profile_offsets(conn, on) == {"photo": {"sass": {"3": 1}}}


def test_too_few_posts_or_a_noisy_shift_never_adjusts(conn, policy):
    from test_production import creator
    cid = creator(conn)
    on = _with(policy, calibration={"enabled": True})
    few = _observe(conn, cid, judge="claude-sonnet-5-5", requested={"sass": 3}, observed={"sass": 2}, n=24)  # no posts
    _second_judge(conn, cid, few, {"sass": 2})
    out = performance.refresh_calibration(conn, on)
    assert out["changed"] == [] and out["held"] == 1
    cells = performance.calibration_cells(conn, on, judge="claude-sonnet-5-5")
    assert "posts" in performance.reliable(cells[0], on)


def test_the_chief_of_staff_takes_the_calibration_step(conn, policy):
    from manyworlds.brain import chief_of_staff
    out = chief_of_staff.tick(conn, policy)
    assert out["performance_calibration"]["changed"] == 0 and "enabled" in out["performance_calibration"]["reason"]


# --- our own content in the research vocabulary -----------------------------------------------------------------------

class ReadLLM:
    def __init__(self):
        self.calls = []

    def model_for(self, role):
        return f"stub-{role}"

    @contextlib.contextmanager
    def tagged(self, agent, job_id, before_call=None, attempt=None):
        self.calls.append({"agent": agent})
        yield

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        self.calls.append({"role": role, "system": system, "content": content})
        n = sum(1 for b in content if str(b.get("text", "")).startswith("Post "))
        return schema.model_validate({"reads": [{"post": i + 1, "annotation": {"observed": {"coded": {
            "candid_level": "candid", "expression": "not_a_listed_value"}}}} for i in range(n)]})


def test_accepted_own_assets_are_coded_once_in_the_research_vocabulary(conn, policy, store, tmp_path):
    from test_review import JudgeStub, ok
    from manyworlds.brain import evidence
    from manyworlds.research import annotation as ann
    from manyworlds.review import gates, own_annotation
    _, job, post = _image_job(conn, store, tmp_path, n=2, shot=DIRECTED_SHOT, post_body={"caption": "tea at 2am"})
    gates.review_job(conn, policy, JudgeStub([ok(), ok(identity=1)], confirm=[ok(appeal=8)]), store, job, tmp_path)
    assert own_annotation.run_pending(conn, policy, ReadLLM(), store, tmp_path) == {"enabled": False}
    on = _with(policy, own_annotation={"enabled": True})
    llm = ReadLLM()
    out = own_annotation.run_pending(conn, on, llm, store, tmp_path / "own")
    assert out == {"enabled": True, "assets": 1, "annotated": 1, "calls": 1}          # the rejected one is never read
    assert llm.calls[0]["agent"] == on.get("performance", "own_annotation", "agent")
    assert llm.calls[1]["role"] == on.get("performance", "own_annotation", "role")
    assert "tea at 2am" in " ".join(str(b.get("text", "")) for b in llm.calls[1]["content"])
    row = conn.execute("SELECT * FROM reference_annotations").fetchone()
    assert row["source"] == "own" and row["post_id"] is None and row["own_post_id"] == post
    assert row["schema"] == ann.ANNOTATION_SCHEMA and row["observed"]["coded"]["candid_level"] == "candid"
    assert row["observed"]["coded"]["expression"] == "unknown"                   # off-list values coerced, as research
    assert own_annotation.run_pending(conn, on, ReadLLM(), store, tmp_path / "own")["annotated"] == 0   # idempotent
    assert evidence.annotated_posts(conn, mode="real", schemas=ann.READ_SCHEMAS) == []   # research readers never see it
    [own] = evidence.own_annotations(conn, mode="sandbox", schemas=ann.READ_SCHEMAS)
    assert own["post_id"] == str(post) and own["performance"]["requested"] == {"sass": 3, "warmth": 1}
    assert evidence.own_annotations(conn, mode="real", schemas=ann.READ_SCHEMAS) == []   # a sandbox creator


def test_a_reference_annotation_must_say_whose_it_is(conn):
    import psycopg
    with pytest.raises(psycopg.errors.CheckViolation):
        with conn.transaction():
            conn.execute("INSERT INTO reference_annotations (schema, source) VALUES ('iia_v2', 'external')")
