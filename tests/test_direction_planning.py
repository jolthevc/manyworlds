"""The planner and the showrunner write scenes, never shot sheets (ShotSheet inside their structured outputs is too
large a grammar: HTTP 400, 2026-11-22); the director writes the shots later (tests/test_director.py). Moves are
recorded on ideas, only moves the models can carry are offered, and the shot helpers pair scenes with shots."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from types import SimpleNamespace

from psycopg.types.json import Jsonb

from manyworlds.brain import archetypes, ideas, planner
from manyworlds.brain.agents import showrunner

from brain_helpers import creator as bare_creator
from test_brain_showrunner import PlanLLM, _maren, _run, idea
from test_production import START, PlanStub, creator, post

KITCHEN = {"set": {"location": "her galley kitchen", "time_of_day": "early morning", "set_id": "home_kitchen"},
           "wardrobe": {"outfit": "a grey hoodie", "outfit_id": "grey_hoodie"},
           "framing": {"shot_size": "medium shot", "camera_move": "drone orbit"},
           "blocking": [{"at_s": 0, "action": "pours coffee into a chipped mug", "eyeline": "prop"}]}
WINDOW = {"set": {"location": "the bakery window downstairs"}, "framing": {"shot_size": "close-up"}}


def test_the_planner_and_showrunner_schemas_have_no_shots():
    assert "shots" not in planner.PlannedPost.model_fields
    assert "shots" not in showrunner.PlannedIdea.model_fields
    assert "ShotSheet" not in str(planner.ContentPlan.model_json_schema())
    assert "ShotSheet" not in str(showrunner.IdeaBatch.model_json_schema())


def test_planned_posts_carry_no_shots(conn, policy):
    cid = creator(conn)
    planner.plan_period(conn, policy, PlanStub([post(0, "photo", ["", "fogged glass of the bakery window"],
                                                     shots=[KITCHEN, WINDOW])]), cid, start=START)   # extra: ignored
    body = conn.execute("SELECT body FROM posts WHERE creator_id = %s", (cid,)).fetchone()["body"]
    assert "shots" not in body
    assert body["scenes"] == ["fogged glass of the bakery window"]            # empty scenes dropped, as before


def test_scenes_and_shots_stay_paired_when_one_list_is_longer():
    p = SimpleNamespace(scenes=["a", "", "c"], shots=[KITCHEN])
    planner.align_shots(p)
    assert len(p.scenes) == len(p.shots) == 2 and p.scenes == ["a", "c"]      # the empty pair is dropped
    assert p.shots[0].set.location == "her galley kitchen" and p.shots[1].set.location == ""
    written = SimpleNamespace(scenes=[""], shots=[KITCHEN])
    planner.align_shots(written)
    assert "galley kitchen" in written.scenes[0] and "pours coffee" in written.scenes[0]   # written from the shot
    assert planner.shots_body(written)[0]["set"]["set_id"] == "home_kitchen"


def test_a_shots_lines_follow_the_voice_rules(policy):
    salesy = {**WINDOW, "performance": {"lines": [{"text": "Drop a comment if you agree", "delivery": "bright"}]}}
    assert planner.shot_voice_issues(policy, [salesy])
    assert planner.shot_voice_issues(policy, [KITCHEN, WINDOW]) == []


def test_the_showrunner_records_the_move(conn, policy):
    cid = _maren(conn)
    scenes = dict(fmt="carousel", scenes=("her galley kitchen at dawn", "the bakery window at dawn"))
    _run(conn, policy, cid, PlanLLM([idea(0, move="b_roll_detail", **scenes),
                                     idea(1, move="walk_to_camera", **scenes)], conn=conn))   # not a carousel move
    rows = conn.execute("SELECT body, attributes FROM ideas ORDER BY body->>'day'").fetchall()
    body = rows[0]["body"]
    assert "shots" not in body and len(body["scenes"]) == 2
    assert body["move"] == "b_roll_detail" and rows[0]["attributes"]["move"] == "b_roll_detail"
    assert "move" not in rows[1]["body"] and "move" not in rows[1]["attributes"]


def test_moves_are_offered_only_when_ready_enough_and_made_now(conn, policy):
    cid = _maren(conn)
    llm = PlanLLM([idea(0)], conn=conn)
    _run(conn, policy, cid, llm)
    prompt = str(llm.calls[0]["content"])
    assert "b_roll_detail" in prompt and "walk_to_camera" in prompt             # ready and partial (story clips)
    assert "storytime_to_camera" not in prompt                                   # talking video isn't made yet
    ready_only = dataclasses.replace(policy, body={**policy.body, "direction": {
        **policy.body["direction"], "offer_readiness": ["ready"]}})
    llm = PlanLLM([idea(0)], conn=conn)
    _run(conn, ready_only, cid, llm)
    prompt = str(llm.calls[0]["content"])
    assert "b_roll_detail" in prompt and "walk_to_camera" not in prompt and "phone_down_reveal" not in prompt
    assert set(showrunner.moves_for(ready_only, ["story_clip"])) == {"point_at_caption", "b_roll_detail"}


def test_the_showrunner_thinks_in_capabilities_and_archetype(conn, policy):
    cid = _maren(conn)
    conn.execute("UPDATE creators SET bible = bible || %s WHERE id = %s",
                 (Jsonb({"archetype": "slow living in a flat above a bakery", "niche": "lifestyle"}), cid))
    llm = PlanLLM([idea(0)], conn=conn)
    _run(conn, policy, cid, llm)
    prompt = str(llm.calls[0]["content"])
    assert "What the studio can render now" in prompt
    assert policy.get("capabilities", "devices", "pov", "description") in prompt
    lifestyle = archetypes.get(policy, "lifestyle_slow")["communicates"]
    assert lifestyle in prompt                                                       # how this creator communicates
    assert "suits" in prompt and "outfit_change_spin" in prompt                      # recipes, tagged by archetype
    assert "CAPABILITIES" in showrunner.SYSTEM and "CAPABILITIES" in planner.system_prompt(policy)


def test_the_planner_gets_the_capability_brief(conn, policy):
    cid = creator(conn)
    stub = PlanStub([post(0)])
    planner.plan_period(conn, policy, stub, cid, start=START)
    assert "What the studio can render now" in stub.calls[0]


def test_the_showrunners_system_prompt_names_recurring_sets_for_the_director():
    assert "director" in showrunner.SYSTEM and "on_screen_text" in showrunner.SYSTEM
    assert "set_id" in showrunner.SYSTEM and "outfit_id" in showrunner.SYSTEM and "Hands" in showrunner.SYSTEM


def test_scheduled_ideas_carry_their_shots_and_move_into_the_post(conn, policy):
    cid = bare_creator(conn)
    conn.execute("UPDATE creators SET profile_version = 1, status = 'active', anchors = %s WHERE id = %s",
                 (Jsonb({"reference_images": [{"uri": "s3://b/r"}]}), cid))
    base = {"caption": "c", "platform": "instagram", "hashtags": [], "canon": "", "pillar": "food", "tone": "warm",
            "series": "", "script": "", "on_screen_text": "", "reasoning": "r"}
    for day, extra in ((0, {"scenes": ["her galley kitchen, medium shot"], "shots": [KITCHEN], "move": "b_roll_detail"}),
                       (1, {"scenes": ["at the market"]})):
        iid = conn.execute("INSERT INTO ideas (creator_id, batch, concept, format, body, status) "
                           "VALUES (%s, %s, 'x', 'photo', %s, 'proposed') RETURNING id",
                           (cid, f"{cid}:{START}", Jsonb({**base, "day": day, **extra}))).fetchone()["id"]
        conn.execute("INSERT INTO evaluations (item_kind, idea_id, creator_id, rubric, rubric_version, evaluator_model, "
                     "prompt_version, scores, samples, role) VALUES ('idea', %s, %s, 'idea', 1, 'm', 'p', '{}', 3, 'evaluator')",
                     (iid, cid))
    assert ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc))["scheduled"] == 2
    directed, plain = [r["body"] for r in conn.execute("SELECT body FROM posts ORDER BY scheduled_for").fetchall()]
    assert directed["shots"] == [KITCHEN] and directed["move"] == "b_roll_detail"
    assert "shots" not in plain and "move" not in plain
