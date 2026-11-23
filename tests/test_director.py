"""The director (manyworlds.brain.director): an accepted idea's scenes become shot sheets when it is scheduled, a
move's shot defaults fill what the director left empty, the voice rules hold, and a failed call never drops the
post: it proceeds undirected."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest
from psycopg.types.json import Jsonb

from manyworlds.brain import archetypes, director, ideas

from brain_helpers import StubLLM, creator as bare_creator
from test_production import START, creator


class DirectorLLM(StubLLM):
    """Answers the director's call with `answer` (validated into the asked schema), or raises `error`."""

    def __init__(self, answer=None, error: Exception | None = None, conn=None):
        super().__init__(conn=conn)
        self.answer, self.error = answer, error

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema)
        self.calls[-1]["system"] = system
        if self.error is not None:
            raise self.error
        if schema.__name__ == "DirectedEdits":   # the edit pass: each answered shot's edit, as the director wrote it
            return schema.model_validate({"edits": [sh.get("edit") or {} for sh in self.answer.get("shots") or []]})
        return schema.model_validate(self.answer)


CANAL = {"set": {"location": "the canal path at dusk", "set_id": "canal_path"},
         "wardrobe": {"outfit": "a long camel coat", "outfit_id": "camel_coat"}}


def test_direct_returns_shot_dicts_and_merges_the_moves_defaults(conn, policy):
    cid = creator(conn)
    llm = DirectorLLM({"shots": [CANAL], "move_shot": 0}, conn=conn)
    body = {"scenes": ["walking along the canal at dusk"], "script": "she walks toward the camera",
            "move": "walk_to_camera"}
    shots = director.direct(conn, policy, llm, body, cid, fmt="story_clip")
    assert len(shots) == 1 and isinstance(shots[0], dict)
    shot = shots[0]
    assert shot["set"]["location"] == "the canal path at dusk"                 # what the director set wins
    move = policy.get("moves", "library", "walk_to_camera", "shot")
    assert shot["framing"]["shot_size"] == move["framing"]["shot_size"]        # the move fills what was empty
    assert shot["framing"]["device"] == "propped" and shot["framing"]["camera_move"] == "static"
    assert [b["action"] for b in shot["blocking"]] == [b["action"] for b in move["blocking"]]
    call = llm.calls[0]
    assert call["role"] == "director" and call["agent"] == "director"
    assert "walk_to_camera" in str(call["content"]) and "orbit_slow" in str(call["content"])   # move and camera moves
    assert "walking along the canal" in str(call["content"])
    cost = conn.execute("SELECT count(*) n FROM cost_entries WHERE agent = 'director'").fetchone()["n"]
    assert cost == 1                                                           # charged to the director


def test_a_move_lands_only_on_the_shot_that_performs_it(conn, policy):
    cid = creator(conn)
    talk = {"set": {"location": "her sofa"}, "framing": {"shot_size": "medium close-up"}}
    llm = DirectorLLM({"shots": [talk, {"set": {"location": "the mug on the table"}}], "move_shot": 1}, conn=conn)
    body = {"scenes": ["on the sofa", "the mug"], "script": "", "move": "b_roll_detail"}
    shots = director.direct(conn, policy, llm, body, cid, fmt="carousel")
    assert shots[0]["cutaway"] is False and shots[1]["cutaway"] is True
    assert shots[1]["framing"]["camera_move"] == "push_in" and shots[0]["framing"]["camera_move"] == ""


def test_one_shot_per_scene(conn, policy):
    cid = creator(conn)
    llm = DirectorLLM({"shots": [CANAL, CANAL, CANAL]}, conn=conn)
    shots = director.direct(conn, policy, llm, {"scenes": ["a", "b"]}, cid, fmt="carousel")
    assert len(shots) == 2                                                     # extra shots dropped
    llm = DirectorLLM({"shots": [CANAL]}, conn=conn)
    shots = director.direct(conn, policy, llm, {"scenes": ["a", "b"]}, cid, fmt="carousel")
    assert len(shots) == 2 and shots[1]["set"]["location"] == ""               # a missing shot is an empty sheet
    with pytest.raises(ValueError):
        director.direct(conn, policy, DirectorLLM({"shots": []}, conn=conn), {"scenes": ["a"]}, cid, fmt="photo")


def test_shots_that_break_the_voice_rules_are_refused(conn, policy):
    cid = creator(conn)
    salesy = {**CANAL, "edit": {"on_screen_text": "drop a comment if you agree"}}
    with pytest.raises(ValueError, match="read as AI"):
        director.direct(conn, policy, DirectorLLM({"shots": [salesy]}, conn=conn), {"scenes": ["a"]}, cid,
                        fmt="photo")


def _says(*texts, **kw):
    return {**kw, "performance": {"lines": [{"text": t, "delivery": "dry", "pause_after_s": 0} for t in texts]}}


def test_a_talking_videos_lines_must_be_its_script(conn, policy):
    cid = creator(conn)
    body = {"scenes": ["outside the shop", "the doorway"], "script": "Right, quick one. The samosas are back!"}
    kept = director.direct(conn, policy, DirectorLLM({"shots": [_says("Right, quick one."),
                                                                _says("The samosas are back!")]}, conn=conn),
                           body, cid, fmt="talking_video")
    assert [line["text"] for s in kept for line in s["performance"]["lines"]] == ["Right, quick one.",
                                                                                  "The samosas are back!"]
    assert kept[0]["performance"]["lines"][0]["pause_after_s"] is None       # 0 means the default pause
    reworded = director.direct(conn, policy, DirectorLLM({"shots": [_says("Okay so, quick one."),
                                                                    _says("Samosas are back.")]}, conn=conn),
                               body, cid, fmt="talking_video")
    assert all(s["performance"]["lines"] == [] for s in reworded)            # production splits the script instead
    assert reworded[0]["set"] == kept[0]["set"]


def _ready_ideas(conn, cid, *items):
    base = {"caption": "c", "platform": "instagram", "hashtags": [], "canon": "", "pillar": "food", "tone": "warm",
            "series": "", "script": "", "on_screen_text": "", "reasoning": "r"}
    for day, (fmt, extra) in enumerate(items):
        iid = conn.execute("INSERT INTO ideas (creator_id, batch, concept, format, body, status) "
                           "VALUES (%s, %s, 'x', %s, %s, 'proposed') RETURNING id",
                           (cid, f"{cid}:{START}", fmt, Jsonb({**base, "day": day, **extra}))).fetchone()["id"]
        conn.execute("INSERT INTO evaluations (item_kind, idea_id, creator_id, rubric, rubric_version, evaluator_model, "
                     "prompt_version, scores, samples, role) VALUES ('idea', %s, %s, 'idea', 1, 'm', 'p', '{}', 3, 'evaluator')",
                     (iid, cid))


def _real_creator(conn):
    cid = bare_creator(conn)
    conn.execute("UPDATE creators SET profile_version = 1, status = 'active', anchors = %s WHERE id = %s",
                 (Jsonb({"reference_images": [{"uri": "s3://b/r"}]}), cid))
    return cid


def test_scheduling_directs_new_posts(conn, policy):
    cid = _real_creator(conn)
    _ready_ideas(conn, cid, ("photo", {"scenes": ["walking along the canal at dusk"]}),
                 ("story", {"scenes": ["the view from the roof"]}))                  # story: not directed
    llm = DirectorLLM({"shots": [CANAL]}, conn=conn)
    out = ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc), llm=llm)
    assert out["scheduled"] == 2 and out["directed"] == 1 and len(llm.calls) == 2   # shots, then their edit
    photo, story = [r["body"] for r in conn.execute("SELECT body FROM posts ORDER BY scheduled_for").fetchall()]
    assert photo["shots"][0]["set"]["set_id"] == "canal_path" and photo["scenes"] == ["walking along the canal at dusk"]
    assert photo["platform_ai_label"] is True and "shots" not in story


def test_a_set_first_named_by_a_scheduled_post_records_that_post(conn, policy):
    cid = _real_creator(conn)
    _ready_ideas(conn, cid, ("photo", {"scenes": ["walking along the canal at dusk"]}))
    ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc), llm=DirectorLLM({"shots": [CANAL]}, conn=conn))
    post_id = conn.execute("SELECT id FROM posts").fetchone()["id"]
    from manyworlds.creator import sets
    b = sets.bibles(conn, cid)
    assert b["sets"]["canal_path"]["created_from_post"] == str(post_id)
    assert b["outfits"]["camel_coat"]["created_from_post"] == str(post_id)


def test_a_failed_director_call_leaves_the_post_undirected(conn, policy):
    cid = _real_creator(conn)
    _ready_ideas(conn, cid, ("photo", {"scenes": ["walking along the canal at dusk"]}))
    out = ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc),
                               llm=DirectorLLM(error=RuntimeError("grammar too large"), conn=conn))
    assert out["scheduled"] == 1 and out["directed"] == 0
    post = conn.execute("SELECT status, body FROM posts").fetchone()
    assert post["status"] == "planned" and "shots" not in post["body"]
    assert conn.execute("SELECT status FROM ideas").fetchone()["status"] == "scheduled"


def _fashion(conn):
    cid = creator(conn)
    conn.execute("UPDATE creators SET bible = bible || %s WHERE id = %s",
                 (Jsonb({"archetype": "quiet-luxury fashion in a small city flat", "niche": "fashion",
                         "one_line": "outfits that cost less than they look"}), cid))
    return cid


def _clips_only(policy):
    """Photos and story clips made, no talking video, dance off."""
    return dataclasses.replace(policy, body={
        **policy.body, "production": {**policy.body["production"], "formats": ["photo", "carousel", "story_clip"]},
        "dance": {**policy.body["dance"], "enabled": False}})


def test_the_brief_is_the_capability_catalogue_and_the_archetype(conn, policy):
    policy = _clips_only(policy)
    cid = _fashion(conn)
    llm = DirectorLLM({"shots": [CANAL]}, conn=conn)
    director.direct(conn, policy, llm, {"scenes": ["by the canal"], "script": ""}, cid, fmt="story_clip")
    content, system = str(llm.calls[0]["content"]), llm.calls[0]["system"]
    assert "quiet-luxury fashion" in content and "outfits that cost less" in content          # the archetype
    fashion = archetypes.get(policy, "fashion_style")["communicates"]
    assert fashion in content                                                                  # how it communicates
    cat = policy.get("capabilities")
    assert cat["devices"]["mirror"]["description"] in content and cat["actions"]["pose"]["write"] in content
    assert "actions.talk" in content.split("Not available now")[1]                 # named as unavailable
    assert cat["actions"]["talk"]["description"] not in content.split("Not available now")[0]
    assert "silent_aesthetic" in content and "pov_day_timestamps" not in content   # recipes for this format only
    for name in list(policy.get("on_screen_text", "styles")) + list(policy.get("burned_captions", "styles")):
        assert name in content
    assert "rain_glass" in content and "match" in content                          # room tone words, transitions
    assert "Motion references: none now" in content                                # dance is off
    for rule in policy.get("direction", "director", "craft"):
        assert rule in system
    assert "a gimbal-smooth phone" in system and "with no people" not in system


def test_without_a_matching_archetype_the_default_guidance_is_given(policy):
    assert director.communication(policy, {"archetype": "retired lighthouse keeper"}) == [
        archetypes.library(policy)["default_communicates"]]
    both = director.communication(policy, {"niche": "beauty", "one_line": "storytelling while doing her makeup"})
    assert len(both) == 2


def test_a_motion_reference_is_kept_only_when_listed_and_rendered(conn, policy):
    cid = creator(conn)
    ref = policy.get("dance", "references")[0]["id"]
    on = dataclasses.replace(policy, body={**policy.body, "dance": {**policy.body["dance"], "enabled": True}})
    shots = director.direct(conn, on, DirectorLLM({"shots": [{**CANAL, "motion_reference": ref},
                                                             {**CANAL, "motion_reference": "a-trend-video"}]},
                                                  conn=conn), {"scenes": ["a", "b"]}, cid, fmt="carousel")
    assert [s["motion_reference"] for s in shots] == [ref, ""]
    llm = DirectorLLM({"shots": [{**CANAL, "motion_reference": ref}]}, conn=conn)
    shots = director.direct(conn, _clips_only(policy), llm, {"scenes": ["a"]}, cid, fmt="photo")
    assert shots[0]["motion_reference"] == ""                     # dance off: cleared
    assert ref in str(_content(conn, on, cid))


def _content(conn, policy, cid):
    llm = DirectorLLM({"shots": [CANAL]}, conn=conn)
    director.direct(conn, policy, llm, {"scenes": ["a"]}, cid, fmt="photo")
    return llm.calls[0]["content"]


def test_overlay_items_keep_the_voice_rules(conn, policy):
    cid = creator(conn)
    salesy = {**CANAL, "edit": {"on_screen_text": [{"kind": "label", "text": "fine"},
                                                    {"kind": "question", "text": "drop a comment below"}]}}
    with pytest.raises(ValueError, match="read as AI"):
        director.direct(conn, policy, DirectorLLM({"shots": [salesy]}, conn=conn), {"scenes": ["a"]}, cid,
                        fmt="story_clip")


def test_direction_can_be_switched_off(conn, policy):
    cid = _real_creator(conn)
    _ready_ideas(conn, cid, ("photo", {"scenes": ["walking along the canal at dusk"]}))
    off = dataclasses.replace(policy, body={**policy.body, "direction": {**policy.body["direction"], "enabled": False}})
    llm = DirectorLLM({"shots": [CANAL]}, conn=conn)
    assert ideas.schedule_ready(conn, off, now=datetime.now(timezone.utc), llm=llm)["directed"] == 0
    assert llm.calls == []
