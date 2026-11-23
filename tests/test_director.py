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


def text(content) -> str:
    """A call's content as plain text: the director sends blocks (the cached static brief, then the creator's part)."""
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))


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
    assert "walk_to_camera" in text(call["content"]) and "orbit_slow" in text(call["content"])   # move and camera moves
    assert "walking along the canal" in text(call["content"])
    cost = conn.execute("SELECT count(*) n FROM cost_entries WHERE agent = 'director'").fetchone()["n"]
    assert cost == 2                                         # both calls (shots, then their edit) charged to it


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
    [shot] = director.direct(conn, policy, DirectorLLM({"shots": [salesy]}, conn=conn), {"scenes": ["a"]}, cid,
                             fmt="photo")
    assert "comment" not in str(shot["edit"]) and shot["set"]["location"]       # edit cleared, shot kept


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
    content, system = text(llm.calls[0]["content"]), llm.calls[0]["system"]
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
    assert ref in text(_content(conn, on, cid))


def _content(conn, policy, cid):
    llm = DirectorLLM({"shots": [CANAL]}, conn=conn)
    director.direct(conn, policy, llm, {"scenes": ["a"]}, cid, fmt="photo")
    return llm.calls[0]["content"]


def test_overlay_items_keep_the_voice_rules(conn, policy):
    cid = creator(conn)
    salesy = {**CANAL, "edit": {"on_screen_text": [{"kind": "label", "text": "fine"},
                                                    {"kind": "question", "text": "drop a comment below"}]}}
    [shot] = director.direct(conn, policy, DirectorLLM({"shots": [salesy]}, conn=conn), {"scenes": ["a"]}, cid,
                             fmt="story_clip")
    assert "comment" not in str(shot["edit"])                                  # the overlay went with the edit


def test_direction_can_be_switched_off(conn, policy):
    cid = _real_creator(conn)
    _ready_ideas(conn, cid, ("photo", {"scenes": ["walking along the canal at dusk"]}))
    off = dataclasses.replace(policy, body={**policy.body, "direction": {**policy.body["direction"], "enabled": False}})
    llm = DirectorLLM({"shots": [CANAL]}, conn=conn)
    assert ideas.schedule_ready(conn, off, now=datetime.now(timezone.utc), llm=llm)["directed"] == 0
    assert llm.calls == []


# --- more context, the cached brief, retries (2026-11-23) ---------------------------------------------------------

def _profiled(conn, policy):
    """A creator with a profile, a dossier, a talent brief and directed posts behind her."""
    from manyworlds.creator import profile
    cid = creator(conn)
    profile.save(conn, policy, cid, {
        "identity": {"apparent_age": 27},
        "look": {"hair": "bob", "wardrobe": "knits", "accessories": "watch", "makeup": "none",
                 "signature_item": "glasses"},
        "persona": {"traits": ["dry"], "vibe": "quiet", "ethos": "e", "humour": "deadpan about kettles",
                    "caption_voice": "lowercase, one dry line", "speaking_voice": "low", "never": ["gym selfies"]},
        "content": {"pillars": [{"name": "night rituals", "about": "a", "why_it_lasts": "b"}],
                    "video_series": [{"name": "kettle ratings", "premise": "she rates kettles at 2am",
                                      "format": "story_clip", "what_it_says": "w", "pillar": "night rituals"}]},
        "archetypes": ["lifestyle_slow"],
        "positioning": {"why_follow": "a calm night routine after late shifts"},
        "content_system": {"series": [{"name": "2am tea", "premise": "the same ritual each night",
                                       "hook": "a time stamp"}]},
        "opinions_and_running_jokes": ["the kettle is always slow"], "dials": {}}, author="test", reason="seed")
    conn.execute("INSERT INTO decisions (kind, creator_id, inputs, output, policy_version) "
                 "VALUES ('talent_brief', %s, '{}', %s, %s)",
                 (cid, Jsonb({"focus": ["more steam close-ups"], "stop": ["talking heads in the hallway"]}),
                  policy.version))
    conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, 'carousel', 'planned', %s)",
                 (cid, Jsonb({"scenes": ["a", "b"], "shots": [{"shot_type": "hand_coffee_table"},
                                                              {"shot_type": "hand_coffee_table"}]})))
    return cid


def test_recent_shot_types_reads_the_directed_posts_newest_first(conn, policy):
    cid = _profiled(conn, policy)
    assert director.recent_shot_types(conn, cid, 5) == ["hand_coffee_table", "hand_coffee_table"]


def test_the_director_sees_the_idea_her_persona_dossier_brief_playbook_and_recent_shots(conn, policy):
    cid = _profiled(conn, policy)
    iid = conn.execute("INSERT INTO ideas (creator_id, concept, format, body, attributes, status) "
                       "VALUES (%s, 'x', 'story_clip', %s, %s, 'scheduled') RETURNING id",
                       (cid, Jsonb({"reasoning": "her saves spike on steam shots"}),
                        Jsonb({"mood": "sleepy"}))).fetchone()["id"]
    body = {"scenes": ["steam off the kettle at 2am"], "script": "steam rises", "series": "kettle ratings",
            "pillar": "night rituals", "tone": "dry", "canon": "her kettle is green", "idea_id": str(iid)}
    answer = {"shots": [{**CANAL, "shot_type": "hand_coffee_table", "sound": {"room_tone": "fridge hum"}}]}
    llm = DirectorLLM(answer, conn=conn)
    director.direct(conn, policy, llm, body, cid, fmt="story_clip")
    content = text(llm.calls[0]["content"])
    assert "her saves spike on steam shots" in content and "sleepy" in content        # the idea's why
    assert "she rates kettles at 2am" in content and "her kettle is green" in content  # its series and canon
    assert "lowercase, one dry line" in content and "the kettle is always slow" in content   # her persona
    assert "a calm night routine after late shifts" in content                         # dossier planning view
    assert "the same ritual each night" in content
    assert "more steam close-ups" in content and "talking heads in the hallway" in content   # talent brief
    play = archetypes.get(policy, "lifestyle_slow")
    assert play["edit_grammar"][0] in content and play["pitfalls"][0] in content      # her archetype's craft
    assert '"hand_coffee_table": 2' in content                                         # recent shot types
    edit = text(llm.calls[1]["content"])                                               # the edit pass sees the shots
    assert "the canal path at dusk" in edit.split(director.EDIT_ASK)[1]
    assert "hand_coffee_table" in edit.split(director.EDIT_ASK)[1] and "fridge hum" in edit.split(director.EDIT_ASK)[1]
    assert "EVERY shot names one" in llm.calls[0]["system"]                            # never an empty shot type


def test_the_static_brief_leads_as_a_cached_block_shared_by_both_calls(conn, policy):
    cid = _profiled(conn, policy)
    llm = DirectorLLM({"shots": [CANAL]}, conn=conn)
    director.direct(conn, policy, llm, {"scenes": ["by the canal"]}, cid, fmt="story_clip")
    shots_call, edit_call = llm.calls[0]["content"], llm.calls[1]["content"]
    static = shots_call[0]
    assert static["cache_control"] == {"type": "ephemeral"}
    assert static["text"] == director.static_brief(policy, "story_clip")
    assert "What the studio can render now" in static["text"] and "Allowed camera moves" in static["text"]
    assert "lowercase, one dry line" not in static["text"] and "by the canal" not in static["text"]   # nothing of hers
    assert edit_call[0] == static and edit_call[1] == shots_call[1]                # the edit pass reuses the prefix


def test_a_post_the_director_failed_on_is_directed_again_later_within_its_attempts(conn, policy):
    cid = _real_creator(conn)
    _ready_ideas(conn, cid, ("photo", {"scenes": ["walking along the canal at dusk"]}))
    ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc),
                         llm=DirectorLLM(error=RuntimeError("timed out"), conn=conn))
    post = conn.execute("SELECT id, body FROM posts").fetchone()
    assert post["body"]["direct_attempts"] == 1 and "shots" not in post["body"]
    llm = DirectorLLM({"shots": [CANAL]}, conn=conn)
    assert ideas.redirect_undirected(conn, policy, llm) == 0 and llm.calls == []      # too soon after the failure
    conn.execute("UPDATE posts SET body = body || %s WHERE id = %s",
                 (Jsonb({"direct_failed_at": "2020-01-01T00:00:00+00:00"}), post["id"]))
    assert ideas.redirect_undirected(conn, policy, llm) == 1
    assert conn.execute("SELECT body FROM posts WHERE id = %s",
                        (post["id"],)).fetchone()["body"]["shots"][0]["set"]["set_id"] == "canal_path"
    # Out of attempts: never asked again.
    spent = policy.get("direction", "redirect", "max_attempts")
    conn.execute("INSERT INTO posts (creator_id, format, status, caption, body) VALUES (%s, 'photo', 'planned', 'c', %s)",
                 (cid, Jsonb({"scenes": ["a bakery window"], "idea_id": str(post["body"]["idea_id"]),
                              "direct_attempts": spent, "direct_failed_at": "2020-01-01T00:00:00+00:00"})))
    llm = DirectorLLM({"shots": [CANAL]}, conn=conn)
    assert ideas.redirect_undirected(conn, policy, llm) == 0 and llm.calls == []
