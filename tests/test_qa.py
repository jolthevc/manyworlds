"""QA and repair (Brain M7): a rejected image gets the repair its review points to, not just a new seed;
a video that keeps failing still ships as the photo it was made from; and the feed is checked before
ideas become posts, so a grid doesn't repeat an outfit, reuse a caption or sound unlike the creator."""

from __future__ import annotations

from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from manyworlds import production
from manyworlds.brain import ideas
from manyworlds.review import video

from test_brain_schedule import START, _creator
from test_production import NOW, PlanStub, creator, finish, post, real_refs
from test_production import START as PROD_START
from test_review import JudgeStub, ok
from test_video_review import ClipStub, clip, rendered


# --- feed QA at scheduling ------------------------------------------------------------------------

def _idea(conn, cid, day, *, caption="a quiet coffee before work", scenes=("at the cafe window",), attributes=None,
          scores=None, experiment=None):
    body = {"scenes": list(scenes), "caption": caption, "day": day, "platform": "instagram", "hashtags": [],
            "canon": "", "pillar": "", "tone": "", "series": "", "script": "", "on_screen_text": "", "reasoning": "r"}
    iid = conn.execute("INSERT INTO ideas (creator_id, batch, concept, format, body, attributes, status, experiment_id, arm) "
                       "VALUES (%s, %s, 'x', 'photo', %s, %s, 'proposed', %s, %s) RETURNING id",
                       (cid, f"{cid}:{START}", Jsonb(body), Jsonb(attributes or {}), experiment,
                        "A" if experiment else None)).fetchone()["id"]
    conn.execute("INSERT INTO evaluations (item_kind, idea_id, creator_id, rubric, rubric_version, evaluator_model, "
                 "prompt_version, scores, samples, role) VALUES ('idea', %s, %s, 'idea', 1, 'm', 'p', %s, 3, 'evaluator')",
                 (iid, cid, Jsonb(scores or {"naturalness": 7, "personality": 6})))
    return iid


def _recent_post(conn, cid, *, caption, outfit=None):
    pid = conn.execute("INSERT INTO posts (creator_id, platform, format, status, caption, body, scheduled_for) "
                       "VALUES (%s, 'instagram', 'photo', 'published', %s, %s, now()) RETURNING id",
                       (cid, caption, Jsonb({"scenes": ["somewhere"]}))).fetchone()["id"]
    conn.execute("INSERT INTO ideas (creator_id, concept, format, body, attributes, status, post_id) "
                 "VALUES (%s, 'x', 'photo', '{}', %s, 'scheduled', %s)",
                 (cid, Jsonb({"outfit": outfit} if outfit else {}), pid))
    return pid


def _status(conn, iid):
    return conn.execute("SELECT status, body->>'drop_reason' AS why FROM ideas WHERE id = %s", (iid,)).fetchone()


def test_a_repeated_outfit_is_kept_out_of_the_grid(conn, policy):
    cid = _creator(conn)
    limit = int(policy.get("brain", "qa", "max_repeat"))
    for i in range(limit):
        _recent_post(conn, cid, caption=f"post {i} about something different", outfit="green cardigan")
    again = _idea(conn, cid, 0, attributes={"outfit": "Green cardigan"})
    fresh = _idea(conn, cid, 1, caption="new boots, finally", attributes={"outfit": "black coat"})
    out = ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc))
    assert out["scheduled"] == 1 and _status(conn, fresh)["status"] == "scheduled"
    s = _status(conn, again)
    assert s["status"] == "dropped" and "repetitive feed" in s["why"] and "green cardigan" in s["why"]


def test_a_near_duplicate_caption_is_dropped(conn, policy):
    cid = _creator(conn)
    _recent_post(conn, cid, caption="rain again. the kettle knows what to do")
    dup = _idea(conn, cid, 0, caption="Rain again... the kettle knows what to do")
    ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc))
    s = _status(conn, dup)
    assert s["status"] == "dropped" and "unoriginal" in s["why"]


def test_an_idea_that_does_not_sound_like_them_is_dropped(conn, policy):
    cid = _creator(conn)
    floor = policy.get("brain", "qa", "min_scores")["naturalness"]
    off = _idea(conn, cid, 0, scores={"naturalness": floor - 1, "personality": 7})
    ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc))
    s = _status(conn, off)
    assert s["status"] == "dropped" and "naturalness" in s["why"]


def test_experiment_ideas_are_not_screened_out_so_arms_stay_comparable(conn, policy):
    cid = _creator(conn)
    exp = conn.execute("INSERT INTO experiments (creator_id, hypothesis, status) VALUES (%s, 'h', 'open') RETURNING id",
                       (cid,)).fetchone()["id"]
    _recent_post(conn, cid, caption="rain again. the kettle knows what to do")
    arm = _idea(conn, cid, 0, caption="rain again. the kettle knows what to do", experiment=exp)
    ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc))
    assert _status(conn, arm)["status"] == "scheduled"


# --- targeted repair ------------------------------------------------------------------------------

def _photo_post(conn, policy, store, tmp_path, lora=True):
    cid = creator(conn, lora=lora)
    real_refs(conn, store, tmp_path, cid)
    from manyworlds.brain import planner
    planner.plan_period(conn, policy, PlanStub([post(0, "photo", ["reading on a balcony at dawn"])]), cid,
                         start=PROD_START)
    pid = conn.execute("SELECT id FROM posts WHERE creator_id = %s", (cid,)).fetchone()["id"]
    production.advance(conn, policy, now=NOW)
    return pid


def _jobs(conn, pid):
    return conn.execute("SELECT payload FROM jobs WHERE post_id = %s AND kind = 'image.zimage' ORDER BY created_at",
                        (pid,)).fetchall()


def test_an_identity_failure_strengthens_the_likeness(conn, policy, store, tmp_path):
    pid = _photo_post(conn, policy, store, tmp_path)
    finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok(identity=2)] * 4), pid)
    production.advance(conn, policy, now=NOW)
    first, second = _jobs(conn, pid)
    step = float(policy.get("review", "repair", "lora_scale_step"))
    assert abs(second["payload"]["lora_scale"] - (first["payload"]["lora_scale"] + step)) < 1e-9
    log = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]["repair_log"]
    assert log[0]["cause"] == "identity" and "lora_scale" in log[0]["action"]


def test_a_wrong_scene_puts_the_scene_first(conn, policy, store, tmp_path):
    pid = _photo_post(conn, policy, store, tmp_path)
    finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok(scene=1)] * 4, confirm=[ok(scene=1)] * 4), pid)
    # scene fit only ranks images (min_scene 0), so force a rejection the way the gate reports a miss
    conn.execute("UPDATE reviews SET passed = false WHERE gate = 'scene'")
    production.advance(conn, policy, now=NOW)
    second = _jobs(conn, pid)[1]["payload"]
    assert second["prompt"].startswith(policy.get("review", "repair", "hints", "scene").format(
        scene="reading on a balcony at dawn"))


def test_unsafe_output_is_never_regenerated(conn, policy, store, tmp_path):
    pid = _photo_post(conn, policy, store, tmp_path)
    finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok(sexualized=True)] * 4), pid)
    assert production.advance(conn, policy, now=NOW)["dropped"] == 1
    assert len(_jobs(conn, pid)) == 1
    assert "safety" in conn.execute("SELECT body->>'drop_reason' r FROM posts WHERE id = %s", (pid,)).fetchone()["r"]


# --- format reroute -------------------------------------------------------------------------------

def test_a_story_clip_whose_clips_keep_failing_ships_as_its_photo(conn, policy, store, tmp_path):
    from test_video_review import _clip_jobs, _story_post
    pid = _story_post(conn, policy, store, tmp_path)
    for _ in range(int(policy.get("review", "video", "max_repairs_per_post")) + 1):
        job = _clip_jobs(conn, pid)[-1]
        rendered(conn, store, tmp_path, job["id"])
        video.run_pending(conn, policy, ClipStub(clip(identity=2)), store, tmp_path)
        state = production.advance(conn, policy, now=NOW)
    assert state["ready"] == 1
    row = conn.execute("SELECT format, status, body, asset_ids FROM posts WHERE id = %s", (pid,)).fetchone()
    assert row["format"] == "photo" and row["body"]["rerouted_from"] == "story_clip" and "video" not in row["body"]
    assert len(row["asset_ids"]) == 1


def test_a_skipped_review_or_a_stray_logo_is_not_treated_as_unsafe(conn, policy, store, tmp_path):
    pid = _photo_post(conn, policy, store, tmp_path)
    finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok(real_brand_marks=True)] * 4), pid)
    assert production.advance(conn, policy, now=NOW)["producing"] == 1          # repaired, not dropped
    log = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]["repair_log"]
    assert log[0]["cause"] == "realism"


def test_a_repair_survives_a_machine_failure(conn, policy, store, tmp_path):
    pid = _photo_post(conn, policy, store, tmp_path)
    finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok(identity=2)] * 4), pid)
    production.advance(conn, policy, now=NOW)                                   # identity repair queued
    conn.execute("UPDATE jobs SET status = 'failed' WHERE post_id = %s AND status = 'queued'", (pid,))
    production.advance(conn, policy, now=NOW)                                   # the machine failed: retry
    third = _jobs(conn, pid)[2]["payload"]
    hint = policy.get("review", "repair", "hints", "identity")
    assert hint in third["prompt"]


def test_an_experiment_video_is_not_rerouted_to_a_photo(conn, policy, store, tmp_path):
    from test_video_review import _clip_jobs, _story_post
    pid = _story_post(conn, policy, store, tmp_path)
    exp = conn.execute("INSERT INTO experiments (creator_id, hypothesis, status) SELECT creator_id, 'h', 'open' "
                       "FROM posts WHERE id = %s RETURNING id", (pid,)).fetchone()["id"]
    conn.execute("INSERT INTO ideas (creator_id, concept, format, status, post_id, experiment_id, arm) "
                 "SELECT creator_id, 'x', 'story_clip', 'scheduled', id, %s, 'A' FROM posts WHERE id = %s", (exp, pid))
    for _ in range(int(policy.get("review", "video", "max_repairs_per_post")) + 1):
        job = _clip_jobs(conn, pid)[-1]
        rendered(conn, store, tmp_path, job["id"])
        video.run_pending(conn, policy, ClipStub(clip(identity=2)), store, tmp_path)
        state = production.advance(conn, policy, now=NOW)
    assert state["dropped"] == 1


# --- the rerouted post's caption --------------------------------------------------------------------------------

class CaptionStub:
    """Answers the caption_rewrite role only; anything else would be a call this flow must not make."""

    def __init__(self, caption="tea by the rainy window, just this", fail=False):
        self.caption, self.fail, self.calls = caption, fail, []

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        self.calls.append({"role": role, "system": system, "content": content, "effort": effort})
        assert role == "caption_rewrite"
        if self.fail:
            raise RuntimeError("model unavailable")
        return schema(caption=self.caption)


def _rerouted(conn, policy, store, tmp_path, llm, caption):
    from test_video_review import _clip_jobs, _story_post
    pid = _story_post(conn, policy, store, tmp_path)
    conn.execute("UPDATE posts SET caption = %s WHERE id = %s", (caption, pid))
    for _ in range(int(policy.get("review", "video", "max_repairs_per_post")) + 1):
        job = _clip_jobs(conn, pid)[-1]
        rendered(conn, store, tmp_path, job["id"])
        video.run_pending(conn, policy, ClipStub(clip(identity=2)), store, tmp_path)
        state = production.advance(conn, policy, now=NOW, llm=llm)
    assert state["ready"] == 1
    return conn.execute("SELECT format, caption, body FROM posts WHERE id = %s", (pid,)).fetchone()


def test_a_rerouted_post_gets_its_caption_rewritten_for_the_photo(conn, policy, store, tmp_path):
    disclosure = policy.get("scope", "disclosure_line")
    llm = CaptionStub(caption="tea by the rainy window, just this")      # the model left the disclosure out
    old = f"watch the steam rise, sound on 🎧 · {disclosure}"
    row = _rerouted(conn, policy, store, tmp_path, llm, old)
    assert row["format"] == "photo" and len(llm.calls) == 1
    call = llm.calls[0]
    assert "story_clip" in call["system"] and "photo" in call["system"] and disclosure in call["system"]
    assert old in call["content"] and call["effort"] == policy.get("production", "reroute_caption", "effort")
    assert row["body"]["caption_before_reroute"] == old
    assert row["caption"].startswith("tea by the rainy window") and disclosure in row["caption"]   # kept by code


def test_a_failed_caption_rewrite_keeps_the_old_caption_and_ships(conn, policy, store, tmp_path):
    row = _rerouted(conn, policy, store, tmp_path, CaptionStub(fail=True), "watch the steam rise")
    assert row["format"] == "photo" and row["caption"] == "watch the steam rise"
    assert "caption_before_reroute" not in row["body"] and "model unavailable" in row["body"]["reroute_caption_error"]


def test_the_caption_rewrite_is_skipped_without_a_model_or_a_caption(policy):
    post = {"id": "p", "format": "story_clip", "caption": "watch this", "creator_id": None}
    body: dict = {}
    assert production.reroute_caption(policy, None, post, body, "photo") == "watch this" and body == {}
    llm = CaptionStub()
    assert production.reroute_caption(policy, llm, {**post, "caption": ""}, body, "photo") == "" and not llm.calls
