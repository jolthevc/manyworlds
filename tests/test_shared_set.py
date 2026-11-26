"""One place per shared set (and, 2026-11-25 maren-final-3, its empty plate: followers are drawn into the leader's
place with nobody in it, never into the leader's photo itself) (perf_compare 2026-11-25: nine shots named pub_bar; one photo was at the bar, eight in a
living room or kitchen). Three causes, three fixes: the photo prompt lost its Set and Wardrobe sections whole to her
long signature item (direction._fit), nothing tied the shots' photos to one place (production.scenes: the first
face shot leads, the others are conditioned on its accepted photo), and scene fit only scores (gate set_match)."""

from __future__ import annotations

import uuid

from psycopg.types.json import Jsonb

from manyworlds import direction, production
from manyworlds.creator import sets
from manyworlds.review import gates

from test_production import creator
from test_review import BIBLE, JudgeStub, creator_with_job, ok

BAR = {"location": "behind the bar of a small, dim neighbourhood pub at closing time", "set_id": "pub_bar",
       "time_of_day": "night", "light_quality": "warm low bar lights, a neon sign behind her", "color_temp": "warm"}
TEE = {"outfit_id": "bar_black_tee", "outfit": "a plain black t-shirt, a dark apron tied at the waist"}
CARDIGAN = ("Her oversized moss-green cardigan. It is a dark, slightly faded moss green in a chunky fisherman-rib knit "
            "and hangs to mid-thigh. It has a shawl collar and five round brown wooden buttons; the second from the "
            "top is a mismatched darker button she sewed on herself. The left patch pocket is visibly stretched and "
            "sagging from years of carrying a notebook and pen. The sleeves fall past her wrists, so the ribbed cuffs "
            "are folded back once. A few soft pills sit at the elbows.")
MAREN = {"appearance": {**BIBLE["appearance"],
                        "face": "oval face with a softly squared jaw, straight thick dark-brown eyebrows, a small pale "
                                "scar splitting the tail of her left eyebrow, medium straight nose, medium-full lips "
                                "with a slightly lifted left corner when she half-smiles, faint tired shadows under "
                                "the eyes, no visible makeup",
                        "eyes": "hazel-green, slightly hooded, almond-shaped, with fine smile lines at the outer corners",
                        "hair": "dark chocolate-brown, straight with a slight wave, shoulder-length, with a grown-out "
                                "curtain fringe parted in the middle, twisted up in a tortoiseshell claw clip",
                        "build": "average height, soft medium build, rounded shoulders, relaxed posture",
                        "signature_details": CARDIGAN}}


def test_a_photo_keeps_its_set_and_wardrobe_before_her_signature_details(policy):
    from manyworlds import policy as policy_mod
    d = policy.get("direction")
    policy = policy_mod.from_dict({**policy.body, "direction": {**d, "prompt_budget": {**d["prompt_budget"], "photo": 400}}})
    shot = {"set": BAR, "wardrobe": TEE, "framing": {"shot_size": "medium shot", "angle": "eye level", "device": "propped"},
            "blocking": [{"at_s": 0, "action": "leans on the counter and starts the story", "eyeline": "lens"}]}
    budget = policy.get("direction", "prompt_budget", "photo")
    out = direction.compile_prompt(shot, MAREN, "photo", policy, trigger="mw_m", scene="behind the bar")
    assert direction.estimate_tokens(out) <= budget
    assert "Set: behind the bar of a small, dim neighbourhood pub" in out and "black t-shirt" in out
    assert "scar splitting the tail of her left eyebrow" in out and "claw clip" in out   # the face is whole
    assert "sagging from years" not in out                                              # the signature gave way
    short = {**MAREN, "appearance": {**MAREN["appearance"], "signature_details": "thin gold wire glasses"}}
    assert direction.compile_prompt(shot, short, "photo", policy, trigger="mw_m").count("thin gold wire glasses") == 1


SHOTS = [{"set": BAR, "wardrobe": TEE, "framing": {"shot_size": "medium shot"}},
         {"set": BAR, "wardrobe": TEE, "framing": {"shot_size": "close-up"}},
         {"set": BAR, "cutaway": True, "framing": {"shot_size": "close-up"}},
         {"set": {"location": "her kitchen", "set_id": "home_kitchen"}, "framing": {"shot_size": "medium shot"}}]


def test_the_shared_set_leader_is_its_first_face_shot(policy):
    assert [sets.shared_leader(policy, "talking_video", SHOTS, i) for i in range(4)] == [None, 0, 0, None]
    assert sets.shared_leader(policy, "photo", SHOTS, 1) is None                       # formats in policy only
    faceless_first = [SHOTS[2], SHOTS[0], SHOTS[1]]
    assert [sets.shared_leader(policy, "talking_video", faceless_first, i) for i in range(3)] == [1, None, 1]


def _post(conn, cid):
    body = {"scenes": [f"behind the bar, shot {k}" for k in range(4)], "shots": SHOTS, "script": "Story time."}
    row = conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, 'talking_video', 'planned', %s) "
                       "RETURNING *", (cid, Jsonb(body))).fetchone()
    return dict(row)


def _accept(conn, job_id, cid):
    """The worker made the job's photo and the reviewer passed and confirmed it."""
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job_id,))
    a = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'image', %s, "
                     "'x', 1) RETURNING id, uri", (job_id, cid, f"s3://b/{uuid.uuid4().hex}.png")).fetchone()
    for gate in ("identity", "realism", "safety"):
        conn.execute("INSERT INTO reviews (asset_id, gate, passed, score, details) VALUES (%s, %s, true, 8, %s)",
                     (a["id"], gate, Jsonb({"stage": "confirm"})))
    return a["uri"]


def _jobs(conn, pid):
    return conn.execute("SELECT id, kind, payload FROM jobs WHERE post_id = %s ORDER BY created_at, idempotency_key",
                        (pid,)).fetchall()


def test_the_shots_of_a_shared_set_wait_for_the_leader_then_take_its_photo_as_their_set(conn, policy):
    cid = creator(conn)
    post = _post(conn, cid)
    production.start(conn, policy, post)
    started = _jobs(conn, post["id"])
    assert len(started) == 2                                            # the bar's leader and the kitchen
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (post["id"],)).fetchone()["body"]
    assert [len(j) for j in body["scene_jobs"]] == [1, 0, 0, 1]
    post = dict(conn.execute("SELECT * FROM posts WHERE id = %s", (post["id"],)).fetchone())
    production.advance_post(conn, policy, post)
    assert len(_jobs(conn, post["id"])) == 2                            # still waiting for the leader's photo
    leader = _accept(conn, body["scene_jobs"][0][0], cid)
    post = dict(conn.execute("SELECT * FROM posts WHERE id = %s", (post["id"],)).fetchone())
    production.advance_post(conn, policy, post)
    # 2026-11-25 (maren-final-3): followers conditioned on the leader's photo copied her face into the cutaways and
    # her phone hands into a shot with new hands. The leader's photo first becomes an empty plate of the place.
    [plate_job] = [j for j in _jobs(conn, post["id"]) if (j["payload"] or {}).get("plate")]
    assert plate_job["kind"] == "image.qwen_edit" and plate_job["payload"]["images"] == [leader]
    assert plate_job["payload"]["prompt"].startswith("Remove the person from picture 1")
    assert plate_job["payload"]["shared_set"] == {"set_id": "pub_bar", "leader": 0, "plate": True}
    assert len(_jobs(conn, post["id"])) == 3                            # followers wait for the plate
    plate = _accept(conn, plate_job["id"], cid)
    post = dict(conn.execute("SELECT * FROM posts WHERE id = %s", (post["id"],)).fetchone())
    production.advance_post(conn, policy, post)
    made = {j["payload"]["scene"]: j for j in _jobs(conn, post["id"]) if not j["payload"].get("plate")}
    face, hands = made["behind the bar, shot 1"], made["behind the bar, shot 2"]
    assert face["kind"] == "image.qwen_edit" and face["payload"]["images"] == ["s3://b/r0", plate]
    assert face["payload"]["shared_set"] == {"set_id": "pub_bar", "leader": 0}
    # a faceless follower: the empty plate alone (no anchor, nobody in it), with words for a picture of a place
    assert hands["kind"] == "image.qwen_edit" and hands["payload"]["images"] == [plate]
    assert hands["payload"]["references"] == [{"kind": "set", "uri": plate}]
    assert hands["payload"]["prompt"].startswith("Picture 1 is the place")
    assert hands["payload"]["shared_set"]["leader"] == 0
    assert "shared_set" not in made["behind the bar, shot 0"]["payload"]


def _reject(conn, job_id, cid):
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job_id,))
    a = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'image', %s, "
                     "'x', 1) RETURNING id", (job_id, cid, f"s3://b/{uuid.uuid4().hex}.png")).fetchone()
    conn.execute("INSERT INTO reviews (asset_id, gate, passed, score, details) VALUES (%s, 'framing', false, NULL, %s)",
                 (a["id"], Jsonb({"stage": "screen"})))


def test_followers_fall_back_to_the_leaders_photo_when_no_plate_passes(conn, policy):
    cid = creator(conn)
    post = _post(conn, cid)
    production.start(conn, policy, post)
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (post["id"],)).fetchone()["body"]
    leader = _accept(conn, body["scene_jobs"][0][0], cid)
    attempts = policy.get("sets", "shared_set", "plate", "attempts")
    for _ in range(attempts):
        production.advance_post(conn, policy, dict(conn.execute("SELECT * FROM posts WHERE id = %s",
                                                                (post["id"],)).fetchone()))
        pending = [j for j in _jobs(conn, post["id"]) if j["payload"].get("plate")
                   and conn.execute("SELECT status FROM jobs WHERE id = %s", (j["id"],)).fetchone()["status"] == "queued"]
        assert len(pending) == 1
        _reject(conn, pending[0]["id"], cid)
    production.advance_post(conn, policy, dict(conn.execute("SELECT * FROM posts WHERE id = %s", (post["id"],)).fetchone()))
    face = next(j for j in _jobs(conn, post["id"]) if j["payload"].get("scene") == "behind the bar, shot 1")
    assert face["payload"]["images"] == ["s3://b/r0", leader]
    assert len([j for j in _jobs(conn, post["id"]) if j["payload"].get("plate")]) == attempts


def test_extra_hands_reject_and_a_face_in_a_cutaway_rejects(conn, policy, store, tmp_path):
    _, job = creator_with_job(conn, store, tmp_path, n_images=3)
    conn.execute("UPDATE jobs SET payload = %s WHERE id = %s", (Jsonb({"shot": SHOTS[2]}), job))   # a cutaway
    judge = JudgeStub([ok(extra_body_parts=True, face_visible=False), ok(face_visible=True), ok(face_visible=False)])
    results = gates.review_job(conn, policy, judge, store, job, tmp_path)
    assert [r["accepted"] for r in results] == [False, False, True]
    assert results[0]["anatomy"] is False and results[1]["framing"] is False
    _, job = creator_with_job(conn, store, tmp_path, n_images=1)
    conn.execute("UPDATE jobs SET payload = %s WHERE id = %s", (Jsonb({"shot": SHOTS[0]}), job))   # her face belongs
    [r] = gates.review_job(conn, policy, JudgeStub([ok(face_visible=True)]), store, job, tmp_path)
    assert r["accepted"] and "framing" not in r


def test_a_plate_with_anyone_in_it_is_rejected(conn, policy, store, tmp_path):
    _, job = creator_with_job(conn, store, tmp_path, n_images=2)
    conn.execute("UPDATE jobs SET payload = %s WHERE id = %s", (Jsonb({"plate": True, "scene": "an empty bar"}), job))
    judge = JudgeStub([ok(person_visible=True), ok(person_visible=False)])
    assert [r["accepted"] for r in gates.review_job(conn, policy, judge, store, job, tmp_path)] == [False, True]


def test_without_conditioning_the_followers_are_text_photos_judged_against_the_leader(conn, policy):
    import dataclasses
    off = dataclasses.replace(policy, body={**policy.body, "sets": {
        **policy.body["sets"], "shared_set": {**policy.body["sets"]["shared_set"], "conditioning": False}}})
    cid = creator(conn)
    post = _post(conn, cid)
    production.start(conn, off, post)
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (post["id"],)).fetchone()["body"]
    leader = _accept(conn, body["scene_jobs"][0][0], cid)
    production.advance_post(conn, off, dict(conn.execute("SELECT * FROM posts WHERE id = %s", (post["id"],)).fetchone()))
    face = next(j for j in _jobs(conn, post["id"]) if j["payload"]["scene"] == "behind the bar, shot 1")
    assert face["kind"] == "image.zimage" and face["payload"]["references"] == [{"kind": "set", "uri": leader}]


def test_a_follower_photo_in_another_place_is_rejected(conn, policy, store, tmp_path):
    _, job = creator_with_job(conn, store, tmp_path, n_images=2)
    p = tmp_path / "leader.png"
    p.write_bytes(b"\x89PNG leader")
    leader = store.put_file(p, "t/leader.png").uri
    conn.execute("UPDATE jobs SET payload = %s WHERE id = %s",
                 (Jsonb({"shot": SHOTS[1], "references": [{"kind": "set", "uri": leader}],
                         "shared_set": {"set_id": "pub_bar", "leader": 0}}), job))
    bar = policy.get("sets", "shared_set", "min_consistency")
    judge = JudgeStub([ok(consistency=bar - 2), ok(consistency=bar + 3)])
    results = gates.review_job(conn, policy, judge, store, job, tmp_path)
    assert [r["accepted"] for r in results] == [False, True]
    assert results[0]["set_match"] is False and results[1]["set_match"] is True
    rows = conn.execute("SELECT r.passed, r.score FROM reviews r JOIN assets a ON a.id = r.asset_id "
                        "WHERE a.job_id = %s AND r.gate = 'set_match' ORDER BY a.uri", (job,)).fetchall()
    assert [(x["passed"], float(x["score"])) for x in rows] == [(False, bar - 2), (True, bar + 3)]
    assert "SET REFERENCE (pub_bar)" in " ".join(str(b.get("text", "")) for b in judge.calls[0]["content"])


def test_without_a_shared_set_the_consistency_score_still_only_reports(conn, policy, store, tmp_path):
    _, job = creator_with_job(conn, store, tmp_path, n_images=1)
    conn.execute("UPDATE jobs SET payload = %s WHERE id = %s",
                 (Jsonb({"shot": SHOTS[1], "references": [{"kind": "set", "uri": "s3://b/plate.png"}]}), job))
    [r] = gates.review_job(conn, policy, JudgeStub([ok(consistency=1)]), store, job, tmp_path)
    assert "set_match" not in r


def test_a_hands_cutaway_is_framed_from_her_own_eyes_and_its_scene_loses_its_negations(policy):
    # 2026-11-25 maren-final-4: "phone held by someone else" and a repair that led with "..., no face" drew her head
    # into all eight hands-at-the-sink cutaways (each rejected for a face in a cutaway)
    from manyworlds.review import repair
    hands = {"set": BAR, "cutaway": True, "framing": {"shot_size": "close-up", "angle": "high angle", "device": "held"},
             "blocking": [{"at_s": 0, "action": "fills a battered steel kettle at the bar sink"}]}
    out = direction.compile_prompt(hands, MAREN, "photo", policy, scene="hands filling a kettle, no face")
    assert "held by someone else" not in out and "point of view from her eyes" in out and "cropped at the forearms" in out
    face = direction.compile_prompt(SHOTS[0], MAREN, "photo", policy)
    assert "cropped at the forearms" not in face
    adjust, _ = repair.plan(policy, "scene", scene="close-up of her hands filling a kettle, warm bar light, no face",
                            identity_repairs=0)
    p = adjust({"prompt": "base"})
    assert "no face" not in p["prompt"] and "filling a kettle" in p["prompt"]


def test_a_directed_photo_keeps_its_performance_with_her_full_appearance(policy):
    """2026-11-26: six photos of Maren directed at different sass and candidness levels compiled to one prompt: her
    long appearance filled the budget and the performance section was trimmed away first."""
    from manyworlds.direction import ControlLevel
    prompts = []
    for level in (1, 2):
        shot = {"set": BAR, "wardrobe": TEE, "framing": {"shot_size": "medium shot", "angle": "eye level", "device": "propped"},
                "blocking": [{"at_s": 0, "action": "leans on the counter after closing", "eyeline": "lens"}],
                "performance": {"intent": "a dry, knowing closing-time look",
                                "controls": [ControlLevel(control="sass", level=level).model_dump()]}}
        out = direction.compile_prompt(shot, MAREN, "photo", policy, trigger="mw_m", scene="behind the bar")
        assert direction.estimate_tokens(out) <= policy.get("direction", "prompt_budget", "photo")
        assert "dry, knowing closing-time look" in out and "scar splitting the tail of her left eyebrow" in out
        prompts.append(out)
    assert prompts[0] != prompts[1]                                   # the level reaches the model


def test_a_trimmed_directed_photo_keeps_the_place_and_whole_performance_clauses(policy):
    from manyworlds import policy as policy_mod
    from manyworlds.direction import ControlLevel
    d = policy.get("direction")
    tight = policy_mod.from_dict({**policy.body, "direction": {**d, "prompt_budget": {**d["prompt_budget"], "photo": 380}}})
    shot = {"set": BAR, "wardrobe": TEE, "framing": {"shot_size": "medium shot", "angle": "eye level", "device": "propped"},
            "blocking": [{"at_s": 0, "action": "leans on the counter after closing", "eyeline": "lens"}],
            "performance": {"intent": "caught mid-laugh behind the bar",
                            "controls": [ControlLevel(control="candidness", level=4).model_dump()],
                            "extreme_reason": "an unguarded snapshot is the point"}}
    out = direction.compile_prompt(shot, MAREN, "photo", tight, trigger="mw_m", scene="behind the bar")
    assert "Set: behind the bar" in out and "caught mid-laugh behind the bar" in out
    perf = out[out.index("Performance:"):].split(". ")[0]
    assert not perf.rstrip().endswith(("camera", "face", "eyes", "body", ":"))


def test_a_beats_eyeline_wins_over_the_performances_gaze(policy):
    from manyworlds.direction import ControlLevel
    def shot(eyeline):
        return {"set": BAR, "wardrobe": TEE, "framing": {"shot_size": "medium shot"},
                "blocking": [{"at_s": 0, "action": "leans on the counter", "eyeline": eyeline}],
                "performance": {"intent": "a quiet moment", "controls": [ControlLevel(control="candidness", level=3).model_dump()]}}
    with_lens = direction.compile_prompt(shot("lens"), MAREN, "photo", policy, trigger="mw_m")
    free = direction.compile_prompt(shot(None), MAREN, "photo", policy, trigger="mw_m")
    gaze = "looking away from the camera"
    assert "looking into the lens" in with_lens and gaze not in with_lens
    assert gaze in free


def test_a_beat_that_blocks_her_hands_wins_over_the_performances_body(policy):
    from manyworlds.direction import ControlLevel
    def shot(hands):
        return {"set": BAR, "wardrobe": TEE, "framing": {"shot_size": "medium shot"},
                "blocking": [{"at_s": 0, "action": "leans on the counter", "hands": hands}],
                "performance": {"intent": "a teasing look", "controls": [ControlLevel(control="sass", level=2).model_dump()]}}
    blocked = direction._performance(direction._coerce(shot("one hand flat on the counter")), "talk", policy)
    free = direction._performance(direction._coerce(shot("")), "talk", policy)
    assert "body:" not in blocked and "hand on hip" not in blocked
    assert blocked != free or "body:" not in free


def test_walk_phrases_only_when_she_moves_through_the_space(policy):
    from manyworlds.direction import ControlLevel
    def shot(action):
        return {"set": BAR, "wardrobe": TEE, "framing": {"shot_size": "medium shot"},
                "blocking": [{"at_s": 0, "action": action}],
                "performance": {"intent": "teasing", "controls": [ControlLevel(control="sass", level=3).model_dump()]}}
    standing = direction._performance(direction._coerce(shot("leans on the counter")), "i2v", policy)
    walking = direction._performance(direction._coerce(shot("walks along the bar toward the camera")), "i2v", policy)
    assert "strut" not in standing and "strut" in walking
