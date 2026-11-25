"""One place per shared set (perf_compare 2026-11-25: nine shots named pub_bar; one photo was at the bar, eight in a
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
    made = {j["payload"]["scene"]: j for j in _jobs(conn, post["id"])}
    face, hands = made["behind the bar, shot 1"], made["behind the bar, shot 2"]
    assert face["kind"] == "image.qwen_edit" and face["payload"]["images"] == ["s3://b/r0", leader]
    assert face["payload"]["shared_set"] == {"set_id": "pub_bar", "leader": 0}
    # 2026-11-25: text-only faceless followers drew a daylight home kitchen 12 times and the post dropped on
    # set_match; a faceless follower is now conditioned on the set photo alone (no anchor, so no face is drawn).
    assert hands["kind"] == "image.qwen_edit" and hands["payload"]["images"] == [leader]
    assert hands["payload"]["references"] == [{"kind": "set", "uri": leader}]
    assert hands["payload"]["shared_set"]["leader"] == 0
    assert "shared_set" not in made["behind the bar, shot 0"]["payload"]


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
