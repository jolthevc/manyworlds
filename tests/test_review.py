"""Review gates decide which generated images count: off-identity, unrealistic or unsafe
images are rejected, recorded per gate, and left out of cost per accepted output."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds import llm as llm_mod
from manyworlds.deploy import scoring
from manyworlds.review import gates

BIBLE = {"appearance": {"apparent_age": 29, "gender_presentation": "woman", "ethnicity_and_skin": "olive skin",
                        "face": "oval face", "eyes": "brown eyes", "hair": "black bob with a clip",
                        "build": "slim", "signature_details": "thin gold wire glasses"}}


class JudgeStub:
    """Stands in for Claude: the screen returns the scores it is given (in candidate order);
    the confirm stage passes everything unless `confirm` scores are given. Records calls."""

    def __init__(self, scores, confirm=None):
        self.scores, self.confirm, self.calls = scores, confirm, []

    def model_for(self, role):
        return f"stub-{role}"

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        n = sum(1 for b in content if str(b.get("text", "")).startswith("Candidate "))
        self.calls.append({"role": role, "content": content, "effort": effort, "n": n})
        first = len(self.calls) == 1
        scores = self.scores if first else (self.confirm or [ok()] * n)
        return schema(reviews=[gates.CandidateReview(index=i, **x) for i, x in enumerate(scores[:n])])


def ok(**kw):
    return {"identity": 9, "realism": 8, "defects": [], "looks_under_min_age": False, "sexualized": False,
            "real_brand_marks": False, "notes": "fine", **kw}


def creator_with_job(conn, store, tmp_path, n_images=4, batch="posts-1", refs=3):
    def put(name):
        p = tmp_path / f"{uuid.uuid4().hex}.png"
        p.write_bytes(b"\x89PNG fake " + name.encode())
        return store.put_file(p, f"t/{p.name}").uri
    references = [{"uri": put(f"ref{i}")} for i in range(refs)]
    cid = conn.execute("INSERT INTO creators (handle, status, bible, anchors) VALUES (%s, 'draft', %s, %s) RETURNING id",
                       (f"c{uuid.uuid4().hex[:6]}", Jsonb(BIBLE), Jsonb({"reference_images": references}))).fetchone()["id"]
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, batch) VALUES ('image.zimage', 'done', %s, %s) "
                       "RETURNING id", (cid, batch)).fetchone()["id"]
    for i in range(n_images):
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'image', %s, 'x', 1)",
                     (job, cid, put(f"img{i}")))
    return cid, job


def test_each_gate_is_recorded_and_any_failure_rejects(conn, policy, store, tmp_path):
    _, job = creator_with_job(conn, store, tmp_path)
    judge = JudgeStub([ok(), ok(identity=4, defects=["no glasses"]), ok(realism=3), ok(real_brand_marks=True)])
    results = gates.review_job(conn, policy, judge, store, job, tmp_path)
    assert [r["accepted"] for r in results] == [True, False, False, False]
    assert [r["identity"] for r in results] == [True, False, True, True]
    assert [r["safety"] for r in results] == [True, True, True, False]
    rows = conn.execute("SELECT gate, count(*) n, sum(passed::int) p FROM reviews WHERE details->>'stage' = 'screen' "
                        "GROUP BY gate ORDER BY gate").fetchall()
    assert [(r["gate"], r["n"], r["p"]) for r in rows] == [("identity", 4, 3), ("realism", 4, 3), ("safety", 4, 3)]


def test_judge_sees_the_references_cached_then_every_candidate(conn, policy, store, tmp_path):
    _, job = creator_with_job(conn, store, tmp_path, n_images=2)
    judge = JudgeStub([ok(), ok()])
    gates.review_job(conn, policy, judge, store, job, tmp_path)
    call = judge.calls[0]
    images = [b for b in call["content"] if b.get("type") == "image"]
    assert len(images) == policy.get("review", "image", "reference_images") + 2
    cached = [i for i, b in enumerate(call["content"]) if "cache_control" in b]
    first_candidate = next(i for i, b in enumerate(call["content"]) if b.get("text") == "Candidate 0:")
    assert cached and cached[-1] < first_candidate          # cache breakpoint before the candidates
    assert call["role"] == policy.get("review", "image", "role")
    assert "glasses" in call["content"][0]["text"]          # the signature details are spelled out


def test_a_candidate_the_judge_skipped_fails_closed(conn, policy, store, tmp_path):
    _, job = creator_with_job(conn, store, tmp_path, n_images=3)
    results = gates.review_job(conn, policy, JudgeStub([ok(), ok()]), store, job, tmp_path)
    assert [r["accepted"] for r in results] == [True, True, False]


def test_pending_skips_reviewed_jobs_and_the_reference_pack(conn, policy, store, tmp_path):
    _, done = creator_with_job(conn, store, tmp_path)
    _, refpack = creator_with_job(conn, store, tmp_path, batch="refpack-x")
    _, no_refs = creator_with_job(conn, store, tmp_path, refs=0)
    assert gates.pending_jobs(conn, policy) == [done]
    gates.run_pending(conn, policy, JudgeStub([ok()] * 4), store, tmp_path)
    assert gates.pending_jobs(conn, policy) == []


def test_rejected_images_do_not_count_toward_cost_per_accepted_output(conn, policy, store, tmp_path):
    _, job = creator_with_job(conn, store, tmp_path)
    now = datetime.now(timezone.utc)
    conn.execute("""INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd, started_at,
                    billed_until, billed_usd, last_busy_at, stopped_at, stop_reason, family)
                    VALUES ('mw-image-3090-vast-t', 'vast', 'vast_3090', 0.2, '{image.zimage}', %s, 1, %s, %s, 0.04, %s, %s,
                    'idle', 'image')""", (now + timedelta(hours=1), now - timedelta(hours=1), now, now, now))
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, outcome, gpu_group) VALUES (%s, 'w', 'vast_3090', 'done', "
                 "'mw-image-3090-vast-t')", (job,))
    gates.review_job(conn, policy, JudgeStub([ok(), ok(identity=2), ok(identity=2), ok(identity=2)]), store, job, tmp_path)
    row = scoring.score_pools(conn, policy, "image", ["vast_3090"], now)[0]
    assert row["outputs"] == 1 and abs(row["usd_per_output"] - 0.04) < 1e-9


def test_refusal_fallback_is_only_requested_from_models_that_support_it(conn, policy):
    sent = []

    class Client:
        class messages:
            @staticmethod
            def parse(**kw):
                sent.append(kw)

                class U:
                    input_tokens = output_tokens = 1
                    cache_read_input_tokens = cache_creation_input_tokens = 0

                class R:
                    model, usage, stop_reason = kw["model"], U, "end_turn"
                    parsed_output = gates.ImageReviews(reviews=[])
                return R

    import dataclasses
    roles = {**policy.get("llm", "roles"), "cheap": "claude-haiku-5-5"}
    pol = dataclasses.replace(policy, body={**policy.body, "llm": {**policy.body["llm"], "roles": roles}})
    c = llm_mod.Claude(conn, pol, client=Client())
    c.structured("judge", system="s", content="x", schema=gates.ImageReviews)
    c.structured("cheap", system="s", content="x", schema=gates.ImageReviews)
    assert sent[0].get("extra_body") == {"fallbacks": "default"}
    assert "extra_body" not in sent[1] and "extra_headers" not in sent[1]


def test_only_screen_passes_reach_the_stronger_confirm_and_it_can_still_reject(conn, policy, store, tmp_path):
    _, job = creator_with_job(conn, store, tmp_path)
    judge = JudgeStub([ok(), ok(identity=3), ok(), ok(realism=2)], confirm=[ok(), ok(identity=4)])   # 4: clearly not her
    results = gates.review_job(conn, policy, judge, store, job, tmp_path)
    cfg = policy.get("review", "image")
    assert [c["role"] for c in judge.calls] == [cfg["role"], cfg["confirm_role"]]
    assert judge.calls[1]["n"] == 2                       # only the two screen passes
    assert [r["accepted"] for r in results] == [True, False, False, False]
    assert [len(r["stages"]) for r in results] == [2, 1, 2, 1]
    accepted = conn.execute("""SELECT count(*) n FROM assets a WHERE a.job_id = %s AND NOT EXISTS
        (SELECT 1 FROM reviews r WHERE r.asset_id = a.id AND NOT r.passed)""", (job,)).fetchone()["n"]
    assert accepted == 1


def test_confirm_goes_only_to_the_best_screen_passes(conn, policy, store, tmp_path):
    _, job = creator_with_job(conn, store, tmp_path, n_images=4)
    top = policy.get("review", "image", "confirm_top_per_job")
    judge = JudgeStub([ok(identity=7), ok(identity=9), ok(identity=8), ok(identity=10)])
    results = gates.review_job(conn, policy, judge, store, job, tmp_path)
    assert judge.calls[1]["n"] == top
    assert all(r["accepted"] for r in results)
    best = sorted(range(4), key=lambda i: -judge.scores[i]["identity"])[:top]
    assert [r["confirmed"] for r in results] == [i in best for i in range(4)]


def test_photo_prompt_carries_the_full_appearance():
    from manyworlds.creator.lab import photo_prompt
    p = photo_prompt(BIBLE, "at a bus stop", trigger="mw_x")
    assert p.startswith("mw_x, ") and "thin gold wire glasses" in p and "bus stop" in p


def test_a_jobs_reviews_appear_all_at_once_when_both_stages_are_done(conn, database_url, policy, store, tmp_path):
    # 2026-11-14: production read a job's screen rows while the confirm call was still running,
    # saw every image reviewed and none confirmed, called the scene rejected and paid for a
    # needless repair; the post then went out with the weaker image.
    from manyworlds import db
    _, job = creator_with_job(conn, store, tmp_path)
    other = db.connect(database_url)
    seen = []

    class Watching(JudgeStub):
        def structured(self, role, **kw):
            seen.append(other.execute("SELECT count(*) n FROM reviews r JOIN assets a ON a.id = r.asset_id "
                                      "WHERE a.job_id = %s", (job,)).fetchone()["n"])
            return super().structured(role, **kw)

    gates.review_job(conn, policy, Watching([ok(), ok(), ok(), ok()]), store, job, tmp_path)
    assert seen == [0, 0]                                  # nothing visible mid-review
    assert other.execute("SELECT count(*) n FROM reviews r JOIN assets a ON a.id = r.asset_id "
                         "WHERE a.job_id = %s", (job,)).fetchone()["n"] > 0
    other.close()


def test_post_images_are_ranked_by_the_scene_the_planner_wrote_but_not_rejected_for_it(conn, policy, store, tmp_path):
    # 2026-11-19 (owner): scene fit is a preference. Rejecting on it (with a beanie forced into every
    # prompt) threw away 9 in 10 of Maren's images. A close-enough image is kept; the best fits are chosen.
    _, job = creator_with_job(conn, store, tmp_path)
    conn.execute("UPDATE jobs SET payload = %s WHERE id = %s", (Jsonb({"scene": "at the window, 4:35am"}), job))
    judge = JudgeStub([ok(scene=9), ok(scene=2), ok(scene=6), ok(scene=8)], confirm=[ok(scene=9), ok(scene=8)])
    results = gates.review_job(conn, policy, judge, store, job, tmp_path)
    assert any("INTENDED SCENE" in str(b.get("text", "")) and "4:35am" in str(b.get("text", ""))
               for b in judge.calls[0]["content"])
    assert [r["accepted"] for r in results] == [True, True, True, True]      # a different scene is not a reject
    assert [r["confirmed"] for r in results] == [True, False, False, True]   # best scene fits go to confirm
    rows = conn.execute("SELECT count(*) n FROM reviews WHERE gate = 'scene'").fetchone()["n"]
    assert rows == 4 + 2


def test_images_without_a_scene_get_no_scene_gate(conn, policy, store, tmp_path):
    _, job = creator_with_job(conn, store, tmp_path)
    judge = JudgeStub([ok(), ok(), ok(), ok()])
    gates.review_job(conn, policy, judge, store, job, tmp_path)
    assert not any("INTENDED SCENE" in str(b.get("text", "")) for b in judge.calls[0]["content"])
    assert conn.execute("SELECT count(*) n FROM reviews WHERE gate = 'scene'").fetchone()["n"] == 0


def test_a_little_drift_is_fine_a_different_face_or_clear_defects_are_not(conn, policy, store, tmp_path):
    # 2026-11-19 (owner): keep images unless the face is clearly someone else or there are clear AI defects.
    _, job = creator_with_job(conn, store, tmp_path)
    judge = JudgeStub([ok(identity=5, realism=5), ok(identity=3), ok(realism=3), ok(identity=6, realism=6)])
    results = gates.review_job(conn, policy, judge, store, job, tmp_path)
    assert [r["accepted"] for r in results] == [True, False, False, True]


# --- repair hints: positive wording in the prompt, what to avoid in the negative prompt --------------------------
NEGATION_STARTS = ("no ", "not ", "without ", "avoid ", "never ")


def _clauses(text):
    return [c.strip().lstrip(".").strip().lower() for c in text.split(",")]


def test_repair_hints_never_put_a_negation_in_the_prompt(policy):
    from manyworlds.review import repair
    for cause in ("identity", "scene", "realism", "hands", "teeth"):
        adjust, _ = repair.plan(policy, cause, scene="reading on a balcony", identity_repairs=0)
        out = adjust({"prompt": "a woman reading", "negative_prompt": "blurry", "lora_scale": 1.0})
        assert not any(c.startswith(NEGATION_STARTS) for c in _clauses(out["prompt"])), (cause, out["prompt"])
        assert out["negative_prompt"].startswith("blurry")
    adjust, _ = repair.plan(policy, "realism", scene="", identity_repairs=0)
    out = adjust({"prompt": "p"})
    avoid = policy.get("review", "repair", "hints", "realism", "avoid")
    assert out["negative_prompt"] == ", ".join(avoid)
    assert "five fingers on each hand" in out["prompt"] and "blank, unprinted surfaces" in out["prompt"].lower()  # no channel
    again = adjust(out)
    assert again["negative_prompt"] == out["negative_prompt"]                 # a second repair adds nothing twice
    assert again["prompt"].count("five fingers on each hand") == 1
    adjust, _ = repair.plan(policy, "realism", scene="", identity_repairs=0, target="i2v")
    wan = adjust({"prompt": "p"})
    assert wan["negative_prompt"].startswith(policy.get("direction", "negatives", "base", "i2v")[:20])   # added to, not replaced
    assert wan["prompt"].startswith("p")


def test_a_negated_clause_in_a_string_hint_moves_to_the_negative_prompt():
    from manyworlds.review import repair
    text, avoid = repair.split_hint(". Relaxed hands or hands out of frame, natural anatomy, no text or logos")
    assert text == ". Relaxed hands or hands out of frame, natural anatomy" and avoid == ["text or logos"]
    assert repair.split_hint({"positive": ". Plain wall", "avoid": "logos"}) == (". Plain wall", ["logos"])
    assert repair.split_hint("The photo must clearly show: {scene}. ") == ("The photo must clearly show: {scene}. ", [])
    assert repair.with_negative({"prompt": "p"}, []) == {"prompt": "p"}


def test_an_image_in_a_breadth_batch_is_judged_under_the_test_overlay(conn, policy, store, tmp_path):
    from test_video_review import SHOT
    prefix = policy.get("operations", "test_overlay", "batches")[0]
    _, job = creator_with_job(conn, store, tmp_path, n_images=1, batch=f"{prefix}r1")
    conn.execute("UPDATE jobs SET payload = %s WHERE id = %s", (Jsonb({"scene": "at a rainy bus stop", "shot": SHOT}), job))
    [r] = gates.review_job(conn, policy, JudgeStub([ok(direction=1)], confirm=[ok(direction=1)]), store, job, tmp_path)
    assert r["accepted"] and r["direction"] is True          # report-only in the overlay; production rejects a 1
