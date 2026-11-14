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
    judge = JudgeStub([ok(), ok(identity=3), ok(), ok(realism=2)], confirm=[ok(), ok(identity=5)])
    results = gates.review_job(conn, policy, judge, store, job, tmp_path)
    cfg = policy.get("review", "image")
    assert [c["role"] for c in judge.calls] == [cfg["role"], cfg["confirm_role"]]
    assert judge.calls[1]["n"] == 2                       # only the two screen passes
    assert [r["accepted"] for r in results] == [True, False, False, False]
    assert [len(r["stages"]) for r in results] == [2, 1, 2, 1]
    accepted = conn.execute("""SELECT count(*) n FROM assets a WHERE a.job_id = %s AND NOT EXISTS
        (SELECT 1 FROM reviews r WHERE r.asset_id = a.id AND NOT r.passed)""", (job,)).fetchone()["n"]
    assert accepted == 1
