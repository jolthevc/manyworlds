"""Fixes from the 2026-11-25 independent review, each through the function production calls: candidate ranking by core
gates only, a rejecting confirm walking down to the next screen passes, a talking take handing on to the next route
after its own repairs, and a talking post whose repairs ran out shipping as its photo."""

from __future__ import annotations

import dataclasses

from psycopg.types.json import Jsonb

from manyworlds import production
from manyworlds.review import gates
from manyworlds.production.steps import Step, advance_step

from test_review import JudgeStub, creator_with_job, ok
from test_talk_repair import _judged, _story
from test_talk_route import ROUTE, _seg


def test_ranking_uses_core_gates_only_never_report_rows(conn):
    cid = conn.execute("INSERT INTO creators (handle) VALUES ('r') RETURNING id").fetchone()["id"]
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id) VALUES ('image.zimage', 'done', %s) RETURNING id",
                       (cid,)).fetchone()["id"]
    uris = {}
    for name, core, risk in (("risky", 8, 9), ("safe", 9, 1)):
        a = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'image', "
                         "%s, 'x', 1) RETURNING id", (job, cid, f"s3://b/{name}.png")).fetchone()["id"]
        uris[name] = f"s3://b/{name}.png"
        for gate, score in (("identity", core), ("realism", core), ("distribution", risk)):
            conn.execute("INSERT INTO reviews (asset_id, gate, passed, score, details) VALUES (%s, %s, true, %s, %s)",
                         (a, gate, score, Jsonb({"stage": "confirm"})))
    best = production._best_confirmed(conn, [str(job)])
    assert best["uri"] == uris["safe"]          # distribution_risk 9 no longer outranks a better photo


class WalkJudge(JudgeStub):
    """Screen passes three; each confirm call takes the next answer from a queue."""
    def __init__(self, screen, confirms):
        super().__init__(screen)
        self.confirms = list(confirms)

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        if self.calls:
            n = sum(1 for b in content if str(b.get("text", "")).startswith("Candidate "))
            self.calls.append({"role": role, "n": n})
            got = self.confirms.pop(0)
            return schema(reviews=[gates.CandidateReview(index=i, **got) for i in range(n)])
        return super().structured(role, system=system, content=content, schema=schema, effort=effort,
                                  creator_id=creator_id)


def test_a_rejecting_confirm_walks_down_to_the_next_screen_pass(conn, policy, store, tmp_path):
    shipped = dataclasses.replace(policy, body={**policy.body, "review": {**policy.body["review"], "image": {
        **policy.body["review"]["image"], "confirm_top_per_job": 1, "clear_pass_margin": None, "confirm_walk_max": 3}}})
    _, job = creator_with_job(conn, store, tmp_path, n_images=3)
    judge = WalkJudge([ok(), ok(identity=8), ok(identity=7)], [ok(identity=2), ok()])
    results = gates.review_job(conn, shipped, judge, store, job, tmp_path)
    assert len(judge.calls) == 3                                   # screen, confirm (rejects), confirm (passes)
    assert sum(1 for r in results if r["confirmed"]) == 1
    assert production._best_confirmed(conn, [str(job)]) is not None   # chosen without a new image job


def test_a_take_hands_on_to_the_next_route_after_its_own_repairs_even_when_scores_vary(conn, policy, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    post, body, chosen = _story(conn)
    body["talk_route"] = ROUTE
    last = len(body["talk_segments"]) - 1
    per = int(policy.get("review", "video", "max_repairs_per_post"))
    kinds = [_seg(conn, policy, post, body, chosen, last)["kind"]]
    for k in range(per + 1):
        _judged(conn, body["talk_segment_jobs"][last][-1], (-0.46, -0.1, -0.4, -0.05)[k % 4])   # never repeats
        kinds.append(_seg(conn, policy, post, body, chosen, last)["kind"])
    assert kinds[:per + 1] == ["comfy.graph"] * (per + 1) and kinds[per + 1] == "talk.infinitetalk"


def test_a_take_whose_repairs_run_out_counts_as_rejected_for_the_photo_reroute(conn):
    body = {}

    def make(attempt):
        raise production.DropPost("lip sync failed and no repair changes it")
    step = Step(["j1"], lambda jobs: ("rejected", None), make, ("clip_repairs", 9), ("follow_up_failures", 9), "clip")
    assert advance_step(body, step)[0] == "dropped"
    assert body["drop_kind"] == "rejected" and "passed review" not in body["drop_reason"]
