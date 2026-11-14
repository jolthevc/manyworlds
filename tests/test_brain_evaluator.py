"""The evaluator scores ideas before they are made and images after, independently of whoever
planned them, and stores every score with what it was scored under."""

from __future__ import annotations

import dataclasses

import pytest

from manyworlds import queue
from manyworlds.brain import evaluation, ideas, runtime
from manyworlds.brain.agents import evaluator

from brain_helpers import ScoringLLM, creator, make_ctx, planned_post


def _ctx(conn, policy, llm):
    job = queue.enqueue(conn, "brain.evaluator", {})
    ctx = make_ctx(conn, policy, job)
    ctx.llm = llm
    return ctx


def test_from_post_is_idempotent(conn):
    post = planned_post(conn, creator(conn))
    a, b = ideas.from_post(conn, post), ideas.from_post(conn, post)
    row = conn.execute("SELECT concept, format, attributes, post_id FROM ideas WHERE id = %s", (a,)).fetchone()
    assert a == b and row["post_id"] == post and row["format"] == "photo"
    assert "Venice" in row["concept"] and "best pasta" in row["concept"] and row["attributes"]["pillar"] == "food"


def test_an_idea_is_scored_and_stored_with_its_provenance(conn, policy):
    idea = ideas.from_post(conn, planned_post(conn, creator(conn)))
    eid = evaluator.evaluate_idea(_ctx(conn, policy, ScoringLLM(7, conn=conn)), idea)
    e = conn.execute("SELECT * FROM evaluations WHERE id = %s", (eid,)).fetchone()
    assert (e["item_kind"], e["idea_id"], e["rubric"], e["rubric_version"]) == ("idea", idea, "idea", 1)
    assert e["evaluator_model"] == "stub-evaluator" and e["prompt_version"] == "eval-v1" and e["format"] == "photo"
    assert e["samples"] == policy.get("brain", "evaluation", "samples") and e["scores"]["novelty"] == 7
    assert evaluation.latest(conn, idea_id=idea)["id"] == eid


def test_the_evaluator_never_sees_the_showrunners_reasoning(conn, policy):
    idea = ideas.from_post(conn, planned_post(conn, creator(conn)))
    conn.execute("UPDATE ideas SET body = body || '{\"reasoning\": \"SECRET PLAN\"}' WHERE id = %s", (idea,))
    llm = ScoringLLM(conn=conn)
    evaluator.evaluate_idea(_ctx(conn, policy, llm), idea)
    assert llm.calls and not any("SECRET PLAN" in str(c["content"]) for c in llm.calls)


def test_a_rerun_does_not_store_a_second_evaluation(conn, policy):
    idea = ideas.from_post(conn, planned_post(conn, creator(conn)))
    ctx = _ctx(conn, policy, ScoringLLM(conn=conn))
    assert evaluator.evaluate_idea(ctx, idea) == evaluator.evaluate_idea(ctx, idea)
    assert conn.execute("SELECT count(*) n FROM evaluations").fetchone()["n"] == 1


def test_checked_items_get_a_strong_model_row_too(conn, policy):
    ev = {**policy.body["brain"]["evaluation"], "check_rate": 1.0}
    pol = dataclasses.replace(policy, body={**policy.body, "brain": {**policy.body["brain"], "evaluation": ev}})
    idea = ideas.from_post(conn, planned_post(conn, creator(conn)))
    evaluator.evaluate_idea(_ctx(conn, pol, ScoringLLM(conn=conn)), idea)
    models = sorted(r["evaluator_model"] for r in conn.execute("SELECT evaluator_model FROM evaluations").fetchall())
    assert models == ["stub-evaluator", "stub-evaluator_check"]


def test_an_image_is_scored_on_the_execution_rubric(conn, policy, store, tmp_path):
    cid = creator(conn)
    post = planned_post(conn, cid)
    p = tmp_path / "img.png"
    p.write_bytes(b"\x89PNG fake")
    job = queue.enqueue(conn, "image.zimage", {}, creator_id=cid, post_id=post)
    asset = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'image', %s, 'x', 1) "
                         "RETURNING id", (job, cid, store.put_file(p, "a/img.png").uri)).fetchone()["id"]
    llm = ScoringLLM(8, conn=conn)
    eid = evaluator.evaluate_asset(_ctx(conn, policy, llm), asset, store=store)
    e = conn.execute("SELECT item_kind, asset_id, rubric, scores FROM evaluations WHERE id = %s", (eid,)).fetchone()
    assert (e["item_kind"], e["asset_id"], e["rubric"], e["scores"]["believability"]) == ("asset", asset, "execution", 8)
    assert any(b.get("type") == "image" for b in llm.calls[0]["content"] if isinstance(b, dict))
    assert "best pasta" in str(llm.calls[0]["content"])                     # the caption is judged too


def test_an_asset_that_cannot_be_fetched_fails_without_a_row(conn, policy, store):
    cid = creator(conn)
    job = queue.enqueue(conn, "image.zimage", {}, creator_id=cid)
    asset = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) "
                         "VALUES (%s, %s, 'image', 'file:///nowhere/x.png', 'x', 1) RETURNING id", (job, cid)).fetchone()["id"]
    with pytest.raises(Exception):
        evaluator.evaluate_asset(_ctx(conn, policy, ScoringLLM(conn=conn)), asset, store=store)
    assert conn.execute("SELECT count(*) n FROM evaluations").fetchone()["n"] == 0


def test_the_evaluator_runs_as_a_brain_job(conn, policy):
    idea = ideas.from_post(conn, planned_post(conn, creator(conn)))
    job = queue.enqueue(conn, "brain.evaluator", {"idea_id": str(idea)})
    out = runtime.tick(conn, policy, ScoringLLM(conn=conn), worker_id="w1")
    assert out[0]["status"] == "done"
    assert conn.execute("SELECT count(*) n FROM evaluations WHERE idea_id = %s", (idea,)).fetchone()["n"] == 1
    assert conn.execute("SELECT agent FROM cost_entries LIMIT 1").fetchone()["agent"] == "evaluator"
