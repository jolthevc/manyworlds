"""A fixed reference set shows how scores move when a rubric or model changes, and calibration
shows how noisy each dimension is on a creator's real ideas."""

from __future__ import annotations

from manyworlds.brain import evaluation, rubrics

from brain_helpers import ScoringLLM, creator, planned_post


def test_drift_is_mean_absolute_shift_on_shared_items():
    before = {"a": {"novelty": 4, "curiosity": 6}, "b": {"novelty": 6, "curiosity": None}, "old": {"novelty": 1}}
    after = {"a": {"novelty": 6, "curiosity": 5}, "b": {"novelty": 5, "curiosity": 7}, "new": {"novelty": 9}}
    assert evaluation.drift(before, after) == {"novelty": 1.5, "curiosity": 1.0}


def test_the_reference_set_is_loaded_and_its_scores_stored_on_the_rubric_version(conn, policy):
    refs = evaluation.reference_items("idea")
    assert len(refs) >= 10 and all(r["id"] and r["idea"] for r in refs)
    scores = evaluation.score_reference(conn, policy, ScoringLLM(5, conn=conn), "idea")
    assert set(scores) == {r["id"] for r in refs} and scores[refs[0]["id"]]["novelty"] == 5
    stored = conn.execute("SELECT reference FROM rubric_versions WHERE rubric = 'idea' AND version = %s",
                          (rubrics.load("idea").version,)).fetchone()["reference"]
    assert stored["scores"] == scores and stored["model"] == "stub-evaluator"


def test_calibration_reports_spread_per_dimension(conn, policy):
    cid = creator(conn)
    for _ in range(3):
        planned_post(conn, cid)
    report = evaluation.calibrate(conn, policy, ScoringLLM(6, conn=conn), cid, posts=3)
    assert report["items"] == 3 and report["dimensions"]["novelty"]["median_spread"] == 0
    assert report["dimensions"]["novelty"]["within_max_spread"] == 1.0 and report["passing_dimensions"] == 10
