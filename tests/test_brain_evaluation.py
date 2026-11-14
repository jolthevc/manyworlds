"""Scores are medians over several samples, with their spread kept, so they can be compared
over time; nothing is invented when the model leaves a dimension out."""

from __future__ import annotations

import uuid

from manyworlds.brain import evaluation, rubrics


class ScriptedLLM:
    """Returns the given per-sample score dicts in order ({dimension: score}); missing keys are
    left out of the response."""

    def __init__(self, samples):
        self.samples, self.calls = list(samples), []

    def model_for(self, role):
        return f"stub-{role}"

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        self.calls.append({"role": role, "content": content})
        scores = self.samples[len(self.calls) - 1]
        data = {k: {"score": v, "why": f"{k} because"} for k, v in scores.items()}
        return schema.model_validate({**data, "overall_note": "n"})


IDEA = rubrics.load("idea")


def _all(v):
    return {d.key: v for d in IDEA.dimensions}


def test_median_and_spread_across_samples(policy):
    llm = ScriptedLLM([_all(4), _all(6), _all(8)])
    s = evaluation.score(llm, policy, IDEA, ["an idea"], role="evaluator", samples=3)
    assert s.medians["novelty"] == 6 and s.spreads["novelty"] == 4 and s.samples == 3
    assert s.model == "stub-evaluator" and len(llm.calls) == 3
    assert s.rationale["novelty"] == ["novelty because"] * 3


def test_out_of_range_scores_are_clamped(policy):
    s = evaluation.score(ScriptedLLM([_all(14), _all(-2), _all(5)]), policy, IDEA, ["x"], role="evaluator", samples=3)
    assert s.medians["novelty"] == 5 and s.spreads["novelty"] == 9      # 10, 1, 5


def test_a_dimension_no_sample_scored_is_null_not_invented(policy):
    partial = {k: 5 for k in list(_all(0))[1:]}                         # first dimension missing every time
    s = evaluation.score(ScriptedLLM([partial, partial]), policy, IDEA, ["x"], role="evaluator", samples=2)
    first = IDEA.dimensions[0].key
    assert s.medians[first] is None and s.spreads[first] is None and s.medians["novelty"] == 5


def test_the_rubric_and_its_anchors_are_in_what_the_model_reads(policy):
    llm = ScriptedLLM([_all(5)])
    evaluation.score(llm, policy, IDEA, ["the idea text"], role="evaluator", samples=1)
    text = " ".join(b if isinstance(b, str) else b.get("text", "") for b in llm.calls[0]["content"])
    assert "the idea text" in text and IDEA.dimensions[0].anchors[9] in text


def test_double_check_sampling_is_deterministic():
    ids = [uuid.uuid4() for _ in range(2000)]
    assert [evaluation.checked(i, 0.1) for i in ids] == [evaluation.checked(i, 0.1) for i in ids]
    assert not any(evaluation.checked(i, 0) for i in ids) and all(evaluation.checked(i, 1) for i in ids)
    share = sum(evaluation.checked(i, 0.1) for i in ids) / len(ids)
    assert 0.05 <= share <= 0.15
