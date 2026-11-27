"""Model benchmarks on our own creative tasks (manyworlds.benchmarks): every candidate gets the same assignment;
every judge family (Claude included) compares blind; independent samples are counted apart from judgements; a
judge that favours its own family is measured; the longer-answer effect is reported; every call is reserved
against the cap, so the cap holds; trials pool into per-task evidence."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace as NS

from manyworlds import benchmarks


class BenchLLM:
    """glm writes a longer answer. Judges: claude picks whichever answer is claude's (self-preference); glm and qwen
    pick glm's."""

    def __init__(self, usd_per_call=0.001):
        self.calls, self.log, self.shadowing, self.usd = [], [], True, usd_per_call

    @property
    def spent(self):
        return round(sum(c.usd for c in self.calls), 6)

    @contextlib.contextmanager
    def tagged(self, agent, job_id):
        yield

    def _call(self, model, *, system, content, schema, effort, creator_id):
        self.calls.append(NS(usd=self.usd, output_tokens=100))
        self.log.append((model, schema.__name__))
        if schema is benchmarks.Piece:
            return benchmarks.Piece(text="a much longer answer by glm here" if "glm" in model else "short by claude")
        a = content.split("=== Answer A ===")[1].split("=== Answer B ===")[0]
        own = "glm" not in a if model.startswith("claude") else "glm" in a
        return benchmarks.Verdict(reason="r", better="A" if own else "B",
                                  a=benchmarks.Scores(originality=4), b=benchmarks.Scores(originality=2))


def _creator(conn):
    return conn.execute("INSERT INTO creators (handle, bible) VALUES ('bench1', '{\"name\": \"Maren\"}') "
                        "RETURNING id").fetchone()["id"]


def test_all_families_judge_and_self_preference_independent_samples_and_length_are_reported(conn, policy):
    assert set(benchmarks.load()["scenarios"]) >= {"closing_time_monologue", "embodiment_low", "house_tour_plan"}
    llm = BenchLLM()
    out = benchmarks.run(conn, policy, llm, creator_id=_creator(conn), scenarios=["closing_time_monologue"],
                         models=["glm-5.3"], baseline="claude-sonnet-5-5",
                         judges=["claude-opus-5-5", "glm-5.3", "qwen4-max"], samples=2, usd=1.0)
    assert out["neutral_judges"] == {"glm-5.3": ["qwen4-max"]} and out["under_judged"] == ["glm-5.3"]
    m = out["models"]["glm-5.3"]
    assert m["judgements"] == 6                                      # every judge on every pair (2 samples x 3)
    assert m["samples"] == {"candidate": 2, "baseline": 0, "tie": 0}  # two independent samples, majority each
    assert m["samples_neutral_judges"] == {"candidate": 2, "baseline": 0, "tie": 0}
    sp = out["self_preference"]
    assert sp["claude"]["self_preference"] == 1.0                    # claude always picks claude; qwen never does
    assert sp["glm"]["self_preference"] == 0.0                       # glm picks glm exactly as often as qwen does
    assert out["length"]["longer_answer_won"] is not None
    assert m["dimensions"]["originality"] in (2.0, 3.0, 4.0) and llm.shadowing is True
    assert out["untested_scenarios"] == [] and out["stopped"] is None
    assert set(out["cost"]) == {"estimated_usd", "committed_usd", "actual_usd"}
    ev = benchmarks.evidence(conn, policy)
    assert ev["models"]["glm-5.3 vs claude-sonnet-5-5"]["dialogue:closing_time_monologue"] == {"candidate": 2,
                                                                                              "baseline": 0, "tie": 0}


def test_the_cap_holds_with_reservations_and_unjudged_scenarios_are_named(conn, policy):
    llm = BenchLLM(usd_per_call=0.5)
    me = _creator(conn)
    plan = benchmarks.run(conn, policy, llm, creator_id=me, scenarios=None, models=["glm-5.3"],
                          baseline="claude-sonnet-5-5", judges=["qwen4-max"], samples=2, usd=1.0, dry_run=True)
    assert llm.calls == [] and plan["estimate_usd"] > 0 and len(plan["scenarios"]) >= 7
    out = benchmarks.run(conn, policy, llm, creator_id=me, scenarios=None, models=["glm-5.3"],
                         baseline="claude-sonnet-5-5", judges=["qwen4-max"], samples=2, usd=1.0)
    assert llm.spent <= 1.0 and out["stopped"].startswith("cap")     # never over the cap
    assert "narrative_scene" in out["untested_scenarios"]            # named, not silently missing
