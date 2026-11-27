"""Model benchmarks on our own creative tasks (manyworlds.benchmarks): every candidate gets the same assignment,
answers are compared blind with the incumbent's, no judge judges its own family, spend stops at the cap, and the
trial lands in the ledger."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace as NS

from manyworlds import benchmarks


class BenchLLM:
    """Answers with the model's name; the judge prefers whichever answer came from glm (wherever it sits)."""

    def __init__(self, usd_per_call=0.001):
        self.calls, self.log, self.shadowing, self.usd = [], [], True, usd_per_call

    @property
    def spent(self):
        return round(sum(c.usd for c in self.calls), 6)

    @contextlib.contextmanager
    def tagged(self, agent, job_id):
        yield

    def _call(self, model, *, system, content, schema, effort, creator_id):
        self.calls.append(NS(usd=self.usd))
        self.log.append((model, schema.__name__))
        if schema is benchmarks.Piece:
            return benchmarks.Piece(text=f"answer by {model}")
        a = content.split("=== Answer A ===")[1].split("=== Answer B ===")[0]
        return benchmarks.Verdict(reason="sounds more like her", better="A" if "glm" in a else "B")


def _creator(conn):
    return conn.execute("INSERT INTO creators (handle, bible) VALUES ('bench1', '{\"name\": \"Maren\"}') "
                        "RETURNING id").fetchone()["id"]


def test_candidates_are_judged_blind_by_other_families_and_recorded(conn, policy):
    assert set(benchmarks.load()["scenarios"]) >= {"bartender_monologue", "embodiment_low", "house_tour_plan"}
    llm = BenchLLM()
    out = benchmarks.run(conn, policy, llm, creator_id=_creator(conn), scenarios=["bartender_monologue"],
                         models=["glm-5.3"], baseline="claude-sonnet-5-5",
                         judges=["claude-opus-5-5", "glm-5.3", "qwen4-max"], samples=2, usd=1.0)
    assert out["judges"] == {"glm-5.3": ["qwen4-max"]}           # neither claude nor glm judges claude vs glm
    s = out["summary"]["glm-5.3"]
    assert s["judgements"] == 2 and s["wins"] == 2 and s["losses"] == 0   # the order flip didn't fool the tally
    assert [m for m, sch in llm.log if sch == "Verdict"] == ["qwen4-max", "qwen4-max"]
    assert llm.shadowing is True                                 # restored after the run
    row = conn.execute("SELECT method, result FROM analysis_runs WHERE method = 'model_bench'").fetchone()
    assert row and len(row["result"]["answers"]) == 4


def test_the_cap_stops_spending_and_a_dry_run_spends_nothing(conn, policy):
    llm = BenchLLM(usd_per_call=0.5)
    plan = benchmarks.run(conn, policy, llm, creator_id=_creator(conn), scenarios=None, models=["glm-5.3"],
                          baseline="claude-sonnet-5-5", judges=["qwen4-max"], samples=2, usd=1.0, dry_run=True)
    assert llm.calls == [] and plan["estimate_usd"] > 0 and len(plan["scenarios"]) >= 7
    benchmarks.run(conn, policy, llm, creator_id=_creator(conn), scenarios=None, models=["glm-5.3"],
                   baseline="claude-sonnet-5-5", judges=["qwen4-max"], samples=2, usd=1.0)
    assert llm.spent <= 1.0 + 0.5                                # stops at the first call past the cap
