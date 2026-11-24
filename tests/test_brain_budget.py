"""Every agent's LLM spend is recorded under its name and stopped at its daily budget."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from manyworlds import ledger, queue
from manyworlds.brain import budget
from manyworlds.llm import Claude

from brain_helpers import creator


class Echo(BaseModel):
    say: str = "hi"


class FakeAnthropic:
    def __init__(self, input_tokens: int, output_tokens: int):
        usage = type("U", (), {"input_tokens": input_tokens, "output_tokens": output_tokens,
                               "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})

        class messages:
            @staticmethod
            def parse(**kw):
                return type("R", (), {"model": kw["model"], "usage": usage, "stop_reason": "end_turn",
                                      "parsed_output": kw["output_format"]()})
        self.messages = messages


def test_llm_cost_is_recorded_under_the_agent_and_job(conn, policy):
    llm = Claude(conn, policy, client=FakeAnthropic(input_tokens=1000, output_tokens=100))
    job = queue.enqueue(conn, "brain.echo", {})
    with llm.tagged("echo", job):
        llm.structured("brain", system="s", content="c", schema=Echo)
    llm.structured("brain", system="s", content="c", schema=Echo)        # outside the tag: no agent
    rows = conn.execute("SELECT agent, job_id, category FROM cost_entries ORDER BY id").fetchall()
    assert (rows[0]["agent"], rows[0]["job_id"], rows[0]["category"]) == ("echo", job, "llm")
    assert rows[1]["agent"] is None and rows[1]["job_id"] is None


def test_an_agent_over_its_daily_budget_is_stopped(conn, policy):
    cap = budget.remaining(conn, policy, "echo")
    ledger.record(conn, category="llm", provider="m", quantity=1, unit="token", usd=cap + 0.01, agent="echo")
    with pytest.raises(budget.OverBudget, match="echo"):
        budget.check(conn, policy, "echo")


def test_spend_by_other_agents_does_not_count(conn, policy):
    ledger.record(conn, category="llm", provider="m", quantity=1, unit="token", usd=999, agent="scouts")
    budget.check(conn, policy, "echo")
    assert budget.spent_today(conn, "echo") == 0


def test_agents_without_their_own_budget_use_the_default(conn, policy):
    assert budget.remaining(conn, policy, "echo") == float(policy.get("brain", "llm_usd_per_day", "default"))


def test_an_agent_that_has_spent_exactly_its_budget_is_stopped(conn, policy):
    ledger.record(conn, category="llm", provider="m", quantity=1, unit="token",
                  usd=budget.daily_cap(policy, "echo"), agent="echo")
    with pytest.raises(budget.OverBudget):
        budget.check(conn, policy, "echo")


def test_per_creator_agents_get_more_budget_as_the_portfolio_grows(conn, policy):
    base = budget.daily_cap(policy, "showrunner", conn)
    per = policy.get("brain", "llm_usd_per_creator_day", "showrunner")
    n0 = budget.active_creators(conn)
    for _ in range(3):
        creator(conn)
    assert budget.daily_cap(policy, "showrunner", conn) == pytest.approx(base + 3 * per)
    assert budget.active_creators(conn) == n0 + 3
    assert budget.daily_cap(policy, "strategist", conn) == policy.get("brain", "llm_usd_per_day", "strategist")


def test_calibrate_and_rubric_drift_are_refused_over_the_evaluators_budget(conn, policy, monkeypatch):
    from manyworlds import cli
    import manyworlds.llm as llm_mod

    class NoCalls:   # the refusal comes before any model is asked
        def __init__(self, *a, **kw):
            self.spent = 0.0

        def tagged(self, *a, **kw):
            raise AssertionError("no LLM call may be set up over budget")
    monkeypatch.setattr(llm_mod, "Claude", NoCalls)
    cap = budget.daily_cap(policy, "evaluator", conn)
    ledger.record(conn, category="llm", provider="m", quantity=1, unit="token", usd=cap + 0.01, agent="evaluator")
    for argv in (["brain", "calibrate", "someone"], ["brain", "rubric-drift", "idea"]):
        with pytest.raises(SystemExit, match="refused: agent evaluator is over its daily LLM budget"):
            cli._brain_ops(cli.build_parser().parse_args(argv), conn, policy)
