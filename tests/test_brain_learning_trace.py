"""The closed loop on a sandbox creator, end to end with stubbed models: analysis → plan → scores →
schedule → simulated publication → outcomes → analysis that the next plan cites."""

from __future__ import annotations

import random

import pytest
from psycopg.types.json import Jsonb

from manyworlds.brain import sandbox
from manyworlds.brain.agents import analyst, showrunner

from brain_helpers import StubLLM, creator


class LoopLLM(StubLLM):
    """Plays every model in the loop. Plans favour high personality once the analyst has said so."""

    def __init__(self, conn):
        super().__init__(conn=conn)
        self.rng, self.batch = random.Random(7), 0

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema)
        if schema is analyst.Analysis:
            return schema.model_validate({
                "hypotheses": [{"statement": "Personality-led posts may reach more people.", "direction": "positive",
                                "supports": ["dimension:personality"], "how_to_confirm": "more of them"}],
                "guidance": {"more_of": ["LEARNED-personality"], "less_of": [], "explore": ["new places"],
                             "notes": "modest n"}})
        if schema is showrunner.IdeaBatch:
            self.batch += 1
            led = "LEARNED-personality" in str(content)          # only a real (non-insufficient) analysis says this
            ideas = [{"day": i % 3, "platform": "instagram", "format": "carousel",
                      "scenes": [f"scene {self.batch}-{i}", f"scene {self.batch}-{i}b"], "caption": f"c{self.batch}{i}", "hashtags": [], "canon": "",
                      "attributes": {"lead": "personality" if led else "place"},
                      "reasoning": "finding: personality" if led else "explore"} for i in range(9)]
            return schema.model_validate({"rationale": "r", "ideas": ideas})
        # evaluator: personality-led ideas score high on personality; the rest spread over the whole scale, so the
        # planted effect has range to show in (3-6 hid it in some random histories)
        led = '"lead": "personality"' in str(content) or "personality" in str(content).split("THE IDEA")[-1]
        dims = {k: {"score": (8 if (k == "personality" and led) else self.rng.randint(1, 9)), "why": "w"}
                for k in schema.model_fields if k != "overall_note"}
        return schema.model_validate({**dims, "overall_note": "n"})


def _sim_creator(conn):
    cid = creator(conn, publish_mode="simulated")
    conn.execute("UPDATE creators SET status = 'active', profile_version = NULL, bible = %s, anchors = %s WHERE id = %s",
                 (Jsonb({"display_name": "Sim"}), Jsonb({"simulation": {"scenario": "personality_wins", "seed": 3}}), cid))
    return cid


def test_sandbox_runs_refuse_real_creators(conn, policy):
    with pytest.raises(ValueError, match="sandbox"):
        sandbox.run_batches(conn, policy, StubLLM(conn=conn), creator(conn, publish_mode="instagram"), batches=1,
                            worker_id="t")


def test_the_loop_closes_and_the_trace_shows_what_changed(conn, policy):
    cid = _sim_creator(conn)
    llm = LoopLLM(conn)
    runs = sandbox.run_batches(conn, policy, llm, cid, batches=4, worker_id="t")
    assert [r["published"] for r in runs] == [9, 9, 9, 9]
    assert runs[0]["analysis"]["insufficient"] is True                  # first batch: nothing to learn from yet
    trace = sandbox.learning_trace(conn, cid)
    assert trace["label"].startswith("SIMULATED")
    last = trace["batches"][-1]
    assert last["analysis_decision_id"] and last["finding_ids"]          # the plan cites the analysis it used
    assert trace["findings"][0]["statement"].startswith("Personality-led posts")
    assert last["composition"]["attributes"]["lead=personality"] > trace["batches"][0]["composition"]["attributes"].get(
        "lead=personality", 0)


def test_only_a_real_finding_changes_the_plan(conn, policy):
    # Batches planned on "insufficient evidence" stay exploratory; the first batch after an analysis
    # with findings follows it, and the trace links that batch to those findings.
    cid = _sim_creator(conn)
    runs = sandbox.run_batches(conn, policy, LoopLLM(conn), cid, batches=4, worker_id="t")
    trace = sandbox.learning_trace(conn, cid)
    by_analysis = {a["id"]: a for a in trace["analyses"]}
    for b in trace["batches"]:
        a = by_analysis.get(b["analysis_decision_id"])
        led = b["composition"]["attributes"].get("lead=personality", 0)
        if a is None or a["insufficient"]:
            assert led == 0 and b["finding_ids"] == []
        else:
            # follows the analysis it was given (its findings may be empty when a hypothesis contradicted
            # the numbers and was rejected, which a random history can produce)
            assert led == 1.0 and b["analysis_decision_id"] == a["id"]
    assert any(not r["analysis"]["insufficient"] for r in runs)


def test_sandbox_runs_refuse_creators_without_a_scenario_or_with_reference_images(conn, policy):
    no_scenario = creator(conn, publish_mode="simulated")
    with pytest.raises(ValueError, match="scenario"):
        sandbox.run_batches(conn, policy, StubLLM(conn=conn), no_scenario, batches=1, worker_id="t")
    rendered = _sim_creator(conn)
    conn.execute("UPDATE creators SET anchors = anchors || %s WHERE id = %s",
                 (Jsonb({"reference_images": [{"uri": "s3://r"}]}), rendered))
    with pytest.raises(ValueError, match="reference images"):
        sandbox.run_batches(conn, policy, StubLLM(conn=conn), rendered, batches=1, worker_id="t")


def test_a_sandbox_run_only_runs_its_own_jobs_on_its_own_budget(conn, policy):
    from manyworlds import ledger, queue
    from manyworlds.brain import budget
    other = queue.enqueue(conn, "brain.echo", {}, creator_id=creator(conn))
    cid = _sim_creator(conn)
    sandbox.run_batches(conn, policy, LoopLLM(conn), cid, batches=1, worker_id="t")
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (other,)).fetchone()["status"] == "queued"
    agents = {r["agent"] for r in conn.execute("SELECT DISTINCT agent FROM cost_entries WHERE agent IS NOT NULL").fetchall()}
    assert agents and all(a.startswith("sandbox:") for a in agents)
    ledger.record(conn, category="llm", provider="m", quantity=1, unit="token", usd=999, agent="sandbox:showrunner")
    budget.check(conn, policy, "showrunner")
