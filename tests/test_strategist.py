"""The strategist (Brain M8): proposes stage moves and launches from the portfolio's numbers; code checks every
proposal against policy (evidence windows, caps, pacing, budget, approval) and applies only what passes."""

from __future__ import annotations

import dataclasses
import inspect

from psycopg.types.json import Jsonb

from manyworlds import queue
from manyworlds.brain import portfolio
from manyworlds.brain.agents import strategist

from brain_helpers import StubLLM, creator, make_ctx


def _theses(conn, n=3):
    return [conn.execute("INSERT INTO theses (summary, status, body, score) VALUES (%s, 'proposed', %s, %s) "
                         "RETURNING id", (f"thesis {i}", Jsonb({"summary": f"thesis {i}"}), 10 - i)).fetchone()["id"]
            for i in range(n)]


class PlanLLM(StubLLM):
    def __init__(self, plan, conn=None):
        super().__init__(conn=conn)
        self.plan = plan

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema)
        return schema.model_validate(self.plan)


def _run(conn, policy, plan):
    job = queue.enqueue(conn, "brain.strategist", {"day": "2026-11-20"})
    ctx = make_ctx(conn, policy, job)
    ctx.llm = PlanLLM(plan, conn=conn)
    return strategist.Strategist().run(ctx), ctx.llm


def _with(policy, **kw):
    s = {**policy.body["brain"]["strategist"], **kw}
    return dataclasses.replace(policy, body={**policy.body, "brain": {**policy.body["brain"], "strategist": s}})


def test_launch_proposals_wait_for_the_owner_while_policy_says_so(conn, policy):
    t = _theses(conn)
    policy = _with(policy, launches_per_week=2)          # the shipped policy allows none (owner 2026-11-21)
    out, llm = _run(conn, policy, {"stage_moves": [], "notes": "n",
                                   "launches": [{"thesis_id": str(t[0]), "cohort": "nov-a", "reason": "untested niche"}]})
    a = conn.execute("SELECT kind, status, subject FROM approvals").fetchall()
    assert [(x["kind"], x["status"]) for x in a] == [("launch", "pending")] and a[0]["subject"]["thesis_id"] == str(t[0])
    assert out["launches"]["requested"] == 1
    assert conn.execute("SELECT status FROM theses WHERE id = %s", (t[0],)).fetchone()["status"] == "chosen"
    assert "thesis 0" in str(llm.calls[0]["content"])                     # it saw the candidates


def test_launches_are_capped_by_creators_pacing_and_budget(conn, policy):
    t = _theses(conn, 4)
    pol = _with(policy, launch_requires_approval=False, launches_per_week=2)
    out, _ = _run(conn, pol, {"stage_moves": [], "notes": "n",
                              "launches": [{"thesis_id": str(x), "cohort": "c", "reason": "r"} for x in t]})
    assert out["launches"]["approved"] == 2 and len(out["launches"]["refused"]) == 2
    assert "per week" in out["launches"]["refused"][0]["why"]
    rows = conn.execute("SELECT status, decided_by FROM approvals").fetchall()
    assert {(r["status"], r["decided_by"]) for r in rows} == {("approved", "strategist")}   # within its authority


def test_an_unknown_or_used_thesis_is_refused(conn, policy):
    t = _theses(conn, 1)
    conn.execute("UPDATE theses SET status = 'launched' WHERE id = %s", (t[0],))
    out, _ = _run(conn, policy, {"stage_moves": [], "notes": "n",
                                 "launches": [{"thesis_id": str(t[0]), "cohort": "c", "reason": "r"},
                                              {"thesis_id": "00000000-0000-0000-0000-000000000000", "cohort": "c",
                                               "reason": "r"}]})
    assert out["launches"]["requested"] == 0 and len(out["launches"]["refused"]) == 2


def test_no_downward_move_before_the_evidence_window(conn, policy):
    cid = creator(conn, publish_mode="instagram")
    out, _ = _run(conn, policy, {"launches": [], "notes": "n",
                                 "stage_moves": [{"creator_id": str(cid), "to": "paused", "reason": "slow start"}]})
    assert out["stage_moves"]["applied"] == [] and "evidence window" in out["stage_moves"]["refused"][0]["why"]
    up, _ = _run(conn, policy, {"launches": [], "notes": "n",
                                "stage_moves": [{"creator_id": str(cid), "to": "iterating", "reason": "new angle"}]})
    assert [m["to"] for m in up["stage_moves"]["applied"]] == ["iterating"]
    st = conn.execute("SELECT stage, history FROM creator_states WHERE creator_id = %s", (cid,)).fetchone()
    assert st["stage"] == "iterating" and st["history"][-1]["reason"] == "new angle"


def test_retirement_needs_the_expected_value_case_and_owner_approval(conn, policy):
    cid = creator(conn, publish_mode="instagram")
    pol = _with(policy, min_posts_before_downward=0, min_days_before_downward=0, min_adjustments_before_retire=0)
    out, _ = _run(conn, pol, {"launches": [], "notes": "n",
                              "stage_moves": [{"creator_id": str(cid), "to": "retired", "reason": "flat"}]})
    # with only the prior (no real evidence) continuing is not worse than the alternative: refused
    assert out["stage_moves"]["applied"] == [] and "expected value" in out["stage_moves"]["refused"][0]["why"]


def test_the_portfolio_reads_real_evidence_only(conn, policy):
    src = inspect.getsource(portfolio) + inspect.getsource(strategist)
    assert "mode=\"sandbox\"" not in src and "mode='sandbox'" not in src and "simulated_" not in src
    cid = creator(conn, publish_mode="simulated")
    snap = portfolio.snapshot(conn, policy)
    me = next(s for s in snap["creators"] if s["creator_id"] == str(cid))
    assert me["real_posts"] == 0 and me["conviction"]["source"] == "prior"


def test_every_run_records_its_numbers_and_choices(conn, policy):
    _theses(conn, 1)
    out, _ = _run(conn, policy, {"stage_moves": [], "launches": [], "notes": "nothing to do yet"})
    d = conn.execute("SELECT kind, inputs, output, rationale FROM decisions WHERE kind = 'portfolio'").fetchone()
    assert d["rationale"] == "nothing to do yet" and "creators" in d["inputs"] and "budget" in d["inputs"]


def test_one_strategist_run_per_day(conn, policy):
    from datetime import datetime, timezone
    portfolio.ensure_daily(conn, policy, now=datetime(2026, 11, 20, 9, tzinfo=timezone.utc))
    portfolio.ensure_daily(conn, policy, now=datetime(2026, 11, 20, 15, tzinfo=timezone.utc))
    jobs = conn.execute("SELECT payload FROM jobs WHERE kind = 'brain.strategist'").fetchall()
    assert [j["payload"] for j in jobs] == [{"day": "2026-11-20"}]


def test_the_shipped_policy_launches_nothing(conn, raw_policy):
    t = _theses(conn, 1)
    pol = __import__("manyworlds.policy", fromlist=["from_dict"]).from_dict(
        {**raw_policy.body, "operations": {**raw_policy.body["operations"], "mode": "live"}})
    out, _ = _run(conn, pol, {"stage_moves": [], "notes": "n",
                              "launches": [{"thesis_id": str(t[0]), "cohort": "c", "reason": "r"}]})
    assert out["launches"]["requested"] == 0 and "per week" in out["launches"]["refused"][0]["why"]
