"""The Brain tunes its own settings within its authority; anything beyond waits for a human, and
it never believes it changed something no service reads."""

from __future__ import annotations

import pytest

from manyworlds import queue
from manyworlds.brain import overrides, runtime

from brain_helpers import StubLLM

PATH = "brain.exploration_share"          # read by Brain agents; bounds 0.1-0.6 in policy


def test_an_override_within_authority_applies(conn, policy):
    r = overrides.propose(conn, policy, PATH, 0.5, author="strategist", reason="r")
    eff = overrides.effective_policy(conn, policy)
    assert r["applied"] and r["version"] == 1
    assert eff.get("brain", "exploration_share") == 0.5 and eff.version == f"{policy.version}+o1"


def test_without_overrides_the_file_policy_stands(conn, policy):
    eff = overrides.effective_policy(conn, policy)
    assert eff.body == policy.body and eff.version == f"{policy.version}+o0"


def test_the_bound_itself_is_allowed(conn, policy):
    assert overrides.propose(conn, policy, PATH, 0.6, author="a", reason="r")["applied"]


def test_beyond_authority_waits_for_approval_and_changes_nothing(conn, policy):
    r = overrides.propose(conn, policy, PATH, 0.7, author="a", reason="r")
    assert not r["applied"]
    assert overrides.effective_policy(conn, policy).get("brain", "exploration_share") == \
        policy.get("brain", "exploration_share")
    pending = conn.execute("SELECT kind, status, requested_by FROM approvals WHERE id = %s",
                           (r["approval_id"],)).fetchone()
    assert (pending["kind"], pending["status"], pending["requested_by"]) == ("policy_override", "pending", "a")
    overrides.decide(conn, r["approval_id"], approved=True, by="owner")
    assert overrides.effective_policy(conn, policy).get("brain", "exploration_share") == 0.7


def test_a_rejected_approval_changes_nothing(conn, policy):
    r = overrides.propose(conn, policy, PATH, 0.7, author="a", reason="r")
    overrides.decide(conn, r["approval_id"], approved=False, by="owner")
    assert overrides.effective_policy(conn, policy).get("brain", "exploration_share") == \
        policy.get("brain", "exploration_share")


def test_an_approval_that_fails_to_apply_stays_pending(conn, policy, monkeypatch):
    r = overrides.propose(conn, policy, PATH, 0.7, author="a", reason="r")

    def broken(*a, **k):
        raise RuntimeError("apply failed")
    monkeypatch.setattr(overrides, "_apply", broken)
    with pytest.raises(RuntimeError):
        overrides.decide(conn, r["approval_id"], approved=True, by="owner")
    assert conn.execute("SELECT status FROM approvals WHERE id = %s", (r["approval_id"],)).fetchone()["status"] == \
        "pending"


def test_settings_no_service_reads_yet_are_refused_not_pretended(conn, policy):
    # The producer, reviewer and autoscaler read the policy file; an override there would be recorded
    # but not obeyed. Until they read the effective policy, such changes are refused outright.
    r = overrides.propose(conn, policy, "production.candidates_per_scene", 6, author="strategist", reason="r")
    assert r == {"applied": False, "not_observed": True,
                 "reason": "production.* is not read by any service through the effective policy yet"}
    assert conn.execute("SELECT count(*) n FROM policy_overrides").fetchone()["n"] == 0
    assert conn.execute("SELECT count(*) n FROM approvals").fetchone()["n"] == 0


def test_limits_and_the_master_budget_are_never_changed_by_the_brain(conn, policy):
    assert not overrides.propose(conn, policy, "limits.daily_usd_max", 1, author="a", reason="r")["applied"]
    assert not overrides.propose(conn, policy, "budget.master_usd_per_month", 1, author="a", reason="r")["applied"]


def test_a_glob_covers_one_path_segment(conn, policy):
    assert overrides.propose(conn, policy, "brain.llm_usd_per_day.showrunner", 4, author="a", reason="r")["applied"]
    assert not overrides.propose(conn, policy, "brain.llm_usd_per_day.showrunner.x", 4, author="a",
                                 reason="r")["applied"]


def test_agents_run_under_the_effective_policy_and_decisions_record_it(conn, policy):
    overrides.propose(conn, policy, PATH, 0.5, author="a", reason="r")
    job = queue.enqueue(conn, "brain.echo", {})
    runtime.tick(conn, policy, StubLLM(conn=conn), worker_id="w1")
    d = conn.execute("SELECT policy_version, override_version FROM decisions WHERE job_id = %s", (job,)).fetchone()
    assert d["override_version"] == 1 and d["policy_version"].endswith("+o1")


def test_agent_budgets_follow_the_effective_policy(conn, policy):
    overrides.propose(conn, policy, "brain.llm_usd_per_day.echo", 0, author="strategist", reason="pause echo")
    queue.enqueue(conn, "brain.echo", {})
    out = runtime.tick(conn, policy, StubLLM(conn=conn), worker_id="w1")
    assert out[0]["status"] == "failed" and "budget" in out[0]["error"]
