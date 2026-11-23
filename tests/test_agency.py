"""The agency's connective tissue (docs/superpowers/plans/2026-11-23-agency.md): requests with one owner and rights
checked in code, tiers the CEO sets, predictions scored against real outcomes, a market lead activated when a market
warrants one, the strategist deciding requests, and the chief of staff keeping it all moving."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds import queue
from manyworlds.brain import chief_of_staff, predictions, proposals
from manyworlds.brain.agents import market_lead, strategist

from brain_helpers import StubLLM, creator, make_ctx

NOW = datetime(2026, 11, 23, 9, tzinfo=timezone.utc)


def _with(policy, path, value):
    body = json.loads(json.dumps(policy.body))
    node = body
    for k in path[:-1]:
        node = node.setdefault(k, {})
    node[path[-1]] = value
    return dataclasses.replace(policy, body=body)


# ---------------------------------------------------------------------------------------------------- requests

def test_a_request_goes_to_its_owner_once_and_a_rejection_needs_new_evidence(conn, policy):
    cid = creator(conn)
    r = proposals.file(conn, policy, kind="tier_change", from_role="talent_manager", creator_id=cid,
                       subject={"tier": "standard"}, argument="video works", now=NOW)
    assert r["to_role"] == "strategist"
    again = proposals.file(conn, policy, kind="tier_change", from_role="talent_manager", creator_id=cid,
                           subject={"tier": "standard"}, argument="again", now=NOW)
    assert "already open" in again["refused"]
    out = proposals.decide(conn, policy, r["id"], decided_by="strategist", accept=False, reason="too early", now=NOW)
    assert out["status"] == "rejected"
    soon = proposals.file(conn, policy, kind="tier_change", from_role="talent_manager", creator_id=cid,
                          subject={"tier": "standard"}, argument="please", now=NOW + timedelta(days=1))
    assert "rejected recently" in soon["refused"]
    later = proposals.file(conn, policy, kind="tier_change", from_role="talent_manager", creator_id=cid,
                           subject={"tier": "standard"}, argument="weeks later",
                           now=NOW + timedelta(days=policy.get("brain", "rights", "cooldown_days") + 1))
    assert later.get("id")
    assert proposals.open_for(conn, "strategist")[0]["label"].startswith("SIMULATED")   # a sandbox creator's request


def test_only_the_owner_role_decides_and_an_accepted_tier_is_applied_within_budget(conn, policy):
    cid = creator(conn)
    r = proposals.file(conn, policy, kind="tier_change", from_role="talent_manager", creator_id=cid,
                       subject={"tier": "standard"}, argument="video works", now=NOW)
    assert "does not decide" in proposals.decide(conn, policy, r["id"], decided_by="analyst", accept=True,
                                                 reason="x")["refused"]
    out = proposals.decide(conn, policy, r["id"], decided_by="strategist", accept=True, reason="earned", now=NOW)
    assert out["status"] == "accepted" and proposals.tier_of(conn, policy, cid) == "standard"
    broke = _with(policy, ("budget", "master_usd_per_month"), 0)
    other = creator(conn)                                  # one tier request a week per creator (the allowance)
    r2 = proposals.file(conn, broke, kind="tier_change", from_role="talent_manager", creator_id=other,
                        subject={"tier": "premium"}, argument="more", now=NOW)
    out = proposals.decide(conn, broke, r2["id"], decided_by="strategist", accept=True, reason="yes", now=NOW)
    assert out["status"] == "open" and "left" in out["refused"]                    # money: up to the owner
    assert conn.execute("SELECT to_role FROM proposals WHERE id = %s", (r2["id"],)).fetchone()["to_role"] == "owner"


def test_a_budget_over_the_ceos_rights_goes_to_the_owner(conn, policy):
    cid = creator(conn)
    cap = policy.get("brain", "rights", "strategist", "max_usd_per_request")
    r = proposals.file(conn, policy, kind="budget", from_role="talent_manager", creator_id=cid,
                       subject={"usd": cap * 3}, argument="premium video month", now=NOW)
    out = proposals.decide(conn, policy, r["id"], decided_by="strategist", accept=True, reason="worth it", now=NOW)
    assert out.get("escalated") and conn.execute("SELECT count(*) n FROM events WHERE type = 'request_for_owner'"
                                                 ).fetchone()["n"] == 1


def test_research_runs_at_once_within_the_allowance(conn, policy):
    cid = creator(conn)
    allowance = policy.get("brain", "rights", "talent_manager", "research_per_week")
    got = [proposals.file(conn, policy, kind="research", from_role="talent_manager", creator_id=cid,
                          subject={"question": f"why q{i}"}, argument="stalled", now=NOW) for i in range(allowance + 1)]
    assert all(g.get("status") == "accepted" for g in got[:allowance]) and "a week" in got[-1]["refused"]
    jobs = conn.execute("SELECT payload FROM jobs WHERE kind = 'brain.analyst'").fetchall()
    assert len(jobs) == allowance and jobs[0]["payload"]["question"] == "why q0"


def test_tiers_cap_the_plan(policy):
    d = proposals.tier_dials(policy, "economy", {"posts_per_day": {"instagram": 5}, "images_per_video": 1})
    eco = policy.get("brain", "tiers", "levels", "economy")
    assert d["posts_per_day"]["instagram"] == eco["posts_per_day_max"]
    assert d["images_per_video"] >= (1 - eco["video_share_max"]) / eco["video_share_max"] - 0.01


# ------------------------------------------------------------------------------------------------- predictions

def _published(conn, cid, reach, at):
    pid = conn.execute("INSERT INTO posts (creator_id, format, status) VALUES (%s, 'photo', 'ready') RETURNING id",
                       (cid,)).fetchone()["id"]
    conn.execute("INSERT INTO publications (post_id, platform, simulated, published_at) VALUES (%s, 'instagram', false, %s)",
                 (pid, at))
    conn.execute("INSERT INTO metric_snapshots (post_id, metrics, post_age_s, captured_at) VALUES (%s, %s, 259200, %s)",
                 (pid, Jsonb({"reach": reach}), at + timedelta(days=3)))


def test_predictions_are_scored_against_real_outcomes_and_make_a_scorecard(conn, policy):
    cid = creator(conn, publish_mode="instagram")
    sure = predictions.record(conn, role="talent_manager", creator_id=cid, claim="reach above 500",
                              metric="median_reach", comparator="above", threshold=500, probability=0.9,
                              horizon_days=14, now=NOW)
    wrong = predictions.record(conn, role="talent_manager", creator_id=cid, claim="reach below 100",
                               metric="median_reach", comparator="below", threshold=100, probability=0.8,
                               horizon_days=14, now=NOW)
    assert predictions.record(conn, role="talent_manager", creator_id=cid, claim="x", metric="vibes",
                              comparator="above", threshold=1, probability=0.5, horizon_days=7) is None
    for i, reach in enumerate((600, 800, 700)):
        _published(conn, cid, reach, NOW + timedelta(days=2 + i))
    assert predictions.resolve_due(conn, policy, now=NOW + timedelta(days=5))["resolved"] == 0   # not due yet
    assert predictions.resolve_due(conn, policy, now=NOW + timedelta(days=15))["resolved"] == 0  # window not mature
    out = predictions.resolve_due(conn, policy, now=NOW + timedelta(days=18))
    assert out["resolved"] == 2
    rows = {r["id"]: r for r in conn.execute("SELECT id, outcome, observed, brier FROM predictions").fetchall()}
    assert rows[sure]["outcome"] is True and float(rows[sure]["observed"]) == 700
    assert rows[wrong]["outcome"] is False and abs(float(rows[wrong]["brier"]) - 0.64) < 1e-6
    [card] = predictions.scorecard(conn)
    assert card["role"] == "talent_manager" and card["n"] == 2 and card["hit_rate"] == 0.5
    assert predictions.scorecard(conn, mode="sandbox") == []                       # real and sandbox never mix


# ------------------------------------------------------------------------------------------------- market lead

class LeadLLM(StubLLM):
    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema)
        return schema.model_validate({
            "thesis": "slow living is underserved at night", "audiences": ["night workers"], "working": ["rituals"],
            "gaps": ["men"], "conviction_mean": 0.5, "conviction_low": 0.3, "conviction_high": 0.7,
            "learnings": ["timestamps lift saves"],
            "cohort": [{"concept": "a night baker", "archetype": "food_home", "why_distinct": "food, not rituals"}],
            "requests": [{"kind": "cohort", "subject": json.dumps({"size": 3}), "argument": "untested gaps"}],
            "predictions": []})


def test_a_market_lead_is_activated_writes_the_thesis_and_proposes_a_cohort(conn, policy):
    m = conn.execute("INSERT INTO markets (name) VALUES ('slow living') RETURNING id").fetchone()["id"]
    need = policy.get("brain", "market_lead", "min_creators")
    for i in range(need - 1):
        conn.execute("INSERT INTO creators (handle, status, market_id) VALUES (%s, 'active', %s)", (f"m{i}", m))
    assert market_lead.ensure_reviews(conn, policy, now=NOW) == []                 # not enough creators yet
    conn.execute("INSERT INTO creators (handle, status, market_id) VALUES ('mlast', 'active', %s)", (m,))
    [job] = market_lead.ensure_reviews(conn, policy, now=NOW)
    ctx = make_ctx(conn, policy, job)
    ctx.llm = LeadLLM(conn=conn)
    out = market_lead.MarketLead().run(ctx)
    row = conn.execute("SELECT dossier, conviction_mean FROM markets WHERE id = %s", (m,)).fetchone()
    assert row["dossier"]["thesis"].startswith("slow living") and float(row["conviction_mean"]) == 0.5
    assert row["dossier"]["learnings"] == ["timestamps lift saves"]
    p = conn.execute("SELECT kind, to_role, subject FROM proposals").fetchone()
    assert (p["kind"], p["to_role"]) == ("cohort", "strategist") and p["subject"]["concepts"][0]["concept"] == "a night baker"
    assert out["requests"][0].get("id")


# --------------------------------------------------------------------------------------------------- strategist

class PlanLLM(StubLLM):
    def __init__(self, plan, conn=None):
        super().__init__(conn=conn)
        self.plan = plan

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema)
        return schema.model_validate(self.plan(content) if callable(self.plan) else self.plan)


def test_the_strategist_decides_every_request_addressed_to_it(conn, policy):
    cid = creator(conn)
    r = proposals.file(conn, policy, kind="tier_change", from_role="talent_manager", creator_id=cid,
                       subject={"tier": "standard"}, argument="video works")
    job = queue.enqueue(conn, "brain.strategist", {"day": "2026-11-23"})
    ctx = make_ctx(conn, policy, job)
    ctx.llm = PlanLLM({"stage_moves": [], "launches": [], "notes": "n",
                       "requests": [{"proposal_id": r["id"], "accept": True, "reason": "earned"},
                                    {"proposal_id": "00000000-0000-0000-0000-000000000000", "accept": True,
                                     "reason": "ghost"}]}, conn=conn)
    out = strategist.Strategist().run(ctx)
    assert "video works" in str(ctx.llm.calls[0]["content"])                       # it saw the request
    decided = out["requests"]
    assert decided[0]["status"] == "accepted" and "not an open request" in decided[1]["refused"]
    assert proposals.tier_of(conn, policy, cid) == "standard"


# ---------------------------------------------------------------------------------------------- chief of staff

def test_the_chief_of_staff_expires_stale_requests_and_reports_failing_agents(conn, policy):
    cid = creator(conn)
    proposals.file(conn, policy, kind="tier_change", from_role="talent_manager", creator_id=cid,
                   subject={"tier": "standard"}, argument="x", now=datetime.now(timezone.utc) - timedelta(days=10))
    for _ in range(policy.get("brain", "chief_of_staff", "failing_jobs")):
        j = queue.enqueue(conn, "brain.talent_manager", {"creator_id": str(cid)})
        conn.execute("UPDATE jobs SET status = 'failed', updated_at = now() WHERE id = %s", (j,))
    out = chief_of_staff.tick(conn, policy)
    assert out["expired_requests"] == 1 and out["failing_agents"] == 1
    types = {r["type"] for r in conn.execute("SELECT type FROM events").fetchall()}
    assert {"requests_expired", "agent_failing"} <= types


# ------------------------------------------------------------------------------------------- request effects

def test_requests_waiting_on_the_owner_never_block_a_creators_other_requests(conn, policy):
    cid = creator(conn)
    cap = int(policy.get("brain", "rights", "max_open_per_creator"))
    for i in range(cap + 1):
        proposals.file(conn, policy, kind="escalate", from_role="talent_manager", creator_id=cid,
                       subject={"ask": f"a set {i}"}, argument="repetition", now=NOW)
    r = proposals.file(conn, policy, kind="research", from_role="talent_manager", creator_id=cid,
                       subject={"question": "why"}, argument="stalled", now=NOW)
    assert r.get("status") == "accepted"


def test_an_escalated_tier_tells_the_owner_at_once(conn, policy):
    cid = creator(conn)
    broke = _with(policy, ("budget", "master_usd_per_month"), 0)
    r = proposals.file(conn, broke, kind="tier_change", from_role="talent_manager", creator_id=cid,
                       subject={"tier": "premium"}, argument="more", now=NOW)
    proposals.decide(conn, broke, r["id"], decided_by="strategist", accept=True, reason="yes", now=NOW)
    assert conn.execute("SELECT count(*) n FROM events WHERE type = 'request_for_owner'").fetchone()["n"] == 1


def test_a_rerun_job_files_its_request_once(conn, policy):
    cid = creator(conn)
    job = queue.enqueue(conn, "brain.talent_manager", {"creator_id": str(cid)})
    first = proposals.file(conn, policy, kind="research", from_role="talent_manager", creator_id=cid,
                           subject={"question": "why"}, argument="stalled", job_id=job, now=NOW)
    again = proposals.file(conn, policy, kind="research", from_role="talent_manager", creator_id=cid,
                           subject={"question": "why"}, argument="stalled", job_id=job, now=NOW)
    assert again["id"] == first["id"] and again["repeat"]
    assert conn.execute("SELECT count(*) n FROM jobs WHERE kind = 'brain.analyst'").fetchone()["n"] == 1


def test_a_reworded_tier_request_is_the_same_request(conn, policy):
    cid = creator(conn)
    proposals.file(conn, policy, kind="tier_change", from_role="talent_manager", creator_id=cid,
                   subject={"tier": "standard"}, argument="a", now=NOW)
    again = proposals.file(conn, policy, kind="tier_change", from_role="talent_manager", creator_id=cid,
                           subject={"tier": "standard", "why": "worded differently"}, argument="b", now=NOW)
    assert "already open" in again["refused"]


def test_tier_upgrades_in_one_run_share_the_months_budget(conn, policy):
    levels = policy.get("brain", "tiers", "levels")
    per = (float(levels["premium"]["usd_per_day"]) - float(levels["economy"]["usd_per_day"])) * 30
    from manyworlds.brain import portfolio
    spent = portfolio.budget(conn, policy, NOW)["spent_usd"]
    reserve = float(policy.get("budget", "reserve_share", default=0.15))
    room = _with(policy, ("budget", "master_usd_per_month"), (per * 1.5 + spent) / (1 - reserve))   # room for one
    got = []
    for _ in range(3):
        cid = creator(conn)
        r = proposals.file(conn, room, kind="tier_change", from_role="talent_manager", creator_id=cid,
                           subject={"tier": "premium"}, argument="earned", now=NOW)
        got.append(proposals.decide(conn, room, r["id"], decided_by="strategist", accept=True, reason="y", now=NOW)["status"])
    assert got == ["accepted", "open", "open"]                                      # the rest go to the owner


def test_an_accepted_experiment_opens_one_and_a_cohort_becomes_theses(conn, policy):
    cid = creator(conn)
    r = proposals.file(conn, policy, kind="experiment", from_role="talent_manager", creator_id=cid,
                       subject={"effect": "format:carousel", "hypothesis": "carousels earn saves"}, argument="x", now=NOW)
    assert r.get("status") == "accepted"
    exp = conn.execute("SELECT effect_key, status FROM experiments WHERE creator_id = %s", (cid,)).fetchone()
    assert exp["effect_key"] == "format:carousel" and exp["status"] == "open"
    bad = proposals.file(conn, policy, kind="experiment", from_role="talent_manager", creator_id=creator(conn),
                         subject={"hypothesis": "vibes"}, argument="x", now=NOW)
    assert bad.get("status") == "rejected"
    m = conn.execute("INSERT INTO markets (name) VALUES ('slow living') RETURNING id").fetchone()["id"]
    c = proposals.file(conn, policy, kind="cohort", from_role="market_lead", market_id=m, argument="a gap",
                       subject={"size": 2, "concepts": [{"concept": "a potter in Leeds", "archetype": "maker",
                                                         "why_distinct": "clay"}, "a night baker"]}, now=NOW)
    out = proposals.decide(conn, policy, c["id"], decided_by="strategist", accept=True, reason="go", now=NOW)
    assert out["status"] == "accepted" and len(out["applied"]["theses"]) == 2
    t = conn.execute("SELECT status, body FROM theses WHERE body->>'market_id' = %s", (str(m),)).fetchall()
    assert {x["status"] for x in t} == {"proposed"} and t[0]["body"]["archetype"] in ("maker", "")
