"""Starting new work vs finishing work already started (owner, 2026-11-25: the fully edited Maren video stopped at its
voice lines a few dollars from done, every photo already paid for). manyworlds.budget."""

from __future__ import annotations

import pytest
from psycopg.types.json import Jsonb

from manyworlds import budget, ledger, queue
from manyworlds.deploy import autoscale

from test_autoscale import T0, providers, registry  # noqa: F401 - registry: autouse image registry stub
from test_production import creator


def _at_cap(conn, policy, extra=0.0):
    ledger.record(conn, category="llm", provider="claude-opus-5-5", quantity=1, unit="token",
                  usd=float(policy.get("budgets", "daily_usd")) + extra)


def _started_post(conn, *, status="producing", done=True):
    cid = creator(conn)
    pid = conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, 'talking_video', %s, %s) "
                       "RETURNING id", (cid, status, Jsonb({}))).fetchone()["id"]
    if done:
        j = queue.enqueue(conn, "fake.image", {}, batch="bench-p", post_id=pid, creator_id=cid)
        conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (j,))
    return pid, cid


def _launched(ps):
    return [m for p in ps.values() for m in p.launched]


def test_new_work_waits_at_the_cap(conn, policy):
    _at_cap(conn, policy)
    queue.enqueue(conn, "train.lora", {}, batch="bench-x")
    ps = providers()
    actions = autoscale.tick(conn, policy, ps, now=T0)
    assert not _launched(ps) and any(a.get("reason") == "daily_total_budget" for a in actions)


def test_a_post_already_in_production_is_finished_past_the_cap_within_the_headroom(conn, policy):
    pid, cid = _started_post(conn)
    queue.enqueue(conn, "train.lora", {}, batch="bench-p", post_id=pid, creator_id=cid)
    _at_cap(conn, policy)
    ps = providers()
    actions = autoscale.tick(conn, policy, ps, now=T0)
    assert _launched(ps)
    assert any(a.get("action") == "finishing_started_posts" for a in actions)
    assert conn.execute("SELECT count(*) n FROM events WHERE type = 'budget_completion'").fetchone()["n"] == 1


@pytest.mark.parametrize("status,done", [("planned", False), ("producing", False)])
def test_a_post_with_nothing_finished_yet_is_new_work(conn, policy, status, done):
    pid, cid = _started_post(conn, status=status, done=done)
    queue.enqueue(conn, "train.lora", {}, batch="bench-p", post_id=pid, creator_id=cid)
    _at_cap(conn, policy)
    ps = providers()
    autoscale.tick(conn, policy, ps, now=T0)
    assert not _launched(ps)


def test_the_headroom_is_a_hard_ceiling(conn, policy):
    pid, cid = _started_post(conn)
    queue.enqueue(conn, "train.lora", {}, batch="bench-p", post_id=pid, creator_id=cid)
    _at_cap(conn, policy, extra=float(policy.get("budgets", "completion_usd")))
    ps = providers()
    autoscale.tick(conn, policy, ps, now=T0)
    assert not _launched(ps) and not budget.completion_open(conn, policy)


def test_an_owner_grant_raises_todays_caps_only(conn, policy):
    _at_cap(conn, policy)
    queue.enqueue(conn, "train.lora", {}, batch="bench-x")
    assert budget.over(conn, policy) == "daily_total_budget"
    out = budget.grant(conn, 5, "finish the Maren video")
    assert out["granted_today"] == 5
    assert budget.caps(conn, policy)["total"] == float(policy.get("budgets", "daily_usd")) + 5
    assert budget.over(conn, policy) is None
    conn.execute("UPDATE budget_grants SET day = day - 1")          # yesterday's grant: today is back to the policy
    assert budget.over(conn, policy) == "daily_total_budget"
    with pytest.raises(ValueError):
        budget.grant(conn, 0, "nothing")
    with pytest.raises(ValueError):
        budget.grant(conn, 3, "  ")
