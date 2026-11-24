"""Research discovery on a cadence: queued from the Brain's cycle when due and operations are live, never while
paused, and once a period (a discovery by hand counts too). The scouts job runs it with Claude and Apify stubbed."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

from manyworlds.brain import discover as disc
from manyworlds.brain.agents import scouts
from manyworlds.research.apify import ApifySource

from brain_helpers import make_ctx
from test_day2 import IG_ITEM, TT_ITEM, FakeLLM

NOW = datetime.now(timezone.utc)


def _paused(policy):
    return dataclasses.replace(policy, body={**policy.body, "operations": {**policy.body["operations"], "mode": "paused"}})


def _jobs(conn):
    return conn.execute("SELECT id, payload FROM jobs WHERE kind = 'brain.scouts'").fetchall()


def test_discovery_is_queued_when_due_and_live(conn, policy):
    job = disc.ensure_due(conn, policy, now=NOW)
    assert job and [j["id"] for j in _jobs(conn)] == [job]


def test_discovery_never_runs_while_paused(conn, policy):
    assert disc.ensure_due(conn, _paused(policy), now=NOW) is None and not _jobs(conn)


def test_discovery_runs_once_a_period(conn, policy):
    every = policy.get("research", "discover", "every_days")
    assert disc.ensure_due(conn, policy, now=NOW)
    assert disc.ensure_due(conn, policy, now=NOW) is None
    assert disc.ensure_due(conn, policy, now=NOW + timedelta(days=every / 2)) is None
    assert len(_jobs(conn)) == 1
    assert disc.ensure_due(conn, policy, now=NOW + timedelta(days=every, hours=1))   # the next period
    assert len(_jobs(conn)) == 2


def test_a_discovery_by_hand_counts_for_the_period(conn, policy):
    conn.execute("INSERT INTO decisions (kind, inputs, output) VALUES ('research_plan', '{}', '{}')")
    assert disc.ensure_due(conn, policy, now=NOW) is None


def test_zero_days_means_by_hand_only(conn, policy):
    off = dataclasses.replace(policy, body={**policy.body, "research": {**policy.body["research"],
                                                                        "discover": {"every_days": 0}}})
    assert disc.ensure_due(conn, off, now=NOW) is None


def test_the_scouts_job_discovers_and_records_it(conn, policy, monkeypatch):
    def fetch(actor, payload):
        return [IG_ITEM] if "instagram" in actor else [TT_ITEM]

    monkeypatch.setattr(scouts.Scouts, "source", lambda self, ctx: ApifySource(ctx.conn, ctx.policy, fetch=fetch))
    job = disc.ensure_due(conn, policy, now=NOW)
    ctx = make_ctx(conn, policy, job)
    ctx.llm = FakeLLM()
    out = scouts.Scouts().run(ctx)
    assert len(out["theses"]) == 3 and out["items"] >= 1 and out["hashtags"] == 3
    assert conn.execute("SELECT count(*) n FROM theses WHERE status = 'proposed'").fetchone()["n"] == 3
    assert conn.execute("SELECT kind FROM decisions WHERE job_id = %s", (job,)).fetchone()["kind"] == "discovery"
