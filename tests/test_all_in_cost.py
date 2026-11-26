"""All-in cost per accepted output (owner 2026-11-25): the capability card carries every machine's whole bill (boot,
downloads, idle, failed and never-started machines) and untagged orchestration calls, not only job seconds."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from psycopg.types.json import Jsonb

from manyworlds.brain import capability_card
from manyworlds.deploy import report

NOW = datetime.now(timezone.utc)


def _machine(conn, name, *, usd, family="image", start=None):
    start = start or NOW - timedelta(hours=3)
    conn.execute(
        """INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd, started_at,
                                   billed_until, billed_usd, cost_usd, stopped_at, stop_reason, family)
           VALUES (%s, 'vast', 'p', 1, %s, %s, 5, %s, %s, %s, %s, %s, 'idle', %s)""",
        (name, [f"{family}.x"], start + timedelta(hours=2), start, start + timedelta(hours=1), usd, usd,
         start + timedelta(hours=1), family))
    return start


def _job(conn, post_id, machine, start, seconds, batch=None):
    jid = conn.execute("INSERT INTO jobs (kind, status, post_id, batch) VALUES ('image.x', 'done', %s, %s) "
                       "RETURNING id", (post_id, batch)).fetchone()["id"]
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, outcome, gpu_group, started_at, finished_at) "
                 "VALUES (%s, 'w', 'p', 'done', %s, %s, %s)",
                 (jid, machine, start, start + timedelta(seconds=seconds)))
    # the worker's attribution entry: job seconds only (what the card used to count)
    conn.execute("INSERT INTO cost_entries (job_id, category, provider, quantity, unit, usd) "
                 "VALUES (%s, 'gpu', 'p', %s, 'gpu_second', %s)", (jid, seconds, seconds / 3600))
    return jid


def _world(conn):
    cid = conn.execute("INSERT INTO creators (handle, status, bible) VALUES ('cost', 'active', %s) RETURNING id",
                       (Jsonb({}),)).fetchone()["id"]
    posts = [conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, 'photo', 'ready', '{}') "
                          "RETURNING id", (cid,)).fetchone()["id"] for _ in range(2)]
    start = _machine(conn, "m-a", usd=1.0)
    _machine(conn, "m-dead", usd=0.5)                                  # never ran a job: spread over its family
    j1 = _job(conn, posts[0], "m-a", start + timedelta(minutes=5), 60, batch="test-breadth-x")
    j2 = _job(conn, posts[1], "m-a", start + timedelta(minutes=10), 180)
    conn.execute("INSERT INTO cost_entries (post_id, category, provider, quantity, unit, usd, agent) "
                 "VALUES (%s, 'llm', 'anthropic', 1, 'token', 0.1, 'director')", (posts[0],))
    conn.execute("INSERT INTO cost_entries (category, provider, quantity, unit, usd, agent) "
                 "VALUES ('llm', 'anthropic', 1, 'token', 0.2, 'showrunner:ideas')")      # booked to no post
    conn.execute("INSERT INTO cost_entries (category, provider, quantity, unit, usd, agent) "
                 "VALUES ('llm', 'anthropic', 1, 'token', 5.0, 'research:desk')")         # not production
    return j1, j2


def test_each_machine_bill_is_split_over_its_jobs_and_dead_machines_over_the_family(conn):
    j1, j2 = _world(conn)
    by_job = report.machine_cost_by_job(conn, since=NOW - timedelta(days=1), now=NOW)
    assert by_job[str(j1)] == pytest.approx(0.25 * 1.5) and by_job[str(j2)] == pytest.approx(0.75 * 1.5)
    assert sum(v for k, v in by_job.items() if not k.startswith("_")) == pytest.approx(1.5)   # no dollar lost


def test_a_family_with_no_working_machine_keeps_its_money_visible(conn):
    _machine(conn, "m-only-dead", usd=0.7, family="talk")
    out = report.machine_cost_by_job(conn, since=NOW - timedelta(days=1), now=NOW)
    assert out["_unallocated"] == {"talk": 0.7}


def test_the_card_is_all_in_and_shows_the_nominal_gap_and_development_share(conn, policy):
    _world(conn)
    m = capability_card.measure(conn, policy, now=NOW)["photo"]
    # gpu 1.5 + direction 0.1 + showrunner 0.2 spread; research is not production cost
    assert m["usd"] == pytest.approx(1.8, abs=1e-4) and m["usd_per_accepted"] == pytest.approx(0.9, abs=1e-4)
    assert m["usd_nominal"] == pytest.approx(0.1 + 240 / 3600, abs=1e-4)
    assert m["usd_per_accepted"] > 5 * m["nominal_per_accepted"]
    assert m["breakdown"]["gpu_machines"] == pytest.approx(1.5, abs=1e-4)
    assert m["breakdown"]["orchestration"] == pytest.approx(0.2, abs=1e-4)
    assert m["development_share"] == 0.5


def test_the_economics_report_separates_development_from_recurring_production(conn, policy):
    from manyworlds import economics
    _world(conn)
    r = economics.report(conn, policy, days=1, now=NOW)
    assert r["development"] == pytest.approx(0.375 + 0.1, abs=1e-4)       # the test-batch post: machines + direction
    assert r["production"] == pytest.approx(1.125, abs=1e-4)
    assert r["orchestration"] == pytest.approx(0.2) and r["research"] == pytest.approx(5.0)
