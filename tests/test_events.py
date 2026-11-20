"""Events (Brain M8): detectors watch durable state and raise events; each is raised once, routed by policy,
and rate-limited so an event storm can't trigger runaway work."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds.brain import events

from brain_helpers import creator

NOW = datetime(2026, 11, 20, 18, tzinfo=timezone.utc)


def _published(conn, cid, hours_ago, reach, *, age_h=24, caption="c"):
    pid = conn.execute("INSERT INTO posts (creator_id, format, status, caption, published_at, body) "
                       "VALUES (%s, 'photo', 'simulated', %s, %s, %s) RETURNING id",
                       (cid, caption, NOW - timedelta(hours=hours_ago), Jsonb({"scenes": ["x"]}))).fetchone()["id"]
    conn.execute("INSERT INTO simulated_metrics (post_id, post_age_s, model, panel_version, metrics) "
                 "VALUES (%s, %s, 'scenario', 't', %s)", (pid, int(age_h * 3600), Jsonb({"reach": reach})))
    return pid


def _baseline(conn, cid, n=10, reach=1000):
    for i in range(n):
        _published(conn, cid, 200 + i * 24, reach)


def test_a_breakout_is_raised_once_and_sent_to_the_analyst(conn, policy):
    cid = creator(conn)
    _baseline(conn, cid)
    ratio = float(policy.get("brain", "events", "breakout_ratio"))
    hit = _published(conn, cid, 30, int(1000 * ratio * 1.5), caption="the kettle saga")
    out = events.tick(conn, policy, now=NOW)
    assert [e["type"] for e in out] == ["breakout"]
    ev = conn.execute("SELECT type, subject, routed_to, job_id FROM events").fetchone()
    assert ev["subject"]["post_id"] == str(hit) and ev["routed_to"] == "analyst" and ev["job_id"]
    job = conn.execute("SELECT kind, payload FROM jobs WHERE id = %s", (ev["job_id"],)).fetchone()
    assert job["kind"] == "brain.analyst" and job["payload"]["creator_id"] == str(cid)
    assert events.tick(conn, policy, now=NOW) == []                       # once per post


def test_an_ordinary_post_raises_nothing(conn, policy):
    cid = creator(conn)
    _baseline(conn, cid)
    _published(conn, cid, 30, 1100)
    assert events.tick(conn, policy, now=NOW) == []


def test_a_run_of_weak_posts_raises_a_drop(conn, policy):
    cid = creator(conn)
    _baseline(conn, cid)
    k = int(policy.get("brain", "events", "drop_posts"))
    for i in range(k):
        _published(conn, cid, 30 + i * 24, 150)
    assert "drop" in [e["type"] for e in events.tick(conn, policy, now=NOW)]


def test_an_event_storm_is_rate_limited(conn, policy):
    cid = creator(conn)
    _baseline(conn, cid)
    limit = int(policy.get("brain", "events", "max_per_type_per_creator_per_day"))
    for i in range(limit + 3):
        _published(conn, cid, 30 + i, 9000 + i)
    out = events.tick(conn, policy, now=NOW)
    assert len([e for e in out if e["type"] == "breakout"]) == limit
    routed = conn.execute("SELECT count(*) n FROM jobs WHERE kind = 'brain.analyst'").fetchone()["n"]
    assert routed == 1                                                    # one analysis covers the storm


def test_the_next_plan_hears_about_recent_breakouts(conn, policy):
    cid = creator(conn)
    _baseline(conn, cid)
    _published(conn, cid, 30, 9000, caption="the kettle saga")
    events.tick(conn, policy, now=NOW)
    notes = events.for_planning(conn, cid, now=NOW)
    assert notes and "the kettle saga" in notes[0]["caption"] and notes[0]["label"]   # sandbox: labelled SIMULATED


def test_the_showrunner_is_told_about_a_breakout(conn, policy):
    from brain_helpers import StubLLM, make_ctx
    from manyworlds import queue
    from manyworlds.brain.agents import showrunner
    cid = creator(conn)
    _baseline(conn, cid)
    _published(conn, cid, 30, 9000, caption="the kettle saga")
    events.tick(conn, policy, now=NOW)
    conn.execute("UPDATE events SET created_at = now()")
    job = queue.enqueue(conn, "brain.showrunner", {"creator_id": str(cid), "start": "2026-11-21"}, creator_id=cid)
    ctx = make_ctx(conn, policy, job)
    prompt, _ = showrunner._prompt(ctx, cid, __import__("datetime").date(2026, 11, 21), 3)
    assert "the kettle saga" in prompt and "far better than usual" in prompt
