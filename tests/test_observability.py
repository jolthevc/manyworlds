"""Structured logs (manyworlds.logs) and event-driven wakeups (manyworlds.wakeup, migration 033)."""

from __future__ import annotations

import json
import logging
import time

import pytest

from manyworlds import db, logs, queue, wakeup


def _record(msg="hello %s", args=("world",), exc_info=None):
    return logging.LogRecord("manyworlds.test", logging.INFO, __file__, 1, msg, args, exc_info)


# --- logs -------------------------------------------------------------------------------------------------------

def test_a_json_line_carries_the_context_fields_that_are_set_and_omits_the_rest():
    fmt = logs.JsonFormatter(role="produce", group="producer")
    with logs.context(job_id="j-1", kind="image.zimage"):
        with logs.context(attempt_id=42):
            line = json.loads(fmt.format(_record()))
    assert line["msg"] == "hello world"
    assert line["level"] == "INFO" and line["logger"] == "manyworlds.test"
    assert line["role"] == "produce" and line["group"] == "producer"
    assert line["job_id"] == "j-1" and line["kind"] == "image.zimage" and line["attempt_id"] == 42
    assert "ts" in line
    for unset in ("post_id", "creator_id", "exc"):
        assert unset not in line


def test_context_is_restored_when_a_block_ends():
    fmt = logs.JsonFormatter()
    with logs.context(post_id="p-1"):
        with logs.context(post_id="p-2", creator_id="c-1"):
            assert logs.current() == {"post_id": "p-2", "creator_id": "c-1"}
        assert logs.current() == {"post_id": "p-1"}
    line = json.loads(fmt.format(_record()))
    assert not set(logs.FIELDS) & set(line)
    with pytest.raises(TypeError):
        with logs.context(password="x"):
            pass


def test_a_json_line_keeps_the_traceback():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        rec = _record("failed", (), sys.exc_info())
    line = json.loads(logs.JsonFormatter().format(rec))
    assert "ValueError: boom" in line["exc"]


def test_plain_text_appends_the_context():
    with logs.context(job_id="j-9"):
        assert logs.TextFormatter().format(_record()).endswith("hello world [job_id=j-9]")
    assert logs.TextFormatter().format(_record()).endswith("hello world")


def test_json_is_chosen_by_the_env_then_the_policy(policy):
    assert logs.json_wanted(policy, env={"MW_LOG_JSON": "1"}) is True
    assert logs.json_wanted(policy, env={"MW_LOG_JSON": "0"}) is False
    assert logs.json_wanted(policy, env={}) is bool(policy.get("logging", "json", default=False))
    assert logs.json_wanted(None, env={}) is False


# --- the trigger -------------------------------------------------------------------------------------------------

def _drain(conn, seconds=1.0) -> list[dict]:
    return [{"channel": n.channel, **json.loads(n.payload or "{}")} for n in conn.notifies(timeout=seconds)]


def test_jobs_notify_on_insert_and_on_finishing_but_not_on_claim(conn, database_url):
    listen = db.connect(database_url)
    try:
        listen.execute("LISTEN mw_jobs")
        jid = queue.enqueue(conn, "fake.image", {})
        conn.execute("UPDATE jobs SET status = 'running' WHERE id = %s", (jid,))
        conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (jid,))
        got = _drain(listen)
    finally:
        listen.close()
    assert [(g["id"], g["status"]) for g in got] == [(str(jid), "queued"), (str(jid), "done")]
    assert got[0]["kind"] == "fake.image" and got[0]["post_id"] is None and got[0]["channel"] == "mw_jobs"


def test_the_worker_role_fires_the_trigger_too(conn, database_url):
    """Rented GPUs update jobs as mw_worker (migration 022): the trigger needs no grant beyond what it has."""
    jid = queue.enqueue(conn, "fake.image", {})
    conn.execute("UPDATE jobs SET status = 'running' WHERE id = %s", (jid,))
    listen, worker = db.connect(database_url), db.connect(database_url)
    try:
        listen.execute("LISTEN mw_jobs")
        worker.execute("SET ROLE mw_worker")
        worker.execute("UPDATE jobs SET status = 'failed' WHERE id = %s", (jid,))
        got = _drain(listen)
    finally:
        listen.close()
        worker.close()
    assert [(g["id"], g["status"]) for g in got] == [(str(jid), "failed")]


def test_reviews_notify_once_per_statement(conn, database_url):
    listen = db.connect(database_url)
    try:
        listen.execute("LISTEN mw_reviews")
        aid = conn.execute("INSERT INTO assets (kind, uri, sha256, bytes) VALUES ('image', 'mem://a', 'x', 1) "
                           "RETURNING id").fetchone()["id"]
        conn.execute("INSERT INTO reviews (asset_id, gate, passed) VALUES (%s, 'a', true), (%s, 'b', true)",
                     (aid, aid))
        got = _drain(listen)
    finally:
        listen.close()
    assert [g["channel"] for g in got] == ["mw_reviews"]


# --- the wait ----------------------------------------------------------------------------------------------------

def _notify(conn, channel="mw_jobs", **payload):
    conn.execute("SELECT pg_notify(%s, %s)", (channel, json.dumps(payload)))


def test_the_wait_returns_early_on_a_notification(conn, database_url):
    lis = wakeup.Listener(connect=lambda: db.connect(database_url), accept=wakeup.accepts(["done"], True))
    try:
        assert lis.ensure()
        _notify(conn, id="j1", status="done")
        t0 = time.monotonic()
        got = lis.wait(20)
        assert time.monotonic() - t0 < 10
        assert got == [{"channel": "mw_jobs", "id": "j1", "status": "done"}]
    finally:
        lis.close()


def test_the_wait_times_out_without_a_matching_notification(conn, database_url):
    lis = wakeup.Listener(connect=lambda: db.connect(database_url), accept=wakeup.accepts(["done"], False))
    try:
        assert lis.ensure()
        _notify(conn, id="j1", status="queued")    # an insert: not what this loop waits for
        _notify(conn, channel="mw_reviews")       # reviews: off for this loop
        t0 = time.monotonic()
        assert lis.wait(0.5) == []
        assert time.monotonic() - t0 >= 0.45
    finally:
        lis.close()


def test_a_burst_is_one_wakeup(conn, database_url):
    lis = wakeup.Listener(connect=lambda: db.connect(database_url), accept=wakeup.accepts(["done"], False),
                          debounce_seconds=0.5)
    try:
        assert lis.ensure()
        for i in range(20):
            _notify(conn, id=f"j{i}", status="done")
        got = lis.wait(20)
        assert len(got) == 20
        t0 = time.monotonic()
        assert lis.wait(0.3) == []   # nothing left over for a second tick
        assert time.monotonic() - t0 >= 0.25
    finally:
        lis.close()


def test_a_listener_that_cannot_connect_sleeps_and_does_not_raise():
    calls, slept = [], []

    def broken():
        calls.append(1)
        raise OSError("no database")

    lis = wakeup.Listener(connect=broken, reconnect_seconds=3600, sleep=slept.append)
    assert lis.wait(5) == [] and lis.wait(5) == []
    assert len(calls) == 1                    # no reconnect storm within reconnect_seconds
    assert len(slept) == 2 and all(4 <= s <= 5 for s in slept)


def test_a_listener_that_drops_mid_wait_sleeps_the_rest_and_reconnects_later():
    slept = []

    class Dead:
        autocommit = False
        closed = False

        def execute(self, *_a, **_k):
            return None

        def notifies(self, **_k):
            raise OSError("connection lost")

        def close(self):
            self.closed = True

    dead = Dead()
    lis = wakeup.Listener(connect=lambda: dead, sleep=slept.append)
    assert lis.wait(2) == []
    assert dead.closed and lis.conn is None and len(slept) == 1


def test_without_a_listener_the_wait_is_a_plain_sleep():
    slept = []
    assert wakeup.wait(None, 7, sleep=slept.append) == [] and slept == [7.0]


def test_the_shipped_policy_wakes_the_producer_on_reviews_and_the_reviewer_on_finished_jobs(policy):
    prod = wakeup.Listener.from_policy(policy, "producer", connect=lambda: None)
    rev = wakeup.Listener.from_policy(policy, "reviewer", connect=lambda: None)
    assert prod.accept("mw_reviews", {}) and prod.accept("mw_jobs", {"status": "dead"})
    assert not prod.accept("mw_jobs", {"status": "queued"})
    assert rev.accept("mw_jobs", {"status": "done"}) and not rev.accept("mw_reviews", {})


def test_listen_uses_the_direct_endpoint_not_the_pooler():
    from manyworlds import db
    pooled = "postgresql://u:p@ep-cool-name-123456-pooler.us-east-2.aws.neon.tech/mw?sslmode=require"
    assert db.direct_url(pooled) == "postgresql://u:p@ep-cool-name-123456.us-east-2.aws.neon.tech/mw?sslmode=require"
    plain = "postgresql://mw:mw@db:5432/mw"
    assert db.direct_url(plain) == plain
