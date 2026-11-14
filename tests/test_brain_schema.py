"""The Brain's tables: everything it knows lives in Postgres, real and simulated data apart."""

from __future__ import annotations

import psycopg
import pytest

from manyworlds import queue

from brain_helpers import make_post


def test_brain_tables_exist(conn):
    names = {r["table_name"] for r in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'").fetchall()}
    assert {"accounts", "creator_states", "creator_memory", "ideas", "evaluations", "rubric_versions",
            "findings", "allocations", "publications", "simulated_metrics", "markets", "clusters",
            "convictions", "budgets", "approvals", "events", "policy_overrides"} <= names


def test_real_metric_snapshots_cannot_hold_simulated_rows(conn):
    post = make_post(conn)
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("INSERT INTO metric_snapshots (post_id, simulated, metrics) VALUES (%s, true, '{}')", (post,))


def test_new_creators_start_in_simulated_mode(conn):
    row = conn.execute("INSERT INTO creators (handle) VALUES ('x') RETURNING publish_mode").fetchone()
    assert row["publish_mode"] == "simulated"


def test_one_decision_per_job_step(conn):
    job = queue.enqueue(conn, "brain.echo", {})
    conn.execute("INSERT INTO decisions (kind, job_id, step) VALUES ('k', %s, 's')", (job,))
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute("INSERT INTO decisions (kind, job_id, step) VALUES ('k', %s, 's')", (job,))
