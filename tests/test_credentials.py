"""Rented GPU machines get scoped credentials: enough to do their jobs, nothing else (deploy/credentials.py)."""

from __future__ import annotations

import psycopg
import pytest

from manyworlds import db, policy as policy_mod, queue
from manyworlds.deploy import autoscale, credentials

from test_autoscale import T0, providers
from test_worker import make_worker


@pytest.fixture()
def worker_url(conn, database_url):
    return credentials.rotate_login(conn, database_url)


def test_login_url_keeps_host_database_and_options():
    url = credentials.login_url("postgresql://owner:s3cret@ep-x.us-east-2.aws.neon.tech/neondb?sslmode=require",
                                "mw_worker_login", "p/w+x")
    assert url == "postgresql://mw_worker_login:p%2Fw%2Bx@ep-x.us-east-2.aws.neon.tech/neondb?sslmode=require"
    assert "owner" not in url and "s3cret" not in url


def test_a_worker_with_the_scoped_login_does_its_jobs(conn, worker_url, store, policy):
    ok = queue.enqueue(conn, "fake.image", {"prompt": "x", "n": 2})
    bad = queue.enqueue(conn, "fake.image", {"fail": "always"}, max_attempts=1)
    assert make_worker(worker_url, store, policy).run(max_jobs=2) == 2
    status = {r["id"]: r["status"] for r in conn.execute("SELECT id, status FROM jobs")}
    assert status == {ok: "done", bad: "failed"}
    assert conn.execute("SELECT count(*) AS n FROM assets WHERE job_id = %s", (ok,)).fetchone()["n"] == 2
    assert conn.execute("SELECT count(*) AS n FROM cost_entries").fetchone()["n"] == 2   # failures charged too


@pytest.mark.parametrize("sql", [
    "SELECT * FROM creators", "SELECT * FROM decisions", "SELECT * FROM research_items",
    "SELECT * FROM cost_entries", "SELECT * FROM accounts",
    "INSERT INTO jobs (kind, payload) VALUES ('fake.image', '{}')",
    "DELETE FROM jobs", "UPDATE gpu_groups SET max_usd = 1000", "DROP TABLE jobs",
])
def test_the_scoped_login_reaches_nothing_else(worker_url, sql):
    c = db.connect(worker_url)
    try:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute(sql)
    finally:
        c.close()


def test_rotating_gives_a_new_password(conn, database_url):
    first = credentials.rotate_login(conn, database_url)
    second = credentials.rotate_login(conn, database_url)
    assert first != second
    with pytest.raises(psycopg.OperationalError):
        db.connect(first).close()
    db.connect(second).close()


def test_machines_get_the_scoped_login_never_the_control_planes(conn, policy, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://owner:master@db/neondb")
    monkeypatch.setenv("WORKER_DATABASE_URL", "postgresql://mw_worker_login:scoped@db/neondb")
    queue.enqueue(conn, "image.zimage", {}, batch="bench-x")
    ps = providers()
    autoscale.tick(conn, policy, ps, now=T0)
    envs = [spec.env for p in ps.values() for _, spec in p.launched]
    assert envs and all(e["DATABASE_URL"].endswith("scoped@db/neondb") for e in envs)
    assert not any("master" in v for e in envs for v in e.values())


def test_no_machine_starts_without_the_scoped_login(conn, policy, monkeypatch):
    monkeypatch.delenv("WORKER_DATABASE_URL", raising=False)
    queue.enqueue(conn, "image.zimage", {}, batch="bench-x")
    ps = providers()
    actions = autoscale.tick(conn, policy, ps, now=T0)
    assert not [p for p in ps.values() if p.launched]
    assert {"action": "not_launching", "reason": "worker_credentials"}.items() <= next(
        a for a in actions if a.get("reason") == "worker_credentials").items()


def test_shared_mode_falls_back_for_development(conn, policy, monkeypatch):
    monkeypatch.delenv("WORKER_DATABASE_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://dev:dev@localhost/mw")
    pol = policy_mod.from_dict({**policy.body, "security": {"worker_database": "shared", "worker_storage": "shared"}})
    assert credentials.worker_env(pol)["DATABASE_URL"] == "postgresql://dev:dev@localhost/mw"
