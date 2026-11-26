"""Rented GPU machines get scoped credentials: enough to do their jobs, nothing else (deploy/credentials.py)."""

from __future__ import annotations

import psycopg
import pytest

from manyworlds import db, policy as policy_mod, queue
from manyworlds.deploy import autoscale, credentials

from test_autoscale import T0, providers, registry  # noqa: F401 - registry: autouse image registry stub
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


def test_control_plane_credentials_never_reach_a_rented_machine(policy, monkeypatch):
    # owner 2026-11-26: the old `shared` mode handed out the control plane's own database URL and R2 key
    import pytest
    monkeypatch.delenv("WORKER_DATABASE_URL", raising=False)
    monkeypatch.delenv("WORKER_AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("WORKER_AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://owner:cp@db/mw")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "cp-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "cp-secret")
    pol = policy_mod.from_dict({**policy.body, "security": {"worker_database": "shared", "worker_storage": "shared"}})
    assert len(credentials.missing(pol)) == 2
    with pytest.raises(credentials.MissingWorkerCredentials):
        credentials.worker_env(pol)
    monkeypatch.setenv("WORKER_DATABASE_URL", "postgresql://w:w@db/mw")
    monkeypatch.setenv("WORKER_AWS_ACCESS_KEY_ID", "scoped-key")
    monkeypatch.setenv("WORKER_AWS_SECRET_ACCESS_KEY", "scoped-secret")
    env = credentials.worker_env(pol)
    assert env["DATABASE_URL"] == "postgresql://w:w@db/mw" and env["AWS_ACCESS_KEY_ID"] == "scoped-key"
    assert not {"cp-key", "cp-secret", "postgresql://owner:cp@db/mw"} & set(env.values())


def test_a_rented_host_cannot_queue_brain_work_forge_the_ledger_or_extend_its_deadline(conn):
    # review 2026-11-25: 022 let mw_worker rewrite any job, insert negative gpu_billed rows and push its deadline out
    import psycopg
    import pytest
    from manyworlds import queue
    brain = queue.enqueue(conn, "brain.intelligence", {"mode": "pilot"})
    gpu = queue.enqueue(conn, "image.zimage", {"prompt": "x"}, batch="test-x")
    conn.execute("INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd, started_at) "
                 "VALUES ('mw-x', 'runpod', 'p', 1, '{image.zimage}', now() + interval '1 hour', 1, now())")
    conn.execute("SET ROLE mw_worker")
    try:
        assert conn.execute("SELECT count(*) n FROM jobs WHERE id = %s", (brain,)).fetchone()["n"] == 0
        assert conn.execute("UPDATE jobs SET status = 'queued' WHERE id = %s RETURNING id", (brain,)).fetchall() == []
        for sql, args in (("UPDATE jobs SET payload = '{\"test\": true}' WHERE id = %s", (gpu,)),
                          ("UPDATE jobs SET batch = 'test-y' WHERE id = %s", (gpu,)),
                          ("INSERT INTO cost_entries (category, provider, quantity, unit, usd) "
                           "VALUES ('gpu_billed', 'p', 1, 's', -500)", ()),
                          ("INSERT INTO cost_entries (category, provider, quantity, unit, usd) "
                           "VALUES ('gpu', 'p', 1, 's', -5)", ()),
                          ("UPDATE gpu_groups SET deadline = now() + interval '1 year' WHERE name = 'mw-x'", ())):
            with pytest.raises(psycopg.Error):
                with conn.transaction():
                    conn.execute(sql, args)
        # what a worker legitimately does still works
        conn.execute("UPDATE jobs SET status = 'running', worker_id = 'w', lease_until = now() WHERE id = %s", (gpu,))
        conn.execute("INSERT INTO cost_entries (category, provider, quantity, unit, usd) VALUES ('gpu', 'p', 1, 's', 0.1)")
        conn.execute("UPDATE gpu_groups SET deadline = now() WHERE name = 'mw-x'")
    finally:
        conn.execute("RESET ROLE")
