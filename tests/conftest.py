"""Each test session gets a throwaway database on the Postgres in DATABASE_URL."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest

from manyworlds import db, policy as policy_mod
from manyworlds.storage import LocalStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def database_url():
    base = os.environ.get("DATABASE_URL")
    if not base:
        pytest.skip("DATABASE_URL not set")
    name = f"mw_test_{uuid.uuid4().hex[:8]}"
    admin = psycopg.connect(base, autocommit=True)
    admin.execute(f'CREATE DATABASE "{name}"')
    url = urlunparse(urlparse(base)._replace(path=f"/{name}"))
    yield url
    admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
    admin.close()


@pytest.fixture(scope="session")
def raw_policy():
    return policy_mod.load(ROOT / "policy" / "default.yaml")


@pytest.fixture(scope="session")
def policy(raw_policy):
    """The shipped policy with operations live: tests describe what runs once the owner turns it on
    (paused behaviour is tested explicitly in test_operations.py). Shadow tests are off: they sample at
    random and would add candidate calls to tests that count calls (test_llm.py turns them on itself)."""
    return policy_mod.from_dict({**raw_policy.body, "operations": {
        **(raw_policy.body.get("operations") or {}), "mode": "live"},
        "llm": {**raw_policy.body["llm"], "shadow": {}}})


@pytest.fixture()
def conn(database_url, policy):
    c = db.connect(database_url)
    db.migrate(c)
    policy_mod.register(c, policy)
    c.execute(
        "TRUNCATE jobs, attempts, assets, cost_entries, reviews, posts, metric_snapshots, "
        "plans, experiments, theses, creators, research_items, research_runs, decisions, gpu_groups, "
        "markets, clusters, accounts, creator_states, creator_memory, ideas, rubric_versions, evaluations, "
        "findings, allocations, budgets, convictions, publications, simulated_metrics, approvals, events, "
        "policy_overrides, simulated_findings, leases, provider_charges, launches, llm_shadow "
        "RESTART IDENTITY CASCADE"
    )
    yield c
    c.close()


@pytest.fixture(autouse=True)
def worker_credentials(monkeypatch):
    """GPU machines need a scoped database login (policy security.worker_database: required); tests that
    launch machines get a stand-in. test_credentials.py covers the real role."""
    monkeypatch.setenv("WORKER_DATABASE_URL", "postgresql://mw_worker_login:test@localhost/test")


@pytest.fixture()
def store(tmp_path):
    return LocalStore(tmp_path / "objects")
