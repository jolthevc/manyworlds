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
def policy():
    return policy_mod.load(ROOT / "policy" / "default.yaml")


@pytest.fixture()
def conn(database_url, policy):
    c = db.connect(database_url)
    db.migrate(c)
    policy_mod.register(c, policy)
    c.execute(
        "TRUNCATE jobs, attempts, assets, cost_entries, reviews, posts, metric_snapshots, "
        "plans, experiments, theses, creators, research_items, research_runs, decisions "
        "RESTART IDENTITY CASCADE"
    )
    yield c
    c.close()


@pytest.fixture()
def store(tmp_path):
    return LocalStore(tmp_path / "objects")
