"""Postgres connection and a minimal forward-only migration runner."""

from __future__ import annotations

import os
from importlib import resources

import psycopg
from psycopg.rows import dict_row


def connect(url: str | None = None) -> psycopg.Connection:
    url = url or os.environ["DATABASE_URL"]
    return psycopg.connect(url, autocommit=True, row_factory=dict_row)


def migrate(conn: psycopg.Connection) -> list[str]:
    """Apply any migration files not yet recorded. Returns the versions applied."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    done = {r["version"] for r in conn.execute("SELECT version FROM schema_migrations")}
    files = sorted(
        f for f in resources.files("manyworlds.migrations").iterdir() if f.name.endswith(".sql")
    )
    applied = []
    for f in files:
        version = f.name.removesuffix(".sql")
        if version in done:
            continue
        with conn.transaction():
            conn.execute(f.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
        applied.append(version)
    return applied
