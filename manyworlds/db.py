"""Postgres connection and a minimal forward-only migration runner."""

from __future__ import annotations

import os
from importlib import resources

import psycopg
from psycopg.rows import dict_row


def connect(url: str | None = None, timeout: float | None = None) -> psycopg.Connection:
    url = url or os.environ["DATABASE_URL"]
    extra = {"connect_timeout": int(timeout)} if timeout else {}
    return psycopg.connect(url, autocommit=True, row_factory=dict_row, **extra)


def direct_url(url: str | None = None) -> str:
    """The database URL without a connection pooler, for LISTEN (manyworlds.wakeup): Neon's pooled endpoint
    (`ep-...-pooler.<region>...`) runs PgBouncer in transaction mode, which never delivers notifications; the direct
    endpoint is the same host without `-pooler`. Other URLs are returned as they are."""
    url = url or os.environ["DATABASE_URL"]
    creds, sep, rest = url.rpartition("@")
    host, slash, tail = rest.partition("/")
    return creds + sep + host.replace("-pooler.", ".", 1) + slash + tail


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
