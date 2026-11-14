"""Shared helpers for the Brain's tests."""

from __future__ import annotations

import uuid


def creator(conn, *, publish_mode: str = "simulated"):
    return conn.execute("INSERT INTO creators (handle, publish_mode) VALUES (%s, %s) RETURNING id",
                        (f"c{uuid.uuid4().hex[:8]}", publish_mode)).fetchone()["id"]


def make_post(conn, creator_id=None):
    cid = creator_id or creator(conn)
    return conn.execute("INSERT INTO posts (creator_id, format) VALUES (%s, 'photo') RETURNING id",
                        (cid,)).fetchone()["id"]
