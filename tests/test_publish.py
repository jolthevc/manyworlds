"""Posts go out through one publisher interface: simulated for sandbox creators (clearly marked,
never 'published'), Instagram for real ones once that adapter exists; a kill switch stops both."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds.publish import scheduler

from brain_helpers import creator

NOW = datetime(2026, 11, 20, 18, tzinfo=timezone.utc)


def _post(conn, cid, status="ready", when=NOW - timedelta(hours=1)):
    return conn.execute("INSERT INTO posts (creator_id, platform, format, status, caption, body, scheduled_for) "
                        "VALUES (%s, 'instagram', 'photo', %s, 'c', %s, %s) RETURNING id",
                        (cid, status, Jsonb({"platform_ai_label": True}), when)).fetchone()["id"]


def _with(policy, path, value):
    body = {**policy.body, path[0]: {**policy.body.get(path[0], {}), path[1]: value}}
    return dataclasses.replace(policy, body=body)


def test_sandbox_posts_are_simulated_not_published(conn, policy):
    cid = creator(conn, publish_mode="simulated")
    due, later = _post(conn, cid), _post(conn, cid, when=NOW + timedelta(hours=2))
    out = scheduler.publish_due(conn, policy, now=NOW)
    assert out["simulated"] == 1
    rows = {r["id"]: r for r in conn.execute("SELECT id, status, published_at FROM posts").fetchall()}
    assert rows[due]["status"] == "simulated" and rows[due]["published_at"] is not None
    assert rows[later]["status"] == "ready"
    pub = conn.execute("SELECT platform, simulated, ai_label FROM publications WHERE post_id = %s", (due,)).fetchone()
    assert (pub["platform"], pub["simulated"], pub["ai_label"]) == ("simulated", True, True)
    assert scheduler.publish_due(conn, policy, now=NOW)["simulated"] == 0          # never twice


def test_the_kill_switch_stops_publishing(conn, policy):
    cid = creator(conn, publish_mode="simulated")
    _post(conn, cid)
    out = scheduler.publish_due(conn, _with(policy, ("autonomy", "publish"), "paused"), now=NOW)
    assert out == {"paused": True} and conn.execute("SELECT count(*) n FROM publications").fetchone()["n"] == 0


def test_real_creators_wait_for_the_instagram_adapter(conn, policy):
    cid = creator(conn, publish_mode="instagram")
    post = _post(conn, cid)
    out = scheduler.publish_due(conn, policy, now=NOW)
    assert out["waiting_for_adapter"] == 1
    assert conn.execute("SELECT status FROM posts WHERE id = %s", (post,)).fetchone()["status"] == "ready"


def test_fast_forward_is_sandbox_only(conn, policy):
    sandbox, real = creator(conn, publish_mode="simulated"), creator(conn, publish_mode="instagram")
    a, b = _post(conn, sandbox, status="planned"), _post(conn, real, status="planned")
    scheduler.publish_due(conn, _with(policy, ("simulation", "fast_forward"), True), now=NOW)
    status = {r["id"]: r["status"] for r in conn.execute("SELECT id, status FROM posts").fetchall()}
    assert status[a] == "simulated" and status[b] == "planned"
