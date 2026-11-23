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


def _sim(conn):
    cid = creator(conn, publish_mode="simulated")
    conn.execute("UPDATE creators SET anchors = '{\"simulation\": {\"scenario\": \"null\"}}' WHERE id = %s", (cid,))
    return cid


def test_sandbox_posts_are_simulated_not_published(conn, policy):
    cid = _sim(conn)
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
    cid = _sim(conn)
    _post(conn, cid)
    out = scheduler.publish_due(conn, _with(policy, ("autonomy", "publish"), "paused"), now=NOW)
    assert out == {"paused": True} and conn.execute("SELECT count(*) n FROM publications").fetchone()["n"] == 0


def test_real_creators_wait_for_their_connected_account(conn, policy):
    cid = creator(conn, publish_mode="instagram")
    post = _post(conn, cid)
    out = scheduler.publish_due(conn, policy, now=NOW)
    assert out["waiting_for_account"] == 1
    assert conn.execute("SELECT status FROM posts WHERE id = %s", (post,)).fetchone()["status"] == "ready"


def test_fast_forward_is_sandbox_only(conn, policy):
    sandbox, real = _sim(conn), creator(conn, publish_mode="instagram")
    a, b = _post(conn, sandbox, status="planned"), _post(conn, real, status="planned")
    scheduler.publish_due(conn, _with(policy, ("simulation", "fast_forward"), True), now=NOW)
    status = {r["id"]: r["status"] for r in conn.execute("SELECT id, status FROM posts").fetchall()}
    assert status[a] == "simulated" and status[b] == "planned"


def test_a_creator_waiting_for_their_account_keeps_their_posts(conn, policy):
    # Maren: sandbox only until her Instagram account is connected; her posts wait for it.
    cid = creator(conn, publish_mode="simulated")
    post = _post(conn, cid)
    out = scheduler.publish_due(conn, policy, now=NOW)
    assert out["waiting_for_account"] == 1
    assert conn.execute("SELECT status FROM posts WHERE id = %s", (post,)).fetchone()["status"] == "ready"


class FakeInstagram:
    platform = "instagram"

    def publish(self, conn, policy, post):
        return {"external_id": "ig1", "permalink": "https://instagram.example/p/1", "ai_label": True, "simulated": False}


def test_real_posts_wait_unless_publishing_is_autonomous(conn, policy, monkeypatch):
    monkeypatch.setitem(scheduler.PUBLISHERS, "instagram", FakeInstagram())
    cid = creator(conn, publish_mode="instagram")
    post = _post(conn, cid)
    out = scheduler.publish_due(conn, _with(policy, ("autonomy", "publish"), "human_approval"), now=NOW)
    assert out["awaiting_approval"] == 1
    assert scheduler.publish_due(conn, policy, now=NOW)["published"] == 1
    assert conn.execute("SELECT status FROM posts WHERE id = %s", (post,)).fetchone()["status"] == "published"


def test_stale_real_posts_are_not_dumped_all_at_once(conn, policy, monkeypatch):
    monkeypatch.setitem(scheduler.PUBLISHERS, "instagram", FakeInstagram())
    cid = creator(conn, publish_mode="instagram")
    late = float(policy.get("publishing", "max_late_hours"))
    old = _post(conn, cid, when=NOW - timedelta(hours=late + 1))
    out = scheduler.publish_due(conn, policy, now=NOW)
    assert out["stale"] == 1
    assert conn.execute("SELECT status FROM posts WHERE id = %s", (old,)).fetchone()["status"] == "ready"


def test_paused_creators_publish_nothing(conn, policy):
    cid = _sim(conn)
    conn.execute("UPDATE creators SET status = 'paused' WHERE id = %s", (cid,))
    _post(conn, cid)
    assert scheduler.publish_due(conn, policy, now=NOW).get("simulated", 0) == 0


# --- test posts never go out ------------------------------------------------------------------------------------
def _test_post(conn, cid, body):
    return conn.execute("INSERT INTO posts (creator_id, platform, format, status, caption, body, scheduled_for) "
                        "VALUES (%s, 'instagram', 'photo', 'ready', 'c', %s, %s) RETURNING id",
                        (cid, Jsonb({"platform_ai_label": True, **body}), NOW - timedelta(hours=1))).fetchone()["id"]


def test_a_breadth_test_post_is_never_published_even_when_scheduled(conn, policy, monkeypatch):
    monkeypatch.setitem(scheduler.PUBLISHERS, "instagram", FakeInstagram())
    real, sandbox = creator(conn, publish_mode="instagram"), _sim(conn)
    posts = [_test_post(conn, real, {"breadth_run": "r1"}), _test_post(conn, sandbox, {"breadth_run": "r1"}),
             _test_post(conn, real, {"batch": "test-breadth-r1"})]
    out = scheduler.publish_due(conn, policy, now=NOW)
    assert out["test_posts"] == 3 and out["published"] == 0 and out["simulated"] == 0
    assert {r["status"] for r in conn.execute("SELECT status FROM posts WHERE id = ANY(%s)", (posts,))} == {"ready"}
    assert conn.execute("SELECT count(*) AS n FROM publications").fetchone()["n"] == 0


def test_a_post_made_in_a_test_batch_is_never_published(conn, policy, monkeypatch):
    monkeypatch.setitem(scheduler.PUBLISHERS, "instagram", FakeInstagram())
    cid = creator(conn, publish_mode="instagram")
    test, real = _post(conn, cid), _post(conn, cid)
    prefix = policy.get("operations", "test_batch_prefixes")[0]
    for pid, batch in ((test, f"{prefix}x"), (real, f"post-{real}")):
        conn.execute("INSERT INTO jobs (kind, status, creator_id, post_id, batch) VALUES ('image.zimage', 'done', %s, "
                     "%s, %s)", (cid, pid, batch))
    out = scheduler.publish_due(conn, policy, now=NOW)
    assert out["test_posts"] == 1 and out["published"] == 1
    status = {r["id"]: r["status"] for r in conn.execute("SELECT id, status FROM posts").fetchall()}
    assert status == {test: "ready", real: "published"}


def test_the_instagram_publisher_refuses_a_test_post_before_any_call(conn, policy, monkeypatch):
    import pytest

    from manyworlds.publish import NotForPublishing, instagram
    monkeypatch.setattr(instagram, "graph", lambda *a, **k: pytest.fail("no Graph call for a test post"))
    cid = creator(conn, publish_mode="instagram")
    pid = _test_post(conn, cid, {"breadth_run": "r1"})
    post = conn.execute("SELECT * FROM posts WHERE id = %s", (pid,)).fetchone()
    with pytest.raises(NotForPublishing, match="breadth"):
        instagram.InstagramPublisher().publish(conn, policy, post)
