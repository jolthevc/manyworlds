"""Instagram publishing and metrics (Brain M4), against Meta's Instagram Graph API, with the HTTP calls faked.
Nothing here talks to Meta: going live needs only the owner's account, token and `mw account connect`."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from psycopg.types.json import Jsonb

from manyworlds.publish import instagram, metrics, scheduler

from brain_helpers import creator

NOW = datetime(2026, 11, 22, 18, tzinfo=timezone.utc)


class FakeGraph:
    """Records calls; answers like the Graph API."""

    def __init__(self, status="FINISHED", quota=0):
        self.calls, self.status, self.quota, self.n = [], status, quota, 0

    def __call__(self, method, path, params=None, token=None):
        self.calls.append((method, path, dict(params or {})))
        if path.endswith("/content_publishing_limit"):
            return {"data": [{"quota_usage": self.quota, "config": {"quota_total": 50, "quota_duration": 86400}}]}
        if method == "POST" and path.endswith("/media"):
            self.n += 1
            return {"id": f"c{self.n}"}
        if method == "GET" and path.startswith("/c"):
            return {"status_code": self.status, "id": path.strip("/")}
        if path.endswith("/media_publish"):
            return {"id": "m1"}
        if method == "GET" and path == "/m1":
            return {"id": "m1", "permalink": "https://www.instagram.com/p/abc/"}
        if path.endswith("/insights"):
            return {"data": [{"name": n, "values": [{"value": v}]} for n, v in
                             (("reach", 900), ("views", 1500), ("likes", 80), ("comments", 6), ("saved", 12),
                              ("shares", 4), ("follows", 3))]}
        raise AssertionError(f"unexpected call {method} {path}")


@pytest.fixture
def ig(conn, store, tmp_path, monkeypatch):
    cid = creator(conn, publish_mode="instagram")
    conn.execute("INSERT INTO accounts (creator_id, platform, external_id, token_env, status) "
                 "VALUES (%s, 'instagram', '17841400000', 'IG_TOKEN_TEST', 'connected')", (cid,))
    monkeypatch.setenv("IG_TOKEN_TEST", "tok-secret")
    fake = FakeGraph()
    monkeypatch.setattr(instagram, "graph", fake)
    monkeypatch.setattr(instagram, "public_jpeg", lambda store, uri, tmp: f"https://cdn.example/{uri.rsplit('/', 1)[-1]}.jpg")
    return cid, fake


def _post(conn, cid, fmt="photo", images=1, video=False):
    body = {"platform_ai_label": True, "images": [{"uri": f"s3://b/i{i}.png"} for i in range(images)],
            "hashtags": ["#Sunday"]}
    if video:
        body["video"] = {"uri": "s3://b/v.mp4"}
    return conn.execute("INSERT INTO posts (creator_id, platform, format, status, caption, body, scheduled_for) "
                        "VALUES (%s, 'instagram', %s, 'ready', 'rain again', %s, %s) RETURNING id",
                        (cid, fmt, Jsonb(body), NOW - timedelta(minutes=5))).fetchone()["id"]


def test_a_photo_is_published_with_the_ai_label_in_two_steps(conn, policy, ig, store):
    cid, fake = ig
    pid = _post(conn, cid)
    first = scheduler.publish_due(conn, policy, now=NOW)
    assert first["pending"] == 1                                         # container made, nothing published yet
    create = [c for c in fake.calls if c[0] == "POST" and c[1].endswith("/media")][0][2]
    assert create["is_ai_generated"] is True and create["image_url"].endswith(".jpg")
    assert create["caption"] == "rain again\n\n#Sunday"
    second = scheduler.publish_due(conn, policy, now=NOW + timedelta(minutes=2))
    assert second["published"] == 1
    pub = conn.execute("SELECT platform, external_id, permalink, ai_label, simulated FROM publications").fetchone()
    assert (pub["platform"], pub["external_id"], pub["ai_label"], pub["simulated"]) == ("instagram", "m1", True, False)
    assert pub["permalink"].startswith("https://www.instagram.com/")
    assert conn.execute("SELECT status FROM posts WHERE id = %s", (pid,)).fetchone()["status"] == "published"


def test_a_carousel_labels_the_album_not_its_children(conn, policy, ig):
    cid, fake = ig
    _post(conn, cid, "carousel", images=3)
    scheduler.publish_due(conn, policy, now=NOW)
    creates = [c[2] for c in fake.calls if c[0] == "POST" and c[1].endswith("/media")]
    children, album = creates[:3], creates[3]
    assert all(ch["is_carousel_item"] is True and "is_ai_generated" not in ch and "caption" not in ch for ch in children)
    assert album["media_type"] == "CAROUSEL" and album["children"] == "c1,c2,c3" and album["is_ai_generated"] is True


def test_videos_go_out_as_reels_and_stories_as_stories(conn, policy, ig):
    cid, fake = ig
    _post(conn, cid, "talking_video", video=True)
    _post(conn, cid, "story")
    scheduler.publish_due(conn, policy, now=NOW)
    creates = [c[2] for c in fake.calls if c[0] == "POST" and c[1].endswith("/media")]
    assert creates[0]["media_type"] == "REELS" and creates[0]["video_url"] and creates[0]["share_to_feed"] is True
    assert creates[1]["media_type"] == "STORIES"


def test_a_container_still_processing_waits_and_an_error_is_reported(conn, policy, ig):
    cid, fake = ig
    pid = _post(conn, cid)
    fake.status = "IN_PROGRESS"
    scheduler.publish_due(conn, policy, now=NOW)
    assert scheduler.publish_due(conn, policy, now=NOW + timedelta(minutes=1))["pending"] == 1
    fake.status = "ERROR"
    out = scheduler.publish_due(conn, policy, now=NOW + timedelta(minutes=2))
    assert out["failed"] == 1
    row = conn.execute("SELECT status, body FROM posts WHERE id = %s", (pid,)).fetchone()
    assert row["status"] == "ready" and "ig" not in row["body"] and "ERROR" in row["body"]["publish_errors"][0]


def test_the_daily_quota_is_respected(conn, policy, ig):
    cid, fake = ig
    fake.quota = 50
    _post(conn, cid)
    assert scheduler.publish_due(conn, policy, now=NOW)["quota"] == 1
    assert not [c for c in fake.calls if c[1].endswith("/media")]


def test_a_creator_without_a_connected_account_waits(conn, policy, ig):
    cid, fake = ig
    conn.execute("UPDATE accounts SET status = 'requested' WHERE creator_id = %s", (cid,))
    _post(conn, cid)
    assert scheduler.publish_due(conn, policy, now=NOW)["waiting_for_account"] == 1
    assert fake.calls == []


def test_the_token_never_appears_in_errors(conn, policy, ig, monkeypatch):
    cid, fake = ig

    def boom(method, path, params=None, token=None):
        raise instagram.GraphError(f"400 bad request for token {token}")
    monkeypatch.setattr(instagram, "graph", boom)
    pid = _post(conn, cid)
    scheduler.publish_due(conn, policy, now=NOW)
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    assert "tok-secret" not in str(body)


def test_real_metrics_are_collected_at_each_age_once(conn, policy, ig):
    cid, fake = ig
    pid = _post(conn, cid)
    conn.execute("UPDATE posts SET status = 'published', external_id = 'm1', published_at = %s WHERE id = %s",
                 (NOW - timedelta(hours=7), pid))
    conn.execute("INSERT INTO publications (post_id, platform, external_id, ai_label, simulated, published_at) "
                 "VALUES (%s, 'instagram', 'm1', true, false, %s)", (pid, NOW - timedelta(hours=7)))
    out = metrics.collect(conn, policy, now=NOW)
    rows = conn.execute("SELECT post_age_s, metrics, simulated FROM metric_snapshots ORDER BY post_age_s").fetchall()
    # collected late: one snapshot at the post's real age (7 h), covering the 1 h and 6 h observations it missed
    assert [r["post_age_s"] for r in rows] == [7 * 3600] and out["snapshots"] == 1
    assert rows[0]["metrics"] == {"reach": 900, "views": 1500, "likes": 80, "comments": 6, "saves": 12, "shares": 4,
                                  "follows": 3} and rows[0]["simulated"] is False
    assert metrics.collect(conn, policy, now=NOW)["snapshots"] == 0      # once per age
    assert metrics.collect(conn, policy, now=NOW + timedelta(hours=18))["snapshots"] == 1   # the 24 h observation


def test_connecting_an_account_checks_the_token_and_leaves_the_sandbox(conn, policy, monkeypatch):
    cid = creator(conn, publish_mode="simulated")
    monkeypatch.setenv("IG_TOKEN_MAREN", "tok")
    calls = []
    monkeypatch.setattr(instagram, "graph", lambda m, p, params=None, token=None: calls.append((m, p, token)) or
                        {"user_id": "1784", "username": "maren.afterhours", "account_type": "MEDIA_CREATOR"})
    out = instagram.connect(conn, policy, cid, token_env="IG_TOKEN_MAREN")
    assert out["instagram_user_id"] == "1784" and calls == [("GET", "/me", "tok")]
    acc = conn.execute("SELECT external_id, token_env, status FROM accounts WHERE creator_id = %s", (cid,)).fetchone()
    assert (acc["external_id"], acc["token_env"], acc["status"]) == ("1784", "IG_TOKEN_MAREN", "connected")
    assert conn.execute("SELECT publish_mode FROM creators WHERE id = %s", (cid,)).fetchone()["publish_mode"] == "instagram"


def test_connecting_without_the_token_in_env_changes_nothing(conn, policy, monkeypatch):
    cid = creator(conn, publish_mode="simulated")
    monkeypatch.delenv("IG_TOKEN_NOPE", raising=False)
    with pytest.raises(instagram.AccountNotConnected):
        instagram.connect(conn, policy, cid, token_env="IG_TOKEN_NOPE")
    assert conn.execute("SELECT publish_mode FROM creators WHERE id = %s", (cid,)).fetchone()["publish_mode"] == "simulated"
