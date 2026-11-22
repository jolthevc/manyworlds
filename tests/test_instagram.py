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
    conn.execute("UPDATE accounts SET connected_at = '2026-01-01' WHERE creator_id = %s", (cid,))   # past the ramp
    monkeypatch.setenv("IG_TOKEN_TEST", "tok-secret")
    fake = FakeGraph()
    monkeypatch.setattr(instagram, "graph", fake)
    monkeypatch.setattr(instagram, "public_jpeg", lambda store, uri, tmp: f"https://cdn.example/{uri.rsplit('/', 1)[-1]}.jpg")
    monkeypatch.setattr(instagram.InstagramPublisher, "store", lambda self: store)   # never the env's real bucket
    return cid, fake


def _post(conn, cid, fmt="photo", images=1, video=False, cover=None):
    body = {"platform_ai_label": True, "images": [{"uri": f"s3://b/i{i}.png"} for i in range(images)],
            "hashtags": ["#Sunday"]}
    if video:
        body["video"] = {"uri": "s3://b/v.mp4"}
    if cover:
        body["cover"] = {"asset_id": "a-cover", "uri": cover}
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
    creates = {c[2]["media_type"]: c[2] for c in fake.calls if c[0] == "POST" and c[1].endswith("/media")}
    assert set(creates) == {"REELS", "STORIES"}              # same scheduled time: no order between them
    assert creates["REELS"]["video_url"] and creates["REELS"]["share_to_feed"] is True


def test_a_reel_uses_the_finish_stages_cover_else_its_first_photo(conn, policy, ig):
    cid, fake = ig
    _post(conn, cid, "talking_video", video=True, cover="s3://b/keyframes/k2.png")
    _post(conn, cid, "talking_video", video=True)                   # no picked cover: the first photo, as before
    scheduler.publish_due(conn, policy, now=NOW)
    reels = [c[2] for c in fake.calls if c[0] == "POST" and c[1].endswith("/media")]
    assert {r["media_type"] for r in reels} == {"REELS"} and len(reels) == 2
    assert {r["cover_url"] for r in reels} == {"https://cdn.example/k2.png.jpg", "https://cdn.example/i0.png.jpg"}


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
    monkeypatch.setattr(instagram, "graph", lambda m, p, params=None, token=None: calls.append((m, p, token.token)) or
                        {"user_id": "1784", "username": "maren.afterhours", "account_type": "MEDIA_CREATOR",
                         "biography": "rain, records · AI-generated character", "profile_picture_url": "https://x"})
    out = instagram.connect(conn, policy, cid, token_env="IG_TOKEN_MAREN")
    assert out["instagram_user_id"] == "1784"
    assert calls == [("GET", "/me", "tok"), ("GET", "/1784/content_publishing_limit", "tok")]
    acc = conn.execute("SELECT external_id, token_env, status FROM accounts WHERE creator_id = %s", (cid,)).fetchone()
    assert (acc["external_id"], acc["token_env"], acc["status"]) == ("1784", "IG_TOKEN_MAREN", "connected")
    assert conn.execute("SELECT publish_mode FROM creators WHERE id = %s", (cid,)).fetchone()["publish_mode"] == "instagram"


def test_connecting_without_the_token_in_env_changes_nothing(conn, policy, monkeypatch):
    cid = creator(conn, publish_mode="simulated")
    monkeypatch.delenv("IG_TOKEN_NOPE", raising=False)
    with pytest.raises(instagram.AccountNotConnected):
        instagram.connect(conn, policy, cid, token_env="IG_TOKEN_NOPE")
    assert conn.execute("SELECT publish_mode FROM creators WHERE id = %s", (cid,)).fetchone()["publish_mode"] == "simulated"


# --- review fixes: never twice, always labelled, bounded retries ------------------------------------

def test_a_failed_permalink_lookup_never_republishes(conn, policy, ig, monkeypatch):
    cid, fake = ig
    pid = _post(conn, cid)
    scheduler.publish_due(conn, policy, now=NOW)
    real = fake.__call__

    def no_permalink(method, path, params=None, token=None):
        if method == "GET" and path == "/m1":
            raise instagram.GraphError("500")
        return real(method, path, params, token)
    monkeypatch.setattr(instagram, "graph", no_permalink)
    assert scheduler.publish_due(conn, policy, now=NOW + timedelta(minutes=2))["published"] == 1
    assert len([c for c in fake.calls if c[1].endswith("/media_publish")]) == 1


def test_a_publish_call_that_errors_is_reconciled_not_repeated(conn, policy, ig, monkeypatch):
    cid, fake = ig
    pid = _post(conn, cid)
    scheduler.publish_due(conn, policy, now=NOW)
    real = fake.__call__

    def lost_reply(method, path, params=None, token=None):
        if path.endswith("/media_publish"):
            fake.calls.append((method, path, params))
            raise instagram.GraphError("timeout")      # Meta may have published anyway
        return real(method, path, params, token)
    monkeypatch.setattr(instagram, "graph", lost_reply)
    scheduler.publish_due(conn, policy, now=NOW + timedelta(minutes=2))
    fake.status = "FINISHED"                             # not PUBLISHED: we can't be sure what happened
    out = scheduler.publish_due(conn, policy, now=NOW + timedelta(minutes=4))
    assert out.get("held") == 1
    assert len([c for c in fake.calls if c[1].endswith("/media_publish")]) == 1
    assert conn.execute("SELECT status FROM posts WHERE id = %s", (pid,)).fetchone()["status"] == "held"
    assert conn.execute("SELECT routed_to FROM events WHERE type = 'publish_held'").fetchone()["routed_to"] == "owner"


@pytest.mark.parametrize("change,why", [
    ({"platform_ai_label": False}, "AI label"),
    ({"hashtags": [f"#t{i}" for i in range(31)]}, "hashtags"),
])
def test_posts_that_break_the_rules_are_held_not_sent(conn, policy, ig, change, why):
    cid, fake = ig
    pid = _post(conn, cid)
    conn.execute("UPDATE posts SET body = body || %s WHERE id = %s", (Jsonb(change), pid))
    assert scheduler.publish_due(conn, policy, now=NOW).get("held") == 1
    assert why in conn.execute("SELECT body->>'hold_reason' r FROM posts WHERE id = %s", (pid,)).fetchone()["r"]
    assert not [c for c in fake.calls if c[1].endswith("/media")]


def test_repeated_failures_stop_after_the_attempt_limit(conn, policy, ig):
    cid, fake = ig
    pid = _post(conn, cid)
    fake.status = "ERROR"
    for i in range(8):
        scheduler.publish_due(conn, policy, now=NOW + timedelta(minutes=i))
    creates = [c for c in fake.calls if c[0] == "POST" and c[1].endswith("/media")]
    assert len(creates) == int(policy.get("instagram", "max_attempts"))
    assert conn.execute("SELECT status FROM posts WHERE id = %s", (pid,)).fetchone()["status"] == "held"


def test_reels_are_asked_only_for_reel_metrics(conn, policy, ig):
    cid, fake = ig
    pid = _post(conn, cid, "talking_video", video=True)
    conn.execute("UPDATE posts SET status = 'published', external_id = 'm1', published_at = %s WHERE id = %s",
                 (NOW - timedelta(hours=2), pid))
    conn.execute("INSERT INTO publications (post_id, platform, external_id, ai_label, simulated, published_at) "
                 "VALUES (%s, 'instagram', 'm1', true, false, %s)", (pid, NOW - timedelta(hours=2)))
    metrics.collect(conn, policy, now=NOW)
    asked = [c[2]["metric"] for c in fake.calls if c[1].endswith("/insights")][0]
    assert "follows" not in asked and "reach" in asked
