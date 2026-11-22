"""Instagram intake (system-user token, setup sheet, verified connection, posting ramp) and comment replies, with
Meta's API faked. Nothing here talks to Meta or to a model."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from psycopg.types.json import Jsonb

from manyworlds import policy as policy_mod
from manyworlds.publish import comments, instagram, scheduler

from brain_helpers import StubLLM, creator

NOW = datetime(2026, 11, 22, 18, tzinfo=timezone.utc)
BIBLE = {"handle": "maren.afterhours", "display_name": "Maren Holt", "one_line": "night-shift baker, records, rain"}


class Meta:
    """Answers like the Graph API for the Unified Login calls; records (method, path, params, host, token)."""

    def __init__(self, bio="night-shift baker · AI-generated character", assigned=None, comments=None, can_publish=True):
        self.calls, self.bio, self.can_publish = [], bio, can_publish
        self.assigned = assigned if assigned is not None else [{"id": "1784", "username": "maren.afterhours"}]
        self.comments, self.replies = comments or [], {}

    def __call__(self, method, path, params=None, token=None):
        self.calls.append((method, path, dict(params or {}), token.host, token.token))
        if path == "/me/assigned_instagram_accounts":
            return {"data": self.assigned}
        if path == "/1784" and method == "GET":
            return {"username": "maren.afterhours", "name": "Maren Holt", "biography": self.bio,
                    "profile_picture_url": "https://cdn/p.jpg", "followers_count": 0, "media_count": 0}
        if path.endswith("/content_publishing_limit"):
            if not self.can_publish:
                raise instagram.GraphError(f"GET {path} -> 403: (#10) permission denied token={token.token}")
            return {"data": [{"quota_usage": 0, "config": {"quota_total": 50, "quota_duration": 86400}}]}
        if path.endswith("/comments"):
            return {"data": self.comments}
        if method == "POST" and path.endswith("/replies"):
            cid = path.strip("/").split("/")[0]
            self.replies[cid] = params["message"]
            return {"id": f"r-{cid}"}
        if method == "GET" and path.endswith("/replies"):
            cid = path.strip("/").split("/")[0]
            return {"data": [{"id": f"r-{cid}", "from": {"id": "1784"}}] if cid in self.replies else []}
        raise AssertionError(f"unexpected call {method} {path}")


@pytest.fixture
def maren(conn, monkeypatch):
    cid = creator(conn)
    conn.execute("UPDATE creators SET bible = %s WHERE id = %s", (Jsonb(BIBLE), cid))
    monkeypatch.setenv("META_SYSTEM_USER_TOKEN", "sys-secret")
    return cid


def test_connecting_by_username_uses_the_system_user_token_and_checks_the_profile(conn, policy, maren, monkeypatch):
    meta = Meta()
    monkeypatch.setattr(instagram, "graph", meta)
    out = instagram.connect(conn, policy, maren, username="@Maren.AfterHours")
    assert out["instagram_user_id"] == "1784" and out["login"] == "unified" and out["warnings"] == []
    assert {c[3] for c in meta.calls} == {"graph.facebook.com"} and {c[4] for c in meta.calls} == {"sys-secret"}
    acc = conn.execute("SELECT external_id, token_env, meta, connected_at FROM accounts WHERE creator_id = %s",
                       (maren,)).fetchone()
    assert acc["external_id"] == "1784" and acc["token_env"] is None and acc["meta"]["login"] == "unified"
    assert acc["connected_at"] is not None
    creds = instagram.account(conn, maren)
    assert (creds.ig_id, creds.host) == ("1784", "graph.facebook.com") and "sys-secret" not in repr(creds)


@pytest.mark.parametrize("meta, why", [
    (Meta(assigned=[]), "not assigned to our app"),
    (Meta(bio="night-shift baker"), "disclosure"),
    (Meta(can_publish=False), "publishing limit"),
])
def test_an_account_that_isnt_set_up_right_is_not_connected(conn, policy, maren, monkeypatch, meta, why):
    monkeypatch.setattr(instagram, "graph", meta)
    with pytest.raises(instagram.AccountNotConnected, match=why) as e:
        instagram.connect(conn, policy, maren, username="maren.afterhours")
    assert "sys-secret" not in str(e.value)
    assert conn.execute("SELECT count(*) n FROM accounts WHERE creator_id = %s", (maren,)).fetchone()["n"] == 0
    assert conn.execute("SELECT publish_mode FROM creators WHERE id = %s", (maren,)).fetchone()["publish_mode"] == "simulated"


def test_one_instagram_account_belongs_to_one_creator(conn, policy, maren, monkeypatch):
    monkeypatch.setattr(instagram, "graph", Meta())
    instagram.connect(conn, policy, maren, username="maren.afterhours")
    other = creator(conn)
    with pytest.raises(instagram.AccountNotConnected, match="already connected"):
        instagram.connect(conn, policy, other, username="maren.afterhours")


def test_the_setup_sheet_carries_the_disclosure_within_instagrams_bio_limit(conn, policy, maren, store):
    conn.execute("UPDATE creators SET bible = %s, anchors = %s WHERE id = %s",
                 (Jsonb({**BIBLE, "one_line": "x" * 400}), Jsonb({"reference_images": [{"uri": "s3://b/a.png"}]}), maren))
    sheet = instagram.setup_sheet(conn, policy, maren, store=type("S", (), {"public_url": lambda s, u, t: f"https://u/{u}"})())
    assert sheet["bio"].endswith("AI-generated character") and len(sheet["bio"]) <= 150
    assert sheet["username"] == "maren.afterhours" and sheet["display_name"] == "Maren Holt"
    assert sheet["profile_photo_url"] == "https://u/s3://b/a.png" and sheet["checklist"]


def _connected(conn, cid, days_ago: float):
    conn.execute("INSERT INTO accounts (creator_id, platform, handle, external_id, status, meta, connected_at) "
                 "VALUES (%s, 'instagram', 'maren.afterhours', '1784', 'connected', %s, %s)",
                 (cid, Jsonb({"login": "unified"}), NOW - timedelta(days=days_ago)))
    conn.execute("UPDATE creators SET publish_mode = 'instagram' WHERE id = %s", (cid,))


def test_a_new_account_ramps_up(policy):
    assert [instagram.ramp_allowance(policy, d) for d in (0, 6.9, 7, 30)] == [1, 1, 2, None]


def test_the_ramp_holds_posts_back_and_caps_planning(conn, policy, maren, monkeypatch):
    _connected(conn, maren, days_ago=1)
    post = conn.execute("INSERT INTO posts (creator_id, platform, format, status, caption, body, scheduled_for) "
                        "VALUES (%s, 'instagram', 'photo', 'ready', 'x', %s, %s) RETURNING id",
                        (maren, Jsonb({"platform_ai_label": True, "images": [{"uri": "s3://b/i.png"}]}),
                         NOW - timedelta(minutes=5))).fetchone()["id"]
    done = conn.execute("INSERT INTO posts (creator_id, format, status) VALUES (%s, 'photo', 'published') RETURNING id",
                        (maren,)).fetchone()["id"]
    conn.execute("INSERT INTO publications (post_id, platform, account_id, external_id, simulated, published_at) "
                 "VALUES (%s, 'instagram', (SELECT id FROM accounts WHERE creator_id = %s), 'm0', false, %s)",
                 (done, maren, datetime.now(timezone.utc) - timedelta(hours=2)))
    monkeypatch.setattr(instagram, "graph", Meta())
    out = scheduler.publish_due(conn, policy, now=NOW)
    assert out.get("ramp") == 1 and out["published"] == 0
    assert conn.execute("SELECT status FROM posts WHERE id = %s", (post,)).fetchone()["status"] == "ready"
    dials = instagram.ramp_dials(conn, policy, maren, {"posts_per_day": {"instagram": 2, "tiktok": 3}}, NOW)
    assert dials["posts_per_day"] == {"instagram": 1.0, "tiktok": 3}


# --- comments ---------------------------------------------------------------------------------------
class ReplyLLM(StubLLM):
    def __init__(self, decide):
        super().__init__()
        self.decide = decide

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema, effort=effort, creator_id=creator_id)
        return schema.model_validate({"items": self.decide})


def _published(conn, cid):
    post = conn.execute("INSERT INTO posts (creator_id, format, status, caption) VALUES (%s, 'photo', 'published', "
                        "'the 4am croissant') RETURNING id", (cid,)).fetchone()["id"]
    conn.execute("INSERT INTO publications (post_id, platform, account_id, external_id, simulated, published_at) "
                 "VALUES (%s, 'instagram', (SELECT id FROM accounts WHERE creator_id = %s), 'm1', false, %s)",
                 (post, cid, NOW - timedelta(hours=3)))


COMMENTS = [
    {"id": "k1", "text": "is this real??", "timestamp": "2026-11-22T16:00:00+0000", "username": "fan1", "from": {"id": "9"}},
    {"id": "k2", "text": "buy followers cheap", "timestamp": "2026-11-22T16:01:00+0000", "username": "spam", "from": {"id": "8"}},
    {"id": "k3", "text": "ok but the lamination", "timestamp": "2026-11-22T16:02:00+0000", "username": "fan2", "from": {"id": "7"}},
    {"id": "k4", "text": "love it", "timestamp": "2026-11-22T16:03:00+0000", "username": "other.creator", "from": {"id": "555"}},
]


@pytest.fixture
def commented(conn, policy, maren, monkeypatch):
    _connected(conn, maren, days_ago=30)
    other = creator(conn)   # another of our creators: never answered
    conn.execute("INSERT INTO accounts (creator_id, platform, handle, external_id, status) "
                 "VALUES (%s, 'instagram', 'other.creator', '555', 'connected')", (other,))
    _published(conn, maren)
    meta = Meta(comments=COMMENTS)
    monkeypatch.setattr(instagram, "graph", meta)
    return maren, meta


def test_comments_are_stored_once_without_handles_and_ours_are_never_answered(conn, policy, commented):
    cid, meta = commented
    assert comments.collect(conn, policy, now=NOW)["new"] == 3
    assert comments.collect(conn, policy, now=NOW + timedelta(minutes=5)) == {"read": 0, "new": 0, "errors": 0}
    rows = {r["external_id"]: r for r in conn.execute("SELECT * FROM comments").fetchall()}
    assert rows["k4"]["status"] == "skipped" and rows["k4"]["reason"] == "one of our accounts"
    assert all(r["author_hash"] and "fan" not in r["author_hash"] for r in rows.values())


def test_replies_are_drafted_by_default_and_rule_breaking_ones_are_dropped(conn, policy, commented):
    cid, meta = commented
    comments.collect(conn, policy, now=NOW)
    llm = ReplyLLM([{"comment": "c1", "action": "reply", "text": "real dough, AI-generated baker. both true"},
                    {"comment": "c2", "action": "skip", "reason": "spam"},
                    {"comment": "c3", "action": "reply", "text": "follow @someone for more #baking"}])
    out = comments.respond(conn, policy, llm, now=NOW)
    assert (out["drafted"], out["skipped"]) == (1, 2) and meta.replies == {}
    assert llm.calls[0]["role"] == "comment_reply" and llm.calls[0]["agent"] == "comments"
    assert "fan1" not in llm.calls[0]["content"]          # the model never sees who commented
    got = {r["external_id"]: r for r in conn.execute("SELECT * FROM comments").fetchall()}
    assert got["k1"]["status"] == "drafted" and got["k3"]["reason"] == "mention, hashtag or link"


def test_autonomous_replies_are_sent_once_and_capped_per_day(conn, policy, commented):
    cid, meta = commented
    policy = _with(policy, "autonomous", max_replies_per_account_day=1)
    comments.collect(conn, policy, now=NOW)
    llm = ReplyLLM([{"comment": "c1", "action": "reply", "text": "AI-generated, but the butter is serious"}])
    assert comments.respond(conn, policy, llm, now=NOW)["replied"] == 1
    assert meta.replies == {"k1": "AI-generated, but the butter is serious"}
    assert comments.respond(conn, policy, llm, now=NOW)["replied"] == 0       # today's allowance is used
    assert len(llm.calls) == 1


def test_a_reply_sent_but_not_recorded_is_found_not_sent_again(conn, policy, commented):
    cid, meta = commented
    policy = _with(policy, "autonomous")
    comments.collect(conn, policy, now=NOW)
    meta.replies["k1"] = "earlier"
    conn.execute("UPDATE comments SET status = 'replying', reply_sent_at = %s WHERE external_id = 'k1'", (NOW,))
    conn.execute("UPDATE comments SET status = 'replying', reply_sent_at = %s WHERE external_id = 'k3'", (NOW,))
    out = comments.respond(conn, policy, ReplyLLM([]), now=NOW)
    assert out["reconciled"] == 2
    got = {r["external_id"]: r for r in conn.execute("SELECT * FROM comments").fetchall()}
    assert (got["k1"]["status"], got["k1"]["reply_external_id"]) == ("replied", "r-k1")
    assert got["k3"]["status"] == "failed" and "k3" not in meta.replies


def test_nothing_is_read_or_answered_when_replies_are_disabled(conn, policy, commented):
    policy = _with(policy, "disabled")
    assert comments.collect(conn, policy, now=NOW) == {} and comments.respond(conn, policy, ReplyLLM([]), now=NOW) == {}


def _with(policy, replies: str, **comment_cfg):
    """The policy with comment replies in this mode (a copy: the shared policy is never changed)."""
    body = {**policy.body, "autonomy": {**policy.body["autonomy"], "comment_replies": replies},
            "comments": {**policy.body["comments"], **comment_cfg}}
    return policy_mod.from_dict(body)
