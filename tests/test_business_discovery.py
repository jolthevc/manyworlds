"""Business Discovery prepared before our account is connected (owner 2026-11-25): fields from policy, what Meta
withholds recorded as missing (never zero), and a one-call live check that settles the unverified parts. Fixtures
follow the documented response shape (docs/research/2026-11-25-business-discovery-notes.md); nothing here has run
against Meta."""

from __future__ import annotations

from manyworlds.research import providers

MEDIA = [
    {"id": "1", "shortcode": "Cabc", "media_type": "IMAGE", "media_product_type": "FEED", "timestamp":
     "2026-11-20T10:00:00+0000", "caption": "lisbon light #travel", "comments_count": 12, "media_url": "https://x/1.jpg",
     "permalink": "https://www.instagram.com/p/Cabc/"},                      # likes hidden: no like_count
    {"id": "2", "shortcode": "Cdef", "media_type": "VIDEO", "media_product_type": "REELS", "timestamp":
     "2026-11-19T10:00:00+0000", "like_count": 900, "comments_count": 30, "view_count": 41000, "reposts_count": 7,
     "media_audio_type": "licensed", "thumbnail_url": "https://x/2.jpg",
     "permalink": "https://www.instagram.com/reel/Cdef/"},                   # licensed audio: no media_url
]


def _answer(fields):
    return {"business_discovery": {"username": "mira.lou", "followers_count": 52000, "media_count": 310,
                                   "biography": "lisbon", "media": {"data": MEDIA, "paging": {"cursors": {}}}}}


def test_the_query_asks_only_for_the_policys_fields(policy):
    seen = []
    g = providers.GraphProvider(policy, call=lambda path, params: seen.append(params["fields"]) or _answer(None))
    g.fetch_posts("instagram", "mira.lou", 2, None)
    q = seen[0]
    assert q.startswith("business_discovery.username(mira.lou){username,biography,followers_count")
    assert ",name," not in q and "view_count" in q and "reposts_count" in q and "children{media_url,media_type}" in q


def test_withheld_fields_are_missing_not_zero(policy):
    g = providers.GraphProvider(policy, call=lambda path, params: _answer(None))
    photo, reel = g.fetch_posts("instagram", "mira.lou", 2, None)[0]
    assert "likes" not in photo["metrics"] and photo["missing"] == ["likes"]
    assert reel["metrics"]["views"] == 41000 and reel["metrics"]["reposts"] == 7
    assert reel["missing"] == ["media"] and reel["video_url"] is None and reel["audio_type"] == "licensed"


def test_the_live_check_reports_which_fields_came_back(policy):
    g = providers.GraphProvider(policy, call=lambda path, params: _answer(None))
    r = g.check("mira.lou")
    assert r["profile_fields_missing"] == ["website"] and r["posts"] == 2
    assert "like_count" in r["media_fields_seen"] and "is_shared_to_feed" in r["media_fields_never_seen"]


def test_uses_pending_counsel_are_refused_in_code(conn, policy):
    """Owner 2026-11-26: where permission is materially uncertain the function stays off. The monitor's repeated
    per-creator snapshots are `history`, pending counsel for Business Discovery; a one-off research read is not."""
    from manyworlds.research import data_policy
    assert "pending counsel" in data_policy.refused(policy, "graph", "posts", "history")
    assert data_policy.refused(policy, "graph", "posts", None) is None
    g = providers.GraphProvider(policy, call=lambda path, params: _answer(None))
    r = providers.Router(conn, policy, adapters={"graph": g}, agent="intelligence:monitor")
    r.route = lambda task, platform: ["graph"]
    posts, used, errors = r.fetch_posts("instagram", "mira.lou", 2)
    assert used is None and "pending counsel" in errors[0]["skipped"]
    r2 = providers.Router(conn, policy, adapters={"graph": g}, agent="intelligence:pilot")
    r2.route = lambda task, platform: ["graph"]
    assert r2.fetch_posts("instagram", "mira.lou", 2)[1] == "graph"
