"""Provider-agnostic acquisition (manyworlds.research.providers) and cost-aware monitoring of the reference creator
universe (manyworlds.research.monitor, the intelligence agent's monitor mode). Everything stubbed: no Apify,
ScrapeCreators, Graph API, LLM or GPU."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

from manyworlds.research import annotation, monitor, providers, pulls
from manyworlds.storage import LocalStore

from test_iia_direction import _ctx

NOW = datetime.now(timezone.utc)


def _with(policy, section: str, key: str, value):
    body = {**policy.body, section: {**policy.body[section], key: value}}
    return dataclasses.replace(policy, body=body)


def _monitor_cfg(policy, **over):
    m = {**policy.body["intelligence"]["monitor"], **over}
    return _with(policy, "intelligence", "monitor", m)


def ig_item(i: int, *, who="mira.lou", kind="Image", likes=100, views=None, caption=None, days=None):
    return {"id": f"{who}-{i}", "shortCode": f"c{i}", "url": f"https://www.instagram.com/p/c{i}/", "type": kind,
            "caption": caption if caption is not None else f"post {i} #daily", "likesCount": likes,
            "commentsCount": 0, "videoViewCount": views, "displayUrl": f"https://cdn.example/{who}/{i}.jpg?oe=1",
            "videoUrl": f"https://cdn.example/{who}/{i}.mp4" if kind == "Video" else None,
            "timestamp": (NOW - timedelta(days=i if days is None else days)).isoformat(), "ownerUsername": who}


def latest(who="mira.lou"):
    """Ten posts: the newest a video far above her median (outperformer, new format), the next an #ad."""
    posts = [ig_item(i, who=who) for i in range(2, 10)]
    return [ig_item(0, who=who, kind="Video", likes=5000, views=90000),
            ig_item(1, who=who, caption="my new routine #ad #glow"), *posts]


class FakeApify:
    def __init__(self, posts=None, fail=False):
        self.calls, self.posts, self.fail = [], posts, fail

    def __call__(self, actor, payload):
        self.calls.append((actor, payload))
        if self.fail:
            raise RuntimeError("apify down")
        if actor == "apify~instagram-profile-scraper":
            return [{"username": u, "fullName": u, "followersCount": 42000,
                     "latestPosts": self.posts if self.posts is not None else latest(u)} for u in payload["usernames"]]
        if actor == "apify~instagram-post-scraper":
            return (self.posts if self.posts is not None else latest(payload["username"][0]))[:payload["resultsLimit"]]
        return []


def _env(monkeypatch, *names):
    for n in ("APIFY_TOKEN", "SCRAPECREATORS_API_KEY", "META_SYSTEM_USER_TOKEN"):
        monkeypatch.delenv(n, raising=False)
    for n in names:
        monkeypatch.setenv(n, "x")


# --- ids, normalisers -----------------------------------------------------------------------------------------------
def test_a_shortcode_is_its_media_id_so_providers_share_one_row():
    assert providers.shortcode_to_id("B") == "1" and providers.shortcode_to_id("BA") == "64"
    assert providers.shortcode_to_id("BAA") == str(64 * 64)
    long = "C" * 20
    assert providers.shortcode_to_id(long) == long and providers.shortcode_to_id(None) is None


def test_scrapecreators_feed_items_normalise_carousels_videos_and_partnerships():
    car = {"pk": "111", "id": "111_9", "code": "abc", "media_type": 8, "taken_at": 1764000000,
           "caption": {"text": "slides #fit"}, "like_count": 10, "comment_count": 2, "is_paid_partnership": True,
           "carousel_media": [{"image_versions2": {"candidates": [{"url": f"https://cdn/{k}.jpg"}]}} for k in range(3)],
           "user": {"username": "mira.lou"}}
    reel = {"pk": "222", "code": "def", "media_type": 2, "product_type": "clips", "taken_at": 1764000000,
            "caption": None, "like_count": 5, "comment_count": 1, "play_count": 900,
            "image_versions2": {"candidates": [{"url": "https://cdn/r.jpg"}]},
            "video_versions": [{"url": "https://cdn/r.mp4"}]}
    c, r = providers.sc_ig_post(car), providers.sc_ig_post(reel)
    assert c["external_id"] == "111" and c["media_type"] == "carousel" and c["children"] == 3 and len(c["images"]) == 3
    assert c["paid_partnership"] and c["hashtags"] == ["fit"] and c["posted_at"].year == 2025
    assert r["media_type"] == "video" and r["metrics"]["views"] == 900 and r["video_url"] == "https://cdn/r.mp4"


def test_business_discovery_posts_take_the_media_id_from_their_shortcode():
    m = {"id": "1789", "shortcode": "BAA", "media_type": "CAROUSEL_ALBUM", "like_count": 3, "comments_count": 1,
         "timestamp": "2026-11-20T10:00:00+0000", "permalink": "https://www.instagram.com/p/BAA/", "caption": "#x",
         "children": {"data": [{"media_url": "https://cdn/1.jpg", "media_type": "IMAGE"},
                               {"media_url": "https://cdn/2.jpg", "media_type": "IMAGE"}]}}
    p = providers.graph_post(m, "mira.lou", 100)
    assert p["external_id"] == str(64 * 64) and p["children"] == 2 and p["images"] == ["https://cdn/1.jpg",
                                                                                       "https://cdn/2.jpg"]


# --- the router -----------------------------------------------------------------------------------------------------
def test_apify_serves_twelve_posts_from_the_profile_scraper_in_one_item(conn, policy, monkeypatch):
    _env(monkeypatch, "APIFY_TOKEN")
    fake = FakeApify()
    r = providers.Router(conn, policy, adapters={"apify": providers.ApifyProvider(policy, fetch=fake)}, agent="t")
    posts, used, _ = r.fetch_posts("instagram", "mira.lou", 12)
    assert used == "apify" and len(posts) == 10 and [a for a, _ in fake.calls] == ["apify~instagram-profile-scraper"]
    row = conn.execute("SELECT provider, quantity, usd, agent FROM cost_entries").fetchone()
    assert row["provider"] == "apify:posts" and float(row["quantity"]) == 1 and row["agent"] == "t"
    assert float(row["usd"]) == policy.get("intelligence", "agent", "usd_per_item", "instagram_profile")
    posts, _, _ = r.fetch_posts("instagram", "mira.lou", 30)           # more than the profile embeds: the post scraper
    assert fake.calls[-1][0] == "apify~instagram-post-scraper"


def test_the_router_falls_back_on_missing_keys_errors_and_incomplete_answers(conn, policy, monkeypatch):
    _env(monkeypatch, "APIFY_TOKEN", "SCRAPECREATORS_API_KEY")       # no Graph token, no connected account

    def sc_down(path, params):
        raise RuntimeError("502")
    adapters = {"graph": providers.GraphProvider(policy, conn=conn),
                "scrapecreators": providers.ScrapeCreatorsProvider(policy, fetch=sc_down),
                "apify": providers.ApifyProvider(policy, fetch=FakeApify())}
    r = providers.Router(conn, policy, adapters=adapters, agent="t")
    posts, used, errors = r.fetch_posts("instagram", "mira.lou", 12)
    assert used == "apify" and posts
    assert errors[0] == {"provider": "graph", "skipped": "META_SYSTEM_USER_TOKEN not set"}
    assert errors[1]["provider"] == "scrapecreators" and "502" in errors[1]["error"]
    tags = {row["provider"] for row in conn.execute("SELECT provider FROM cost_entries").fetchall()}
    assert tags == {"scrapecreators:posts", "apify:posts"}           # the failed request is charged too

    def sc_undated(path, params):                                     # answers, but without dates
        return {"items": [{"pk": "9", "media_type": 1, "like_count": 1}], "more_available": False}
    adapters["scrapecreators"] = providers.ScrapeCreatorsProvider(policy, fetch=sc_undated)
    posts, used, errors = providers.Router(conn, policy, adapters=adapters).fetch_posts("instagram", "mira.lou", 12)
    assert used == "apify" and {"provider": "scrapecreators", "incomplete": True} in errors


def test_scrapecreators_pages_by_cursor_and_bills_a_credit_a_page(conn, policy, monkeypatch):
    _env(monkeypatch, "SCRAPECREATORS_API_KEY")
    calls = []

    def sc(path, params):
        calls.append(params.get("next_max_id"))
        page = len(calls)
        return {"items": [{"pk": f"{page}{k}", "code": f"x{page}{k}", "media_type": 1, "taken_at": 1764000000 - page,
                           "like_count": 1, "caption": {"text": "hi"}} for k in range(12)],
                "more_available": page < 3, "next_max_id": f"cur{page}"}
    r = providers.Router(conn, policy, adapters={"scrapecreators": providers.ScrapeCreatorsProvider(policy, fetch=sc)})
    posts, used, _ = r.fetch_posts("instagram", "mira.lou", 30)
    assert used == "scrapecreators" and len(posts) == 30 and calls == [None, "cur1", "cur2"]
    row = conn.execute("SELECT quantity, usd FROM cost_entries").fetchone()
    per = policy.get("research", "providers", "catalog", "scrapecreators", "usd_per_credit")
    assert float(row["quantity"]) == 3 and float(row["usd"]) == pytest.approx(3 * per)


def test_hikerapi_is_never_routed(policy):
    cat = policy.get("research", "providers", "catalog")
    assert cat["hikerapi"]["enabled"] is False and cat["hikerapi"]["terms"] == "private_api"
    for task in policy.get("research", "providers", "routes").values():
        for route in (task.values() if isinstance(task, dict) else [task]):
            assert "hikerapi" not in route


# --- monitoring -----------------------------------------------------------------------------------------------------
def _creator(conn, handle="mira.lou", role="mid"):
    return pulls.upsert_creator(conn, "instagram", handle, role=role, followers=40000)


def _downloads():
    got = []

    def download(url, dest, max_bytes):
        got.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest
    return got, download


def _router(conn, policy, fake):
    return providers.Router(conn, policy, adapters={"apify": providers.ApifyProvider(policy, fetch=fake)},
                            agent="intelligence:monitor")


def test_a_check_pulls_cheap_metadata_selects_few_posts_and_copies_media_only_for_them(conn, policy, monkeypatch,
                                                                                         tmp_path):
    _env(monkeypatch, "APIFY_TOKEN")
    policy = _monitor_cfg(policy, daily_usd=1.0)
    cid = _creator(conn)
    fake = FakeApify()
    got, download = _downloads()
    out = monitor.tick(conn, policy, _router(conn, policy, fake), now=NOW, store=LocalStore(tmp_path / "s"),
                       download=download, workdir=tmp_path)
    assert len(out["checked"]) == 1 and out["checked"][0]["new"] == 10
    assert [a for a, _ in fake.calls] == ["apify~instagram-profile-scraper"]      # one batched item, no post scraper
    sel = {r["external_id"]: r["selected"]["reasons"] for r in conn.execute(
        "SELECT external_id, selected FROM reference_posts WHERE selected IS NOT NULL").fetchall()}
    assert set(sel) == {"mira.lou-0", "mira.lou-1"}
    assert any(x.startswith("outperformer") for x in sel["mira.lou-0"]) and "new_format:video" in sel["mira.lou-0"]
    assert "commercial" in sel["mira.lou-1"]
    assert sorted(u.split("/")[-1].split("?")[0] for u in got) == ["0.jpg", "0.mp4", "1.jpg"]   # selected only
    row = conn.execute("SELECT tier, next_check_at, last_checked_at, content_hash, last_post_id FROM reference_creators "
                       "WHERE id = %s", (cid,)).fetchone()
    assert row["last_checked_at"] and row["content_hash"] and row["last_post_id"] == "mira.lou-0"
    every = policy.get("intelligence", "monitor", "tiers", row["tier"], "every_days")
    assert timedelta(days=every * 0.8) <= row["next_check_at"] - NOW <= timedelta(days=every * 1.2)
    # the next check: nothing new, nothing re-selected, nothing downloaded again, no duplicate rows
    later = row["next_check_at"] + timedelta(minutes=1)
    out = monitor.tick(conn, policy, _router(conn, policy, fake), now=later, store=LocalStore(tmp_path / "s"),
                       download=download, workdir=tmp_path)
    assert out["checked"][0]["new"] == 0 and out["checked"][0]["unchanged"] and out["checked"][0]["selected"] == []
    assert len(got) == 3 and conn.execute("SELECT count(*) n FROM reference_posts").fetchone()["n"] == 10


def test_checks_stop_at_the_days_budget_and_hot_creators_go_first(conn, policy, monkeypatch, tmp_path):
    _env(monkeypatch, "APIFY_TOKEN")
    per = policy.get("intelligence", "monitor", "usd_per_check", "instagram")
    policy = _monitor_cfg(policy, daily_usd=per * 1.5, prefetch={"platforms": []})
    cold, hot = _creator(conn, "cold.one"), _creator(conn, "hot.one")
    for cid, tier in ((cold, "cold"), (hot, "hot")):
        conn.execute("UPDATE reference_creators SET tier = %s, next_check_at = %s WHERE id = %s",
                     (tier, NOW - timedelta(hours=1), cid))
    out = monitor.tick(conn, policy, _router(conn, policy, FakeApify()), now=NOW)
    assert [c["handle"] for c in out["checked"]] == ["hot.one"] and out["skipped_budget"] == 1


def test_a_failed_check_backs_off_and_a_forgotten_creator_leaves_the_universe(conn, policy, monkeypatch):
    from manyworlds.research import references
    _env(monkeypatch, "APIFY_TOKEN")
    cid = _creator(conn)
    out = monitor.tick(conn, policy, _router(conn, policy, FakeApify(fail=True)), now=NOW)
    row = conn.execute("SELECT check_failures, next_check_at FROM reference_creators WHERE id = %s", (cid,)).fetchone()
    assert out["errors"] and row["check_failures"] == 1 and row["next_check_at"] > NOW
    references.forget(conn, None, "mira.lou")
    assert conn.execute("SELECT count(*) n FROM reference_creators").fetchone()["n"] == 0


def test_tiers_follow_relevance_growth_performance_and_dormancy(conn, policy):
    w = policy.get("intelligence", "monitor", "bands")
    assert monitor.tier_for(policy, {"relevance": 1, "growth": 1, "performance": 1, "intelligence": 1}, NOW, NOW) == "hot"
    assert monitor.tier_for(policy, {"relevance": 0.2}, NOW, NOW) == "cold" and w["hot"] > w["warm"]
    assert monitor.tier_for(policy, {"relevance": 1, "growth": 1, "performance": 1, "intelligence": 1},
                            NOW - timedelta(days=400), NOW) == "dormant"


def test_the_monthly_estimate_scales_with_the_universe(policy):
    a, b = monitor.monthly_estimate(policy, 1000), monitor.monthly_estimate(policy, 5000)
    assert b["checks_per_month"] == pytest.approx(5 * a["checks_per_month"], rel=0.01)
    assert b["total_usd"] == pytest.approx(b["check_usd"] + b["vision_usd"], abs=0.02)


def test_monitor_is_queued_once_a_day_only_when_enabled(conn, policy):
    assert monitor.ensure_due(conn, policy, now=NOW) is None                        # shipped off
    on = _monitor_cfg(policy, enabled=True)
    assert monitor.ensure_due(conn, on, now=NOW) and monitor.ensure_due(conn, on, now=NOW + timedelta(minutes=5))
    assert conn.execute("SELECT count(*) n FROM jobs WHERE kind = 'brain.intelligence'").fetchone()["n"] == 1


# --- the agent's monitor mode: vision only for selected posts, never twice for unchanged content --------------------
def test_monitor_mode_reads_only_selected_posts_and_never_rereads_unchanged_content(conn, policy, monkeypatch,
                                                                                     tmp_path):
    from manyworlds.brain.agents import intelligence
    _env(monkeypatch, "APIFY_TOKEN")
    _creator(conn)
    agent = intelligence.Intelligence()
    agent.fetch = FakeApify()
    got, agent.download = _downloads()
    store = LocalStore(tmp_path / "s")
    agent.store = lambda policy: store
    ctx = _ctx(conn, policy, {"mode": "monitor", "usd": 1}, batch="test-iia-mon-x")
    out = agent.run(ctx)
    reads = conn.execute("SELECT a.post_id, a.content_hash, p.external_id FROM reference_annotations a "
                         "JOIN reference_posts p ON p.id = a.post_id WHERE a.schema = %s",
                         (annotation.ANNOTATION_SCHEMA,)).fetchall()
    assert {r["external_id"] for r in reads} == {"mira.lou-0", "mira.lou-1"} and all(r["content_hash"] for r in reads)
    assert out["monitor"]["annotate"]["annotated"] == 2
    data = conn.execute("SELECT DISTINCT agent FROM cost_entries WHERE category = 'data'").fetchall()
    assert {r["agent"] for r in data} == {"intelligence:monitor"}
    # a second run: unchanged content is not read again; an edited caption is read once more
    ctx2 = _ctx(conn, policy, {"mode": "monitor", "usd": 1}, batch="test-iia-mon-y")
    assert "annotate" not in agent.run(ctx2)["monitor"]
    conn.execute("UPDATE reference_posts SET caption = 'edited', content_hash = 'new' WHERE external_id = 'mira.lou-1'")
    ctx3 = _ctx(conn, policy, {"mode": "monitor", "usd": 1}, batch="test-iia-mon-z")
    assert agent.run(ctx3)["monitor"]["annotate"]["annotated"] == 1
    assert conn.execute("SELECT a.content_hash FROM reference_annotations a JOIN reference_posts p ON p.id = a.post_id "
                        "WHERE p.external_id = 'mira.lou-1'").fetchone()["content_hash"] == "new"


def test_mw_research_monitor_estimate_and_dry_run_queue_nothing(conn, policy, capsys, database_url, monkeypatch):
    from manyworlds import cli
    monkeypatch.setenv("DATABASE_URL", database_url)
    assert cli.main(["research", "monitor", "--estimate", "5000"]) == 0
    assert '"creators": 5000' in capsys.readouterr().out
    assert cli.main(["research", "monitor", "--dry-run"]) == 0
    assert conn.execute("SELECT count(*) n FROM jobs WHERE kind = 'brain.intelligence'").fetchone()["n"] == 0


def test_a_scout_copies_videos_only_for_selected_posts_and_enrolls_its_creators(conn, policy, tmp_path):
    from test_iia_direction import _agent
    agent = _agent(LocalStore(tmp_path / "objects"))
    ctx = _ctx(conn, policy, {"mode": "scout", "hypothesis": "h", "hashtags": ["x"], "handles": ["instagram:mira.lou"],
                              "platforms": ["instagram"], "creators": 1, "posts": 8, "usd": 5})
    out = agent.run(ctx)
    assert out["scout"]["creators"][0]["tier"] in monitor.TIERS
    rows = conn.execute("SELECT p.media_type, p.media, p.selected FROM reference_posts p JOIN reference_creators c "
                        "ON c.id = p.creator_id WHERE c.handle = 'mira.lou'").fetchall()
    videos = [r for r in rows if r["media_type"] == "video"]
    copied = [r for r in videos if any(m["kind"] == "video" for m in r["media"])]
    assert 0 < len(copied) < len(videos) and all(r["selected"] for r in copied)
    assert all(any(m["kind"] == "image" for m in r["media"]) for r in rows)      # every post's images, for the read
    assert conn.execute("SELECT count(*) n FROM reference_creators WHERE next_check_at IS NULL").fetchone()["n"] == 0
