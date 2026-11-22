"""The trend watcher, with ScrapeCreators and the models faked: collection with velocity, outliers vs controls,
the daily brief and its findings, the hard daily budget, and privacy (no handles stored)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from manyworlds import ledger, policy as policy_mod
from manyworlds.brain import evidence
from manyworlds.research import scrapecreators, trends

from brain_helpers import StubLLM, creator


def _pol(policy, **over):
    cfg = {**policy.body["trends"], "enabled": True, "markets": [{"name": "night routine", "queries": ["night routine"],
                                                                  "platforms": ["tiktok"]}],
           "trending_feed": False, "min_n": 2, "min_outliers_for_brief": 3, **over}
    return policy_mod.from_dict({**policy.body, "trends": cfg})


def _aweme(i, views, hook=False):
    return {"aweme_info": {"aweme_id": f"v{i}", "desc": f"post {i}", "create_time": int(datetime.now(timezone.utc).timestamp()) - 3600,
                           "statistics": {"play_count": views, "digg_count": views // 10, "comment_count": 3,
                                          "share_count": 1, "collect_count": 2},
                           "author": {"unique_id": f"person{i}", "follower_count": 1000},
                           "music": {"id_str": "s1" if hook else f"s{i + 100}", "title": "song"},
                           "video": {"duration": 15000, "cover": {"url_list": []}}}}


class FakeSource(scrapecreators.ScrapeCreators):
    def __init__(self, scale=1):
        self.calls, self.scale = [], scale
        super().__init__(fetch=self._fetch)

    def _fetch(self, path, params):
        self.calls.append(path)
        if path == "/v1/tiktok/search/keyword":
            items = [_aweme(i, (100_000 if i < 6 else 1_000) * self.scale, hook=i < 6) for i in range(30)]
            return {"search_item_list": items, "cursor": 30}
        if path == "/v1/tiktok/video/transcript":
            return {"transcript_with_timestamp": [{"start": 0, "end": 2, "text": "nobody tells you this"}]}
        raise AssertionError(path)


class FeatureLLM(StubLLM):
    """Outliers (views >= 100k) are question hooks on talking heads; the rest are aesthetic montages."""

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema, effort=effort, creator_id=creator_id)
        if role == "trend_features":
            import json
            posts = [json.loads(b["text"]) for b in content if b.get("type") == "text"]
            out = []
            for p in posts:
                big = int(p["caption_opening"].split()[-1]) < 6
                out.append({"key": p["key"], "format": "talking_head" if big else "aesthetic_montage",
                            "hook_type": "question" if big else "none", "hook_summary": "asks a blunt question",
                            "opens_on_face": big, "on_screen_text": True, "speaks_to_camera": big, "people": 1,
                            "setting": "kitchen", "tone": "deadpan", "topic": "nights", "pacing": "steady",
                            "call_to_action": False, "why_it_works": "direct" if big else ""})
            return schema.model_validate({"posts": out})
        return schema.model_validate({
            "working": [{"statement": "Question hooks on talking heads are 3x over-represented among outliers (n=6)",
                         "feature": "hook_type", "value": "question", "lift": 3.0, "n": 6}],
            "rising": [], "cooling": [], "hooks_to_try": [{"pattern": "blunt question", "example": "why is 4am so loud?"}],
            "avoid": ["silent montages"], "summary": "Talk to camera, open with a question."})


def test_collection_stores_posts_without_handles_and_snapshots_for_velocity(conn, policy):
    pol, src, now = _pol(policy, refresh_hours=0), FakeSource(), datetime.now(timezone.utc)
    out = trends.collect(conn, pol, src, now=now)
    assert out["requests"] == 1 and out["items"] == 30
    trends.collect(conn, pol, FakeSource(scale=2), now=now + timedelta(hours=1))
    row = conn.execute("SELECT * FROM research_items WHERE external_id = 'v0'").fetchone()
    assert row["author_hash"] and "person0" not in str(row.values()) and row["market"] == "night routine"
    assert row["duration_s"] == 15 and row["views"] == 200_000 and row["sound_id"] == "s1"
    assert conn.execute("SELECT count(*) n FROM research_snapshots WHERE item_id = %s", (row["id"],)).fetchone()["n"] == 2
    spent = conn.execute("SELECT sum(usd) u, min(agent) a FROM cost_entries WHERE agent LIKE 'trends%'").fetchone()
    assert spent["a"] == "trends:collect" and float(spent["u"]) == pytest.approx(2 * pol.get("trends", "usd_per_credit"))


def test_a_niche_is_not_searched_again_before_its_refresh(conn, policy):
    pol, src, now = _pol(policy), FakeSource(), datetime.now(timezone.utc)
    trends.collect(conn, pol, src, now=now)
    assert trends.collect(conn, pol, src, now=now)["requests"] == 0 and len(src.calls) == 1


def test_outliers_and_controls_are_described_and_the_brief_becomes_findings(conn, policy):
    pol, src, now = _pol(policy, controls_per_group=6), FakeSource(), datetime.now(timezone.utc)
    llm = FeatureLLM()
    trends.collect(conn, pol, src, now=now)
    a = trends.analyze(conn, pol, llm, src, now=now)
    got = conn.execute("SELECT sample, count(*) n FROM research_features GROUP BY sample").fetchall()
    assert {r["sample"]: r["n"] for r in got} == {"outlier": 6, "control": 6} and a["transcripts"] == 12
    assert {c["agent"] for c in llm.calls} == {"trends:analyze"}
    s = trends.stats(conn, pol, "night routine", now)
    top = s["patterns"][0]
    assert (top["feature"], top["share_of_outliers"], top["share_of_controls"]) in {
        ("format", 1.0, 0.0), ("hook_type", 1.0, 0.0), ("opens_on_face", 1.0, 0.0), ("speaks_to_camera", 1.0, 0.0)}
    out = trends.synthesize(conn, pol, llm, now=now)
    assert out == {"briefs": 1, "findings": 1}
    f = conn.execute("SELECT * FROM findings WHERE evidence = 'research'").fetchone()
    assert f["scope"] == "market" and f["grade"] == "speculative" and f["n"] == 6
    assert trends.synthesize(conn, pol, llm, now=now)["briefs"] == 0          # one brief per niche per day
    cid = creator(conn)
    niche = evidence.niche_trends(conn, cid, pol)
    assert niche["niche"] == "night routine" and niche["hooks_to_try"][0]["example"] == "why is 4am so loud?"


def test_the_daily_budget_stops_every_stage(conn, policy):
    pol, src, now = _pol(policy, daily_usd_max=1.0), FakeSource(), datetime.now(timezone.utc)
    ledger.record(conn, category="data", provider="scrapecreators:collect", quantity=1, unit="credit", usd=0.99,
                  agent="trends:collect")
    out = trends.tick(conn, pol, FeatureLLM(), src, now=now)
    assert out["collect"]["requests"] == 0 and out["collect"]["budget"] and src.calls == []
    assert out["analyze"]["described"] == 0


def test_nothing_runs_until_the_owner_enables_it(conn, policy):
    src = FakeSource()
    assert trends.tick(conn, policy, FeatureLLM(), src) == {"enabled": False} and src.calls == []


def test_tiktok_and_instagram_payloads_normalise():
    t = scrapecreators.normalize_tiktok(_aweme(3, 5000), "m", "q")
    assert (t["views"], t["saves"], t["duration_s"], t["platform"]) == (5000, 2, 15.0, "tiktok")
    ig = scrapecreators.normalize_instagram({"id": "1", "code": "abc", "like_count": 9, "video_duration": 12.5,
                                             "owner": {"username": "x", "follower_count": 50},
                                             "clips_music_attribution_info": {"audio_id": "7", "song_name": "a",
                                                                              "artist_name": "b"}}, "m", "q")
    assert ig["url"] == "https://www.instagram.com/reel/abc/" and ig["sound_title"] == "a - b" and ig["likes"] == 9
