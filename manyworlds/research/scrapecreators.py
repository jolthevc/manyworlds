"""ScrapeCreators as a data source: public TikTok and Instagram posts, one credit per request.

Per its docs (read 2026-11-21): base https://api.scrapecreators.com, key in the `x-api-key` header, no rate limit,
HTTP 402 when credits run out. A TikTok keyword search page or trending feed returns many videos for one credit,
which is what makes it much cheaper than per-item scrapers for watching trends. Logged-out public data only;
author handles are salted-hashed before they touch the database (research.apify.author_hash).

Endpoints used: /v1/tiktok/search/keyword (query, date_posted, sort_by, region, cursor), /v1/tiktok/get-trending-feed
(region), /v1/tiktok/video/transcript (url), /v2/instagram/reels/search (query, date_posted, page). Field names
vary, so normalisation reads each from several names and tolerates missing ones.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .apify import _first, _ts, author_hash

API = "https://api.scrapecreators.com"


class OutOfCredits(RuntimeError):
    pass


def get(path: str, params: dict, *, key: str | None = None, timeout: int = 60) -> dict:
    key = key or os.environ.get("SCRAPECREATORS_API_KEY")
    if not key:
        raise RuntimeError("SCRAPECREATORS_API_KEY is not set")
    url = f"{API}{path}?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(url, headers={"x-api-key": key, "User-Agent": "manyworlds-research/0.2"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300].replace(key, "<key>")
        if e.code == 402:
            raise OutOfCredits("ScrapeCreators: out of credits") from None
        raise RuntimeError(f"ScrapeCreators {path} -> {e.code}: {body}") from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        raise RuntimeError(f"ScrapeCreators {path} -> {type(e).__name__}") from None


def _url_list(v) -> str | None:
    lst = (v or {}).get("url_list") if isinstance(v, dict) else None
    return lst[0] if lst else None


def normalize_tiktok(item: dict, market: str, query: str) -> dict | None:
    a = item.get("aweme_info") or item.get("aweme_detail") or item
    ext = _first(a, "aweme_id", "id")
    if not ext:
        return None
    st = a.get("statistics") or {}
    author = a.get("author") or {}
    music = a.get("music") or {}
    dur = _first(a, "video.duration", "duration")
    dur = float(dur) / 1000 if dur and float(dur) > 600 else dur     # milliseconds in TikTok's own payload
    handle = _first(author, "unique_id", "uniqueId")
    tags = [t.get("hashtag_name") for t in (a.get("text_extra") or []) if isinstance(t, dict) and t.get("hashtag_name")]
    return {
        "platform": "tiktok", "external_id": str(ext), "query_tag": query, "market": market, "source": "scrapecreators",
        "author_hash": author_hash(handle), "author_followers": _first(author, "follower_count") or None,
        "media_type": "carousel" if a.get("image_post_info") else "video",
        "caption": (a.get("desc") or "")[:600], "hashtags": [str(t).lower() for t in tags][:30],
        "likes": _first(st, "digg_count"), "comments": _first(st, "comment_count"), "views": _first(st, "play_count"),
        "shares": _first(st, "share_count"), "saves": _first(st, "collect_count"), "duration_s": dur,
        "posted_at": _ts(a.get("create_time")), "sound_id": str(_first(music, "id_str", "id") or "") or None,
        "sound_title": (music.get("title") or None), "cover_url": _url_list((a.get("video") or {}).get("cover")),
        "url": f"https://www.tiktok.com/@{handle}/video/{ext}" if handle else a.get("share_url"),
    }


def normalize_instagram(item: dict, market: str, query: str) -> dict | None:
    ext = _first(item, "id", "pk", "shortcode", "code")
    if not ext:
        return None
    owner = item.get("owner") or item.get("user") or {}
    music = item.get("clips_music_attribution_info") or {}
    code = _first(item, "shortcode", "code")
    cap = item.get("caption")
    cap = cap.get("text") if isinstance(cap, dict) else cap
    return {
        "platform": "instagram", "external_id": str(ext), "query_tag": query, "market": market, "source": "scrapecreators",
        "author_hash": author_hash(_first(owner, "username")), "author_followers": _first(owner, "follower_count"),
        "media_type": "video", "caption": (cap or "")[:600], "hashtags": [],
        "likes": _first(item, "like_count"), "comments": _first(item, "comment_count"),
        "views": _first(item, "play_count", "video_play_count", "view_count", "video_view_count"),
        "shares": None, "saves": None, "duration_s": _first(item, "video_duration"),
        "posted_at": _ts(_first(item, "taken_at", "taken_at_timestamp")),
        "sound_id": str(music.get("audio_id") or "") or None,
        "sound_title": " - ".join(x for x in (music.get("song_name"), music.get("artist_name")) if x) or None,
        "cover_url": _first(item, "display_url", "thumbnail_src"),
        "url": f"https://www.instagram.com/reel/{code}/" if code else item.get("url"),
    }


class ScrapeCreators:
    """The calls the trend watcher makes. `fetch(path, params)` is injectable for tests."""

    def __init__(self, fetch=None):
        self.fetch = fetch or get

    def tiktok_search(self, query: str, *, market: str, region: str, date_posted: str, sort_by: str,
                      cursor: Any = None) -> tuple[list[dict], Any]:
        r = self.fetch("/v1/tiktok/search/keyword", {"query": query, "date_posted": date_posted, "sort_by": sort_by,
                                                      "region": region, "cursor": cursor})
        items = [x for x in (normalize_tiktok(i, market, query) for i in r.get("search_item_list") or []) if x]
        return items, r.get("cursor")

    def tiktok_trending(self, *, market: str, region: str) -> list[dict]:
        r = self.fetch("/v1/tiktok/get-trending-feed", {"region": region})
        return [x for x in (normalize_tiktok(i, market, "trending") for i in r.get("aweme_list") or []) if x]

    def instagram_reels(self, query: str, *, market: str, date_posted: str, page: int = 1) -> list[dict]:
        r = self.fetch("/v2/instagram/reels/search", {"query": query, "date_posted": date_posted, "page": page})
        return [x for x in (normalize_instagram(i, market, query) for i in r.get("reels") or []) if x]

    def transcript(self, platform: str, url: str) -> str | None:
        """The spoken words, or None. TikTok's native captions cost one credit; no paid AI transcription."""
        if platform != "tiktok" or not url:
            return None
        r = self.fetch("/v1/tiktok/video/transcript", {"url": url, "use_ai_as_fallback": "false"})
        segs = r.get("transcript_with_timestamp")
        if segs:
            return " ".join(str(s.get("text", "")).strip() for s in segs if float(s.get("start") or 0) < 20)[:600]
        text = r.get("transcript") or ""
        lines = [ln for ln in text.splitlines() if ln.strip() and "-->" not in ln and ln.strip() != "WEBVTT"]
        return " ".join(lines)[:600] or None
