"""Apify as a DataSource: logged-out public posts by hashtag on Instagram and TikTok.

Actor output fields vary between actor versions, so normalisation reads each field
from several possible names and tolerates missing ones. Only what the Brain needs is
kept; author handles are salted-hashed before they touch the database.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable

from psycopg.types.json import Jsonb

from .. import ledger
from ..policy import Policy

API = "https://api.apify.com/v2"


def _first(d: dict, *keys, default=None):
    for k in keys:
        cur: Any = d
        for part in k.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if cur not in (None, ""):
            return cur
    return default


def _ts(v) -> datetime | None:
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


def author_hash(handle: str | None) -> str | None:
    if not handle:
        return None
    salt = os.environ.get("MW_HASH_SALT", "manyworlds")
    return hashlib.sha256(f"{salt}:{handle.lower()}".encode()).hexdigest()[:24]


def normalize_instagram(item: dict, tag: str) -> dict | None:
    ext = _first(item, "id", "shortCode", "url")
    if not ext:
        return None
    kind = str(_first(item, "type", "productType", default="")).lower()
    media = "carousel" if "sidecar" in kind or "carousel" in kind else "video" if (
        "video" in kind or "clips" in kind or "reel" in kind) else "photo"
    tags = _first(item, "hashtags", default=[]) or []
    return {
        "platform": "instagram", "external_id": str(ext), "query_tag": tag,
        "author_hash": author_hash(_first(item, "ownerUsername", "owner.username")),
        "author_followers": _first(item, "ownerFollowersCount", "owner.followersCount"),
        "media_type": media, "caption": (_first(item, "caption", default="") or "")[:600],
        "hashtags": [str(t).lstrip("#").lower() for t in tags][:30],
        "likes": _first(item, "likesCount", "likes"), "comments": _first(item, "commentsCount", "comments"),
        "views": _first(item, "videoViewCount", "videoPlayCount", "playsCount"), "shares": None,
        "duration_s": _first(item, "videoDuration"), "posted_at": _ts(_first(item, "timestamp", "takenAt")),
    }


def normalize_tiktok(item: dict, tag: str) -> dict | None:
    ext = _first(item, "id", "webVideoUrl")
    if not ext:
        return None
    tags = [(_first(h, "name") if isinstance(h, dict) else h) for h in (_first(item, "hashtags", default=[]) or [])]
    is_photo = bool(_first(item, "isSlideshow", "imagePost"))
    return {
        "platform": "tiktok", "external_id": str(ext), "query_tag": tag,
        "author_hash": author_hash(_first(item, "authorMeta.name", "author.uniqueId")),
        "author_followers": _first(item, "authorMeta.fans", "authorMeta.followers"),
        "media_type": "carousel" if is_photo else "video",
        "caption": (_first(item, "text", "desc", default="") or "")[:600],
        "hashtags": [str(t).lower() for t in tags if t][:30],
        "likes": _first(item, "diggCount", "likes"), "comments": _first(item, "commentCount"),
        "views": _first(item, "playCount", "views"), "shares": _first(item, "shareCount"),
        "duration_s": _first(item, "videoMeta.duration"), "posted_at": _ts(_first(item, "createTimeISO", "createTime")),
    }


NORMALIZERS: dict[str, Callable[[dict, str], dict | None]] = {
    "instagram": normalize_instagram, "tiktok": normalize_tiktok,
}


def _actor_input(platform: str, tag: str, limit: int) -> dict:
    if platform == "instagram":
        return {"hashtags": [tag], "resultsLimit": limit}
    return {"hashtags": [tag], "resultsPerPage": limit, "shouldDownloadVideos": False,
            "shouldDownloadCovers": False, "shouldDownloadSubtitles": False}


def run_actor(actor: str, payload: dict, *, token: str | None = None, timeout: int = 300) -> list[dict]:
    token = token or os.environ["APIFY_TOKEN"]
    req = urllib.request.Request(
        f"{API}/acts/{actor}/run-sync-get-dataset-items?timeout={timeout - 10}",
        data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                 "User-Agent": "manyworlds-research/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"[]")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Apify {actor} -> {e.code}: {e.read().decode()[:500]}") from e


class ApifySource:
    """Pulls posts for hashtags and stores normalised items. `fetch` is injectable for tests."""

    def __init__(self, conn, policy: Policy, fetch: Callable[[str, dict], list[dict]] | None = None):
        self.conn, self.policy = conn, policy
        self.fetch = fetch or (lambda actor, payload: run_actor(actor, payload))

    def pull(self, platform: str, tag: str, limit: int) -> dict:
        actor = self.policy.get("research", "apify_actors", platform)
        query = {"platform": platform, "tag": tag, "limit": limit}
        run_id = self.conn.execute(
            "INSERT INTO research_runs (source, query) VALUES (%s, %s) RETURNING id",
            (f"apify:{actor}", Jsonb(query))).fetchone()["id"]
        try:
            raw = self.fetch(actor, _actor_input(platform, tag, limit))
        except Exception as e:  # noqa: BLE001 - a failed pull is recorded, not fatal
            self.conn.execute("UPDATE research_runs SET error = %s WHERE id = %s", (str(e)[:2000], run_id))
            return {"tag": tag, "platform": platform, "items": 0, "error": str(e)[:300]}
        rows = [r for r in (NORMALIZERS[platform](i, tag) for i in raw if isinstance(i, dict)) if r]
        for r in rows:
            self.conn.execute(
                """
                INSERT INTO research_items (run_id, platform, external_id, query_tag, author_hash,
                    author_followers, media_type, caption, hashtags, likes, comments, views, shares,
                    duration_s, posted_at)
                VALUES (%(run)s, %(platform)s, %(external_id)s, %(query_tag)s, %(author_hash)s,
                    %(author_followers)s, %(media_type)s, %(caption)s, %(hashtags)s, %(likes)s,
                    %(comments)s, %(views)s, %(shares)s, %(duration_s)s, %(posted_at)s)
                ON CONFLICT (platform, external_id) DO UPDATE SET
                    likes = EXCLUDED.likes, comments = EXCLUDED.comments, views = EXCLUDED.views,
                    shares = EXCLUDED.shares, fetched_at = now()
                """, {**r, "run": run_id})
        usd = len(raw) * float(self.policy.get("research", "usd_per_1k_items", platform, default=2.5)) / 1000
        self.conn.execute("UPDATE research_runs SET items = %s, usd = %s WHERE id = %s", (len(rows), usd, run_id))
        ledger.record(self.conn, category="data", provider=f"apify:{platform}", quantity=len(raw),
                      unit="item", usd=usd, policy_version=self.policy.version)
        return {"tag": tag, "platform": platform, "items": len(rows)}


def purge_expired(conn) -> int:
    """Delete raw items past their retention date (GDPR guardrail)."""
    return len(conn.execute("DELETE FROM research_items WHERE purge_after < now() RETURNING id").fetchall())
