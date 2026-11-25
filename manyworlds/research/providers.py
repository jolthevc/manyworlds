"""Provider-agnostic acquisition of public creator data (docs/research/2026-11-25-research-providers.md).

Three tasks, one shape of answer whichever provider serves them:

    fetch_profiles(platform, handles)        -> [profile]   {handle, followers, display_name, bio, profile_url, ...}
    fetch_posts(platform, handle, n, since)  -> [post]      the shape manyworlds.research.pulls stores (external_id,
                                                            url, media_type, posted_at, caption, metrics, images,
                                                            video_url, handle, followers) plus shortcode, hashtags,
                                                            children, paid_partnership, product_type
    search(platform, query)                  -> [post]      a hashtag or keyword page

A route is an ordered list of providers per task (policy `research.providers.routes.<task>`). The router tries them in
order and falls back on an error, an unavailable provider (no key, no connected account, over its hourly limit) or an
answer missing the task's required fields (`research.providers.required.<task>`, for at least `min_complete_share`
of the items). Every call that may cost money is written to the ledger (category data, provider
`<provider>:<task>`, the caller's agent tag and job), failed calls included.

Adapters: `apify` (the actors the pilot uses: logged-out public scraping), `scrapecreators` (per-request credits;
manyworlds.research.scrapecreators.get), `graph` (Instagram Graph API Business Discovery through our own connected
professional account: free, official, Business/Creator accounts only). HikerAPI is not built: it reads Instagram's
private mobile API through managed sessions and its terms forbid commercial use without written permission
(`research.providers.catalog.hikerapi`). Each adapter's transport is injectable for tests.

Instagram post ids: Apify and ScrapeCreators give the numeric media id; Business Discovery gives a shortcode (and its
own Graph id). `shortcode_to_id` turns a shortcode into the media id, so a post is one row whoever fetched it.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .. import ledger
from ..policy import Policy
from . import pulls
from .apify import _first, _ts

TASKS = ("profiles", "posts", "posts_media", "search")
_B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"


class Unavailable(RuntimeError):
    """The provider cannot serve this call here (no key, no account, wrong platform, over its limit)."""


class Incomplete(RuntimeError):
    """The answer lacks the task's required fields."""


def shortcode_to_id(code: str | None) -> str | None:
    """An Instagram shortcode as its numeric media id (the classic 11-character codes; longer ones are returned as
    they are)."""
    if not code:
        return None
    code = str(code)
    if len(code) > 12 or any(ch not in _B64 for ch in code):
        return code
    n = 0
    for ch in code:
        n = n * 64 + _B64.index(ch)
    return str(n)


def _cfg(policy: Policy) -> dict:
    return policy.get("research", "providers", default={}) or {}


def _catalog(policy: Policy, name: str) -> dict:
    return (_cfg(policy).get("catalog") or {}).get(name) or {}


def _tags(caption: str | None) -> list[str]:
    import re
    return [t.lower() for t in re.findall(r"#(\w+)", caption or "")][:30]


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


# --- normalisers --------------------------------------------------------------------------------------------------
def _apify_ig(item: dict) -> dict | None:
    p = pulls.normalize_instagram_post(item)
    if not p:
        return None
    children = _first(item, "childPosts", default=[]) or []
    caption = p.get("caption") or ""
    p.update(shortcode=_first(item, "shortCode"), hashtags=[str(t).lstrip("#").lower() for t in
                                                            (_first(item, "hashtags", default=[]) or [])] or _tags(caption),
             children=len(children) or (len(p["images"]) if p["media_type"] == "carousel" else 0),
             paid_partnership=bool(_first(item, "isSponsored", "paidPartnership")),
             product_type=_first(item, "productType"))
    return p


def _apify_tt(item: dict) -> dict | None:
    p = pulls.normalize_tiktok_post(item)
    if not p:
        return None
    tags = [(_first(h, "name") if isinstance(h, dict) else h) for h in (_first(item, "hashtags", default=[]) or [])]
    p.update(shortcode=None, hashtags=[str(t).lower() for t in tags if t] or _tags(p.get("caption")),
             children=len(p["images"]) if p["media_type"] == "carousel" else 0,
             paid_partnership=bool(_first(item, "isAd", "isSponsored")), product_type=None)
    return p


def _img(v) -> str | None:
    cands = (v or {}).get("candidates") if isinstance(v, dict) else None
    return cands[0].get("url") if cands and isinstance(cands[0], dict) else None


def sc_ig_post(item: dict) -> dict | None:
    """An item of ScrapeCreators' /v2/instagram/user/posts (Instagram's feed shape)."""
    pk = _first(item, "pk", "id")
    if not pk:
        return None
    pk = str(pk).split("_")[0]
    kind = int(_first(item, "media_type", default=1) or 1)
    media = {8: "carousel", 2: "video"}.get(kind, "photo")
    cap = item.get("caption")
    caption = (cap.get("text") if isinstance(cap, dict) else cap) or ""
    kids = item.get("carousel_media") or []
    images = [_img(c.get("image_versions2")) for c in kids if isinstance(c, dict)] or [_img(item.get("image_versions2"))]
    vids = item.get("video_versions") or []
    owner = item.get("user") or item.get("owner") or {}
    code = _first(item, "code", "shortcode")
    return {"external_id": pk, "shortcode": code, "url": f"https://www.instagram.com/p/{code}/" if code else None,
            "media_type": media, "posted_at": _ts(_first(item, "taken_at")), "caption": caption[:2000],
            "metrics": _clean({"likes": _first(item, "like_count"), "comments": _first(item, "comment_count"),
                               "views": _first(item, "play_count", "ig_play_count", "view_count")
                               if media == "video" else None,
                               "duration_s": _first(item, "video_duration")}),
            "images": [i for i in images if i], "video_url": (vids[0].get("url") if vids else None)
            if media == "video" else None,
            "handle": _first(owner, "username"), "followers": _first(owner, "follower_count"),
            "hashtags": _tags(caption), "children": len(kids), "product_type": _first(item, "product_type"),
            "paid_partnership": bool(_first(item, "is_paid_partnership"))}


def sc_ig_node(node: dict) -> dict | None:
    """A post in Instagram's web (GraphQL) shape: ScrapeCreators' profile and single-post answers, Apify-like keys."""
    code = _first(node, "shortcode")
    ext = shortcode_to_id(code) or _first(node, "id")
    if not ext:
        return None
    t = str(_first(node, "__typename", default="")).lower()
    media = "carousel" if "sidecar" in t else "video" if (node.get("is_video") or "video" in t) else "photo"
    edges = ((node.get("edge_media_to_caption") or {}).get("edges") or [])
    caption = ((edges[0].get("node") or {}).get("text") if edges else "") or ""
    kids = [(e.get("node") or {}) for e in ((node.get("edge_sidecar_to_children") or {}).get("edges") or [])]
    return {"external_id": str(ext), "shortcode": code, "url": f"https://www.instagram.com/p/{code}/" if code else None,
            "media_type": media, "posted_at": _ts(_first(node, "taken_at_timestamp")), "caption": caption[:2000],
            "metrics": _clean({"likes": _first(node, "edge_liked_by.count", "edge_media_preview_like.count"),
                               "comments": _first(node, "edge_media_to_comment.count",
                                                  "edge_media_to_parent_comment.count"),
                               "views": _first(node, "video_play_count", "video_view_count") if media == "video"
                               else None, "duration_s": _first(node, "video_duration")}),
            "images": [k.get("display_url") for k in kids if k.get("display_url")] or
                      ([node["display_url"]] if node.get("display_url") else []),
            "video_url": node.get("video_url") if media == "video" else None,
            "handle": _first(node, "owner.username"), "followers": None, "hashtags": _tags(caption),
            "children": len(kids), "product_type": _first(node, "product_type"),
            "paid_partnership": bool(_first(node, "is_paid_partnership"))}


def sc_tt_post(item: dict) -> dict | None:
    a = item.get("aweme_info") or item
    ext = _first(a, "aweme_id", "id")
    if not ext:
        return None
    st = a.get("statistics") or {}
    author = a.get("author") or {}
    imgs = ((a.get("image_post_info") or {}).get("images") or [])
    slides = [_first(i, "display_image.url_list.0") for i in imgs if isinstance(i, dict)]
    dur = _first(a, "video.duration", "duration")
    dur = float(dur) / 1000 if dur and float(dur) > 600 else dur
    caption = a.get("desc") or ""
    tags = [t.get("hashtag_name") for t in (a.get("text_extra") or []) if isinstance(t, dict) and t.get("hashtag_name")]
    media = "carousel" if imgs else "video"
    handle = _first(author, "unique_id")
    return {"external_id": str(ext), "shortcode": None,
            "url": f"https://www.tiktok.com/@{handle}/video/{ext}" if handle else None, "media_type": media,
            "posted_at": _ts(a.get("create_time")), "caption": caption[:2000],
            "metrics": _clean({"views": _first(st, "play_count"), "likes": _first(st, "digg_count"),
                               "comments": _first(st, "comment_count"), "shares": _first(st, "share_count"),
                               "saves": _first(st, "collect_count"), "duration_s": dur}),
            "images": [s for s in slides if s] or ([_first(a, "video.cover.url_list.0")]
                                                   if _first(a, "video.cover.url_list.0") else []),
            "video_url": _first(a, "video.play_addr.url_list.0") if media == "video" else None,
            "handle": handle, "followers": _first(author, "follower_count"),
            "hashtags": [str(t).lower() for t in tags] or _tags(caption), "children": len(slides),
            "product_type": None, "paid_partnership": bool(_first(a, "is_ads", "commerce_info.bc_label_test_text"))}


def graph_post(m: dict, handle: str, followers=None) -> dict | None:
    code = _first(m, "shortcode") or (str(m.get("permalink") or "").rstrip("/").split("/")[-1] or None)
    ext = shortcode_to_id(code) or _first(m, "id")
    if not ext:
        return None
    kind = str(m.get("media_type") or "").upper()
    media = "carousel" if kind == "CAROUSEL_ALBUM" else "video" if kind == "VIDEO" else "photo"
    kids = ((m.get("children") or {}).get("data") or [])
    caption = m.get("caption") or ""
    return {"external_id": str(ext), "shortcode": code, "url": m.get("permalink"), "media_type": media,
            "posted_at": _ts(m.get("timestamp")), "caption": caption[:2000],
            "metrics": _clean({"likes": m.get("like_count"), "comments": m.get("comments_count"),
                               "views": m.get("view_count") if media == "video" else None}),
            "images": [k.get("media_url") for k in kids if k.get("media_url") and k.get("media_type") != "VIDEO"]
                      or ([m.get("thumbnail_url") or m.get("media_url")] if media != "carousel"
                          and (m.get("thumbnail_url") or m.get("media_url")) else []),
            "video_url": m.get("media_url") if media == "video" else None,
            "handle": handle, "followers": followers, "hashtags": _tags(caption), "children": len(kids),
            "product_type": m.get("media_product_type"), "paid_partnership": None}


# --- adapters -----------------------------------------------------------------------------------------------------
class Provider:
    """An adapter. Each method returns (items, units, usd); `unit` names what units count."""
    name = "?"
    unit = "request"
    platforms: tuple = ()

    def __init__(self, policy: Policy):
        self.policy = policy
        self.cat = _catalog(policy, self.name)

    def available(self, conn, platform: str) -> str | None:
        """None when it can serve `platform` here, else why not."""
        if self.cat.get("enabled") is False:
            return "disabled"
        if platform not in (self.cat.get("platforms") or self.platforms):
            return f"no {platform}"
        env = self.cat.get("env")
        if env and not os.environ.get(env):
            return f"{env} not set"
        return None


class ApifyProvider(Provider):
    """The pilot's actors (policy intelligence.agent.actors, research.apify_actors); priced per item from policy."""
    name, unit, platforms = "apify", "item", ("instagram", "tiktok")

    def __init__(self, policy: Policy, fetch: Callable[[str, dict], list[dict]] | None = None):
        super().__init__(policy)
        from .apify import run_actor
        self.fetch = fetch or run_actor
        self.agent_cfg = policy.get("intelligence", "agent", default={}) or {}

    def _price(self, key: str) -> float:
        return float((self.agent_cfg.get("usd_per_item") or {}).get(key, 0))

    def fetch_profiles(self, platform: str, handles: list[str]):
        if platform == "instagram":
            actor = self.agent_cfg["actors"]["instagram_profile"]
            raw = [r for r in self.fetch(actor, {"usernames": list(handles)}) if isinstance(r, dict)]
            out = []
            for r in raw:
                if not _first(r, "username") or r.get("error"):
                    continue
                out.append({"platform": platform, "handle": r["username"], "followers": _first(r, "followersCount"),
                            "display_name": _first(r, "fullName"), "bio": _first(r, "biography"),
                            "posts_count": _first(r, "postsCount"), "is_business": _first(r, "isBusinessAccount"),
                            "profile_url": _first(r, "url") or f"https://www.instagram.com/{r['username']}/",
                            "latest_posts": [p for p in (_apify_ig({**x, "ownerUsername": r["username"]})
                                                         for x in (r.get("latestPosts") or [])
                                                         if isinstance(x, dict)) if p]})
            return out, len(raw), len(raw) * self._price("instagram_profile")
        out, n, usd = [], 0, 0.0
        for h in handles:                         # TikTok: one profile-scraper result carries the author
            posts, k, u = self.fetch_posts(platform, h, 1, None)
            n, usd = n + k, usd + u
            if posts:
                out.append({"platform": platform, "handle": h, "followers": posts[0].get("followers"),
                            "display_name": None, "bio": None, "profile_url": f"https://www.tiktok.com/@{h}",
                            "latest_posts": posts})
        return out, n, usd

    def fetch_posts(self, platform: str, handle: str, n: int, since: datetime | None):
        if platform == "instagram":
            lim = int(self.cat.get("profile_posts", 12))
            if n <= lim and not since:            # the profile scraper embeds the latest posts: one item, not n
                profs, k, usd = self.fetch_profiles(platform, [handle])
                return (profs[0]["latest_posts"][:n] if profs else []), k, usd
            actor = self.agent_cfg["actors"]["instagram_posts"]
            payload = {"username": [handle], "resultsLimit": n, "skipPinnedPosts": True}
            if since:
                payload["onlyPostsNewerThan"] = since.date().isoformat()
            raw = [r for r in self.fetch(actor, payload) if isinstance(r, dict)]
            posts = [p for p in (_apify_ig(r) for r in raw if not r.get("error")) if p]
            return posts, len(raw), len(raw) * self._price("instagram_posts")
        actor = self.agent_cfg["actors"]["tiktok_profile"]
        raw = [r for r in self.fetch(actor, {"profiles": [handle], "resultsPerPage": n,
                                             "shouldDownloadVideos": False}) if isinstance(r, dict)]
        posts = [p for p in (_apify_tt(r) for r in raw if not r.get("error")) if p]
        return posts, len(raw), len(raw) * self._price("tiktok_profile")

    def search(self, platform: str, query: str, n: int):
        from .apify import _actor_input
        actor = self.policy.get("research", "apify_actors", platform)
        raw = [r for r in self.fetch(actor, _actor_input(platform, query.lstrip("#"), n)) if isinstance(r, dict)]
        norm = _apify_ig if platform == "instagram" else _apify_tt
        price = float((self.agent_cfg.get("usd_per_hashtag_item") or {}).get(platform, 0))
        return [p for p in (norm(r) for r in raw) if p], len(raw), len(raw) * price


class ScrapeCreatorsProvider(Provider):
    """Per-request credits (research.providers.catalog.scrapecreators.usd_per_credit); a feed page is one credit."""
    name, unit, platforms = "scrapecreators", "credit", ("instagram", "tiktok")

    def __init__(self, policy: Policy, fetch: Callable[[str, dict], dict] | None = None):
        super().__init__(policy)
        from .scrapecreators import get
        self.fetch = fetch or get
        self.per = float(self.cat.get("usd_per_credit", 0))

    def fetch_profiles(self, platform: str, handles: list[str]):
        out = []
        for h in handles:
            if platform == "instagram":
                r = self.fetch("/v1/instagram/profile", {"handle": h, "trim": "false"})
                u = ((r.get("data") or {}).get("user")) or {}
                if not u:
                    continue
                nodes = [(e.get("node") or {}) for e in ((u.get("edge_owner_to_timeline_media") or {})
                                                         .get("edges") or [])]
                out.append({"platform": platform, "handle": u.get("username") or h,
                            "followers": _first(u, "edge_followed_by.count", "follower_count"),
                            "display_name": u.get("full_name"), "bio": u.get("biography"),
                            "posts_count": _first(u, "edge_owner_to_timeline_media.count"),
                            "is_business": u.get("is_business_account"),
                            "profile_url": f"https://www.instagram.com/{u.get('username') or h}/",
                            "latest_posts": [p for p in (sc_ig_node({**n, "owner": {"username": h}}) for n in nodes)
                                             if p]})
            else:
                r = self.fetch("/v1/tiktok/profile", {"handle": h})
                u, st = r.get("user") or {}, r.get("stats") or r.get("statsV2") or {}
                if not u:
                    continue
                out.append({"platform": platform, "handle": u.get("uniqueId") or h,
                            "followers": _first(st, "followerCount"), "display_name": u.get("nickname"),
                            "bio": u.get("signature"), "posts_count": _first(st, "videoCount"),
                            "profile_url": f"https://www.tiktok.com/@{h}", "latest_posts": []})
        return out, len(handles), len(handles) * self.per

    def fetch_posts(self, platform: str, handle: str, n: int, since: datetime | None):
        posts: list[dict] = []
        cursor, calls = None, 0
        pages = max(1, int(self.cat.get("max_pages", 5)))
        while len(posts) < n and calls < pages:
            if platform == "instagram":
                r = self.fetch("/v2/instagram/user/posts", {"handle": handle, "next_max_id": cursor, "trim": "false"})
                items = [p for p in (sc_ig_post(i) for i in r.get("items") or []) if p]
                cursor = r.get("next_max_id") if r.get("more_available") else None
            else:
                r = self.fetch("/v3/tiktok/profile/videos", {"handle": handle, "max_cursor": cursor})
                items = [p for p in (sc_tt_post(i) for i in r.get("aweme_list") or []) if p]
                cursor = r.get("max_cursor") if r.get("has_more") else None
            calls += 1
            posts += items
            if since and items and all(p["posted_at"] and p["posted_at"] < since for p in items):
                break
            if not cursor or not items:
                break
        if since:
            posts = [p for p in posts if not p["posted_at"] or p["posted_at"] >= since]
        return posts[:n], calls, calls * self.per

    def search(self, platform: str, query: str, n: int):
        if platform == "instagram":
            r = self.fetch("/v1/instagram/search/hashtag", {"hashtag": query.lstrip("#")})
            items = [p for p in (sc_ig_post(i) or sc_ig_node(i) for i in (r.get("posts") or r.get("items") or []))
                     if p]
            return items, 1, self.per
        r = self.fetch("/v1/tiktok/search/keyword", {"query": query})
        return [p for p in (sc_tt_post(i) for i in r.get("search_item_list") or []) if p], 1, self.per


GRAPH_MEDIA = ("id,shortcode,caption,like_count,comments_count,media_type,media_product_type,media_url,thumbnail_url,"
               "permalink,timestamp,view_count,children{media_url,media_type}")


class GraphProvider(Provider):
    """Instagram Graph API Business Discovery through our own connected professional account (Unified Login's system
    user). Free; reaches Business and Creator accounts only; no media_url for videos with licensed audio."""
    name, unit, platforms = "graph", "call", ("instagram",)

    def __init__(self, policy: Policy, call: Callable[[str, dict], dict] | None = None, conn=None):
        super().__init__(policy)
        self.conn = conn
        self.call = call

    def _ig_id(self, conn) -> str | None:
        row = conn.execute("SELECT external_id FROM accounts WHERE platform = 'instagram' AND status = 'connected' "
                           "AND external_id IS NOT NULL ORDER BY created_at LIMIT 1").fetchone()
        return row["external_id"] if row else None

    def available(self, conn, platform: str) -> str | None:
        why = super().available(conn, platform)
        if why:
            return why
        if self.call is None and not self._ig_id(conn):
            return "no connected Instagram account"
        per_hour = int(self.cat.get("max_calls_per_hour", 0) or 0)
        if per_hour:
            used = conn.execute("SELECT coalesce(sum(quantity), 0) n FROM cost_entries WHERE provider LIKE 'graph:%%' "
                                "AND created_at > now() - interval '1 hour'").fetchone()["n"]
            if float(used) >= per_hour:
                return "hourly call limit"
        return None

    def _get(self, conn, fields: str) -> dict:
        if self.call is not None:
            return self.call("business_discovery", {"fields": fields})
        from ..publish import instagram
        instagram._use(self.policy)
        token = os.environ[self.cat.get("env", "META_SYSTEM_USER_TOKEN")]
        return instagram.graph("GET", f"/{self._ig_id(conn)}", {"fields": fields}, token=token)

    def _user(self, handle: str, n: int, after: str | None = None) -> str:
        page = f".after({after})" if after else ""
        return (f"business_discovery.username({handle}){{username,name,biography,followers_count,media_count,"
                f"media{page}.limit({n}){{{GRAPH_MEDIA}}}}}")

    def fetch_profiles(self, platform: str, handles: list[str]):
        out = []
        for h in handles:
            bd = (self._get(self.conn, self._user(h, int(self.cat.get("profile_posts", 12)))) or {}) \
                .get("business_discovery") or {}
            if not bd:
                continue
            f = bd.get("followers_count")
            out.append({"platform": platform, "handle": bd.get("username") or h, "followers": f,
                        "display_name": bd.get("name"), "bio": bd.get("biography"),
                        "posts_count": bd.get("media_count"), "is_business": True,
                        "profile_url": f"https://www.instagram.com/{h}/",
                        "latest_posts": [p for p in (graph_post(m, h, f) for m in
                                                     ((bd.get("media") or {}).get("data") or [])) if p]})
        return out, len(handles), 0.0

    def fetch_posts(self, platform: str, handle: str, n: int, since: datetime | None):
        posts, after, calls = [], None, 0
        while len(posts) < n and calls < int(self.cat.get("max_pages", 5)):
            bd = (self._get(self.conn, self._user(handle, min(n, 50), after)) or {}).get("business_discovery") or {}
            calls += 1
            media = bd.get("media") or {}
            posts += [p for p in (graph_post(m, handle, bd.get("followers_count")) for m in media.get("data") or [])
                      if p]
            after = ((media.get("paging") or {}).get("cursors") or {}).get("after")
            if not after or (since and posts and posts[-1]["posted_at"] and posts[-1]["posted_at"] < since):
                break
        if since:
            posts = [p for p in posts if not p["posted_at"] or p["posted_at"] >= since]
        return posts[:n], calls, 0.0

    def search(self, platform: str, query: str, n: int):
        raise Unavailable("hashtag search needs Instagram Public Content Access (App Review); not used")


ADAPTERS = {"apify": ApifyProvider, "scrapecreators": ScrapeCreatorsProvider, "graph": GraphProvider}


# --- the router ---------------------------------------------------------------------------------------------------
def _has(item: dict, path: str) -> bool:
    cur: Any = item
    for part in path.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur not in (None, "", [], {})


def complete(policy: Policy, task: str, items: list[dict]) -> bool:
    need = list((_cfg(policy).get("required") or {}).get(task) or [])
    if not need or not items:
        return bool(items) or not need
    share = float(_cfg(policy).get("min_complete_share", 0.8))
    ok = sum(1 for i in items if all(_has(i, k) for k in need))
    return ok >= share * len(items)


class Router:
    """Routes each task through `research.providers.routes.<task>`, records every call in the ledger, and keeps a
    log of what was tried (`self.log`). `adapters` maps names to built adapters (tests inject fakes)."""

    def __init__(self, conn, policy: Policy, *, adapters: dict[str, Provider] | None = None, agent: str = "research",
                 job_id=None):
        self.conn, self.policy, self.agent, self.job_id = conn, policy, agent, job_id
        self.adapters = adapters if adapters is not None else {n: cls(policy) if n != "graph" else cls(policy, conn=conn)
                                                                for n, cls in ADAPTERS.items()}
        self.log: list[dict] = []

    def route(self, task: str, platform: str) -> list[str]:
        r = (_cfg(self.policy).get("routes") or {}).get(task) or []
        return list(r.get(platform) or []) if isinstance(r, dict) else list(r)

    def _charge(self, name: str, task: str, units: float, usd: float) -> None:
        if units <= 0 and usd <= 0:
            return
        unit = getattr(self.adapters.get(name), "unit", "request")
        ledger.record(self.conn, category="data", provider=f"{name}:{task}", quantity=units, unit=unit,
                      usd=round(float(usd), 6), job_id=self.job_id, policy_version=self.policy.version,
                      agent=self.agent)

    def _run(self, task: str, platform: str, call: Callable[[Provider], tuple], *, check: str | None = None):
        """The first complete answer on the route; else the first incomplete one (already paid for); else none."""
        errors, partial = [], None
        for name in self.route(task, platform):
            a = self.adapters.get(name)
            why = "no adapter" if a is None else a.available(self.conn, platform)
            if why:
                errors.append({"provider": name, "skipped": why})
                continue
            t0 = time.monotonic()
            try:
                items, units, usd = call(a)
            except Exception as e:  # noqa: BLE001 - one provider's failure falls through to the next
                # A failed call may still have been billed: the adapter cannot say, so charge its estimate of one
                # request where the provider bills per request (not Apify, which bills results).
                if a.unit in ("credit", "request"):
                    self._charge(name, task, 1, float(a.cat.get("usd_per_credit", 0)))
                errors.append({"provider": name, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                self.log.append({"task": task, "provider": name, "ok": False, "s": round(time.monotonic() - t0, 2)})
                continue
            self._charge(name, task, units, usd)
            self.log.append({"task": task, "provider": name, "ok": True, "units": units, "usd": round(usd, 6),
                             "items": len(items), "s": round(time.monotonic() - t0, 2)})
            if not complete(self.policy, check or task, items if isinstance(items, list) else [items]):
                errors.append({"provider": name, "incomplete": True})
                partial = partial or (items, name)
                continue
            return items, name, errors
        if partial and partial[0]:
            return partial[0], partial[1], errors
        return [], None, errors

    def fetch_profiles(self, platform: str, handles: list[str]) -> tuple[list[dict], str | None, list]:
        return self._run("profiles", platform, lambda a: a.fetch_profiles(platform, handles))

    def fetch_posts(self, platform: str, handle: str, n: int, since: datetime | None = None, *,
                    media: bool = False) -> tuple[list[dict], str | None, list]:
        task = "posts_media" if media else "posts"
        return self._run(task, platform, lambda a: a.fetch_posts(platform, handle, n, since))

    def search(self, platform: str, query: str, n: int = 30) -> tuple[list[dict], str | None, list]:
        return self._run("search", platform, lambda a: a.search(platform, query, n))


def since_days(days: float | None) -> datetime | None:
    return datetime.now(timezone.utc) - timedelta(days=float(days)) if days else None
