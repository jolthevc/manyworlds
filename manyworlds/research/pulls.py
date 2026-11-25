"""Per-creator pulls for the influencer intelligence agent: named public creators and their recent posts, from Apify
(logged-out public data), into reference_creators and reference_posts, with research copies of their media in the
research store (manyworlds.research.references).

The actors and their per-item prices are policy (`intelligence.agent.actors`, `usd_per_item`, verified on the
Apify store 2026-11-24):
    instagram_profile  apify~instagram-profile-scraper   {usernames: [...]}                  a profile per item
    instagram_posts    apify~instagram-post-scraper      {username: [...], resultsLimit, onlyPostsNewerThan, skipPinnedPosts}
    tiktok_profile     clockworks~tiktok-profile-scraper {profiles: [...], resultsPerPage, shouldDownloadVideos}
Candidate pools come from the hashtag actors the trend watcher uses (research.apify_actors). Output fields vary
between actor versions, so each field is read from several names and missing ones are tolerated. An account that
asked to be forgotten (research_forget) is never pulled.

    candidates(...)         creators found in hashtag pulls: handle, followers, mean engagement, posts seen
    fresh(conn, ...)        a creator already pulled within intelligence.agent.repull_after_days: reused, not pulled
    pull_creator(...)      one creator's profile and posts, stored; media copied to the research store
    engagement(post)        the traction number a post is compared on: views for a video, else likes+comments+shares
    baselines(conn, cid)    each post's engagement over its creator's median (reference_posts.metrics.over_own_median)
"""

from __future__ import annotations

import hashlib
import statistics
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from psycopg.types.json import Jsonb

from .. import ledger
from ..policy import Policy
from . import data_policy, references
from .apify import _first, _ts

Fetch = Callable[[str, dict], list[dict]]


def _cfg(policy: Policy) -> dict:
    return policy.get("intelligence", "agent", default={}) or {}


def _price(policy: Policy, name: str) -> float:
    return float((_cfg(policy).get("usd_per_item") or {}).get(name, 0))


def charge(conn, policy: Policy, *, provider: str, items: int, usd: float, job_id=None, stage: str = "scout") -> None:
    ledger.record(conn, category="data", provider=provider, quantity=items, unit="item", usd=round(usd, 6),
                  job_id=job_id, policy_version=policy.version, agent=f"intelligence:{stage}")


# --- normalisation ------------------------------------------------------------------------------------------------
def normalize_instagram_post(item: dict) -> dict | None:
    ext = _first(item, "id", "shortCode")
    if not ext:
        return None
    kind = str(_first(item, "type", "productType", default="")).lower()
    media = "carousel" if "sidecar" in kind or "carousel" in kind else "video" if (
        "video" in kind or "clips" in kind or "reel" in kind) else "photo"
    images = [_first(c, "displayUrl", "imageUrl") for c in (_first(item, "childPosts", default=[]) or [])
              if isinstance(c, dict)] or list(_first(item, "images", default=[]) or [])
    if not images and _first(item, "displayUrl"):
        images = [_first(item, "displayUrl")]
    return {"external_id": str(ext), "url": _first(item, "url"), "media_type": media,
            "posted_at": _ts(_first(item, "timestamp", "takenAt")),
            "caption": (_first(item, "caption", default="") or "")[:2000],
            "metrics": {k: v for k, v in {"likes": _first(item, "likesCount", "likes"),
                                          "comments": _first(item, "commentsCount", "comments"),
                                          "views": _first(item, "videoViewCount", "videoPlayCount", "playsCount"),
                                          "duration_s": _first(item, "videoDuration")}.items() if v is not None},
            "images": [str(i) for i in images if i],
            "video_url": _first(item, "videoUrl") if media == "video" else None,
            "handle": _first(item, "ownerUsername", "owner.username"),
            "followers": _first(item, "ownerFollowersCount", "owner.followersCount")}


def normalize_tiktok_post(item: dict) -> dict | None:
    ext = _first(item, "id")
    if not ext:
        return None
    slides = [_first(i, "imageURL.urlList.0", "url") if isinstance(i, dict) else i
              for i in (_first(item, "slideshowImageLinks", "imagePost.images", default=[]) or [])]
    is_photo = bool(_first(item, "isSlideshow")) or bool(slides)
    media_urls = _first(item, "mediaUrls", default=[]) or []
    return {"external_id": str(ext), "url": _first(item, "webVideoUrl"),
            "media_type": "carousel" if is_photo else "video",
            "posted_at": _ts(_first(item, "createTimeISO", "createTime")),
            "caption": (_first(item, "text", "desc", default="") or "")[:2000],
            "metrics": {k: v for k, v in {"views": _first(item, "playCount"), "likes": _first(item, "diggCount"),
                                          "comments": _first(item, "commentCount"),
                                          "shares": _first(item, "shareCount"), "saves": _first(item, "collectCount"),
                                          "duration_s": _first(item, "videoMeta.duration")}.items() if v is not None},
            "images": [str(s) for s in slides if s] or ([_first(item, "videoMeta.coverUrl")]
                                                         if _first(item, "videoMeta.coverUrl") else []),
            "video_url": (media_urls[0] if media_urls else _first(item, "videoMeta.downloadAddr")) if not is_photo
            else None,
            "handle": _first(item, "authorMeta.name", "author.uniqueId"),
            "followers": _first(item, "authorMeta.fans", "authorMeta.followers"),
            "display_name": _first(item, "authorMeta.nickName"), "bio": _first(item, "authorMeta.signature")}


NORMALIZE = {"instagram": normalize_instagram_post, "tiktok": normalize_tiktok_post}


def engagement(post: dict) -> float:
    m = post.get("metrics") or {}
    if post.get("media_type") == "video" and m.get("views"):
        return float(m["views"])
    return float(sum(float(m.get(k) or 0) for k in ("likes", "comments", "shares")))


# --- candidates from hashtag pulls --------------------------------------------------------------------------------
def candidates(conn, policy: Policy, fetch: Fetch, hashtags: dict[str, list[str]], *, job_id=None,
               budget_usd: float | None = None) -> list[dict]:
    """Creators seen in hashtag pulls (the trend watcher's actors, research.apify_actors), with what one pull shows:
    followers, mean engagement and posts seen. Forgotten accounts are left out. Every item is charged. With
    budget_usd (intelligence.agent.max_candidate_pool_usd), a hashtag is pulled only while the pool's spend so far
    plus that hashtag's full price stays within it; the rest are left out."""
    from .apify import _actor_input
    data_policy.require(policy, "apify", "search")
    per = int(_cfg(policy).get("items_per_hashtag", 50))
    seen: dict[tuple[str, str], dict] = {}
    spent = 0.0
    for platform, tags in hashtags.items():
        actor = policy.get("research", "apify_actors", platform)
        price = float((_cfg(policy).get("usd_per_hashtag_item") or {}).get(platform, 0))
        for tag in tags:
            if budget_usd is not None and spent + per * price > float(budget_usd):
                continue
            raw = fetch(actor, _actor_input(platform, tag, per))
            spent += len(raw) * price
            charge(conn, policy, provider=f"apify:{actor}", items=len(raw), usd=len(raw) * price, job_id=job_id)
            for item in raw:
                p = NORMALIZE[platform](item) if isinstance(item, dict) else None
                if not p or not p.get("handle") or references.forgotten(conn, platform, p["handle"]):
                    continue
                c = seen.setdefault((platform, str(p["handle"]).lower()),
                                    {"platform": platform, "handle": p["handle"], "followers": p.get("followers"),
                                     "engagement": [], "tags": set()})
                c["engagement"].append(engagement(p))
                c["tags"].add(tag)
                c["followers"] = c["followers"] or p.get("followers")
    out = []
    for c in seen.values():
        out.append({**c, "engagement": statistics.mean(c["engagement"]), "posts_seen": len(c["engagement"]),
                    "tags": sorted(c["tags"])})
    return out


def select(policy: Policy, pool: list[dict], *, named: list[dict] = (), n: int | None = None) -> list[dict]:
    """The sample by role quotas (intelligence.agent.quotas): named references always, then leaders (most
    followers), mid-sized (in mid_band) by engagement, adjacent (other hashtags' creators), and low comparators
    (the bottom of engagement for their size), always sampled. Each gets `role`."""
    cfg = _cfg(policy)
    n = int(n or cfg.get("creators", 40))
    quotas = cfg.get("quotas") or {}
    lo, hi = (cfg.get("mid_band") or [10000, 150000])[:2]
    chosen: dict[tuple, dict] = {}

    def take(rows, role, k):
        for r in rows:
            if k <= 0 or len(chosen) >= n:
                return
            key = (r["platform"], str(r["handle"]).lower())
            if key not in chosen:
                chosen[key] = {**r, "role": role}
                k -= 1

    take([{**c, "named": True} for c in named], "other", len(named))
    rest = [c for c in pool if (c["platform"], str(c["handle"]).lower()) not in chosen]
    rate = lambda c: c["engagement"] / max(float(c.get("followers") or 0), 1.0)   # noqa: E731
    k = lambda role: max(1, round(n * float(quotas.get(role, 0)))) if quotas.get(role) else 0   # noqa: E731
    take(sorted(rest, key=rate), "low_comparator", k("low_comparator"))
    take(sorted(rest, key=lambda c: -(c.get("followers") or 0)), "leader", k("leader"))
    take(sorted([c for c in rest if lo <= (c.get("followers") or 0) <= hi], key=lambda c: -rate(c)), "mid", k("mid"))
    take(sorted(rest, key=lambda c: -rate(c)), "riser", k("riser"))
    take(sorted(rest, key=lambda c: -c["posts_seen"]), "adjacent", n)
    return list(chosen.values())


# --- one creator --------------------------------------------------------------------------------------------------
def default_download(url: str, dest: Path, max_bytes: int) -> Path | None:
    req = urllib.request.Request(url, headers={"User-Agent": "manyworlds-research/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read(max_bytes + 1)
    if len(data) > max_bytes:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def upsert_creator(conn, platform: str, handle: str, **fields) -> str:
    cols = {k: v for k, v in fields.items() if v is not None}
    if cols.get("bio"):   # a public bio without contact details (manyworlds.research.data_policy: red)
        cols["bio"] = data_policy.clean_bio(cols["bio"])
    row = conn.execute(
        """INSERT INTO reference_creators (platform, handle, display_name, profile_url, bio, followers, role,
                                           selected_because, found_via)
           VALUES (%(platform)s, %(handle)s, %(display_name)s, %(profile_url)s, %(bio)s, %(followers)s,
                   coalesce(%(role)s, 'other'), %(why)s, %(found)s)
           ON CONFLICT (platform, lower(handle)) DO UPDATE SET
               display_name = coalesce(EXCLUDED.display_name, reference_creators.display_name),
               bio = coalesce(EXCLUDED.bio, reference_creators.bio),
               followers = coalesce(EXCLUDED.followers, reference_creators.followers),
               role = CASE WHEN %(role)s IS NULL THEN reference_creators.role ELSE EXCLUDED.role END,
               last_seen = now(), updated_at = now()
           RETURNING id""",
        {"platform": platform, "handle": handle, "display_name": cols.get("display_name"),
         "profile_url": cols.get("profile_url"), "bio": cols.get("bio"), "followers": cols.get("followers"),
         "role": cols.get("role"), "why": cols.get("selected_because"),
         "found": Jsonb(cols.get("found_via") or {})}).fetchone()
    if cols.get("followers") is not None:
        conn.execute("UPDATE reference_creators SET follower_history = follower_history || %s WHERE id = %s",
                     (Jsonb([{"at": datetime.now(timezone.utc).isoformat(), "followers": cols["followers"]}]),
                      row["id"]))
    return str(row["id"])


def content_hash(p: dict) -> str:
    """What a post shows that can change: its caption, with its media type. A published post's media cannot be
    replaced, and media URLs and carousel child lists differ by provider and signature, so they are left out.
    Annotations remember it, so unchanged content is never read again (manyworlds.research.monitor)."""
    text = f"{p.get('media_type') or ''}|{str(p.get('caption') or '').strip()}"
    return hashlib.sha256(text.encode()).hexdigest()[:20]


def upsert_post(conn, policy: Policy, creator_id: str, platform: str, p: dict) -> str:
    now = datetime.now(timezone.utc)
    row = conn.execute(
        """INSERT INTO reference_posts (creator_id, platform, external_id, url, media_type, posted_at, caption, metrics,
                                        snapshots, content_hash, caption_features)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (platform, external_id) DO UPDATE SET metrics = EXCLUDED.metrics,
               snapshots = reference_posts.snapshots || EXCLUDED.snapshots, fetched_at = now(),
               caption = coalesce(EXCLUDED.caption, reference_posts.caption),
               caption_features = CASE WHEN EXCLUDED.caption IS NULL THEN reference_posts.caption_features
                                       ELSE EXCLUDED.caption_features END,
               content_hash = CASE WHEN EXCLUDED.caption IS NULL THEN reference_posts.content_hash
                                   ELSE EXCLUDED.content_hash END
           RETURNING id""",
        (creator_id, platform, p["external_id"], p.get("url"), p.get("media_type"), p.get("posted_at"),
         p.get("caption") or None, Jsonb(p.get("metrics") or {}),
         Jsonb([{"at": now.isoformat(), **(p.get("metrics") or {})}]), content_hash(p),
         Jsonb(data_policy.caption_features(p.get("caption"))))
    ).fetchone()
    return str(row["id"])


def store_media(conn, policy: Policy, store, download, post_id: str, p: dict, workdir: Path) -> list[dict]:
    """Research copies of a post's images (up to images_per_post) and video in the research store, under the post's
    retention (media_purge_after). Each kind is copied once: a post that already has images (or its video) keeps
    them, so a video added later for a selected post (manyworlds.research.monitor) never copies the images again."""
    cfg = _cfg(policy)
    if store is None:
        return []
    have = conn.execute("SELECT media FROM reference_posts WHERE id = %s", (post_id,)).fetchone()
    media = list((have or {}).get("media") or [])
    kinds = {m.get("kind") for m in media}
    limit = int(cfg.get("max_media_bytes", 60_000_000))
    wanted = [] if "image" in kinds else [
        ("image", u, f"img{i}.jpg") for i, u in enumerate(p.get("images", [])[:int(cfg.get("images_per_post", 3))])]
    if p.get("video_url") and cfg.get("download_videos", True) and "video" not in kinds:
        wanted.append(("video", p["video_url"], "video.mp4"))
    if not wanted:
        return media
    before = len(media)
    for kind, url, name in wanted:
        try:
            local = download(url, workdir / post_id / name, limit)
        except Exception:  # noqa: BLE001 - a missing copy lowers coverage; the post's record stays
            local = None
        if local is None:
            continue
        o = store.put_file(Path(local), references.media_key(post_id, name))
        media.append({"uri": o.uri, "kind": kind, "bytes": o.bytes})
        Path(local).unlink(missing_ok=True)
    if len(media) > before:
        conn.execute("UPDATE reference_posts SET media = %s, media_purge_after = %s WHERE id = %s",
                     (Jsonb(media), references.media_purge_after(policy), post_id))
    return media


def fresh(conn, policy: Policy, platform: str, handle: str, *, days: float | None = None) -> dict | None:
    """A creator whose posts were pulled, or who was checked by the monitor, within repull_after_days and who has
    posts stored: {creator_id, handle, posts, usd: 0, reused: True}. A scout reuses them instead of paying for them
    again (2026-11-25: a pilot's retries re-pulled every creator). None: pull them."""
    days = float(days if days is not None else _cfg(policy).get("repull_after_days", 0))
    if days <= 0:
        return None
    row = conn.execute(
        """SELECT c.id, c.handle, count(p.id) AS n,
                  greatest(c.last_checked_at, max(p.fetched_at)) >= now() - make_interval(secs => %s) AS recent
           FROM reference_creators c LEFT JOIN reference_posts p ON p.creator_id = c.id
           WHERE c.platform = %s AND lower(c.handle) = lower(%s) GROUP BY c.id""",
        (days * 86400, platform, handle.lstrip("@"))).fetchone()
    if not row or not row["n"] or not row["recent"]:
        return None
    return {"creator_id": str(row["id"]), "handle": row["handle"], "posts": int(row["n"]), "usd": 0.0, "reused": True}


def pull_creator(conn, policy: Policy, fetch: Fetch, c: dict, *, store, download, workdir: Path, job_id=None,
                 posts: int | None = None, views: bool = False
                 ) -> dict:
    """One creator: their profile and up to posts_per_creator recent posts (posts_newer_than_days), stored, with
    research copies of their media. Returns {creator_id, posts, usd}; {skipped} for a forgotten account.

    Instagram posts come from intelligence.agent.instagram_posts_from: `profile` takes the latest posts the profile
    scraper embeds in its one item (research.providers.catalog.apify.profile_posts of them, no reel views), `posts`
    the post scraper (a billed item a post, with reel views). `views` (reel views needed) always uses the post
    scraper; so does a profile that was found but embeds no posts. A profile not found (an error item) pulls no more."""
    cfg = _cfg(policy)
    actors = cfg.get("actors") or {}
    platform, handle = c["platform"], str(c["handle"]).lstrip("@")
    data_policy.require(policy, "apify", "posts")
    if references.forgotten(conn, platform, handle):
        return {"skipped": "forgotten", "handle": handle}
    n = int(posts or cfg.get("posts_per_creator", 30))
    usd = 0.0
    profile: dict = {}
    post_actor = actors["instagram_posts" if platform == "instagram" else "tiktok_profile"]
    if platform == "instagram":
        prof = fetch(actors["instagram_profile"], {"usernames": [handle]})
        usd += len(prof) * _price(policy, "instagram_profile")
        charge(conn, policy, provider=f"apify:{actors['instagram_profile']}", items=len(prof),
               usd=len(prof) * _price(policy, "instagram_profile"), job_id=job_id)
        first = prof[0] if prof and isinstance(prof[0], dict) else {}
        profile = {"display_name": _first(first, "fullName"), "bio": _first(first, "biography"),
                   "followers": _first(first, "followersCount"),
                   "profile_url": _first(first, "url") or f"https://www.instagram.com/{handle}/"}
        found = bool(_first(first, "username")) and not first.get("error")
        latest = [{**x, "ownerUsername": x.get("ownerUsername") or _first(first, "username"),
                   "ownerFollowersCount": x.get("ownerFollowersCount") or profile["followers"]}
                  for x in (first.get("latestPosts") or []) if isinstance(x, dict)] if found else []
        if not found:
            raw, cost, post_actor = [], 0.0, None
        elif latest and not views and str(cfg.get("instagram_posts_from", "posts")) == "profile":
            raw, cost, post_actor = latest[:n], 0.0, None        # paid for with the profile item
        else:
            since = (datetime.now(timezone.utc) - timedelta(days=int(cfg.get("posts_newer_than_days", 120)))).date()
            raw = fetch(actors["instagram_posts"], {"username": [handle], "resultsLimit": n,
                                                    "onlyPostsNewerThan": since.isoformat(), "skipPinnedPosts": True})
            cost = len(raw) * _price(policy, "instagram_posts")
    else:
        # Apify's paid video download only when every video is kept; selected videos come from the post's own link
        download_videos = bool(cfg.get("download_videos", True)) and cfg.get("videos_for", "all") == "all"
        raw = fetch(actors["tiktok_profile"], {"profiles": [handle], "resultsPerPage": n,
                                               "shouldDownloadVideos": download_videos})
        cost = len(raw) * (_price(policy, "tiktok_profile")
                           + (_price(policy, "tiktok_video_download") if download_videos else 0))
        first = raw[0] if raw and isinstance(raw[0], dict) else {}
        profile = {"display_name": _first(first, "authorMeta.nickName"), "bio": _first(first, "authorMeta.signature"),
                   "followers": _first(first, "authorMeta.fans"), "profile_url": f"https://www.tiktok.com/@{handle}"}
    usd += cost
    if post_actor:
        charge(conn, policy, provider=f"apify:{post_actor}", items=len(raw), usd=cost, job_id=job_id)
    cid = upsert_creator(conn, platform, handle, role=c.get("role"), **profile,
                         selected_because=c.get("why") or ("named in the brief" if c.get("named") else c.get("role")),
                         found_via={"tags": c.get("tags") or [], "named": bool(c.get("named"))})
    stored, by_id = 0, {}
    every_video = cfg.get("videos_for", "all") == "all"
    for item in raw:
        p = NORMALIZE[platform](item) if isinstance(item, dict) else None
        if not p:
            continue
        pid = upsert_post(conn, policy, cid, platform, p)
        by_id[pid] = p
        store_media(conn, policy, store, download, pid, p if every_video else {**p, "video_url": None}, workdir)
        stored += 1
    baselines(conn, cid)
    if not every_video:      # videos only for the posts the deterministic filters pick (manyworlds.research.monitor)
        from . import monitor
        for x in monitor.select(conn, policy, cid, by_id, now=datetime.now(timezone.utc),
                                 window_days=float(cfg.get("posts_newer_than_days", 120))):
            if (by_id.get(x["id"]) or {}).get("video_url"):
                store_media(conn, policy, store, download, x["id"], by_id[x["id"]], workdir)
    span = conn.execute("SELECT extract(epoch FROM max(posted_at) - min(posted_at)) / 86400 AS d, count(*) n, "
                        "count(*) FILTER (WHERE metrics ? 'views') v FROM reference_posts WHERE creator_id = %s",
                        (cid,)).fetchone()
    conn.execute("UPDATE reference_creators SET coverage = %s WHERE id = %s",
                 (Jsonb({"posts": int(span["n"]), "with_views": int(span["v"]),
                         "span_days": round(float(span["d"] or 0), 1),
                         "enough": int(span["n"]) >= int(cfg.get("min_posts_per_creator", 6))}), cid))
    return {"creator_id": cid, "handle": handle, "posts": stored, "usd": round(usd, 4)}


def baselines(conn, creator_id) -> dict[str, float]:
    """Each post's engagement over its creator's median (metrics.over_own_median), the creator's median in scores."""
    rows = conn.execute("SELECT id, media_type, metrics FROM reference_posts WHERE creator_id = %s",
                        (creator_id,)).fetchall()
    eng = {str(r["id"]): engagement(dict(r)) for r in rows}
    vals = [v for v in eng.values() if v > 0]
    if not vals:
        return {}
    med = statistics.median(vals)
    out = {}
    for pid, v in eng.items():
        ratio = round(v / med, 3) if med else None
        out[pid] = ratio
        conn.execute("UPDATE reference_posts SET metrics = metrics || %s WHERE id = %s",
                     (Jsonb({"over_own_median": ratio}), pid))
    conn.execute("UPDATE reference_creators SET scores = scores || %s WHERE id = %s",
                 (Jsonb({"median_engagement": med}), creator_id))
    return out
