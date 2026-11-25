"""Cost-aware monitoring of the reference creator universe (policy `intelligence.monitor`; migration 045;
docs/research/2026-11-25-research-providers.md).

A large universe of public creators is watched at different frequencies. Each creator has a tier from four scores in
[0, 1] (relevance, growth, performance, intelligence value; weights and bands in policy) and is checked on its tier's
cadence. A check pulls cheap metadata only (one provider call through manyworlds.research.providers: ids, captions,
counts), dedupes posts by id, and picks the few posts worth a vision read with deterministic filters:

    outperformer   engagement at least `outperform` x the creator's own median (once the baseline has enough posts)
    new_format     a media type or product type the creator had not used in their stored posts
    new_tags       at least `new_tags_min` hashtags the creator never used before (a cheap aesthetic signal)
    commercial     a sponsorship marker in the caption (intelligence.commerce.sponsored_markers) or the platform's
                   paid-partnership flag

Only selected posts get research copies of their media (videos only for them, once: a post with media is left alone)
and a vision read later (the intelligence agent's annotate, `selected_only`). Post content hashes key annotations, so
unchanged content is never analysed again. Nothing here calls a model.

    enroll(conn, policy, creator_id)   give a creator a tier and a first check date (idempotent)
    tick(conn, policy, router, ...)    check the creators that are due, within the day's budget
    due_estimate(conn, policy)         what the due checks would cost (the dry run)
    monthly_estimate(policy, n)        $/month for a universe of n creators at the policy's tier mix
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from psycopg.types.json import Jsonb

from ..policy import Policy
from . import pulls, references

TIERS = ("hot", "warm", "cold", "dormant")


def cfg(policy: Policy) -> dict:
    return policy.get("intelligence", "monitor", default={}) or {}


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("|".join(sorted(map(str, ids))).encode()).hexdigest()[:20]


def spent_today(conn, now: datetime) -> float:
    day = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return float(conn.execute("SELECT coalesce(sum(usd), 0) u FROM cost_entries WHERE agent = 'intelligence:monitor' "
                              "AND created_at >= %s", (day,)).fetchone()["u"])


# --- tiers --------------------------------------------------------------------------------------------------------
def scores(conn, policy: Policy, creator: dict, now: datetime) -> dict:
    """relevance, growth, performance, intelligence value, each in [0, 1]."""
    c = cfg(policy)
    s = creator.get("scores") or {}
    roles = c.get("role_relevance") or {}
    relevance = float(s["relevance"]) if s.get("relevance") is not None else float(
        roles.get(creator.get("role") or "other", roles.get("other", 0.5)))
    hist = [h for h in (creator.get("follower_history") or []) if h.get("followers")]
    growth = 0.0
    if len(hist) >= 2:
        a, b = hist[0], hist[-1]
        days = max((datetime.fromisoformat(b["at"]) - datetime.fromisoformat(a["at"])).total_seconds() / 86400, 1.0)
        rate = (float(b["followers"]) / max(float(a["followers"]), 1.0) - 1) * 30 / days
        growth = max(0.0, min(1.0, rate / float(c.get("growth_full", 0.1))))
    perf = 0.0
    med = (s.get("median_engagement") or 0)
    if med and creator.get("followers"):
        perf = max(0.0, min(1.0, float(med) / float(creator["followers"]) / float(c.get("engagement_full", 0.05))))
    n_sel = conn.execute("SELECT count(*) n FROM reference_posts WHERE creator_id = %s AND selected IS NOT NULL "
                         "AND (selected->>'at')::timestamptz > %s", (creator["id"], now - timedelta(days=90))
                         ).fetchone()["n"]
    cited = conn.execute("SELECT count(*) n FROM pattern_cards WHERE sources @> %s",
                         (Jsonb([{"creator_id": str(creator["id"])}]),)).fetchone()["n"]
    intel = max(0.0, min(1.0, (int(n_sel) + 2 * int(cited)) / float(c.get("intelligence_full", 6))))
    return {"relevance": round(relevance, 3), "growth": round(growth, 3), "performance": round(perf, 3),
            "intelligence": round(intel, 3)}


def tier_for(policy: Policy, sc: dict, last_post_at: datetime | None, now: datetime) -> str:
    c = cfg(policy)
    w = c.get("weights") or {}
    total = sum(float(w.get(k, 0)) * float(sc.get(k, 0)) for k in ("relevance", "growth", "performance",
                                                                     "intelligence"))
    bands = c.get("bands") or {}
    if last_post_at and now - last_post_at > timedelta(days=float(c.get("dormant_after_days", 120))):
        return "dormant"
    if total >= float(bands.get("hot", 0.7)):
        return "hot"
    if total >= float(bands.get("warm", 0.4)):
        return "warm"
    return "cold"


def _next(policy: Policy, tier: str, creator_id, now: datetime) -> datetime:
    """The tier's cadence, spread by a fixed fraction per creator so a large universe does not fall due at once."""
    t = (cfg(policy).get("tiers") or {}).get(tier) or {}
    days = float(t.get("every_days", 30))
    jitter = float(cfg(policy).get("jitter", 0.1))
    frac = int(hashlib.sha256(str(creator_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return now + timedelta(days=days * (1 - jitter / 2 + jitter * frac))


def _row(conn, creator_id) -> dict:
    return dict(conn.execute("SELECT * FROM reference_creators WHERE id = %s", (creator_id,)).fetchone())


def _last_post_at(conn, creator_id):
    return conn.execute("SELECT max(posted_at) t FROM reference_posts WHERE creator_id = %s",
                        (creator_id,)).fetchone()["t"]


def enroll(conn, policy: Policy, creator_id, *, now: datetime | None = None, due_now: bool = False) -> str:
    """A tier and a first check date (its tier's cadence, or now) for a creator not yet monitored; a creator already
    monitored keeps its own."""
    now = now or datetime.now(timezone.utc)
    c = _row(conn, creator_id)
    if c.get("next_check_at"):
        return c["tier"]
    sc = scores(conn, policy, c, now)
    tier = tier_for(policy, sc, _last_post_at(conn, creator_id), now)
    conn.execute("UPDATE reference_creators SET tier = %s, tier_scores = %s, next_check_at = %s WHERE id = %s",
                 (tier, Jsonb(sc), now if due_now else _next(policy, tier, creator_id, now), creator_id))
    return tier


# --- selection ----------------------------------------------------------------------------------------------------
def _markers(policy: Policy) -> list[str]:
    return [str(m).lower() for m in (policy.get("intelligence", "commerce", "sponsored_markers", default=[]) or [])]


def select(conn, policy: Policy, creator_id, fresh: dict[str, dict], *, now: datetime,
           window_days: float | None = None) -> list[dict]:
    """Deterministic filters over the creator's recent unselected posts; marks the chosen (reference_posts.selected)
    and returns [{id, reasons}]. `fresh` maps post ids to this check's normalised posts (hashtags, flags)."""
    c = cfg(policy)
    s = c.get("select") or {}
    rows = [dict(r) for r in conn.execute(
        "SELECT id, external_id, media_type, caption, metrics, selected, posted_at FROM reference_posts "
        "WHERE creator_id = %s ORDER BY posted_at DESC NULLS LAST", (creator_id,)).fetchall()]
    window = now - timedelta(days=float(window_days or s.get("window_days", 45)))
    enough = len(rows) >= int(s.get("min_posts_for_baseline", 6))
    markers = _markers(policy)
    chosen = []
    for i, r in enumerate(rows):
        if r["selected"] is not None or (r["posted_at"] and r["posted_at"] < window):
            continue
        p = fresh.get(str(r["id"])) or {}
        older = rows[i + 1:]
        if not older:                         # nothing to compare the creator's first post with
            continue
        reasons = []
        ratio = (r["metrics"] or {}).get("over_own_median")
        if enough and ratio is not None and float(ratio) >= float(s.get("outperform", 2.0)):
            reasons.append(f"outperformer:{ratio}")
        seen_types = {o["media_type"] for o in older}
        if s.get("new_format", True) and len(older) >= int(s.get("min_posts_for_baseline", 6)) \
                and r["media_type"] not in seen_types:
            reasons.append(f"new_format:{r['media_type']}")
        tags = set(p.get("hashtags") or [])
        if tags and len(older) >= int(s.get("min_posts_for_baseline", 6)):
            old_tags = set()
            for o in older:
                old_tags |= {t.lower() for t in references_tags(o.get("caption"))}
            new = tags - old_tags
            if len(new) >= int(s.get("new_tags_min", 2)):
                reasons.append("new_tags:" + ",".join(sorted(new)[:5]))
        cap = str(r.get("caption") or "").lower()
        if p.get("paid_partnership") or any(m in cap for m in markers):
            reasons.append("commercial")
        if reasons:
            chosen.append({"id": str(r["id"]), "reasons": reasons, "ratio": float(ratio or 0)})
    chosen.sort(key=lambda x: (-len(x["reasons"]), -x["ratio"]))
    chosen = chosen[:int(s.get("max_per_check", 3))]
    for x in chosen:
        conn.execute("UPDATE reference_posts SET selected = %s WHERE id = %s",
                     (Jsonb({"at": now.isoformat(), "reasons": x["reasons"]}), x["id"]))
    return chosen


def references_tags(caption: str | None) -> list[str]:
    import re
    return re.findall(r"#(\w+)", caption or "")


# --- the check ----------------------------------------------------------------------------------------------------
def store_selected_media(conn, policy: Policy, store, download, post_ids: list[str], by_id: dict[str, dict],
                         workdir: Path) -> int:
    """Research copies (images and video) for selected posts only, once each."""
    n = 0
    for pid in post_ids:
        p = by_id.get(pid)
        if p and (p.get("images") or p.get("video_url")):
            if pulls.store_media(conn, policy, store, download, pid, p, workdir):
                n += 1
    return n


def check(conn, policy: Policy, router, creator: dict, *, now: datetime, store=None, download=None,
          workdir: Path | None = None, prefetched: dict | None = None) -> dict:
    """One creator's check: cheap metadata, dedupe, selection, media for the selected, a new tier and date.
    `prefetched`: this creator's profile from a batched profiles call ({provider, profile}; profile None: not found)."""
    c = cfg(policy)
    cid, platform, handle = creator["id"], creator["platform"], creator["handle"]
    if references.forgotten(conn, platform, handle):
        conn.execute("UPDATE reference_creators SET next_check_at = NULL WHERE id = %s", (cid,))
        return {"handle": handle, "skipped": "forgotten"}
    n = int(((c.get("tiers") or {}).get(creator.get("tier") or "cold") or {}).get("posts", 12))
    if prefetched is not None:
        prof = prefetched.get("profile")
        posts = [{**p, "followers": p.get("followers") or prof.get("followers")}
                 for p in (prof or {}).get("latest_posts") or []][:n]
        provider, errors = (prefetched["provider"] if prof else None), [{"provider": prefetched["provider"],
                                                                         "error": "profile not found"}]
    else:
        posts, provider, errors = router.fetch_posts(platform, handle, n)
    if provider is None:
        fails = int(creator.get("check_failures") or 0) + 1
        retry = now + timedelta(days=min(float(c.get("retry_days", 1)) * 2 ** (fails - 1), 30))
        conn.execute("UPDATE reference_creators SET check_failures = %s, next_check_at = %s, last_checked_at = %s "
                     "WHERE id = %s", (fails, retry, now, cid))
        return {"handle": handle, "error": errors[-1] if errors else "no provider"}
    known = {r["external_id"] for r in conn.execute(
        "SELECT external_id FROM reference_posts WHERE platform = %s AND external_id = ANY(%s)",
        (platform, [p["external_id"] for p in posts])).fetchall()}
    by_id: dict[str, dict] = {}
    for p in posts:
        pid = pulls.upsert_post(conn, policy, str(cid), platform, p)
        by_id[pid] = p
    new = [p["external_id"] for p in posts if p["external_id"] not in known]
    followers = next((p.get("followers") for p in posts if p.get("followers")), None)
    if followers and followers != creator.get("followers"):
        conn.execute("UPDATE reference_creators SET followers = %s, follower_history = follower_history || %s "
                     "WHERE id = %s", (followers, Jsonb([{"at": now.isoformat(), "followers": followers}]), cid))
    h = ids_hash([p["external_id"] for p in posts])
    unchanged = h == creator.get("content_hash")
    pulls.baselines(conn, cid)
    chosen = select(conn, policy, cid, by_id, now=now)
    media = 0
    if chosen and store is not None and download is not None:
        missing = [x["id"] for x in chosen if not (by_id.get(x["id"]) or {}).get("images")]
        if missing:            # a cheap answer without media URLs (Business Discovery on licensed audio): one more call
            again, _, _ = router.fetch_posts(platform, handle, n, media=True)
            ext = {p["external_id"]: p for p in again}
            for pid in missing:
                old = by_id.get(pid) or {}
                if old.get("external_id") in ext:
                    by_id[pid] = ext[old["external_id"]]
        media = store_selected_media(conn, policy, store, download, [x["id"] for x in chosen], by_id,
                                     workdir or Path("/tmp"))
    fresh = _row(conn, cid)
    sc = scores(conn, policy, fresh, now)
    last_at = _last_post_at(conn, cid)
    tier = tier_for(policy, sc, last_at, now)
    newest = max(posts, key=lambda p: p.get("posted_at") or datetime.min.replace(tzinfo=timezone.utc), default=None)
    conn.execute("""UPDATE reference_creators SET tier = %s, tier_scores = %s, next_check_at = %s, last_checked_at = %s,
                        last_post_id = %s, content_hash = %s, check_failures = 0, last_seen = now() WHERE id = %s""",
                 (tier, Jsonb(sc), _next(policy, tier, cid, now), now,
                  newest["external_id"] if newest else creator.get("last_post_id"), h, cid))
    return {"handle": handle, "provider": provider, "posts": len(posts), "new": len(new), "unchanged": unchanged,
            "selected": [x["id"] for x in chosen], "media": media, "tier": tier}


def per_check_usd(policy: Policy, platform: str) -> float:
    est = cfg(policy).get("usd_per_check") or {}
    return float(est.get(platform, est.get("default", 0.003)))


def due(conn, policy: Policy, now: datetime, limit: int) -> list[dict]:
    order = "CASE tier WHEN 'hot' THEN 0 WHEN 'warm' THEN 1 WHEN 'cold' THEN 2 ELSE 3 END"
    return [dict(r) for r in conn.execute(
        f"SELECT * FROM reference_creators WHERE next_check_at IS NOT NULL AND next_check_at <= %s "
        f"ORDER BY {order}, next_check_at LIMIT %s", (now, limit)).fetchall()]


def prefetch(conn, policy: Policy, router, todo: list[dict], left: float) -> tuple[dict, float]:
    """Batched profile calls for the platforms in `prefetch.platforms` (Apify's profile scraper takes many handles in
    one run and embeds each one's latest posts: one item a creator), `prefetch.batch` handles a call, within the
    budget. Returns ({creator id: {provider, profile}}, budget left)."""
    pc = cfg(policy).get("prefetch") or {}
    out: dict = {}
    for platform in pc.get("platforms") or []:
        rows = [r for r in todo if r["platform"] == platform and not references.forgotten(conn, platform, r["handle"])]
        size = max(1, int(pc.get("batch", 25)))
        for i in range(0, len(rows), size):
            chunk = rows[i:i + size]
            if left < per_check_usd(policy, platform) * len(chunk):
                break
            before = sum(float(x.get("usd") or 0) for x in router.log)
            profs, provider, _ = router.fetch_profiles(platform, [r["handle"] for r in chunk])
            left -= sum(float(x.get("usd") or 0) for x in router.log) - before
            if provider is None:
                continue                     # each creator falls back to its own posts call
            got = {str(p["handle"]).lower(): p for p in profs}
            for r in chunk:
                out[str(r["id"])] = {"provider": provider, "profile": got.get(str(r["handle"]).lower())}
    return out, left


def tick(conn, policy: Policy, router, *, now: datetime | None = None, budget_usd: float | None = None, store=None,
         download: Callable | None = None, workdir: Path | None = None) -> dict:
    """Check the creators that are due (hot first), within the day's budget (intelligence.monitor.daily_usd, or a
    smaller `budget_usd`). Creators never enrolled are enrolled first (up to max_enroll_per_tick)."""
    now = now or datetime.now(timezone.utc)
    c = cfg(policy)
    for r in conn.execute("SELECT id FROM reference_creators WHERE next_check_at IS NULL AND tier <> 'dormant' "
                          "AND last_checked_at IS NULL LIMIT %s", (int(c.get("max_enroll_per_tick", 500)),)).fetchall():
        enroll(conn, policy, r["id"], now=now, due_now=True)          # their first check: now
    left = float(c.get("daily_usd", 0)) - spent_today(conn, now)
    if budget_usd is not None:
        left = min(left, float(budget_usd))
    out = {"checked": [], "errors": [], "skipped_budget": 0, "budget_left_usd": round(left, 4)}
    todo = due(conn, policy, now, int(c.get("max_checks_per_tick", 200)))
    pre, left = prefetch(conn, policy, router, todo, left)
    for cr in todo:
        est = 0.0 if str(cr["id"]) in pre else per_check_usd(policy, cr["platform"])
        if left < est:
            out["skipped_budget"] += 1
            continue
        before = sum(float(x.get("usd") or 0) for x in router.log)
        try:
            r = check(conn, policy, router, cr, now=now, store=store, download=download, workdir=workdir,
                      prefetched=pre.get(str(cr["id"])))
        except Exception as e:  # noqa: BLE001 - one creator's failed check must not stop the rest
            out["errors"].append({"handle": cr["handle"], "error": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        left -= sum(float(x.get("usd") or 0) for x in router.log) - before
        (out["errors"] if r.get("error") else out["checked"]).append(r)
    out["selected"] = sum(len(r.get("selected") or []) for r in out["checked"])
    out["budget_left_usd"] = round(left, 4)
    return out


def due_estimate(conn, policy: Policy, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    rows = conn.execute("SELECT platform, tier, count(*) n FROM reference_creators WHERE next_check_at <= %s "
                        "OR (next_check_at IS NULL AND last_checked_at IS NULL) GROUP BY 1, 2", (now,)).fetchall()
    by = [{"platform": r["platform"], "tier": r["tier"], "due": int(r["n"]),
           "usd": round(int(r["n"]) * per_check_usd(policy, r["platform"]), 4)} for r in rows]
    return {"due": by, "usd": round(sum(x["usd"] for x in by), 4), "daily_usd": float(cfg(policy).get("daily_usd", 0)),
            "spent_today_usd": round(spent_today(conn, now), 4)}


def monthly_estimate(policy: Policy, creators: int, *, usd_per_check: float | None = None,
                     vision_usd_per_post: float | None = None) -> dict:
    """$/month for a universe of `creators` at the policy's tier mix (intelligence.monitor.mix), checks plus the
    vision reads of the posts the filters are expected to select (select_rate per check)."""
    c = cfg(policy)
    mix = c.get("mix") or {}
    per = float(usd_per_check if usd_per_check is not None else per_check_usd(policy, "instagram"))
    checks = {t: creators * float(mix.get(t, 0)) * 30 / float(((c.get("tiers") or {}).get(t) or {}).get("every_days", 30))
              for t in TIERS}
    n_checks = sum(checks.values())
    vision = float(vision_usd_per_post if vision_usd_per_post is not None else c.get("usd_per_vision_post", 0.004))
    selected = n_checks * float(c.get("select_rate", 0.1))
    return {"creators": creators, "checks_per_month": math.ceil(n_checks),
            "checks_by_tier": {k: math.ceil(v) for k, v in checks.items()}, "check_usd": round(n_checks * per, 2),
            "selected_posts": math.ceil(selected), "vision_usd": round(selected * vision, 2),
            "total_usd": round(n_checks * per + selected * vision, 2)}


def ensure_due(conn, policy: Policy, *, now: datetime | None = None):
    """One monitor run a day (a brain.intelligence job, mode monitor), from the Brain's cycle (cli.brain_steps), while
    intelligence.monitor.enabled and operations are live. Returns the job id, or None."""
    from .. import operations, queue
    if not cfg(policy).get("enabled") or not operations.is_live(policy):
        return None
    now = now or datetime.now(timezone.utc)
    day = now.date().isoformat()
    return queue.enqueue(conn, "brain.intelligence", {"mode": "monitor", "day": day},
                         idempotency_key=f"intelligence-monitor:{day}")
