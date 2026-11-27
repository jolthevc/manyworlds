"""The trend watcher: what is working, rising and cooling on TikTok and Instagram, per niche, all day, within a
hard daily budget (policy `trends`, `daily_usd_max`; every cent goes to the ledger under agent `trends`).

Each tick does what the budget still allows, in this order:

1. collect: for each niche (policy `trends.markets`), public posts from keyword searches and the trending feed
   (ScrapeCreators: one credit returns a page of posts). A post seen again gets a snapshot, so we know how fast it
   grows (velocity), not just how big it is.
2. analyze: the posts that beat their niche's median views by the widest margin (outliers), plus a random control
   sample of ordinary posts, are described by a cheap vision model from the cover frame, the caption's opening
   and the first seconds of speech: format, hook type, how it opens, pacing, tone, topic. Outliers alone would
   only show what is common; the control sample shows what is over-represented among winners.
3. synthesize (once a day per niche): numbers first (which features are over-represented among outliers, which
   are gaining or losing share week on week, which sounds are rising, the best hooks) then one model call
   writes the brief. Patterns become findings (evidence `research`, never `proven`: other people's posts show
   correlation; our own experiments prove things), read by the showrunner through brain/evidence.py.

Privacy: logged-out public data only; handles only as salted hashes; raw items purged after 30 days; features
and briefs hold no handles or captions. Nothing here posts, follows, likes or logs in anywhere.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Literal

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from .. import ledger
from ..policy import Policy

AGENT = "trends"


# --- budget ------------------------------------------------------------------------------------------------
def spent_today(conn, now: datetime) -> float:
    day = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return float(conn.execute("SELECT coalesce(sum(usd), 0) AS u FROM cost_entries WHERE agent LIKE %s AND created_at >= %s",
                              (AGENT + "%", day)).fetchone()["u"])


def left(conn, policy: Policy, now: datetime, stage: str | None = None) -> float:
    """Dollars the watcher may still spend today (overall, and within a stage's own share if given)."""
    cfg = _cfg(policy)
    from . import budget
    total = min(float(cfg.get("daily_usd_max", 3.0)) - spent_today(conn, now), budget.left(conn, policy, now))
    if stage is None:
        return total
    share = float((cfg.get("stage_share") or {}).get(stage, 1.0)) * float(cfg.get("daily_usd_max", 3.0))
    day = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    used = float(conn.execute("SELECT coalesce(sum(usd), 0) AS u FROM cost_entries WHERE agent = %s AND created_at >= %s",
                              (f"{AGENT}:{stage}", day)).fetchone()["u"])
    return min(total, share - used)


def _cfg(policy: Policy) -> dict:
    return policy.get("trends", default={}) or {}


def _charge_credits(conn, policy: Policy, stage: str, credits: int) -> None:
    usd = credits * float(_cfg(policy).get("usd_per_credit", 0.002))
    ledger.record(conn, category="data", provider=f"scrapecreators:{stage}", quantity=credits, unit="credit", usd=usd,
                  policy_version=policy.version, agent=f"{AGENT}:{stage}")


# --- 1. collect ----------------------------------------------------------------------------------------------
COLS = ("platform", "external_id", "query_tag", "market", "source", "author_hash", "author_followers", "media_type",
        "caption", "hashtags", "likes", "comments", "views", "shares", "saves", "duration_s", "posted_at", "sound_id",
        "sound_title", "cover_url", "url")


def store(conn, run_id, items: list[dict]) -> int:
    n = 0
    for it in items:
        row = {c: it.get(c) for c in COLS}
        row["hashtags"] = row["hashtags"] or []
        item = conn.execute(
            f"""INSERT INTO research_items (run_id, {', '.join(COLS)}) VALUES (%(run)s, {', '.join(f'%({c})s' for c in COLS)})
                ON CONFLICT (platform, external_id) DO UPDATE SET likes = EXCLUDED.likes, comments = EXCLUDED.comments,
                  views = EXCLUDED.views, shares = EXCLUDED.shares, saves = EXCLUDED.saves, fetched_at = now(),
                  cover_url = coalesce(EXCLUDED.cover_url, research_items.cover_url),
                  market = coalesce(research_items.market, EXCLUDED.market)
                RETURNING id""", {**row, "run": run_id}).fetchone()
        conn.execute("INSERT INTO research_snapshots (item_id, captured_at, views, likes, comments, shares, saves) "
                     "VALUES (%s, clock_timestamp(), %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                     (item["id"], row["views"], row["likes"], row["comments"], row["shares"], row["saves"]))
        n += 1
    return n


def _due(conn, market: str, now: datetime, hours: float) -> bool:
    last = conn.execute("SELECT max(created_at) AS t FROM research_runs WHERE source = 'scrapecreators' "
                        "AND query->>'market' = %s AND error IS NULL", (market,)).fetchone()["t"]
    return last is None or now - last >= timedelta(hours=hours)


def collect(conn, policy: Policy, src, *, now: datetime) -> dict:
    from .data_policy import refused
    why = refused(policy, "scrapecreators", "search")   # the data policy first (manyworlds.research.data_policy)
    if why:
        return {"requests": 0, "items": 0, "errors": 0, "refused": why}
    cfg = _cfg(policy)
    out = {"requests": 0, "items": 0, "errors": 0}
    calls = []
    for m in cfg.get("markets") or []:
        if not _due(conn, m["name"], now, float(cfg.get("refresh_hours", 6))):
            continue
        for q in m.get("queries") or [m["name"]]:
            if "tiktok" in m.get("platforms", ["tiktok", "instagram"]):
                calls.append((m["name"], "tiktok", q))
            if "instagram" in m.get("platforms", ["tiktok", "instagram"]):
                calls.append((m["name"], "instagram", q))
    if cfg.get("trending_feed", True) and _due(conn, "_trending", now, float(cfg.get("refresh_hours", 6))):
        calls.append(("_trending", "tiktok", None))
    per_request = float(cfg.get("usd_per_credit", 0.002))
    for market, platform, q in calls:
        if left(conn, policy, now, "collect") < per_request:
            out["budget"] = "collect share used"
            break
        run_id = conn.execute("INSERT INTO research_runs (source, query) VALUES ('scrapecreators', %s) RETURNING id",
                              (Jsonb({"market": market, "platform": platform, "q": q}),)).fetchone()["id"]
        try:
            if q is None:
                items = src.tiktok_trending(market=market, region=cfg.get("region", "US"))
            elif platform == "tiktok":
                items, _ = src.tiktok_search(q, market=market, region=cfg.get("region", "US"),
                                             date_posted=cfg.get("tiktok_date_posted", "this-week"),
                                             sort_by=cfg.get("tiktok_sort_by", "relevance"))
            else:
                items = src.instagram_reels(q, market=market, date_posted=cfg.get("instagram_date_posted", "last-week"))
        except Exception as e:  # noqa: BLE001 - one failed call is recorded; the rest go on
            conn.execute("UPDATE research_runs SET error = %s WHERE id = %s", (str(e)[:500], run_id))
            out["errors"] += 1
            if type(e).__name__ == "OutOfCredits":
                break
            continue
        _charge_credits(conn, policy, "collect", 1)
        n = store(conn, run_id, items)
        conn.execute("UPDATE research_runs SET items = %s, usd = %s WHERE id = %s", (n, per_request, run_id))
        out["requests"] += 1
        out["items"] += n
    return out


# --- 2. analyze ----------------------------------------------------------------------------------------------
Format = Literal["talking_head", "pov_skit", "voiceover_broll", "tutorial_howto", "day_in_life_vlog", "reaction",
                 "slideshow_text", "aesthetic_montage", "interview", "before_after", "other"]
Hook = Literal["question", "bold_claim", "relatable_pain", "curiosity_gap", "visual_surprise", "direct_address",
               "storytime", "list_promise", "contrarian", "pov_setup", "none"]
Tone = Literal["funny", "deadpan", "wholesome", "dramatic", "calm_aesthetic", "informative", "chaotic", "sincere",
               "edgy", "romantic"]


class PostFeatures(BaseModel):
    key: str = Field(description="the post's key, e.g. p3")
    format: Format
    hook_type: Hook
    hook_summary: str = Field(description="the opening move in your own words, at most 12 words; never quote it")
    opens_on_face: bool
    on_screen_text: bool
    speaks_to_camera: bool
    people: int = Field(description="people visible: 0, 1 or 2 for two or more")
    setting: str = Field(description="a few words, e.g. 'small kitchen at night'")
    tone: Tone
    topic: str = Field(description="a few words")
    pacing: Literal["fast_cuts", "steady", "single_take", "unknown"]
    call_to_action: bool
    why_it_works: str = Field(description="for a strong post: the likely reason, at most 20 words; else empty")


class FeatureBatch(BaseModel):
    posts: list[PostFeatures]


FEATURE_SYSTEM = (
    "You study short-form video for a creator studio. For each post you get its cover frame, the opening of its "
    "caption, the first seconds of speech when there is any, its length and its sound. Describe how it is made, "
    "using the given categories exactly. Describe; don't judge taste. Never copy text from a post: summarise in "
    "your own words. Use only what you are given; say 'unknown' or pick 'other' rather than guess.")


def _medians(conn, now: datetime, days: int) -> dict[tuple, float]:
    rows = conn.execute("""SELECT market, platform, percentile_cont(0.5) WITHIN GROUP (ORDER BY views) AS m, count(*) n
                           FROM research_items WHERE views IS NOT NULL AND market IS NOT NULL AND posted_at > %s
                           GROUP BY market, platform""", (now - timedelta(days=days),)).fetchall()
    return {(r["market"], r["platform"]): float(r["m"]) for r in rows if r["n"] >= 5}


def _velocity(conn, item_id) -> float | None:
    s = conn.execute("SELECT captured_at, views FROM research_snapshots WHERE item_id = %s AND views IS NOT NULL "
                     "ORDER BY captured_at LIMIT 2", (item_id,)).fetchall()
    if len(s) < 2:
        return None
    hours = (s[1]["captured_at"] - s[0]["captured_at"]).total_seconds() / 3600
    return round((s[1]["views"] - s[0]["views"]) / hours, 1) if hours > 0 else None


def _cover(url: str | None, max_bytes: int) -> bytes | None:
    if not url:
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read(max_bytes + 1)
        return data if 0 < len(data) <= max_bytes else None
    except Exception:  # noqa: BLE001 - signed CDN links expire; the post is described without its frame
        return None


def pick(conn, policy: Policy, now: datetime) -> list[dict]:
    """Posts to describe: the biggest outliers per niche and platform, and a random control sample."""
    cfg = _cfg(policy)
    days = int(cfg.get("window_days", 7))
    med = _medians(conn, now, days)
    rows = conn.execute(
        """SELECT i.* FROM research_items i LEFT JOIN research_features f ON f.item_id = i.id
           WHERE f.id IS NULL AND i.media_type = 'video' AND i.views IS NOT NULL AND i.market IS NOT NULL
             AND i.posted_at > %s""", (now - timedelta(days=days),)).fetchall()
    groups = defaultdict(list)
    for r in rows:
        m = med.get((r["market"], r["platform"]))
        if m is not None:
            r = dict(r)
            r["outlier"] = round(math.log10(r["views"] + 1) - math.log10(m + 1), 3)
            groups[(r["market"], r["platform"])].append(r)
    picked, rng = [], random.Random(now.strftime("%Y%m%d%H"))
    k_out, k_ctl = int(cfg.get("outliers_per_group", 8)), int(cfg.get("controls_per_group", 4))
    for g in groups.values():
        g.sort(key=lambda r: r["outlier"], reverse=True)
        top = [r for r in g if r["outlier"] >= float(cfg.get("min_outlier", 0.5))][:k_out]
        rest = [r for r in g if r not in top]
        ctl = rng.sample(rest, min(k_ctl, len(rest)))
        picked += [{**r, "sample": "outlier"} for r in top] + [{**r, "sample": "control"} for r in ctl]
    return picked


def analyze(conn, policy: Policy, llm, src, *, now: datetime) -> dict:
    from ..llm import image_block
    cfg = _cfg(policy)
    out = {"described": 0, "transcripts": 0}
    todo = pick(conn, policy, now)
    size = int(cfg.get("feature_batch", 8))
    est = float(cfg.get("usd_per_feature_batch_estimate", 0.01))
    for i in range(0, len(todo), size):
        if left(conn, policy, now, "analyze") < est + size * float(cfg.get("usd_per_credit", 0.002)):
            out["budget"] = "analyze share used"
            break
        batch = todo[i:i + size]
        content, keys = [], {}
        for j, r in enumerate(batch):
            key = f"p{j + 1}"
            keys[key] = r
            speech = None
            if r["platform"] == "tiktok" and r.get("url"):
                try:
                    speech = src.transcript("tiktok", r["url"])
                    _charge_credits(conn, policy, "analyze", 1)
                    out["transcripts"] += 1
                except Exception:  # noqa: BLE001 - described without speech
                    speech = None
            img = _cover(r.get("cover_url"), int(cfg.get("cover_max_bytes", 400_000)))
            if img:
                content.append(image_block(img, "image/jpeg"))
            content.append({"type": "text", "text": json.dumps({
                "key": key, "cover_frame": "the image just above" if img else "not available",
                "platform": r["platform"], "seconds": float(r["duration_s"]) if r.get("duration_s") is not None else None,
                "caption_opening": (r.get("caption") or "")[:160], "first_seconds_of_speech": speech,
                "sound": r.get("sound_title")}, ensure_ascii=False)})
        try:
            with llm.tagged(f"{AGENT}:analyze", None):
                got = llm.structured("trend_features", system=FEATURE_SYSTEM, content=content, schema=FeatureBatch,
                                     effort="low")
        except Exception as e:  # noqa: BLE001 - try again next tick
            out["errors"] = out.get("errors", []) + [f"{type(e).__name__}: {str(e)[:200]}"]
            continue
        for f in got.posts:
            r = keys.get(f.key)
            if r is None:
                continue
            conn.execute(
                """INSERT INTO research_features (item_id, platform, market, sample, posted_at, duration_s, views, outlier,
                       velocity, sound_id, features, model)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (item_id) DO NOTHING""",
                (r["id"], r["platform"], r["market"], r["sample"], r["posted_at"], r.get("duration_s"), r["views"],
                 r["outlier"], _velocity(conn, r["id"]), r.get("sound_id"),
                 Jsonb(f.model_dump(exclude={"key"})), llm.model_for("trend_features")))
            out["described"] += 1
    return out


# --- 3. synthesize -------------------------------------------------------------------------------------------
CATEGORIES = ("format", "hook_type", "tone", "pacing", "opens_on_face", "on_screen_text", "speaks_to_camera",
              "call_to_action", "people", "length")


def _length(s) -> str:
    s = float(s or 0)
    return "unknown" if s <= 0 else "under_10s" if s < 10 else "10_20s" if s < 20 else "20_40s" if s < 40 else \
        "40_90s" if s < 90 else "over_90s"


def stats(conn, policy: Policy, market: str, now: datetime) -> dict:
    """Numbers first: over-representation among outliers vs controls, week-on-week share change, rising sounds."""
    cfg = _cfg(policy)
    days = int(cfg.get("window_days", 7))
    rows = conn.execute("SELECT sample, created_at, posted_at, duration_s, outlier, velocity, sound_id, features "
                        "FROM research_features WHERE market = %s AND created_at > %s",
                        (market, now - timedelta(days=2 * days))).fetchall()
    cur = [r for r in rows if r["created_at"] > now - timedelta(days=days)]
    prev = [r for r in rows if r["created_at"] <= now - timedelta(days=days)]

    def value(r, c):
        return _length(r["duration_s"]) if c == "length" else str(r["features"].get(c))

    def shares(rs, c):
        n = len(rs) or 1
        return {k: v / n for k, v in Counter(value(r, c) for r in rs).items()}

    out_cur = [r for r in cur if r["sample"] == "outlier"]
    ctl_cur = [r for r in cur if r["sample"] == "control"]
    out_prev = [r for r in prev if r["sample"] == "outlier"]
    min_n = int(cfg.get("min_n", 5))
    patterns = []
    for c in CATEGORIES:
        so, sc, sp = shares(out_cur, c), shares(ctl_cur, c), shares(out_prev, c)
        counts = Counter(value(r, c) for r in out_cur)
        for v, s in so.items():
            if counts[v] < min_n or v in ("unknown", "other", "None"):
                continue
            ratio = s / max(sc.get(v, 0.0), 1.0 / max(len(ctl_cur), 1) / 2)
            patterns.append({"feature": c, "value": v, "n": counts[v], "share_of_outliers": round(s, 2),
                             "share_of_controls": round(sc.get(v, 0.0), 2), "lift": round(ratio, 2),
                             "share_change_vs_last_week": round(s - sp[v], 2) if out_prev and v in sp else None})
    patterns.sort(key=lambda p: p["lift"], reverse=True)
    sounds = Counter(r["sound_id"] for r in cur if r["sound_id"])
    before = Counter(r["sound_id"] for r in prev if r["sound_id"])
    rising = [{"sound_id": s, "posts": n, "posts_week_before": before.get(s, 0)}
              for s, n in sounds.most_common(10) if n >= 3 and n > 2 * before.get(s, 0)]
    best = sorted(out_cur, key=lambda r: r["outlier"] or 0, reverse=True)[:int(cfg.get("hook_examples", 12))]
    return {"market": market, "posts_described": len(cur), "outliers": len(out_cur), "controls": len(ctl_cur),
            "median_outlier": round(statistics.median([r["outlier"] for r in out_cur]), 2) if out_cur else None,
            "patterns": patterns[:int(cfg.get("patterns_max", 25))], "rising_sounds": rising,
            "top_hooks": [{"hook": r["features"].get("hook_summary"), "type": r["features"].get("hook_type"),
                           "format": r["features"].get("format"), "why": r["features"].get("why_it_works"),
                           "outlier": r["outlier"], "views_per_hour": r["velocity"]} for r in best]}


class Pattern(BaseModel):
    statement: str = Field(description="one sentence a content planner can act on, with the numbers behind it")
    feature: str
    value: str
    lift: float
    n: int


class HookIdea(BaseModel):
    pattern: str = Field(description="the hook pattern, generalised")
    example: str = Field(description="an original example line in that pattern; never a copy of a post")


class Brief(BaseModel):
    working: list[Pattern] = Field(description="what is over-represented among outliers now")
    rising: list[str] = Field(description="gaining share this week (formats, hooks, sounds), with numbers")
    cooling: list[str] = Field(description="losing share or saturated, with numbers")
    hooks_to_try: list[HookIdea]
    avoid: list[str]
    summary: str = Field(description="three sentences at most")


BRIEF_SYSTEM = (
    "You are the trend analyst for a studio of openly AI-generated social creators. You get numbers about public "
    "short-form posts in one niche: which features are over-represented among posts that beat the niche's median "
    "views (lift = share among outliers / share among ordinary posts), how shares changed since last week, rising "
    "sounds, and the opening moves of the strongest posts. Write a brief a content planner can act on today. Only "
    "claim what the numbers show, and say how many posts back each claim; small samples are hints, not facts. "
    "This is correlation in other people's posts, not proof. Never copy anyone's words; hook examples must be "
    "original. Nothing that imitates a real person or uses a real brand.")


def _market_id(conn, name: str):
    return conn.execute("INSERT INTO markets (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name "
                        "RETURNING id", (name,)).fetchone()["id"]


def synthesize(conn, policy: Policy, llm, *, now: datetime) -> dict:
    cfg = _cfg(policy)
    out = {"briefs": 0, "findings": 0}
    day = now.date()
    for m in cfg.get("markets") or []:
        name = m["name"]
        if conn.execute("SELECT 1 FROM trend_briefs WHERE market = %s AND day = %s", (name, day)).fetchone():
            continue
        s = stats(conn, policy, name, now)
        if s["outliers"] < int(cfg.get("min_outliers_for_brief", 10)):
            continue
        if left(conn, policy, now, "synthesize") < float(cfg.get("usd_per_brief_estimate", 0.05)):
            out["budget"] = "synthesize share used"
            break
        try:
            with llm.tagged(f"{AGENT}:synthesize", None):
                brief = llm.structured("trend_analyst", system=BRIEF_SYSTEM, content=json.dumps(s, default=str),
                                       schema=Brief, effort="medium")
        except Exception as e:  # noqa: BLE001 - next tick
            out["errors"] = out.get("errors", []) + [f"{type(e).__name__}: {str(e)[:200]}"]
            continue
        conn.execute("INSERT INTO trend_briefs (market, day, stats, brief, model) VALUES (%s, %s, %s, %s, %s) "
                     "ON CONFLICT (market, day) DO NOTHING",
                     (name, day, Jsonb(json.loads(json.dumps(s, default=str))), Jsonb(brief.model_dump()),
                      llm.model_for("trend_analyst")))
        out["briefs"] += 1
        mid = _market_id(conn, name)
        conn.execute("UPDATE findings SET status = 'superseded', updated_at = now() WHERE scope = 'market' "
                     "AND market_id = %s AND evidence = 'research' AND status = 'open'", (mid,))
        tentative = int(cfg.get("tentative_min_n", 20))
        for p in brief.working[:int(cfg.get("findings_per_brief", 6))]:
            conn.execute(
                """INSERT INTO findings (scope, market_id, statement, evidence, grade, effect, n, sources)
                   VALUES ('market', %s, %s, 'research', %s, %s, %s, %s)""",
                (mid, p.statement[:500], "tentative" if p.n >= tentative and p.lift >= 2 else "speculative",
                 p.lift, p.n, Jsonb([{"trend_brief": str(day), "feature": p.feature, "value": p.value}])))
            out["findings"] += 1
    return out


def tick(conn, policy: Policy, llm, src, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    if not _cfg(policy).get("enabled"):
        return {"enabled": False}
    out = {"collect": collect(conn, policy, src, now=now), "analyze": analyze(conn, policy, llm, src, now=now),
           "synthesize": synthesize(conn, policy, llm, now=now)}
    out["spent_today"] = round(spent_today(conn, now), 4)
    return out


def brief(conn, market: str) -> dict | None:
    row = conn.execute("SELECT day, brief, stats FROM trend_briefs WHERE market = %s ORDER BY day DESC LIMIT 1",
                       (market,)).fetchone()
    return {"day": str(row["day"]), **row["brief"], "numbers": row["stats"]} if row else None
