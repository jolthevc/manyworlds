"""Content planner: one Brain call writes a creator's posts for the coming period.

The creator's profile (manyworlds.creator.profile) drives it: posts per day per platform,
photos per video, the tone mix, hashtag limits, content pillars, video series, the arc and
the signature item. Each post records its pillar, tone, series and the profile version it
was planned under, so the learning loop can see what performed. Recent posts are passed
back in for continuity. Captions that read as AI-written are dropped (`caption_issues`).

The producer calls `ensure_plans` so creators are always planned a few days ahead.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, time, timedelta, timezone

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from ..creator import profile as profile_mod
from ..llm import StructuredLLM
from ..policy import Policy

SYSTEM = """You plan posts for an original fictional social media creator at Manyworlds, a studio
whose creators are all openly labeled as AI. Posts show a specific, recognisable life: recurring
places, habits, opinions and jokes, and the creator's signature item, which appears naturally and
is never mentioned. Write like the person, not like a brand or an assistant: no tidy lists of
virtues, no inspirational lines, no calls to action, no hashtag stuffing. No claims of real
first-hand experience with products or services. Every image is a single candid smartphone photo;
describe what the camera sees. Every video must say something that adds to the creator's arc,
the audience's trust in them, or their world, never filler."""

ANALYSIS_FIELDS = ("look", "persona", "audience", "content", "arc", "places", "home_and_places", "interests",
                   "opinions_and_running_jokes", "display_name", "one_line", "niche")
BIBLE_FIELDS = ("display_name", "archetype", "niche", "one_line", "style_and_wardrobe", "home_and_places",
                "personality", "interests", "opinions_and_running_jokes", "caption_voice", "posting_habits", "never")
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


class PlannedPost(BaseModel):
    day: int = Field(description="0-based day within the period")
    platform: str
    format: str
    pillar: str = Field(default="", description="one of the creator's content pillars")
    tone: str = Field(default="", description="one of the creator's tones")
    series: str = Field(default="", description="for video posts: one of the creator's video series")
    scenes: list[str] = Field(description="one per image: setting, action, wardrobe, light, framing. "
                                          "The creator is the only clear face; no legible text, logos or brands")
    script: str = Field(default="", description="for narrated_edit: what they say, in their speaking voice, "
                                                "about 10-25 seconds; for story_clip: what moves in the shot")
    on_screen_text: str = Field(default="", description="short text overlay, if the format uses one")
    caption: str = Field(description="in the creator's caption voice")
    hashtags: list[str] = Field(description="without '#', lowercase; often none")
    canon: str = Field(description="one small fact this post adds to the creator's life, for continuity")


class ContentPlan(BaseModel):
    rationale: str
    posts: list[PlannedPost]


def _cfg(policy: Policy) -> dict:
    return policy.get("planner", default={})


def caption_issues(policy: Policy, text: str) -> list[str]:
    """Why a caption (or script) reads as AI-written, per policy `captions`; empty if fine."""
    cfg = policy.get("captions", default={})
    low = text.lower()
    issues = [f"phrase '{p}'" for p in cfg.get("banned_phrases", []) if p.lower() in low]
    if cfg.get("no_em_dash", True) and "—" in text:
        issues.append("em dash")
    if len(EMOJI.findall(text)) > int(cfg.get("max_emoji", 2)):
        issues.append("too many emoji")
    return issues


def _recent(conn, creator_id, n: int) -> list[dict]:
    rows = conn.execute(
        "SELECT format, caption, body->>'pillar' AS pillar, body->>'series' AS series, body->'scenes' AS scenes, "
        "body->>'canon' AS canon FROM posts WHERE creator_id = %s AND status <> 'dropped' "
        "ORDER BY coalesce(scheduled_for, created_at) DESC LIMIT %s", (creator_id, n)).fetchall()
    return [dict(r) for r in rows]


def _thesis(conn, creator_id) -> dict:
    row = conn.execute("SELECT body FROM theses WHERE creator_id = %s ORDER BY created_at DESC LIMIT 1",
                       (creator_id,)).fetchone()
    return {k: row["body"].get(k) for k in ("archetype", "audience", "format_mix")} if row else {}


def _counts(policy: Policy, dials: dict, days: int) -> dict:
    """Posts per platform over the period, and how many of them are videos."""
    per = {p: int(round(float(n) * days)) for p, n in (dials.get("posts_per_day") or {}).items()}
    total = sum(per.values())
    videos = int(math.floor(total / (float(dials.get("images_per_video", 3)) + 1)))
    return {"per_platform": per, "total": total, "videos": videos}


def _clean(policy: Policy, p: PlannedPost, cfg: dict, platforms: list[str], days: int, prof: dict | None,
           hashtags_max: dict) -> tuple[PlannedPost | None, str]:
    formats = cfg.get("formats", {"photo": [1, 1]})
    if p.format not in formats:
        return None, f"format {p.format}"
    if p.platform not in platforms:
        return None, f"platform {p.platform}"
    lo, hi = formats[p.format]
    p.scenes = [s.strip() for s in p.scenes if s.strip()]
    if len(p.scenes) < lo:
        return None, f"{p.format} with {len(p.scenes)} images"
    p.scenes = p.scenes[:hi]
    if prof:
        series = {s["name"] for s in prof["content"]["video_series"]}
        if p.format in cfg.get("video_formats", []) and p.series not in series:
            return None, f"video outside the creator's series ({p.series!r})"
    issues = caption_issues(policy, f"{p.caption} {p.script} {p.on_screen_text}")
    if issues:
        return None, "reads as AI: " + ", ".join(issues)
    p.day = min(max(p.day, 0), days - 1)
    p.hashtags = [h.strip().lstrip("#").lower() for h in p.hashtags if h.strip()][:int(hashtags_max.get(p.platform, 1))]
    return p, ""


def plan_period(conn, policy: Policy, llm: StructuredLLM, creator_id, *, start: date | None = None,
                force: bool = False) -> dict:
    """Write and store the creator's plan for `days` days from `start` (default tomorrow, UTC).
    A period that already has a plan is returned as it is unless `force`."""
    cfg = _cfg(policy)
    days = int(cfg.get("days", 3))
    start = start or (datetime.now(timezone.utc).date() + timedelta(days=1))
    existing = conn.execute("SELECT id FROM plans WHERE creator_id = %s AND period_start = %s",
                            (creator_id, start)).fetchone()
    if existing and not force:
        return {"plan_id": existing["id"], "existing": True,
                "posts": conn.execute("SELECT count(*) n FROM posts WHERE plan_id = %s", (existing["id"],)).fetchone()["n"]}

    c = conn.execute("SELECT handle, bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
    scope = policy.get("scope", default={})
    platforms = list(scope.get("platforms") or ["instagram"])
    formats = cfg.get("formats", {"photo": [1, 1]})
    cur = profile_mod.current(conn, creator_id)
    version, prof = cur if cur else (None, None)
    dials = (prof or {}).get("dials") or policy.get("creators", "dials_default", default={})
    counts = _counts(policy, dials, days)
    hashtags_max = dials.get("hashtags_max") or {}
    if prof:
        who = {k: prof[k] for k in ANALYSIS_FIELDS if k in prof}
        who["commercialization_rules"] = prof["commercialization"].get("rules", [])
        mix = dials.get("tone_mix") or {}
        tone_counts = {t: max(1, round(w * counts["total"])) for t, w in mix.items()}
    else:
        who = {k: c["bible"].get(k) for k in BIBLE_FIELDS}
        tone_counts = {}
    prompt = (
        f"The creator:\n{json.dumps(who, default=str)}\n"
        f"What this creator is testing: {json.dumps(_thesis(conn, creator_id), default=str)}\n"
        f"Recent posts, newest first (keep continuity; don't repeat scenes, jokes or captions):\n"
        f"{json.dumps(_recent(conn, creator_id, int(cfg.get('recent_posts', 30))), default=str)}\n\n"
        f"Plan exactly {counts['total']} posts over {days} days (day 0 to {days - 1}), spread evenly: "
        f"{counts['per_platform']} per platform. {counts['videos']} of them are videos "
        f"({cfg.get('video_formats')}), each in one of the creator's video series; the rest are photo posts. "
        f"Formats and images per post: { {f: f'{lo}-{hi} images' for f, (lo, hi) in formats.items()} }. "
        + (f"Tone mix across the period: {tone_counts}. " if tone_counts else "")
        + f"Hashtags per post at most {hashtags_max or 1}; often none. "
        f"Never touch: {scope.get('excluded_topics')}. No sponsored, product or review posts, no selling. "
        f"Vary places, times of day, light and framing; the creator's face should be visible in most images, "
        f"wearing their signature item.")
    role = cfg.get("role", "brain")
    out = llm.structured(role, system=SYSTEM, content=prompt, schema=ContentPlan,
                         effort=cfg.get("effort", "medium"), creator_id=creator_id)
    kept, dropped = [], []
    for p in out.posts:
        q, why = _clean(policy, p, cfg, platforms, days, prof, hashtags_max)
        (kept.append(q) if q else dropped.append({"caption": p.caption[:80], "why": why}))
    if not kept:
        raise ValueError(f"the plan had no usable posts: {dropped[:5]}")

    plan = conn.execute("INSERT INTO plans (creator_id, period_start, body) VALUES (%s, %s, %s) RETURNING id",
                        (creator_id, start, Jsonb({"rationale": out.rationale, "requested": counts, "kept": len(kept),
                                                   "dropped": dropped, "days": days, "profile_version": version,
                                                   "policy_version": policy.version}))).fetchone()
    hours = [int(h) for h in cfg.get("post_hours_utc", [17])]
    per_day: dict[int, int] = {}
    ids = []
    for p in sorted(kept, key=lambda p: p.day):
        k = per_day.get(p.day, 0)
        per_day[p.day] = k + 1
        when = datetime.combine(start + timedelta(days=p.day), time(hours[k % len(hours)]), tzinfo=timezone.utc)
        body = {"scenes": p.scenes, "hashtags": p.hashtags, "canon": p.canon, "pillar": p.pillar, "tone": p.tone,
                "series": p.series, "script": p.script, "on_screen_text": p.on_screen_text,
                "platform_ai_label": True}   # disclosure: every post sets the platform's AI flag
        ids.append(conn.execute(
            "INSERT INTO posts (creator_id, plan_id, platform, format, status, caption, body, scheduled_for, "
            "profile_version) VALUES (%s, %s, %s, %s, 'planned', %s, %s, %s, %s) RETURNING id",
            (creator_id, plan["id"], p.platform, p.format, p.caption, Jsonb(body), when, version)).fetchone()["id"])
    conn.execute(
        "INSERT INTO decisions (kind, creator_id, inputs, output, model, policy_version) VALUES (%s,%s,%s,%s,%s,%s)",
        ("content_plan", creator_id, Jsonb({"period_start": str(start), "requested": counts,
                                            "profile_version": version}),
         Jsonb({"plan_id": str(plan["id"]), "posts": len(ids), "rationale": out.rationale}),
         getattr(llm, "model_for", lambda r: "?")(role), policy.version))
    return {"plan_id": plan["id"], "existing": False, "posts": len(ids), "dropped_by_validation": len(dropped),
            "dropped": dropped}


def next_start(conn, creator_id, today: date) -> date:
    """The day after the last planned post, or tomorrow."""
    row = conn.execute("SELECT max(scheduled_for) t FROM posts WHERE creator_id = %s AND status <> 'dropped'",
                       (creator_id,)).fetchone()
    last = row["t"].date() if row["t"] else today
    return max(today + timedelta(days=1), last + timedelta(days=1))


def ensure_plans(conn, policy: Policy, llm: StructuredLLM, *, now: datetime | None = None) -> list[dict]:
    """Plan the next period for every creator with less than plan_ahead_days planned."""
    cfg = _cfg(policy)
    now = now or datetime.now(timezone.utc)
    ahead = timedelta(days=float(cfg.get("plan_ahead_days", 2)))
    out = []
    for c in conn.execute(
            "SELECT c.id, c.handle FROM creators c WHERE c.status = ANY(%s) AND c.profile_version IS NOT NULL "
            "AND jsonb_array_length(coalesce(c.anchors->'reference_images', '[]'::jsonb)) > 0",
            (list(cfg.get("auto_statuses", ["active"])),)).fetchall():
        start = next_start(conn, c["id"], now.date())
        if datetime.combine(start, time(0), tzinfo=timezone.utc) - now > ahead:
            continue
        try:
            r = plan_period(conn, policy, llm, c["id"], start=start)
            out.append({"creator": c["handle"], "start": str(start), "posts": r["posts"]})
        except Exception as e:  # noqa: BLE001 - one creator's bad plan must not stop the others
            out.append({"creator": c["handle"], "error": str(e)[:300]})
    return out
