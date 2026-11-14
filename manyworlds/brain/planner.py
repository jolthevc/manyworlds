"""Content planner: one Brain call writes a creator's posts for the coming period.

Each planned post is a format, a platform, one scene per image, a caption in the creator's
voice, and one small "canon" fact it adds to their life. Recent posts are passed back in so
the creator stays consistent and doesn't repeat itself. Production (manyworlds.production)
turns each scene into image jobs; publishing waits for approval (policy autonomy.publish).
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from ..llm import StructuredLLM
from ..policy import Policy

SYSTEM = """You plan posts for an original fictional social media creator at Manyworlds, a studio
whose creators are all openly labeled as AI. Posts show an ordinary, phone-native life: specific,
mundane, recurring places and habits, small opinions and jokes. No forced storylines, no
milestones that need proof, no claims of real first-hand experience with products or services.
Every image is a single candid smartphone photo of the creator; describe what the camera sees."""

BIBLE_FIELDS = ("display_name", "archetype", "niche", "one_line", "style_and_wardrobe", "home_and_places",
                "personality", "interests", "opinions_and_running_jokes", "caption_voice", "posting_habits", "never")


class PlannedPost(BaseModel):
    day: int = Field(description="0-based day within the period")
    platform: str
    format: str
    scenes: list[str] = Field(description="one per image: setting, action, wardrobe, light, framing. "
                                          "The creator is the only clear face; no legible text, logos or brands")
    caption: str = Field(description="in the creator's caption voice")
    hashtags: list[str] = Field(description="without '#', lowercase")
    canon: str = Field(description="one small fact this post adds to the creator's life, for continuity")


class ContentPlan(BaseModel):
    rationale: str
    posts: list[PlannedPost]


def _cfg(policy: Policy) -> dict:
    return policy.get("planner", default={})


def _recent(conn, creator_id, n: int) -> list[dict]:
    rows = conn.execute(
        "SELECT format, caption, body->'scenes' AS scenes, body->>'canon' AS canon, status FROM posts "
        "WHERE creator_id = %s AND status <> 'dropped' ORDER BY coalesce(scheduled_for, created_at) DESC LIMIT %s",
        (creator_id, n)).fetchall()
    return [dict(r) for r in rows]


def _thesis(conn, creator_id) -> dict:
    row = conn.execute("SELECT body FROM theses WHERE creator_id = %s ORDER BY created_at DESC LIMIT 1",
                       (creator_id,)).fetchone()
    return {k: row["body"].get(k) for k in ("archetype", "audience", "format_mix")} if row else {}


def _clean(p: PlannedPost, cfg: dict, platforms: list[str], days: int) -> PlannedPost | None:
    formats = cfg.get("formats", {"photo": [1, 1]})
    if p.format not in formats or p.platform not in platforms or not p.scenes:
        return None
    lo, hi = formats[p.format]
    if len(p.scenes) < lo:
        return None
    p.scenes = [s.strip() for s in p.scenes[:hi] if s.strip()]
    p.day = min(max(p.day, 0), days - 1)
    p.hashtags = [h.strip().lstrip("#").lower() for h in p.hashtags if h.strip()][:int(cfg.get("max_hashtags", 5))]
    return p


def plan_period(conn, policy: Policy, llm: StructuredLLM, creator_id, *, start: date | None = None,
                force: bool = False) -> dict:
    """Write and store the creator's plan for `days` days from `start` (default tomorrow, UTC).
    A period that already has a plan is returned as it is unless `force`."""
    cfg = _cfg(policy)
    days = int(cfg.get("days", 7))
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
    n_posts = int(cfg.get("posts_per_period", days))
    prompt = (
        f"The creator:\n{json.dumps({k: c['bible'].get(k) for k in BIBLE_FIELDS}, default=str)}\n"
        f"What this creator is testing: {json.dumps(_thesis(conn, creator_id), default=str)}\n"
        f"Recent posts, newest first (keep continuity, don't repeat scenes or jokes):\n"
        f"{json.dumps(_recent(conn, creator_id, int(cfg.get('recent_posts', 21))), default=str)}\n\n"
        f"Plan exactly {n_posts} posts over {days} days (day 0 to {days - 1}). "
        f"Platforms: {platforms}. Formats and images per post: "
        f"{ {f: f'{lo}-{hi} images' for f, (lo, hi) in formats.items()} }. "
        f"Never touch: {scope.get('excluded_topics')}. No sponsored or product-review posts. "
        f"Vary places, times of day, light and framing; the creator's face should be visible in most images.")
    out = llm.structured(cfg.get("role", "brain"), system=SYSTEM, content=prompt, schema=ContentPlan,
                         effort=cfg.get("effort", "medium"), creator_id=creator_id)
    posts = [p for p in (_clean(p, cfg, platforms, days) for p in out.posts) if p]
    if not posts:
        raise ValueError("the plan had no usable posts")

    plan = conn.execute("INSERT INTO plans (creator_id, period_start, body) VALUES (%s, %s, %s) RETURNING id",
                        (creator_id, start, Jsonb({"rationale": out.rationale, "requested": n_posts,
                                                   "kept": len(posts), "policy_version": policy.version}))).fetchone()
    hours = [int(h) for h in cfg.get("post_hours_utc", [17])]
    per_day: dict[int, int] = {}
    ids = []
    for p in sorted(posts, key=lambda p: p.day):
        k = per_day.get(p.day, 0)
        per_day[p.day] = k + 1
        when = datetime.combine(start + timedelta(days=p.day), time(hours[k % len(hours)]), tzinfo=timezone.utc)
        body = {"scenes": p.scenes, "hashtags": p.hashtags, "canon": p.canon,
                "platform_ai_label": True}   # disclosure: every post sets the platform's AI flag
        ids.append(conn.execute(
            "INSERT INTO posts (creator_id, plan_id, platform, format, status, caption, body, scheduled_for) "
            "VALUES (%s, %s, %s, %s, 'planned', %s, %s, %s) RETURNING id",
            (creator_id, plan["id"], p.platform, p.format, p.caption, Jsonb(body), when)).fetchone()["id"])
    conn.execute(
        "INSERT INTO decisions (kind, creator_id, inputs, output, model, policy_version) VALUES (%s,%s,%s,%s,%s,%s)",
        ("content_plan", creator_id, Jsonb({"period_start": str(start), "requested": n_posts}),
         Jsonb({"plan_id": str(plan["id"]), "posts": len(ids), "rationale": out.rationale}),
         getattr(llm, "model_for", lambda r: "?")(cfg.get("role", "brain")), policy.version))
    return {"plan_id": plan["id"], "existing": False, "posts": len(ids), "dropped_by_validation": len(out.posts) - len(ids)}
