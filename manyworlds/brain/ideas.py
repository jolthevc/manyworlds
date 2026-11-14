"""Ideas: one future post or Story as an experiment with measurable characteristics.

The showrunner (M3) writes ideas directly. `from_post` turns a post planned by today's planner
into an idea, so existing plans can be scored and calibrated before the showrunner exists.
"""

from __future__ import annotations

import uuid

from psycopg.types.json import Jsonb

# Plan fields that describe the idea; anything else in a post body is bookkeeping.
_PLAN_FIELDS = ("scenes", "script", "on_screen_text", "hashtags", "canon", "pillar", "tone", "series")


def concept_text(fmt: str, body: dict, caption: str | None) -> str:
    parts = [f"{fmt}:"]
    scenes = body.get("scenes") or []
    if scenes:
        parts.append(" | ".join(scenes))
    if body.get("script"):
        parts.append(f"script: {body['script']}")
    if caption:
        parts.append(f"caption: {caption}")
    return " ".join(parts)


def from_post(conn, post_id) -> uuid.UUID:
    existing = conn.execute("SELECT id FROM ideas WHERE post_id = %s", (post_id,)).fetchone()
    if existing:
        return existing["id"]
    p = conn.execute("SELECT creator_id, format, caption, body FROM posts WHERE id = %s", (post_id,)).fetchone()
    body = {k: v for k, v in (p["body"] or {}).items() if k in _PLAN_FIELDS}
    body["caption"] = p["caption"]
    attributes = {k: body[k] for k in ("pillar", "tone", "series") if body.get(k)}
    return conn.execute(
        "INSERT INTO ideas (creator_id, concept, format, body, attributes, status, post_id) "
        "VALUES (%s, %s, %s, %s, %s, 'planned', %s) RETURNING id",
        (p["creator_id"], concept_text(p["format"], body, p["caption"]), p["format"], Jsonb(body),
         Jsonb(attributes), post_id)).fetchone()["id"]
