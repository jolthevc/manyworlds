"""Ideas: one future post or Story as an experiment with measurable characteristics.

The showrunner (M3) writes ideas directly. `from_post` turns a post planned by today's planner
into an idea, so existing plans can be scored and calibrated before the showrunner exists.
"""

from __future__ import annotations

import logging
import uuid

from psycopg.types.json import Jsonb

log = logging.getLogger("manyworlds.ideas")

# Plan fields that describe the idea; anything else in a post body is bookkeeping.
_PLAN_FIELDS = ("scenes", "script", "on_screen_text", "hashtags", "canon", "pillar", "tone", "series", "shots")


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


# Plan fields a scheduled post carries (the reasoning stays on the idea, never on the post).
_POST_FIELDS = ("scenes", "script", "on_screen_text", "hashtags", "canon", "pillar", "tone", "series", "staging", "flavor")
# Direction (manyworlds.direction), carried only when the idea has it: an undirected post's body stays as before.
_DIRECTED_FIELDS = ("shots", "move")


def direct_posts(conn, policy, llm, creator_id, posts) -> int:
    """Shot sheets for newly scheduled posts (manyworlds.brain.director), for the formats in policy
    `direction.direct_formats` while `direction.enabled`. `posts` is [(post_id, format, body)]. A director
    failure never drops a post: it is logged, nothing is recorded, and the post is made undirected."""
    from . import budget, director
    if llm is None or not posts or not policy.get("direction", "enabled", default=True):
        return 0
    if budget.tag_for(conn, director.AGENT, creator_id).startswith(budget.SANDBOX):
        return 0   # sandbox test creators are never rendered: directing them would only spend
    formats = set(policy.get("direction", "direct_formats", default=[]) or [])
    directed = 0
    for post_id, fmt, body in posts:
        if fmt not in formats or body.get("shots"):
            continue
        try:
            shots = director.direct(conn, policy, llm, {**body, "format": fmt}, creator_id, fmt=fmt)
        except Exception as e:  # noqa: BLE001 - an undirected post is made exactly as before
            log.warning("director failed for post %s (%s: %s); the post proceeds undirected", post_id,
                        type(e).__name__, str(e)[:300])
            continue
        conn.execute("UPDATE posts SET body = body || %s, updated_at = now() WHERE id = %s AND status = 'planned'",
                     (Jsonb({"shots": shots}), post_id))
        directed += 1
    return directed


def schedule_ready(conn, policy, *, now, llm=None) -> dict:
    """Turn showrunner batches into planned posts once every idea has a volume-evaluator evaluation.
    After `brain.showrunner.max_wait_hours` a batch is scheduled anyway and its unevaluated ideas are
    dropped: no idea becomes a post without an evaluation. With an `llm`, each new post is then directed
    (`direct_posts`), outside the scheduling transaction so a slow or failed call never holds or undoes it."""
    from datetime import date, timedelta

    from psycopg.types.json import Jsonb

    from . import feed_qa, memory, planner
    max_wait = timedelta(hours=float(policy.get("brain", "showrunner", "max_wait_hours", default=6)))
    out = {"scheduled": 0, "dropped": 0, "batches": 0, "directed": 0}
    for b in conn.execute("SELECT batch, creator_id, min(created_at) AS oldest FROM ideas WHERE status = 'proposed' "
                          "AND batch IS NOT NULL GROUP BY batch, creator_id ORDER BY 3").fetchall():
        rows = conn.execute(
            """SELECT i.id, i.format, i.body, i.attributes, i.experiment_id,
                      (SELECT scores FROM evaluations e WHERE e.idea_id = i.id AND e.role = 'evaluator'
                       ORDER BY created_at DESC LIMIT 1) AS scores,
                      EXISTS (SELECT 1 FROM evaluations e WHERE e.idea_id = i.id
                                                      AND e.role = 'evaluator') AS evaluated,
                      EXISTS (SELECT 1 FROM jobs j WHERE j.kind = 'brain.evaluator' AND j.payload->>'idea_id' = i.id::text
                              AND j.status IN ('failed', 'dead'))
                      AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.kind = 'brain.evaluator'
                              AND j.payload->>'idea_id' = i.id::text AND j.status IN ('queued', 'running')) AS unevaluable
               FROM ideas i WHERE i.batch = %s AND i.status = 'proposed' ORDER BY (i.body->>'day')::int, i.created_at""",
            (b["batch"],)).fetchall()
        # Wait for evaluations still possible; an idea whose evaluation failed for good is dropped now.
        if not all(r["evaluated"] or r["unevaluable"] for r in rows) and now - b["oldest"] < max_wait:
            continue
        start = date.fromisoformat(b["batch"].rsplit(":", 1)[1])
        # Feed QA (manyworlds.brain.feed_qa): repetition, originality and voice, against recent posts.
        ready, qa_dropped = feed_qa.screen(conn, policy, b["creator_id"], [r for r in rows if r["evaluated"]])
        version = conn.execute("SELECT profile_version FROM creators WHERE id = %s", (b["creator_id"],)).fetchone()
        with conn.transaction():
            for r in rows:
                if not r["evaluated"]:
                    conn.execute("UPDATE ideas SET status = 'dropped', updated_at = now(), "
                                 "body = body || '{\"drop_reason\": \"not evaluated (failed or timed out)\"}' "
                                 "WHERE id = %s AND status = 'proposed'", (r["id"],))
                    out["dropped"] += 1
            for r, why in qa_dropped:
                conn.execute("UPDATE ideas SET status = 'dropped', updated_at = now(), body = body || %s "
                             "WHERE id = %s AND status = 'proposed'", (Jsonb({"drop_reason": why}), r["id"]))
                out["dropped"] += 1
            times = planner.slot_times(policy, start, [int(r["body"].get("day", 0)) for r in ready])
            # Claim the ideas: a concurrent scheduler (another producer) that got here first has
            # already taken them, and they must not become posts twice.
            claimed = {c["id"] for c in conn.execute(
                "UPDATE ideas SET status = 'scheduling', updated_at = now() WHERE id = ANY(%s) AND status = 'proposed' "
                "RETURNING id", ([r["id"] for r in ready],)).fetchall()}
            pairs = [(r, when) for r, when in zip(ready, times) if r["id"] in claimed]
            if not pairs:
                continue
            created = []
            for r, when in pairs:
                body = {k: r["body"].get(k) for k in _POST_FIELDS}
                body.update({k: r["body"][k] for k in _DIRECTED_FIELDS if r["body"].get(k)})
                body.update(platform_ai_label=True, idea_id=str(r["id"]))   # disclosure: every post sets the AI flag
                post = conn.execute(
                    "INSERT INTO posts (creator_id, platform, format, status, caption, body, scheduled_for, "
                    "profile_version) VALUES (%s, %s, %s, 'planned', %s, %s, %s, %s) RETURNING id",
                    (b["creator_id"], r["body"].get("platform", "instagram"), r["format"], r["body"].get("caption"),
                     Jsonb(body), when, version["profile_version"])).fetchone()["id"]
                conn.execute("UPDATE ideas SET status = 'scheduled', post_id = %s, updated_at = now() WHERE id = %s",
                             (post, r["id"]))
                created.append((post, r["format"], {**body, "caption": r["body"].get("caption")}))
                if r["body"].get("canon"):
                    memory.add(conn, b["creator_id"], "canon", r["body"]["canon"], source_post=post,
                               key=r["body"].get("canon_key") or None)
                out["scheduled"] += 1
            conn.execute("INSERT INTO decisions (kind, creator_id, inputs, output, policy_version) "
                         "VALUES ('batch_scheduled', %s, %s, %s, %s)",
                         (b["creator_id"], Jsonb({"batch": b["batch"]}),
                          Jsonb({"scheduled": len(ready), "dropped": len(rows) - len(ready)}), policy.version))
        out["directed"] += direct_posts(conn, policy, llm, b["creator_id"], created)
        out["batches"] += 1
    return out
