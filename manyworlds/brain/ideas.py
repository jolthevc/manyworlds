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


# Plan fields a scheduled post carries (the reasoning stays on the idea, never on the post).
_POST_FIELDS = ("scenes", "script", "on_screen_text", "hashtags", "canon", "pillar", "tone", "series", "staging", "flavor")


def schedule_ready(conn, policy, *, now) -> dict:
    """Turn showrunner batches into planned posts once every idea has a volume-evaluator evaluation.
    After `brain.showrunner.max_wait_hours` a batch is scheduled anyway and its unevaluated ideas are
    dropped: no idea becomes a post without an evaluation."""
    from datetime import date, timedelta

    from psycopg.types.json import Jsonb

    from . import feed_qa, memory, planner
    max_wait = timedelta(hours=float(policy.get("brain", "showrunner", "max_wait_hours", default=6)))
    out = {"scheduled": 0, "dropped": 0, "batches": 0}
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
            for r, when in pairs:
                body = {k: r["body"].get(k) for k in _POST_FIELDS}
                body.update(platform_ai_label=True, idea_id=str(r["id"]))   # disclosure: every post sets the AI flag
                post = conn.execute(
                    "INSERT INTO posts (creator_id, platform, format, status, caption, body, scheduled_for, "
                    "profile_version) VALUES (%s, %s, %s, 'planned', %s, %s, %s, %s) RETURNING id",
                    (b["creator_id"], r["body"].get("platform", "instagram"), r["format"], r["body"].get("caption"),
                     Jsonb(body), when, version["profile_version"])).fetchone()["id"]
                conn.execute("UPDATE ideas SET status = 'scheduled', post_id = %s, updated_at = now() WHERE id = %s",
                             (post, r["id"]))
                if r["body"].get("canon"):
                    memory.add(conn, b["creator_id"], "canon", r["body"]["canon"], source_post=post,
                               key=r["body"].get("canon_key") or None)
                out["scheduled"] += 1
            conn.execute("INSERT INTO decisions (kind, creator_id, inputs, output, policy_version) "
                         "VALUES ('batch_scheduled', %s, %s, %s, %s)",
                         (b["creator_id"], Jsonb({"batch": b["batch"]}),
                          Jsonb({"scheduled": len(ready), "dropped": len(rows) - len(ready)}), policy.version))
        out["batches"] += 1
    return out
