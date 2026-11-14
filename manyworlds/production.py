"""Production: turn planned posts into finished, reviewed images. No Claude calls here
(the producer loop also runs brain.planner.ensure_plans, which does).

Each tick (`advance`):
  planned   -> producing  when the post is due within production.lead_hours: one image job
                          per scene (the creator's full appearance + LoRA, see lab.photo_prompt)
  producing -> ready      when every scene has an image that passed both review stages
                          (manyworlds.review.gates); the best one per scene is chosen
  producing -> dropped    when a scene used up review.max_repairs_per_post regenerations

A scene whose job failed, or whose images were all rejected, gets a new job with new seeds.
`ready` posts wait for approval (policy autonomy.publish) before anything is published.
Only formats listed in production.formats are made; video posts wait as 'planned' until their
format is added there.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from . import queue
from .creator.lab import photo_prompt
from .policy import Policy


def _cfg(policy: Policy) -> dict:
    return policy.get("production", default={})


def ensure_lora(conn, policy: Policy, creator_id) -> dict | None:
    """The creator's identity LoRA: attached to anchors from the latest finished training."""
    anchors = conn.execute("SELECT anchors FROM creators WHERE id = %s", (creator_id,)).fetchone()["anchors"] or {}
    if anchors.get("lora"):
        return anchors["lora"]
    row = conn.execute(
        """SELECT a.id, a.uri, j.payload->>'trigger' AS trigger FROM assets a JOIN jobs j ON j.id = a.job_id
           WHERE a.creator_id = %s AND a.kind = 'lora' AND j.status = 'done'
           ORDER BY a.created_at DESC LIMIT 1""", (creator_id,)).fetchone()
    if not row:
        return None
    lora = {"uri": row["uri"], "trigger": row["trigger"], "asset_id": str(row["id"]),
            "scale": float(policy.get("creator_lab", "lora_scale", default=1.0))}
    conn.execute("UPDATE creators SET anchors = anchors || %s WHERE id = %s", (Jsonb({"lora": lora}), creator_id))
    return lora


def _scene_job(conn, policy: Policy, post: dict, bible: dict, lora: dict | None, scene_i: int, attempt: int):
    cfg, img = _cfg(policy), policy.get("generation", "image", default={})
    payload = {"prompt": photo_prompt(bible, post["body"]["scenes"][scene_i], trigger=lora and lora.get("trigger")),
               "n": int(cfg.get("candidates_per_scene", 4)),
               "seed": int(str(post["id"]).replace("-", "")[:7], 16) + scene_i * 1000 + attempt * 10,
               "width": img.get("width", 896), "height": img.get("height", 1152),
               "params": {"steps": img.get("steps", 9), "guidance": img.get("guidance", 0.0)}}
    if lora:
        payload.update(lora_uri=lora["uri"], lora_scale=lora.get("scale", 1.0))
    return queue.enqueue(conn, cfg.get("image_kind", "image.zimage"), payload, creator_id=post["creator_id"],
                         batch=f"post-{post['id']}", post_id=post["id"],
                         idempotency_key=f"post:{post['id']}:{scene_i}:{attempt}")


def _set(conn, post_id, status: str, body: dict, **cols) -> None:
    sets = ", ".join(f"{k} = %({k})s" for k in cols)
    conn.execute(f"UPDATE posts SET status = %(status)s, body = %(body)s, updated_at = now()"
                 f"{', ' + sets if sets else ''} WHERE id = %(id)s",
                 {"status": status, "body": Jsonb(body), "id": post_id, **cols})


def start(conn, policy: Policy, post: dict) -> None:
    c = conn.execute("SELECT bible FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()
    lora = ensure_lora(conn, policy, post["creator_id"])
    body = dict(post["body"])
    body["scene_jobs"] = [[str(_scene_job(conn, policy, post, c["bible"], lora, i, 0))]
                          for i in range(len(body["scenes"]))]
    body["repairs"] = 0
    _set(conn, post["id"], "producing", body)


def _best_confirmed(conn, job_ids: list[str]) -> dict | None:
    """The best image of these jobs that passed every review row and was confirmed."""
    return conn.execute(
        """SELECT a.id, a.uri, sum(r.score) FILTER (WHERE r.details->>'stage' = 'confirm') AS score
           FROM assets a JOIN reviews r ON r.asset_id = a.id
           WHERE a.job_id = ANY(%s::uuid[]) AND a.kind = 'image'
           GROUP BY a.id, a.uri
           HAVING bool_and(r.passed) AND bool_or(r.details->>'stage' = 'confirm')
           ORDER BY score DESC NULLS LAST, a.uri LIMIT 1""", (job_ids,)).fetchone()


def _scene_state(conn, job_ids: list[str]) -> str:
    """'waiting' (still generating or being reviewed), 'chosen', 'rejected' (every image was
    reviewed and none passed) or 'failed' (the job itself failed: infrastructure, not taste)."""
    if _best_confirmed(conn, job_ids):
        return "chosen"
    last = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_ids[-1],)).fetchone()["status"]
    if last in ("queued", "running"):
        return "waiting"
    if last == "done":
        unreviewed = conn.execute(
            "SELECT count(*) n FROM assets a WHERE a.job_id = %s AND a.kind = 'image' "
            "AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id = a.id)", (job_ids[-1],)).fetchone()["n"]
        # The reviewer runs both stages in one pass, so reviewed means decided.
        if unreviewed:
            return "waiting"
        return "rejected"
    return "failed"


def _edit_job(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], attempt: int):
    """The narrated edit for a post whose photos are chosen: their voice reads the script."""
    anchors = conn.execute("SELECT anchors FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["anchors"]
    voice = (anchors.get("voice") or {}).get("voice", "af_heart")
    cfg = _cfg(policy).get("edit", {}) or {}
    payload = {"images": [a["uri"] for a in chosen], "script": body.get("script", ""), "voice": voice,
               "on_screen_text": body.get("on_screen_text", ""), **cfg}
    return queue.enqueue(conn, "image.narrated", payload, creator_id=post["creator_id"], batch=f"post-{post['id']}",
                         post_id=post["id"], idempotency_key=f"post:{post['id']}:edit:{attempt}")


def advance_post(conn, policy: Policy, post: dict) -> str:
    body = dict(post["body"])
    max_repairs = int(policy.get("review", "max_repairs_per_post", default=4))
    max_failures = int(_cfg(policy).get("max_job_failures_per_post", 6))
    chosen, waiting = [], False
    for i, jobs in enumerate(body["scene_jobs"]):
        state = _scene_state(conn, jobs)
        if state == "chosen":
            chosen.append(_best_confirmed(conn, jobs))
            continue
        if state == "waiting":
            waiting = True
            continue
        # Rejected images use a creative repair; failed jobs (machines, downloads) are counted
        # apart, so an infrastructure outage can't use up a post's repairs.
        key, limit = ("repairs", max_repairs) if state == "rejected" else ("job_failures", max_failures)
        if body.get(key, 0) >= limit:
            body["drop_reason"] = (f"scene {i}: no image passed review after {body[key]} repairs" if key == "repairs"
                                   else f"scene {i}: its job failed {body[key]} times")
            _set(conn, post["id"], "dropped", body, qa_status="failed")
            return "dropped"
        c = conn.execute("SELECT bible FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()
        lora = ensure_lora(conn, policy, post["creator_id"])
        jobs.append(str(_scene_job(conn, policy, post, c["bible"], lora, i, len(jobs))))
        body[key] = body.get(key, 0) + 1
        waiting = True
    if waiting:
        _set(conn, post["id"], "producing", body)
        return "producing"
    body["images"] = [{"asset_id": str(a["id"]), "uri": a["uri"]} for a in chosen]
    asset_ids = [a["id"] for a in chosen]
    if post["format"] == "narrated_edit":
        edits = body.setdefault("edit_jobs", [])
        if not edits:
            edits.append(str(_edit_job(conn, policy, post, body, chosen, 0)))
            _set(conn, post["id"], "producing", body)
            return "producing"
        job = conn.execute("SELECT status FROM jobs WHERE id = %s", (edits[-1],)).fetchone()
        if job["status"] in ("queued", "running"):
            _set(conn, post["id"], "producing", body)
            return "producing"
        video = conn.execute("SELECT id, uri FROM assets WHERE job_id = %s AND kind = 'video' LIMIT 1",
                             (edits[-1],)).fetchone()
        if not video:
            if len(edits) > max_failures:
                body["drop_reason"] = f"the edit failed {len(edits)} times"
                _set(conn, post["id"], "dropped", body, qa_status="failed")
                return "dropped"
            edits.append(str(_edit_job(conn, policy, post, body, chosen, len(edits))))
            _set(conn, post["id"], "producing", body)
            return "producing"
        body["video"] = {"asset_id": str(video["id"]), "uri": video["uri"]}
        asset_ids = [video["id"], *asset_ids]
    _set(conn, post["id"], "ready", body, qa_status="passed", asset_ids=asset_ids)
    return "ready"


def advance(conn, policy: Policy, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    lead = timedelta(hours=float(_cfg(policy).get("lead_hours", 48)))
    # Jobs that used their last attempt and lost the lease are dead, so their scene gets repaired
    # instead of waiting forever.
    out_reaped = queue.reap(conn)
    out = {"started": 0, "producing": 0, "ready": 0, "dropped": 0}
    out["reaped"] = out_reaped
    # A planned post in a format production doesn't make, whose time has passed, won't happen.
    made = list(_cfg(policy).get("formats", ["photo", "carousel"]))
    out["expired"] = len(conn.execute(
        """UPDATE posts SET status = 'dropped', updated_at = now(),
                  body = body || jsonb_build_object('drop_reason', 'format ' || format || ' not produced yet')
           WHERE status = 'planned' AND scheduled_for < %s AND NOT (format = ANY(%s)) RETURNING id""",
        (now, made)).fetchall())
    for post in conn.execute(
            # Only creators with reference images: the reviewer judges identity against them.
            "SELECT p.* FROM posts p JOIN creators c ON c.id = p.creator_id WHERE p.status = 'planned' "
            "AND coalesce(p.scheduled_for, now()) <= %s AND p.format = ANY(%s) "
            "AND jsonb_array_length(coalesce(c.anchors->'reference_images', '[]'::jsonb)) > 0 "
            "ORDER BY p.scheduled_for", (now + lead, list(_cfg(policy).get("formats", ["photo", "carousel"])))).fetchall():
        start(conn, policy, post)
        out["started"] += 1
    for post in conn.execute("SELECT * FROM posts WHERE status = 'producing' ORDER BY scheduled_for").fetchall():
        out[advance_post(conn, policy, post)] += 1
    return out
