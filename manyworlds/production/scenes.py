"""Scene photos: the creator's identity LoRA, one image job per scene (registered sets, reference conditioning,
targeted repairs), starting a planned post, and the scene Steps advance_post walks."""

from __future__ import annotations

from psycopg.types.json import Jsonb

from .. import queue
from ..creator.lab import photo_prompt
from ..policy import Policy
from ..review import repair
from .common import FOLLOW_UP_FORMATS, _cfg, _seed, _set, _shot, _with_shot, candidates_per_scene, piece_key
from .steps import DropPost, Step, _best_confirmed, _scene_state


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


def is_video_format(policy: Policy, fmt: str) -> bool:
    """Whether a format is a video: one production makes from photos (FOLLOW_UP_FORMATS, the keys of
    followups.FOLLOW_UPS), or one of planner.video_formats."""
    return fmt in FOLLOW_UP_FORMATS or fmt in (policy.get("planner", "video_formats", default=[]) or [])


def scene_size(policy: Policy, fmt: str) -> dict:
    """A scene photo's generation settings: generation.image, under the format's generation.image_by_format entry,
    else (any video format) the `video` entry: every video's scene photos are 9:16, the frame the video is made in."""
    by_format = policy.get("generation", "image_by_format", default={}) or {}
    over = by_format.get(fmt)
    if over is None and is_video_format(policy, fmt):
        over = by_format.get("video")
    return {**(policy.get("generation", "image", default={}) or {}), **(over or {})}


def _scene_job(conn, policy: Policy, post: dict, bible: dict, lora: dict | None, scene_i: int, attempt: int,
               adjust=None):
    cfg, img = _cfg(policy), scene_size(policy, post["format"])   # Stories and every video format are 9:16
    from ..creator import sets
    # A registered set or outfit (creator.sets): its canonical description goes into the shot, its plate and outfit
    # reference into `references` (the consistency check compares). Nothing registered: the shot as it was.
    shot, references = sets.apply(conn, policy, post["creator_id"], _shot(post["body"], scene_i))
    # A shared set (sets.shared_set): the post's first accepted photo of the place is this scene's SET reference,
    # the image the scene is conditioned on (image.qwen_edit, sets.shared_set.conditioning) and judged against
    # (gate set_match). The leader's photo is made first, from text (_scene_steps holds the others).
    shared = _shared_reference(conn, policy, post, scene_i)
    if shared:
        references = [{"kind": "set", "uri": shared["uri"]}, *[r for r in references if r.get("kind") != "set"]]
    # Reference-conditioned (sets.reference_conditioning, off until benchmarked): image.qwen_edit from the anchor,
    # plate and outfit reference; the creator's LoRA (and its trigger word) is for Z-Image only.
    from ..direction import faceless
    force = bool(shared) and bool(sets.shared_cfg(policy).get("conditioning")) and not faceless(shot)
    images = sets.conditioning(conn, policy, post["creator_id"], references, force=force)
    trigger = None if images else (lora and lora.get("trigger"))
    payload = {"prompt": photo_prompt(bible, post["body"]["scenes"][scene_i], trigger=trigger,
                                      shot=shot, policy=policy),
               "scene": post["body"]["scenes"][scene_i],   # the reviewer scores the image against it
               "n": candidates_per_scene(conn, policy, post["creator_id"]),   # within her production tier
               "seed": _seed(post, scene_i * 1000, attempt),
               "width": img.get("width", 896), "height": img.get("height", 1152),
               "params": {"steps": img.get("steps", 9), "guidance": img.get("guidance", 0.0)}}
    _with_shot(payload, shot)
    if shared:
        payload["shared_set"] = {"set_id": shared["set_id"], "leader": shared["leader"]}
    if references:
        payload["references"] = references
    kind, pools = cfg.get("image_kind", "image.zimage"), None
    if images:
        kind, pools = sets.edit_kind(policy), sets.edit_pools(policy)
        payload = sets.edit_payload(policy, payload, images)
    elif lora:
        payload.update(lora_uri=lora["uri"], lora_scale=lora.get("scale", 1.0))
    if adjust:   # a targeted repair (manyworlds.review.repair)
        payload = adjust(payload)
    return queue.enqueue(conn, kind, payload, creator_id=post["creator_id"],
                         batch=f"post-{post['id']}", post_id=post["id"], pools=pools,
                         idempotency_key=f"{piece_key(post)}:{scene_i}:{attempt}")


def _leader(policy: Policy, post: dict, i: int) -> int | None:
    from ..creator import sets
    body = post["body"]
    return sets.shared_leader(policy, post["format"], body.get("shots"), i)


def _shared_reference(conn, policy: Policy, post: dict, i: int) -> dict | None:
    """{uri, set_id, leader} of the shared set's accepted leader photo for scene i, or None (no shared set, or
    scene i leads it, or its leader has no accepted photo yet)."""
    leader = _leader(policy, post, i)
    jobs = (post["body"].get("scene_jobs") or [])
    if leader is None or leader >= len(jobs) or not jobs[leader]:
        return None
    best = _best_confirmed(conn, [str(j) for j in jobs[leader]])
    if not best:
        return None
    sid = ((post["body"]["shots"][i].get("set") or {}).get("set_id") or "").strip()
    return {"uri": best["uri"], "set_id": sid, "leader": leader}


def start(conn, policy: Policy, post: dict) -> None:
    c = conn.execute("SELECT bible FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()
    lora = ensure_lora(conn, policy, post["creator_id"])
    body = dict(post["body"])
    # a shared set's followers wait for its leader's photo (_scene_steps starts them)
    body["scene_jobs"] = [[] if _leader(policy, post, i) is not None else
                          [str(_scene_job(conn, policy, post, c["bible"], lora, i, 0))]
                          for i in range(len(body["scenes"]))]
    body["repairs"] = 0
    _set(conn, post["id"], "producing", body)


def _scene_steps(conn, policy: Policy, post: dict, body: dict) -> list[Step]:
    max_failures = int(_cfg(policy).get("max_job_failures_per_post", 6))
    repairs = ("repairs", int(policy.get("review", "max_repairs_per_post", default=4)))

    def remake(i):
        def make(attempt):
            bible = conn.execute("SELECT bible FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["bible"]
            adjust = None
            if attempt > 0:   # a retry: repair what the latest reviewed job's review points to
                # (a job that failed on its machine has no review; the repair before it still applies)
                last = next((j for j in reversed(body["scene_jobs"][i]) if repair.reviewed(conn, j)), None)
                diag = repair.diagnose(conn, last) if last else {"cause": None, "failed": {}, "defects": []}
                if diag["cause"] == "safety" and policy.get("review", "repair", "drop_on_safety", default=True):
                    raise DropPost(f"safety: {diag['failed']} {diag['defects']}; unsafe output is never regenerated")
                if diag["cause"]:
                    done = sum(1 for e in body.get("repair_log", []) if e["scene"] == i and e["cause"] == "identity")
                    adjust, action = repair.plan(policy, diag["cause"], scene=body["scenes"][i], identity_repairs=done)
                    body.setdefault("repair_log", []).append(
                        {"scene": i, "attempt": attempt, "cause": diag["cause"], "action": action,
                         "defects": diag["defects"]})
            return _scene_job(conn, policy, post, bible, ensure_lora(conn, policy, post["creator_id"]), i, attempt,
                              adjust=adjust)
        return make

    def hold(i):
        leader = _leader(policy, post, i)
        if leader is None:
            return None
        return lambda: _shared_reference(conn, policy, post, i) is None   # until the leader's photo is accepted

    return [Step(jobs, lambda j: _scene_state(conn, j), remake(i), repairs, ("job_failures", max_failures),
                 "image", f"scene {i}: ", hold=hold(i)) for i, jobs in enumerate(body["scene_jobs"])]
