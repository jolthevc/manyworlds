"""Production: turn planned posts into finished, reviewed images. No Claude calls here
(the producer loop also runs brain.planner.ensure_plans, which does).

Each tick (`advance`):
  planned   -> producing  when the post is due within production.lead_hours: one image job
                          per scene (the creator's full appearance + LoRA, see lab.photo_prompt)
  producing -> ready      when every scene has an image that passed both review stages
                          (manyworlds.review.gates); the best one per scene is chosen
  producing -> dropped    when a scene used up review.max_repairs_per_post regenerations

A scene whose job failed, or whose images were all rejected, gets a new job with new seeds.
Narrated edits and story clips are made from the chosen photos once all are chosen; a story clip
must also pass the video review gate (manyworlds.review.video), with review.video.max_repairs_per_post
new seeds before the post is dropped.
`ready` posts are published by manyworlds.publish.scheduler per policy autonomy.publish (autonomous, or held).
Only formats listed in production.formats are made; video posts wait as 'planned' until their
format is added there.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from psycopg.types.json import Jsonb

from . import queue
from .creator.lab import clip_prompt, photo_prompt, talk_prompt
from .review import repair
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


def _seed(post: dict, offset: int, attempt: int) -> int:
    """A deterministic seed per post, piece and attempt: a rerun asks for the same output, a retry for a new one.
    Offsets keep pieces apart (scenes: i*1000, story clip: 500, talk: 700 + segment*1000)."""
    return int(str(post["id"]).replace("-", "")[:7], 16) + offset + attempt * 10


class DropPost(Exception):
    """Raised by a step's job maker when the post must not be retried (e.g. unsafe output)."""


def _scene_job(conn, policy: Policy, post: dict, bible: dict, lora: dict | None, scene_i: int, attempt: int,
               adjust=None):
    cfg, img = _cfg(policy), policy.get("generation", "image", default={})
    img = {**img, **(policy.get("generation", "image_by_format", post["format"], default={}) or {})}   # Stories are 9:16
    payload = {"prompt": photo_prompt(bible, post["body"]["scenes"][scene_i], trigger=lora and lora.get("trigger")),
               "scene": post["body"]["scenes"][scene_i],   # the reviewer scores the image against it
               "n": int(cfg.get("candidates_per_scene", 4)),
               "seed": _seed(post, scene_i * 1000, attempt),
               "width": img.get("width", 896), "height": img.get("height", 1152),
               "params": {"steps": img.get("steps", 9), "guidance": img.get("guidance", 0.0)}}
    if lora:
        payload.update(lora_uri=lora["uri"], lora_scale=lora.get("scale", 1.0))
    if adjust:   # a targeted repair (manyworlds.review.repair)
        payload = adjust(payload)
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


# Every piece of a post (each scene's image, then the edit or clip made from them) is a Step:
# the job ids tried so far, kept in the post body, and how to read and retry the latest one.
# A state is (waiting | chosen | rejected | failed, result): rejected output gets a creative
# repair; failed jobs (machines, downloads) are counted apart, so an infrastructure outage
# can't use up a post's repairs.


@dataclass
class Step:
    jobs: list[str]                                    # job ids in the post body, oldest first
    state: Callable[[list[str]], tuple[str, Any]]
    make: Callable[[int], Any]                         # attempt number -> new job id
    repairs: tuple[str, int]                           # body counter and limit for rejected output
    failures: tuple[str, int]                          # body counter and limit for failed jobs
    thing: str                                         # "image", "edit", "clip": for the drop reason
    label: str = ""                                    # e.g. "scene 1: "


def advance_step(body: dict, step: Step) -> tuple[str, Any]:
    """('chosen', result), ('waiting', None) or ('dropped', None), starting a retry when needed."""
    if not step.jobs:
        step.jobs.append(str(step.make(0)))
        return "waiting", None
    state, result = step.state(step.jobs)
    if state in ("chosen", "waiting"):
        return state, result
    key, limit = step.repairs if state == "rejected" else step.failures
    if body.get(key, 0) >= limit:
        body["drop_reason"] = (f"{step.label}no {step.thing} passed review after {body[key]} repairs"
                               if state == "rejected" else f"{step.label}the {step.thing} job failed {body[key] + 1} times")
        return "dropped", None
    body[key] = body.get(key, 0) + 1
    try:
        step.jobs.append(str(step.make(len(step.jobs))))
    except DropPost as e:
        body["drop_reason"] = f"{step.label}{e}"
        return "dropped", None
    return "waiting", None


def _status(conn, job_id) -> str:
    return conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()["status"]


def _scene_state(conn, job_ids: list[str]) -> tuple[str, Any]:
    """chosen (an image passed both review stages), waiting (generating or being reviewed),
    rejected (every image reviewed, none passed) or failed (the job itself failed)."""
    best = _best_confirmed(conn, job_ids)
    if best:
        return "chosen", best
    last = _status(conn, job_ids[-1])
    if last in ("queued", "running"):
        return "waiting", None
    if last != "done":
        return "failed", None
    # The reviewer writes both stages at once, so reviewed means decided.
    unreviewed = conn.execute(
        "SELECT count(*) n FROM assets a WHERE a.job_id = %s AND a.kind = 'image' "
        "AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id = a.id)", (job_ids[-1],)).fetchone()["n"]
    return ("waiting", None) if unreviewed else ("rejected", None)


def _edit_state(conn, job_ids: list[str]) -> tuple[str, Any]:
    """An edit is not reviewed (its photos and script were): done with a video is chosen."""
    if _status(conn, job_ids[-1]) in ("queued", "running"):
        return "waiting", None
    video = conn.execute("SELECT id, uri FROM assets WHERE job_id = %s AND kind = 'video' LIMIT 1",
                         (job_ids[-1],)).fetchone()
    return ("chosen", video) if video else ("failed", None)


def _clip_state(conn, job_ids: list[str]) -> tuple[str, Any]:
    """Like _scene_state, judged by the video gate. A clip without keyframes can't be reviewed,
    so it counts as failed."""
    status = _status(conn, job_ids[-1])
    if status in ("queued", "running"):
        return "waiting", None
    video = conn.execute(
        """SELECT a.id, a.uri, count(r.id) AS n, bool_and(r.passed) AS passed,
                  bool_or(r.details->>'stage' = 'confirm') AS confirmed
           FROM assets a LEFT JOIN reviews r ON r.asset_id = a.id
           WHERE a.job_id = %s AND a.kind = 'video' GROUP BY a.id, a.uri LIMIT 1""", (job_ids[-1],)).fetchone()
    keyframes = conn.execute("SELECT count(*) n FROM assets WHERE job_id = %s AND kind = 'keyframe'",
                             (job_ids[-1],)).fetchone()["n"]
    if status != "done" or not video or not keyframes:
        return "failed", None
    if not video["n"]:
        return "waiting", None
    return ("chosen", video) if video["passed"] and video["confirmed"] else ("rejected", None)


def _edit_job(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], attempt: int):
    """The narrated edit for a post whose photos are chosen: their voice reads the script."""
    anchors = conn.execute("SELECT anchors FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["anchors"]
    voice = (anchors.get("voice") or {}).get("voice", "af_heart")
    cfg = _cfg(policy).get("edit", {}) or {}
    payload = {"images": [a["uri"] for a in chosen], "script": body.get("script", ""), "voice": voice,
               "on_screen_text": body.get("on_screen_text", ""), **cfg}
    return queue.enqueue(conn, "image.narrated", payload, creator_id=post["creator_id"], batch=f"post-{post['id']}",
                         post_id=post["id"], idempotency_key=f"post:{post['id']}:edit:{attempt}")


def _clip_job(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], attempt: int):
    """A story clip from the post's approved first photo: the scene, moving as the planner wrote."""
    bible = conn.execute("SELECT bible FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["bible"]
    vid, scene, motion = policy.get("generation", "video", default={}), body["scenes"][0], body.get("script", "")
    payload = {"image_uri": chosen[0]["uri"], "prompt": clip_prompt(bible, scene, motion),
               "scene": f"{scene} Movement: {motion}" if motion else scene,   # the video judge scores against it
               "width": vid.get("width", 704), "height": vid.get("height", 1280), "frames": vid.get("frames", 121),
               "keyframes": int(policy.get("review", "video", "keyframes", default=4)),
               "seed": _seed(post, 500, attempt),
               "params": {"steps": vid.get("steps", 30), "guidance": vid.get("guidance", 5.0), "fps": vid.get("fps", 24)}}
    kind = _cfg(policy).get("video_kind", "video.wan_i2v")
    if kind.startswith("hosted."):   # a hosted API renders it (generation.hosted_video): no GPU machine
        payload["params"] = dict(policy.get("generation", "hosted_video", default={}) or {})
    return queue.enqueue(conn, kind, payload, creator_id=post["creator_id"],
                         batch=f"post-{post['id']}", post_id=post["id"], idempotency_key=f"post:{post['id']}:clip:{attempt}")

def _talk_job(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], attempt: int, *,
              segment: int | None = None, start_uri: str | None = None):
    """A talking video: the creator says the script to camera from the approved photo, lip-synced,
    in their own voice (Kokoro on the talk worker). For a long script, one segment of it."""
    row = conn.execute("SELECT bible, anchors FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()
    voice = (row["anchors"] or {}).get("voice") or {}
    params = dict(policy.get("generation", "talk", default={}) or {})
    default_voice = params.pop("voice", "af_heart")
    scene = body["scenes"][0]
    script = body["talk_segments"][segment] if segment is not None else body.get("script", "")
    seed = _seed(post, 700 + (segment or 0) * 1000, attempt)
    payload = {"image_uri": start_uri or chosen[0]["uri"], "script": script, "voice": voice.get("voice", default_voice),
               "speed": float(voice.get("speed", 1.0)), "prompt": talk_prompt(row["bible"], scene),
               "scene": f"{scene} Speaking to camera: {script}",   # the video judge scores against it
               "keyframes": int(policy.get("review", "video", "keyframes", default=4)), "seed": seed, "params": params}
    if segment is not None:
        payload["segment"] = {"index": segment, "of": len(body["talk_segments"])}
    key = f"post:{post['id']}:talk:{attempt}" if segment is None else f"post:{post['id']}:talk:{segment}:{attempt}"
    return queue.enqueue(conn, _cfg(policy).get("talk_kind", "talk.infinitetalk"), payload,
                         creator_id=post["creator_id"], batch=f"post-{post['id']}", post_id=post["id"],
                         idempotency_key=key)


def _talk_long_cfg(policy: Policy) -> dict:
    return policy.get("generation", "talk_long", default={}) or {}


def talk_segments(policy: Policy, script: str) -> list[str]:
    from .talk_segments import split_script
    cfg = _talk_long_cfg(policy)
    return split_script(script, max_seconds=float(cfg.get("max_segment_seconds", 25)),
                        words_per_second=float(cfg.get("words_per_second", 2.5)))


def _last_frame(conn, video: dict) -> str:
    """The last keyframe of a chosen segment (keyframes always end on the clip's last frame)."""
    row = conn.execute(
        """SELECT k.uri FROM assets v JOIN assets k ON k.job_id = v.job_id AND k.kind = 'keyframe'
           WHERE v.id = %s ORDER BY (k.meta->>'frame')::int DESC NULLS LAST, k.uri DESC LIMIT 1""",
        (video["id"],)).fetchone()
    if not row:
        raise DropPost("a chosen segment has no keyframes to continue from")
    return row["uri"]


def join_talk(conn, policy: Policy, post: dict, videos: list[dict], store=None) -> dict:
    """The post's segments joined into one video asset. Idempotent: a rerun reuses the joined asset."""
    import tempfile
    from pathlib import Path

    from . import media, storage
    store = store or storage.from_url()
    key = f"assets/{post['creator_id']}/post-{post['id']}/talk-joined.mp4"
    uri = store.uri_for(key)
    existing = conn.execute("SELECT id, uri FROM assets WHERE uri = %s AND kind = 'video'", (uri,)).fetchone()
    if existing:
        return existing
    with tempfile.TemporaryDirectory(prefix="mw-join-") as d:
        paths = [store.get_file(v["uri"], Path(d) / f"segment_{i:02d}.mp4") for i, v in enumerate(videos)]
        out = media.concat(paths, Path(d) / "joined.mp4",
                           timeout=float(_talk_long_cfg(policy).get("join_timeout_s", 600)))
        stored = store.put_file(out, key)
    return conn.execute(
        """INSERT INTO assets (creator_id, kind, uri, sha256, bytes, meta, recipe)
           VALUES (%s, 'video', %s, %s, %s, %s, %s) RETURNING id, uri""",
        (post["creator_id"], stored.uri, stored.sha256, stored.bytes,
         Jsonb({"lip_sync": True, "segments": [str(v["id"]) for v in videos]}),
         Jsonb({"joined": [v["uri"] for v in videos], "join": _talk_long_cfg(policy).get("join", "cut")}))).fetchone()


def _advance_long_talk(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], store=None):
    """A script longer than one segment: every segment is a talk job of its own, judged by the video gate and
    repaired on its own; `cut` renders them all from the approved photo at once, `chain` starts each from the
    last frame of the one before. Once all are chosen they are joined into the post's video."""
    chain = _talk_long_cfg(policy).get("join", "cut") == "chain"
    jobs = body.setdefault("talk_segment_jobs", [[] for _ in body["talk_segments"]])
    repairs = int(policy.get("review", "video", "max_repairs_per_post", default=2))
    failures = int(_cfg(policy).get("max_job_failures_per_post", 6))
    results: list[tuple[str, Any]] = []
    for i in range(len(body["talk_segments"])):
        if chain and i > 0 and results[-1][0] != "chosen":
            results.append(("waiting", None))   # starts from the previous segment's last frame
            continue
        start_uri = _last_frame(conn, results[-1][1]) if chain and i > 0 else None
        step = Step(jobs[i], lambda j: _clip_state(conn, j),
                    lambda attempt, i=i, s=start_uri: _talk_job(conn, policy, post, body, chosen, attempt,
                                                                 segment=i, start_uri=s),
                    (f"segment_repairs_{i}", repairs), ("follow_up_failures", failures), "clip", f"segment {i}: ")
        results.append(advance_step(body, step))
        if results[-1][0] == "dropped":
            return "dropped", None
    if any(state != "chosen" for state, _ in results):
        return "waiting", None
    return "chosen", join_talk(conn, policy, post, [video for _, video in results], store)


# Formats made from the post's chosen photos: body key for their jobs, job maker, state, name.
FOLLOW_UPS = {"narrated_edit": ("edit_jobs", _edit_job, _edit_state, "edit"),
              "story_clip": ("clip_jobs", _clip_job, _clip_state, "clip"),
              "talking_video": ("talk_jobs", _talk_job, _clip_state, "clip")}


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

    return [Step(jobs, lambda j: _scene_state(conn, j), remake(i), repairs, ("job_failures", max_failures),
                 "image", f"scene {i}: ") for i, jobs in enumerate(body["scene_jobs"])]


def _follow_up(conn, policy: Policy, post: dict, body: dict, chosen: list[dict]) -> Step:
    key, make, state, thing = FOLLOW_UPS[post["format"]]
    return Step(body.setdefault(key, []), lambda j: state(conn, j),
                lambda attempt: make(conn, policy, post, body, chosen, attempt),
                ("clip_repairs", int(policy.get("review", "video", "max_repairs_per_post", default=2))),
                ("follow_up_failures", int(_cfg(policy).get("max_job_failures_per_post", 6))), thing)


def advance_post(conn, policy: Policy, post: dict, store=None) -> str:
    body = dict(post["body"])
    results = []
    for step in _scene_steps(conn, policy, post, body):
        results.append(advance_step(body, step))
        if results[-1][0] == "dropped":
            break   # no more repairs for a post that is being dropped
    states, chosen, video = {state for state, _ in results}, [r for _, r in results], None
    if states == {"chosen"}:
        body["images"] = [{"asset_id": str(a["id"]), "uri": a["uri"]} for a in chosen]
        if post["format"] in FOLLOW_UPS:   # the edit or clip, once every photo is chosen
            if post["format"] == "talking_video" and "talk_segments" not in body:
                segments = talk_segments(policy, body.get("script", ""))
                body["talk_segments"] = segments if len(segments) > 1 else []   # [] = one clip
            if body.get("talk_segments"):   # a long script: one job per segment, then joined
                state, video = _advance_long_talk(conn, policy, post, body, chosen, store)
            else:
                state, video = advance_step(body, _follow_up(conn, policy, post, body, chosen))
            target = (_cfg(policy).get("reroute_failed_video") or {}).get(post["format"])
            # Only when review kept rejecting the clips (not machine failures), and never for an experiment
            # idea, whose arm would then be measured on a photo.
            rejected = "passed review" in (body.get("drop_reason") or "")
            in_experiment = conn.execute("SELECT 1 FROM ideas WHERE post_id = %s AND experiment_id IS NOT NULL",
                                         (post["id"],)).fetchone()
            if state == "dropped" and target and rejected and not in_experiment:
                # The video kept failing; its start photo passed review. Ship that instead of nothing.
                body.update(rerouted_from=post["format"], reroute_reason=body.pop("drop_reason", ""))
                _set(conn, post["id"], "ready", body, qa_status="passed", asset_ids=[a["id"] for a in chosen],
                     format=target)
                return "ready"
            states = {state}
    if "dropped" in states:
        _set(conn, post["id"], "dropped", body, qa_status="failed")
        return "dropped"
    if "waiting" in states:
        _set(conn, post["id"], "producing", body)
        return "producing"
    asset_ids = [a["id"] for a in chosen]
    if video:
        body["video"] = {"asset_id": str(video["id"]), "uri": video["uri"]}
        asset_ids = [video["id"], *asset_ids]
    _set(conn, post["id"], "ready", body, qa_status="passed", asset_ids=asset_ids)
    return "ready"


def advance(conn, policy: Policy, *, now: datetime | None = None, store=None) -> dict:
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
        out[advance_post(conn, policy, post, store)] += 1
    return out
