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
A creator with a designed voice (voices.engine: voxcpm) has every spoken piece (a talk take or segment, a cutaway,
a narrated edit's script) voiced first by a voice.speak job; the piece's job starts once that is done, with its wav
as `audio_uri`. A Kokoro creator's pieces are voiced on their own workers, as before.
Walk-and-talk (generation.talk_motion): a talking take whose shot moves (a walking or standing-up beat, a camera
move) gets a silent motion clip first, from the take's photo on the story-clip path, alongside its voice line; the
talk job then lip-syncs that clip (`video_uri`, InfiniteTalk video-to-video, only on the ComfyUI talk engine). A take
longer than one motion clip (max_motion_seconds) gets a chain of them, each from the last frame of the one before,
joined here before the lip sync, up to max_motion_chain; past that the take ends on a hold. Production waits for no
review of the motion clips; the take is judged.
Finish (`finish_video`, policy `finishing` and `burned_captions`, each with its `formats`): once a video post's video
is made (and joined), it is finished once on the control plane: grade, grain and loudness (manyworlds.finishing) and
the spoken words burned in as captions (manyworlds.captions, timed from the voice lines), in one ffmpeg pass where
this ffmpeg has libass; the cover frame is picked from its keyframes. The post's `video` is the finished file
(post-<id>/final.mp4, reused on a rerun), `video_raw` the one it was made from.
`ready` posts are published by manyworlds.publish.scheduler per policy autonomy.publish (autonomous, or held).
Only formats listed in production.formats are made; video posts wait as 'planned' until their
format is added there.
"""

from __future__ import annotations

import math
import re
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
    Offsets keep pieces apart (scenes: i*1000, story clip: 500, talk: 700 + segment*1000, voice: 800 + segment*1000,
    motion clip: 900 + segment*1000 + 3*clip in its chain)."""
    return int(str(post["id"]).replace("-", "")[:7], 16) + offset + attempt * 10


def unmarked(text: str) -> str:
    """Text as spoken: without the *emphasis* asterisks (they only style the burned-in captions)."""
    return (text or "").replace("*", "")


class DropPost(Exception):
    """Raised by a step's job maker when the post must not be retried (e.g. unsafe output)."""


def _shot(body: dict, i: int) -> dict | None:
    """Scene i's shot sheet (manyworlds.direction; body.shots is aligned with body.scenes), or None for a post
    planned without shots, which keeps today's prompts exactly."""
    shots = body.get("shots") or []
    return shots[i] if 0 <= i < len(shots) and isinstance(shots[i], dict) else None


def _with_shot(payload: dict, shot: dict | None) -> dict:
    """The job carries its shot, so the review gate can judge the direction (manyworlds.review)."""
    if shot is not None:
        payload["shot"] = shot
    return payload


def _scene_job(conn, policy: Policy, post: dict, bible: dict, lora: dict | None, scene_i: int, attempt: int,
               adjust=None):
    cfg, img = _cfg(policy), policy.get("generation", "image", default={})
    img = {**img, **(policy.get("generation", "image_by_format", post["format"], default={}) or {})}   # Stories are 9:16
    from .creator import sets
    # A registered set or outfit (creator.sets): its canonical description goes into the shot, its plate and outfit
    # reference into `references` (the consistency check compares). Nothing registered: the shot as it was.
    shot, references = sets.apply(conn, policy, post["creator_id"], _shot(post["body"], scene_i))
    # Reference-conditioned (sets.reference_conditioning, off until benchmarked): image.qwen_edit from the anchor,
    # plate and outfit reference; the creator's LoRA (and its trigger word) is for Z-Image only.
    images = sets.conditioning(conn, policy, post["creator_id"], references)
    trigger = None if images else (lora and lora.get("trigger"))
    payload = {"prompt": photo_prompt(bible, post["body"]["scenes"][scene_i], trigger=trigger,
                                      shot=shot, policy=policy),
               "scene": post["body"]["scenes"][scene_i],   # the reviewer scores the image against it
               "n": int(cfg.get("candidates_per_scene", 4)),
               "seed": _seed(post, scene_i * 1000, attempt),
               "width": img.get("width", 896), "height": img.get("height", 1152),
               "params": {"steps": img.get("steps", 9), "guidance": img.get("guidance", 0.0)}}
    _with_shot(payload, shot)
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
                  bool_or(r.details->>'stage' = 'confirm' OR r.details->>'clear_pass' = 'true') AS confirmed
           FROM assets a LEFT JOIN reviews r ON r.asset_id = a.id
           WHERE a.job_id = %s AND a.kind = 'video' GROUP BY a.id, a.uri LIMIT 1""", (job_ids[-1],)).fetchone()
    keyframes = conn.execute("SELECT count(*) n FROM assets WHERE job_id = %s AND kind = 'keyframe'",
                             (job_ids[-1],)).fetchone()["n"]
    if status != "done" or not video or not keyframes:
        return "failed", None
    if not video["n"]:
        return "waiting", None
    return ("chosen", video) if video["passed"] and video["confirmed"] else ("rejected", None)


def _edit_job(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], attempt: int, *,
              audio: dict | None = None):
    """The narrated edit for a post whose photos are chosen: their voice reads the script (Kokoro on the edit
    worker, or the designed voice's finished line, `audio`)."""
    anchors = conn.execute("SELECT anchors FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["anchors"]
    voice = (anchors.get("voice") or {}).get("voice", "af_heart")
    cfg = _cfg(policy).get("edit", {}) or {}
    payload = {"images": [a["uri"] for a in chosen], "script": unmarked(body.get("script", "")), "voice": voice,
               "on_screen_text": body.get("on_screen_text", ""), **cfg}
    _with_audio(payload, audio, captions=True)
    if captions_burned(policy, "narrated_edit"):   # burned in at the finish stage instead of drawn by the worker
        payload["captions"] = False
    return queue.enqueue(conn, "image.narrated", payload, creator_id=post["creator_id"], batch=f"post-{post['id']}",
                         post_id=post["id"], idempotency_key=f"post:{post['id']}:edit:{attempt}")


def _clip_job(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], attempt: int):
    """A story clip from the post's approved first photo: the scene, moving as the planner wrote."""
    bible = conn.execute("SELECT bible FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["bible"]
    vid, scene, motion = policy.get("generation", "video", default={}), body["scenes"][0], body.get("script", "")
    shot = _shot(body, 0)   # a story clip is one shot, from scene 0's photo
    payload = {"image_uri": chosen[0]["uri"], "prompt": clip_prompt(bible, scene, motion, shot=shot, policy=policy),
               "scene": f"{scene} Movement: {motion}" if motion else scene,   # the video judge scores against it
               "width": vid.get("width", 704), "height": vid.get("height", 1280), "frames": vid.get("frames", 121),
               "keyframes": int(policy.get("review", "video", "keyframes", default=4)),
               "seed": _seed(post, 500, attempt),
               "params": {"steps": vid.get("steps", 30), "guidance": vid.get("guidance", 5.0), "fps": vid.get("fps", 24)}}
    _with_shot(payload, shot)
    kind = _story_clip_kind(conn, policy, post)
    _clip_params(policy, kind, payload)
    return queue.enqueue(conn, kind, payload, creator_id=post["creator_id"],
                         batch=f"post-{post['id']}", post_id=post["id"], idempotency_key=f"post:{post['id']}:clip:{attempt}")


def _story_clip_kind(conn, policy: Policy, post: dict) -> str:
    kind = _cfg(policy).get("video_kind", "video.wan_i2v")
    if (_cfg(policy).get("routing") or {}).get("story_clip"):   # the cheapest per accepted clip (manyworlds.routing)
        from . import routing
        kind, _ = routing.choose(conn, policy, "story_clip", key=str(post["id"]))
    return kind


def _clip_params(policy: Policy, kind: str, payload: dict) -> dict:
    """A clip job's render settings for its kind (a video.* job keeps the payload's own)."""
    if kind.startswith("hosted."):   # a hosted API renders it (generation.hosted_video): no GPU machine
        payload["params"] = dict(policy.get("generation", "hosted_video", default={}) or {})
    elif kind.startswith("comfy."):  # the ComfyUI worker's Wan 2.2 graph (generation.comfy_video)
        payload["params"] = dict(policy.get("generation", "comfy_video", default={}) or {})
    return payload

def _talk_job(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], attempt: int, *,
              segment: int | None = None, start_uri: str | None = None, audio: dict | None = None,
              motion: dict | None = None):
    """A talking video: the creator says the script to camera from the approved photo, lip-synced,
    in their own voice (Kokoro on the talk worker, or the designed voice's finished line, `audio`: the worker then
    speaks nothing, and `script` stays for the judge). For a long script, one segment of it. A directed post's take
    (body.talk_shots) uses its shot's scene, prompt and, when every shot has its own photo, that photo. With
    `motion` (the take's silent motion clip, walk-and-talk), the clip is lip-synced instead of the photo
    (`video_uri`, generation.talk_motion.v2v settings), when the talk engine is ComfyUI (`talk_v2v`)."""
    row = conn.execute("SELECT bible, anchors FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()
    voice = (row["anchors"] or {}).get("voice") or {}
    params = dict(policy.get("generation", "talk", default={}) or {})
    default_voice = params.pop("voice", "af_heart")
    params.pop("max_take_seconds", None)   # production's take limit, not a render setting
    i = _take_index(body, segment)
    scene, shot = body["scenes"][i], _shot(body, i)
    from .direction import talk_profile_for
    profile = talk_profile_for(shot, policy)   # 720p close-ups (direction.hd_close_ups; off as shipped)
    if profile:
        params.update(policy.get("generation", "talk_profiles", profile, default={}) or {})
    motion = motion if motion and talk_v2v(policy) else None   # the research-script engine has no V2V
    if motion:
        params.update(_motion_cfg(policy).get("v2v") or {})
    script = unmarked(_take_script(body, segment))
    seed = _seed(post, 700 + (segment or 0) * 1000, attempt)
    payload = {"image_uri": start_uri or _take_photo(body, chosen, i)["uri"], "script": script,
               "voice": voice.get("voice", default_voice), "speed": float(voice.get("speed", 1.0)),
               "prompt": talk_prompt(row["bible"], scene, shot=shot, policy=policy),
               "scene": f"{scene} Speaking to camera: {script}",   # the video judge scores against it
               "keyframes": int(policy.get("review", "video", "keyframes", default=4)), "seed": seed, "params": params}
    _with_shot(payload, shot)
    _with_audio(payload, audio)
    if motion:
        payload["video_uri"] = motion["uri"]
    if segment is not None:
        payload["segment"] = {"index": segment, "of": len(body["talk_segments"])}
    key = f"post:{post['id']}:talk:{attempt}" if segment is None else f"post:{post['id']}:talk:{segment}:{attempt}"
    return queue.enqueue(conn, _cfg(policy).get("talk_kind", "talk.infinitetalk"), payload,
                         creator_id=post["creator_id"], batch=f"post-{post['id']}", post_id=post["id"],
                         idempotency_key=key)


def _talk_long_cfg(policy: Policy) -> dict:
    return policy.get("generation", "talk_long", default={}) or {}


def talk_segments(policy: Policy, script: str, *, hybrid: bool = False) -> list[str]:
    from .talk_segments import split_script
    cfg = _talk_long_cfg(policy)
    seconds = _hybrid_cfg(policy).get("segment_seconds", 10) if hybrid else cfg.get("max_segment_seconds", 25)
    return split_script(script, max_seconds=float(seconds), words_per_second=float(cfg.get("words_per_second", 2.5)))


def _hybrid_cfg(policy: Policy) -> dict:
    return policy.get("generation", "talk_hybrid", default={}) or {}


def hybrid_modes(segments: list[str], on_camera_share: float) -> list[str]:
    """Which segments are lip-synced on camera ('talk') and which are cutaways over the post's other photos
    ('narrated'), in the same voice. The hook and the sign-off are always on camera; middle segments join them,
    shortest first, until on-camera speech is at least `on_camera_share` of the words. Only the on-camera part
    needs the expensive lip-sync render."""
    n = len(segments)
    if n <= 2:
        return ["talk"] * n
    modes = ["talk"] + ["narrated"] * (n - 2) + ["talk"]
    words = [len(s.split()) for s in segments]
    total = sum(words) or 1
    for i in sorted(range(1, n - 1), key=lambda i: words[i]):
        if sum(w for w, m in zip(words, modes) if m == "talk") / total >= on_camera_share:
            break
        modes[i] = "talk"
    return modes


def shot_takes(policy: Policy, shots: list, *, cutaways: bool = True) -> list[tuple[str, str, int]]:
    """A directed talking video's takes, one per shot that speaks: (text, mode, shot index), in order. The text is
    what the shot's lines say (`direction.spoken`, deliveries never included); mode is 'talk' (lip-synced on camera)
    or 'narrated' for a `cutaway` shot (its photo under the same voice). Without `cutaways` (hybrid off, or one
    photo), a cutaway's lines join the on-camera take before it (or after it, for a leading cutaway); when only
    cutaways speak, their lines become one on-camera take on the first shot that isn't a cutaway. A take longer than generation.talk.max_take_seconds is split at sentence ends, as a safety
    net: the scene changes only where the shots do. [] when no shot speaks."""
    from .direction import ShotSheet, spoken
    from .talk_segments import split_script
    sheets = [ShotSheet.model_validate(s) if isinstance(s, dict) else s for s in shots]
    takes = [(spoken(s), s.cutaway, i) for i, s in enumerate(sheets)]
    takes = [t for t in takes if t[0]]
    if not takes:
        return []
    if all(cut for _, cut, _ in takes):
        # Only cutaways speak: a talking video needs her face, so the lines go to the first on-camera shot.
        anchor = next((i for i, s in enumerate(sheets) if not s.cutaway), None)
        takes = ([(" ".join(text for text, _, _ in takes), False, anchor)] if anchor is not None
                 else [(text, False, i) for text, _, i in takes])
    elif not cutaways:
        merged: list[tuple[str, bool, int]] = []
        lead = ""   # cutaway lines before the first on-camera take
        for text, cut, i in takes:
            if cut:
                if merged:
                    merged[-1] = (f"{merged[-1][0]} {text}", False, merged[-1][2])
                else:
                    lead = f"{lead} {text}".strip()
                continue
            merged.append((f"{lead} {text}".strip(), False, i))
            lead = ""
        takes = merged
    max_take = float(policy.get("generation", "talk", "max_take_seconds", default=60))
    wps = float(_talk_long_cfg(policy).get("words_per_second", 2.5))
    return [(part, "narrated" if cut else "talk", i) for text, cut, i in takes
            for part in split_script(text, max_seconds=max_take, words_per_second=wps)]


def _directed(policy: Policy, body: dict) -> bool:
    """Whether a talking video's segments follow its shots (generation.talk_hybrid.segment_by: scene)."""
    return bool(body.get("shots")) and _hybrid_cfg(policy).get("segment_by", "scene") == "scene"


def _take_shot(body: dict, segment: int | None) -> int:
    """The shot (and scene) a talk segment belongs to: from body.talk_shots for a directed post, else scene 0."""
    shots = body.get("talk_shots") or []
    k = segment or 0
    return int(shots[k]) if k < len(shots) else 0


def _photo_per_shot(body: dict, chosen: list[dict]) -> bool:
    """Whether every shot has its own chosen photo (chosen photos follow the scenes in order)."""
    n_shots = len(body.get("shots") or [])
    return bool(n_shots) and len(chosen) >= n_shots


def _take_photo(body: dict, chosen: list[dict], i: int) -> dict:
    """The chosen photo for shot i when the post has one per shot, else the first."""
    return chosen[i] if _photo_per_shot(body, chosen) and i < len(chosen) else chosen[0]


def _take_index(body: dict, segment: int | None) -> int:
    """The scene (and shot) a talk take is made from: its shot, or scene 0 when that is out of range."""
    i = _take_shot(body, segment)
    return i if i < len(body["scenes"]) else 0


def _take_script(body: dict, segment: int | None) -> str:
    return (body["talk_segments"][segment] if segment is not None
            else body.get("talk_script") or body.get("script", ""))


# Walk-and-talk (generation.talk_motion): a take whose shot moves is a silent motion clip first (the story-clip path,
# from the take's photo, prompted from the shot for image-to-video), then the talk job lip-syncs that clip. The clip is
# a Step of its own, made alongside the voice line; production does not wait for a review of it (done with a video is
# enough: the lip-synced take is judged; the job is marked payload.intermediate), and a rejected take is remade over
# the same clip.
def _motion_cfg(policy: Policy) -> dict:
    return policy.get("generation", "talk_motion", default={}) or {}


def _motion_pattern(words: str) -> str:
    """A trigger phrase as a whole-word pattern; its first word may also end in s/es/ed/ing."""
    first, *rest = words.lower().split()
    return r"\b" + re.escape(first) + r"(?:s|es|ed|ing)?" + "".join(r"\s+" + re.escape(w) for w in rest) + r"\b"


def moves(policy: Policy, shot: dict | None) -> bool:
    """Whether a shot has movement a talking photo can't give (generation.talk_motion.trigger): a camera move in
    camera_moves, or a beat whose action says one of beat_words. False when walk-and-talk is off or there's no shot."""
    cfg = _motion_cfg(policy)
    if shot is None or not cfg.get("enabled"):
        return False
    from .direction import ShotSheet, camera_move
    s = shot if isinstance(shot, ShotSheet) else ShotSheet.model_validate(shot)
    trigger = cfg.get("trigger") or {}
    move = camera_move(s, policy)
    if move and move in (trigger.get("camera_moves") or []):
        return True
    patterns = [_motion_pattern(str(w)) for w in trigger.get("beat_words") or [] if str(w).strip()]
    return any(re.search(p, b.action.lower()) for b in s.blocking for p in patterns)


def talk_v2v(policy: Policy) -> bool:
    """Whether the talk worker can lip-sync a video (walk-and-talk): only its ComfyUI engine can (talk_comfy; the
    research-script engine, talk_infinitetalk, animates a photo only). The worker image chooses the engine
    (MW_TALK_ENGINE, docker/worker-talkcomfy.Dockerfile), so this reads the talk kind's family's image in policy:
    worker_images.<family>.env.MW_TALK_ENGINE when set, else whether the image is one of
    generation.talk_motion.v2v_images."""
    family = policy.family(str(_cfg(policy).get("talk_kind", "talk.infinitetalk")))
    image = policy.get("worker_images", family, default={}) or {}
    engine = (image.get("env") or {}).get("MW_TALK_ENGINE")
    if engine is not None:
        return str(engine) == "comfy"
    return str(image.get("image") or "") in [str(x) for x in _motion_cfg(policy).get("v2v_images") or []]


def _needs_motion(policy: Policy, post: dict, body: dict, segment: int | None) -> bool:
    """Whether this talk take is walk-and-talk: a talking video on InfiniteTalk (the only V2V talk model) in the
    ComfyUI talk engine, whose take's shot moves."""
    if post["format"] != "talking_video":
        return False
    talk = policy.get("generation", "talk", default={}) or {}
    if str(talk.get("model") or "infinitetalk") != "infinitetalk" or not talk_v2v(policy):
        return False
    return moves(policy, _shot(body, _take_index(body, segment)))


def _wan_frames(seconds: float, fps: float) -> int:
    """At least `seconds` of frames, as Wan's 4k+1 frame counts."""
    n = max(1, math.ceil(seconds * fps))
    return 4 * math.ceil((n - 1) / 4) + 1


def _take_seconds(policy: Policy, body: dict, segment: int | None, audio: dict | None) -> float:
    """How long a take speaks: its finished voice line once voiced, else estimated from its words."""
    from .talk_segments import estimate_seconds
    return float(((audio or {}).get("meta") or {}).get("seconds") or estimate_seconds(
        unmarked(_take_script(body, segment)), float(_talk_long_cfg(policy).get("words_per_second", 2.5))))


def motion_lengths(policy: Policy, seconds: float) -> list[float]:
    """The motion clips a take of `seconds` gets, as their lengths: clips of max_motion_seconds (Wan's trained length)
    chained until the take is covered, at most max_motion_chain of them; a remainder under chain_min_seconds gets
    no clip of its own. Whatever is left ends on a hold (the talk job continues from the last frame)."""
    cfg = _motion_cfg(policy)
    longest = float(cfg.get("max_motion_seconds", 5))
    most = max(1, int(cfg.get("max_motion_chain", 1)))
    least = float(cfg.get("chain_min_seconds", 0))
    out, left = [], float(seconds)
    while left > 0 and len(out) < most and (not out or left >= least):
        out.append(min(longest, left))
        left -= out[-1]
    return out or [max(0.0, min(longest, float(seconds)))]


def _motion_job(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], attempt: int, *,
                segment: int | None = None, start_uri: str | None = None, audio: dict | None = None,
                clip: int = 0, seconds: float | None = None):
    """A silent motion clip of the take (clip `clip` of its chain): its photo (or the frame it continues from) moving
    as its shot's blocking and camera say, `seconds` long (default: the take's voice line, up to
    max_motion_seconds)."""
    cfg = _motion_cfg(policy)
    bible = conn.execute("SELECT bible FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["bible"]
    i = _take_index(body, segment)
    scene, shot = body["scenes"][i], _shot(body, i)
    if seconds is None:
        seconds = _take_seconds(policy, body, segment, audio)
    seconds = min(float(seconds), float(cfg.get("max_motion_seconds", 5)))
    vid = policy.get("generation", "video", default={})
    kind = cfg.get("kind") or _story_clip_kind(conn, policy, post)
    payload = {"image_uri": start_uri or _take_photo(body, chosen, i)["uri"],
               "prompt": clip_prompt(bible, scene, "", shot=shot, policy=policy),   # compile_prompt(shot, ..., "i2v")
               "scene": scene, "width": vid.get("width", 704), "height": vid.get("height", 1280),
               "keyframes": int(cfg.get("keyframes", 1)),
               "seed": _seed(post, 900 + (segment or 0) * 1000 + 3 * clip, attempt),
               "params": {"steps": vid.get("steps", 30), "guidance": vid.get("guidance", 5.0), "fps": vid.get("fps", 24)},
               "intermediate": True}   # walk-and-talk's source clip, not a post's output: the take is reviewed
    _clip_params(policy, kind, payload)
    frames = _wan_frames(seconds, float(payload["params"].get("fps") or vid.get("fps", 24)))
    payload["frames"] = frames
    if kind.startswith("comfy."):   # the ComfyUI graph reads its length from params
        payload["params"]["frames"] = frames
    _with_shot(payload, shot)
    if clip:
        payload["chain"] = {"index": clip}
    part = "" if segment is None else f"{segment}:"
    link = f"c{clip}:" if clip else ""
    return queue.enqueue(conn, kind, payload, creator_id=post["creator_id"], batch=f"post-{post['id']}",
                         post_id=post["id"], idempotency_key=f"post:{post['id']}:motion:{part}{link}{attempt}")


def _chain_jobs(body: dict, segment: int | None, clip: int) -> list[str]:
    """The job ids of motion clip `clip` (> 0) of a take's chain, kept in the body beside clip 0's."""
    if segment is None:
        chain = body.setdefault("motion_chain_jobs", [])
    else:
        chain = body.setdefault("motion_segment_chain_jobs", {}).setdefault(str(segment), [])
    while len(chain) < clip:
        chain.append([])
    return chain[clip - 1]


def _advance_motion(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], jobs: list[str], *,
                    segment: int | None = None, start_uri: str | None = None, audio: dict | None = None,
                    label: str = "", store=None) -> tuple[str, Any]:
    """('chosen', the take's motion video asset) once it is made: one motion clip, or a chain of them (motion_lengths),
    each from the last frame of the one before, joined into one. Each clip is retried on failure like any other step;
    a clip with no last frame to continue from ends the chain there (the take ends on a hold)."""
    failures = ("follow_up_failures", int(_cfg(policy).get("max_job_failures_per_post", 6)))
    videos: list[dict] = []
    for k, seconds in enumerate(motion_lengths(policy, _take_seconds(policy, body, segment, audio))):
        start = start_uri
        if k:
            try:
                start = _last_frame(conn, videos[-1])
            except DropPost:
                break
        step = Step(jobs if k == 0 else _chain_jobs(body, segment, k), lambda j: _edit_state(conn, j),
                    lambda attempt, k=k, s=start, secs=seconds: _motion_job(
                        conn, policy, post, body, chosen, attempt, segment=segment, start_uri=s, audio=audio,
                        clip=k, seconds=secs),
                    ("motion_repairs", 0), failures, "motion clip", label)
        state, video = advance_step(body, step)
        if state != "chosen":
            return state, None
        videos.append(video)
    if len(videos) == 1:
        return "chosen", videos[0]
    return "chosen", join_motion(conn, policy, post, videos, segment=segment, store=store)


def join_motion(conn, policy: Policy, post: dict, videos: list[dict], *, segment: int | None = None,
                store=None) -> dict:
    """A take's chained motion clips joined end to end into one silent clip, for the lip sync. Each clip after the
    first starts on the last frame of the one before (its start image), so that repeated frame is trimmed at the
    join (media.trim_start). Idempotent: the same clips reuse the joined asset."""
    import hashlib
    import tempfile
    from pathlib import Path

    from . import media, storage
    store = store or storage.from_url()
    tag = hashlib.sha256("|".join(str(v["id"]) for v in videos).encode()).hexdigest()[:12]
    key = f"assets/{post['creator_id']}/post-{post['id']}/motion-{'take' if segment is None else segment}-{tag}.mp4"
    existing = conn.execute("SELECT id, uri FROM assets WHERE uri = %s AND kind = 'video'",
                            (store.uri_for(key),)).fetchone()
    if existing:
        return existing
    with tempfile.TemporaryDirectory(prefix="mw-motion-") as d:
        timeout = float(_talk_long_cfg(policy).get("join_timeout_s", 600))
        paths = [media.trim_start(store.get_file(v["uri"], Path(d) / f"motion_{i:02d}.mp4"),
                                  Path(d) / f"motion_{i:02d}_cut.mp4", frames=1 if i else 0, timeout=timeout)
                 for i, v in enumerate(videos)]
        out = media.concat(paths, Path(d) / "motion.mp4", timeout=timeout)
        stored = store.put_file(out, key)
    return conn.execute(
        """INSERT INTO assets (creator_id, kind, uri, sha256, bytes, meta, recipe)
           VALUES (%s, 'video', %s, %s, %s, %s, %s) RETURNING id, uri""",
        (post["creator_id"], stored.uri, stored.sha256, stored.bytes,
         Jsonb({"intermediate": True, "motion_chain": [str(v["id"]) for v in videos]}),
         Jsonb({"joined": [v["uri"] for v in videos]}))).fetchone()


def _cutaway_job(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], attempt: int, *, segment: int,
                 audio: dict | None = None):
    """A cutaway segment of a hybrid talking video: the post's other photos, moving gently, under the creator's
    voice saying this part of the script (the narrated-edit worker; no lip-sync render). With `audio`, the
    designed voice's finished line instead of Kokoro."""
    anchors = conn.execute("SELECT anchors FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["anchors"]
    out = _hybrid_cfg(policy).get("output", {}) or {}
    if body.get("talk_shots") and _photo_per_shot(body, chosen):   # a directed cutaway: its own shot's photo
        images = [chosen[_take_shot(body, segment)]]
    else:
        images = chosen[1:] or chosen
    payload = {"images": [a["uri"] for a in images], "script": unmarked(body["talk_segments"][segment]),
               "voice": (anchors.get("voice") or {}).get("voice", "af_heart"), "on_screen_text": "",
               **(_cfg(policy).get("edit", {}) or {}), **{k: out[k] for k in ("width", "height", "fps") if k in out}}
    _with_audio(payload, audio, captions=True)
    if captions_burned(policy, "talking_video"):   # the joined video's captions are burned in at the finish stage
        payload["captions"] = False
    return queue.enqueue(conn, "image.narrated", payload, creator_id=post["creator_id"], batch=f"post-{post['id']}",
                         post_id=post["id"], idempotency_key=f"post:{post['id']}:cutaway:{segment}:{attempt}")


# The designed voice (creator/voice_lab.py, worker/handlers/tts_voxcpm.py): for a creator whose voice engine is
# voxcpm, each spoken piece is a voice.speak job first, a Step of its own (retried like any other on failure; nothing
# reviews it here, its takes are chosen by WER on the worker), and the piece's job takes its wav. A rejected talk
# clip is remade over the same voice line.
def _designed_voice(conn, post: dict) -> dict | None:
    """The creator's designed voice (anchors.voice), or None while they speak with Kokoro."""
    from . import voices
    anchors = conn.execute("SELECT anchors FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["anchors"]
    return (anchors or {}).get("voice") if voices.engine(anchors) == "voxcpm" else None


def _sentence_lines(text: str) -> list[dict]:
    from .talk_segments import sentences
    return [{"text": s} for s in sentences(text)]


def voice_lines(body: dict, segment: int | None = None) -> list[dict]:
    """What one take of a talking video says, line by line, for voice.speak: [{text, delivery, pause_after_s}].
    A directed take (body.talk_shots) speaks its shots' lines with their deliveries and pauses, cut to this take's
    words (a take split for length gets only its part; a line cut in two keeps its pause on the second part).
    Otherwise, or when the take's words don't follow the shots' lines, one line per sentence, no delivery.
    *Emphasis* asterisks are never spoken: they are taken out here (the finish stage reads them from the script)."""
    out = []
    for line in _marked_voice_lines(body, segment):
        text = " ".join(unmarked(line["text"]).split())
        if text:
            out.append({**line, "text": text})
    return out


def _marked_voice_lines(body: dict, segment: int | None = None) -> list[dict]:
    from itertools import groupby

    takes = body.get("talk_segments") or [body.get("talk_script") or body.get("script", "")]
    k = segment or 0
    text = takes[k] if k < len(takes) else ""
    if body.get("talk_shots") and body.get("shots"):
        from .direction import spoken_lines
        lines = [line for shot in body["shots"] for line in spoken_lines(shot)]
        stream = [(w, li, wi == len(line["text"].split()) - 1)
                  for li, line in enumerate(lines) for wi, w in enumerate(line["text"].split())]
        start, words = sum(len(t.split()) for t in takes[:k]), text.split()
        part = stream[start:start + len(words)]
        if words and [w for w, _, _ in part] == words:
            out = []
            for li, group in groupby(part, key=lambda x: x[1]):
                group = list(group)
                out.append({"text": " ".join(w for w, _, _ in group), "delivery": lines[li]["delivery"] or None,
                            "pause_after_s": lines[li]["pause_after_s"] if group[-1][2] else None})
            return out
    return _sentence_lines(text)


def _voice_state(conn, job_ids: list[str]) -> tuple[str, Any]:
    """A voice line is not reviewed here: done with an audio asset is chosen."""
    status = _status(conn, job_ids[-1])
    if status in ("queued", "running"):
        return "waiting", None
    audio = conn.execute("SELECT id, uri, meta FROM assets WHERE job_id = %s AND kind = 'audio' ORDER BY uri LIMIT 1",
                         (job_ids[-1],)).fetchone()
    return ("chosen", audio) if status == "done" and audio else ("failed", None)


def _voice_job(conn, policy: Policy, post: dict, voice: dict, lines: list[dict], attempt: int, *,
               segment: int | None = None):
    from .creator import voice_lab
    payload = voice_lab.speak_payload(policy, voice, lines, seed=_seed(post, 800 + (segment or 0) * 1000, attempt))
    key = f"post:{post['id']}:voice:{attempt}" if segment is None else f"post:{post['id']}:voice:{segment}:{attempt}"
    return queue.enqueue(conn, voice_lab.speak_kind(policy), payload, creator_id=post["creator_id"],
                         batch=f"post-{post['id']}", post_id=post["id"], idempotency_key=key)


def _advance_voice(conn, policy: Policy, post: dict, body: dict, voice: dict | None, jobs: list[str],
                   lines: list[dict], *, segment: int | None = None, label: str = "") -> tuple[str, Any]:
    """('chosen', audio asset) once the piece's voice line exists; ('chosen', None) when there is nothing to voice
    here (a Kokoro creator, or no words), so the piece is made as before."""
    if voice is None or not lines:
        return "chosen", None
    step = Step(jobs, lambda j: _voice_state(conn, j),
                lambda attempt: _voice_job(conn, policy, post, voice, lines, attempt, segment=segment),
                ("voice_repairs", 0), ("follow_up_failures", int(_cfg(policy).get("max_job_failures_per_post", 6))),
                "voice", label)
    return advance_step(body, step)


def _with_audio(payload: dict, audio: dict | None, *, captions: bool = False) -> dict:
    """The piece speaks the finished voice line (its wav) instead of TTS on its own worker; an edit also gets
    each line's timing, so its captions appear exactly while they are said."""
    if audio:
        payload["audio_uri"] = audio["uri"]
        if captions:
            payload["cues"] = [{k: line.get(k) for k in ("text", "start_s", "end_s")}
                               for line in (audio.get("meta") or {}).get("lines") or []]
    return payload


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
    timeout = float(_talk_long_cfg(policy).get("join_timeout_s", 600))
    with tempfile.TemporaryDirectory(prefix="mw-join-") as d:
        paths = [store.get_file(v["uri"], Path(d) / f"segment_{i:02d}.mp4") for i, v in enumerate(videos)]
        if (post.get("body") or {}).get("talk_modes") or _hybrid_cfg(policy).get("normalize_all"):
            out_cfg = _hybrid_cfg(policy).get("output", {}) or {}
            paths = [media.normalize(p, p.with_name(f"norm_{p.name}"), width=int(out_cfg.get("width", 720)),
                                     height=int(out_cfg.get("height", 1280)), fps=int(out_cfg.get("fps", 30)),
                                     lufs=float(out_cfg.get("lufs", -14)), timeout=timeout) for p in paths]
        seconds = [media.duration(p) for p in paths]   # where each segment starts in the joined video (captions)
        out = media.concat(paths, Path(d) / "joined.mp4", timeout=timeout)
        stored = store.put_file(out, key)
    return conn.execute(
        """INSERT INTO assets (creator_id, kind, uri, sha256, bytes, meta, recipe)
           VALUES (%s, 'video', %s, %s, %s, %s, %s) RETURNING id, uri""",
        (post["creator_id"], stored.uri, stored.sha256, stored.bytes,
         Jsonb({"lip_sync": True, "segments": [str(v["id"]) for v in videos], "segment_seconds": seconds}),
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
    modes = body.get("talk_modes") or ["talk"] * len(body["talk_segments"])
    voice = _designed_voice(conn, post)
    voice_jobs = body.setdefault("voice_segment_jobs", [[] for _ in body["talk_segments"]]) if voice else None
    for i in range(len(body["talk_segments"])):
        # The designed voice says this segment first (every segment's line at once, even while a chain waits).
        # A segment started before the voice was designed keeps going as it began.
        voiced = voice if voice and (voice_jobs[i] or not jobs[i]) else None
        vstate, audio = _advance_voice(conn, policy, post, body, voiced, voice_jobs[i] if voiced else [],
                                       voice_lines(body, i) if voiced else [], segment=i, label=f"segment {i}: ")
        if vstate == "dropped":
            return "dropped", None
        # Walk-and-talk: the take's motion clip is made alongside its voice line. A segment started without one
        # keeps going as it began.
        motion_jobs = body.get("motion_segment_jobs")
        wants_motion = modes[i] != "narrated" and (bool(motion_jobs and motion_jobs[i])
                                                    or (not jobs[i] and _needs_motion(policy, post, body, i)))
        if vstate != "chosen" and not wants_motion:
            results.append(("waiting", None))
            continue
        if modes[i] == "narrated":   # a cutaway over the other photos (hybrid): not a lip-sync render
            step = Step(jobs[i], lambda j: _edit_state(conn, j),
                        lambda attempt, i=i, a=audio: _cutaway_job(conn, policy, post, body, chosen, attempt,
                                                                   segment=i, audio=a),
                        (f"segment_repairs_{i}", repairs), ("follow_up_failures", failures), "edit", f"segment {i}: ")
            results.append(advance_step(body, step))
            if results[-1][0] == "dropped":
                return "dropped", None
            continue
        # A directed post chains only within one shot (a take split for length): a new shot is a new scene.
        follows = chain and i > 0 and modes[i - 1] == "talk" and _take_shot(body, i) == _take_shot(body, i - 1)
        if follows and results[-1][0] != "chosen":
            results.append(("waiting", None))   # starts from the previous segment's last frame
            continue
        start_uri = _last_frame(conn, results[-1][1]) if follows else None
        mstate, motion = "chosen", None
        if wants_motion:
            motion_jobs = body.setdefault("motion_segment_jobs", [[] for _ in body["talk_segments"]])
            mstate, motion = _advance_motion(conn, policy, post, body, chosen, motion_jobs[i], segment=i,
                                             start_uri=start_uri, audio=audio, label=f"segment {i}: ", store=store)
            if mstate == "dropped":
                return "dropped", None
        if vstate != "chosen" or mstate != "chosen":
            results.append(("waiting", None))   # the take needs its voice line and its motion clip
            continue
        step = Step(jobs[i], lambda j: _clip_state(conn, j),
                    lambda attempt, i=i, s=start_uri, a=audio, m=motion: _talk_job(
                        conn, policy, post, body, chosen, attempt, segment=i, start_uri=s, audio=a, motion=m),
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


def _follow_up(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], audio: dict | None = None,
               motion: dict | None = None) -> Step:
    key, make, state, thing = FOLLOW_UPS[post["format"]]
    voiced = {"audio": audio} if audio else {}
    if motion:   # walk-and-talk: only a talk job is made with a motion clip
        voiced["motion"] = motion
    return Step(body.setdefault(key, []), lambda j: state(conn, j),
                lambda attempt: make(conn, policy, post, body, chosen, attempt, **voiced),
                ("clip_repairs", int(policy.get("review", "video", "max_repairs_per_post", default=2))),
                ("follow_up_failures", int(_cfg(policy).get("max_job_failures_per_post", 6))), thing)


# Follow-ups that speak: voiced first by the designed voice, when the creator has one.
SPOKEN = ("narrated_edit", "talking_video")


def _advance_follow_up(conn, policy: Policy, post: dict, body: dict, chosen: list[dict],
                       store=None) -> tuple[str, Any]:
    audio, vstate = None, "chosen"
    started = bool(body.get(FOLLOW_UPS[post["format"]][0]))
    voice = _designed_voice(conn, post) if post["format"] in SPOKEN else None
    # A piece started before the voice was designed (or before walk-and-talk) keeps going as it began.
    if voice and (body.get("voice_jobs") or not started):
        vstate, audio = _advance_voice(conn, policy, post, body, voice, body.setdefault("voice_jobs", []),
                                       voice_lines(body))
        if vstate == "dropped":
            return vstate, None
    mstate, motion = "chosen", None   # walk-and-talk: the motion clip, made alongside the voice line
    if body.get("motion_jobs") or (not started and _needs_motion(policy, post, body, None)):
        mstate, motion = _advance_motion(conn, policy, post, body, chosen, body.setdefault("motion_jobs", []),
                                         store=store)
        if mstate == "dropped":
            return mstate, None
    if vstate != "chosen" or mstate != "chosen":
        return "waiting", None
    return advance_step(body, _follow_up(conn, policy, post, body, chosen, audio, motion))


# --- finish: one pass of grade, grain, loudness and burned-in captions over a made video ------------------------
# Once a video post's video exists (joined, for a long talking video), it is finished on the control plane before
# the post is ready: manyworlds.finishing (look, grain, two-pass loudness) and manyworlds.captions (the spoken words,
# timed from the voice lines). With libass in this ffmpeg the captions' ASS rides in finishing's own filter chain
# (after the grain, so text gets none): one re-encode. Without it, finishing then the caption overlay: two. A rerun
# reuses post-<id>/final.mp4. A finish that keeps failing (finishing.max_failures) ships the unfinished video and
# tells the owner.
def _switched_on(policy: Policy, block: str, fmt: str) -> bool:
    cfg = policy.get(block, default={}) or {}
    formats = cfg.get("formats")
    return bool(cfg.get("enabled")) and fmt in FOLLOW_UPS and (formats is None or fmt in formats)


def finishes(policy: Policy, fmt: str) -> bool:
    """Whether a video post of this format is graded, grained and loudness-normalised (finishing.enabled, .formats)."""
    return _switched_on(policy, "finishing", fmt)


def captions_burned(policy: Policy, fmt: str) -> bool:
    """Whether a video post of this format gets burned-in captions (burned_captions.enabled, .formats)."""
    return _switched_on(policy, "burned_captions", fmt)


def caption_lines(meta: dict) -> list[tuple[dict, int]]:
    """A voice line's timings as captions.estimate_timings' lines ({text, start_s, end_s}), from voice.speak's meta:
    one per clause when it has clauses (finer spans), else one per line; each with the number of filler words in
    front. A clause the voice began with a filler (meta.fillers, or the clause's own `filler`) gets the filler in
    front of its text: the clause's span holds the filler too, so the filler takes its share of the time, and those
    leading words are then dropped from the captions."""
    def timed(items) -> bool:
        return bool(items) and all(isinstance(x, dict) and x.get("text") and x.get("start_s") is not None
                                   and x.get("end_s") is not None for x in items)
    clauses, lines = meta.get("clauses") or [], meta.get("lines") or []
    if timed(clauses):
        said = {f.get("clause"): f.get("word") for f in meta.get("fillers") or [] if isinstance(f, dict)}
        out = []
        for i, c in enumerate(clauses):
            filler = said.get(c.get("index", i)) or c.get("filler")
            text = f"{filler}, {c['text']}" if filler else c["text"]
            out.append(({"text": text, "start_s": float(c["start_s"]), "end_s": float(c["end_s"])},
                        len(str(filler).split()) if filler else 0))
        return out
    if timed(lines):
        return [({"text": ln["text"], "start_s": float(ln["start_s"]), "end_s": float(ln["end_s"])}, 0)
                for ln in lines]
    return []


def _pieces(conn, video: dict) -> tuple[dict, list[dict]]:
    """The video's asset row and its pieces in order (a joined video's segments, else the video itself), each with
    its job (id, kind, payload)."""
    row = conn.execute("SELECT id, uri, meta, job_id FROM assets WHERE id = %s", (video["id"],)).fetchone()
    ids = (row["meta"] or {}).get("segments") or []
    rows = [conn.execute("SELECT id, uri, meta, job_id FROM assets WHERE id = %s", (i,)).fetchone() for i in ids] or [row]
    if any(r is None for r in rows):
        raise ValueError(f"a segment of {video['uri']} has no asset")
    return row, [{**r, "job": conn.execute("SELECT id, kind, payload FROM jobs WHERE id = %s", (r["job_id"],)).fetchone()
                  if r["job_id"] else None} for r in rows]


def _audio_meta(conn, *, uri: str | None = None, job_id=None) -> dict:
    row = (conn.execute("SELECT meta FROM assets WHERE uri = %s AND kind = 'audio' LIMIT 1", (uri,)).fetchone() if uri
           else conn.execute("SELECT meta FROM assets WHERE job_id = %s AND kind = 'audio' ORDER BY uri LIMIT 1",
                             (job_id,)).fetchone() if job_id else None)
    return (row["meta"] or {}) if row else {}


def _piece_timing(conn, piece: dict, script: str) -> tuple[list[tuple[dict, int]], float | None]:
    """A piece's caption lines in its own time, and its length. Its designed voice line's clauses or lines (a narrated
    piece's voice starts after the edit's lead); else a Kokoro edit's own cues; else the script over the spoken part
    (a Kokoro talk clip's own audio, or the edit between its lead and tail)."""
    from .worker.handlers.edit_narrated import LEAD_S, TAIL_S
    job = piece["job"] or {}
    payload, vmeta = job.get("payload") or {}, piece["meta"] or {}
    narrated = job.get("kind") == "image.narrated"
    lead = float(vmeta.get("lead_s", LEAD_S)) if narrated else 0.0
    tail = float(vmeta.get("tail_s", TAIL_S)) if narrated else 0.0
    voice = _audio_meta(conn, uri=payload["audio_uri"]) if payload.get("audio_uri") else {}
    seconds = vmeta.get("seconds")
    if seconds is None and voice.get("seconds") is not None:
        seconds = float(voice["seconds"]) + lead + tail
    seconds = float(seconds) if seconds is not None else None
    lines = caption_lines(voice) if voice else []
    if lines:
        return [({**ln, "start_s": ln["start_s"] + lead, "end_s": ln["end_s"] + lead}, f) for ln, f in lines], seconds
    cues = vmeta.get("cues") if narrated else None
    if cues and all(isinstance(c, dict) and c.get("text") and c.get("start_s") is not None for c in cues):
        return [({"text": c["text"], "start_s": float(c["start_s"]), "end_s": float(c["end_s"])}, 0)
                for c in cues], seconds
    spoken = None
    if not narrated and job.get("id"):
        spoken = _audio_meta(conn, job_id=job["id"]).get("seconds")
    if spoken is None and seconds is not None:
        spoken = seconds - lead - tail
    if not script.strip() or not spoken or float(spoken) <= 0:
        return [], seconds
    return [({"text": script, "start_s": lead, "end_s": lead + float(spoken)}, 0)], seconds


def caption_words(conn, policy: Policy, body: dict, video: dict) -> list | None:
    """The post's spoken words, timed in its finished video (captions.Word). Each piece's lines (_piece_timing) are
    placed at the piece's start in the joined video (the join's measured segment lengths, else each piece's own),
    fillers dropped, and the post's script (with *emphasis*) lines up the shown words. None when a piece's length
    is unknown, so later captions can't be placed."""
    from . import captions
    row, pieces = _pieces(conn, video)
    measured = (row["meta"] or {}).get("segment_seconds") or []
    segmented = len(body.get("talk_segments") or []) == len(pieces) > 1
    words, scripts, t = [], [], 0.0
    for i, piece in enumerate(pieces):
        script = _take_script(body, i if segmented else None)
        lines, seconds = _piece_timing(conn, piece, unmarked(script))
        for line, fillers in lines:
            words += captions.estimate_timings([line], offset=t, policy=policy)[fillers:]
        scripts.append(script)
        if i < len(pieces) - 1:
            length = measured[i] if i < len(measured) and measured[i] is not None else seconds
            if length is None:
                return None
            t += float(length)
    return captions._align_script(words, " ".join(scripts)) if words else []


def caption_placement(policy: Policy, shot: dict | None) -> str:
    """Where a video's captions go: the first entry of burned_captions.placement_by_shot_size whose words appear in
    the shot's shot size, else default_placement."""
    cfg = policy.get("burned_captions", default={}) or {}
    size = str(((shot or {}).get("framing") or {}).get("shot_size") or "").lower()
    for words, place in (cfg.get("placement_by_shot_size") or {}).items():
        if str(words).lower() in size:
            return str(place)
    return str(cfg.get("default_placement") or "lower")


def _caption_style(conn, creator_id) -> str | None:
    """The creator's caption style (profile dial caption_style), or None for burned_captions.default_style."""
    from .creator import profile
    current = profile.current(conn, creator_id)
    style = ((current[1] if current else {}).get("dials") or {}).get("caption_style")
    return str(style) if style else None


def caption_plan(conn, policy: Policy, post: dict, body: dict, video: dict, *, width: int, height: int):
    """The captions for a post's finished video of width x height, or None when nothing is said (or can't be timed)."""
    from . import captions
    words = caption_words(conn, policy, body, video)
    if not words:
        return None
    shot = _shot(body, _take_index(body, 0 if body.get("talk_segments") else None))
    return captions.make_plan(words, policy, width=width, height=height, style=_caption_style(conn, post["creator_id"]),
                              placement=caption_placement(policy, shot))


def _cover(conn, policy: Policy, store, body: dict, pieces: list[dict], workdir) -> dict | None:
    """The cover frame (finishing.pick_cover): among the pieces' keyframes, else the post's chosen photos."""
    from pathlib import Path

    from . import finishing
    job_ids = [str(p["job"]["id"]) for p in pieces if p["job"]]
    frames = conn.execute("SELECT id, uri FROM assets WHERE job_id = ANY(%s::uuid[]) AND kind = 'keyframe' ORDER BY uri",
                          (job_ids,)).fetchall() if job_ids else []
    frames = frames or [{"id": im.get("asset_id"), "uri": im["uri"]} for im in body.get("images") or []]
    if not frames:
        return None
    local = {}
    for i, f in enumerate(frames):
        local[store.get_file(f["uri"], Path(workdir) / f"cover_{i:03d}{Path(f['uri']).suffix or '.jpg'}")] = f
    best = local[finishing.pick_cover(list(local), policy=policy)]
    return {"asset_id": str(best["id"]) if best["id"] else None, "uri": best["uri"]}


def finish_video(conn, policy: Policy, post: dict, body: dict, video: dict, store=None) -> dict:
    """The post's finished video asset (or `video` itself when nothing applies to its format). Idempotent: the
    finished file is assets/<creator>/post-<id>/final.mp4, and a rerun reuses its asset."""
    import dataclasses
    import tempfile
    from pathlib import Path

    from . import captions, finishing, media, storage
    fmt = post["format"]
    grade, burn = finishes(policy, fmt), captions_burned(policy, fmt)
    if not (grade or burn):
        return video
    store = store or storage.from_url()
    key = f"assets/{post['creator_id']}/post-{post['id']}/final.mp4"
    existing = conn.execute("SELECT id, uri, meta FROM assets WHERE uri = %s AND kind = 'video'",
                            (store.uri_for(key),)).fetchone()
    if existing:
        return existing
    _, pieces = _pieces(conn, video)
    look = None
    with tempfile.TemporaryDirectory(prefix="mw-finish-") as tmp:
        d = Path(tmp)
        raw = store.get_file(video["uri"], d / "raw.mp4")
        plan = None
        if burn:
            info = media.probe(raw)
            if not (info["width"] and info["height"]):
                raise RuntimeError(f"can't read the frame size of {video['uri']}")
            # The finish stage upscales a smaller video to finishing.output first, so its captions plan at that size.
            width, height = (finishing.finished_size(info["width"], info["height"], policy) if grade
                             else (info["width"], info["height"]))
            plan = caption_plan(conn, policy, post, body, video, width=width, height=height)
        if not grade and plan is None:
            return video   # captions only, and nothing is said
        out, passes, cover = d / "final.mp4", 0, None
        if grade:
            handle = conn.execute("SELECT handle FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["handle"]
            shot = _shot(body, _take_index(body, 0 if body.get("talk_segments") else None)) or {}
            look = finishing.look_for(policy, creator=handle, set_id=(shot.get("set") or {}).get("set_id"))
            one_pass = plan is not None and media.has_libass()
            ass, fpolicy = None, policy
            if one_pass:   # the captions ride in finishing's filter chain, after the grain
                if plan.fonts_dir is None or not (plan.fonts_dir / str(plan.style.get("file", ""))).exists():
                    raise FileNotFoundError(f"caption font {plan.style.get('file')!r} not found ({plan.fonts_dir})")
                ass = captions.write_ass(plan, d / "captions.ass")
                fpolicy = dataclasses.replace(policy, body={
                    **policy.body, "finishing": {**finishing.cfg(policy), "fonts_dir": str(plan.fonts_dir)}})
            graded = finishing.finish(raw, out if plan is None or one_pass else d / "graded.mp4", look=look,
                                      hook_ass=ass, policy=fpolicy)
            passes += 1
            if plan is not None and not one_pass:   # no libass here: the captions' Pillow overlay, a second pass
                captions.burn(graded, plan, out, workdir=d / "captions", policy=policy)
                passes += 1
            try:
                cover = _cover(conn, policy, store, body, pieces, d)
            except Exception:  # noqa: BLE001 - a post without a picked cover still ships (the publisher has a fallback)
                cover = None
        else:
            captions.burn(raw, plan, out, workdir=d / "captions", policy=policy)
            passes += 1
        stored = store.put_file(out, key)
    meta = {"finished": grade, "look": look, "captions": plan.style_name if plan else None,
            "placement": plan.placement if plan else None, "passes": passes, "raw": str(video["id"]), "cover": cover}
    return conn.execute(
        """INSERT INTO assets (creator_id, kind, uri, sha256, bytes, meta, recipe)
           VALUES (%s, 'video', %s, %s, %s, %s, %s) RETURNING id, uri, meta""",
        (post["creator_id"], stored.uri, stored.sha256, stored.bytes, Jsonb(meta),
         Jsonb({"finished_from": video["uri"], "policy_version": policy.version}))).fetchone()


def _finish_failed(conn, policy: Policy, post: dict, body: dict, error: Exception) -> bool:
    """Count a failed finish. True once finishing.max_failures is used up: the post then ships its unfinished video
    and the owner hears of it; until then it waits for the next tick."""
    from . import notify
    body["finish_failures"] = int(body.get("finish_failures", 0)) + 1
    body["finish_error"] = f"{type(error).__name__}: {error}"[:500]
    if body["finish_failures"] < int(policy.get("finishing", "max_failures", default=3)):
        return False
    body["finish_skipped"] = body["finish_error"]
    notify.owner_event(conn, "finish_failed", subject={"post_id": str(post["id"])},
                       payload={"format": post["format"], "error": body["finish_error"],
                                "failures": body["finish_failures"]},
                       creator_id=post["creator_id"], dedupe_key=f"finish_failed:{post['id']}")
    return True


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
                hybrid = bool(_hybrid_cfg(policy).get("enabled")) and len(chosen) >= 2
                takes = shot_takes(policy, body["shots"], cutaways=hybrid) if _directed(policy, body) else []
                if takes:
                    # Directed: one take per shot that speaks, cut only where the shots change; cutaway shots are
                    # narrated over their own photo (hybrid), the rest lip-synced from theirs.
                    body["talk_shots"] = [i for _, _, i in takes]
                    if len(takes) > 1:
                        body["talk_segments"] = [text for text, _, _ in takes]
                        if any(mode == "narrated" for _, mode, _ in takes):
                            body["talk_modes"] = [mode for _, mode, _ in takes]
                    else:
                        body["talk_segments"], body["talk_script"] = [], takes[0][0]   # [] = one clip
                    post = {**post, "body": body}
                else:
                    # Hybrid (generation.talk_hybrid): with cutaway photos, short segments, only some on camera.
                    segments = talk_segments(policy, body.get("script", ""), hybrid=hybrid)
                    body["talk_segments"] = segments if len(segments) > 1 else []   # [] = one clip
                    if hybrid and len(segments) > 2:
                        body["talk_modes"] = hybrid_modes(segments,
                                                          float(_hybrid_cfg(policy).get("on_camera_share", 0.4)))
                        post = {**post, "body": body}
            if body.get("talk_segments"):   # a long script: one job per segment, then joined
                state, video = _advance_long_talk(conn, policy, post, body, chosen, store)
            else:
                state, video = _advance_follow_up(conn, policy, post, body, chosen, store)
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
        final = video
        try:
            final = finish_video(conn, policy, {**post, "body": body}, body, video, store)
        except Exception as e:  # noqa: BLE001 - ffmpeg or storage: tried again next tick, then shipped unfinished
            if not _finish_failed(conn, policy, post, body, e):
                _set(conn, post["id"], "producing", body)
                return "producing"
        if str(final["id"]) != str(video["id"]):
            meta = final.get("meta") or {}
            body["video_raw"] = {"asset_id": str(video["id"]), "uri": video["uri"]}
            body["finish"] = {k: meta.get(k) for k in ("finished", "look", "captions", "placement", "passes")}
            if meta.get("cover"):
                body["cover"] = meta["cover"]
        body["video"] = {"asset_id": str(final["id"]), "uri": final["uri"]}
        asset_ids = [final["id"], *asset_ids]
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
