"""Moving pictures: story clips (Wan image-to-video), clip edits (several silent clips joined), dance and motion
transfer (Wan-Animate-2) and walk-and-talk's silent motion clips (chained from each last frame, joined)."""

from __future__ import annotations

import math
import re
from typing import Any

from psycopg.types.json import Jsonb

from .. import queue
from ..policy import Policy
from .common import _cfg, _seed, _shot, _with_crops, _with_prompts, _with_shot, unmarked
from .steps import DropPost, Step, _clip_state, _edit_state, advance_step
from .takes import _take_index, _take_photo, _take_script, _talk_long_cfg


def _clip_job(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], attempt: int, *, index: int = 0):
    """A story clip from one approved photo (scene `index`; a clip edit has one per scene): the scene, moving as
    the planner wrote (the script's motion is the first clip's; later clips move as their shots say)."""
    bible = conn.execute("SELECT bible FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["bible"]
    vid, scene = policy.get("generation", "video", default={}), body["scenes"][index]
    motion = body.get("script", "") if index == 0 else ""
    shot = _shot(body, index)
    image = chosen[index]
    key = f"post:{post['id']}:clip:{attempt}" if index == 0 else f"post:{post['id']}:clip{index}:{attempt}"
    keyframes = int(policy.get("review", "video", "keyframes", default=4))
    reference = dance_reference(policy, shot)
    if reference:   # the shot's motion comes from a licensed reference clip (Wan-Animate-2), reviewed like any clip
        payload = dance_payload(policy, bible, scene, shot, image_uri=image["uri"], reference=reference,
                                seed=_seed(post, 500 + 50 * index, attempt))
        payload["scene"] = f"{scene} Movement: {motion}" if motion else scene
        _with_crops(payload, policy)
        return queue.enqueue(conn, _dance_kind(policy), payload, creator_id=post["creator_id"],
                             batch=f"post-{post['id']}", post_id=post["id"], idempotency_key=key)
    payload = {"image_uri": image["uri"],
               "scene": f"{scene} Movement: {motion}" if motion else scene,   # the video judge scores against it
               "width": vid.get("width", 704), "height": vid.get("height", 1280), "frames": vid.get("frames", 121),
               "keyframes": keyframes,
               "seed": _seed(post, 500 + 50 * index, attempt),
               "params": {"steps": vid.get("steps", 30), "guidance": vid.get("guidance", 5.0), "fps": vid.get("fps", 24)}}
    _with_prompts(payload, "i2v", bible, scene, motion=motion, shot=shot, policy=policy)
    _with_shot(payload, shot)
    _with_crops(payload, policy)
    kind = _story_clip_kind(conn, policy, post)
    _clip_params(policy, kind, payload)
    return queue.enqueue(conn, kind, payload, creator_id=post["creator_id"],
                         batch=f"post-{post['id']}", post_id=post["id"], idempotency_key=key)


# Clip edits (generation.clip_edit): a silent story clip of several shots, one clip per scene photo, joined with
# each shot's transition in. Outfit checks, recipe steps, a walk through a city: the faceless and visual shapes
# that need no voice. Each clip is its own job, reviewed by the video gate and repaired on its own.
def clip_edit(policy: Policy, post: dict, chosen: list[dict]) -> bool:
    cfg = policy.get("generation", "clip_edit", default={}) or {}
    return post["format"] == "story_clip" and bool(cfg.get("enabled")) and len(chosen) > 1


def _advance_clip_edit(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], store=None):
    cap = int(policy.get("generation", "clip_edit", "max_clips", default=4))
    n = min(len(chosen), cap)
    lists = body.setdefault("clip_edit_jobs", [[] for _ in range(n)])
    repairs = int(policy.get("review", "video", "max_repairs_per_post", default=2))
    failures = int(_cfg(policy).get("max_job_failures_per_post", 6))
    results = []
    for i in range(n):
        step = Step(lists[i], lambda j: _clip_state(conn, j),
                    lambda attempt, i=i: _clip_job(conn, policy, post, body, chosen, attempt, index=i),
                    (f"clip_repairs_{i}", repairs), (f"clip_failures_{i}", failures), "clip", f"clip {i}: ")
        results.append(advance_step(body, step))
    states = {s for s, _ in results}
    if "dropped" in states:
        return "dropped", None
    if "waiting" in states:
        return "waiting", None
    return "chosen", join_clips(conn, policy, post, [v for _, v in results], store, body=body)


def join_clips(conn, policy: Policy, post: dict, videos: list[dict], store=None, *, body: dict) -> dict:
    """A clip edit's clips joined, each with its shot's transition in. Idempotent: a rerun reuses the asset."""
    import tempfile
    from pathlib import Path

    from .. import media, storage
    from ..direction import transition_kind
    store = store or storage.from_url()
    key = f"assets/{post['creator_id']}/post-{post['id']}/clips-joined.mp4"
    existing = conn.execute("SELECT id, uri FROM assets WHERE uri = %s AND kind = 'video'", (store.uri_for(key),)).fetchone()
    if existing:
        return existing
    texts = [str((((_shot(body, i) or {}).get("edit") or {}).get("transition_in")) or "").strip()
             for i in range(1, len(videos))]
    transitions = [transition_kind(t, policy) if t else "cut" for t in texts]
    transitions = transitions if any(t != "cut" for t in transitions) else None
    with tempfile.TemporaryDirectory(prefix="mw-clips-") as d:
        paths = [store.get_file(v["uri"], Path(d) / f"clip_{i:02d}.mp4") for i, v in enumerate(videos)]
        seconds = [media.duration(p) for p in paths]
        out = media.concat(paths, Path(d) / "joined.mp4", transitions=transitions, policy=policy)
        stored = store.put_file(out, key)
    return conn.execute(
        """INSERT INTO assets (creator_id, kind, uri, sha256, bytes, meta, recipe)
           VALUES (%s, 'video', %s, %s, %s, %s, %s) RETURNING id, uri""",
        (post["creator_id"], stored.uri, stored.sha256, stored.bytes,
         Jsonb({"segments": [str(v["id"]) for v in videos], "segment_seconds": seconds}),
         Jsonb({"joined": [v["uri"] for v in videos], "transitions": transitions}))).fetchone()


# Dance and motion transfer (policy `dance`; worker/handlers/dance_animate.py): a shot sheet's `motion_reference` names
# a driving clip in dance.references; while dance.enabled, that shot is rendered by Wan-Animate-2 from the shot's photo
# (the worker checks the reference again: ours or licensed, never a platform download).
def _dance_kind(policy: Policy) -> str:
    return str(_cfg(policy).get("dance_kind", "comfy.wan_animate2"))


def dance_reference(policy: Policy, shot: dict | None) -> str | None:
    """The shot's motion reference (a listed dance.references id, direction.motion_reference) when dance is on, else
    None (the shot renders as before)."""
    from .. import dance
    from ..direction import motion_reference
    if shot is None or not dance.enabled(policy) or not str(shot.get("motion_reference") or "").strip():
        return None
    return motion_reference(shot, policy) or None


def dance_payload(policy: Policy, bible: dict, scene: str, shot: dict | None, *, image_uri: str, reference: str,
                  seed: int) -> dict:
    """A comfy.wan_animate2 job's payload (the handler's contract, as `mw bench` sends it): the photo, the reference,
    the shot's image-to-video prompt and the policy's dance.generation settings, with dance.identity.keyframes
    keyframes (the video gate's dance identity row scores each). The animate graph reads its negative text from
    params (worker/comfy_animate.py), so the compiled negative goes there too."""
    from .. import dance
    keyframes = int((dance.cfg(policy).get("identity") or {}).get(
        "keyframes", policy.get("review", "video", "keyframes", default=4)))
    payload = {"image_uri": image_uri, "reference": reference, "seed": seed, "keyframes": keyframes,
               "params": dict(dance.cfg(policy).get("generation") or {}),
               "reference_entry": dance.reference(reference, policy)}   # a worker's baked policy may predate it
    _with_prompts(payload, "i2v", bible, scene, shot=shot, policy=policy)
    if payload.get("negative_prompt"):
        payload["params"]["negative_prompt"] = payload["negative_prompt"]
    return _with_shot(payload, shot)


def _story_clip_kind(conn, policy: Policy, post: dict) -> str:
    kind = _cfg(policy).get("video_kind", "comfy.wan_i2v")
    if (_cfg(policy).get("routing") or {}).get("story_clip"):   # the cheapest per accepted clip (manyworlds.routing)
        from .. import routing
        kind, _ = routing.choose(conn, policy, "story_clip", key=str(post["id"]))
    return kind


def _clip_params(policy: Policy, kind: str, payload: dict) -> dict:
    """A clip job's render settings for its kind (a video.* job keeps the payload's own)."""
    if kind.startswith("hosted."):   # a hosted API renders it (generation.hosted_video): no GPU machine
        payload["params"] = dict(policy.get("generation", "hosted_video", default={}) or {})
    elif kind.startswith("comfy."):  # the ComfyUI worker's Wan 2.2 graph (generation.comfy_video)
        payload["params"] = dict(policy.get("generation", "comfy_video", default={}) or {})
    return payload


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
    from ..direction import ShotSheet, camera_move
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
    ComfyUI talk engine, whose take's shot moves, or has a motion reference while dance is on (its animate clip is
    then the motion clip)."""
    if post["format"] != "talking_video":
        return False
    shot = _shot(body, _take_index(body, segment))
    # The take's model is what its talk job will run: generation.talk merged with the shot's talk profile, as
    # talk._talk_job builds it (a LongCat profile sets model: longcat, which has no V2V).
    from ..direction import talk_profile_for
    talk = dict(policy.get("generation", "talk", default={}) or {})
    profile = talk_profile_for(shot, policy)
    if profile:
        talk.update(policy.get("generation", "talk_profiles", profile, default={}) or {})
    if str(talk.get("model") or "infinitetalk") != "infinitetalk" or not talk_v2v(policy):
        return False
    return bool(dance_reference(policy, shot)) or moves(policy, shot)


def _wan_frames(seconds: float, fps: float) -> int:
    """At least `seconds` of frames, as Wan's 4k+1 frame counts."""
    n = max(1, math.ceil(seconds * fps))
    return 4 * math.ceil((n - 1) / 4) + 1


def _take_seconds(policy: Policy, body: dict, segment: int | None, audio: dict | None) -> float:
    """How long a take speaks: its finished voice line once voiced, else estimated from its words."""
    from ..talk_segments import estimate_seconds
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
    part = "" if segment is None else f"{segment}:"
    link = f"c{clip}:" if clip else ""
    key = f"post:{post['id']}:motion:{part}{link}{attempt}"
    seed = _seed(post, 900 + (segment or 0) * 1000 + 3 * clip, attempt)
    reference = dance_reference(policy, shot)
    if reference:   # the take's motion is the reference's (Wan-Animate-2), as long as the reference clip
        payload = dance_payload(policy, bible, scene, shot, image_uri=start_uri or _take_photo(body, chosen, i)["uri"],
                                reference=reference, seed=seed)
        payload.update(scene=scene, intermediate=True)
        return queue.enqueue(conn, _dance_kind(policy), payload, creator_id=post["creator_id"],
                             batch=f"post-{post['id']}", post_id=post["id"], idempotency_key=key)
    if seconds is None:
        seconds = _take_seconds(policy, body, segment, audio)
    seconds = min(float(seconds), float(cfg.get("max_motion_seconds", 5)))
    vid = policy.get("generation", "video", default={})
    kind = cfg.get("kind") or _story_clip_kind(conn, policy, post)
    payload = {"image_uri": start_uri or _take_photo(body, chosen, i)["uri"],
               "scene": scene, "width": vid.get("width", 704), "height": vid.get("height", 1280),
               "keyframes": int(cfg.get("keyframes", 1)),
               "seed": seed,
               "params": {"steps": vid.get("steps", 30), "guidance": vid.get("guidance", 5.0), "fps": vid.get("fps", 24)},
               "intermediate": True}   # walk-and-talk's source clip, not a post's output: the take is reviewed
    _with_prompts(payload, "i2v", bible, scene, shot=shot, policy=policy)
    _clip_params(policy, kind, payload)
    frames = _wan_frames(seconds, float(payload["params"].get("fps") or vid.get("fps", 24)))
    payload["frames"] = frames
    if kind.startswith("comfy."):   # the ComfyUI graph reads its length from params
        payload["params"]["frames"] = frames
    _with_shot(payload, shot)
    if clip:
        payload["chain"] = {"index": clip}
    return queue.enqueue(conn, kind, payload, creator_id=post["creator_id"], batch=f"post-{post['id']}",
                         post_id=post["id"], idempotency_key=key)


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
    lengths = motion_lengths(policy, _take_seconds(policy, body, segment, audio))
    if dance_reference(policy, _shot(body, _take_index(body, segment))):
        lengths = lengths[:1]   # one animate clip, as long as its reference: no chain (the talk holds past its end)
    for k, seconds in enumerate(lengths):
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

    from .. import media, storage
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


def _last_frame(conn, video: dict) -> str:
    """The last keyframe of a chosen segment (keyframes always end on the clip's last frame)."""
    row = conn.execute(
        """SELECT k.uri FROM assets v JOIN assets k ON k.job_id = v.job_id AND k.kind = 'keyframe'
           WHERE v.id = %s ORDER BY (k.meta->>'frame')::int DESC NULLS LAST, k.uri DESC LIMIT 1""",
        (video["id"],)).fetchone()
    if not row:
        raise DropPost("a chosen segment has no keyframes to continue from")
    return row["uri"]
