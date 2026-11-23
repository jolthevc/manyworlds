"""Follow-ups: the one video made from a post's chosen photos (narrated edit, story clip or talking video), voiced
first by the designed voice and, for walk-and-talk, over its motion clip."""

from __future__ import annotations

from typing import Any

from .. import queue
from ..policy import Policy
from .common import SPOKEN, _cfg, unmarked
from .finish import captions_burned
from .motion import _advance_motion, _clip_job, _needs_motion
from .steps import Step, _clip_state, _edit_state, advance_step
from .talk import _talk_job
from .voice import _advance_voice, _designed_voice, _with_audio, voice_lines


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


# Formats made from the post's chosen photos: body key for their jobs, job maker, state, name.
FOLLOW_UPS = {"narrated_edit": ("edit_jobs", _edit_job, _edit_state, "edit"),
              "story_clip": ("clip_jobs", _clip_job, _clip_state, "clip"),
              "talking_video": ("talk_jobs", _talk_job, _clip_state, "clip")}


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
