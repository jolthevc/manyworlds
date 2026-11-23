"""Shared helpers of the production package: the production policy block, deterministic seeds, spoken text, a
scene's shot sheet on a job's payload, the post row's status write and a made video's pieces."""

from __future__ import annotations

import logging

from psycopg.types.json import Jsonb

from ..creator.lab import prompts
from ..policy import Policy

log = logging.getLogger("manyworlds.production")


def _cfg(policy: Policy) -> dict:
    return policy.get("production", default={})


def _seed(post: dict, offset: int, attempt: int) -> int:
    """A deterministic seed per post, piece and attempt: a rerun asks for the same output, a retry for a new one.
    Offsets keep pieces apart (scenes: i*1000, story clip: 500, talk: 700 + segment*1000, voice: 800 + segment*1000,
    motion clip: 900 + segment*1000 + 3*clip in its chain)."""
    return int(str(post["id"]).replace("-", "")[:7], 16) + offset + attempt * 10


def unmarked(text: str) -> str:
    """Text as spoken: without the *emphasis* asterisks (they only style the burned-in captions)."""
    return (text or "").replace("*", "")


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


def _with_crops(payload: dict, policy: Policy) -> dict:
    """A reviewed clip asks its worker for hand and mouth crops of its keyframes (review.detail.crops; worker/crops.py),
    which the video judge sees beside the keyframes. Workers without MediaPipe make none."""
    c = policy.get("review", "detail", "crops", default={}) or {}
    if c.get("enabled"):
        payload["crops"] = {k: c[k] for k in ("enabled", "hands", "mouth", "size", "margin") if k in c}
    return payload


def _with_prompts(payload: dict, target: str, bible: dict, scene: str, *, motion: str = "", shot: dict | None = None,
                  policy: Policy) -> dict:
    """The job's `prompt`, and `negative_prompt` when the direction compiler wrote one (only for a target whose model
    has a negative channel, creator.lab.prompts); none leaves the worker's own default negative."""
    p = prompts(target, bible, scene, motion=motion, shot=shot, policy=policy)
    payload["prompt"] = p["prompt"]
    if p.get("negative"):
        payload["negative_prompt"] = p["negative"]
    return payload


def _set(conn, post_id, status: str, body: dict, **cols) -> None:
    sets = ", ".join(f"{k} = %({k})s" for k in cols)
    conn.execute(f"UPDATE posts SET status = %(status)s, body = %(body)s, updated_at = now()"
                 f"{', ' + sets if sets else ''} WHERE id = %(id)s",
                 {"status": status, "body": Jsonb(body), "id": post_id, **cols})


# Formats made from the post's chosen photos (the keys of followups.FOLLOW_UPS, named here so the scene and finish
# modules need not import the job makers).
FOLLOW_UP_FORMATS = ("narrated_edit", "story_clip", "talking_video")


# Follow-ups that speak: voiced first by the designed voice, when the creator has one.
SPOKEN = ("narrated_edit", "talking_video")


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
