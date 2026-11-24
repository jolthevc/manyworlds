"""The song piece of a music_video post (policy `music`, OFF; production.formats has no music_video): the song is made
first (music.song: the mix, a vocal stem and an instrumental stem), then one sung take is lip-synced to the vocal stem
(production.followups, the talk job with the stem as its audio), and the finish lays the full mix under the video in
place of the stem (production.mixing). The song's spec is the brief's `song` (brain/lyrics.py wrote it).

A song job is a Step of its own: retried with a new seed when its machine fails; while music.check.enforce, a song
whose objective check failed (lyric WER, length, loudness) is remade with a new seed, up to
music.video.max_repairs_per_post."""

from __future__ import annotations

import logging
from typing import Any

from .. import music, queue
from ..policy import Policy
from .common import _cfg, _seed
from .steps import DropPost, Step, _status, advance_step

log = logging.getLogger("manyworlds.production")


def _song_state(conn, job_ids: list[str], *, enforce: bool) -> tuple[str, Any]:
    """chosen ({role: {asset_id, uri, meta}} with a mix and, for a sung song, a vocal stem), waiting, rejected (its
    check failed while enforced) or failed."""
    status = _status(conn, job_ids[-1])
    if status in ("queued", "running"):
        return "waiting", None
    if status != "done":
        return "failed", None
    rows = conn.execute("SELECT id, uri, meta FROM assets WHERE job_id = %s AND kind = 'audio' ORDER BY uri",
                        (job_ids[-1],)).fetchall()
    media = {str((r["meta"] or {}).get("role")): {"asset_id": str(r["id"]), "uri": r["uri"], "meta": r["meta"] or {}}
             for r in rows if (r["meta"] or {}).get("role") in music.ROLES}
    if "mix" not in media:
        return "failed", None
    if enforce and (media["mix"]["meta"].get("check") or {}).get("passed") is False:
        return "rejected", None
    return "chosen", media


def _song_job(conn, policy: Policy, post: dict, body: dict, attempt: int):
    spec = body.get("song") or {}
    try:
        payload = music.song_payload(policy, spec, seed=_seed(post, 600, attempt), check=True,
                                     max_seconds=float(policy.get("music", "video", "max_seconds", default=30)))
    except music.SongRefused as e:   # the brief can't be made: no retry helps
        raise DropPost(f"the song can't be made: {e}") from None
    return queue.enqueue(conn, music.kind(policy), payload, creator_id=post["creator_id"], batch=f"post-{post['id']}",
                         post_id=post["id"], idempotency_key=f"post:{post['id']}:song:{attempt}")


def advance_song(conn, policy: Policy, post: dict, body: dict) -> tuple[str, Any]:
    """('chosen', the song's media by role), ('waiting', None) or ('dropped', None). The chosen media are kept in
    body.song_media ({role: {asset_id, uri, seconds}}) for the talk job and the finish."""
    if not body.get("song"):
        body["drop_reason"] = "a music_video post without a song (body.song)"
        return "dropped", None
    enforce = bool(policy.get("music", "check", "enforce", default=False))
    step = Step(body.setdefault("song_jobs", []), lambda j: _song_state(conn, j, enforce=enforce),
                lambda attempt: _song_job(conn, policy, post, body, attempt),
                ("song_repairs", int(policy.get("music", "video", "max_repairs_per_post", default=2))),
                ("follow_up_failures", int(_cfg(policy).get("max_job_failures_per_post", 6))), "song")
    try:
        state, media = advance_step(body, step)
    except DropPost as e:   # the first job's spec was refused (later attempts are caught by advance_step)
        body["drop_reason"] = str(e)
        return "dropped", None
    if state == "chosen":
        body["song_media"] = {role: {"asset_id": m["asset_id"], "uri": m["uri"], "seconds": m["meta"].get("seconds")}
                              for role, m in media.items()}
        if not body.get("song_id"):
            _register(conn, post, body)
    return state, media


def _register(conn, post: dict, body: dict) -> None:
    """The chosen song as a songs row, the post linked to it (manyworlds.songs.for_post); measured later on the
    control plane (songs.measure_pending). Bookkeeping: a failure here never holds the post up."""
    from .. import songs
    try:
        with conn.transaction():
            songs.for_post(conn, post, body, body["song_jobs"][-1])
    except Exception:  # noqa: BLE001 - the song row is evidence, not a step of the post
        log.exception("registering the song of post %s failed", post.get("id"))


def sung_text(body: dict) -> str:
    """The words the take sings (the judge's script): the song's sung words."""
    return music.sung_words((body.get("song") or {}).get("lyrics") or "")
