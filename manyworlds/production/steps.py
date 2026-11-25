"""Steps: each piece of a post (a scene's image, then the edit, clip, voice line or upscale made from them) is a
Step, retried and repaired by advance_step; and the state readers that say whether a Step's latest job is waiting,
chosen, rejected or failed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class DropPost(Exception):
    """Raised by a step's job maker when the post must not be retried (e.g. unsafe output)."""


def _best_confirmed(conn, job_ids: list[str]) -> dict | None:
    """The best image of these jobs that passed every review row and was confirmed, or cleared the screen by the
    clear-pass margin (no confirm needed: review.image.clear_pass_margin)."""
    return conn.execute(
        """SELECT a.id, a.uri,
                  coalesce(sum(r.score) FILTER (WHERE r.details->>'stage' = 'confirm'),
                           sum(r.score) FILTER (WHERE r.details->>'clear_pass' = 'true')) AS score
           FROM assets a JOIN reviews r ON r.asset_id = a.id
           WHERE a.job_id = ANY(%s::uuid[]) AND a.kind = 'image'
           GROUP BY a.id, a.uri
           HAVING bool_and(r.passed)
              AND bool_or(r.details->>'stage' = 'confirm' OR r.details->>'clear_pass' = 'true')
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
    hold: Callable[[], bool] | None = None             # true while the first job must wait (a shared set's leader)


def advance_step(body: dict, step: Step) -> tuple[str, Any]:
    """('chosen', result), ('waiting', None) or ('dropped', None), starting a retry when needed."""
    if not step.jobs:
        if step.hold is not None and step.hold():
            return "waiting", None
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
