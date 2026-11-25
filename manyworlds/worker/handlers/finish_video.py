"""finish.video: a made video finished (sound mix, grade, grain, loudness, burned-in captions and on-screen text, the
cover frame) on the finisher, off the producer's loop. No GPU: ffmpeg on a dedicated-CPU Fly machine of the control
plane's app (fly.toml process `finisher`, pool `fly_finisher`), which the autoscaler starts while finish jobs wait
(deploy/autoscale.ensure_finisher) and which stops itself when its queue is empty.

Payload: production.finish.finish_plan, prepared by the producer: the post and its body, the made video and the file
to finish, its pieces and keyframes, the creator's handle, the caption style and timed words, the output keys, and
the policy sections the finish reads (rendered under them, over this image's own policy). The handler reads no
database: it renders (production.finish.render_finish), uploads post-<id>/final.mp4 and cover.jpg (fixed keys, so a
rerun overwrites them) and returns what it made in the job's result (`finish`); the producer writes the asset rows
and the post's body (record_finish). No outputs go through the runner: they are the post's, not the job's.
"""

from __future__ import annotations

from typing import Any

from ...policy import Policy
from ..runner import HandlerResult, JobContext, NonRetryable

KIND = "finish.video"
NEEDS = ("post", "body", "video", "source_uri", "final_key", "cover_key")


def policy_for(payload: dict[str, Any], base: Policy | None = None) -> Policy:
    """This image's policy with the payload's finish sections over it: the producer's settings win."""
    from ... import policy as policy_mod
    base = base or policy_mod.load()
    sections = {k: v for k, v in (payload.get("policy") or {}).items() if v is not None}
    return policy_mod.from_dict({**base.body, **sections})


def finish(payload: dict[str, Any], ctx: JobContext, *, policy: Policy | None = None) -> HandlerResult:
    from ...production import finish as fin
    missing = [k for k in NEEDS if payload.get(k) in (None, "")]
    if missing:
        raise NonRetryable(f"payload needs {missing} (production.finish.finish_plan)")
    made = fin.render_finish(payload, policy_for(payload, policy), ctx.store, ctx.workdir / "finish")
    if made:
        made.pop("path", None)   # this machine's temporary file: the producer fetches the upload when it needs it
    return HandlerResult(outputs=[], recipe={"finished_from": payload["source_uri"],
                                             "policy_version": payload.get("policy_version")},
                         meta={"finish": made})


HANDLERS = {KIND: finish}


# finish.join: a long talking video's takes joined (normalised to one size and rate, transitions, upload) on the
# finisher instead of the producer's small VM (production.talk: join_plan prepares the payload, record_join writes the
# asset row from the result's `join`). Breadth run 2026-11-23: the join timed out on the producer every loop.
JOIN_KIND = "finish.join"
JOIN_NEEDS = ("key", "segments")


def join(payload: dict[str, Any], ctx: JobContext, *, policy: Policy | None = None) -> HandlerResult:
    from ...production import talk
    missing = [k for k in JOIN_NEEDS if not payload.get(k)]
    if missing:
        raise NonRetryable(f"payload needs {missing} (production.talk.join_plan)")
    made = talk.render_join(payload, policy_for(payload, policy), ctx.store, ctx.workdir / "join")
    return HandlerResult(outputs=[], recipe={"joined": [s["uri"] for s in payload["segments"]]},
                         meta={"join": made})


HANDLERS[JOIN_KIND] = join


# finish.measure_song: a song's anatomy measured (manyworlds.song_measure: librosa, pyloudnorm) on the finisher, not the
# producer's 512 MB VM (2026-11-24: a 2-minute 48 kHz stereo song was OOM-killed there). The payload is
# songs.measure_payload (no database here); the measure goes back in the job's result (`song_features`) and the
# producer's tick stores it (songs.collect_measurements).
MEASURE_KIND = "finish.measure_song"
MEASURE_NEEDS = ("song_id", "mix_uri")


def measure_song(payload: dict[str, Any], ctx: JobContext, *, policy: Policy | None = None) -> HandlerResult:
    from ... import songs
    missing = [k for k in MEASURE_NEEDS if not payload.get(k)]
    if missing:
        raise NonRetryable(f"payload needs {missing} (songs.measure_payload)")
    out = songs.measure_files(payload, policy_for(payload, policy), ctx.store, ctx.workdir / "measure")
    return HandlerResult(outputs=[], recipe={"measured": payload["mix_uri"], "policy_version": payload.get("policy_version")},
                         meta={"song_features": out})


HANDLERS[MEASURE_KIND] = measure_song


# finish.measure_video: a research reference's video measured (manyworlds.research.video_measure: cuts, shot lengths,
# pace, sampled frames) on the finisher. It reads and writes only the research store (RESEARCH_STORAGE_URL), which
# rented GPU machines never hold; the measure goes back in the job's result (`video_measure`) and the control plane
# stores it (video_measure.collect).
VIDEO_MEASURE_KIND = "finish.measure_video"
VIDEO_MEASURE_NEEDS = ("post_id", "media_uri")


def measure_video(payload: dict[str, Any], ctx: JobContext, *, policy: Policy | None = None) -> HandlerResult:
    from ...research import references, video_measure
    missing = [k for k in VIDEO_MEASURE_NEEDS if not payload.get(k)]
    if missing:
        raise NonRetryable(f"payload needs {missing} (research.video_measure.payload)")
    if not references.is_research_uri(payload["media_uri"]):
        raise NonRetryable("finish.measure_video measures research references only")
    store = references.research_store()
    if store is None:
        raise NonRetryable(f"{references.STORE_ENV} is not set on this machine")
    out = video_measure.measure_files(payload, store, ctx.workdir / "measure_video")
    return HandlerResult(outputs=[], recipe={"measured": payload["media_uri"], "version": out.get("version")},
                         meta={"video_measure": out})


HANDLERS[VIDEO_MEASURE_KIND] = measure_video
