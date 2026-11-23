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
