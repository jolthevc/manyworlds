"""`brain.music_pipeline`: one song through the department (manyworlds.music_pipeline) for a real run, with real
takes (QueueTakes: music.song jobs on the GPU). Payload {creator_id, brief, key} to start, or {draft} to go on. While
the takes are being made the job ends with state 'waiting' and enqueues its own continuation (the draft in the
payload, so no role is asked again), not before music.pipeline.poll_seconds. Nothing enqueues it while music is off
(music.enabled); the sandbox runner (manyworlds.music_sandbox) runs the same pipeline in-process with stubbed takes."""

from __future__ import annotations

from ... import music, music_pipeline, queue
from ...worker.runner import NonRetryable
from ..agent import AgentContext, register

NAME = "music_pipeline"


class MusicPipeline:
    name = NAME

    def run(self, ctx: AgentContext) -> dict:
        p = ctx.job.payload or {}
        if not music.enabled(ctx.policy) and not p.get("test"):
            return {"skipped": "music.enabled is off"}
        if p.get("draft"):
            out = music_pipeline.run_song(ctx, draft=p["draft"])
        elif p.get("creator_id"):
            out = music_pipeline.run_song(ctx, p["creator_id"], p.get("brief") or {}, key=p.get("key") or str(ctx.job.id),
                                          experiment=p.get("experiment"))
        else:
            raise NonRetryable("music_pipeline payload needs creator_id (and a brief) or a draft")
        if out["state"] == "waiting":
            n = int(p.get("wait", 0)) + 1
            if n > int(music_pipeline.cfg(ctx.policy).get("max_waits", 200)):
                raise NonRetryable("the takes never finished (music.pipeline.max_waits)")
            nxt = queue.enqueue(ctx.conn, f"brain.{NAME}", {"draft": music_pipeline.jsonable(out["draft"]), "wait": n,
                                                            "test": p.get("test")},
                                creator_id=ctx.job.creator_id, parent_job_id=ctx.job.id,
                                idempotency_key=f"{out['draft']['key']}:wait:{n}")
            ctx.conn.execute("UPDATE jobs SET run_after = now() + make_interval(secs => %s) WHERE id = %s",
                             (float(music_pipeline.cfg(ctx.policy).get("poll_seconds", 120)), nxt))
        return {k: v for k, v in out.items() if k != "draft"}


register(MusicPipeline())
