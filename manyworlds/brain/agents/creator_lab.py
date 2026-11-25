"""The creator lab's Claude steps for a launch (manyworlds.brain.launches): bible, anchors, profile; and casting's
(manyworlds.brain.casting): casting (concepts, judge, shortlist), audition (a shortlisted concept's bible, sandbox
creator and reference pack), audition_anchors (its anchors and mini-feed posts).

Each `brain.creator_lab` job does one step of one launch and records it as a decision, so a rerun after a
crash does not create a second creator or ask Claude again. The GPU steps (reference pack, voice sample,
LoRA) are ordinary jobs the launch advancer queues itself.
"""

from __future__ import annotations

import tempfile

from ...creator import lab, profile
from ...worker.runner import NonRetryable
from ..agent import AgentContext, register, remember


class _Lab:
    def bible(self, conn, policy, llm, thesis: dict):
        bible = lab.write_bible(conn, policy, llm, thesis)
        return lab.create_creator(conn, bible, thesis["id"], policy_version=policy.version,
                                  voice_blend=policy.get("creator_lab", "voice_blend"))["id"]

    def anchors(self, conn, policy, llm, creator_id):
        from ... import storage
        with tempfile.TemporaryDirectory() as tmp:
            return lab.select_anchors(conn, policy, llm, storage.from_url(), creator_id, tmp)

    def profile(self, conn, policy, llm, creator_id):
        out = profile.build(conn, policy, llm, creator_id)
        from .. import casting
        casting.seed_profile(conn, policy, creator_id)   # a cast creator keeps the brand core her audition proved
        return out


LAB = _Lab()


class CreatorLab:
    name = "creator_lab"

    def run(self, ctx: AgentContext) -> dict:
        p = ctx.job.payload or {}
        if p.get("step") in CASTING_STEPS:
            return self.casting(ctx, p)
        launch = ctx.conn.execute("SELECT * FROM launches WHERE id::text = %s", (p.get("launch_id"),)).fetchone()
        if not launch or p.get("step") not in ("bible", "anchors", "profile"):
            raise NonRetryable("creator_lab payload needs a launch_id and a step (bible, anchors, profile)")
        step = p["step"]

        def compute() -> dict:
            if step == "bible":
                t = ctx.conn.execute("SELECT id, body FROM theses WHERE id = %s", (launch["thesis_id"],)).fetchone()
                if not t:
                    raise NonRetryable(f"launch {launch['id']} has no thesis")
                if (row := ctx.conn.execute("SELECT creator_id FROM theses WHERE id = %s", (t["id"],)).fetchone()) \
                        and row["creator_id"]:
                    cid = row["creator_id"]   # created before a crash
                else:
                    cid = LAB.bible(ctx.conn, ctx.policy, ctx.llm, {"id": t["id"], **(t["body"] or {})})
                # the pattern cards the casting was shown (manyworlds.creator.lab.casting_cards), for attribution
                cards = [{"key": c["key"], "id": c.get("id"), "version": c.get("version"), "stage": "casting"}
                         for c in lab.casting_cards(ctx.conn, ctx.policy, {"id": t["id"], **(t["body"] or {})})]
                return {"kind": "creator_launch_step", "inputs": {"launch_id": str(launch["id"]), "step": step},
                        "output": {"creator_id": str(cid), "pattern_cards": cards},
                        "rationale": "bible written from the thesis",
                        "creator_id": cid}
            cid = launch["creator_id"]
            out = (LAB.anchors if step == "anchors" else LAB.profile)(ctx.conn, ctx.policy, ctx.llm, cid)
            return {"kind": "creator_launch_step", "inputs": {"launch_id": str(launch["id"]), "step": step},
                    "output": {"result": out if isinstance(out, (dict, int, str)) else str(out)},
                    "rationale": f"{step} for the launch", "creator_id": cid}

        return remember(ctx, step, compute)

    def casting(self, ctx: AgentContext, p: dict) -> dict:
        from .. import casting
        step = p["step"]
        if step == "casting":
            launch = ctx.conn.execute("SELECT * FROM launches WHERE id::text = %s", (p.get("launch_id"),)).fetchone()
            thesis_id = launch["thesis_id"] if launch else p.get("thesis_id")
            if not thesis_id:
                raise NonRetryable("a casting step needs a launch_id (or a thesis_id)")
            run = casting.launch_run(launch) if launch else str(p.get("run") or f"t{str(thesis_id)[:8]}")

            def compute() -> dict:
                out = casting.cast(ctx.conn, ctx.policy, ctx.llm, thesis_id, run=run,
                                   launch_id=launch["id"] if launch else None)
                return {"kind": "casting_step", "inputs": {"thesis_id": str(thesis_id), "run": run},
                        "output": out, "rationale": "casting: concepts, judge, shortlist"}
            return remember(ctx, step, compute)
        audition_id = p.get("audition_id")
        if not audition_id or not ctx.conn.execute("SELECT 1 FROM auditions WHERE id::text = %s",
                                                   (str(audition_id),)).fetchone():
            raise NonRetryable(f"{step} needs an existing audition_id")

        def compute_audition() -> dict:
            if step == "audition":
                out = casting.audition_bible(ctx.conn, ctx.policy, ctx.llm, audition_id)
            else:
                from ... import storage
                with tempfile.TemporaryDirectory() as tmp:
                    out = casting.audition_anchors(ctx.conn, ctx.policy, ctx.llm, storage.from_url(), audition_id, tmp)
            return {"kind": "casting_step", "inputs": {"audition_id": str(audition_id), "step": step},
                    "output": out, "rationale": f"casting: {step}"}
        return remember(ctx, step, compute_audition)


CASTING_STEPS = ("casting", "audition", "audition_anchors")
register(CreatorLab())
