"""The scouts: research discovery as a Brain job (manyworlds.brain.discover). `discover.ensure_due` queues one
`brain.scouts` job per `research.discover.every_days` while operations are live; the job plans the research, pulls
it (Apify, within `research.max_items_per_run`) and writes proposed theses, on the scouts' daily LLM budget."""

from __future__ import annotations

from ...research.apify import ApifySource
from .. import discover
from ..agent import AgentContext, register, remember


class Scouts:
    name = "scouts"

    def source(self, ctx: AgentContext):
        return ApifySource(ctx.conn, ctx.policy)

    def run(self, ctx: AgentContext) -> dict:
        def compute() -> dict:
            out = discover.discover(ctx.conn, ctx.policy, ctx.llm, self.source(ctx))
            return {"kind": "discovery", "inputs": dict(ctx.job.payload or {}),
                    "output": {"theses": [str(t["id"]) for t in out["theses"]],
                               "items": sum(int(p.get("items", 0) or 0) for p in out["pulls"]),
                               "hashtags": sum(len(p["hashtags"]) for p in out["plan"]["plans"])},
                    "rationale": "scheduled research discovery"}
        return remember(ctx, "discover", compute)


register(Scouts())
