"""A minimal agent that exercises the whole runtime path: one LLM call, one decision."""

from __future__ import annotations

from pydantic import BaseModel

from ..agent import AgentContext, register, remember


class _Ack(BaseModel):
    ok: bool = True


class Echo:
    name = "echo"

    def run(self, ctx: AgentContext) -> dict:
        def step() -> dict:
            ctx.llm.structured("brain", system="Acknowledge.", content="ping", schema=_Ack, effort="low")
            return {"kind": "echo", "inputs": {}, "output": dict(ctx.job.payload or {}), "rationale": "echo test"}
        return remember(ctx, "echo", step)


register(Echo())
