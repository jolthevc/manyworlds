"""A minimal agent that exercises the whole runtime path: one LLM call, one decision."""

from __future__ import annotations

from pydantic import BaseModel

from ..agent import AgentContext, record_decision, register


class _Ack(BaseModel):
    ok: bool = True


class Echo:
    name = "echo"

    def run(self, ctx: AgentContext) -> dict:
        ctx.llm.structured("brain", system="Acknowledge.", content="ping", schema=_Ack, effort="low")
        payload = dict(ctx.job.payload or {})
        record_decision(ctx, "echo", step="echo", inputs={}, output=payload, rationale="echo test")
        return payload


register(Echo())
