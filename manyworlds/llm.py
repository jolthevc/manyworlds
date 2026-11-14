"""Claude behind one small wrapper: structured outputs, images, and cost in the ledger.

Every call is validated against a Pydantic model (messages.parse), checked for a
refusal, and charged to cost_entries at the policy's per-token prices. Refusals on
Opus 5.5 fall back server-side (`fallbacks: "default"`), so an over-cautious decline
on a harmless lifestyle prompt is retried on another model inside the same call.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from . import ledger
from .policy import Policy

T = TypeVar("T", bound=BaseModel)
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class Refused(RuntimeError):
    pass


@dataclass
class Usage:
    model: str
    input_tokens: int
    output_tokens: int
    cache_read: int
    cache_write: int
    usd: float


class StructuredLLM(Protocol):
    def structured(self, role: str, *, system: str, content: Any, schema: type[T],
                   effort: str = "high", creator_id=None) -> T: ...


def image_block(data: bytes, media_type: str = "image/png") -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                        "data": base64.standard_b64encode(data).decode()}}


class Claude:
    def __init__(self, conn, policy: Policy, client=None):
        import anthropic

        self.conn = conn
        self.policy = policy
        self.client = client or anthropic.Anthropic()
        self.calls: list[Usage] = []

    def model_for(self, role: str) -> str:
        model = self.policy.get("llm", "roles", role)
        if not model:
            raise KeyError(f"no model configured for LLM role {role!r}")
        return model

    def _charge(self, model: str, usage, creator_id) -> Usage:
        prices = self.policy.get("llm", "usd_per_mtok", model) or [0, 0, 0, 0]
        cr = getattr(usage, "cache_read_input_tokens", 0) or 0
        cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
        usd = (usage.input_tokens * prices[0] + usage.output_tokens * prices[1]
               + cr * prices[2] + cw * prices[3]) / 1e6
        u = Usage(model, usage.input_tokens, usage.output_tokens, cr, cw, usd)
        ledger.record(self.conn, category="llm", provider=model,
                      quantity=usage.input_tokens + usage.output_tokens + cr + cw, unit="token",
                      usd=usd, creator_id=creator_id, policy_version=self.policy.version)
        self.calls.append(u)
        return u

    def structured(self, role: str, *, system: str, content: Any, schema: type[T],
                   effort: str = "high", creator_id=None) -> T:
        model = self.model_for(role)
        # Server-side refusal fallback exists only for some models (not Haiku); policy lists them.
        fallback = {}
        if model in (self.policy.get("llm", "server_fallback_models", default=[]) or []):
            fallback = {"extra_headers": {"anthropic-beta": FALLBACK_BETA}, "extra_body": {"fallbacks": "default"}}
        response = self.client.messages.parse(
            model=model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=schema,
            output_config={"effort": effort},
            **fallback,
        )
        self._charge(response.model, response.usage, creator_id)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise Refused(f"{model} declined ({getattr(details, 'category', None)}): "
                          f"{getattr(details, 'explanation', '')}")
        if response.parsed_output is None:
            raise RuntimeError(f"{model} returned no parsable output (stop_reason={response.stop_reason})")
        return response.parsed_output

    @property
    def spent(self) -> float:
        return round(sum(c.usd for c in self.calls), 6)
