"""Every LLM call goes through here: structured outputs, images, and cost in the ledger.

Each role (policy `llm.roles`) names a model, or `{model, fallback}`. A model is either Claude (called with the
Anthropic SDK, validated with messages.parse, checked for refusals; Opus/Sonnet refusals fall back server-side
with `fallbacks: "default"`) or an open-weight model in `llm.open_models` on an OpenAI-compatible host in
`llm.hosts` (Together, DeepInfra, ...; JSON-schema output validated here, one retry on invalid output). When
the role's model fails (host down, invalid output, no key, not a vision model) the `fallback` model answers.

Shadow testing (`llm.shadow.<role>`): a share of a role's calls also go to a candidate model; both outputs are
stored in `llm_shadow` and compared by `manyworlds.shadow`. The candidate never changes the answer, its
failures are only recorded, and its spend is capped per day (`llm.shadow_usd_per_day`).

Every call, failed attempts included, is charged at the policy's per-token prices.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Protocol, TypeVar

from psycopg.types.json import Jsonb
from pydantic import BaseModel, ValidationError

from . import ledger
from .policy import Policy

T = TypeVar("T", bound=BaseModel)
FALLBACK_BETA = "server-side-fallback-2026-07-01"
USER_AGENT = "manyworlds/1.0"
SHADOW_AGENT = "shadow"     # ledger tag for shadow calls: outside every agent's budget and every output's cost

log = logging.getLogger("manyworlds.llm")


class Refused(RuntimeError):
    pass


class OpenModelError(RuntimeError):
    """An open-weight model couldn't answer (no key, host error, invalid output): the fallback model answers."""


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


def _post_json(url: str, body: dict, key: str, timeout: float) -> dict:
    """POST JSON with a bearer key (in the header, never the URL). Errors carry no key."""
    from .deploy.providers import redact
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          # Together sits behind a firewall that refuses urllib's default agent (403)
                                          "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        text = e.read().decode(errors="replace")[:500]
        raise OpenModelError(redact(f"{url} -> {e.code}: {text}").replace(key, "<key>")) from None
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise OpenModelError(redact(f"{url} -> {type(e).__name__}: {e}").replace(key, "<key>")) from None


def _openai_content(content: Any, vision: bool) -> Any:
    """Anthropic-style content (a string or text/image blocks) as OpenAI chat content."""
    if isinstance(content, str):
        return content
    out = []
    for block in content:
        if block.get("type") == "text":
            out.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image":
            if not vision:
                raise OpenModelError("not a vision model")
            src = block["source"]
            out.append({"type": "image_url", "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"}})
        else:
            raise OpenModelError(f"unsupported content block {block.get('type')!r}")
    return out


def _json_text(text: str) -> str:
    """Model output as bare JSON (some hosts wrap it in a ``` fence)."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t.strip()


class LLM:
    def __init__(self, conn, policy: Policy, client=None, post: Callable = _post_json):
        self.conn = conn
        self.policy = policy
        self._client = client        # Anthropic client, made on first Claude call
        self._post = post
        self.calls: list[Usage] = []
        self._tag: dict = {}   # agent and job the current calls are charged to (Brain runtime)

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    # --- routing -------------------------------------------------------------------------------------------
    def route(self, role: str) -> tuple[str, str | None]:
        """(model, fallback) for a role."""
        spec = self.policy.get("llm", "roles", role)
        if not spec:
            raise KeyError(f"no model configured for LLM role {role!r}")
        if isinstance(spec, dict):
            return str(spec["model"]), (str(spec["fallback"]) if spec.get("fallback") else None)
        return str(spec), None

    def model_for(self, role: str) -> str:
        return self.route(role)[0]

    def is_open(self, model: str) -> bool:
        return bool(self.policy.get("llm", "open_models", model))

    # --- charging ------------------------------------------------------------------------------------------
    def _record(self, model: str, input_tokens: int, output_tokens: int, cr: int, cw: int, usd: float,
                creator_id, agent: str | None = None) -> Usage:
        u = Usage(model, input_tokens, output_tokens, cr, cw, usd)
        ledger.record(self.conn, category="llm", provider=model,
                      quantity=input_tokens + output_tokens + cr + cw, unit="token",
                      usd=usd, creator_id=creator_id, policy_version=self.policy.version,
                      agent=agent if agent is not None else self._tag.get("agent"), job_id=self._tag.get("job_id"))
        self.calls.append(u)
        return u

    def _charge(self, model: str, usage, creator_id, agent: str | None = None) -> Usage:
        prices = self.policy.get("llm", "usd_per_mtok", model) or [0, 0, 0, 0]
        cr = getattr(usage, "cache_read_input_tokens", 0) or 0
        cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
        usd = (usage.input_tokens * prices[0] + usage.output_tokens * prices[1]
               + cr * prices[2] + cw * prices[3]) / 1e6
        return self._record(model, usage.input_tokens, usage.output_tokens, cr, cw, usd, creator_id, agent)

    @contextlib.contextmanager
    def tagged(self, agent: str, job_id, before_call=None):
        """Charge every call inside the block to this agent and job (per-agent budgets, traces).
        `before_call` runs before each call, e.g. the Brain runtime's budget check, so an agent
        that keeps calling is stopped mid-job, not only before it starts."""
        before, self._tag = self._tag, {"agent": agent, "job_id": job_id, "before_call": before_call}
        try:
            yield
        finally:
            self._tag = before

    # --- calls ---------------------------------------------------------------------------------------------
    def structured(self, role: str, *, system: str, content: Any, schema: type[T],
                   effort: str = "high", creator_id=None) -> T:
        if self._tag.get("before_call"):
            self._tag["before_call"]()
        model, fallback = self.route(role)
        n0 = len(self.calls)
        try:
            out = self._call(model, system=system, content=content, schema=schema, effort=effort,
                             creator_id=creator_id)
        except OpenModelError as e:
            if not fallback:
                raise
            log.warning("%s failed for role %s (%s); falling back to %s", model, role, e, fallback)
            model = fallback
            out = self._call(fallback, system=system, content=content, schema=schema, effort=effort,
                             creator_id=creator_id)
        self._shadow(role, model, out, sum(c.usd for c in self.calls[n0:]), system=system, content=content,
                     schema=schema, effort=effort, creator_id=creator_id)
        return out

    def _call(self, model: str, *, system, content, schema: type[T], effort: str, creator_id,
              agent: str | None = None) -> T:
        if self.is_open(model):
            return self._open(model, system=system, content=content, schema=schema, creator_id=creator_id,
                              agent=agent)
        return self._claude(model, system=system, content=content, schema=schema, effort=effort,
                            creator_id=creator_id, agent=agent)

    def _claude(self, model: str, *, system, content, schema: type[T], effort: str, creator_id,
                agent: str | None = None) -> T:
        # Server-side refusal fallback exists only for some models (not Haiku); policy lists them.
        fallback = {}
        if model in (self.policy.get("llm", "server_fallback_models", default=[]) or []):
            fallback = {"extra_headers": {"anthropic-beta": FALLBACK_BETA}, "extra_body": {"fallbacks": "default"}}
        response = self.client.messages.parse(
            model=model,
            max_tokens=int(self.policy.get("llm", "max_output_tokens", default=16000)),
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=schema,
            output_config={"effort": effort},
            **fallback,
        )
        self._charge(response.model, response.usage, creator_id, agent)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise Refused(f"{model} declined ({getattr(details, 'category', None)}): "
                          f"{getattr(details, 'explanation', '')}")
        if response.parsed_output is None:
            raise RuntimeError(f"{model} returned no parsable output (stop_reason={response.stop_reason})")
        return response.parsed_output

    def _open(self, model: str, *, system, content, schema: type[T], creator_id, agent: str | None = None) -> T:
        spec = self.policy.get("llm", "open_models", model)
        host = self.policy.get("llm", "hosts", spec["host"])
        if not host:
            raise OpenModelError(f"no llm.hosts entry {spec['host']!r} for {model}")
        key = os.environ.get(host["key_env"])
        if not key:
            raise OpenModelError(f"{host['key_env']} is not set")
        body = {"model": spec["id"],
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": _openai_content(content, bool(spec.get("vision")))}],
                "max_tokens": int(spec.get("max_tokens", self.policy.get("llm", "max_output_tokens", default=16000))),
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": schema.__name__, "schema": schema.model_json_schema(), "strict": False}}}
        prices = spec.get("usd_per_mtok") or [0, 0]
        timeout = float(host.get("timeout_seconds", 120))
        last = None
        for _ in range(int(spec.get("attempts", 2))):
            r = self._post(f"{host['base_url'].rstrip('/')}/chat/completions", body, key, timeout)
            usage = r.get("usage") or {}
            tin, tout = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
            self._record(model, tin, tout, 0, 0, (tin * prices[0] + tout * prices[1]) / 1e6, creator_id, agent)
            try:
                return schema.model_validate_json(_json_text(r["choices"][0]["message"]["content"]))
            except (ValidationError, KeyError, IndexError, TypeError) as e:
                last = e
        raise OpenModelError(f"{model} gave no valid {schema.__name__}: {str(last)[:300]}")

    # --- shadow testing ------------------------------------------------------------------------------------
    def _shadow(self, role: str, model: str, out: BaseModel, primary_usd: float, **call) -> None:
        cfg = self.policy.get("llm", "shadow", role)
        if not cfg:
            return
        # One candidate or several (`candidates`): each sampled call goes to one of them, so a role can try
        # several models side by side against the same primary.
        pool = [str(c) for c in (cfg.get("candidates") or [cfg.get("candidate")]) if c and str(c) != model]
        if not pool or random.random() >= float(cfg.get("share", 0)):
            return
        candidate = random.choice(pool)
        try:
            cap = float(self.policy.get("llm", "shadow_usd_per_day", default=1.0))
            start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            spent = self.conn.execute("SELECT coalesce(sum(usd), 0) AS u FROM cost_entries WHERE agent = %s "
                                      "AND created_at >= %s", (SHADOW_AGENT, start)).fetchone()["u"]
            if float(spent) >= cap:
                return
            n0, t0 = len(self.calls), time.monotonic()
            try:
                got, error = self._call(candidate, agent=SHADOW_AGENT, **call).model_dump(mode="json"), None
            except Exception as e:  # noqa: BLE001 - a candidate's failure is a result, never the caller's problem
                got, error = None, f"{type(e).__name__}: {e}"[:500]
            self.conn.execute(
                "INSERT INTO llm_shadow (role, primary_model, candidate_model, primary_output, candidate_output, "
                "candidate_error, primary_usd, candidate_usd, candidate_seconds, agent) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (role, model, candidate, Jsonb(out.model_dump(mode="json")), Jsonb(got) if got is not None else None,
                 error, primary_usd, sum(c.usd for c in self.calls[n0:]), round(time.monotonic() - t0, 3),
                 self._tag.get("agent")))
        except Exception as e:  # noqa: BLE001 - shadow bookkeeping must never fail the real call
            log.warning("shadow call for %s failed: %s", role, type(e).__name__)

    @property
    def spent(self) -> float:
        return round(sum(c.usd for c in self.calls), 6)


Claude = LLM   # the name most callers know; it routes to open-weight models too
