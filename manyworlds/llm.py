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

Inside a brain job on the brain worker (`batching`), calls are journaled and the roles in `llm.batch.roles` go
through the Message Batches API at half price (manyworlds.llm_batch).
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

from . import ledger, llm_batch
from .llm_batch import Deferred  # noqa: F401 - re-exported: a batched call not answered yet
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


def audio_block(data: bytes, fmt: str = "mp3") -> dict:
    """A clip of audio for an open model that hears (llm.open_models.<model>.audio: true; the music listener).
    Claude takes no audio: a call with one to a Claude model fails as an OpenModelError."""
    return {"type": "audio", "source": {"type": "base64", "format": fmt,
                                        "data": base64.standard_b64encode(data).decode()}}


def _has_audio(content: Any) -> bool:
    return not isinstance(content, str) and any(isinstance(b, dict) and b.get("type") == "audio" for b in content)


def _post_json(url: str, body: dict, key: str, timeout: float) -> dict:
    """POST JSON with a bearer key (in the header, never the URL). Errors carry no key."""
    from .deploy.providers import redact
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                          # Together sits behind a firewall that refuses urllib's default agent (403)
                                          "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _read_stream(r) if body.get("stream") else json.loads(r.read())
    except urllib.error.HTTPError as e:
        text = e.read().decode(errors="replace")[:500]
        raise OpenModelError(redact(f"{url} -> {e.code}: {text}").replace(key, "<key>")) from None
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise OpenModelError(redact(f"{url} -> {type(e).__name__}: {e}").replace(key, "<key>")) from None


def _read_stream(lines) -> dict:
    """A streamed (server-sent events) chat completion as the one-piece response shape: some hosted models
    (Qwen on Together) only answer streamed. Usage comes in the last event (`stream_options.include_usage`)."""
    text, usage = [], {}
    for raw in lines:
        line = raw.decode(errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        event = json.loads(line[5:])
        for choice in event.get("choices") or []:
            text.append((choice.get("delta") or {}).get("content") or "")
        usage = event.get("usage") or usage
    return {"choices": [{"message": {"content": "".join(text)}}], "usage": usage}


def _openai_content(content: Any, vision: bool, audio: bool = False) -> Any:
    """Anthropic-style content (a string or text/image/audio blocks) as OpenAI chat content."""
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
        elif block.get("type") == "audio":
            if not audio:
                raise OpenModelError("not an audio model")
            src = block["source"]
            out.append({"type": "input_audio", "input_audio": {"data": src["data"], "format": src["format"]}})
        else:
            raise OpenModelError(f"unsupported content block {block.get('type')!r}")
    return out


_DROPPED_KEYS = ("default", "title", "$defs")


def strict_json_schema(model: type[BaseModel]) -> dict:
    """A model's JSON schema in the strict shape structured outputs compile cheaply: `$ref`s inlined, every
    object's properties all required with `additionalProperties: false` (dict-typed fields too, which become
    empty objects), `anyOf [X, null]` collapsed to X, "" added to every string enum for "unset", and `default`
    and `title` dropped. Measured 2026-11-22: ShotSheet's ordinary schema fails with "The compiled grammar is
    too large"; this shape works. Models read back from it turn "" into None (manyworlds.direction._Sheet)."""
    raw = model.model_json_schema()
    defs = raw.get("$defs") or {}

    def conv(node: Any) -> Any:
        if isinstance(node, list):
            return [conv(n) for n in node]
        if not isinstance(node, dict):
            return node
        node = dict(node)
        keep = {k: node[k] for k in ("description",) if k in node}
        if "$ref" in node:
            return {**conv(defs[node["$ref"].rsplit("/", 1)[-1]]), **keep}
        if isinstance(node.get("allOf"), list) and len(node["allOf"]) == 1:
            return {**conv(node["allOf"][0]), **keep}
        if isinstance(node.get("anyOf"), list):
            options = [o for o in node["anyOf"] if not (isinstance(o, dict) and o.get("type") == "null")]
            if len(options) == 1:
                return {**conv(options[0]), **keep}
        out: dict = {}
        for k, v in node.items():
            if k in _DROPPED_KEYS:
                continue
            if k == "properties":
                out[k] = {name: conv(sub) for name, sub in v.items()}
            elif k in ("items", "anyOf", "oneOf", "prefixItems"):
                out[k] = conv(v)
            elif k == "additionalProperties":
                continue   # set below for every object
            else:
                out[k] = v
        if isinstance(out.get("enum"), list) and all(isinstance(e, str) for e in out["enum"]) and "" not in out["enum"]:
            out["enum"] = [*out["enum"], ""]
        if out.get("type") == "object" or "properties" in out:
            out.setdefault("type", "object")
            out["properties"] = out.get("properties") or {}
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        return out

    out = conv({k: v for k, v in raw.items() if k != "$defs"})
    # `strict_omit` (a model ClassVar): property paths left out of what the model is asked to fill, each a tuple of
    # property names through objects and arrays, e.g. ("shots", "edit"); the model's own defaults fill them.
    for path in getattr(model, "strict_omit", ()) or ():
        node = out
        for name in path[:-1]:
            node = node["properties"][name]
            node = node.get("items", node)
        node["properties"].pop(path[-1], None)
        node["required"] = [r for r in node["required"] if r != path[-1]]
    return out


def _first_text(response) -> str:
    """The answer's text: the first text block (a thinking block may come first)."""
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "text":
            return block.text
    return ""


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
        self.trial_usd: float | None = None   # set for a test run: every call goes to every shadow candidate
        self.trial_spent = 0.0
        self.shadowing = True   # off while model selection judges shadow answers (no shadows of the judging)
        self._overrides: dict | None = None   # approved role switches, re-read every llm.override_cache_seconds
        self._overrides_at = 0.0
        self.batching = False   # journal calls in brain jobs and batch llm.batch.roles (the brain worker sets it)

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    # --- routing -------------------------------------------------------------------------------------------
    def route(self, role: str) -> tuple[str, str | None]:
        """(model, fallback) for a role: an approved switch (`llm.roles.<role>` in policy_overrides, from model
        selection) wins over the policy file, in every service that calls models, not only the Brain."""
        spec = self._role_override(role) or self.policy.get("llm", "roles", role)
        if not spec:
            raise KeyError(f"no model configured for LLM role {role!r}")
        if isinstance(spec, dict):
            return str(spec["model"]), (str(spec["fallback"]) if spec.get("fallback") else None)
        return str(spec), None

    def _role_override(self, role: str):
        if self.conn is None:
            return None
        ttl = float(self.policy.get("llm", "override_cache_seconds", default=60))
        if self._overrides is None or time.monotonic() - self._overrides_at > ttl:
            try:
                rows = self.conn.execute(
                    "SELECT DISTINCT ON (path) path, value FROM policy_overrides WHERE path LIKE 'llm.roles.%%' "
                    "ORDER BY path, version DESC").fetchall()
                self._overrides = {r["path"].removeprefix("llm.roles."): r["value"] for r in rows}
            except Exception as e:  # noqa: BLE001 - no override table or connection: the policy file decides
                log.warning("role overrides unreadable (%s); using the policy file", type(e).__name__)
                self._overrides = {}
            self._overrides_at = time.monotonic()
        return self._overrides.get(role)

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
    def tagged(self, agent: str, job_id, before_call=None, attempt: int | None = None):
        """Charge every call inside the block to this agent and job (per-agent budgets, traces).
        `before_call` runs before each call, e.g. the Brain runtime's budget check, so an agent
        that keeps calling is stopped mid-job, not only before it starts. `attempt`: the job's attempt, which keys
        its journaled calls (manyworlds.llm_batch); each block is one run of the job."""
        before, self._tag = self._tag, {"agent": agent, "job_id": job_id, "before_call": before_call,
                                        "attempt": attempt}
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
        if llm_batch.active(self):
            return llm_batch.journaled(self, role, model, fallback, system=system, content=content, schema=schema,
                                       effort=effort, creator_id=creator_id)
        return self.answer(role, model, fallback, system=system, content=content, schema=schema, effort=effort,
                           creator_id=creator_id)

    def answer(self, role: str, model: str, fallback: str | None, *, system: str, content: Any, schema: type[T],
               effort: str, creator_id) -> T:
        """One call now: the role's model, its fallback if an open model fails, and the role's shadow test."""
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
            # Effort is Claude's (output_config.effort). The OpenAI-compatible hosts in llm.hosts have no portable
            # reasoning parameter we have verified, so an open model answers at its own default; said at debug level.
            log.debug("effort %r not sent to open model %s (no verified reasoning parameter on its host)", effort, model)
            return self._open(model, system=system, content=content, schema=schema, creator_id=creator_id,
                              agent=agent)
        if _has_audio(content):
            raise OpenModelError(f"{model} takes no audio")
        return self._claude(model, system=system, content=content, schema=schema, effort=effort,
                            creator_id=creator_id, agent=agent)

    def _system(self, system):
        """The system prompt as a cached block when it is long enough to cache (policy
        llm.cache_system_min_chars): agents send the same long instructions on every call (showrunner, director,
        evaluator rubrics, talent manager), so the repeats are read from the cache at a tenth of the input price."""
        floor = int(self.policy.get("llm", "cache_system_min_chars", default=0) or 0)
        if not floor or not isinstance(system, str) or len(system) < floor:
            return system
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    def _claude(self, model: str, *, system, content, schema: type[T], effort: str, creator_id,
                agent: str | None = None) -> T:
        system = self._system(system)
        # Server-side refusal fallback exists only for some models (not Haiku); policy lists them.
        fallback = {}
        if model in (self.policy.get("llm", "server_fallback_models", default=[]) or []):
            fallback = {"extra_headers": {"anthropic-beta": FALLBACK_BETA}, "extra_body": {"fallbacks": "default"}}
        if getattr(schema, "strict_required", False):
            # A schema too large for messages.parse (manyworlds.direction.DirectedShots): the strict shape, sent as
            # a JSON-schema output format, validated here.
            response = self.client.messages.create(
                model=model,
                max_tokens=int(self.policy.get("llm", "max_output_tokens", default=16000)),
                system=system,
                messages=[{"role": "user", "content": content}],
                output_config={"effort": effort,
                               "format": {"type": "json_schema", "schema": strict_json_schema(schema)}},
                **fallback,
            )
            self._charge(response.model, response.usage, creator_id, agent)
            self._raise_if_refused(model, response)
            try:
                return schema.model_validate_json(_json_text(_first_text(response)))
            except ValidationError as e:
                raise RuntimeError(f"{model} returned no valid {schema.__name__} "
                                   f"(stop_reason={response.stop_reason}): {str(e)[:300]}") from None
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
        self._raise_if_refused(model, response)
        if response.parsed_output is None:
            raise RuntimeError(f"{model} returned no parsable output (stop_reason={response.stop_reason})")
        return response.parsed_output

    def batch_params(self, model: str, *, system, content, schema: type[T], effort: str) -> dict:
        """A Claude call's Messages parameters for the Message Batches API: the same model, prompt (cached system
        block included), effort and output schema as `_claude` sends (messages.parse's own schema transform), less
        the server-side refusal fallback, which a batch request can't carry."""
        if getattr(schema, "strict_required", False):
            fmt = {"type": "json_schema", "schema": strict_json_schema(schema)}
        else:
            try:
                from anthropic.lib._parse._transform import to_json_output_format
                fmt = to_json_output_format(schema)
            except ImportError:   # an SDK without the helper: the strict shape is accepted too
                fmt = {"type": "json_schema", "schema": strict_json_schema(schema)}
        return {"model": model, "max_tokens": int(self.policy.get("llm", "max_output_tokens", default=16000)),
                "system": self._system(system), "messages": [{"role": "user", "content": content}],
                "output_config": {"effort": effort, "format": fmt}}

    @staticmethod
    def _raise_if_refused(model: str, response) -> None:
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise Refused(f"{model} declined ({getattr(details, 'category', None)}): "
                          f"{getattr(details, 'explanation', '')}")

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
                             {"role": "user", "content": _openai_content(content, bool(spec.get("vision")),
                                                                         bool(spec.get("audio")))}],
                "max_tokens": int(spec.get("max_tokens", self.policy.get("llm", "max_output_tokens", default=16000))),
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": schema.__name__, "strict": False,
                    "schema": strict_json_schema(schema) if getattr(schema, "strict_required", False)
                    else schema.model_json_schema()}}}
        if spec.get("stream"):
            body.update(stream=True, stream_options={"include_usage": True})
        prices = spec.get("usd_per_mtok") or [0, 0]
        timeout = float(spec.get("timeout_seconds", host.get("timeout_seconds", 120)))   # slow thinkers need more
        last = None
        for _ in range(int(spec.get("attempts", 2))):
            try:
                r = self._post(f"{host['base_url'].rstrip('/')}/chat/completions", body, key, timeout)
            except OpenModelError as e:
                # A timeout or dropped connection is billed by the host with no usage returned: record an estimate
                # so the ledger matches the bill (2026-11-22: ~$1.80 of Together's bill was missing). HTTP errors
                # (a refused request) are not billed.
                if " -> " in str(e) and not any(f" -> {c}" in str(e) for c in range(400, 600)):
                    est = self.policy.get("llm", "failed_call_estimate", default={}) or {}
                    parts = [body["messages"][0]["content"], body["messages"][1]["content"]]
                    text = "".join(p if isinstance(p, str) else "".join(b.get("text", "") for b in p) for p in parts)
                    images = sum(1 for p in parts if not isinstance(p, str) for b in p if b.get("type") == "image_url")
                    tin = int(len(text) / float(est.get("chars_per_token", 4))) + images * int(est.get("image_tokens", 1500))
                    tout = int(est.get("output_tokens", 3000))
                    self._record(model, tin, tout, 0, 0, (tin * prices[0] + tout * prices[1]) / 1e6, creator_id, agent)
                raise
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
        if not cfg or not self.shadowing:
            return
        # One candidate or several (`candidates`): each sampled call goes to one of them, so a role can try
        # several models side by side against the same primary. In a trial (`self.trial_usd`, set for a test
        # run) every call goes to every candidate, so one short run compares them all on the same inputs.
        pool = [str(c) for c in (cfg.get("candidates") or [cfg.get("candidate")]) if c and str(c) != model]
        if not pool:
            return
        if self.trial_usd is None:
            if random.random() >= float(cfg.get("share", 0)):
                return
            pool = [random.choice(pool)]
        for candidate in pool:
            self._shadow_one(role, model, candidate, out, primary_usd, **call)

    def _shadow_one(self, role: str, model: str, candidate: str, out: BaseModel, primary_usd: float, **call) -> None:
        try:
            if self.trial_usd is not None:
                if self.trial_spent >= self.trial_usd:
                    return
            else:
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
            usd = sum(c.usd for c in self.calls[n0:])
            self.trial_spent += usd
            system = call.get("system")
            request = (f"{system if isinstance(system, str) else json.dumps(system, default=str)}\n---\n"
                       f"{llm_batch.content_text(call.get('content'))}")
            limit = int(self.policy.get("llm", "shadow_request_chars", default=12000))
            self.conn.execute(
                "INSERT INTO llm_shadow (role, primary_model, candidate_model, primary_output, candidate_output, "
                "candidate_error, primary_usd, candidate_usd, candidate_seconds, agent, creator_id, request) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (role, model, candidate, Jsonb(out.model_dump(mode="json")), Jsonb(got) if got is not None else None,
                 error, primary_usd, usd, round(time.monotonic() - t0, 3), self._tag.get("agent"),
                 call.get("creator_id"), request[-limit:] if limit else None))
        except Exception as e:  # noqa: BLE001 - shadow bookkeeping must never fail the real call
            log.warning("shadow call for %s failed: %s", role, type(e).__name__)

    @property
    def spent(self) -> float:
        return round(sum(c.usd for c in self.calls), 6)


Claude = LLM   # the name most callers know; it routes to open-weight models too
