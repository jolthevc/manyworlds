"""Structured logs: one JSON object a line, with what is being worked on.

`configure(json_logs, role=...)` sets up the root logger once per process (manyworlds.cli.main). Each line carries
ts, level, logger, msg, the role (the `mw` command: produce, review, worker...) and the Fly process group, plus the
context fields set around the work in hand:

    with logs.context(job_id=job.id, kind=job.kind):
        ...   # every log line in here (any logger, this thread) carries job_id and kind

Context lives in contextvars, so it nests (an inner block adds to or overrides the outer one and restores it on
exit) and threads don't share it. Unset fields are left out. Messages are logged as given: never pass a secret or a
DSN to a logger (manyworlds.deploy.providers.redact cleans provider errors).
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Iterator

# The context fields a line may carry, in output order.
FIELDS = ("post_id", "job_id", "attempt_id", "creator_id", "kind")

_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("mw_log_context", default={})
_role: dict[str, str | None] = {"role": None}


@contextlib.contextmanager
def context(**fields: Any) -> Iterator[None]:
    """Add fields to every log line in this block (this thread). None values unset a field."""
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise TypeError(f"unknown log context field(s): {sorted(unknown)}; known: {FIELDS}")
    merged = {**_context.get(), **fields}
    token = _context.set({k: v for k, v in merged.items() if v is not None})
    try:
        yield
    finally:
        _context.reset(token)


def current() -> dict[str, Any]:
    """The context fields set now (a copy)."""
    return dict(_context.get())


def _text(v: Any) -> Any:
    return v if isinstance(v, (int, float, bool)) or v is None else str(v)


class JsonFormatter(logging.Formatter):
    """A log record as one JSON line: ts, level, logger, msg, role, group, the context fields that are set, and
    exc (the traceback) when there is one. Stdlib only."""

    def __init__(self, role: str | None = None, group: str | None = None):
        super().__init__()
        self.role = role
        self.group = group

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        role = self.role or _role["role"]
        if role:
            out["role"] = role
        if self.group:
            out["group"] = self.group
        ctx = _context.get()
        for k in FIELDS:
            if ctx.get(k) is not None:
                out[k] = _text(ctx[k])
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        elif record.exc_text:
            out["exc"] = record.exc_text
        return json.dumps(out, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """The plain format (logging.basicConfig's), with the context fields appended as k=v when any are set."""

    def __init__(self):
        super().__init__(logging.BASIC_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        ctx = _context.get()
        extra = " ".join(f"{k}={ctx[k]}" for k in FIELDS if ctx.get(k) is not None)
        if not extra:
            return line
        head, sep, tail = line.partition("\n")   # keep a traceback below the context
        return f"{head} [{extra}]{sep}{tail}"


def json_wanted(policy=None, env=None) -> bool:
    """MW_LOG_JSON=1/0 decides; without it, policy logging.json (off when there is no policy)."""
    env = os.environ if env is None else env
    flag = env.get("MW_LOG_JSON")
    if flag is not None and flag.strip() != "":
        return flag.strip().lower() in ("1", "true", "yes", "on")
    if policy is None:
        return False
    return bool(policy.get("logging", "json", default=False))


def configure(json_logs: bool, *, role: str | None = None, level: str | int | None = None, env=None) -> bool:
    """Point the root logger at stderr in the chosen format, like logging.basicConfig: only when the root logger has
    no handlers yet (under pytest its capture handlers are there, so tests keep their own logging untouched). The
    process group comes from FLY_PROCESS_GROUP. Returns whether it configured anything."""
    env = os.environ if env is None else env
    _role["role"] = role
    root = logging.getLogger()
    if root.handlers:
        return False
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter(role=role, group=env.get("FLY_PROCESS_GROUP") or None) if json_logs
                         else TextFormatter())
    root.addHandler(handler)
    root.setLevel(level or env.get("LOG_LEVEL", "INFO"))
    return True
