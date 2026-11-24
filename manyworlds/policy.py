"""Versioned policy: every tunable number lives in a YAML file, never in code.

The version is a hash of the policy's canonical JSON, so any change produces a new
version, and every cost entry and decision can name the exact policy it ran under.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from psycopg.types.json import Jsonb


@dataclass(frozen=True)
class Policy:
    body: dict[str, Any]
    version: str

    def get(self, *path: str, default: Any = None) -> Any:
        node: Any = self.body
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def pool(self, name: str) -> dict[str, Any]:
        pools = self.body.get("gpu_pools", {})
        if name not in pools:
            raise KeyError(f"unknown GPU pool {name!r}; known: {sorted(pools)}")
        return pools[name]

    def family(self, kind: str) -> str:
        """A job kind's worker family (its worker_images entry, machine limits and no-output guard):
        job_kinds.<kind>.family when set, else the kind's prefix (image.zimage -> image)."""
        override = self.get("job_kinds", kind, "family")
        return str(override) if override else kind.split(".")[0]

    def usd_per_gpu_second(self, pool: str) -> float:
        return float(self.pool(pool)["usd_per_hour"]) / 3600.0


def from_dict(body: dict[str, Any]) -> Policy:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return Policy(body=body, version=hashlib.sha256(canonical.encode()).hexdigest()[:16])


def load(path: str | os.PathLike | None = None, *, strict: bool | None = None) -> Policy:
    """The policy file at `path` (MW_POLICY, else policy/default.yaml), checked against the schema of its hot
    sections (manyworlds.policy_schema): see `check` for what happens when it doesn't match."""
    path = Path(path or os.environ.get("MW_POLICY", "policy/default.yaml"))
    policy = from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
    check(policy, strict=strict, source=str(path))
    return policy


# Validation (2026-11-23). Policy.get falls back to code defaults, so a misspelt or missing key in a hot section would
# silently run on the code's number; the schema catches it. strict (tests, CI: MW_POLICY_STRICT=1) raises. Otherwise
# the problem is logged as an error once per policy version and the policy is used as it is: the services that load it
# include the autoscaler and the GPU enforcer, which stop paid machines, and must never stop working over a typo that
# CI (tests/test_policy_schema.py) already fails on.

log = logging.getLogger("manyworlds.policy")
_CHECKED: dict[str, list[str]] = {}


def strict_default() -> bool:
    return os.environ.get("MW_POLICY_STRICT", "").strip().lower() in ("1", "true", "yes")


def check(policy: Policy, *, strict: bool | None = None, source: str = "policy") -> list[str]:
    """The schema problems of `policy` (empty when it validates). strict (default: MW_POLICY_STRICT) raises
    policy_schema.PolicyInvalid on any; otherwise each version's problems are logged once."""
    from . import policy_schema
    strict = strict_default() if strict is None else strict
    problems = _CHECKED.get(policy.version)
    if problems is None:
        problems = _CHECKED[policy.version] = policy_schema.errors(policy.body)
        if problems and not strict:
            log.error("policy %s (%s) does not match its schema; running on it as it is:\n  %s",
                      source, policy.version, "\n  ".join(problems))
    if problems and strict:
        raise policy_schema.PolicyInvalid(f"policy {source} ({policy.version}) does not match its schema "
                                          "(manyworlds.policy_schema):\n  " + "\n  ".join(problems))
    return problems


def register(conn, policy: Policy) -> None:
    conn.execute(
        "INSERT INTO policies (version_hash, body) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (policy.version, Jsonb(policy.body)),
    )


# Overlays (2026-11-23, the breadth test): a small YAML file deep-merged over the policy for one purpose, e.g.
# policy/test_overlay.yaml for the breadth test's own posts. Nothing reads an overlay unless asked: the driver applies
# it to its posts, and services that meet a batch named by `operations.test_overlay.batches` may apply it to that
# batch's work (`for_batch`).

ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base: Any, over: Any) -> Any:
    """`over` merged into `base`: dicts key by key, everything else (lists included) replaced. An empty dict in
    `over` replaces (an overlay's `reroute_failed_video: {}` means "none", not "no change")."""
    if not isinstance(base, dict) or not isinstance(over, dict) or not over:
        return over
    out = dict(base)
    for key, value in over.items():
        out[key] = deep_merge(base[key], value) if key in base else value
    return out


def resolve(path: str | os.PathLike) -> Path:
    """An overlay's file: as given when it exists (absolute, or relative to the working directory), else relative
    to the repository root (a service started elsewhere still finds policy/...)."""
    p = Path(path)
    return p if p.is_absolute() or p.exists() else ROOT / p


_OVERLAYS: dict[tuple, Policy] = {}


def with_overlay(base: Policy, path: str | os.PathLike) -> Policy:
    """`base` with the overlay file at `path` deep-merged over it (a new version). Applying the same overlay twice
    gives the same policy."""
    p = resolve(path)
    import hashlib
    import json
    body = hashlib.sha256(json.dumps(base.body, sort_keys=True, default=str).encode()).hexdigest()
    key = (body, str(p), p.stat().st_mtime_ns)   # the body, not the version: a replaced body can keep its version
    if key not in _OVERLAYS:
        over = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(over, dict):
            raise ValueError(f"policy overlay {p} must be a mapping")
        # an empty overlay changes nothing (deep_merge's "an empty dict replaces" is for keys inside it)
        _OVERLAYS[key] = from_dict(deep_merge(base.body, over) if over else base.body)
    return _OVERLAYS[key]


def overlay_batches(policy: Policy) -> list[str]:
    """Batch prefixes whose work runs under the test overlay (operations.test_overlay.batches)."""
    return [str(b) for b in (policy.get("operations", "test_overlay", "batches", default=[]) or [])]


class OverlayMissing(RuntimeError):
    """Work in a test-overlay batch whose overlay file is gone. Running it under the plain policy would spend money
    the test never planned (2026-11-24: without the overlay's merged machine, three cold L40S boots instead of one),
    so callers refuse that work and tell the owner (deploy.autoscale) instead of falling back."""

    def __init__(self, path: str, batches: list[str]):
        self.path, self.batches = str(path), batches
        super().__init__(f"test overlay {path} is missing; refusing work in test batches {batches}")


def for_batches(policy: Policy, batches: list[str | None] | None) -> Policy:
    """The policy for work in these batches: with the test overlay (operations.test_overlay.path) when any of them
    starts with one of its prefixes, else `policy` itself. A missing overlay file raises OverlayMissing for such
    work: never a silent fall back to the plain policy."""
    path, prefixes = policy.get("operations", "test_overlay", "path"), overlay_batches(policy)
    tests = [str(b) for b in (batches or []) if b and str(b).startswith(tuple(prefixes))] if prefixes else []
    if not path or not tests:
        return policy
    try:
        return with_overlay(policy, path)
    except FileNotFoundError:
        log.error("test overlay %s is missing (operations.test_overlay.path): work in batches %s is refused",
                  path, tests)
        raise OverlayMissing(path, tests) from None


def for_batch(policy: Policy, batch: str | None) -> Policy:
    return for_batches(policy, [batch])
