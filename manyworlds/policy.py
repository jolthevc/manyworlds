"""Versioned policy: every tunable number lives in a YAML file, never in code.

The version is a hash of the policy's canonical JSON, so any change produces a new
version, and every cost entry and decision can name the exact policy it ran under.
"""

from __future__ import annotations

import hashlib
import json
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

    def usd_per_gpu_second(self, pool: str) -> float:
        return float(self.pool(pool)["usd_per_hour"]) / 3600.0


def from_dict(body: dict[str, Any]) -> Policy:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return Policy(body=body, version=hashlib.sha256(canonical.encode()).hexdigest()[:16])


def load(path: str | os.PathLike | None = None) -> Policy:
    path = Path(path or os.environ.get("MW_POLICY", "policy/default.yaml"))
    return from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def register(conn, policy: Policy) -> None:
    conn.execute(
        "INSERT INTO policies (version_hash, body) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (policy.version, Jsonb(policy.body)),
    )
