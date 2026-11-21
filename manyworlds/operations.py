"""The master switch (owner, 2026-11-21): nothing runs on its own until the owner says so.

policy `operations.mode`:
  paused  (default) the producer plans, produces and publishes nothing; the Brain worker runs only jobs marked
          as tests (payload `test: true`); the autoscaler starts GPU machines only for test batches
          (`operations.test_batch_prefixes`, e.g. benchmarks). Safety enforcement (deadlines, caps, idle stops,
          the GitHub backstop) always runs.
  live    everything runs as designed (production, showrunner, strategist, launches within their own gates).
Only the owner sets `live`.
"""

from __future__ import annotations

from .policy import Policy


def mode(policy: Policy) -> str:
    return str(policy.get("operations", "mode", default="paused"))


def is_live(policy: Policy) -> bool:
    return mode(policy) == "live"


def test_batch_patterns(policy: Policy) -> list[str] | None:
    """SQL LIKE patterns of the batches GPU machines may serve now; None means any (live)."""
    if is_live(policy):
        return None
    return [f"{p}%" for p in policy.get("operations", "test_batch_prefixes", default=["bench-", "test-"]) or []]
