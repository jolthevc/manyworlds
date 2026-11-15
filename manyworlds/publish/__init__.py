"""Publishing: one interface, a simulated implementation (sandbox creators) and Instagram (M4).

A publisher returns {external_id, permalink, ai_label, simulated}. The scheduler writes one
`publications` row per post either way; outcomes later land in `simulated_metrics` or
`metric_snapshots` with the same metric keys, so learning reads both the same way (through
manyworlds.brain.evidence, which keeps them apart).
"""

from __future__ import annotations

from typing import Protocol


class Publisher(Protocol):
    platform: str

    def publish(self, conn, policy, post: dict) -> dict: ...
