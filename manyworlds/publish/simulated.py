"""Simulated publishing for sandbox creators: nothing leaves the system. The post is marked
`simulated` (never `published`) and its outcomes come from the scenario engine
(manyworlds.brain.simulation). An engineering test of the loop, not audience evidence."""

from __future__ import annotations


class SimulatedPublisher:
    platform = "simulated"

    def publish(self, conn, policy, post: dict) -> dict:
        return {"external_id": f"sim-{post['id']}", "permalink": None,
                "ai_label": bool((post.get("body") or {}).get("platform_ai_label", True)), "simulated": True}
