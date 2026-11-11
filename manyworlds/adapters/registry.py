"""Capability registry and router.

The router picks, per interface, the enabled capability with the lowest expected cost
per accepted output:  cost per call / observed acceptance rate. Capabilities without
enough history get an optimistic prior so new models are still tried. Stats are filled
in by the review stage (day 4); until then the configured cost alone decides.
"""

from __future__ import annotations

from .base import Capability

MIN_SAMPLES = 20
PRIOR_ACCEPTANCE = 0.5


class Registry:
    def __init__(self):
        self._caps: dict[str, Capability] = {}
        self._stats: dict[str, dict] = {}

    def register(self, cap: Capability, *, stats: dict | None = None) -> None:
        self._caps[cap.name] = cap
        self._stats[cap.name] = stats or {}

    def for_interface(self, interface: str) -> list[Capability]:
        return [c for c in self._caps.values() if c.interface == interface]

    def expected_cost_per_accepted(self, name: str) -> float:
        cap, stats = self._caps[name], self._stats.get(name, {})
        cost = stats.get("usd_per_call", cap.hosted_usd_per_call or cap.config.get("est_usd_per_call", 0.0))
        n = stats.get("reviewed", 0)
        acceptance = stats.get("accepted", 0) / n if n >= MIN_SAMPLES else PRIOR_ACCEPTANCE
        return float(cost) / max(acceptance, 1e-3)

    def route(self, interface: str) -> Capability:
        options = self.for_interface(interface)
        if not options:
            raise LookupError(f"no capability registered for {interface}")
        return min(options, key=lambda c: self.expected_cost_per_accepted(c.name))
