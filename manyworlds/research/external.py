"""The one switch for third-party creator research (owner 2026-11-27: discontinued; a strategic decision, not a problem
to work around). Policy `intelligence.external.enabled` false stops every path that collected or analysed other
creators: the scraper call itself (research.apify.run_actor), the provider router (the data policy blocks apify,
scrapecreators and Business Discovery), discovery and the monitor's schedules, the intelligence agent's external
modes, creator proposition reads and external gold sets. Web research, research questions and everything about
our own content are unaffected."""

from __future__ import annotations

from ..policy import Policy


class Discontinued(RuntimeError):
    """Third-party creator research was asked for while it is discontinued."""


EXTERNAL_MODES = ("scout", "annotate", "distill", "pilot", "commerce", "monitor", "resume")


def allowed(policy: Policy | None) -> bool:
    if policy is None:
        from .. import policy as policy_mod
        policy = policy_mod.load()
    return bool((policy.get("intelligence", "external", default={}) or {}).get("enabled", False))


def reason(policy: Policy | None) -> str:
    if policy is None:
        from .. import policy as policy_mod
        policy = policy_mod.load()
    return str((policy.get("intelligence", "external", default={}) or {}).get("reason")
               or "third-party creator research is discontinued")


def require(policy: Policy | None, what: str) -> None:
    if not allowed(policy):
        raise Discontinued(f"{what}: {reason(policy)}")
