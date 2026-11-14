"""Per-agent daily LLM budgets (policy `brain.llm_usd_per_day`), from the ledger's agent tag.

The runtime checks an agent's budget before running its job; an agent past it fails the job
as non-retryable, so a busy or looping agent stops instead of spending on.
"""

from __future__ import annotations

from ..policy import Policy
from ..worker.runner import NonRetryable


class OverBudget(NonRetryable):
    pass


def spent_today(conn, agent: str) -> float:
    """This UTC day's LLM spend charged to `agent`."""
    row = conn.execute(
        "SELECT coalesce(sum(usd), 0) AS u FROM cost_entries WHERE agent = %s AND category = 'llm' "
        "AND created_at >= date_trunc('day', now() AT TIME ZONE 'utc') AT TIME ZONE 'utc'", (agent,)).fetchone()
    return float(row["u"])


def daily_cap(policy: Policy, agent: str) -> float:
    caps = policy.get("brain", "llm_usd_per_day", default={}) or {}
    return float(caps.get(agent, caps.get("default", 0)))


def remaining(conn, policy: Policy, agent: str) -> float:
    return daily_cap(policy, agent) - spent_today(conn, agent)


def check(conn, policy: Policy, agent: str) -> None:
    left = remaining(conn, policy, agent)
    if left <= 0:
        raise OverBudget(f"agent {agent} is over its daily LLM budget of ${daily_cap(policy, agent):.2f} "
                         f"(spent ${spent_today(conn, agent):.2f})")
