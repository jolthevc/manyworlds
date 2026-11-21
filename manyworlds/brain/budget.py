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
    """This UTC day's LLM spend charged to `agent` (all sandbox agents together for a `sandbox:` tag)."""
    match = "sandbox:%" if agent.startswith("sandbox:") else agent
    row = conn.execute(
        "SELECT coalesce(sum(usd), 0) AS u FROM cost_entries WHERE agent LIKE %s AND category = 'llm' "
        "AND created_at >= date_trunc('day', now() AT TIME ZONE 'utc') AT TIME ZONE 'utc'", (match,)).fetchone()
    return float(row["u"])


def active_creators(conn) -> int:
    """Real creators the Brain works for: not retired, not sandbox test creators."""
    return int(conn.execute("SELECT count(*) AS n FROM creators WHERE status <> 'retired' "
                            "AND anchors->'simulation'->>'scenario' IS NULL").fetchone()["n"])


def daily_cap(policy: Policy, agent: str, conn=None) -> float:
    """The agent's fixed daily budget, plus `llm_usd_per_creator_day` for each active creator when the agent's
    work grows with the portfolio (the showrunner and evaluator run per creator: a fixed cap would stall them
    at a few dozen creators)."""
    caps = policy.get("brain", "llm_usd_per_day", default={}) or {}
    if agent.startswith("sandbox:"):
        return float(caps.get("sandbox", 0))
    base = float(caps.get(agent, caps.get("default", 0)))
    per = float((policy.get("brain", "llm_usd_per_creator_day", default={}) or {}).get(agent, 0))
    return base + (per * active_creators(conn) if per and conn is not None else 0)


def remaining(conn, policy: Policy, agent: str) -> float:
    return daily_cap(policy, agent, conn) - spent_today(conn, agent)


def check(conn, policy: Policy, agent: str) -> None:
    left = remaining(conn, policy, agent)
    if left <= 0:
        raise OverBudget(f"agent {agent} is over its daily LLM budget of ${daily_cap(policy, agent, conn):.2f} "
                         f"(spent ${spent_today(conn, agent):.2f})")


SANDBOX = "sandbox:"


def tag_for(conn, agent: str, creator_id) -> str:
    """The ledger tag and budget an agent's job is charged to: `sandbox:<agent>` for sandbox test creators
    (they have a scenario), so test runs share one sandbox budget and never use up real agents' budgets."""
    if creator_id is None:
        return agent
    row = conn.execute("SELECT anchors->'simulation'->>'scenario' AS s FROM creators WHERE id = %s",
                       (creator_id,)).fetchone()
    return f"{SANDBOX}{agent}" if row and row["s"] else agent
