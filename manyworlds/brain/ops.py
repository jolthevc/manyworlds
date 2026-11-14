"""Operator views of the Brain (`mw brain trace`, `mw brain approvals`)."""

from __future__ import annotations

from . import overrides


def trace(conn, job_id) -> dict:
    """A job, its decisions, the work it commissioned (recursively), and LLM spend per agent
    across the whole tree."""
    def node(jid) -> dict:
        job = conn.execute("SELECT id, kind, status, payload, result, error, parent_job_id, created_at, updated_at "
                           "FROM jobs WHERE id = %s", (jid,)).fetchone()
        decisions = conn.execute("SELECT id, kind, agent, step, inputs, output, rationale, model, policy_version, "
                                 "override_version, created_at FROM decisions WHERE job_id = %s ORDER BY created_at",
                                 (jid,)).fetchall()
        children = [node(c["id"]) for c in conn.execute(
            "SELECT id FROM jobs WHERE parent_job_id = %s ORDER BY created_at", (jid,)).fetchall()]
        return {"job": job, "decisions": decisions, "children": children}

    tree = node(job_id)
    spend = conn.execute(
        """WITH RECURSIVE t AS (SELECT id FROM jobs WHERE id = %s
                                UNION ALL SELECT j.id FROM jobs j JOIN t ON j.parent_job_id = t.id)
           SELECT agent, round(sum(usd)::numeric, 6)::float AS usd FROM cost_entries
           WHERE job_id IN (SELECT id FROM t) AND category = 'llm' AND agent IS NOT NULL GROUP BY agent""",
        (job_id,)).fetchall()
    return {**tree, "llm_usd_by_agent": {r["agent"]: r["usd"] for r in spend}}


def approvals(conn, *, status: str = "pending") -> list[dict]:
    return conn.execute("SELECT * FROM approvals WHERE status = %s ORDER BY created_at", (status,)).fetchall()


def decide(conn, approval_id, *, approved: bool, by: str) -> None:
    overrides.decide(conn, approval_id, approved=approved, by=by)
