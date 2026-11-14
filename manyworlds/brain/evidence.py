"""The only sanctioned reader of findings and metrics for Brain agents.

Simulated results test the machinery; real Instagram results are the only evidence of demand.
Every read names a mode: `real` never returns a simulated finding or metric, whatever the
creator's history; `sandbox` returns both and labels every simulated row "SIMULATED". The
strategist and allocator read through `for_allocation`, which is always real. A test fails if
any other module under manyworlds/brain reads these tables directly.
"""

from __future__ import annotations

import uuid
from typing import Literal

from psycopg.types.json import Jsonb

Mode = Literal["real", "sandbox"]
LABEL = "SIMULATED"
GRADES = ("proven", "tentative", "speculative")


def mode_for(conn, creator_id) -> Mode:
    row = conn.execute("SELECT publish_mode FROM creators WHERE id = %s", (creator_id,)).fetchone()
    return "real" if row and row["publish_mode"] == "instagram" else "sandbox"


def _label(row: dict) -> dict:
    return {**row, "label": LABEL} if row.get("evidence") == "simulated" else {**row, "label": None}


def findings(conn, *, mode: Mode, scope: str | None = None, creator_id=None, cluster_id=None, market_id=None,
             grades: tuple[str, ...] = GRADES) -> list[dict]:
    if mode not in ("real", "sandbox"):
        raise ValueError(f"unknown evidence mode {mode!r}")
    rows = conn.execute(
        """SELECT * FROM findings
           WHERE (%(scope)s::text IS NULL OR scope = %(scope)s)
             AND (%(creator)s::uuid IS NULL OR creator_id = %(creator)s)
             AND (%(cluster)s::uuid IS NULL OR cluster_id = %(cluster)s)
             AND (%(market)s::uuid IS NULL OR market_id = %(market)s)
             AND grade = ANY(%(grades)s)
             AND (%(mode)s = 'sandbox' OR evidence <> 'simulated')
           ORDER BY created_at""",
        {"scope": scope, "creator": creator_id, "cluster": cluster_id, "market": market_id,
         "grades": list(grades), "mode": mode}).fetchall()
    return [_label(r) for r in rows]


def for_allocation(conn, **filters) -> list[dict]:
    """Findings the strategist may act on with money: real evidence only, always."""
    filters.pop("mode", None)
    return findings(conn, mode="real", **filters)


def metrics(conn, post_id, *, mode: Mode) -> list[dict]:
    if mode == "real":
        return conn.execute("SELECT * FROM metric_snapshots WHERE post_id = %s ORDER BY captured_at",
                            (post_id,)).fetchall()
    if mode == "sandbox":
        return [{**r, "label": LABEL} for r in conn.execute(
            "SELECT * FROM simulated_metrics WHERE post_id = %s ORDER BY captured_at", (post_id,)).fetchall()]
    raise ValueError(f"unknown evidence mode {mode!r}")


def record_finding(conn, *, scope: str, statement: str, evidence: str, grade: str, creator_id=None, cluster_id=None,
                   market_id=None, effect=None, interval: tuple | None = None, n: int | None = None,
                   sources=()) -> uuid.UUID:
    """A finding about a sandbox creator, or drawn from one or from simulated metrics, can only be
    recorded as simulated: the label is checked here, not trusted, so nothing simulated can reach
    a real-mode read when the creator later goes live."""
    if evidence != "simulated":
        if creator_id and mode_for(conn, creator_id) == "sandbox":
            raise ValueError(f"creator {creator_id} is in the sandbox: its findings must be evidence='simulated'")
        for s in sources:
            if isinstance(s, dict) and s.get("table") == "simulated_metrics":
                raise ValueError("a finding drawn from simulated_metrics must be evidence='simulated'")
            if isinstance(s, dict) and s.get("creator_id") and mode_for(conn, s["creator_id"]) == "sandbox":
                raise ValueError(f"source creator {s['creator_id']} is in the sandbox: evidence must be 'simulated'")
    low, high = interval if interval else (None, None)
    return conn.execute(
        """INSERT INTO findings (scope, creator_id, cluster_id, market_id, statement, evidence, grade, effect,
                                 interval_low, interval_high, n, sources)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
        (scope, creator_id, cluster_id, market_id, statement, evidence, grade, effect, low, high, n,
         Jsonb(list(sources)))).fetchone()["id"]
