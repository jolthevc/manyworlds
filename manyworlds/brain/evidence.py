"""The only sanctioned reader of findings and metrics for Brain agents.

Simulated results test the machinery; real Instagram results are the only evidence of demand.
The wall is structural: simulated findings and metrics live only in `simulated_findings` and
`simulated_metrics`, and the real tables reject simulated rows (CHECK constraints), so no query
on the real tables can see them. Only this module touches the simulated tables (a test enforces
it). Every read names a mode: `real` reads the real tables only; `sandbox` adds the simulated
ones and labels every simulated row "SIMULATED" (metrics in sandbox mode are the simulated ones). The
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
    filters = """ (%(scope)s::text IS NULL OR scope = %(scope)s)
             AND (%(creator)s::uuid IS NULL OR creator_id = %(creator)s)
             AND (%(cluster)s::uuid IS NULL OR cluster_id = %(cluster)s)
             AND (%(market)s::uuid IS NULL OR market_id = %(market)s)
             AND grade = ANY(%(grades)s)
"""
    params = {"scope": scope, "creator": creator_id, "cluster": cluster_id, "market": market_id,
              "grades": list(grades)}
    sql = f"SELECT * FROM findings WHERE {filters}"
    if mode == "sandbox":
        sql += f" UNION ALL SELECT * FROM simulated_findings WHERE {filters}"
    rows = conn.execute(f"SELECT * FROM ({sql}) f ORDER BY created_at", params).fetchall()
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
    table = "simulated_findings" if evidence == "simulated" else "findings"
    return conn.execute(
        f"""INSERT INTO {table} (scope, creator_id, cluster_id, market_id, statement, evidence, grade, effect,
                                 interval_low, interval_high, n, sources)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
        (scope, creator_id, cluster_id, market_id, statement, evidence, grade, effect, low, high, n,
         Jsonb(list(sources)))).fetchone()["id"]


def record_simulated_metrics(conn, post_id, rows: list[dict], *, model: str, panel_version: str) -> None:
    """Outcomes from the scenario engine; captured_at is the publication time plus each age."""
    pub = conn.execute("SELECT coalesce(published_at, now()) AS t FROM posts WHERE id = %s", (post_id,)).fetchone()["t"]
    with conn.transaction():
        for r in rows:
            conn.execute("INSERT INTO simulated_metrics (post_id, captured_at, post_age_s, model, panel_version, metrics) "
                         "VALUES (%s, %s + make_interval(secs => %s), %s, %s, %s, %s)",
                         (post_id, pub, r["post_age_s"], r["post_age_s"], model, panel_version,
                          Jsonb(r["metrics"])))   # never the scenario's hidden truth (e.g. which posts broke out)
