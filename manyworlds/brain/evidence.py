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
             grades: tuple[str, ...] = GRADES, statuses: tuple[str, ...] = ("open",)) -> list[dict]:
    if mode not in ("real", "sandbox"):
        raise ValueError(f"unknown evidence mode {mode!r}")
    filters = """ (%(scope)s::text IS NULL OR scope = %(scope)s)
             AND (%(creator)s::uuid IS NULL OR creator_id = %(creator)s)
             AND (%(cluster)s::uuid IS NULL OR cluster_id = %(cluster)s)
             AND (%(market)s::uuid IS NULL OR market_id = %(market)s)
             AND grade = ANY(%(grades)s)
             AND status = ANY(%(statuses)s)
"""
    params = {"scope": scope, "creator": creator_id, "cluster": cluster_id, "market": market_id,
              "grades": list(grades), "statuses": list(statuses)}
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


def post_outcomes(conn, creator_id, n: int, mode: Mode, *, policy=None) -> dict:
    """How the creator's last `n` published posts did: {post_id (str): {"reach": ..., "age_h": ..., ...}}, read at
    the analysis target age (policy brain.analysis.target_age_s; the first snapshot at or after it, as the analyst
    and the predictions ledger read it: a young post's reach would read as failure). Posts not yet that old are
    left out. `real` reads real publications and metric_snapshots only; `sandbox` reads the simulated ones, each
    outcome labelled SIMULATED. The numbers are the snapshot's own (reach, plus saves, shares, likes, comments and
    views where the platform gave them)."""
    if mode not in ("real", "sandbox"):
        raise ValueError(f"unknown evidence mode {mode!r}")
    target = int((policy.get("brain", "analysis", "target_age_s", default=259200) if policy else 259200))
    posts = conn.execute(
        "SELECT DISTINCT ON (p.id) p.id, u.published_at FROM posts p JOIN publications u ON u.post_id = p.id "
        "WHERE p.creator_id = %s AND u.simulated = %s ORDER BY p.id, u.published_at DESC",
        (creator_id, mode == "sandbox")).fetchall()
    posts = sorted(posts, key=lambda r: r["published_at"], reverse=True)[:int(n)]
    keys = ("reach", "saves", "shares", "likes", "comments", "views")
    out: dict = {}
    for p in posts:
        rows = [r for r in metrics(conn, p["id"], mode=mode) if (r.get("post_age_s") or 0) >= target]
        if not rows:
            continue
        first = min(rows, key=lambda r: r.get("post_age_s") or 0)
        m = first.get("metrics") or {}
        o = {k: m[k] for k in keys if m.get(k) is not None}
        if "reach" not in o:
            continue
        o["age_h"] = round(int(first.get("post_age_s") or 0) / 3600)
        if mode == "sandbox":
            o["label"] = LABEL
        out[str(p["id"])] = o
    return out


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


def supersede(conn, *, creator_id) -> int:
    """Close a creator's open creator-level analysis findings (both tables) when a new analysis replaces
    them. Proven findings (experiments) and findings at other scopes are left alone."""
    n = 0
    for table in ("findings", "simulated_findings"):
        n += conn.execute(f"UPDATE {table} SET status = 'superseded', updated_at = now() "
                          "WHERE creator_id = %s AND status = 'open' AND scope = 'creator' AND grade <> 'proven'",
                          (creator_id,)).rowcount
    return n


def niche_trends(conn, creator_id, policy) -> dict | None:
    """What is working in the creator's niche on public posts (the trend watcher's latest brief and its findings).
    Research evidence: correlation in other people's posts, never proof; the same in real and sandbox modes."""
    c = conn.execute("SELECT c.bible->>'niche' AS niche, m.name AS market FROM creators c "
                     "LEFT JOIN markets m ON m.id = c.market_id WHERE c.id = %s", (creator_id,)).fetchone()
    names = [n for n in ((c or {}).get("market"), (c or {}).get("niche"),
                         (policy.get("trends", "default_market") if policy else None)) if n]
    for name in names:
        row = conn.execute("SELECT day, brief FROM trend_briefs WHERE market = %s ORDER BY day DESC LIMIT 1",
                           (name,)).fetchone()
        if row:
            b = row["brief"]
            return {"niche": name, "as_of": str(row["day"]), "working": [p["statement"] for p in b.get("working", [])],
                    "rising": b.get("rising", []), "cooling": b.get("cooling", []),
                    "hooks_to_try": b.get("hooks_to_try", []), "avoid": b.get("avoid", [])}
    return None
