"""Commercial intelligence, a capability (not an agent): the consumer categories our talent can be valuable in.

The registry is the `commercial_categories` table: per category its market size, advertiser density, influencer-
marketing activity, virtual-creator sponsorship precedent, production fit (can we render it well: apparel yes,
watches maybe, cars hard), affiliate and licensing fit, as bands. It is seeded from policy (strategy.commercial.seed,
general knowledge, source `estimate`) and refreshed by the intelligence agent's `commerce` mode (source `research`;
a seed never overwrites research). Casting, the talent manager and the CEO read it through this module.

    index(policy, row)        0-1: the bands' values (strategy.commercial.band_values) weighted (weights)
    categories(conn, policy)  every category, with its index, best first (seeded on first read)
    for_keys(conn, policy, keys)
    brief(rows)               compact lines for a prompt
"""

from __future__ import annotations

from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from ..policy import Policy

BANDS = ("market_size", "advertiser_density", "influencer_activity", "virtual_precedent", "production_fit",
         "affiliate_fit", "licensing_fit")


def _cfg(policy: Policy) -> dict:
    return policy.get("strategy", "commercial", default={}) or {}


def seed(conn, policy: Policy) -> int:
    """Insert or update the policy seed's categories; a row research has refreshed keeps its fields."""
    n = 0
    for c in _cfg(policy).get("seed") or []:
        if not c.get("key"):
            continue
        n += conn.execute(
            f"""INSERT INTO commercial_categories (key, name, {", ".join(BANDS)}, hashtags, notes, source)
                VALUES (%s, %s, {", ".join(["%s"] * len(BANDS))}, %s, %s, 'estimate')
                ON CONFLICT (key) DO UPDATE SET name = excluded.name,
                  {", ".join(f"{b} = excluded.{b}" for b in BANDS)}, hashtags = excluded.hashtags, notes = excluded.notes
                WHERE commercial_categories.source = 'estimate'""",
            (c["key"], c.get("name") or c["key"], *[c.get(b) for b in BANDS], Jsonb(list(c.get("hashtags") or [])),
             c.get("notes"))).rowcount
    return n


def index(policy: Policy, row: dict) -> float:
    """0-1 commercial index of a category from its bands (an unknown band counts 0)."""
    cfg = _cfg(policy)
    values, weights = cfg.get("band_values") or {}, cfg.get("weights") or {}
    total = sum(float(w) for w in weights.values()) or 1.0
    return round(sum(float(w) * float((values.get(b) or {}).get(str(row.get(b)), 0.0)) for b, w in weights.items())
                 / total, 3)


def production_fit(policy: Policy, row: dict) -> float:
    return float(((_cfg(policy).get("band_values") or {}).get("production_fit") or {}).get(
        str(row.get("production_fit")), 0.0))


def categories(conn, policy: Policy) -> list[dict]:
    """Every category with its index, best first. An empty registry is seeded from policy first."""
    if not conn.execute("SELECT 1 FROM commercial_categories LIMIT 1").fetchone():
        seed(conn, policy)
    rows = [dict(r) for r in conn.execute("SELECT * FROM commercial_categories ORDER BY key").fetchall()]
    for r in rows:
        r["index"] = index(policy, r)
        r["fit"] = production_fit(policy, r)
    return sorted(rows, key=lambda r: (-r["index"], r["key"]))


def keys(conn, policy: Policy) -> list[str]:
    return [r["key"] for r in categories(conn, policy)]


def for_keys(conn, policy: Policy, wanted) -> list[dict]:
    wanted = [str(k) for k in wanted or []]
    by = {r["key"]: r for r in categories(conn, policy)}
    return [by[k] for k in wanted if k in by]


def mean_index(conn, policy: Policy, wanted) -> float | None:
    """0-10: the mean index of the known categories among `wanted`; None when none is known."""
    rows = for_keys(conn, policy, wanted)
    return round(10 * sum(r["index"] for r in rows) / len(rows), 2) if rows else None


def brief(rows: list[dict], *, n: int | None = None) -> str:
    """Compact lines: key, name, bands, index, source."""
    lines = []
    for r in rows[:n] if n else rows:
        bands = ", ".join(f"{b.replace('_', ' ')} {r.get(b)}" for b in BANDS if r.get(b))
        lines.append(f"- {r['key']} ({r.get('name')}): {bands}; index {r.get('index')} [{r.get('source')}]")
    return "\n".join(lines)


def record_research(conn, key: str, *, fields: dict, evidence: dict) -> bool:
    """Research findings for a category (the intelligence agent's commerce mode): the given band fields replace
    the estimate's, the evidence is merged, source becomes `research`."""
    fields = {k: v for k, v in (fields or {}).items() if k in BANDS}
    sets = "".join(f", {k} = %s" for k in fields)
    row = conn.execute(
        f"UPDATE commercial_categories SET source = 'research', evidence = evidence || %s, refreshed_at = %s{sets} "
        "WHERE key = %s RETURNING key",
        (Jsonb(evidence), datetime.now(timezone.utc), *fields.values(), key)).fetchone()
    return row is not None


def adjacent(policy: Policy, texts) -> dict[str, list[str]]:
    """Categories a creator's own details touch (strategy.commercial.adjacency): {category key: [matched words]},
    most matches first. The input is any text or list of texts (personal style, world details, wardrobe)."""
    import json
    import re
    words = set(re.findall(r"[a-z]+", json.dumps(texts, default=str).lower()))
    out = {k: sorted(set(w) & words) for k, w in (_cfg(policy).get("adjacency") or {}).items()}
    return dict(sorted(((k, v) for k, v in out.items() if v), key=lambda kv: (-len(kv[1]), kv[0])))


def stale(conn, policy: Policy, *, max_age_days: float, limit: int) -> list[dict]:
    """Categories never researched or researched longer ago than max_age_days, oldest first."""
    categories(conn, policy)
    return [dict(r) for r in conn.execute(
        "SELECT * FROM commercial_categories WHERE source = 'estimate' OR refreshed_at < now() - make_interval(days => %s) "
        "ORDER BY (source = 'research'), refreshed_at LIMIT %s", (int(max_age_days), int(limit))).fetchall()]
