"""Gold sets: how far each coded attribute can be trusted (owner 2026-11-26: measured reliability before scale).

Several independent reads of the same items: the production annotator (iia_annotate), the same annotator again
(`repeat`: its self-consistency), reference models (policy intelligence.gold.reference_roles: Opus, Sonnet), code
measures where an attribute has one, and the owner's spot checks (method human). Opus is a reference annotator, not
ground truth: agreement is reported between every pair, and an attribute is called reliable only when the production
read agrees with the references (Krippendorff's alpha >= min_alpha) and with itself.

    alpha(units, level)          Krippendorff's alpha over {item: {coder: value}} (nominal, ordinal, interval).
    run_own(conn, policy, llm, store, workdir, n, usd)
                                 a gold run on our own accepted images (rights-clean): every read of every item into
                                 the spine (attribute_values, purpose production|repeat|gold), within max_usd_per_run.
    report(conn, policy)         agreement by attribute and by class (objective | subjective | inferred), per annotator
                                 pair; the attributes that fail (`unreliable`) and those with too few items
                                 (`untested`); recorded as an analysis_runs row (method gold_agreement).
    unreliable(conn)             the latest report's unreliable fields (manyworlds.brain.recipes leaves them out).

Multi-label fields (garments, environment...) are scored label by label as present/absent items, pooled per field.
Ordinal fields use their vocabulary order as ranks (an interval metric on ranks: close to Krippendorff's ordinal metric
for evenly used scales, and simpler to explain).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path

from psycopg.types.json import Jsonb

from ..policy import Policy
from . import annotation as ann

METHOD_VERSION = "gold-v1"


def cfg(policy: Policy) -> dict:
    return policy.get("intelligence", "gold", default={}) or {}


# --- the statistic --------------------------------------------------------------------------------------------------
def _delta(a, b, level: str) -> float:
    if level == "nominal":
        return 0.0 if a == b else 1.0
    return (float(a) - float(b)) ** 2


def alpha(units: dict, level: str = "nominal") -> float | None:
    """Krippendorff's alpha over {item: {coder: value}}; items with fewer than two values are not pairable. None when
    there is nothing to measure (no pairable item, or every value the same: no expected disagreement)."""
    values = [[v for v in coders.values() if v is not None] for coders in units.values()]
    values = [vs for vs in values if len(vs) >= 2]
    n = sum(len(vs) for vs in values)
    if n < 2:
        return None
    observed = 0.0
    for vs in values:
        m = len(vs)
        observed += sum(_delta(a, b, level) for a, b in itertools.permutations(vs, 2)) / (m - 1)
    observed /= n
    pooled = [v for vs in values for v in vs]
    expected = sum(_delta(a, b, level) for a, b in itertools.permutations(pooled, 2)) / (n * (n - 1))
    if expected == 0:
        return None
    return round(1.0 - observed / expected, 4)


# --- reading the spine ----------------------------------------------------------------------------------------------
def _coder(row) -> str:
    return f"{row['purpose']}:{row['annotator']}" if row["purpose"] != "production" else f"production:{row['annotator']}"


def _items(conn, unit_ids: list[str]) -> dict:
    """{attribute: {unit: {coder: [values]}}} over every read of these units."""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in conn.execute("SELECT unit_id, attribute, value, num, method, annotator, purpose FROM attribute_values "
                          "WHERE unit_id = ANY(%s::uuid[])", (unit_ids,)).fetchall():
        coder = _coder(r) if r["method"] != "code" else f"code:{r['annotator']}"
        if r["method"] == "human":
            coder = f"human:{r['annotator']}"
        out[r["attribute"]][str(r["unit_id"])][coder].append(r["num"] if r["num"] is not None else r["value"])
    return out


def _scored(attr: str, by_unit: dict, level: str, policy: Policy) -> dict:
    """{item key: {coder: value}} ready for alpha: one value per coder for single fields; label-presence items for
    multi-label fields; vocabulary ranks for ordinal ones. Unknown values are left out (not a disagreement)."""
    out: dict = {}
    if attr in ann.MULTI_FIELDS:
        labels = {v for coders in by_unit.values() for vs in coders.values() for v in vs
                  if v not in (None, "(none)", ann.UNKNOWN)}
        for unit, coders in by_unit.items():
            for label in labels:
                out[(unit, label)] = {c: (label in vs) for c, vs in coders.items()}
        return out
    order = {v: i for i, v in enumerate(ann.vocab(policy).get(attr) or [])}
    for unit, coders in by_unit.items():
        row = {}
        for c, vs in coders.items():
            v = vs[0] if vs else None
            if v in (None, "", ann.UNKNOWN):
                continue
            if level == "ordinal":
                if str(v) not in order:
                    continue
                v = order[str(v)]
            elif level == "interval":
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
            row[c] = v
        out[unit] = row
    return out


def _pair(units: dict, a: str, b: str) -> dict:
    return {k: {a: v[a], b: v[b]} for k, v in units.items() if a in v and b in v}


def report(conn, policy: Policy, *, unit_ids: list[str] | None = None, requester: str = "gold") -> dict:
    """Agreement by attribute and class over the units that have gold reads (or `unit_ids`); stored as an
    analysis_runs row. `production` is the production annotator; each other coder is compared with it, and every
    reference with every other."""
    c = cfg(policy)
    registry = policy.get("intelligence", "attributes", default={}) or {}
    if unit_ids is None:
        unit_ids = [str(r["unit_id"]) for r in conn.execute(
            "SELECT DISTINCT unit_id FROM attribute_values WHERE purpose IN ('gold', 'repeat', 'spot_check')"
        ).fetchall()]
    items = _items(conn, unit_ids)
    floor, min_items = float(c.get("min_alpha", 0.67)), int(c.get("min_items", 30))
    attrs, unreliable_, untested = {}, [], []
    for attr, by_unit in sorted(items.items()):
        meta = registry.get(attr) or {}
        level = str(meta.get("level") or "nominal")
        units = _scored(attr, by_unit, level, policy)
        coders = sorted({cd for v in units.values() for cd in v})
        prod = next((cd for cd in coders if cd.startswith("production:")), None)
        pairs = {}
        for a, b in itertools.combinations(coders, 2):
            sub = _pair(units, a, b)
            pairs[f"{a} ~ {b}"] = {"alpha": alpha(sub, level), "items": len(sub)}
        refs = [cd for cd in coders if cd.startswith(("gold:", "human:", "code:"))]
        vs_ref = [pairs.get(f"{a} ~ {b}") or pairs.get(f"{b} ~ {a}") for a, b in
                  ((prod, r) for r in refs)] if prod else []
        vs_ref = [p for p in vs_ref if p and p["alpha"] is not None and p["items"] >= min_items]
        rep = next((pairs.get(f"{a} ~ {b}") or pairs.get(f"{b} ~ {a}") for a, b in
                    ((prod, cd) for cd in coders if cd.startswith("repeat:"))), None) if prod else None
        overall = alpha(units, level)
        verdict = "untested"
        if vs_ref:
            best = max(p["alpha"] for p in vs_ref)
            self_ok = rep is None or rep["alpha"] is None or rep["items"] < min_items or rep["alpha"] >= floor
            verdict = "reliable" if best >= floor and self_ok else "unreliable"
        (unreliable_ if verdict == "unreliable" else untested if verdict == "untested" else []).append(attr)
        attrs[attr] = {"class": meta.get("class"), "tier": meta.get("tier"), "level": level, "alpha_all": overall,
                       "vs_reference": vs_ref, "repeat": rep, "pairs": pairs, "verdict": verdict,
                       "items": len(units)}
    by_class: dict = defaultdict(list)
    for attr, r in attrs.items():
        a = max((p["alpha"] for p in r["vs_reference"]), default=None)
        if a is not None:
            by_class[r["class"] or "unclassed"].append(a)
    classes = {k: {"median_alpha": sorted(v)[len(v) // 2], "attributes": len(v)} for k, v in by_class.items()}
    result = {"units": len(unit_ids), "attributes": attrs, "by_class": classes, "unreliable": sorted(unreliable_),
              "untested": sorted(untested), "min_alpha": floor, "min_items": min_items}
    conn.execute("INSERT INTO analysis_runs (requester, method, method_version, spec, family, family_size, data, "
                 "result, policy_version) VALUES (%s, 'gold_agreement', %s, %s, 'gold', 1, %s, %s, %s)",
                 (requester, METHOD_VERSION, Jsonb({"min_alpha": floor, "min_items": min_items}),
                  Jsonb({"units": len(unit_ids)}), Jsonb(result), policy.version))
    return result


def unreliable(conn) -> set[str]:
    row = conn.execute("SELECT result FROM analysis_runs WHERE method = 'gold_agreement' "
                       "ORDER BY created_at DESC, id DESC LIMIT 1").fetchone()
    return set((row["result"] or {}).get("unreliable") or []) if row else set()


# --- a gold run on our own accepted images ------------------------------------------------------------------------
def own_items(conn, n: int) -> list[dict]:
    """Accepted own image assets (every review passed and confirmed or a clear pass; never set plates), newest first,
    spread across posts (one per post first)."""
    rows = [dict(r) for r in conn.execute(
        """SELECT DISTINCT ON (j.post_id) a.id, a.uri, a.kind, a.job_id, j.post_id, p.format,
                  p.body->>'caption' AS caption, a.created_at
           FROM assets a JOIN jobs j ON j.id = a.job_id JOIN posts p ON p.id = j.post_id
           WHERE a.kind = 'image' AND a.deleted_at IS NULL
             AND coalesce(j.payload->>'plate', '') NOT IN ('true')
             AND EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id = a.id)
             AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id = a.id AND NOT r.passed)
             AND EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id = a.id
                         AND (r.details->>'stage' = 'confirm' OR r.details->>'clear_pass' = 'true'))
           ORDER BY j.post_id, a.created_at DESC""").fetchall()]
    rows.sort(key=lambda r: r["created_at"], reverse=True)
    return rows[:n]


def _spent(conn, agent: str, since) -> float:
    return float(conn.execute("SELECT coalesce(sum(usd), 0) u FROM cost_entries WHERE agent = %s AND created_at >= %s",
                              (agent, since)).fetchone()["u"])


def run_own(conn, policy: Policy, llm, store, workdir: Path, *, n: int | None = None, usd: float | None = None) -> dict:
    """Every gold read (production, repeat, each reference role) of up to n own accepted images, batched like the
    research read, within the run cap and the monthly research ceiling; then the agreement report."""
    from datetime import datetime, timezone

    from ..brain.agents.intelligence import ReadBatch, read_system
    from ..llm import image_block
    from ..review import own_annotation as own
    from . import budget, spine
    c = cfg(policy)
    agent = str(c.get("agent", "research:gold"))
    cap = min(float(usd if usd is not None else c.get("max_usd_per_run", 15.0)), budget.left(conn, policy))
    started = datetime.now(timezone.utc)
    items = own_items(conn, int(n or c.get("own_assets", 60)))
    ocfg = policy.get("performance", "own_annotation", default={}) or {}
    per_call = max(1, int(ocfg.get("assets_per_call", 4)))
    prod_role = str(ocfg.get("role", "iia_annotate"))
    passes = [("production", prod_role)] + ([("repeat", prod_role)] if c.get("repeat", True) else []) + \
             [("gold", str(r)) for r in c.get("reference_roles") or []]
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    units, calls, stopped = {}, 0, None
    for a in items:
        post = spine.upsert_unit(conn, source="own", level="post", ref_table="posts", ref_id=a["post_id"],
                                 platform="own", format=a.get("format"))
        units[str(a["id"])] = spine.upsert_unit(conn, source="own", level="shot", ref_table="assets", ref_id=a["id"],
                                                parent_id=post, platform="own", medium="photo")
    for purpose, role in passes:
        for i in range(0, len(items), per_call):
            if _spent(conn, agent, started) >= cap:
                stopped = f"run cap ${cap:.2f} reached"
                break
            chunk = items[i:i + per_call]
            blocks: list[dict] = [{"type": "text", "text": (
                f"{own.OWN_NOTE}Annotate each of these {len(chunk)} posts (numbered from 1) into the schema, "
                f"observed.coded from the coded lists.")}]
            seen = []
            for j, a in enumerate(chunk, 1):
                pics = own._images(conn, store, a, workdir, ocfg)
                if not pics:
                    continue
                blocks.append({"type": "text", "text": f"Post {j}: photo; caption: "
                                                       f"{(a.get('caption') or '')[:600]!r}"})
                blocks += [image_block(b, own._media_type(b)) for b in pics]
                seen.append((j, a))
            if not seen:
                continue
            with llm.tagged(agent, None):
                got = llm.structured(role, system=own.OWN_NOTE + read_system(policy), content=blocks,
                                     schema=ReadBatch, effort="low")
            calls += 1
            reads = {r.post: r.annotation for r in (getattr(got, "reads", None) or [])
                     if isinstance(getattr(r, "post", None), int)}
            model = llm.model_for(role)
            for j, a in seen:
                read = reads.get(j)
                if read is None:
                    continue
                coded = getattr(read.observed, "coded", None)
                raw = coded.model_dump(mode="json") if hasattr(coded, "model_dump") else (coded or {})
                clean, _ = ann.coerce_coded(raw, ann.vocab(policy))
                clean = clean.model_dump(mode="json") if hasattr(clean, "model_dump") else clean
                spine.put_read(conn, units[str(a["id"])], {"coded": clean, "vocab_versions": ann.vocab_versions(policy)},
                               medium="photo", annotator=f"{role}:{model}", purpose=purpose)
                if purpose == "production":
                    own.store_read(conn, policy, a, read, model)
        if stopped:
            break
    out = report(conn, policy, unit_ids=list(units.values()), requester="gold:own")
    return {"items": len(items), "calls": calls, "spent_usd": round(_spent(conn, agent, started), 4),
            "cap_usd": round(cap, 2), "stopped": stopped, "passes": [p for p, _ in passes], "report": out}


def spot_check(conn, policy: Policy, unit_id: str, labels: dict[str, str], who: str = "owner") -> int:
    """The owner's labels for a unit (attribute -> value; a multi-label field as comma-separated labels), stored as
    method human, purpose spot_check: they join every agreement report as a coder."""
    from . import spine
    n = 0
    for attr, raw in labels.items():
        vals = [v.strip() for v in str(raw).split(",")] if attr in ann.MULTI_FIELDS else [str(raw).strip()]
        conn.execute("DELETE FROM attribute_values WHERE unit_id = %s AND attribute = %s AND method = 'human' "
                     "AND annotator = %s", (unit_id, attr, who))
        for v in vals:
            spine.put_value(conn, unit_id, attr, value=v, method="human", annotator=who, purpose="spot_check")
            n += 1
    return n
