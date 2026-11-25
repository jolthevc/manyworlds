"""The portfolio map (code, no model): which archetypes, audience cells and commercial categories the portfolio
covers, where creators overlap, and the gaps worth casting into (owner 2026-11-25: "underexposed to sophisticated male
lifestyle 35-55 with premium apparel, travel, auto and accessories opportunity; photography can support it").

Per creator (portfolio creators only: no sandbox clones, auditions or retired ones):
    archetypes     the dossier's (else the bible's archetype)
    desire         the brand core's desire
    cell           the audience cell (strategy.portfolio_map: gender x age band) from the profile's audience, else
                   the creator's own presentation and apparent age
    categories     the brand commerce categories
Coverage: per archetype, per cell, per category, the handles. Overlaps: pairs sharing an archetype and a category,
with their positioning word overlap (manyworlds.creator.dossier.overlap). Gaps: library archetypes nobody covers;
cells nobody covers; categories nobody covers, ranked by commercial index x production fit (manyworlds.brain.commerce);
and the strategy's examples as casting prompts. `text()` is the CEO's view; `mw brain portfolio-map` prints `build()`.
"""

from __future__ import annotations

import re

from ..creator import dossier
from ..policy import Policy
from . import archetypes, commerce


def _cfg(policy: Policy) -> dict:
    return policy.get("strategy", "portfolio_map", default={}) or {}


def _gender(text: str | None) -> str | None:
    t = str(text or "").lower()
    if re.search(r"\b(women|woman|female|feminine|she|her)\b", t):
        return "female" if not re.search(r"\b(men|man|male|masculine)\b", t) else None
    if re.search(r"\b(men|man|male|masculine|he|him)\b", t):
        return "male"
    return None


def _ages(text) -> tuple[int, int] | None:
    nums = [int(n) for n in re.findall(r"\d{2}", str(text or ""))]
    if not nums:
        return None
    return (min(nums), max(nums)) if len(nums) > 1 else (nums[0], nums[0])


def cell_of(policy: Policy, prof: dict, bible: dict) -> list[str]:
    """The audience cells (gender age-band) a creator speaks to: the profile's audience (age range, gender), else
    her own presentation and apparent age. A mixed-gender audience covers both genders' cells."""
    cfg = _cfg(policy)
    genders = list(cfg.get("genders") or ["female", "male"])
    bands = [tuple(b) for b in cfg.get("age_bands") or [[18, 24], [25, 34], [35, 54], [55, 99]]]
    aud = (prof or {}).get("audience") or {}
    app = (bible or {}).get("appearance") or {}
    g = _gender(aud.get("gender"))
    gs = [g] if g else ([] if aud.get("gender") else [_gender(app.get("gender_presentation"))])
    if aud.get("gender") and not g:
        gs = genders                              # "all", "mixed", "everyone"
    gs = [x for x in gs if x]
    span = _ages(aud.get("age_range")) or ((int(app["apparent_age"]),) * 2 if app.get("apparent_age") else None)
    if not gs or not span:
        return []
    lo, hi = span
    cells = [f"{x} {a}-{b}" for x in gs for a, b in bands if a <= hi and lo <= b]
    return cells


def creators(conn, policy: Policy) -> list[dict]:
    rows = conn.execute(
        """SELECT c.id, c.handle, c.bible, c.vertical, coalesce(s.stage, 'exploring') AS stage, p.body AS prof
           FROM creators c LEFT JOIN creator_states s ON s.creator_id = c.id
           LEFT JOIN LATERAL (SELECT body FROM creator_profiles WHERE creator_id = c.id ORDER BY version DESC LIMIT 1) p
             ON true
           WHERE c.status <> 'retired' AND c.anchors->'simulation' IS NULL AND c.anchors->'audition' IS NULL
           ORDER BY c.created_at""").fetchall()
    out = []
    for r in rows:
        prof, bible = r["prof"] or {}, r["bible"] or {}
        names = list(prof.get("archetypes") or archetypes.for_creator(policy, prof or bible) or [])
        out.append({"creator_id": str(r["id"]), "handle": r["handle"], "vertical": r["vertical"], "stage": r["stage"],
                    "archetypes": names, "desire": (prof.get("brand_core") or {}).get("desire"),
                    "cells": cell_of(policy, prof, bible),
                    "categories": list((prof.get("brand_commerce") or {}).get("categories") or []), "_prof": prof})
    return out


def build(conn, policy: Policy) -> dict:
    cfg = _cfg(policy)
    people = creators(conn, policy)
    cats = commerce.categories(conn, policy)
    coverage = {"archetypes": {}, "cells": {}, "categories": {}}
    for c in people:
        for a in c["archetypes"]:
            coverage["archetypes"].setdefault(a, []).append(c["handle"])
        for x in c["cells"]:
            coverage["cells"].setdefault(x, []).append(c["handle"])
        for k in c["categories"]:
            coverage["categories"].setdefault(k, []).append(c["handle"])
    overlaps = []
    for i, a in enumerate(people):
        for b in people[i + 1:]:
            arch = sorted(set(a["archetypes"]) & set(b["archetypes"]))
            shared = sorted(set(a["categories"]) & set(b["categories"]))
            cells = sorted(set(a["cells"]) & set(b["cells"]))
            if arch and (shared or cells):
                overlaps.append({"a": a["handle"], "b": b["handle"], "archetypes": arch, "categories": shared,
                                 "cells": cells, "positioning_overlap": dossier.overlap(a["_prof"], b["_prof"])})
    genders = list(cfg.get("genders") or ["female", "male"])
    bands = [tuple(b) for b in cfg.get("age_bands") or [[18, 24], [25, 34], [35, 54], [55, 99]]]
    all_cells = [f"{g} {a}-{b}" for g in genders for a, b in bands]
    top = int(cfg.get("top_gaps", 5))
    cat_gaps = sorted((c for c in cats if c["key"] not in coverage["categories"]),
                      key=lambda c: (-(c["index"] * c["fit"]), c["key"]))
    gaps = {"archetypes": [a for a in archetypes.names(policy) if a not in coverage["archetypes"]],
            "cells": [x for x in all_cells if x not in coverage["cells"]],
            "categories": [{"key": c["key"], "name": c.get("name"), "index": c["index"], "production_fit": c["fit"],
                            "source": c.get("source")} for c in cat_gaps[:top]]}
    s = policy.get("strategy", default={}) or {}
    return {"focus": s.get("focus"), "strategy_version": s.get("version"),
            "creators": [{k: v for k, v in c.items() if k != "_prof"} for c in people],
            "coverage": coverage, "overlaps": overlaps, "gaps": gaps,
            "examples": list(s.get("examples") or []),
            "opportunities": opportunities(gaps, top)}


def opportunities(gaps: dict, top: int) -> list[str]:
    """Plain lines pairing uncovered audience cells with the best uncovered categories that production can render."""
    cats = [c for c in gaps.get("categories") or [] if c.get("production_fit", 0) >= 0.5]
    names = ", ".join(c.get("name") or c["key"] for c in cats[:top])
    return [f"underexposed to the {cell} audience" + (f"; open categories production can render: {names}" if names else "")
            for cell in (gaps.get("cells") or [])[:top]]


def text(m: dict) -> str:
    """The map in a few lines for the CEO's prompt."""
    lines = [f"Focus: {m.get('focus')} (strategy {m.get('strategy_version')})."]
    for c in m["creators"]:
        lines.append(f"- {c['handle']} [{c['stage']}]: archetypes {c['archetypes']}; audience {c['cells']}; "
                     f"categories {c['categories']}; desire: {c.get('desire') or '?'}")
    if m["overlaps"]:
        lines.append("Overlaps: " + "; ".join(f"{o['a']}~{o['b']} ({', '.join(o['archetypes'])}; positioning "
                                              f"{o['positioning_overlap']})" for o in m["overlaps"]))
    g = m["gaps"]
    lines.append(f"Uncovered audience cells: {g['cells']}")
    lines.append(f"Uncovered archetypes: {g['archetypes']}")
    lines.append("Best uncovered categories (index x production fit): "
                 + "; ".join(f"{c['key']} {c['index']}x{c['production_fit']}" for c in g["categories"]))
    if m.get("opportunities"):
        lines.append("Opportunities: " + " | ".join(m["opportunities"]))
    return "\n".join(lines)
