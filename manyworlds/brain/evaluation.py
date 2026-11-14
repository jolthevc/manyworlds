"""Creative scoring: an item is scored `samples` times against a rubric's anchors, and the median
and spread per dimension are kept, so scores are comparable over time and noisy dimensions show.

Scores are integers clamped to 1-10. A dimension the model leaves out of a sample is skipped for
that sample; one no sample scored is stored as null, never invented. A deterministic share of
items (`checked`) is also scored by the strong model to measure how far the cheaper evaluator
agrees with it.
"""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, create_model

from ..policy import Policy
from .rubrics import Rubric

SYSTEM = """You are an independent creative evaluator for a studio of openly AI-generated, fictional
Instagram creators. You score posts against a fixed rubric so scores can be compared over months.
Use the anchors literally: odd scores are the anchor descriptions, even scores sit between them;
5 is an ordinary, competent post. Score every dimension independently, and be calibrated rather
than generous: most posts are 4-6 on most dimensions. Ordinary everyday posts are legitimate; low
novelty or curiosity is not a flaw in a quiet personal moment."""


class DimScore(BaseModel):
    score: int = Field(description="1-10, using the anchors")
    why: str = Field(description="one short sentence")


_schemas: dict[tuple[str, int], type[BaseModel]] = {}


def schema_for(r: Rubric) -> type[BaseModel]:
    key = (r.rubric, r.version)
    if key not in _schemas:
        fields: dict[str, Any] = {d.key: (DimScore | None, Field(default=None, description=d.name))
                                  for d in r.dimensions}
        fields["overall_note"] = (str, Field(default="", description="one sentence on the item overall"))
        _schemas[key] = create_model(f"{r.rubric.title()}V{r.version}Scores", **fields)
    return _schemas[key]


@dataclass
class Scored:
    medians: dict[str, float | None]
    spreads: dict[str, int | None]
    rationale: dict[str, list[str]]
    model: str
    samples: int
    notes: list[str] = field(default_factory=list)


def _blocks(content: list) -> list[dict]:
    return [{"type": "text", "text": c} if isinstance(c, str) else c for c in content]


def score(llm, policy: Policy, r: Rubric, content: list, *, role: str, samples: int, creator_id=None) -> Scored:
    schema = schema_for(r)
    prompt = [{"type": "text", "text": r.as_text()}, *_blocks(content),
              {"type": "text", "text": f"Score every dimension of the {r.rubric} rubric for this item."}]
    per_dim: dict[str, list[int]] = {d.key: [] for d in r.dimensions}
    why: dict[str, list[str]] = {d.key: [] for d in r.dimensions}
    notes: list[str] = []
    for _ in range(samples):
        out = llm.structured(role, system=SYSTEM, content=prompt, schema=schema, effort="low", creator_id=creator_id)
        for d in r.dimensions:
            s = getattr(out, d.key, None)
            if s is None or s.score is None:
                continue
            per_dim[d.key].append(max(1, min(10, int(s.score))))
            why[d.key].append(s.why)
        if getattr(out, "overall_note", ""):
            notes.append(out.overall_note)
    return Scored(medians={k: (statistics.median(v) if v else None) for k, v in per_dim.items()},
                  spreads={k: (max(v) - min(v) if v else None) for k, v in per_dim.items()},
                  rationale=why, model=llm.model_for(role), samples=samples, notes=notes)


def checked(item_id, rate: float) -> bool:
    """Whether this item also gets a strong-model score: the same answer for the same item every time."""
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    return int(hashlib.sha256(str(item_id).encode()).hexdigest()[:8], 16) / 2**32 < rate


def latest(conn, *, idea_id=None, asset_id=None, model: str | None = None) -> dict | None:
    """The most recent evaluation of an idea or asset (optionally by one evaluator model)."""
    return conn.execute(
        """SELECT * FROM evaluations
           WHERE (%(idea)s::uuid IS NULL OR idea_id = %(idea)s) AND (%(asset)s::uuid IS NULL OR asset_id = %(asset)s)
             AND (%(model)s::text IS NULL OR evaluator_model = %(model)s)
           ORDER BY created_at DESC LIMIT 1""", {"idea": idea_id, "asset": asset_id, "model": model}).fetchone()


def reference_items(rubric_name: str) -> list[dict]:
    """The fixed reference set for a rubric (policy/rubrics/reference/<rubric>.yaml)."""
    import yaml

    from .rubrics import ROOT
    body = yaml.safe_load((ROOT / "reference" / f"{rubric_name}.yaml").read_text(encoding="utf-8"))
    return [{**item, "creator": body.get("creator", "")} for item in body["items"]]


def score_reference(conn, policy: Policy, llm, rubric_name: str) -> dict[str, dict[str, float | None]]:
    """Score the reference set under the current rubric and evaluator model, and store the result on
    the rubric version, so the next version or model can be compared item by item."""
    from psycopg.types.json import Jsonb

    from . import rubrics
    r = rubrics.load(rubric_name)
    rubrics.register(conn, r)
    samples = int(policy.get("brain", "evaluation", "samples", default=3))
    scores = {}
    for item in reference_items(rubric_name):
        s = score(llm, policy, r, [f"THE CREATOR: {item['creator']}", f"THE IDEA: {item['idea']}"],
                  role="evaluator", samples=samples)
        scores[item["id"]] = s.medians
    conn.execute("UPDATE rubric_versions SET reference = %s WHERE rubric = %s AND version = %s",
                 (Jsonb({"model": llm.model_for("evaluator"), "samples": samples, "scores": scores}),
                  r.rubric, r.version))
    return scores


def drift(before: dict, after: dict) -> dict[str, float]:
    """Mean absolute shift per dimension over the reference items both runs scored."""
    shifts: dict[str, list[float]] = {}
    for item in set(before) & set(after):
        for dim, b in before[item].items():
            a = after[item].get(dim)
            if a is not None and b is not None:
                shifts.setdefault(dim, []).append(abs(a - b))
    return {d: round(statistics.mean(v), 3) for d, v in shifts.items()}


def calibrate(conn, policy: Policy, llm, creator_id, *, posts: int) -> dict:
    """Score a creator's latest planned/produced posts as ideas and report how noisy each dimension is:
    the median spread across samples, and the share of items within `brain.evaluation.max_spread`."""
    from . import ideas
    from .agents import evaluator
    max_spread = float(policy.get("brain", "evaluation", "max_spread", default=2))
    rows = conn.execute("SELECT id FROM posts WHERE creator_id = %s AND status <> 'dropped' "
                        "ORDER BY coalesce(scheduled_for, created_at) DESC LIMIT %s", (creator_id, posts)).fetchall()
    spreads: dict[str, list[int]] = {}
    for row in rows:
        idea_id = ideas.from_post(conn, row["id"])
        cid, fmt, content = evaluator.idea_content(conn, idea_id)
        [eid, *_] = evaluator.score_and_store(conn, policy, llm, "idea", "idea", content, creator_id=cid, fmt=fmt,
                                             idea_id=idea_id, item_key=idea_id)
        for dim, sp in conn.execute("SELECT spreads FROM evaluations WHERE id = %s", (eid,)).fetchone()["spreads"].items():
            if sp is not None:
                spreads.setdefault(dim, []).append(sp)
    dims = {d: {"median_spread": statistics.median(v), "within_max_spread": round(sum(x <= max_spread for x in v) / len(v), 3)}
            for d, v in spreads.items()}
    return {"items": len(rows), "max_spread": max_spread, "dimensions": dims,
            "passing_dimensions": sum(1 for d in dims.values() if d["median_spread"] <= max_spread)}
