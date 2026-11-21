"""Creative scoring: an item is scored `samples` times against a rubric's anchors, and the median
and spread per dimension are kept, so scores are comparable over time and noisy dimensions show.

Scores are integers clamped to 1-10. A dimension the model leaves out of a sample is skipped for
that sample; one no sample scored is stored as null, never invented. A deterministic share of
items (`checked`) is also scored by the strong model to measure how far the cheaper evaluator
agrees with it.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

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


def score(llm, policy: Policy, r: Rubric, content: list, *, role: str, samples: int, creator_id=None,
          more: Callable[[dict], int] | None = None) -> Scored:
    """`samples` scorings, then as many more as `more(medians)` asks for: a score near a decision line is
    worth re-checking, one far from it is not (at scale every sample is a paid call)."""
    schema = schema_for(r)
    # The rubric is the same for every item: cached, so repeat calls pay a fraction of its tokens.
    prompt = [{"type": "text", "text": r.as_text(), "cache_control": {"type": "ephemeral"}}, *_blocks(content),
              {"type": "text", "text": f"Score every dimension of the {r.rubric} rubric for this item."}]
    per_dim: dict[str, list[int]] = {d.key: [] for d in r.dimensions}
    why: dict[str, list[str]] = {d.key: [] for d in r.dimensions}
    notes: list[str] = []

    def medians() -> dict:
        return {k: (statistics.median(v) if v else None) for k, v in per_dim.items()}

    def sample() -> None:
        out = llm.structured(role, system=SYSTEM, content=prompt, schema=schema, effort=effort(policy),
                             creator_id=creator_id)
        for d in r.dimensions:
            s = getattr(out, d.key, None)
            if s is None or s.score is None:
                continue
            per_dim[d.key].append(max(1, min(10, int(s.score))))
            why[d.key].append(s.why)
        if getattr(out, "overall_note", ""):
            notes.append(out.overall_note)

    for _ in range(samples):
        sample()
    extra = int(more(medians())) if more else 0
    for _ in range(extra):
        sample()
    return Scored(medians=medians(), spreads={k: (max(v) - min(v) if v else None) for k, v in per_dim.items()},
                  rationale=why, model=llm.model_for(role), samples=samples + extra, notes=notes)


# How items are laid out for the evaluator. Part of the prompt version: change them and the
# version changes, so scores from different prompts are never silently compared.
IDEA_TEMPLATE = "THE IDEA ({format}): {concept}\nPLAN: {plan}"
CREATOR_TEMPLATE = "THE CREATOR (@{handle}): {facts}"
ASSET_TEMPLATES = ("REFERENCE PHOTOS OF THE CREATOR (judge identity against these):", "PLANNED SCENE: {scenes}",
                   "THE IMAGE:", "CAPTION: {caption}")


def effort(policy: Policy) -> str:
    return str(policy.get("brain", "evaluation", "effort", default="low"))


def prompt_version(policy: Policy) -> str:
    """The policy label plus a hash of everything that shapes what the evaluator reads and how hard it
    thinks: the system prompt, the templates and the effort."""
    label = policy.get("brain", "evaluation", "prompt_version", default="eval")
    material = "\n".join([SYSTEM, IDEA_TEMPLATE, CREATOR_TEMPLATE, *ASSET_TEMPLATES, effort(policy)])
    return f"{label}-{hashlib.sha256(material.encode()).hexdigest()[:8]}"


def checked(item_id, rate: float) -> bool:
    """Whether this item also gets a strong-model score: the same answer for the same item every time."""
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    return int(hashlib.sha256(str(item_id).encode()).hexdigest()[:8], 16) / 2**32 < rate


def latest(conn, *, idea_id=None, asset_id=None, role: str = "evaluator") -> dict | None:
    """The most recent evaluation of an idea or asset by one evaluator role (the volume evaluator by
    default), so a time series never mixes the volume model with the strong-model checks."""
    return conn.execute(
        """SELECT * FROM evaluations
           WHERE (%(idea)s::uuid IS NULL OR idea_id = %(idea)s) AND (%(asset)s::uuid IS NULL OR asset_id = %(asset)s)
             AND role = %(role)s
           ORDER BY created_at DESC LIMIT 1""", {"idea": idea_id, "asset": asset_id, "role": role}).fetchone()


def reference_items(rubric_name: str) -> list[dict]:
    """The fixed reference set for a rubric (policy/rubrics/reference/<rubric>.yaml)."""
    import yaml

    from .rubrics import ROOT
    body = yaml.safe_load((ROOT / "reference" / f"{rubric_name}.yaml").read_text(encoding="utf-8"))
    return [{**item, "creator": body.get("creator", "")} for item in body["items"]]


def reference_content(item: dict) -> list[str]:
    """A reference idea laid out exactly as live ideas are (same templates), so drift measures the
    prompt live scoring actually uses."""
    fmt, _, rest = item["idea"].partition(":")
    creator = CREATOR_TEMPLATE.format(handle="reference", facts=json.dumps({"one_line": item["creator"]}))
    return [creator, IDEA_TEMPLATE.format(format=fmt.strip(), concept=rest.strip(), plan="{}")]


def score_reference(conn, policy: Policy, llm, rubric_name: str) -> dict[str, dict[str, float | None]]:
    """Score the reference set under the current rubric, evaluator model and prompt version. Each
    (model, prompt version) keeps its own baseline on the rubric version, and the drift from the
    previous run is stored with it, so comparisons across models or prompts are always measured."""
    from psycopg.types.json import Jsonb

    from . import rubrics
    r = rubrics.load(rubric_name)
    rubrics.register(conn, r)
    samples = int(policy.get("brain", "evaluation", "samples", default=3))
    scores = {item["id"]: score(llm, policy, r, reference_content(item), role="evaluator", samples=samples).medians
              for item in reference_items(rubric_name)}
    key = f"{llm.model_for('evaluator')}|{prompt_version(policy)}"
    ref = conn.execute("SELECT reference FROM rubric_versions WHERE rubric = %s AND version = %s",
                       (r.rubric, r.version)).fetchone()["reference"] or {}
    runs, drifts = dict(ref.get("runs") or {}), list(ref.get("drift") or [])
    if runs:
        previous = max(runs, key=lambda k: runs[k]["at"])
        drifts.append({"before": previous, "after": key, "drift": drift(runs[previous]["scores"], scores)})
    runs[key] = {"model": llm.model_for("evaluator"), "prompt_version": prompt_version(policy), "samples": samples,
                 "scores": scores, "at": conn.execute("SELECT now()::text AS t").fetchone()["t"]}
    conn.execute("UPDATE rubric_versions SET reference = %s WHERE rubric = %s AND version = %s",
                 (Jsonb({"runs": runs, "drift": drifts}), r.rubric, r.version))
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
