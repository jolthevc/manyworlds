"""The simulated audience: a deterministic, seeded scenario engine for sandbox creators.

A scenario (policy/scenarios/<name>.yaml over _base.yaml) is a hidden truth: how rubric dimensions,
formats and attributes move a post's reach, plus account growth or decline, noise and breakouts.
Each simulated publication gets outcomes at the policy observation ages, with the same metric keys
real Instagram ingestion uses. The analyst must discover the hidden effects from the outcomes.
This tests the learning loop; it is never evidence of real demand (outcomes are stored only in
`simulated_metrics`, through manyworlds.brain.evidence).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import yaml

from ..policy import Policy
from . import evidence, rubrics

PANEL_VERSION = "scenario-v1"


def _root() -> Path:
    return rubrics.ROOT.parent / "scenarios"


def load(name: str) -> dict:
    base = yaml.safe_load((_root() / "_base.yaml").read_text(encoding="utf-8"))
    over = yaml.safe_load((_root() / f"{name}.yaml").read_text(encoding="utf-8")) or {}
    return {**base, **over, "name": name}


def simulate(scenario: dict, features: dict, *, index: int, rng_key: str, policy: Policy) -> list[dict]:
    """Outcomes for one post. features: {format, scores: {dimension: 1-10}, attributes: {key: value}};
    index: how many posts the creator published before this one (for growth or decline)."""
    rng = random.Random(f"{scenario['name']}:{rng_key}")
    log_reach = math.log(float(scenario["base_reach"])) + float(scenario["growth_per_post"]) * index
    scores, attrs, fmt = features.get("scores") or {}, features.get("attributes") or {}, features.get("format")
    for e in scenario.get("effects") or []:
        if "dimension" in e and (e.get("format") in (None, fmt)) and scores.get(e["dimension"]) is not None:
            log_reach += float(e["per_point"]) * (float(scores[e["dimension"]]) - 5)
        elif "attribute" in e and str(attrs.get(e["attribute"], "")).lower() == str(e.get("value", "")).lower():
            log_reach += float(e["lift"])
        elif "format" in e and "dimension" not in e and e["format"] == fmt:
            log_reach += float(e["lift"])
    log_reach += rng.gauss(0, float(scenario["noise_sd"]))
    b = scenario.get("breakout") or {}
    breakout = rng.random() < float(b.get("probability", 0))
    if breakout:
        log_reach += math.log(float(b.get("multiplier", 1)))
    reach72 = math.exp(log_reach)
    curve = {int(k): float(v) for k, v in scenario["curve"].items()}
    rows = []
    for age in policy.get("simulation", "ages_s", default=sorted(curve)):
        reach = int(round(reach72 * curve.get(int(age), 1.0)))
        metrics = {"reach": reach, **{k: int(round(reach * float(r))) for k, r in scenario["rates"].items()}}
        rows.append({"post_age_s": int(age), "metrics": metrics, "breakout": breakout})
    return rows


def features_of(conn, post_id) -> dict:
    """What the scenario sees of a post: its format, its idea's scores and attributes."""
    row = conn.execute(
        """SELECT p.format, i.attributes, e.scores FROM posts p LEFT JOIN ideas i ON i.post_id = p.id
           LEFT JOIN LATERAL (SELECT scores FROM evaluations WHERE idea_id = i.id AND role = 'evaluator'
                              ORDER BY created_at DESC LIMIT 1) e ON true
           WHERE p.id = %s LIMIT 1""", (post_id,)).fetchone()
    return {"format": row["format"], "scores": row["scores"] or {}, "attributes": row["attributes"] or {}}


def simulate_post(conn, policy: Policy, post_id, *, scenario: str, seed: int) -> list[dict]:
    """Write a simulated publication's outcomes (once) and return them."""
    if evidence.metrics(conn, post_id, mode="sandbox"):
        return []
    post = conn.execute("SELECT creator_id FROM posts WHERE id = %s", (post_id,)).fetchone()
    index = conn.execute("SELECT count(*) n FROM publications pb JOIN posts p ON p.id = pb.post_id "
                         "WHERE p.creator_id = %s AND pb.simulated AND pb.post_id <> %s",
                         (post["creator_id"], post_id)).fetchone()["n"]
    rows = simulate(load(scenario), features_of(conn, post_id), index=index, rng_key=f"{post_id}:{seed}",
                    policy=policy)
    evidence.record_simulated_metrics(conn, post_id, rows, model=f"scenario:{scenario}", panel_version=PANEL_VERSION)
    return rows
