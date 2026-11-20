"""Experiments: the step from "this pattern showed up" to "this causes more reach".

The analyst's findings are observational: a pattern in what the creator happened to post. A tentative one
(it cleared the corrected interval) becomes an experiment with two arms: A follows the finding (an
attribute value, a high score on a rubric dimension, or a format), B is the control (anything else). The
showrunner gets randomised slots each batch, a day and an arm, and writes a natural idea for each, so arms
are spread across days and the plan's other content instead of being chosen where they'd look good.

The outcome is reach as log-lift over the creator's baseline (as in manyworlds.brain.analysis), read once
each arm has its sample: a fixed horizon, so looking every batch can't inflate false positives. The sample
is sized when the experiment opens, for `power` to detect `mde` given the creator's outcome spread. A clear difference becomes a `proven` finding (evidence `experimental`, or `simulated` for
sandbox creators, through manyworlds.brain.evidence); no difference is recorded on the experiment, which
is a result too. One experiment per creator and effect, ever.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from statistics import NormalDist

from psycopg.types.json import Jsonb

from ..policy import Policy
from . import analysis, evidence


def _cfg(policy: Policy) -> dict:
    return policy.get("brain", "experiments", default={}) or {}


def arms_for(effect_key: str) -> list[dict] | None:
    """Arm A follows the effect, arm B is the control. None for effects an idea can't be told to follow."""
    kind, _, key = effect_key.partition(":")
    if kind == "attribute" and "=" in key:
        k, v = key.split("=", 1)
        return [{"name": "A", "attribute": {k: v}, "instruction": f"{k}: {v} (make the idea genuinely that)"},
                {"name": "B", "attribute": {}, "instruction": f"{k}: not {v} (any other {k}; otherwise a normal idea)"}]
    if kind == "dimension" and key:
        return [{"name": "A", "attribute": {}, "dimension": key,
                 "instruction": f"an idea where {key} is the heart of it (it should score high on {key})"},
                {"name": "B", "attribute": {}, "instruction": f"a good ordinary idea where {key} is not the point"}]
    if kind == "format" and key:
        return [{"name": "A", "attribute": {}, "format": key, "instruction": f"format {key}"},
                {"name": "B", "attribute": {}, "instruction": f"any format except {key}"}]
    return None


def sample_size(policy: Policy, sd: float) -> int:
    """Posts per arm for `power` to detect a difference of `mde` at two-sided `alpha`, given outcome spread sd."""
    cfg = _cfg(policy)
    z = NormalDist().inv_cdf(1 - float(cfg.get("alpha", 0.05)) / 2) + NormalDist().inv_cdf(float(cfg.get("power", 0.8)))
    n = math.ceil(2 * z * z * sd * sd / float(cfg.get("mde", 0.4)) ** 2)
    return max(int(cfg.get("min_sample", 10)), min(int(cfg.get("max_sample", 60)), n))


def _outcome_sd(conn, policy: Policy, creator_id, mode) -> float:
    lifts = [r["lift"] for r in analysis.normalise(analysis.dataset(conn, creator_id, mode=mode, policy=policy), policy)
             if r.get("lift") is not None]
    if len(lifts) >= int(policy.get("brain", "analysis", "min_n_any", default=8)):
        return statistics.stdev(lifts)
    return float(_cfg(policy).get("default_sd", 0.6))


def open_for(conn, creator_id) -> list[dict]:
    return conn.execute("SELECT * FROM experiments WHERE creator_id = %s AND status = 'open' ORDER BY created_at",
                        (creator_id,)).fetchall()


def concluded_for(conn, creator_id) -> list[dict]:
    return conn.execute("SELECT * FROM experiments WHERE creator_id = %s AND status = 'concluded' "
                        "ORDER BY concluded_at", (creator_id,)).fetchall()


def open_from_findings(conn, policy: Policy, creator_id) -> list[str]:
    """Open experiments for the creator's tentative findings, strongest first, up to the cap."""
    cfg = _cfg(policy)
    room = int(cfg.get("max_open_per_creator", 1)) - len(open_for(conn, creator_id))
    mode = evidence.mode_for(conn, creator_id)
    found = evidence.findings(conn, mode=mode, creator_id=creator_id, grades=("tentative",))
    found.sort(key=lambda f: -abs(float(f["effect"] or 0)))
    tested = {r["effect_key"] for r in conn.execute(
        "SELECT effect_key FROM experiments WHERE creator_id = %s AND effect_key IS NOT NULL", (creator_id,)).fetchall()}
    opened, need = [], None
    for f in found:
        if room <= 0:
            break
        keys = next((s.get("effects") for s in f["sources"] or [] if isinstance(s, dict) and s.get("effects")), None)
        key = keys[0] if keys else None
        arms = arms_for(key) if key else None
        if not arms or key in tested:
            continue
        need = need or sample_size(policy, _outcome_sd(conn, policy, creator_id, mode))
        eid = conn.execute(
            "INSERT INTO experiments (creator_id, hypothesis, arms, metric, min_sample, status, effect_key, source_finding) "
            "VALUES (%s, %s, %s, 'reach_lift', %s, 'open', %s, %s) RETURNING id",
            (creator_id, f["statement"], Jsonb(arms), need, key, f["id"])).fetchone()["id"]
        opened.append(str(eid))
        tested.add(key)
        room -= 1
    return opened


def assign(conn, policy: Policy, creator_id, *, start, days: int) -> list[dict]:
    """This batch's experiment slots: for each open experiment, `slots_per_batch` slots split evenly between
    the arms, each on a random day. Seeded by experiment and batch start, so a rerun gets the same slots."""
    per = int(_cfg(policy).get("slots_per_batch", 4))
    slots = []
    for exp in open_for(conn, creator_id):
        rng = random.Random(int(hashlib.sha256(f"{exp['id']}:{start}".encode()).hexdigest()[:12], 16))
        arms = (["A", "B"] * (per // 2 + 1))[:per]
        rng.shuffle(arms)
        by_name = {a["name"]: a for a in exp["arms"]}
        for i, arm in enumerate(arms):
            slots.append({"slot": f"{exp['id']}:{i}", "experiment_id": str(exp["id"]), "arm": arm,
                          "day": rng.randrange(days), "instruction": by_name[arm]["instruction"]})
    return slots


def compare(rows: list[dict], policy: Policy, *, seed: int = 0) -> dict:
    """Difference in mean outcome `y` between arm A and arm B, with a normal interval at `alpha`.
    (`seed` is accepted for symmetry with the analysis code; the interval is analytic.)"""
    a = [r["y"] for r in rows if r["arm"] == "A"]
    b = [r["y"] for r in rows if r["arm"] == "B"]
    alpha = float(_cfg(policy).get("alpha", 0.05))
    if len(a) < 2 or len(b) < 2:
        return {"n_a": len(a), "n_b": len(b), "effect": None, "significant": False}
    est = statistics.fmean(a) - statistics.fmean(b)
    se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
    z = NormalDist().inv_cdf(1 - alpha / 2)
    low, high = est - z * se, est + z * se
    return {"n_a": len(a), "n_b": len(b), "effect": round(est, 4), "se": round(se, 4), "low": round(low, 4),
            "high": round(high, 4), "significant": low > 0 or high < 0}


def rows_for(conn, policy: Policy, exp: dict, mode: evidence.Mode) -> list[dict]:
    """The experiment's posts with outcomes, by the arm they were assigned (intent to treat)."""
    data = analysis.normalise(analysis.dataset(conn, exp["creator_id"], mode=mode, policy=policy), policy)
    return [{"post_id": r["post_id"], "arm": r["arm"], "lift": r["lift"]} for r in data
            if r.get("experiment_id") == str(exp["id"]) and r.get("arm") and r.get("lift") is not None]


def conclude(conn, policy: Policy, exp: dict, rows: list[dict]) -> dict | None:
    """Read the result once each arm has its minimum sample; None while it hasn't."""
    need = int(exp["min_sample"] or _cfg(policy).get("min_sample", 10))
    if min(sum(r["arm"] == "A" for r in rows), sum(r["arm"] == "B" for r in rows)) < need:
        return None
    cap = math.log(float(policy.get("brain", "analysis", "breakout_ratio", default=3.0)))
    res = compare([{"arm": r["arm"], "y": max(-cap, min(cap, r["lift"]))} for r in rows], policy)
    status = ("confirmed" if res["effect"] > 0 else "reversed") if res["significant"] else "no_difference"
    finding_id = None
    if status != "no_difference":
        mode = evidence.mode_for(conn, exp["creator_id"])
        a = next(x for x in exp["arms"] if x["name"] == "A")
        verb = "beat" if status == "confirmed" else "did worse than"
        statement = (f"Experiment: posts with {a['instruction']} {verb} the control by {abs(res['effect']):.2f} "
                     f"log-lift in reach ({res['n_a']} vs {res['n_b']} posts). Tested: {exp['hypothesis']}")
        source = [{"creator_id": str(exp["creator_id"])}, {"experiment_id": str(exp["id"])},
                  {"posts": [r["post_id"] for r in rows]}] + ([{"table": "simulated_metrics"}] if mode == "sandbox" else [])
        finding_id = evidence.record_finding(
            conn, scope="creator", creator_id=exp["creator_id"], statement=statement,
            evidence="simulated" if mode == "sandbox" else "experimental", grade="proven", effect=res["effect"],
            interval=(res["low"], res["high"]), n=res["n_a"] + res["n_b"], sources=source)
    result = {"status": status, **res, "finding_id": str(finding_id) if finding_id else None}
    conn.execute("UPDATE experiments SET status = 'concluded', result = %s, concluded_at = now(), updated_at = now() "
                 "WHERE id = %s", (Jsonb(result), exp["id"]))
    return result


def review(conn, policy: Policy, creator_id) -> dict:
    """The analyst's experiment step: read results that are due, then open new experiments."""
    mode = evidence.mode_for(conn, creator_id)
    concluded = []
    for exp in open_for(conn, creator_id):
        out = conclude(conn, policy, exp, rows_for(conn, policy, exp, mode))
        if out:
            concluded.append({"experiment_id": str(exp["id"]), **out})
    return {"concluded": concluded, "opened": open_from_findings(conn, policy, creator_id)}
