"""The analyst's statistics, in code (the LLM only words what these numbers show).

Outcome: each post's reach at the target age, as a log-lift over the median of the creator's
previous `baseline_window` posts, so a growing or shrinking account doesn't masquerade as a
creative effect. Breakouts (≥ `breakout_ratio` × baseline) are reported, and capped at that ratio
when estimating effects, so one lucky post can't create a pattern.

Effects: per rubric dimension, the slope of outcome on score; per format and per attribute value,
the mean difference from the rest; per format × dimension, the slope within the format. Each has a
bootstrap standard error. Many effects are tested at once, so a `tentative` grade needs the effect
to clear a Bonferroni-corrected interval at family-wide `alpha` and `min_n_tentative` posts;
anything else is `speculative`. `proven` is reserved for experiments. Below `min_n_any` posts
nothing is estimated: the evidence is insufficient and the Brain keeps exploring.
"""

from __future__ import annotations

import math
import random
import statistics
from statistics import NormalDist

from ..policy import Policy
from . import evidence


def _cfg(policy: Policy) -> dict:
    return policy.get("brain", "analysis", default={}) or {}


def dataset(conn, creator_id, *, mode: evidence.Mode, policy: Policy) -> list[dict]:
    """Published (or simulated) posts in time order with their idea's scores and attributes and their reach
    at the target age, read in the creator's mode. Posts not yet observed at the target age are left out."""
    target = int(_cfg(policy).get("target_age_s", 259200))
    posts = conn.execute(
        """SELECT p.id, p.published_at, p.format, i.attributes, i.experiment_id, i.arm, e.scores
           FROM posts p LEFT JOIN ideas i ON i.post_id = p.id
           LEFT JOIN LATERAL (SELECT scores FROM evaluations WHERE idea_id = i.id AND role = 'evaluator'
                              ORDER BY created_at DESC LIMIT 1) e ON true
           WHERE p.creator_id = %s AND p.status IN ('published', 'simulated') AND p.published_at IS NOT NULL
           ORDER BY p.published_at""", (creator_id,)).fetchall()
    rows = []
    for p in posts:
        obs = [m for m in evidence.metrics(conn, p["id"], mode=mode)
               if m.get("post_age_s") is not None and m["post_age_s"] <= target]
        if not obs:
            continue
        latest = max(obs, key=lambda m: m["post_age_s"])
        if latest["post_age_s"] < target:
            continue   # too young: its reach is still a fraction of what it will be, and would read as failure
        rows.append({"post_id": str(p["id"]), "published_at": p["published_at"], "format": p["format"],
                     "scores": p["scores"] or {}, "attributes": p["attributes"] or {},
                     "reach": latest["metrics"].get("reach"), "age_s": latest["post_age_s"],
                     "experiment_id": str(p["experiment_id"]) if p["experiment_id"] else None, "arm": p["arm"]})
    return rows


def normalise(rows: list[dict], policy: Policy) -> list[dict]:
    cfg = _cfg(policy)
    window, need = int(cfg.get("baseline_window", 20)), int(cfg.get("min_baseline", 3))
    out = []
    for i, r in enumerate(rows):
        prev = [x["reach"] for x in rows[max(0, i - window):i] if x.get("reach")]
        lift = (math.log(r["reach"] / statistics.median(prev))
                if len(prev) >= need and r.get("reach") else None)
        out.append({**r, "lift": lift})
    return out


def _slope(pairs: list[tuple[float, float]]) -> float | None:
    xs = [x for x, _ in pairs]
    mx, my = statistics.fmean(xs), statistics.fmean(y for _, y in pairs)
    sxx = sum((x - mx) ** 2 for x in xs)
    return None if sxx == 0 else sum((x - mx) * (y - my) for x, y in pairs) / sxx


def _diff(pairs: list[tuple[bool, float]]) -> float | None:
    a = [y for g, y in pairs if g]
    b = [y for g, y in pairs if not g]
    return None if not a or not b else statistics.fmean(a) - statistics.fmean(b)


def _bootstrap_se(pairs, stat, draws: int, rng: random.Random) -> float | None:
    vals = []
    for _ in range(draws):
        v = stat([pairs[rng.randrange(len(pairs))] for _ in pairs])
        if v is not None:
            vals.append(v)
    return statistics.stdev(vals) if len(vals) > 2 else None


def effects(rows: list[dict], policy: Policy, *, seed: int = 0) -> dict:
    cfg = _cfg(policy)
    cap = math.log(float(cfg.get("breakout_ratio", 3.0)))
    usable = [r for r in rows if r.get("lift") is not None]
    breakouts = [{"post_id": r["post_id"], "lift": round(r["lift"], 3), "format": r["format"]}
                 for r in usable if r["lift"] >= cap]
    n, min_any = len(usable), int(cfg.get("min_n_any", 8))
    result = {"n": n, "insufficient": n < min_any, "effects": [], "breakouts": breakouts}
    if result["insufficient"]:
        return result
    for r in usable:
        r["y"] = max(-cap, min(cap, r["lift"]))
    rng, draws = random.Random(seed), int(cfg.get("bootstrap", 300))
    min_group = int(cfg.get("min_group", 5))
    candidates: list[tuple[str, str, list, object]] = []
    dims = sorted({d for r in usable for d, v in (r["scores"] or {}).items() if v is not None})
    for d in dims:
        pairs = [(float(r["scores"][d]), r["y"]) for r in usable if r["scores"].get(d) is not None]
        if len(pairs) >= min_any:
            candidates.append(("dimension", d, pairs, _slope))
    formats = sorted({r["format"] for r in usable})
    for f in formats:
        pairs = [(r["format"] == f, r["y"]) for r in usable]
        k = sum(g for g, _ in pairs)
        if k >= min_group and n - k >= min_group:
            candidates.append(("format", f, pairs, _diff))
        within = [r for r in usable if r["format"] == f]
        if len(within) >= min_any:
            for d in dims:
                fp = [(float(r["scores"][d]), r["y"]) for r in within if r["scores"].get(d) is not None]
                if len(fp) >= min_any:
                    candidates.append(("format_dimension", f"{f}:{d}", fp, _slope))
    attrs = sorted({(k, str(v).lower()) for r in usable for k, v in (r["attributes"] or {}).items()})
    for k, v in attrs:
        pairs = [(str((r["attributes"] or {}).get(k, "")).lower() == v, r["y"]) for r in usable]
        hits = sum(g for g, _ in pairs)
        if hits >= min_group and n - hits >= min_group:
            candidates.append(("attribute", f"{k}={v}", pairs, _diff))
    m = max(1, len(candidates))
    alpha = float(cfg.get("alpha", 0.1))
    z_family = NormalDist().inv_cdf(1 - alpha / (2 * m))     # Bonferroni across everything tested
    min_tentative = int(cfg.get("min_n_tentative", 20))
    min_group_tentative = int(cfg.get("min_group_tentative", 10))   # each side of a group comparison
    for kind, key, pairs, stat in candidates:
        est, se = stat(pairs), _bootstrap_se(pairs, stat, draws, rng)
        if est is None or se is None:
            continue
        low, high = est - z_family * se, est + z_family * se
        smallest = len(pairs) if stat is _slope else min(sum(1 for g, _ in pairs if g), sum(1 for g, _ in pairs if not g))
        tentative = (len(pairs) >= min_tentative and smallest >= min_group_tentative and (low > 0 or high < 0))
        result["effects"].append({"kind": kind, "key": key, "effect": round(est, 4), "se": round(se, 4),
                                  "low": round(low, 4), "high": round(high, 4), "n": len(pairs),
                                  "grade": "tentative" if tentative else "speculative"})
    result["effects"].sort(key=lambda e: -abs(e["effect"]) / (e["se"] or 1))
    result["tested"] = m
    return result


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _spearman(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None
    rx, ry = _ranks([x for x, _ in pairs]), _ranks([y for _, y in pairs])
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx, syy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return None if sxx == 0 or syy == 0 else sxy / math.sqrt(sxx * syy)


def rubric_validity(rows: list[dict], policy: Policy, *, seed: int = 0) -> list[dict]:
    """How well each rubric dimension's idea score predicted the post's outcome: Spearman rank correlation
    with a bootstrap interval at `alpha`. A dimension that predicts nothing is a candidate to reweight or drop
    when the rubric is next revised (spec 4.4: predictive usefulness beats plausibility). Report only."""
    cfg = _cfg(policy)
    usable = [r for r in rows if r.get("lift") is not None]
    rng, draws = random.Random(seed), int(cfg.get("bootstrap", 300))
    z = NormalDist().inv_cdf(1 - float(cfg.get("alpha", 0.1)) / 2)
    out = []
    for d in sorted({d for r in usable for d, v in (r["scores"] or {}).items() if v is not None}):
        pairs = [(float(r["scores"][d]), r["lift"]) for r in usable if (r["scores"] or {}).get(d) is not None]
        if len(pairs) < int(cfg.get("min_n_any", 8)):
            continue
        rho, se = _spearman(pairs), _bootstrap_se(pairs, _spearman, draws, rng)
        if rho is None or se is None:
            continue
        out.append({"dimension": d, "n": len(pairs), "rho": round(rho, 3), "low": round(rho - z * se, 3),
                    "high": round(rho + z * se, 3), "predictive": rho - z * se > 0})
    return sorted(out, key=lambda x: -x["rho"])
