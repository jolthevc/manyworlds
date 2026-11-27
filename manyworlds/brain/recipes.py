"""Recipes: which combinations of coded attributes go with engagement, for whom and in what context, with honest
uncertainty (docs/research/2026-11-26-recipes.md). Deterministic, in code; no model is asked anything here.

Data: the research reference posts with their newest annotation, read through manyworlds.brain.evidence
(`annotated_posts`), turned into `Row`s by `prepare`:
  outcome   log of over_own_median (the post's engagement over its creator's own median: creator-level effects are
            already divided out). A post without it is excluded and counted by why (never read as zero); a ratio
            under intelligence.recipes.min_ratio is clipped there before the log, and counted.
  fields    a coded field is used for a post only when the medium is asked it (annotation.applies: a video field is
            not "absent" on a photo, it is not there) and the read's stored vocabulary hash for the field is the
            current one or one declared compatible (annotation.current_fields). An unknown value excludes the field
            for that post; an empty label list is a read that saw none of the labels (every label absent).

Inference: the creator is the unit, not the post (a creator with 300 posts must not outvote ten creators with 30). An
effect is a contrast of mean outcomes within one creator x medium (x any policy covariates) stratum, each arm (each
cell of an interaction) holding at least min_posts_per_arm posts; a creator's strata are combined by the contrast's
precision; the creators' effects are averaged with equal weight. Uncertainty: a bootstrap over creators (seeded) for
the CI, and a sign-flip permutation test of the creators' effects for p (exact for a few creators, else random flips),
with a plain sign test beside it.

    main_effect(rows, attr)            with vs without the attribute
    interaction(rows, a, b)            AB - A - B + neither: what the pair adds beyond the two alone
    conditional(rows, attr, context)   the effect within each context value, and a Wald test of whether they differ
                                       (covariance from a joint bootstrap over creators)
    recipe(rows, attrs)                every attribute of the combination (at most max_recipe_attrs) vs the rest

Protocol (`run`): creators are split by a hash of their id into discovery and holdout. Every declared test runs on
the discovery creators; Benjamini-Hochberg over the whole declared family (its size fixed before any result is seen;
a test without enough data counts as p = 1); what passes fdr_q is retested on the holdout creators (BH over the
retested, same direction). Stages: confirmed (held on holdout), discovery (passed discovery; holdout lacks data:
UNCONFIRMED), failed (not significant in discovery, or not replicated), insufficient (too little data to test).
Only a confirmed result may back a `correlation` pattern card (`card_tests`); `validated` stays reserved for our own
experiments (manyworlds.brain.patterns.basis).

`screen` builds a bounded data-driven family: attributes that pass an outcome-blind prevalence floor in discovery,
their main effects and the pairwise interactions between different fields, capped at max_pairs.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from ..research import annotation as ann

KINDS = ("main", "interaction", "conditional", "recipe")
CONTEXT_FIELDS = ("medium", "platform", "archetype", "follower_band", "month", "age_band")
STAGES = ("confirmed", "discovery", "failed", "insufficient")
TIE = 1e-9            # numerical tolerance: a creator effect this close to zero is a tie, not a sign
_HASH_SPAN = 16 ** 8  # the split reads the first 8 hex digits of a sha256


def cfg(policy) -> dict:
    return (policy.get("intelligence", "recipes", default={}) if policy is not None else {}) or {}


# --- rows -----------------------------------------------------------------------------------------------------------
@dataclass
class Row:
    post_id: str
    creator_id: str
    medium: str
    y: float | None                      # log over_own_median; None: excluded (see missing)
    attrs: frozenset                     # "field=value" over the usable, known fields
    known: frozenset                     # coded fields usable for this post with a known value
    values: dict = field(default_factory=dict)     # single-value and banded fields: field -> value (for contexts)
    context: dict = field(default_factory=dict)    # medium, platform, archetype, follower_band, month, age_band
    missing: str | None = None           # why y is None
    excluded: dict = field(default_factory=dict)   # field -> why it is not usable (unversioned, old_vocab, unknown)
    clipped: bool = False


def _band(value, bands: dict | None) -> str | None:
    if value is None:
        return None
    for name, (lo, hi) in (bands or {}).items():
        if float(lo) <= float(value) <= float(hi):
            return str(name)
    return None


def _dt(v) -> datetime | None:
    if v is None or isinstance(v, datetime):
        return v
    try:
        return datetime.fromisoformat(str(v))
    except ValueError:
        return None


def prepare(raw: list[dict], policy) -> list[Row]:
    """Rows for the analysis from evidence.annotated_posts-shaped dicts (see the module docstring for the rules)."""
    c = cfg(policy)
    current, compat, bands_ = ann.vocab_versions(policy), ann.vocab_compatible(policy), ann.bands(policy)
    floor = float(c.get("min_ratio", 0.05))
    out = []
    for r in raw:
        medium = str(r.get("media_type") or "unknown")
        ratio, missing, clipped = r.get("over_own_median"), None, False
        if ratio is None:
            missing = "no_creator_baseline" if r.get("metrics") else "no_public_counts"
            y = None
        else:
            ratio = float(ratio)
            if ratio < floor:
                ratio, clipped = floor, True
            y = math.log(ratio)
        obs = r.get("observed") or {}
        stored = obs.get("vocab_versions") or {}
        usable = ann.current_fields(stored, current, compat)
        coded = obs.get("coded") or {}
        known, values, excluded = set(), {}, {}
        for f in ann.CODED_FIELDS:
            if not ann.applies(f, medium):
                continue                     # not asked of this medium: not there, never "absent"
            if f not in usable:
                excluded[f] = "unversioned" if f not in stored else "old_vocab"
                continue
            v = coded.get(f)
            if f in ann.NUMBER_FIELDS:
                b = ann.band(f, v, bands_)
                if b is None:
                    excluded[f] = "unknown"
                    continue
                values[f] = b
            elif f in ann.MULTI_FIELDS:
                labels = [x for x in (v or []) if x not in (None, "")]
                if labels and all(x == ann.UNKNOWN for x in labels):
                    excluded[f] = "unknown"
                    continue
            else:
                if v in (None, "", ann.UNKNOWN):
                    excluded[f] = "unknown"
                    continue
                values[f] = str(v)
            known.add(f)
        attrs = frozenset(ann.coded_attributes({f: coded.get(f) for f in known}, medium, bands_))
        posted, fetched = _dt(r.get("posted_at")), _dt(r.get("fetched_at"))
        age = (fetched - posted).total_seconds() / 86400 if posted and fetched else None
        context = {"medium": medium, "platform": r.get("platform"), "archetype": r.get("archetype"),
                   "follower_band": _band(r.get("followers"), c.get("follower_bands")),
                   "month": posted.strftime("%Y-%m") if posted else None,
                   "age_band": _band(age, c.get("age_bands_days"))}
        out.append(Row(post_id=str(r.get("post_id")), creator_id=str(r.get("creator_id")), medium=medium, y=y,
                       attrs=attrs, known=frozenset(known), values=values, context=context, missing=missing,
                       excluded=excluded, clipped=clipped))
    return out


def load(conn, policy, *, creator_ids=None, outcome: tuple[str, str] | None = None) -> list[Row]:
    """The annotated research posts as rows (through manyworlds.brain.evidence, real mode). `outcome` (metric,
    horizon) analyses that spine outcome instead of over_own_median. Fields the latest gold run flagged unreliable
    (manyworlds.research.gold) are left out of every row, so no test rests on an attribute we can't trust."""
    from ..research import gold
    from . import evidence
    rows = prepare(evidence.annotated_posts(conn, mode="real", schemas=ann.READ_SCHEMAS, creator_ids=creator_ids,
                                            outcome=outcome), policy)
    return drop_fields(rows, gold.unreliable(conn))


def drop_fields(rows: list[Row], fields: set[str]) -> list[Row]:
    """The rows without these coded fields (each recorded in `excluded` as unreliable)."""
    if not fields:
        return rows
    for r in rows:
        gone = fields & set(r.known)
        if gone:
            r.known = frozenset(set(r.known) - gone)
            r.attrs = frozenset(a for a in r.attrs if field_of(a) not in gone)
            r.values = {k: v for k, v in r.values.items() if k not in gone}
            r.excluded.update({f: "unreliable" for f in gone})
    return rows


def select(rows: list[Row], filters: dict | None) -> list[Row]:
    """Rows whose context matches every filter ({medium|platform|archetype|follower_band: value or [values]})."""
    out = rows
    for k, want in (filters or {}).items():
        if want in (None, "", []):
            continue
        allowed = {str(x) for x in (want if isinstance(want, (list, tuple, set)) else [want])}
        out = [r for r in out if str(r.context.get(k)) in allowed]
    return out


def field_of(attr: str) -> str:
    return attr.split("=", 1)[0]


def has(row: Row, attr: str) -> bool | None:
    """True/False when the post's read can say, None when the field is not usable for it."""
    if field_of(attr) not in row.known:
        return None
    return attr in row.attrs


def context_of(row: Row, name: str):
    return row.context.get(name) if name in CONTEXT_FIELDS else row.values.get(name)


def side(creator_id, policy) -> str:
    """discovery or holdout: a fixed hash of the creator id (salted by intelligence.recipes.split_salt)."""
    c = cfg(policy)
    h = int(hashlib.sha256(f"{c.get('split_salt', '')}:{creator_id}".encode()).hexdigest()[:8], 16) / _HASH_SPAN
    return "holdout" if h < float(c.get("holdout_share", 0.5)) else "discovery"


# --- statistics -----------------------------------------------------------------------------------------------------
def bh(ps: list[float | None], m: int | None = None) -> list[float]:
    """Benjamini-Hochberg q values over a family of size m (default len(ps)); a missing p counts as 1."""
    p = [1.0 if x is None else float(x) for x in ps]
    m = max(int(m or len(p)), len(p), 1)
    order = sorted(range(len(p)), key=lambda i: p[i])
    q, running = [1.0] * len(p), 1.0
    for rank in range(len(order) - 1, -1, -1):
        i = order[rank]
        running = min(running, p[i] * m / (rank + 1))
        q[i] = min(1.0, running)
    return q


def sign_test_p(vals: list[float]) -> float:
    """Two-sided binomial sign test on the creators' effects (ties dropped)."""
    pos, neg = sum(v > TIE for v in vals), sum(v < -TIE for v in vals)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def flip_p(vals: list[float], c: dict, rng: random.Random) -> float:
    """Sign-flip permutation p of the mean of the creators' effects (two-sided): exact up to exact_flip_max_creators
    creators, else bootstrap.iterations random flips."""
    n = len(vals)
    if n == 0:
        return 1.0
    obs = abs(sum(vals)) - TIE
    if n <= int(c.get("exact_flip_max_creators", 12)):
        hits = 0
        for mask in range(1 << n):
            s = 0.0
            for i, v in enumerate(vals):
                s += v if (mask >> i) & 1 else -v
            hits += abs(s) >= obs
        return hits / (1 << n)
    iters = int((c.get("bootstrap") or {}).get("iterations", 2000))
    hits = sum(abs(sum(v if rng.random() < 0.5 else -v for v in vals)) >= obs for _ in range(iters))
    return (hits + 1) / (iters + 1)


def _rng(c: dict, key: str) -> random.Random:
    return random.Random(f"{(c.get('bootstrap') or {}).get('seed', 0)}:{key}")


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def _gammq(a: float, x: float) -> float:
    """The regularised upper incomplete gamma Q(a, x) (series below a+1, continued fraction above)."""
    if x <= 0:
        return 1.0
    lead = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:
        ap, d = a, 1.0 / a
        s = d
        for _ in range(1000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(lead))
    tiny = 1e-300
    b = x + 1 - a
    cc, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        cc = b + an / cc
        cc = tiny if abs(cc) < tiny else cc
        d = 1 / d
        delta = d * cc
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return min(1.0, math.exp(lead) * h)


def chi2_sf(x: float, df: int) -> float:
    return 1.0 if df <= 0 else _gammq(df / 2.0, max(0.0, x) / 2.0)


def _solve(m: list[list[float]], v: list[float]) -> list[float] | None:
    """m^-1 v by Gauss-Jordan with partial pivoting; None when m is singular."""
    n = len(v)
    a = [row[:] + [v[i]] for i, row in enumerate(m)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-15:
            return None
        a[col], a[piv] = a[piv], a[col]
        for r in range(n):
            if r != col:
                f = a[r][col] / a[col][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
    return [a[i][n] / a[i][i] for i in range(n)]


# --- estimation -----------------------------------------------------------------------------------------------------
def _creator_effects(rows: list[Row], cell, coefs: dict, c: dict) -> tuple[dict, dict]:
    """{creator: effect} from the contrast `coefs` ({cell: coefficient}) of mean outcomes within each creator x medium
    (x covariates) stratum whose every cell has min_posts_per_arm posts, a creator's strata weighted by the contrast's
    precision; and what went in: {n_posts, n_creators, cells: Counter, used: [rows in contributing strata]}."""
    min_arm = int(c.get("min_posts_per_arm", 3))
    covs = [str(x) for x in (c.get("covariates") or [])]
    strata: dict = defaultdict(lambda: defaultdict(list))
    eligible = []
    for r in rows:
        if r.y is None:
            continue
        k = cell(r)
        if k is None or k not in coefs:
            continue
        eligible.append((r, k))
        strata[(r.creator_id, r.medium, *(r.context.get(v) for v in covs))][k].append(r)
    per_creator: dict = defaultdict(list)
    used = []
    for key in sorted(strata, key=lambda t: tuple(str(x) for x in t)):
        cells = strata[key]
        if any(len(cells.get(k, [])) < min_arm for k in coefs):
            continue
        est = sum(coef * statistics.fmean(r.y for r in cells[k]) for k, coef in coefs.items())
        w = 1.0 / sum(coef * coef / len(cells[k]) for k, coef in coefs.items())
        per_creator[key[0]].append((est, w))
        used += [(r, k) for k in coefs for r in cells[k]]
    effects = {cid: sum(e * w for e, w in xs) / sum(w for _, w in xs) for cid, xs in per_creator.items()}
    return effects, {"n_posts": len(eligible), "n_creators": len({r.creator_id for r, _ in eligible}),
                     "cells": Counter(k for _, k in eligible), "used": used}


def _balance(used: list[tuple[Row, object]], names: dict) -> dict:
    """Each arm's posts by posting month and by post age band (shares within the arm), so imbalance is visible."""
    out: dict = {"month": {}, "age_band": {}}
    by_arm: dict = defaultdict(list)
    for r, k in used:
        by_arm[names.get(k, str(k))].append(r)
    for dim in out:
        for arm, rs in sorted(by_arm.items()):
            cnt = Counter(str(r.context.get(dim)) for r in rs)
            out[dim][arm] = {v: round(n / len(rs), 3) for v, n in sorted(cnt.items())}
    return out


def _coverage(rows: list[Row], fields: list[str], policy) -> dict:
    """Of the posts whose medium is asked every field: how many had the outcome, why the others did not, and for each
    field how many reads could not be used for it (unversioned, old_vocab, unknown)."""
    asked = [r for r in rows if all(ann.applies(f, r.medium) for f in fields if f in ann.CODED_FIELDS)]
    with_metric = sum(r.y is not None for r in asked)
    cur = ann.vocab_versions(policy)
    return {"posts": len(asked), "with_metric": with_metric,
            "share_with_metric": round(with_metric / len(asked), 3) if asked else 0.0,
            "missing": dict(Counter(r.missing for r in asked if r.y is None)),
            "clipped": sum(r.clipped for r in asked),
            "fields_excluded": {f: dict(Counter(r.excluded[f] for r in asked if f in r.excluded)) for f in fields
                                if f in ann.CODED_FIELDS},
            "vocab_versions": {f: cur.get(f) for f in fields if f in cur}}


def _summary(effects: dict, info: dict, c: dict, key: str) -> dict:
    """Effect (multiplier), bootstrap CI, flip p, sign test p and the floors, from the creators' effects."""
    vals = [effects[k] for k in sorted(effects)]
    n = len(vals)
    out = {"n_posts": info["n_posts"], "n_creators": info["n_creators"], "n_creators_both_arms": n,
           "creators_positive": sum(v > TIE for v in vals), "creators_negative": sum(v < -TIE for v in vals),
           "arms": {str(k): v for k, v in sorted(info["cells"].items(), key=lambda t: str(t[0]))}}
    reasons = []
    if info["n_creators"] < int(c.get("min_creators", 6)):
        reasons.append(f"{info['n_creators']} creators with eligible posts (< {c.get('min_creators', 6)})")
    if n < int(c.get("min_creators_both_arms", 5)):
        reasons.append(f"{n} creators with every arm filled (< {c.get('min_creators_both_arms', 5)})")
    if reasons:
        return {**out, "sufficient": False, "reason": "; ".join(reasons), "effect": None, "log_effect": None,
                "ci": None, "p": None, "sign_p": None}
    b = c.get("bootstrap") or {}
    iters, level = int(b.get("iterations", 2000)), float(b.get("ci", 0.95))
    rng = _rng(c, key)
    boots = sorted(statistics.fmean(rng.choice(vals) for _ in range(n)) for _ in range(iters))
    mean = statistics.fmean(vals)
    lo, hi = _percentile(boots, (1 - level) / 2), _percentile(boots, 1 - (1 - level) / 2)
    return {**out, "sufficient": True, "reason": None, "effect": round(math.exp(mean), 4),
            "log_effect": round(mean, 5), "ci": [round(math.exp(lo), 4), round(math.exp(hi), 4)], "ci_level": level,
            "p": round(flip_p(vals, c, _rng(c, key + ":flip")), 5), "sign_p": round(sign_test_p(vals), 5)}


def _key(kind: str, attrs: list[str], context: str | None = None) -> str:
    return f"{kind}:{'+'.join(attrs)}" + (f"|{context}" if context else "")


def _contrast(rows, policy, kind, attrs, cell, coefs, names) -> dict:
    c = cfg(policy)
    effects, info = _creator_effects(rows, cell, coefs, c)
    s = _summary(effects, info, c, _key(kind, attrs))
    fields = sorted({field_of(a) for a in attrs})
    return {"kind": kind, "attrs": list(attrs), **s, "balance": _balance(info["used"], names),
            "coverage": _coverage(rows, fields, policy), "_effects": effects}


def main_effect(rows: list[Row], attr: str, policy) -> dict:
    """Posts with the attribute against posts without it, within each creator."""
    return _contrast(rows, policy, "main", [attr], lambda r: has(r, attr), {True: 1.0, False: -1.0},
                     {True: "with", False: "without"})


def interaction(rows: list[Row], a: str, b: str, policy) -> dict:
    """What the pair adds beyond the two alone: (AB - A - B + neither), within each creator."""
    def cell(r):
        x, y = has(r, a), has(r, b)
        return None if x is None or y is None else (x, y)
    return _contrast(rows, policy, "interaction", [a, b], cell,
                     {(True, True): 1.0, (True, False): -1.0, (False, True): -1.0, (False, False): 1.0},
                     {(True, True): "both", (True, False): "a_only", (False, True): "b_only",
                      (False, False): "neither"})


def recipe(rows: list[Row], attrs: list[str], policy) -> dict:
    """Posts with every attribute of the combination against the rest, within each creator."""
    attrs = list(attrs)[:int(cfg(policy).get("max_recipe_attrs", 3))]

    def cell(r):
        hs = [has(r, a) for a in attrs]
        if any(h is False for h in hs):
            return False
        return True if all(h is True for h in hs) else None
    return _contrast(rows, policy, "recipe", attrs, cell, {True: 1.0, False: -1.0},
                     {True: "combination", False: "rest"})


def conditional(rows: list[Row], attr: str, context_field: str, policy) -> dict:
    """The attribute's effect within each value of a context (medium, platform, archetype, follower band, month, age
    band, or a coded single-value field), and a heterogeneity test: a Wald test that the context effects are equal,
    its covariance from a joint bootstrap over creators (contexts can share creators). p is the heterogeneity p."""
    c = cfg(policy)
    key = _key("conditional", [attr], context_field)
    counts = Counter(context_of(r, context_field) for r in rows if r.y is not None and has(r, attr) is not None)
    counts.pop(None, None)
    values = [v for v, _ in sorted(counts.items(), key=lambda t: (-t[1], str(t[0])))][:int(c.get("max_contexts", 6))]
    by, effects = {}, {}
    for v in values:
        sub = [r for r in rows if context_of(r, context_field) == v]
        res = main_effect(sub, attr, policy)
        effects[v] = res.pop("_effects")
        res.pop("coverage", None)
        by[str(v)] = res
    ok = [v for v in values if by[str(v)]["sufficient"]]
    overall = main_effect(rows, attr, policy)
    overall.pop("_effects")
    base = {"kind": "conditional", "attrs": [attr], "context": context_field, "by_context": by,
            "overall": {k: overall.get(k) for k in ("effect", "ci", "p", "n_creators_both_arms")},
            "coverage": overall["coverage"], "balance": overall["balance"],
            "n_posts": overall["n_posts"], "n_creators": overall["n_creators"],
            "n_creators_both_arms": overall["n_creators_both_arms"], "creators_positive": None,
            "creators_negative": None, "effect": overall.get("effect"), "ci": overall.get("ci"),
            "log_effect": overall.get("log_effect"), "sign_p": None}
    if len(ok) < 2:
        return {**base, "sufficient": False, "p": None, "heterogeneity": None,
                "reason": f"{len(ok)} context values with enough data (< 2)"}
    est = [statistics.fmean(effects[v].values()) for v in ok]
    creators = sorted({cid for v in ok for cid in effects[v]})
    rng = _rng(c, key + ":het")
    draws = []
    for _ in range(int((c.get("bootstrap") or {}).get("iterations", 2000))):
        sample = [rng.choice(creators) for _ in creators]
        vec = []
        for v in ok:
            got = [effects[v][cid] for cid in sample if cid in effects[v]]
            if not got:
                break
            vec.append(statistics.fmean(got))
        if len(vec) == len(ok):
            draws.append([x - vec[0] for x in vec[1:]])
    d = [e - est[0] for e in est[1:]]
    k = len(d)
    het = {"stat": None, "df": k, "p": None, "draws": len(draws)}
    if len(draws) > k:
        means = [statistics.fmean(row[i] for row in draws) for i in range(k)]
        cov = [[sum((row[i] - means[i]) * (row[j] - means[j]) for row in draws) / (len(draws) - 1)
                for j in range(k)] for i in range(k)]
        sol = _solve(cov, d)
        if sol is not None:
            stat = sum(x * y for x, y in zip(d, sol))
            het.update(stat=round(stat, 4), p=round(chi2_sf(stat, k), 6))
    return {**base, "sufficient": het["p"] is not None, "p": het["p"], "heterogeneity": het,
            "reason": None if het["p"] is not None else "the context effects' covariance could not be estimated"}


def evaluate(rows: list[Row], test: dict, policy) -> dict:
    """One declared test ({kind, attrs, context}) on the rows."""
    kind, attrs = test.get("kind"), list(test.get("attrs") or [])
    if kind == "main":
        res = main_effect(rows, attrs[0], policy)
    elif kind == "interaction":
        res = interaction(rows, attrs[0], attrs[1], policy)
    elif kind == "recipe":
        res = recipe(rows, attrs, policy)
    elif kind == "conditional":
        res = conditional(rows, attrs[0], str(test.get("context")), policy)
    else:
        raise ValueError(f"unknown test kind {kind!r}")
    res.pop("_effects", None)
    return res


# --- the protocol ---------------------------------------------------------------------------------------------------
def split(rows: list[Row], policy) -> tuple[list[Row], list[Row]]:
    disc = [r for r in rows if side(r.creator_id, policy) == "discovery"]
    hold = [r for r in rows if side(r.creator_id, policy) == "holdout"]
    return disc, hold


def _same_direction(a: dict, b: dict) -> bool:
    if a.get("kind") == "conditional":
        return True                      # heterogeneity has no direction
    return (a.get("log_effect") or 0) * (b.get("log_effect") or 0) > 0


_SHOWN = ("effect", "log_effect", "ci", "ci_level", "n_posts", "n_creators", "n_creators_both_arms",
          "creators_positive", "creators_negative", "sign_p", "p")


def run(rows: list[Row], policy, tests: list[dict]) -> dict:
    """Every declared test on the discovery creators, BH over the whole family, what passes retested on the holdout
    creators (BH over those, same direction). Each result: the estimate its stage rests on (holdout once retested,
    else discovery) at the top, with `discovery` and `holdout` beside it, q, stage and why."""
    c = cfg(policy)
    fdr = float(c.get("fdr_q", 0.1))
    disc, hold = split(rows, policy)
    family = len(tests)                  # fixed before any result is seen
    first = [evaluate(disc, t, policy) for t in tests]
    q1 = bh([r["p"] if r.get("sufficient") else None for r in first], family)
    surfaced = [i for i, r in enumerate(first) if r.get("sufficient") and q1[i] <= fdr]
    second = {i: evaluate(hold, tests[i], policy) for i in surfaced}
    retest = [i for i in surfaced if second[i].get("sufficient")]
    q2 = dict(zip(retest, bh([second[i]["p"] for i in retest]))) if retest else {}
    results = []
    for i, t in enumerate(tests):
        d = first[i]
        res = {"test": t, "kind": d["kind"], "attrs": d["attrs"], **({"context": d["context"]} if "context" in d
                                                                       else {}),
               "discovery": {**{k: d.get(k) for k in _SHOWN}, "q": round(q1[i], 5), "reason": d.get("reason"),
                             **({"by_context": d["by_context"], "heterogeneity": d["heterogeneity"]}
                                if d["kind"] == "conditional" else {})},
               "coverage": d["coverage"], "balance": d["balance"],
               "vocab_versions": d["coverage"].get("vocab_versions")}
        if not d.get("sufficient"):
            stage, why, top = "insufficient", d.get("reason"), d
            res["q"] = None
        elif q1[i] > fdr:
            stage, why, top = "failed", f"not significant in discovery (q {q1[i]:.3f} > {fdr})", d
            res["q"] = round(q1[i], 5)
        else:
            h = second[i]
            res["holdout"] = {**{k: h.get(k) for k in _SHOWN}, "reason": h.get("reason"),
                              **({"by_context": h["by_context"], "heterogeneity": h["heterogeneity"]}
                                 if h["kind"] == "conditional" else {})}
            if not h.get("sufficient"):
                stage, why, top = "discovery", f"UNCONFIRMED: holdout has too little data ({h.get('reason')})", d
                res["q"] = round(q1[i], 5)
            else:
                res["holdout"]["q"] = round(q2[i], 5)
                res["q"] = round(q2[i], 5)
                if q2[i] <= fdr and _same_direction(d, h):
                    stage, why, top = "confirmed", "held on the holdout creators", h
                else:
                    stage, why, top = "failed", "not replicated on the holdout creators", h
        res.update({k: top.get(k) for k in _SHOWN})
        res.update(stage=stage, reason=why)
        results.append(res)
    return {"family_size": family, "fdr_q": fdr,
            "creators": {"discovery": len({r.creator_id for r in disc}), "holdout": len({r.creator_id for r in hold})},
            "posts": {"discovery": len(disc), "holdout": len(hold)},
            "policy_recipes": {k: c.get(k) for k in ("min_creators", "min_posts_per_arm", "min_creators_both_arms",
                                                     "holdout_share", "split_salt", "covariates")},
            "results": results,
            "confirmed": sum(r["stage"] == "confirmed" for r in results)}


def screen(rows: list[Row], policy, attrs: list[str] | None = None) -> list[dict]:
    """A bounded data-driven family, built on the discovery creators without looking at outcomes: the attributes
    held by between prevalence_floor and 1 - prevalence_floor of the discovery posts that know the field (the most
    balanced max_attributes), their main effects (screen.main_effects) and the pairwise interactions between
    attributes of different fields, at most max_pairs. Every test returned belongs to the family."""
    s = cfg(policy).get("screen") or {}
    floor = float(s.get("prevalence_floor", 0.15))
    disc = [r for r in rows if r.y is not None and side(r.creator_id, policy) == "discovery"]
    universe = sorted(set(attrs) if attrs else {a for r in disc for a in r.attrs})
    prev = {}
    for a in universe:
        elig = [r for r in disc if field_of(a) in r.known]
        if elig:
            share = sum(a in r.attrs for r in elig) / len(elig)
            if floor <= share <= 1 - floor:
                prev[a] = share
    ranked = sorted(prev, key=lambda a: (abs(prev[a] - 0.5), a))[:int(s.get("max_attributes", 30))]
    tests = [{"kind": "main", "attrs": [a]} for a in ranked] if s.get("main_effects", True) else []
    pairs = [(a, b) for a, b in itertools.combinations(ranked, 2) if field_of(a) != field_of(b)]
    tests += [{"kind": "interaction", "attrs": [a, b]} for a, b in pairs[:int(s.get("max_pairs", 60))]]
    return tests


def answer(rows: list[Row], policy, spec: dict) -> dict:
    """A research question's spec ({tests, filters, screen}) answered on the rows: filters first, then the declared
    tests plus the screen's (when asked), all one family."""
    sub = select(rows, spec.get("filters"))
    tests = list(spec.get("tests") or [])
    if spec.get("screen"):
        tests += [t for t in screen(sub, policy) if t not in tests]
    out = run(sub, policy, tests)
    out["filters"] = spec.get("filters") or {}
    return out


def card_tests(result: dict) -> list[dict]:
    """A confirmed result as a pattern card's evidence test (manyworlds.brain.patterns.basis reads q, n_high, n_low):
    nothing for any other stage, so only holdout-confirmed results can make a card a `correlation`."""
    if result.get("stage") != "confirmed":
        return []
    h = result.get("holdout") or {}
    n = int(h.get("n_creators_both_arms") or 0)
    return [{"attribute": " + ".join(result.get("attrs") or []), "kind": result.get("kind"),
             "effect": result.get("effect"), "ci": result.get("ci"), "high": result.get("effect"), "low": 1.0,
             "p": h.get("p"), "q": result.get("q"), "n_high": n, "n_low": n, "unit": "creators",
             "stage": "confirmed"}]


def own_rows(conn, policy) -> list[Row]:
    """Our own accepted content coded in the same vocabulary (manyworlds.review.own_annotation), as rows without an
    outcome: for comparing what we make with what works elsewhere (inquiry.contrast_sources), never for effects.
    Sandbox creators count here (mode sandbox): only attributes are read, never metrics."""
    from . import evidence
    raw = [{"post_id": r["asset_id"], "creator_id": r["creator_id"], "platform": "own",
            "media_type": "video" if str(r.get("format") or "").endswith(("video", "clip", "edit")) else "photo",
            "observed": r["observed"], "over_own_median": None, "metrics": {}}
           for r in evidence.own_annotations(conn, mode="sandbox", schemas=ann.READ_SCHEMAS)]
    return prepare(raw, policy)
