"""Exploratory methods for research questions, beside the pre-registered tests of manyworlds.brain.recipes (owner
2026-11-26: agents need room for genuinely novel questions, not a few fixed reports). The model chooses a method and
its arguments; this code computes the numbers. Deterministic; no model is asked anything here.

Exploration finds where to look; it is never a finding by itself. Every run is recorded in the trial ledger
(analysis_runs, `record`), so any later test of what it surfaced counts it in its multiple-testing family, and a
claim it suggests still has to pass recipes.run (discovery and holdout creators, BH FDR) before it is believed.

    prevalence(rows, attr, by)      how common an attribute is in each group of a context (creator-weighted: each
                                    creator's own share, averaged), with the median outcome of posts with and without
    contrast_sources(external, own, attrs)
                                    each attribute's share among the external posts that beat their creator's median
                                    against its share in our own accepted content: what winners use that we don't, and
                                    what we lean on that winners rarely show
    outliers(rows, k)               the posts furthest above their creator's median, and the attributes over-represented
                                    in them against the same creators' other posts (lift, creator-weighted)
    trend(rows, attr)               the attribute's monthly share and the single change point that best splits it
                                    (a binomial likelihood ratio), with the shares before and after
    groups(rows, by)                the outcome by a context's values (median and creators per group)

Rows are recipes.Row (log outcome over the creator's own median; context: medium, platform, archetype, follower band,
month, age band; coded attributes "field=value").
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from psycopg.types.json import Jsonb

from . import recipes

METHODS = ("prevalence", "contrast_sources", "outliers", "trend", "groups", "format_fit", "evolution")
VERSION = "inquiry-v1"


def _by(row: recipes.Row, name: str | None):
    return "all" if not name else recipes.context_of(row, name)


def _creator_share(rows: list[recipes.Row], attr: str) -> tuple[float | None, int, int]:
    """(mean over creators of each creator's share of posts with attr, creators, posts) over rows that know the
    attribute's field."""
    by_creator: dict = defaultdict(list)
    for r in rows:
        h = recipes.has(r, attr)
        if h is not None:
            by_creator[r.creator_id].append(1.0 if h else 0.0)
    if not by_creator:
        return None, 0, 0
    shares = [sum(v) / len(v) for v in by_creator.values()]
    return round(sum(shares) / len(shares), 4), len(by_creator), sum(len(v) for v in by_creator.values())


def _median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 4) if xs else None


def prevalence(rows: list[recipes.Row], attr: str, by: str | None = None) -> dict:
    groups: dict = defaultdict(list)
    for r in rows:
        groups[str(_by(r, by))].append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        share, creators, posts = _creator_share(rs, attr)
        with_ = [r.y for r in rs if recipes.has(r, attr) and r.y is not None]
        without = [r.y for r in rs if recipes.has(r, attr) is False and r.y is not None]
        out[g] = {"share": share, "creators": creators, "posts": posts,
                  "median_log_outcome_with": _median(with_), "median_log_outcome_without": _median(without)}
    return {"method": "prevalence", "attr": attr, "by": by, "groups": out}


def contrast_sources(external: list[recipes.Row], own: list[recipes.Row], attrs: list[str] | None = None, *,
                     min_creators: int = 3, top: int = 12) -> dict:
    """Shares among external winners (y > 0: above their creator's median) vs our own content, largest gaps first."""
    winners = [r for r in external if r.y is not None and r.y > 0]
    universe = sorted(set(attrs) if attrs else {a for r in winners for a in r.attrs} | {a for r in own for a in r.attrs})
    rows = []
    for a in universe:
        w, wc, _ = _creator_share(winners, a)
        o, oc, op = _creator_share(own, a)
        if w is None or o is None or wc < min_creators:
            continue
        rows.append({"attr": a, "winners_share": w, "winner_creators": wc, "own_share": o, "own_items": op,
                     "gap": round(w - o, 4)})
    rows.sort(key=lambda x: -abs(x["gap"]))
    return {"method": "contrast_sources", "winners": len(winners), "own": len(own),
            "under_used": [x for x in rows if x["gap"] > 0][:top], "over_used": [x for x in rows if x["gap"] < 0][:top],
            "note": "a gap is where to experiment, not proof: winners' shares are correlational and creator-weighted"}


def outliers(rows: list[recipes.Row], k: int = 20, *, min_creators: int = 3, top: int = 15) -> dict:
    scored = sorted([r for r in rows if r.y is not None], key=lambda r: -r.y)
    best = scored[:k]
    ids = {r.creator_id for r in best}
    chosen = {r.post_id for r in best}
    rest = [r for r in scored if r.creator_id in ids and r.post_id not in chosen]
    lifts = []
    for a in sorted({a for r in best for a in r.attrs}):
        s_top, c_top, _ = _creator_share(best, a)
        s_rest, c_rest, _ = _creator_share(rest, a)
        if s_top is None or s_rest is None or c_top < min_creators:
            continue
        lifts.append({"attr": a, "share_in_outliers": s_top, "share_otherwise": s_rest,
                      "lift": round((s_top + 0.01) / (s_rest + 0.01), 3), "creators": c_top})
    lifts.sort(key=lambda x: -x["lift"])
    return {"method": "outliers", "k": len(best), "creators": len(ids),
            "posts": [{"post_id": r.post_id, "creator_id": r.creator_id, "x_own_median": round(math.exp(r.y), 2),
                       "medium": r.medium} for r in best],
            "over_represented": lifts[:top], "under_represented": sorted(lifts, key=lambda x: x["lift"])[:top]}


def _ll(k: int, n: int) -> float:
    if n == 0 or k in (0, n):
        return 0.0
    p = k / n
    return k * math.log(p) + (n - k) * math.log(1 - p)


def trend(rows: list[recipes.Row], attr: str, *, min_months: int = 4) -> dict:
    months: dict = defaultdict(lambda: [0, 0])
    for r in rows:
        m, h = r.context.get("month"), recipes.has(r, attr)
        if m and h is not None:
            months[m][0] += 1 if h else 0
            months[m][1] += 1
    series = [{"month": m, "share": round(k / n, 4), "posts": n} for m, (k, n) in sorted(months.items())]
    out = {"method": "trend", "attr": attr, "series": series, "change_point": None}
    if len(series) < min_months:
        out["reason"] = f"fewer than {min_months} months with data"
        return out
    ks = [months[s["month"]][0] for s in series]
    ns = [months[s["month"]][1] for s in series]
    whole = _ll(sum(ks), sum(ns))
    best = None
    for i in range(1, len(series)):
        a = _ll(sum(ks[:i]), sum(ns[:i])) + _ll(sum(ks[i:]), sum(ns[i:]))
        gain = 2 * (a - whole)
        if best is None or gain > best[0]:
            best = (gain, i)
    gain, i = best
    before, after = sum(ks[:i]) / sum(ns[:i]), sum(ks[i:]) / sum(ns[i:])
    out["change_point"] = {"from_month": series[i]["month"], "share_before": round(before, 4),
                           "share_after": round(after, 4), "lr_stat": round(gain, 3),
                           "p_approx": round(recipes.chi2_sf(gain, 1), 5),
                           "direction": "rising" if after > before else "falling",
                           "note": "the best of several split points: p is optimistic; a lead to test, not a finding"}
    return out


def groups(rows: list[recipes.Row], by: str) -> dict:
    gs: dict = defaultdict(list)
    for r in rows:
        if r.y is not None:
            gs[str(_by(r, by))].append(r)
    return {"method": "groups", "by": by, "groups": {
        g: {"median_x_own_median": round(math.exp(statistics.median([r.y for r in rs])), 3),
            "posts": len(rs), "creators": len({r.creator_id for r in rs})} for g, rs in sorted(gs.items())}}


def format_fit(rows: list[recipes.Row], by: str = "proposition", *, min_creators: int = 3) -> dict:
    """Which formats carry which creators (owner 2026-11-26, creator-content fit): within each group of a creator-level
    context (proposition by default), each medium's share of posts and its posts' median outcome over their own
    creator's median. A medium a group's creators rarely use, or that underperforms their own median, is a
    hypothesis that it doesn't fit them; never a rule (public posts, correlation, current window only)."""
    groups: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.y is not None:
            groups[str(_by(r, by))][r.medium].append(r)
    out = {}
    for g, media in sorted(groups.items()):
        total = sum(len(v) for v in media.values())
        out[g] = {m: {"share": round(len(rs) / total, 3), "posts": len(rs), "creators": len({r.creator_id for r in rs}),
                      "median_x_own_median": round(math.exp(statistics.median([r.y for r in rs])), 3)}
                  for m, rs in sorted(media.items()) if len({r.creator_id for r in rs}) >= min_creators}
    return {"method": "format_fit", "by": by, "groups": out}


def evolution(rows: list[recipes.Row], *, min_posts: int = 6) -> dict:
    """How creators' format strategies change over time (owner 2026-11-26), as far as the posts held reach back: per
    creator, each medium's share in the earlier and the later half of their posts by month, and the outcome in each
    half; then, across creators, whether those who moved toward a medium did better afterwards than before. A lead for
    questions, not a finding: the window is what was collected, usually months, never a career."""
    by_creator: dict = defaultdict(list)
    for r in rows:
        if r.y is not None and r.context.get("month"):
            by_creator[r.creator_id].append(r)
    shifts: dict = defaultdict(list)
    creators = {}
    for cid, rs in by_creator.items():
        if len(rs) < min_posts:
            continue
        rs = sorted(rs, key=lambda r: r.context["month"])
        half = len(rs) // 2
        early, late = rs[:half], rs[half:]
        media = sorted({r.medium for r in rs})
        share = lambda part, m: sum(r.medium == m for r in part) / len(part)   # noqa: E731
        moved = {m: round(share(late, m) - share(early, m), 3) for m in media}
        gain = round(statistics.median([r.y for r in late]) - statistics.median([r.y for r in early]), 4)
        creators[cid] = {"posts": len(rs), "from": early[0].context["month"], "to": late[-1].context["month"],
                         "share_change": moved, "outcome_change_log": gain}
        for m, d in moved.items():
            if abs(d) >= 0.2:
                shifts[f"toward_{m}" if d > 0 else f"away_from_{m}"].append(gain)
    return {"method": "evolution", "creators": creators,
            "shifts": {k: {"creators": len(v), "median_outcome_change_log": round(statistics.median(v), 4)}
                       for k, v in sorted(shifts.items())},
            "note": "outcomes are each post over its creator's own median, so a change is relative to the creator; "
                    "who chose to shift is not random (selection), so read it as a lead"}


def run(method: str, rows: list[recipes.Row], args: dict, *, own_rows: list[recipes.Row] | None = None) -> dict:
    if method == "prevalence":
        return prevalence(rows, str(args["attr"]), args.get("by"))
    if method == "contrast_sources":
        return contrast_sources(rows, own_rows or [], args.get("attrs"))
    if method == "outliers":
        return outliers(rows, int(args.get("k") or 20))
    if method == "trend":
        return trend(rows, str(args["attr"]))
    if method == "groups":
        return groups(rows, str(args.get("by") or "medium"))
    if method == "format_fit":
        return format_fit(rows, str(args.get("by") or "proposition"))
    if method == "evolution":
        return evolution(rows)
    raise ValueError(f"unknown method {method!r}")


def record(conn, policy, *, requester: str, method: str, spec: dict, result: dict, question_id=None,
           family: str | None = None, family_size: int = 1, data: dict | None = None) -> int:
    """One analysis in the trial ledger (analysis_runs). Returns its id."""
    import json
    row = conn.execute(
        "INSERT INTO analysis_runs (requester, question_id, method, method_version, spec, family, family_size, data, "
        "result, policy_version) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
        (requester, question_id, method, VERSION if method in METHODS else "recipes", Jsonb(spec), family,
         int(family_size), Jsonb(data or {}), Jsonb(json.loads(json.dumps(result, default=str))),
         getattr(policy, "version", None))).fetchone()
    return int(row["id"])


def trials(conn, family: str) -> int:
    """How many analyses a family has run (every one counts toward its multiple-testing denominator)."""
    return int(conn.execute("SELECT coalesce(sum(family_size), 0) n FROM analysis_runs WHERE family = %s",
                            (family,)).fetchone()["n"])
