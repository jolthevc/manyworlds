"""The music department's learning loop (spec 2026-11-23-music-department.md, section 2): which song features go
with outcomes, what good looks like per artist and genre, which past songs a new one resembles, and checkable
predictions about a song. Numbers in code; an LLM never estimates an effect.

    song_rows(conn, policy, mode=, creator_id=, genre=)   songs with features and their outcome (below)
    effects(rows, policy)                                   per feature, category and feature mix: effect, interval, grade
    analyse(conn, policy, scope=, ...) / record(...)        an analysis per artist, genre or the portfolio; findings
                                                            (domain music) replacing that scope's previous ones
    bands(conn, policy, creator_id=, genre=)                advisory "what good looks like" bands; position(...)
    nearest(conn, policy, song_id)                          the most alike songs (standardised features) with outcomes
    song_context(conn, policy, song_id)                     everything the roles and the critic read about one song
    file_prediction(conn, policy, role=, song_id=, ...)     a checkable claim about a song (brain.predictions)

A song's outcome: each post using the song gets the analyst's log-lift over its creator's own baseline
(brain.analysis.normalise: the median reach of the creator's previous posts, songs or not), net of the mean lift of
its format across the data (a music video is not compared with a photo), capped at the breakout ratio; the song's
outcome is the mean over its posts. Outcomes come only through manyworlds.brain.evidence (song_posts): `real`
never sees simulated data; `sandbox` findings are recorded as simulated. Effects reuse the analyst's machinery
(brain.analysis: slopes and differences with bootstrap errors, a Bonferroni-corrected interval for `tentative`),
with the thresholds in policy music.analysis. Bands are advice to the roles and the critic, never a gate on media.
"""

from __future__ import annotations

import math
import random
import statistics
from statistics import NormalDist
from typing import Any

from .. import song_measure, songs
from ..policy import Policy
from . import analysis, evidence, predictions

DOMAIN = "music"


def _cfg(policy: Policy) -> dict:
    return policy.get("music", "analysis", default={}) or {}


# --- data ----------------------------------------------------------------------------------------------------------

def _song_list(conn, *, creator_id=None, genre=None, song_ids=None) -> list[dict]:
    ids = [str(s) for s in song_ids] if song_ids is not None else None
    return conn.execute(
        "SELECT id, creator_id, genre, title, version, parent_id, created_at FROM songs WHERE creator_id IS NOT NULL "
        "AND (%s::uuid IS NULL OR creator_id = %s) AND (%s::text IS NULL OR genre = %s) "
        "AND (%s::uuid[] IS NULL OR id = ANY(%s::uuid[])) ORDER BY created_at",
        (creator_id, creator_id, genre, genre, ids, ids)).fetchall()


def outcomes(conn, policy: Policy, creator_ids, *, mode: evidence.Mode) -> dict[str, dict]:
    """{song_id: {y, n_posts, reach_median, saves_median, shares_median}} for the songs used by these creators'
    posts: see the module docstring for y."""
    posts = evidence.song_posts(conn, mode=mode, creator_ids=list(creator_ids), policy=policy)
    by_creator: dict[str, list[dict]] = {}
    for p in posts:
        by_creator.setdefault(p["creator_id"], []).append(p)
    lifted = [r for rows in by_creator.values() for r in analysis.normalise(rows, policy) if r.get("lift") is not None]
    if not lifted:
        return {}
    cap = math.log(float(policy.get("brain", "analysis", "breakout_ratio", default=3.0)))
    by_format: dict[str, list[float]] = {}
    for r in lifted:
        by_format.setdefault(r["format"], []).append(r["lift"])
    fmean = {f: statistics.fmean(v) for f, v in by_format.items()}
    per_song: dict[str, list[dict]] = {}
    for r in lifted:
        if r.get("song_id"):
            per_song.setdefault(r["song_id"], []).append({**r, "y": max(-cap, min(cap, r["lift"] - fmean[r["format"]]))})
    out = {}
    for sid, rs in per_song.items():
        def med(k):
            vals = [float(x[k]) for x in rs if x.get(k) is not None]
            return round(statistics.median(vals), 2) if vals else None
        out[sid] = {"y": round(statistics.fmean(x["y"] for x in rs), 4), "n_posts": len(rs),
                    "reach_median": med("reach"), "saves_median": med("saves"), "shares_median": med("shares"),
                    **({"label": evidence.LABEL} if mode == "sandbox" else {})}
    return out


def song_rows(conn, policy: Policy, *, mode: evidence.Mode, creator_id=None, genre=None, song_ids=None) -> list[dict]:
    """Songs (of one artist, one genre, or all) with their latest features and, where posts using them have been
    observed at the target age, their outcome (`y` is None otherwise)."""
    rows = _song_list(conn, creator_id=creator_id, genre=genre, song_ids=song_ids)
    if not rows:
        return []
    feats = songs.latest_features(conn, [r["id"] for r in rows])
    outs = outcomes(conn, policy, {str(r["creator_id"]) for r in rows}, mode=mode)
    out = []
    for r in rows:
        sid = str(r["id"])
        o = outs.get(sid) or {}
        out.append({"song_id": sid, "creator_id": str(r["creator_id"]), "genre": r["genre"], "title": r["title"],
                    "version": r["version"], "features": feats.get(sid) or {}, "y": o.get("y"),
                    "outcome": o or None})
    return out


# --- effects -------------------------------------------------------------------------------------------------------

def _z(values: list[float]) -> list[float] | None:
    if len(values) < 2:
        return None
    m, s = statistics.fmean(values), statistics.pstdev(values)
    return None if s == 0 else [(v - m) / s for v in values]


def effects(rows: list[dict], policy: Policy, *, seed: int = 0) -> dict:
    """Effects of song features on outcome (`y`, log-lift): per numeric feature the slope per standard deviation,
    per category value the difference from the rest, per feature mix (music.analysis.mix_features pairs) the slope on
    the product of the standardised values. Graded as the analyst grades (Bonferroni across everything tested)."""
    cfg = _cfg(policy)
    usable = [r for r in rows if r.get("y") is not None]
    n, min_any = len(usable), int(cfg.get("min_n_any", 8))
    result: dict[str, Any] = {"n": n, "insufficient": n < min_any, "effects": []}
    if result["insufficient"]:
        return result
    min_group = int(cfg.get("min_group", 4))
    candidates: list[tuple[str, str, list, Any]] = []
    zs: dict[str, dict[str, float]] = {}
    for name in song_measure.numeric_names(policy):
        have = [r for r in usable if isinstance(r["features"].get(name), (int, float))]
        if len(have) < min_any:
            continue
        z = _z([float(r["features"][name]) for r in have])
        if z is None:
            continue
        zs[name] = {r["song_id"]: v for r, v in zip(have, z)}
        candidates.append(("feature", name, [(v, r["y"]) for r, v in zip(have, z)], analysis._slope))
    cats = [f["name"] for f in song_measure.registry(policy) if f.get("kind") == "category"]
    for name in cats:
        values = sorted({str(r["features"][name]) for r in usable if r["features"].get(name) is not None})
        for v in values:
            pairs = [(str(r["features"].get(name)) == v, r["y"]) for r in usable if r["features"].get(name) is not None]
            k = sum(g for g, _ in pairs)
            if k >= min_group and len(pairs) - k >= min_group:
                candidates.append(("category", f"{name}={v}", pairs, analysis._diff))
    mix = [m for m in cfg.get("mix_features") or [] if m in zs]
    for i, a in enumerate(mix):
        for b in mix[i + 1:]:
            pairs = [(zs[a][r["song_id"]] * zs[b][r["song_id"]], r["y"]) for r in usable
                     if r["song_id"] in zs[a] and r["song_id"] in zs[b]]
            if len(pairs) >= min_any:
                candidates.append(("mix", f"{a}*{b}", pairs, analysis._slope))
    m = max(1, len(candidates))
    z_family = NormalDist().inv_cdf(1 - float(cfg.get("alpha", 0.1)) / (2 * m))
    rng, draws = random.Random(seed), int(cfg.get("bootstrap", 300))
    min_tentative = int(cfg.get("min_n_tentative", 16))
    for kind, key, pairs, stat in candidates:
        est, se = stat(pairs), analysis._bootstrap_se(pairs, stat, draws, rng)
        if est is None or se is None:
            continue
        low, high = est - z_family * se, est + z_family * se
        smallest = len(pairs) if stat is analysis._slope else min(sum(1 for g, _ in pairs if g),
                                                                  sum(1 for g, _ in pairs if not g))
        tentative = len(pairs) >= min_tentative and smallest >= min_group and (low > 0 or high < 0)
        result["effects"].append({"kind": kind, "key": key, "effect": round(est, 4), "se": round(se, 4),
                                  "low": round(low, 4), "high": round(high, 4), "n": len(pairs),
                                  "grade": "tentative" if tentative else "speculative"})
    result["effects"].sort(key=lambda e: -abs(e["effect"]) / (e["se"] or 1))
    result["tested"] = m
    return result


def analyse(conn, policy: Policy, *, scope: str, creator_id=None, genre: str | None = None,
            mode: evidence.Mode | None = None, seed: int = 0) -> dict:
    """One analysis: scope 'creator' (one artist), 'genre' or 'portfolio'. The mode defaults to the artist's own
    (creator scope) and to real otherwise."""
    if scope not in ("creator", "genre", "portfolio"):
        raise ValueError(f"unknown music analysis scope {scope!r}")
    if scope == "creator" and creator_id is None or scope == "genre" and not genre:
        raise ValueError(f"a {scope} analysis needs its subject")
    mode = mode or (evidence.mode_for(conn, creator_id) if scope == "creator" else "real")
    rows = song_rows(conn, policy, mode=mode, creator_id=creator_id if scope == "creator" else None,
                     genre=genre if scope == "genre" else None)
    result = effects(rows, policy, seed=seed)
    return {**result, "scope": scope, "mode": mode, "creator_id": str(creator_id) if creator_id else None,
            "genre": genre, "songs": [r["song_id"] for r in rows if r.get("y") is not None],
            "creators": sorted({r["creator_id"] for r in rows if r.get("y") is not None})}


def _unit(policy: Policy, name: str) -> str:
    return next((str(f.get("unit") or "") for f in song_measure.registry(policy) if f["name"] == name), "")


def statement(e: dict, policy: Policy, *, subject: str) -> str:
    """A finding's words, from its numbers alone."""
    better = "better" if e["effect"] > 0 else "worse"
    tail = (f"({e['effect']:+.2f} log-lift, interval {e['low']:+.2f} to {e['high']:+.2f}, n={e['n']} songs, "
            f"{e['grade']})")
    if e["kind"] == "feature":
        unit = _unit(policy, e["key"])
        return f"{subject}: songs with a higher {e['key']}{f' ({unit})' if unit else ''} did {better} per SD {tail}"
    if e["kind"] == "category":
        return f"{subject}: songs with {e['key']} did {better} than the rest {tail}"
    a, b = e["key"].split("*", 1)
    return f"{subject}: {a} and {b} together (both high or both low) went with {better} outcomes {tail}"


def record(conn, policy: Policy, result: dict) -> list[str]:
    """The analysis as findings (domain music), replacing the open ones of the same scope and subject: every
    tentative effect and the strongest music.analysis.max_speculative speculative ones. Sandbox data is recorded
    as simulated (evidence.record_finding checks it). No finding when the evidence is insufficient."""
    scope, mode = result["scope"], result["mode"]
    cid = result.get("creator_id") if scope == "creator" else None
    genre = result.get("genre") if scope == "genre" else None
    evidence.supersede(conn, creator_id=cid, domain=DOMAIN, scope=scope, genre=genre)
    if result.get("insufficient"):
        return []
    subject = {"creator": "this artist", "genre": f"{genre} songs", "portfolio": "across the portfolio"}[scope]
    keep = [e for e in result["effects"] if e["grade"] == "tentative"]
    keep += [e for e in result["effects"] if e["grade"] == "speculative"][:int(_cfg(policy).get("max_speculative", 3))]
    kind = "simulated" if mode == "sandbox" else "observational"
    sources = [{"creator_id": c} for c in result.get("creators") or []]
    if mode == "sandbox":
        sources.append({"table": "simulated_metrics"})
    ids = []
    for e in keep:
        fid = evidence.record_finding(
            conn, scope=scope, creator_id=cid, statement=statement(e, policy, subject=subject), evidence=kind,
            grade=e["grade"], effect=e["effect"], interval=(e["low"], e["high"]), n=e["n"], domain=DOMAIN,
            genre=genre, sources=sources + [{"effect": f"{e['kind']}:{e['key']}", "songs": result.get("songs") or []}])
        ids.append(str(fid))
    return ids


def findings(conn, *, mode: evidence.Mode, creator_id=None, genre=None, scope=None) -> list[dict]:
    """The music findings a role may read (through evidence)."""
    return evidence.findings(conn, mode=mode, domain=DOMAIN, creator_id=creator_id, genre=genre, scope=scope)


# --- bands ---------------------------------------------------------------------------------------------------------

def _quantile(values: list[float], q: float) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _from_best(rows: list[dict], policy: Policy) -> dict[str, list[float]]:
    b = _cfg(policy).get("bands") or {}
    scored = sorted([r for r in rows if r.get("y") is not None], key=lambda r: -r["y"])
    best = scored[:max(1, math.ceil(len(scored) * float(b.get("best_share", 0.34))))]
    qlo, qhi = b.get("quantiles") or (0.2, 0.8)
    out = {}
    for name in b.get("features") or []:
        vals = [float(r["features"][name]) for r in best if isinstance(r["features"].get(name), (int, float))]
        if vals:
            out[name] = [round(_quantile(vals, float(qlo)), 3), round(_quantile(vals, float(qhi)), 3)]
    return out


def bands(conn, policy: Policy, *, creator_id=None, genre: str | None = None, mode: evidence.Mode | None = None) -> dict:
    """What good looks like (advisory): {source: artist | genre_history | genre_playbook | None, n, bands:
    {feature: [low, high]}}. The artist's best songs once it has music.analysis.bands.min_songs with outcomes, else
    the genre's best songs, else the genre's starting bands (music.genre_bands)."""
    need = int((_cfg(policy).get("bands") or {}).get("min_songs", 8))
    if creator_id is not None:
        rows = song_rows(conn, policy, mode=mode or evidence.mode_for(conn, creator_id), creator_id=creator_id)
        scored = [r for r in rows if r.get("y") is not None]
        if len(scored) >= need:
            return {"source": "artist", "n": len(scored), "bands": _from_best(scored, policy)}
    if genre:
        rows = song_rows(conn, policy, mode=mode or "real", genre=genre)
        scored = [r for r in rows if r.get("y") is not None]
        if len(scored) >= need:
            return {"source": "genre_history", "n": len(scored), "bands": _from_best(scored, policy)}
        playbook = (policy.get("music", "genre_bands", default={}) or {}).get(str(genre).lower())
        if playbook:
            return {"source": "genre_playbook", "n": 0, "bands": {k: list(v) for k, v in playbook.items()}}
    return {"source": None, "n": 0, "bands": {}}


def position(features: dict, band: dict) -> list[dict]:
    """Where a song sits against bands: [{feature, value, band, position: below | within | above}]."""
    out = []
    for name, (lo, hi) in (band.get("bands") or {}).items():
        v = features.get(name)
        if not isinstance(v, (int, float)):
            continue
        out.append({"feature": name, "value": v, "band": [lo, hi],
                    "position": "below" if v < lo else "above" if v > hi else "within"})
    return out


# --- similarity ----------------------------------------------------------------------------------------------------

def nearest(conn, policy: Policy, song_id, *, k: int | None = None, mode: evidence.Mode | None = None) -> list[dict]:
    """The k songs most like this one (the artist's and the genre's, other than itself), by distance over
    standardised features (registry numbers, standardised across the pool; at least min_shared in common), each with
    its outcome where observed and the features it is most alike on ("C may do well because it shares A's ...")."""
    sim = _cfg(policy).get("similarity") or {}
    k = int(k if k is not None else sim.get("k", 5))
    me = songs.get(conn, song_id)
    if me is None:
        return []
    pool = conn.execute(
        "SELECT id FROM songs WHERE id <> %s AND creator_id IS NOT NULL AND "
        "((%s::uuid IS NOT NULL AND creator_id = %s) OR (%s::text IS NOT NULL AND genre = %s))",
        (song_id, me["creator_id"], me["creator_id"], me["genre"], me["genre"])).fetchall()
    ids = [str(r["id"]) for r in pool]
    if not ids:
        return []
    feats = songs.latest_features(conn, [str(song_id), *ids])
    mine = feats.get(str(song_id)) or {}
    names = [n for n in song_measure.numeric_names(policy) if isinstance(mine.get(n), (int, float))]
    stats = {}
    for n in names:
        vals = [float(f[n]) for f in feats.values() if isinstance(f.get(n), (int, float))]
        if len(vals) >= 2 and statistics.pstdev(vals) > 0:
            stats[n] = (statistics.fmean(vals), statistics.pstdev(vals))
    mode = mode or (evidence.mode_for(conn, me["creator_id"]) if me["creator_id"] else "real")
    rows = {r["song_id"]: r for r in song_rows(conn, policy, mode=mode, song_ids=ids)}
    out = []
    for sid in ids:
        other = feats.get(sid) or {}
        diffs = {n: ((float(mine[n]) - m) / s) - ((float(other[n]) - m) / s)
                 for n, (m, s) in stats.items() if isinstance(other.get(n), (int, float))}
        if len(diffs) < int(sim.get("min_shared", 6)):
            continue
        r = rows.get(sid) or {}
        out.append({"song_id": sid, "title": r.get("title"), "creator_id": r.get("creator_id"),
                    "genre": r.get("genre"),
                    "distance": round(math.sqrt(sum(d * d for d in diffs.values()) / len(diffs)), 4),
                    "shared": len(diffs), "alike_on": sorted(diffs, key=lambda n: abs(diffs[n]))[:3],
                    "outcome": r.get("outcome")})
    return sorted(out, key=lambda x: (x["distance"], x["song_id"]))[:k]


def song_context(conn, policy: Policy, song_id, *, nearest_k: int | None = None) -> dict:
    """What the roles and the critic read about one song: its row, features, bands and where it sits in them, the
    nearest songs with their outcomes, and the open music findings for its artist and genre (in the artist's mode)."""
    s = songs.get(conn, song_id)
    if s is None:
        raise KeyError(f"no song {song_id}")
    mode = evidence.mode_for(conn, s["creator_id"]) if s["creator_id"] else "real"
    feats = songs.latest_features(conn, [song_id]).get(str(song_id)) or {}
    band = bands(conn, policy, creator_id=s["creator_id"], genre=s["genre"], mode=mode)
    found = (findings(conn, mode=mode, creator_id=s["creator_id"]) if s["creator_id"] else []) + \
        (findings(conn, mode=mode, genre=s["genre"], scope="genre") if s["genre"] else [])
    return {"song": {"id": str(s["id"]), "title": s["title"], "version": s["version"], "genre": s["genre"],
                     "subgenre": s["subgenre"], "brief": s["brief"] or {}, "spec": s["spec"] or {},
                     "lyrics": s["lyrics"], "creator_id": str(s["creator_id"]) if s["creator_id"] else None,
                     "parent_id": str(s["parent_id"]) if s["parent_id"] else None},
            "mode": mode, "features": feats, "bands": band, "position": position(feats, band),
            "nearest": nearest(conn, policy, song_id, k=nearest_k, mode=mode),
            "findings": [{"statement": f["statement"], "grade": f["grade"], "label": f.get("label")} for f in found]}


# --- predictions ---------------------------------------------------------------------------------------------------

def file_prediction(conn, policy: Policy, *, role: str, song_id, claim: str, metric: str, comparator: str,
                    threshold: float, probability: float, horizon_days: int | None = None, decision_id=None):
    """A checkable claim about one song (brain.predictions, metrics song_median_reach | song_reach_lift), scored by
    the chief of staff once due. None when it can't be checked (no artist, unknown metric)."""
    s = songs.get(conn, song_id)
    if s is None or s["creator_id"] is None:
        return None
    predictions.configure(policy)
    days = int(horizon_days if horizon_days is not None
               else policy.get("music", "critic", "prediction", "horizon_days", default=14))
    return predictions.record(conn, role=role, claim=claim, metric=metric, comparator=comparator,
                              threshold=threshold, probability=probability, horizon_days=days,
                              creator_id=s["creator_id"], decision_id=decision_id, song_id=s["id"])
