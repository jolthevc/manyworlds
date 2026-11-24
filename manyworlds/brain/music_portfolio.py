"""The music side of the portfolio (spec 2026-11-23-music-department.md, section 6). Code, not judgement.

Music artists are creators (creators.vertical = 'music') in the same portfolio as content creators: the strategist
(CEO) sees them in its snapshot with their genre; head_of_music runs the music share of the budget
(music.portfolio.budget_share of the month's usable budget) through tiers and requests, never by itself.

    artists(conn, policy)          every music artist: genre, tier, songs (made, released, measured), outcome
    diversification(conn, policy)  feature-space distance between artists (each artist's centroid of standardised
                                   song features; RMS distance), pairs closer than music.portfolio.min_artist_distance
    gaps(policy, artists, div)     genres in music.portfolio.genres nobody covers, and artists too alike
    snapshot(conn, policy)         all of it plus the music budget, for head_of_music and the strategist
    budget(conn, policy)           the music share in dollars and what music artists spent this month

Outcomes are real only (brain.music_analysis in mode real): simulated outcomes never steer money.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from datetime import datetime, timezone

from .. import ledger, song_measure, songs
from ..policy import Policy
from . import music_analysis, portfolio, proposals


def _cfg(policy: Policy) -> dict:
    return policy.get("music", "portfolio", default={}) or {}


def genre_of(conn, creator_id, bible: dict | None = None) -> str | None:
    """The artist's genre: the bible's music.genre, else the music dossier's sonic identity, else its songs' most
    common genre."""
    g = ((bible or {}).get("music") or {}).get("genre") if isinstance((bible or {}).get("music"), dict) else None
    if g:
        return str(g).lower()
    from .agents.music_talent_manager import latest_dossier
    d = latest_dossier(conn, creator_id)
    g = (((d or {}).get("output") or {}).get("sonic_identity") or {}).get("genre")
    if g:
        return str(g).lower()
    rows = conn.execute("SELECT genre FROM songs WHERE creator_id = %s AND genre IS NOT NULL", (creator_id,)).fetchall()
    common = Counter(str(r["genre"]).lower() for r in rows).most_common(1)
    return common[0][0] if common else None


def artists(conn, policy: Policy, *, include_sandbox: bool = False) -> list[dict]:
    rows = conn.execute(
        "SELECT id, handle, status, publish_mode, bible, anchors FROM creators WHERE vertical = 'music' "
        "AND status <> 'retired' AND (%s OR anchors->'simulation' IS NULL) ORDER BY created_at",
        (include_sandbox,)).fetchall()
    out = []
    for r in rows:
        counts = conn.execute(
            "SELECT count(*) FILTER (WHERE status <> 'dropped') made, count(*) FILTER (WHERE release_id IS NOT NULL) "
            "released FROM songs WHERE creator_id = %s", (r["id"],)).fetchone()
        scored = [x for x in music_analysis.song_rows(conn, policy, mode="real", creator_id=r["id"])
                  if x.get("y") is not None]
        out.append({"creator_id": str(r["id"]), "handle": r["handle"], "status": r["status"],
                    "publish_mode": r["publish_mode"], "genre": genre_of(conn, r["id"], r["bible"]),
                    "tier": proposals.tier_of(conn, policy, r["id"]), "songs": counts["made"],
                    "released": counts["released"], "songs_with_outcomes": len(scored),
                    "outcome_mean": round(statistics.fmean(x["y"] for x in scored), 3) if scored else None})
    return out


def _centroids(conn, policy: Policy, artist_ids: list[str]) -> dict[str, dict[str, float]]:
    rows = conn.execute("SELECT id, creator_id FROM songs WHERE creator_id = ANY(%s::uuid[]) AND status <> 'dropped'",
                        (artist_ids,)).fetchall()
    feats = songs.latest_features(conn, [r["id"] for r in rows])
    names = song_measure.numeric_names(policy, sources=tuple(_cfg(policy).get("feature_sources")
                                                             or ("sound", "structure", "vocal", "lyrics")))
    stats = {}
    for n in names:
        vals = [float(f[n]) for f in feats.values() if isinstance(f.get(n), (int, float))]
        if len(vals) >= 2 and statistics.pstdev(vals) > 0:
            stats[n] = (statistics.fmean(vals), statistics.pstdev(vals))
    by: dict[str, list[dict]] = {}
    for r in rows:
        if str(r["id"]) in feats:
            by.setdefault(str(r["creator_id"]), []).append(feats[str(r["id"])])
    out = {}
    for cid, fs in by.items():
        out[cid] = {n: statistics.fmean((float(f[n]) - m) / s for f in fs if isinstance(f.get(n), (int, float)))
                    for n, (m, s) in stats.items() if any(isinstance(f.get(n), (int, float)) for f in fs)}
    return out


def diversification(conn, policy: Policy, artist_rows: list[dict]) -> dict:
    """Pairwise distance between artists' sound (RMS over shared standardised features)."""
    ids = [a["creator_id"] for a in artist_rows]
    cents = _centroids(conn, policy, ids)
    handles = {a["creator_id"]: a["handle"] for a in artist_rows}
    pairs = []
    keys = sorted(cents)
    need = int(_cfg(policy).get("min_shared_features", 4))
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            shared = [n for n in cents[a] if n in cents[b]]
            if len(shared) < need:
                continue
            d = math.sqrt(sum((cents[a][n] - cents[b][n]) ** 2 for n in shared) / len(shared))
            pairs.append({"a": handles[a], "b": handles[b], "distance": round(d, 3), "shared": len(shared)})
    pairs.sort(key=lambda p: p["distance"])
    floor = float(_cfg(policy).get("min_artist_distance", 0.5))
    return {"pairs": pairs, "too_close": [p for p in pairs if p["distance"] < floor],
            "min_distance": pairs[0]["distance"] if pairs else None,
            "mean_distance": round(statistics.fmean(p["distance"] for p in pairs), 3) if pairs else None,
            "measured_artists": len(cents)}


def gaps(policy: Policy, artist_rows: list[dict], div: dict) -> list[dict]:
    covered = {a["genre"] for a in artist_rows if a.get("genre")}
    out = [{"kind": "genre", "genre": g, "why": "no artist in this genre"}
           for g in _cfg(policy).get("genres") or [] if g not in covered]
    out += [{"kind": "too_close", "artists": [p["a"], p["b"]], "distance": p["distance"],
             "why": "these artists sound alike: differentiate one or don't launch a third"} for p in div["too_close"]]
    per = Counter(a["genre"] for a in artist_rows if a.get("genre"))
    cap = int(_cfg(policy).get("max_artists_per_genre", 3))
    out += [{"kind": "crowded", "genre": g, "artists": n, "why": f"more than {cap} artists in one genre"}
            for g, n in per.items() if n > cap]
    return out


def budget(conn, policy: Policy, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    b = portfolio.budget(conn, policy, now)
    share = float(_cfg(policy).get("budget_share", 0.0))
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    spent = float(conn.execute(
        "SELECT coalesce(sum(e.usd), 0) u FROM cost_entries e JOIN creators c ON c.id = e.creator_id "
        "WHERE c.vertical = 'music' AND e.category <> ALL(%s) AND e.created_at >= %s",
        (list(ledger.ATTRIBUTION_ONLY), start)).fetchone()["u"])
    usd = round(float(b["usable_usd"]) * share, 2)
    return {"share": share, "usd": usd, "spent_usd": round(spent, 2), "remaining_usd": round(usd - spent, 2)}


def snapshot(conn, policy: Policy, *, include_sandbox: bool = False, now: datetime | None = None) -> dict:
    rows = artists(conn, policy, include_sandbox=include_sandbox)
    div = diversification(conn, policy, rows)
    return {"artists": rows, "genres": dict(Counter(a["genre"] or "unset" for a in rows)),
            "diversification": div, "gaps": gaps(policy, rows, div), "budget": budget(conn, policy, now)}


def summary(conn, policy: Policy, now: datetime | None = None) -> dict:
    """What the strategist's snapshot carries about music (no per-pair detail beyond the closest)."""
    s = snapshot(conn, policy, now=now)
    return {"artists": [{k: a[k] for k in ("handle", "genre", "tier", "songs", "released", "outcome_mean")}
                        for a in s["artists"]], "genres": s["genres"], "gaps": s["gaps"][:10],
            "closest_pair": (s["diversification"]["pairs"] or [None])[0], "budget": s["budget"]}
