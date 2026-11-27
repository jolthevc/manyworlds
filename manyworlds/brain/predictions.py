"""The predictions ledger: what each role expected, scored later against what happened.

This is how the agency shows that it decides better over time, not just more. A talent manager's conviction and
hypotheses, the CEO's calls and a market lead's thesis are written as checkable claims ("median reach of her posts
in the next 21 days above 800", with a probability). When due, the resolver measures the outcome from published
posts through manyworlds.brain.evidence: real metrics for real creators; sandbox creators resolve against
simulated outcomes and stay labelled `sandbox`, so the real scorecard never mixes them in.

    record()       one claim (metric from METRICS, above/below a threshold, probability, horizon)
    resolve_due()  every due claim measured; outcome, observed value and Brier score ((p - outcome)^2) stored;
                   a claim with no posts to measure stays open until `brain.predictions.give_up_days` after due
    scorecard()    per role and month: n, mean Brier, hit rate, and calibration (claims grouped by probability)

Metrics: median_reach (median reach at the target age, brain.analysis.target_age_s, of posts published in the
window), reach_growth (log of that median over the creator's median before the window). Nothing the agents control
themselves (how many posts they publish) is a metric: a claim about it would only flatter the scorecard. A claim
resolves once the window's last posts have reached the target age, and its horizon is clamped to policy.
Song metrics (a claim about one song: `song_id`): song_median_reach (median reach of the posts using the song,
published in the window), song_reach_lift (log of that median over the creator's median before the window, posts
without the song).
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta, timezone

from ..policy import Policy
from . import evidence

SONG_METRICS = ("song_median_reach", "song_reach_lift")
METRICS = ("median_reach", "reach_growth", *SONG_METRICS)


def record(conn, *, role: str, claim: str, metric: str, comparator: str, threshold: float, probability: float,
           horizon_days: int, creator_id=None, market_id=None, decision_id=None, now: datetime | None = None,
           song_id=None):
    """The prediction's id, or None when it can't be checked (unknown metric or comparator, no subject; a song
    metric without its song)."""
    now = now or datetime.now(timezone.utc)
    if metric not in METRICS or comparator not in ("above", "below") or creator_id is None:
        return None
    if (metric in SONG_METRICS) != (song_id is not None):
        return None
    p = min(max(float(probability), 0.0), 1.0)
    cfg = _cfg_p()
    horizon_days = min(max(int(horizon_days), int(cfg["min_horizon_days"])), int(cfg["max_horizon_days"]))
    mode = evidence.mode_for(conn, creator_id)
    return conn.execute(
        "INSERT INTO predictions (role, creator_id, market_id, claim, metric, comparator, threshold, probability, "
        "horizon_days, mode, decision_id, resolve_at, created_at, song_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, "
        "%s, %s, %s, %s, %s) RETURNING id",
        (role, creator_id, market_id, claim[:500], metric, comparator, float(threshold), p, int(horizon_days), mode,
         decision_id, now + timedelta(days=int(horizon_days)), now, song_id)).fetchone()["id"]


_POLICY: dict = {"min_horizon_days": 7, "max_horizon_days": 180, "target_age_s": 259200}


def _cfg_p() -> dict:
    return _POLICY


def configure(policy: Policy) -> None:
    """Horizon bounds and the reach age from policy (brain.predictions, brain.analysis.target_age_s)."""
    _POLICY.update({k: v for k, v in (policy.get("brain", "predictions", default={}) or {}).items()
                    if k in ("min_horizon_days", "max_horizon_days")})
    _POLICY["target_age_s"] = int(policy.get("brain", "analysis", "target_age_s", default=259200))


def _reach(conn, post_id, mode) -> float | None:
    """Reach at the target age (the first snapshot at or after it): a young post's reach would read as failure."""
    target = int(_POLICY["target_age_s"])
    rows = [r for r in evidence.metrics(conn, post_id, mode=mode) if (r.get("post_age_s") or 0) >= target]
    if not rows:
        return None
    first = min(rows, key=lambda r: r.get("post_age_s") or 0)
    v = (first.get("metrics") or {}).get("reach")
    return float(v) if v is not None else None


def _reaches(conn, creator_id, mode, start, end, *, song_id=None, without_song=None) -> list[float]:
    """Reaches of the creator's posts published in (start, end]: only those using `song_id`, or (without_song)
    only those not using that song."""
    simulated = mode == "sandbox"
    posts = conn.execute("SELECT p.id FROM posts p JOIN publications u ON u.post_id = p.id "
                         "WHERE p.creator_id = %s AND u.simulated = %s AND u.published_at > %s AND u.published_at <= %s "
                         "AND (%s::uuid IS NULL OR p.song_id = %s) "
                         "AND (%s::uuid IS NULL OR p.song_id IS DISTINCT FROM %s)",
                         (creator_id, simulated, start, end, song_id, song_id, without_song, without_song)).fetchall()
    return [r for r in (_reach(conn, p["id"], mode) for p in posts) if r is not None]


def measure(conn, pred: dict) -> float | None:
    start, end, mode = pred["created_at"], pred["resolve_at"], pred["mode"]
    if pred["metric"] not in METRICS:
        return None
    if pred["metric"] in SONG_METRICS:
        song = pred.get("song_id")
        if song is None:
            return None
        window = _reaches(conn, pred["creator_id"], mode, start, end, song_id=song)
        if not window:
            return None
        if pred["metric"] == "song_median_reach":
            return statistics.median(window)
        before = _reaches(conn, pred["creator_id"], mode, start - (end - start), start, without_song=song)
        if not before or statistics.median(before) <= 0 or statistics.median(window) <= 0:
            return None
        return math.log(statistics.median(window) / statistics.median(before))
    window = _reaches(conn, pred["creator_id"], mode, start, end)
    if not window:
        return None
    if pred["metric"] == "median_reach":
        return statistics.median(window)
    before = _reaches(conn, pred["creator_id"], mode, start - (end - start), start)
    if not before or statistics.median(before) <= 0 or statistics.median(window) <= 0:
        return None
    return math.log(statistics.median(window) / statistics.median(before))


def resolve_due(conn, policy: Policy, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    configure(policy)
    give_up = timedelta(days=float(policy.get("brain", "predictions", "give_up_days", default=14)))
    mature = timedelta(seconds=int(_POLICY["target_age_s"]))   # the window's last posts have reached the target age
    out = {"resolved": 0, "unmeasurable": 0}
    for pred in conn.execute("SELECT * FROM predictions WHERE resolved_at IS NULL AND resolve_at <= %s",
                             (now - mature,)).fetchall():
        observed = measure(conn, pred)
        if observed is None:
            if now - pred["resolve_at"] - mature > give_up:   # nothing published to judge it by: closed, no score
                conn.execute("UPDATE predictions SET resolved_at = %s WHERE id = %s", (now, pred["id"]))
                out["unmeasurable"] += 1
            continue
        held = observed > float(pred["threshold"]) if pred["comparator"] == "above" else observed < float(pred["threshold"])
        brier = (float(pred["probability"]) - (1.0 if held else 0.0)) ** 2
        conn.execute("UPDATE predictions SET resolved_at = %s, observed = %s, outcome = %s, brier = %s WHERE id = %s",
                     (now, observed, held, round(brier, 4), pred["id"]))
        out["resolved"] += 1
    return out


def resolved_for(conn, creator_id, *, role: str | None = None, n: int = 10) -> list[dict]:
    """A role's scored claims about one creator, newest first: what it said, how sure it was, what happened and its
    Brier score, so the role sees its own track record. Only claims with an outcome (unmeasurable ones are closed
    without one); a sandbox creator's claims are labelled SIMULATED."""
    rows = conn.execute(
        "SELECT claim, metric, comparator, threshold, probability, observed, outcome, brier, mode, created_at, "
        "resolved_at FROM predictions WHERE creator_id = %s AND outcome IS NOT NULL AND (%s::text IS NULL OR role = %s) "
        "ORDER BY resolved_at DESC LIMIT %s", (creator_id, role, role, int(n))).fetchall()
    return [{"claim": r["claim"], "metric": r["metric"], "comparator": r["comparator"],
             "threshold": float(r["threshold"]), "probability": float(r["probability"]),
             "observed": round(float(r["observed"]), 3) if r["observed"] is not None else None,
             "held": bool(r["outcome"]), "brier": float(r["brier"]) if r["brier"] is not None else None,
             "made": r["created_at"].date().isoformat(), "resolved": r["resolved_at"].date().isoformat(),
             **({"label": evidence.LABEL} if r["mode"] == "sandbox" else {})} for r in rows]


def baseline_briers(conn, *, mode: str) -> dict:
    """Each resolved prediction's Brier under the naive base-rate forecaster: the share of earlier resolved
    predictions on the same metric that held (0.5 before any), so a role's skill is measured against "say what
    usually happens", not only against a coin. {prediction id: brier}."""
    seen: dict = {}
    out = {}
    for r in conn.execute("SELECT id, metric, outcome FROM predictions WHERE mode = %s AND outcome IS NOT NULL "
                          "ORDER BY resolved_at, id", (mode,)).fetchall():
        held, n = seen.get(r["metric"], (0, 0))
        p = held / n if n else 0.5
        out[r["id"]] = (p - (1.0 if r["outcome"] else 0.0)) ** 2
        seen[r["metric"]] = (held + (1 if r["outcome"] else 0), n + 1)
    return out


def scorecard(conn, *, mode: str = "real", role: str | None = None) -> list[dict]:
    """Per role and month: how good the claims were. Lower Brier is better; 0.25 is a coin flip at p = 0.5.
    `brier_base_rate` is the naive base-rate forecaster's on the same predictions, and `skill` = 1 - brier /
    brier_base_rate (above 0: better than saying what usually happens; owner 2026-11-26: compare against simple
    baselines)."""
    base = baseline_briers(conn, mode=mode)
    rows = conn.execute(
        "SELECT role, to_char(created_at, 'YYYY-MM') AS month, count(*) n, avg(brier) brier, "
        "avg(CASE WHEN outcome THEN 1 ELSE 0 END) held, avg(probability) mean_p FROM predictions "
        "WHERE mode = %s AND outcome IS NOT NULL AND metric = ANY(%s) AND (%s::text IS NULL OR role = %s) "
        "GROUP BY 1, 2 ORDER BY 1, 2", (mode, list(METRICS), role, role)).fetchall()
    out = []
    for r in rows:
        buckets = conn.execute(
            "SELECT width_bucket(probability, 0, 1, 5) b, count(*) n, avg(probability) p, "
            "avg(CASE WHEN outcome THEN 1 ELSE 0 END) held FROM predictions WHERE mode = %s AND role = %s "
            "AND to_char(created_at, 'YYYY-MM') = %s AND outcome IS NOT NULL AND metric = ANY(%s) GROUP BY 1 ORDER BY 1",
            (mode, r["role"], r["month"], list(METRICS))).fetchall()
        ids = [x["id"] for x in conn.execute(
            "SELECT id FROM predictions WHERE mode = %s AND role = %s AND to_char(created_at, 'YYYY-MM') = %s "
            "AND outcome IS NOT NULL AND metric = ANY(%s)", (mode, r["role"], r["month"], list(METRICS))).fetchall()]
        bb = [base[i] for i in ids if i in base]
        brier_base = round(sum(bb) / len(bb), 4) if bb else None
        skill = (round(1 - float(r["brier"]) / brier_base, 3) if brier_base else None)
        out.append({"role": r["role"], "month": r["month"], "n": r["n"], "brier": round(float(r["brier"]), 4),
                    "hit_rate": round(float(r["held"]), 3), "mean_probability": round(float(r["mean_p"]), 3),
                    "brier_base_rate": brier_base, "skill": skill,
                    "calibration": [{"said": round(float(b["p"]), 2), "happened": round(float(b["held"]), 2),
                                     "n": b["n"]} for b in buckets],
                    **({"label": evidence.LABEL} if mode == "sandbox" else {})})
    return out
