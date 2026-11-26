"""Where a batch's exploration goes: Thompson sampling over what is known and what is untried.

Each candidate direction gets a belief about its effect on reach (log-lift): effects the analyst measured
use their estimate and standard error; untried directions (formats and stagings the creator hasn't used)
use a prior centred on zero with `prior_sd`. One draw from each belief; the highest draws are this batch's
exploration directions. Uncertain-but-promising directions keep getting tried, clear losers rarely do, and
nothing is ruled out on thin evidence. The draws are recorded on the plan decision, so the trace shows why
a batch explored what it did.

How much of a batch explores is the exploration temperature (`shares`, `batch_targets`, `reconcile`: below).
"""

from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta, timezone

from ..policy import Policy

KINDS = ("attribute", "format", "dimension", "staging", "flavor")


def _cfg(policy: Policy) -> dict:
    return policy.get("brain", "exploration", default={}) or {}


def describe(direction: str) -> str:
    kind, _, key = direction.partition(":")
    if kind == "attribute" and "=" in key:
        k, v = key.split("=", 1)
        return f"{k}: {v}"
    return {"format": f"the {key} format", "dimension": f"ideas strong on {key}",
            "staging": f"the {key} staging", "flavor": f"a {key.replace('_', ' ')} post"}.get(kind, direction)


def directions(policy: Policy, *, effects: list[dict], untried: list[str], seed_key: str) -> list[dict]:
    """The top `directions` Thompson draws. `effects` are analysis effects (kind, key, effect, se)."""
    cfg = _cfg(policy)
    prior = float(cfg.get("prior_sd", 0.3))
    rng = random.Random(int(hashlib.sha256(seed_key.encode()).hexdigest()[:12], 16))
    beliefs = {}
    for e in effects:
        if e.get("kind") in KINDS and e.get("se"):
            beliefs[f"{e['kind']}:{e['key']}"] = (float(e["effect"]), float(e["se"]))
    for d in untried:
        beliefs.setdefault(d, (0.0, prior))
    draws = [{"direction": d, "mean": round(m, 3), "sd": round(s, 3), "draw": round(rng.gauss(m, s), 3)}
             for d, (m, s) in sorted(beliefs.items())]
    draws.sort(key=lambda x: -x["draw"])
    return [{**d, "describe": describe(d["direction"])} for d in draws[:int(cfg.get("directions", 3))]]


def untried(conn, creator_id, *, formats, stagings, flavors=()) -> list[str]:
    """Formats, stagings and flavors production can make that this creator has never had an idea for."""
    used_formats = {r["format"] for r in conn.execute(
        "SELECT DISTINCT format FROM ideas WHERE creator_id = %s", (creator_id,)).fetchall()}
    used_stagings = {r["s"] for r in conn.execute(
        "SELECT DISTINCT attributes->>'staging' AS s FROM ideas WHERE creator_id = %s", (creator_id,)).fetchall()}
    used_flavors = {r["f"] for r in conn.execute(
        "SELECT DISTINCT attributes->>'flavor' AS f FROM ideas WHERE creator_id = %s", (creator_id,)).fetchall()}
    return [f"format:{f}" for f in formats if f not in used_formats] + \
           [f"staging:{s}" for s in stagings if s not in used_stagings] + \
           [f"flavor:{f}" for f in flavors if f not in used_flavors]


# --- the exploration temperature (owner 2026-11-26; policy brain.exploration_mix) -------------------------------------
# Every idea is labelled refine | adjacent | novel. The shares are set per creator by stage (and plateau), may be set
# for one creator through brain.authority, and are nudged toward the portfolio total (brain.exploration_share). A
# batch's exact counts close the gap her recent batches left; code trims an oversized batch and flags ideas over
# their level's count instead of asking the model again. Labels are never rewritten: what was planned is what is
# measured (ideas.attributes.exploration, joined to outcomes by evidence.attribution).
LEVELS = ("refine", "adjacent", "novel")
EXPLORING = ("adjacent", "novel")


def _mix_cfg(policy: Policy) -> dict:
    return policy.get("brain", "exploration_mix", default={}) or {}


def stage_of(conn, creator_id) -> str | None:
    row = conn.execute("SELECT stage FROM creator_states WHERE creator_id = %s", (creator_id,)).fetchone()
    return row["stage"] if row else None


def _window_counts(conn, since, creator_id=None) -> dict[str, int]:
    rows = conn.execute(
        "SELECT attributes->>'exploration' AS level, count(*) AS n FROM ideas "
        "WHERE created_at > %s AND (%s::uuid IS NULL OR creator_id = %s) AND attributes ? 'exploration' "
        "GROUP BY 1", (since, creator_id, creator_id)).fetchall()
    out = {lv: 0 for lv in LEVELS}
    for r in rows:
        if r["level"] in out:
            out[r["level"]] = int(r["n"])
    return out


def _plateaued(conn, policy: Policy, creator_id, cfg: dict, stage: str | None) -> bool:
    """A creator in one of plateau.stages whose real follower count did not grow over plateau.growth_days (account
    insights through evidence.growth; a sandbox creator has none, so never)."""
    from . import evidence
    p = cfg.get("plateau") or {}
    if stage not in (p.get("stages") or []):
        return False
    g = evidence.growth(conn, creator_id, evidence.mode_for(conn, creator_id), days=int(p.get("growth_days", 28)))
    return bool(g) and g.get("net_followers") is not None and g["net_followers"] <= 0


def portfolio_share(conn, policy: Policy, *, now: datetime | None = None) -> dict:
    """The portfolio's realised adjacent + novel share over portfolio.window_days, beside its target."""
    cfg = _mix_cfg(policy).get("portfolio") or {}
    now = now or datetime.now(timezone.utc)
    counts = _window_counts(conn, now - timedelta(days=float(cfg.get("window_days", 14))))
    n = sum(counts.values())
    return {"target": float(policy.get("brain", "exploration_share", default=0.3)), "ideas": n, "counts": counts,
            "realised": round(sum(counts[x] for x in EXPLORING) / n, 4) if n else None}


def shares(conn, policy: Policy, creator_id, *, now: datetime | None = None) -> dict:
    """{stage, plateau, own, shares {refine, adjacent, novel}, correction, portfolio}: her stage's shares (plateau
    shares, where larger, on a plateau), or her own (exploration_mix.creators.<id>, set through brain.authority), the
    stage's scaled toward the portfolio total by at most portfolio.max_correction; refine is the rest."""
    cfg = _mix_cfg(policy)
    by = cfg.get("by_stage") or {}
    stage = stage_of(conn, creator_id)
    base = dict(by.get(stage) or by.get("default") or {})
    plateau = _plateaued(conn, policy, creator_id, cfg, stage)
    if plateau:
        p = cfg.get("plateau") or {}
        base = {k: max(float(base.get(k, 0)), float(p.get(k, 0))) for k in EXPLORING}
    own = {k: float(v) for k, v in ((cfg.get("creators") or {}).get(str(creator_id)) or {}).items() if k in EXPLORING}
    port = portfolio_share(conn, policy, now=now)
    pc = cfg.get("portfolio") or {}
    factor = 1.0
    if not own and port["ideas"] >= int(pc.get("min_ideas", 20)) and port["realised"] is not None:
        mc = float(pc.get("max_correction", 0.25))
        raw = port["target"] / port["realised"] if port["realised"] > 0 else 1.0 + mc
        factor = max(1.0 - mc, min(1.0 + mc, raw))
    out = {k: max(0.0, float(base.get(k, 0)) * factor) for k in EXPLORING}
    out.update(own)   # her own setting stands as set (within the authority's bounds)
    total = sum(out.values())
    if total > 1.0:
        out = {k: v / total for k, v in out.items()}
    out = {"refine": round(max(0.0, 1.0 - sum(out.values())), 4), **{k: round(v, 4) for k, v in out.items()}}
    return {"stage": stage, "plateau": plateau, "own": bool(own), "shares": out, "correction": round(factor, 4),
            "portfolio": port}


def allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    """`total` split over LEVELS in proportion to `weights` by largest remainder (ties: refine first)."""
    total = max(int(total), 0)
    w = {lv: max(0.0, float(weights.get(lv, 0))) for lv in LEVELS}
    s = sum(w.values())
    if total == 0 or s <= 0:
        return {lv: (total if lv == "refine" else 0) for lv in LEVELS}
    exact = {lv: total * w[lv] / s for lv in LEVELS}
    out = {lv: int(exact[lv]) for lv in LEVELS}
    for lv in sorted(LEVELS, key=lambda x: (-(exact[x] - out[x]), LEVELS.index(x)))[:total - sum(out.values())]:
        out[lv] += 1
    return out


def batch_targets(conn, policy: Policy, creator_id, n: int, *, now: datetime | None = None) -> dict:
    """This batch's exact counts per level for `n` ideas: what closes the gap between her shares and what her last
    window_days of ideas were labelled, so the mix holds over batches, not only within one. -> shares() plus
    {window (counts), targets}."""
    now = now or datetime.now(timezone.utc)
    sh = shares(conn, policy, creator_id, now=now)
    have = _window_counts(conn, now - timedelta(days=float(_mix_cfg(policy).get("window_days", 14))), creator_id)
    ideal = allocate(sum(have.values()) + int(n), sh["shares"])
    need = {lv: max(0, ideal[lv] - have[lv]) for lv in LEVELS}
    targets = allocate(n, need if sum(need.values()) else sh["shares"])
    return {**sh, "window": have, "targets": targets}


def reconcile(levels: list[str], targets: dict[str, int], size: int, *, keep=frozenset()) -> dict:
    """Fit a planned batch to its targets without another model call. `levels`: each idea's label, in order. Over
    `size` ideas, the surplus is trimmed from the level most over its count (its last ideas first; never an index in
    `keep`, e.g. an experiment slot). Ideas still over their level's count are flagged. -> {kept, trimmed, flagged,
    planned (counts), shortfall (counts)}."""
    idx = list(range(len(levels)))
    trimmed: list[int] = []
    while len(idx) > size:
        counts = {lv: sum(levels[i] == lv for i in idx) for lv in LEVELS}
        over = sorted(LEVELS, key=lambda lv: -(counts[lv] - targets.get(lv, 0)))
        cut = next((i for lv in over for i in reversed(idx) if levels[i] == lv and i not in keep), None)
        if cut is None:
            break
        idx.remove(cut)
        trimmed.append(cut)
    seen: dict[str, int] = {lv: 0 for lv in LEVELS}
    flagged = []
    for i in idx:
        seen[levels[i]] += 1
        if seen[levels[i]] > targets.get(levels[i], 0):
            flagged.append(i)
    return {"kept": idx, "trimmed": sorted(trimmed), "flagged": flagged, "planned": seen,
            "shortfall": {lv: max(0, targets.get(lv, 0) - seen[lv]) for lv in LEVELS}}


def record_window(conn, creator_id, mix: dict, planned: dict) -> None:
    """Her evidence window (creator_states.evidence_window) carries the exploration mix: the shares she plans under,
    the last batch's targets and what was planned."""
    from psycopg.types.json import Jsonb
    window = {"exploration": {"stage": mix.get("stage"), "shares": mix.get("shares"), "targets": mix.get("targets"),
                              "planned": planned, "window": mix.get("window"), "plateau": mix.get("plateau"),
                              "at": datetime.now(timezone.utc).isoformat()}}
    conn.execute("INSERT INTO creator_states (creator_id, evidence_window) VALUES (%s, %s) ON CONFLICT (creator_id) "
                 "DO UPDATE SET evidence_window = creator_states.evidence_window || EXCLUDED.evidence_window, "
                 "updated_at = now()", (creator_id, Jsonb(window)))
