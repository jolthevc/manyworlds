"""Where a batch's exploration goes: Thompson sampling over what is known and what is untried.

Each candidate direction gets a belief about its effect on reach (log-lift): effects the analyst measured
use their estimate and standard error; untried directions (formats and stagings the creator hasn't used)
use a prior centred on zero with `prior_sd`. One draw from each belief; the highest draws are this batch's
exploration directions. Uncertain-but-promising directions keep getting tried, clear losers rarely do, and
nothing is ruled out on thin evidence. The draws are recorded on the plan decision, so the trace shows why
a batch explored what it did.
"""

from __future__ import annotations

import hashlib
import random

from ..policy import Policy

KINDS = ("attribute", "format", "dimension", "staging")


def _cfg(policy: Policy) -> dict:
    return policy.get("brain", "exploration", default={}) or {}


def describe(direction: str) -> str:
    kind, _, key = direction.partition(":")
    if kind == "attribute" and "=" in key:
        k, v = key.split("=", 1)
        return f"{k}: {v}"
    return {"format": f"the {key} format", "dimension": f"ideas strong on {key}",
            "staging": f"the {key} staging"}.get(kind, direction)


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


def untried(conn, creator_id, *, formats, stagings) -> list[str]:
    """Formats and stagings production can make that this creator has never had an idea for."""
    used_formats = {r["format"] for r in conn.execute(
        "SELECT DISTINCT format FROM ideas WHERE creator_id = %s", (creator_id,)).fetchall()}
    used_stagings = {r["s"] for r in conn.execute(
        "SELECT DISTINCT attributes->>'staging' AS s FROM ideas WHERE creator_id = %s", (creator_id,)).fetchall()}
    return [f"format:{f}" for f in formats if f not in used_formats] + \
           [f"staging:{s}" for s in stagings if s not in used_stagings]
