"""Which way of making an output to use: the one that measurably costs least per accepted output.

A story clip can come from a hosted API (hosted.video), the ComfyUI worker's Wan 2.2 graph (comfy.wan_i2v) or the
older self-hosted pipeline (video.wan_i2v). Pool scoring (deploy/scoring.py) already picks the cheapest GPU pool for
one job kind; this is the same idea one level up, across job kinds that make the same thing.

A kind's cost over the window is everything its jobs cost: hosted API charges and review LLM calls charged to them,
plus the billed time of its family's GPU machines (boot, download and idle included), split between the family's
kinds by GPU seconds worked. Divided by its accepted outputs (every review row passed, confirmed or cleared).

Routing (policy production.routing.<output>): kinds with at least `min_accepted` are ranked by cost per accepted
output; the cheapest gets the work except an `explore` share, which goes to the others in policy order, so a new or
improved option keeps getting measured. Before any kind is ranked, the policy order decides. The choice per post is
deterministic (a hash of the post), so a retry lands on the same kind.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from .policy import Policy


def _family(kind: str) -> str:
    return kind.split(".")[0]


def kind_costs(conn, kinds: list[str], *, since: datetime) -> dict[str, dict]:
    out = {}
    for kind in kinds:
        direct = float(conn.execute(
            "SELECT coalesce(sum(e.usd), 0) AS u FROM cost_entries e JOIN jobs j ON j.id = e.job_id "
            "WHERE j.kind = %s AND j.created_at >= %s AND e.category IN ('hosted_api', 'llm')",
            (kind, since)).fetchone()["u"])
        # The family's machines, billed in full, shared between its kinds by the GPU time each used.
        fam = conn.execute(
            """WITH g AS (SELECT name, cost_usd FROM gpu_groups WHERE family = %(f)s AND started_at >= %(s)s),
                    w AS (SELECT a.gpu_group, j.kind, sum(coalesce(a.gpu_seconds, 0)) AS sec
                          FROM attempts a JOIN jobs j ON j.id = a.job_id
                          WHERE a.gpu_group IN (SELECT name FROM g) GROUP BY 1, 2)
               SELECT coalesce(sum(g.cost_usd * coalesce(
                        (SELECT sec FROM w WHERE w.gpu_group = g.name AND w.kind = %(k)s), 0)
                        / nullif((SELECT sum(sec) FROM w WHERE w.gpu_group = g.name), 0)), 0) AS u
               FROM g""", {"f": _family(kind), "s": since, "k": kind}).fetchone()["u"]
        accepted = int(conn.execute(
            """SELECT count(*) AS n FROM (
                 SELECT a.id FROM assets a JOIN jobs j ON j.id = a.job_id JOIN reviews r ON r.asset_id = a.id
                 WHERE j.kind = %s AND j.created_at >= %s AND a.kind IN ('video', 'image')
                 GROUP BY a.id
                 HAVING bool_and(r.passed)
                    AND bool_or(r.details->>'stage' = 'confirm' OR r.details->>'clear_pass' = 'true')) x""",
            (kind, since)).fetchone()["n"])
        usd = direct + float(fam or 0)
        out[kind] = {"usd": round(usd, 4), "accepted": accepted,
                     "usd_per_accepted": round(usd / accepted, 4) if accepted else None}
    return out


def choose(conn, policy: Policy, output: str, *, key: str, now: datetime | None = None) -> tuple[str, dict]:
    """The job kind for one `output` (e.g. story_clip) for the item `key` (a post), and why."""
    cfg = policy.get("production", "routing", output, default={}) or {}
    kinds = list(cfg.get("kinds") or [])
    if not kinds:
        raise KeyError(f"no production.routing.{output}.kinds")
    now = now or datetime.now(timezone.utc)
    costs = kind_costs(conn, kinds, since=now - timedelta(days=float(cfg.get("days", 14))))
    ranked = sorted((k for k in kinds if costs[k]["accepted"] >= int(cfg.get("min_accepted", 5))),
                    key=lambda k: costs[k]["usd_per_accepted"])
    draw = int(hashlib.sha256(f"{output}:{key}".encode()).hexdigest()[:8], 16) / 2**32
    if not ranked:
        return kinds[0], {"why": "no kind measured yet: policy order", "costs": costs}
    best = ranked[0]
    others = [k for k in kinds if k != best]
    if others and draw < float(cfg.get("explore", 0.1)):
        pick = others[int(draw / float(cfg.get("explore", 0.1)) * len(others)) % len(others)]
        return pick, {"why": f"exploring (best is {best})", "costs": costs}
    return best, {"why": "cheapest per accepted output", "costs": costs}
