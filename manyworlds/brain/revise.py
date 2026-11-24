"""Critic-revise for ideas just under the feed-QA floor (policy `brain.revise`).

An idea whose idea-rubric score sits under a `brain.qa.min_scores` floor is dropped at scheduling
(manyworlds.brain.feed_qa). When it misses only by a little (every dimension under its floor is within
`margin` of it), the idea is often fixable with words: one cheap rewrite (brain/agents/idea_reviser.py,
role `idea_revise`) gets the plan, the evaluator's reasons for the weak dimensions and the creator's voice
rules, and returns the same idea improved. The rewrite is a new idea in the same batch, evaluated like any
other, and scheduled or dropped on its own score; the original is kept with status `revised`. This is the
brief being fixed before the GPU: nothing has been rendered yet.

Rules, all in policy: off unless `enabled`; at most `max_per_idea` revisions in an idea's line (an idea that is
itself a revision counts its parent's), `max_per_batch` per batch; experiment ideas are never revised (their arms
must stay comparable, and feed QA exempts them anyway); never after the batch's `brain.showrunner.max_wait_hours`
(the batch then falls back as before: a near-floor idea is dropped, a revision not yet evaluated is dropped).
"""

from __future__ import annotations

from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from .. import queue
from ..policy import Policy

KIND = "brain.idea_reviser"
# What a revision carries on its body to say where it came from; the evaluator never sees it (an idea is
# scored on its plan, not on its history).
BOOKKEEPING = ("revision", "revision_of", "original", "revise_reason", "revise_requested_at", "revise_job",
               "revise_dims", "revised_by")


def cfg(policy: Policy) -> dict:
    return policy.get("brain", "revise", default={}) or {}


def near_floor(policy: Policy, scores: dict | None) -> list[str] | None:
    """The dimensions under their feed-QA floor when every one of them is within `margin` of it (the idea is worth
    one rewrite); None when nothing is under a floor or anything is further under (far below: dropped as before)."""
    floors = policy.get("brain", "qa", "min_scores", default={}) or {}
    margin = float(cfg(policy).get("margin", 0))
    under = []
    for dim, floor in floors.items():
        s = (scores or {}).get(dim)
        if s is None or float(s) >= float(floor):
            continue
        if float(floor) - float(s) > margin:
            return None
        under.append(dim)
    return under or None


def eligible(policy: Policy, row: dict) -> list[str] | None:
    """The weak dimensions to revise for, or None: revision off, an experiment idea, already asked for once, at the
    line's revision limit, or not near a floor."""
    c = cfg(policy)
    body = row.get("body") or {}
    if not c.get("enabled") or row.get("experiment_id") or body.get("revise_requested_at"):
        return None
    if int(body.get("revision") or 0) >= int(c.get("max_per_idea", 1)):
        return None
    return near_floor(policy, row.get("scores"))


def requested_in_batch(conn, batch: str) -> int:
    """Revisions this batch has asked for so far (whatever came of them)."""
    return int(conn.execute("SELECT count(*) AS n FROM ideas WHERE batch = %s AND body ? 'revise_requested_at'",
                            (batch,)).fetchone()["n"])


def request(conn, policy: Policy, row: dict, dims: list[str], *, batch: str, creator_id, why: str) -> str:
    """Queue the rewrite of one idea and mark it, so the batch waits for it and it is never asked for twice."""
    job = queue.enqueue(conn, KIND, {"idea_id": str(row["id"]), "dims": dims, "reason": why,
                                     "requested_by": "schedule_ready"},
                        creator_id=creator_id, idempotency_key=f"revise:{row['id']}")
    conn.execute("UPDATE ideas SET updated_at = now(), body = body || %s WHERE id = %s AND status = 'proposed'",
                 (Jsonb({"revise_requested_at": datetime.now(timezone.utc).isoformat(), "revise_job": str(job),
                         "revise_dims": dims, "revise_reason": why}), row["id"]))
    return str(job)


def plan_for(conn, policy: Policy, batch: str, qa_dropped: list[tuple[dict, str]]) -> list[tuple]:
    """Which of a batch's QA drops get a rewrite now: (row, dims, why) in batch order, within `max_per_batch`."""
    if not cfg(policy).get("enabled"):
        return []
    room = int(cfg(policy).get("max_per_batch", 0)) - requested_in_batch(conn, batch)
    out = []
    for r, why in qa_dropped:
        if room <= 0:
            break
        dims = eligible(policy, r)
        if dims:
            out.append((r, dims, why))
            room -= 1
    return out
