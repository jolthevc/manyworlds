"""Sandbox runs of the closed learning loop (`mw brain simulate`, `mw brain learning`).

A sandbox creator (publish_mode `simulated`) is run batch after batch in fast-forward: the analyst
reads outcomes so far, the showrunner plans the next period with that analysis, the evaluator scores
every idea, evaluated ideas are scheduled, the period is simulated-published without making images,
and the scenario engine produces outcomes. Everything is labelled SIMULATED: it proves the loop
works, it is not evidence of demand. Real creators are refused.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone

from psycopg.types.json import Jsonb

from .. import queue
from ..policy import Policy
from ..publish import scheduler
from . import evidence, ideas, planner, runtime


def clone(conn, source_handle: str, handle: str, *, scenario: str, seed: int = 0):
    """A sandbox copy of a creator (bible, latest profile, memory comes from the profile). No reference
    images, so production never renders it; the scenario plays its audience."""
    src = conn.execute("SELECT id, bible, profile_version FROM creators WHERE handle = %s", (source_handle,)).fetchone()
    cid = conn.execute(
        "INSERT INTO creators (handle, status, bible, anchors, publish_mode, profile_version) "
        "VALUES (%s, 'draft', %s, %s, 'simulated', %s) RETURNING id",
        (handle, Jsonb(src["bible"]), Jsonb({"simulation": {"scenario": scenario, "seed": seed}}),
         src["profile_version"])).fetchone()["id"]
    if src["profile_version"]:
        conn.execute("INSERT INTO creator_profiles (creator_id, version, body, author, reason) "
                     "SELECT %s, version, body, 'brain', 'sandbox clone' FROM creator_profiles "
                     "WHERE creator_id = %s AND version = %s", (cid, src["id"], src["profile_version"]))
    return cid


def _drain(conn, policy: Policy, llm, creator_id, worker_id: str, max_rounds: int = 200) -> None:
    """Run this creator's brain jobs (and only these) until none is waiting."""
    lease = int(policy.get("brain", "runtime", "lease_seconds", default=600))
    for _ in range(max_rounds):
        ids = [r["id"] for r in conn.execute(
            "SELECT id FROM jobs WHERE kind LIKE 'brain.%%' AND creator_id = %s AND attempts < max_attempts "
            # a job whose run was killed (lease expired) is taken again, or one dead run blocks every later one
            "AND (status = 'queued' OR (status = 'running' AND lease_until < now())) "
            "ORDER BY priority DESC, created_at", (creator_id,)).fetchall()]
        if not ids:
            return
        for jid in ids:
            job = queue.claim_job(conn, jid, worker_id, lease)
            if job:
                runtime.run_job(conn, policy, llm, job, worker_id=worker_id, now=datetime.now(timezone.utc), lease_s=lease)


def run_batches(conn, policy: Policy, llm, creator_id, *, batches: int, worker_id: str) -> list[dict]:
    c = conn.execute("SELECT anchors FROM creators WHERE id = %s", (creator_id,)).fetchone()
    anchors = (c or {}).get("anchors") or {}
    if evidence.mode_for(conn, creator_id) != "sandbox":
        raise ValueError("sandbox runs are for simulated creators only")
    if not (anchors.get("simulation") or {}).get("scenario"):
        raise ValueError("sandbox runs need a test scenario (anchors.simulation.scenario); a sandbox creator without one "
                         "is waiting for a real account")
    if anchors.get("reference_images"):
        raise ValueError("sandbox creators must have no reference images, so production never renders them")
    ff = dataclasses.replace(policy, body={**policy.body, "simulation": {**(policy.get("simulation") or {}),
                                                                         "fast_forward": True}})
    days = int(policy.get("brain", "showrunner", "days", default=3))
    runs = []
    for _ in range(batches):
        start = planner.next_start(conn, creator_id, date.today())
        job = queue.enqueue(conn, "brain.analyst", {"creator_id": str(creator_id), "then_plan": str(start)},
                            creator_id=creator_id, idempotency_key=f"analyst:{creator_id}:{start}")   # same key as planning
        _drain(conn, policy, llm, creator_id, worker_id)
        # A near-floor idea is rewritten and re-evaluated before its batch is scheduled (manyworlds.brain.revise):
        # run those jobs too, then schedule again (max_per_idea bounds the rounds).
        for _ in range(int(policy.get("brain", "revise", "max_per_idea", default=1)) + 1):
            if not ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc)).get("revising"):
                break
            _drain(conn, policy, llm, creator_id, worker_id)
        else:
            ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc))
        end = datetime.combine(start + timedelta(days=days), time(0), tzinfo=timezone.utc)
        published = scheduler.publish_due(conn, ff, now=end, creator_id=creator_id)
        analysis = conn.execute("SELECT id, output FROM decisions WHERE job_id = %s AND kind = 'analysis'",
                                (job,)).fetchone()
        runs.append({"start": str(start), "published": published.get("simulated", 0),
                     "analysis": {"id": str(analysis["id"]) if analysis else None,
                                  "insufficient": (analysis["output"] or {}).get("insufficient") if analysis else None,
                                  "n": (analysis["output"] or {}).get("n") if analysis else None}})
    return runs


def _composition(conn, batch: str) -> dict:
    rows = conn.execute(
        """SELECT i.format, i.attributes, e.scores FROM ideas i
           LEFT JOIN LATERAL (SELECT scores FROM evaluations WHERE idea_id = i.id AND role = 'evaluator'
                              ORDER BY created_at DESC LIMIT 1) e ON true
           WHERE i.batch = %s AND i.status IN ('scheduled', 'scheduling')""", (batch,)).fetchall()
    n = len(rows) or 1
    attrs = Counter(f"{k}={str(v).lower()}" for r in rows for k, v in (r["attributes"] or {}).items())
    dims = {d for r in rows for d in (r["scores"] or {})}
    means = {d: round(sum(r["scores"][d] for r in rows if (r["scores"] or {}).get(d) is not None) /
                      max(1, sum(1 for r in rows if (r["scores"] or {}).get(d) is not None)), 2) for d in sorted(dims)}
    return {"ideas": len(rows), "formats": {f: round(c / n, 2) for f, c in Counter(r["format"] for r in rows).items()},
            "attributes": {k: round(c / n, 2) for k, c in attrs.items()}, "mean_scores": means}


def learning_trace(conn, creator_id) -> dict:
    """What the Brain learned for this creator and how each plan used it."""
    mode = evidence.mode_for(conn, creator_id)
    analyses = conn.execute("SELECT id, output, created_at FROM decisions WHERE kind = 'analysis' AND creator_id = %s "
                            "ORDER BY created_at", (creator_id,)).fetchall()
    plans = conn.execute("SELECT id, inputs, output, rationale, created_at FROM decisions WHERE kind = 'content_batch' "
                         "AND creator_id = %s ORDER BY created_at", (creator_id,)).fetchall()
    return {
        "label": "SIMULATED (sandbox engineering test, not real demand)" if mode == "sandbox" else "REAL",
        "findings": [{k: f[k] for k in ("id", "statement", "grade", "effect", "interval_low", "interval_high", "n")}
                     for f in evidence.findings(conn, mode=mode, creator_id=creator_id)],
        "analyses": [{"id": str(a["id"]), "n": a["output"].get("n"), "insufficient": a["output"].get("insufficient"),
                      "top_effects": (a["output"].get("effects") or [])[:5], "hypotheses": a["output"].get("hypotheses"),
                      "guidance": a["output"].get("guidance")} for a in analyses],
        "batches": [{"decision_id": str(p["id"]), "batch": p["output"].get("batch"),
                     "analysis_decision_id": p["inputs"].get("analysis_decision_id"),
                     "finding_ids": p["inputs"].get("finding_ids") or [], "rationale": p["rationale"],
                     "composition": _composition(conn, p["output"].get("batch"))} for p in plans],
    }
