"""Launches: an approved launch walks the creator lab's steps by itself (Brain M8).

Deterministic, run by the producer loop like production.advance. Each launch is a row in `launches`:

  bible       brain.creator_lab writes the bible and creates the creator (draft, sandbox)
  references  the reference pack (image jobs) and a voice sample are rendered
  anchors     brain.creator_lab picks the reference images (Claude looks at the renders)
  lora        the identity LoRA is trained
  profile     brain.creator_lab builds profile v1
  ready       the creator is active (sandbox until its account exists); an account request goes to the owner

Claude steps run as `brain.creator_lab` jobs within that agent's budget; GPU steps are ordinary jobs under the
autoscaler's caps. A step whose jobs fail is retried up to `brain.launches.max_step_retries` times, then the
launch stops and a `launch_failed` event goes to the owner. Launches start from approved `launch` approvals
(the strategist proposes; while policy `brain.strategist.launch_requires_approval` holds, the owner approves).
"""

from __future__ import annotations

from psycopg.types.json import Jsonb

from .. import queue
from ..creator import lab
from ..policy import Policy

LLM_STEPS = ("bible", "anchors", "profile")
NEXT = {"bible": "references", "references": "anchors", "anchors": "lora", "lora": "profile", "profile": "ready"}


def _cfg(policy: Policy) -> dict:
    return policy.get("brain", "launches", default={}) or {}


UUID_RE = "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


def start_approved(conn) -> int:
    """A launch row for every approved launch request that doesn't have one yet, only for a thesis that is
    chosen and has no creator (an approval can never re-launch or touch an existing creator). A rejected
    request returns its thesis to the pool."""
    conn.execute(
        """UPDATE theses t SET status = 'proposed' FROM approvals a
           WHERE a.kind = 'launch' AND a.status = 'rejected' AND a.subject->>'thesis_id' ~ %s
             AND t.id = (a.subject->>'thesis_id')::uuid AND t.status = 'chosen' AND t.creator_id IS NULL""", (UUID_RE,))
    rows = conn.execute(
        """INSERT INTO launches (approval_id, thesis_id, cohort)
           SELECT a.id, t.id, a.subject->>'cohort' FROM approvals a
           JOIN theses t ON a.subject->>'thesis_id' ~ %s AND t.id = (a.subject->>'thesis_id')::uuid
           WHERE a.kind = 'launch' AND a.status = 'approved' AND t.creator_id IS NULL
             AND t.status IN ('chosen', 'proposed')
             AND NOT EXISTS (SELECT 1 FROM launches l WHERE l.approval_id = a.id)
           ON CONFLICT (approval_id) DO NOTHING RETURNING id""", (UUID_RE,)).fetchall()
    return len(rows)


def _status(conn, job_ids: list[str], *, min_done: int | None = None) -> str:
    """done, failed (for good), over_budget (a Claude step refused by its daily budget), or waiting. With
    `min_done`, a step is done once nothing is still running and at least that many jobs succeeded."""
    if not job_ids:
        return "waiting"
    rows = conn.execute("SELECT status, error FROM jobs WHERE id = ANY(%s::uuid[])", (job_ids,)).fetchall()
    st = [r["status"] for r in rows]
    if any(s in ("queued", "running") for s in st):
        return "waiting"
    done = sum(s == "done" for s in st)
    if min_done is not None:
        return "done" if done >= min_done else "failed"
    if any(s in ("failed", "dead") for s in st):
        if any("budget" in (r["error"] or "") for r in rows if r["status"] in ("failed", "dead")):
            return "over_budget"
        return "failed"
    return "done" if st and done == len(st) else "waiting"


def _start_step(conn, policy: Policy, launch: dict, step: str, attempt: int) -> list[str]:
    if step in LLM_STEPS:
        job = queue.enqueue(conn, "brain.creator_lab", {"launch_id": str(launch["id"]), "step": step},
                            creator_id=launch["creator_id"], idempotency_key=f"launch:{launch['id']}:{step}:{attempt}")
        return [str(job)]
    cid = launch["creator_id"]
    if step == "references":
        batch = lab.enqueue_reference_pack(conn, policy, cid)
        ids = [str(r["id"]) for r in conn.execute("SELECT id FROM jobs WHERE batch = %s", (batch,)).fetchall()]
        return ids + [str(lab.enqueue_voice_sample(conn, cid))]
    if step == "lora":
        return [str(lab.enqueue_lora_training(conn, policy, cid))]
    raise ValueError(f"unknown launch step {step!r}")


def _fail(conn, launch: dict, step: str, error: str) -> None:
    conn.execute("UPDATE launches SET status = 'failed', error = %s, updated_at = now() WHERE id = %s",
                 (error, launch["id"]))
    conn.execute("INSERT INTO events (type, subject, payload, routed_to, creator_id, dedupe_key) "
                 "VALUES ('launch_failed', %s, %s, 'owner', %s, %s) ON CONFLICT DO NOTHING",
                 (Jsonb({"launch_id": str(launch["id"])}), Jsonb({"step": step, "error": error[:500]}),
                  launch["creator_id"], f"launch_failed:{launch['id']}"))


def _begin(conn, policy: Policy, launch: dict, step: str, attempt: int) -> list[str] | None:
    """Start a step; a step that cannot start (e.g. too few usable reference images to train on) stops the
    launch with the reason instead of failing the producer loop."""
    try:
        return _start_step(conn, policy, launch, step, attempt)
    except (ValueError, LookupError, KeyError) as e:
        _fail(conn, launch, step, f"step {step} could not start: {e}")
        return None


def _retry(conn, policy: Policy, launch: dict, step: str, jobs: dict) -> str:
    limit = int(_cfg(policy).get("max_step_retries", 2))
    if launch["retries"] >= limit:
        _fail(conn, launch, step, f"step {step} failed {launch['retries'] + 1} times")
        return "failed"
    # Failed jobs of a GPU step are put back (the lab's own idempotency keys are reused); a Claude step gets a new job.
    if step in LLM_STEPS:
        jobs[step] = _start_step(conn, policy, launch, step, launch["retries"] + 1)
    else:
        conn.execute("UPDATE jobs SET status = 'queued', attempts = 0, worker_faults = 0, error = NULL, worker_id = NULL, "
                     "lease_until = NULL, updated_at = now() WHERE id = ANY(%s::uuid[]) AND status IN ('failed', 'dead')",
                     (jobs[step],))
    conn.execute("UPDATE launches SET retries = retries + 1, jobs = %s, updated_at = now() WHERE id = %s",
                 (Jsonb(jobs), launch["id"]))
    return "retrying"


def _ready(conn, launch: dict) -> None:
    cid = launch["creator_id"]
    # Never downgrade a creator that already publishes for real.
    conn.execute("UPDATE creators SET status = 'active' WHERE id = %s", (cid,))
    conn.execute("INSERT INTO accounts (creator_id, platform, status) VALUES (%s, 'instagram', 'requested') "
                 "ON CONFLICT (creator_id, platform) DO NOTHING", (cid,))
    handle = conn.execute("SELECT handle FROM creators WHERE id = %s", (cid,)).fetchone()["handle"]
    conn.execute("INSERT INTO events (type, subject, payload, routed_to, creator_id, dedupe_key) "
                 "VALUES ('account_needed', %s, %s, 'owner', %s, %s) ON CONFLICT DO NOTHING",
                 (Jsonb({"creator_id": str(cid)}), Jsonb({"handle": handle, "platform": "instagram"}), cid,
                  f"account_needed:{cid}:instagram"))


def advance(conn, policy: Policy) -> dict:
    from datetime import datetime, timedelta, timezone

    cfg = _cfg(policy)
    now = datetime.now(timezone.utc)
    out = {"started": start_approved(conn), "advanced": 0, "ready": 0, "failed": 0, "waiting_budget": 0}
    for launch in conn.execute("SELECT * FROM launches WHERE status = 'running' ORDER BY created_at").fetchall():
        step, jobs = launch["step"], dict(launch["jobs"] or {})
        if jobs.get("wait_until"):
            if now < datetime.fromisoformat(jobs["wait_until"]):
                out["waiting_budget"] += 1
                continue
            jobs.pop("wait_until")
            jobs[step] = _start_step(conn, policy, launch, step, int(jobs.get("budget_waits", 0)))
            conn.execute("UPDATE launches SET jobs = %s, updated_at = now() WHERE id = %s", (Jsonb(jobs), launch["id"]))
            continue
        if now - launch["updated_at"] > timedelta(hours=float(cfg.get("step_timeout_hours", 48))):
            _fail(conn, launch, step, f"step {step} made no progress for {cfg.get('step_timeout_hours')} hours")
            out["failed"] += 1
            continue
        if step not in jobs:
            started = _begin(conn, policy, launch, step, 0)
            if started is None:
                out["failed"] += 1
                continue
            jobs[step] = started
            conn.execute("UPDATE launches SET jobs = %s, updated_at = now() WHERE id = %s", (Jsonb(jobs), launch["id"]))
            out["advanced"] += 1
            continue
        state = _status(conn, jobs[step], min_done=int(cfg.get("min_reference_renders", 8)) if step == "references" else None)
        if state == "waiting":
            continue
        if state == "over_budget":
            # Its agent is over today's LLM budget: wait for tomorrow's, without using a retry (spec 4.3).
            jobs["wait_until"] = (now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).isoformat()
            jobs["budget_waits"] = int(jobs.get("budget_waits", 0)) + 1
            conn.execute("UPDATE launches SET jobs = %s WHERE id = %s", (Jsonb(jobs), launch["id"]))
            out["waiting_budget"] += 1
            continue
        if state == "failed":
            if _retry(conn, policy, launch, step, jobs) == "failed":
                out["failed"] += 1
            continue
        if step == "bible":   # the creator lab created the creator: remember who
            cid = conn.execute("SELECT creator_id FROM theses WHERE id = %s", (launch["thesis_id"],)).fetchone()
            if not cid or not cid["creator_id"]:
                _retry(conn, policy, launch, step, jobs)
                continue
            launch = {**launch, "creator_id": cid["creator_id"]}
            conn.execute("UPDATE launches SET creator_id = %s WHERE id = %s", (cid["creator_id"], launch["id"]))
        nxt = NEXT[step]
        if nxt == "ready":
            _ready(conn, launch)
            conn.execute("UPDATE launches SET step = 'ready', status = 'done', retries = 0, updated_at = now() "
                         "WHERE id = %s", (launch["id"],))
            out["ready"] += 1
            continue
        started = _begin(conn, policy, {**launch, "step": nxt}, nxt, 0)
        if started is None:
            out["failed"] += 1
            continue
        jobs[nxt] = started
        conn.execute("UPDATE launches SET step = %s, jobs = %s, retries = 0, updated_at = now() WHERE id = %s",
                     (nxt, Jsonb(jobs), launch["id"]))
        out["advanced"] += 1
    return out
