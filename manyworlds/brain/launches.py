"""Launches: an approved launch walks the creator lab's steps by itself (Brain M8).

Deterministic, run by the producer loop like production.advance. Each launch is a row in `launches`:

  casting     (while brain.casting.enabled; manyworlds.brain.casting) 3-5 candidate concepts, a judge, auditions of the
              best as a mini-feed through normal production, the winner picked from rubric x acceptance x cost; the
              winner IS the creator (its bible is written), so the launch goes on at `references`
  bible       brain.creator_lab writes the bible and creates the creator (draft, sandbox); a launch without casting
  references  the reference pack (image jobs) and a voice sample are rendered
  anchors     brain.creator_lab picks the reference images (Claude looks at the renders)
  lora        the identity LoRA is trained
  voice       her voice is cast automatically (creator/voice_lab.cast, voice.cast: candidates from the bible,
              each judged by the worker's own ears; the best qualifying one is stored); skipped while policy
              voice.casting.at_launch is false
  profile     brain.creator_lab builds profile v1
  ready       the creator is active (sandbox until its account exists); an account request goes to the owner

Claude steps run as `brain.creator_lab` jobs within that agent's budget; GPU steps are ordinary jobs under the
autoscaler's caps. A step whose jobs fail is retried up to `brain.launches.max_step_retries` times, then the
launch stops and a `launch_failed` event goes to the owner. Launches start from approved `launch` approvals
(the strategist proposes; while policy `brain.strategist.launch_requires_approval` holds, the owner approves).
"""

from __future__ import annotations

import logging
import re

from psycopg.types.json import Jsonb

from .. import notify, queue
from ..creator import lab
from ..policy import Policy

log = logging.getLogger("manyworlds.launches")

LLM_STEPS = ("bible", "anchors", "profile")
NEXT = {"bible": "references", "references": "anchors", "anchors": "lora", "lora": "voice", "voice": "profile",
        "profile": "dossier", "dossier": "style", "style": "ready"}
FRESH_ON_RETRY = ("voice",)   # a cast where no candidate qualified is cast again with new seeds, not re-queued
AGENT_STEPS = {"dossier": "talent_manager", "style": "style_director"}   # steps another Brain agent runs (a new job on retry, like LLM_STEPS)


def _cfg(policy: Policy) -> dict:
    return policy.get("brain", "launches", default={}) or {}


UUID_RE = "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


def start_approved(conn, *, casting_on: bool = False) -> int:
    """A launch row for every approved launch request that doesn't have one yet, only for a thesis that is
    chosen and has no creator (an approval can never re-launch or touch an existing creator). A rejected
    request returns its thesis to the pool."""
    conn.execute(
        """UPDATE theses t SET status = 'proposed' FROM approvals a
           WHERE a.kind = 'launch' AND a.status = 'rejected' AND a.subject->>'thesis_id' ~ %s
             AND t.id = (a.subject->>'thesis_id')::uuid AND t.status = 'chosen' AND t.creator_id IS NULL""", (UUID_RE,))
    rows = conn.execute(
        """INSERT INTO launches (approval_id, thesis_id, cohort, step)
           SELECT a.id, t.id, a.subject->>'cohort', %s FROM approvals a
           JOIN theses t ON a.subject->>'thesis_id' ~ %s AND t.id = (a.subject->>'thesis_id')::uuid
           WHERE a.kind = 'launch' AND a.status = 'approved' AND t.creator_id IS NULL
             AND t.status IN ('chosen', 'proposed')
             AND NOT EXISTS (SELECT 1 FROM launches l WHERE l.approval_id = a.id)
           ON CONFLICT (approval_id) DO NOTHING RETURNING id""",
        ("casting" if casting_on else "bible", UUID_RE)).fetchall()
    return len(rows)


def _uuid(value) -> str | None:
    return str(value).lower() if value and re.match(UUID_RE, str(value), re.I) else None


def place(conn, thesis_id, creator_id) -> dict:
    """The creator's market (and cluster) from its thesis, so market leads see it (brain.agents.market_lead counts
    a market's creators). A thesis names its market by id (`body.market_id`: a cohort's theses, brain.proposals) or
    by name (`body.market`: discovery); a named market is found or created by its unique name, as the trend analyst
    does (research.trends). A cluster is taken only by id (`body.cluster_id`) and only when it belongs to that market;
    clusters have no unique name, so none is made here. A thesis that names neither leaves the creator unplaced (logged).
    A creator already placed keeps its market and cluster."""
    row = conn.execute("SELECT body FROM theses WHERE id = %s", (thesis_id,)).fetchone()
    body = (row or {}).get("body") or {}
    market_id = None
    if (mid := _uuid(body.get("market_id"))) and conn.execute("SELECT 1 FROM markets WHERE id = %s", (mid,)).fetchone():
        market_id = mid
    cluster_id = None
    if (cl := _uuid(body.get("cluster_id"))):
        c = conn.execute("SELECT market_id FROM clusters WHERE id = %s", (cl,)).fetchone()
        if c and (market_id is None or str(c["market_id"]) == market_id):
            cluster_id, market_id = cl, str(c["market_id"])
    name = str(body.get("market") or "").strip()
    if market_id is None and name:
        market_id = str(conn.execute(
            "INSERT INTO markets (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
            (name,)).fetchone()["id"])
    if market_id is None:
        log.info("launch: thesis %s names no market; creator %s stays unplaced", thesis_id, creator_id)
        return {"market_id": None, "cluster_id": None}
    conn.execute("UPDATE creators SET market_id = coalesce(market_id, %s), cluster_id = coalesce(cluster_id, %s) "
                 "WHERE id = %s", (market_id, cluster_id, creator_id))
    return {"market_id": market_id, "cluster_id": cluster_id}


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
    if step in AGENT_STEPS:
        job = queue.enqueue(conn, f"brain.{AGENT_STEPS[step]}", {"creator_id": str(launch["creator_id"]), "mode": "onboard"},
                            creator_id=launch["creator_id"], idempotency_key=f"launch:{launch['id']}:{step}:{attempt}")
        return [str(job)]
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
    if step == "voice":
        from ..creator import voice_lab
        return [str(voice_lab.cast(conn, policy, cid, attempt=attempt))]
    raise ValueError(f"unknown launch step {step!r}")


def _casts_at_launch(policy: Policy) -> bool:
    return bool(policy.get("voice", "casting", "at_launch", default=True))


def _voice_cast(conn, policy: Policy, launch: dict, job_ids: list[str]) -> bool:
    """Did the finished cast give her a voice? (judged here unless the producer's settle_casts already did)."""
    from ..creator import voice_lab
    cid = launch["creator_id"]
    voice = (conn.execute("SELECT anchors FROM creators WHERE id = %s", (cid,)).fetchone()["anchors"] or {}).get(
        "voice") or {}
    if voice.get("cast_job_id") in job_ids:
        return True
    return voice_lab.pick_cast(conn, policy, cid, job_ids[-1]).get("picked") is not None


def _fail(conn, launch: dict, step: str, error: str) -> None:
    conn.execute("UPDATE launches SET status = 'failed', error = %s, updated_at = now() WHERE id = %s",
                 (error, launch["id"]))
    notify.owner_event(conn, "launch_failed", subject={"launch_id": str(launch["id"])},
                       payload={"step": step, "error": error[:500]}, creator_id=launch["creator_id"],
                       dedupe_key=f"launch_failed:{launch['id']}")


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
    if step in LLM_STEPS or step in AGENT_STEPS or step in FRESH_ON_RETRY:
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
    notify.owner_event(conn, "account_needed", subject={"creator_id": str(cid)},
                       payload={"handle": handle, "platform": "instagram"}, creator_id=cid,
                       dedupe_key=f"account_needed:{cid}:instagram")


def advance(conn, policy: Policy) -> dict:
    from datetime import datetime, timedelta, timezone

    cfg = _cfg(policy)
    now = datetime.now(timezone.utc)
    from . import casting
    out = {"started": start_approved(conn, casting_on=casting.enabled(policy)), "advanced": 0, "ready": 0, "failed": 0,
           "waiting_budget": 0}
    from ..creator import voice_lab
    out["casts"] = voice_lab.settle_casts(conn, policy)   # every finished cast is judged and stored by itself
    for launch in conn.execute("SELECT * FROM launches WHERE status = 'running' ORDER BY created_at").fetchall():
        step, jobs = launch["step"], dict(launch["jobs"] or {})
        if step == "casting":   # auditions have their own deadline (brain.casting.audition.deadline_hours)
            state, value = casting.launch_tick(conn, policy, launch)
            if state == "failed":
                _fail(conn, launch, step, f"casting: {value}")
                out["failed"] += 1
            elif state == "done":
                launch = {**launch, "creator_id": value}
                conn.execute("UPDATE launches SET creator_id = %s WHERE id = %s", (value, launch["id"]))
                place(conn, launch["thesis_id"], value)
                started = _begin(conn, policy, {**launch, "step": NEXT["bible"]}, NEXT["bible"], 0)
                if started is None:
                    out["failed"] += 1
                    continue
                jobs[NEXT["bible"]] = started
                conn.execute("UPDATE launches SET step = %s, jobs = %s, retries = 0, updated_at = now() WHERE id = %s",
                             (NEXT["bible"], Jsonb(jobs), launch["id"]))
                out["advanced"] += 1
            continue
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
            place(conn, launch["thesis_id"], cid["creator_id"])
        if step == "voice" and not _voice_cast(conn, policy, launch, jobs[step]):
            if _retry(conn, policy, launch, step, jobs) == "failed":   # no candidate qualified: cast again
                out["failed"] += 1
            continue
        nxt = NEXT[step]
        if nxt == "voice" and not _casts_at_launch(policy):
            nxt = NEXT[nxt]
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
