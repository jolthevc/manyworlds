"""The brain's call journal and Anthropic's Message Batches (half price for calls nobody is waiting on).

Brain agents run as jobs and nothing waits on them by the second, so their Claude calls can go through the Message
Batches API at 50% of the price (most batches end within the hour). An agent is ordinary sequential code, so a
batched call can't return at once; instead:

  1. While batching is on (`LLM.batching`, set by the brain worker only) and a call is made inside a job
     (`LLM.tagged`), every call is journaled in `llm_journal` under the job, its attempt, role, model and schema.
  2. A call of a role in policy `llm.batch.roles` (on a Claude model) is queued in the journal and raises
     `Deferred`. The brain runtime gives the job back to the queue (no attempt charged, `queue.defer`).
  3. `flush` (each brain worker loop) sends queued calls as one batch, polls open batches, writes answers into the
     journal, charges them at the batch price, and wakes the jobs.
  4. The job runs again from the start. Every call it already made is answered from the journal for nothing (agents'
     `remember` steps skip what they already did), so only the next new call is paid for.

Matching a call to its journal row on a rerun: rows of the same job, attempt, role, model, schema and system prompt,
not yet used in this run, the identical prompt first; a prompt that changed a little since (a portfolio snapshot
whose spend moved) matches the earliest row whose words overlap at least `near_match_min` (Jaccard). A call asked
twice with the same prompt (the evaluator's extra samples) takes a row each.

Safety valves: a batched call waiting longer than `max_wait_minutes`, an errored or expired one, or one the batch
API refused is answered by an ordinary call instead (`batch_fallback`). A retried job (a real failure) has a new
attempt number, so it asks afresh instead of replaying the answer it failed on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from psycopg.types.json import Jsonb
from pydantic import BaseModel, ValidationError

from . import ledger
from .policy import Policy

log = logging.getLogger("manyworlds.llm_batch")

LEASE = "llm-batch-flush"


class Deferred(BaseException):
    """A batched call is not answered yet: the job steps aside and runs again when it is. A BaseException, so an
    agent's `except Exception` (one bad item must not stop the others) can't swallow it."""

    def __init__(self, row_id: int):
        super().__init__(f"waiting on batched call {row_id}")
        self.row_id = row_id


def cfg(policy: Policy) -> dict:
    return policy.get("llm", "batch", default={}) or {}


def active(llm) -> bool:
    """Journaling is on for this call: the brain worker's LLM, inside a job, with batching enabled in policy."""
    return bool(getattr(llm, "batching", False) and llm._tag.get("job_id") and llm.conn is not None
                and cfg(llm.policy).get("enabled"))


def batchable(llm, role: str, model: str) -> bool:
    return role in (cfg(llm.policy).get("roles") or []) and not llm.is_open(model)


# --- keys --------------------------------------------------------------------------------------------------------
def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def content_text(content: Any) -> str:
    """The prompt as text: text blocks as they are, an image as a digest token (an image changing is a different
    prompt for the near match, a few bytes of text changing is not)."""
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif b.get("type") == "image":
            parts.append(f"[image:{_sha(b.get('source'))[:16]}]")
    return "\n".join(parts)


_WORD = re.compile(r"\w+")


def similarity(a: str, b: str) -> float:
    """Jaccard overlap of the two prompts' words."""
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)


# --- the journal ---------------------------------------------------------------------------------------------------
def journaled(llm, role: str, model: str, fallback: str | None, *, system, content, schema: type[BaseModel],
              effort: str, creator_id):
    """The answer to one call made inside a brain job (see the module docstring)."""
    c = cfg(llm.policy)
    tag = llm._tag
    job_id, attempt = tag["job_id"], int(tag.get("attempt") or 0)
    seen: set = tag.setdefault("seen", set())
    sys_h, text = _sha(system), content_text(content)
    c_h = _sha([text, content if not isinstance(content, str) else None])
    rows = llm.conn.execute(
        "SELECT id, status, mode, content_hash, content_text, output, result, error, usd, shadowed, created_at "
        "FROM llm_journal WHERE job_id = %s AND attempt = %s AND role = %s AND model = %s AND schema_name = %s "
        "AND system_hash = %s ORDER BY id",
        (job_id, attempt, role, model, schema.__name__, sys_h)).fetchall()
    free = [r for r in rows if r["id"] not in seen]
    row = next((r for r in free if r["content_hash"] == c_h), None)
    if row is None:
        floor = float(c.get("near_match_min", 0.9))
        row = next((r for r in free if similarity(r["content_text"], text) >= floor), None)
    call = dict(system=system, content=content, schema=schema, effort=effort, creator_id=creator_id)
    if row is not None:
        seen.add(row["id"])
        if row["status"] == "done":
            out = _read(llm, model, schema, row)
            if not row["shadowed"]:
                llm.conn.execute("UPDATE llm_journal SET shadowed = true WHERE id = %s", (row["id"],))
                llm._shadow(role, model, out, float(row["usd"]), **call)
            return out
        waited = datetime.now(timezone.utc) - row["created_at"]
        if row["status"] == "failed" or waited > timedelta(minutes=float(c.get("max_wait_minutes", 120))):
            log.warning("batched %s call %s %s; answering it directly", role, row["id"],
                        f"failed ({row['error']})" if row["status"] == "failed" else f"waited {waited}")
            out, usd = _direct(llm, role, model, fallback, **call)
            llm.conn.execute(
                "UPDATE llm_journal SET status = 'done', mode = 'batch_fallback', output = %s, usd = usd + %s, "
                "shadowed = true, request = NULL, done_at = now() WHERE id = %s",
                (Jsonb(out.model_dump(mode="json")), usd, row["id"]))
            return out
        raise Deferred(row["id"])
    if batchable(llm, role, model):
        new = llm.conn.execute(
            "INSERT INTO llm_journal (job_id, attempt, role, model, schema_name, system_hash, content_hash, "
            "content_text, mode, status, request, agent, creator_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'batch', 'queued', %s, %s, %s) RETURNING id",
            (job_id, attempt, role, model, schema.__name__, sys_h, c_h, text,
             Jsonb(llm.batch_params(model, system=system, content=content, schema=schema, effort=effort)),
             tag.get("agent"), creator_id)).fetchone()["id"]
        seen.add(new)
        raise Deferred(new)
    out, usd = _direct(llm, role, model, fallback, **call)
    new = llm.conn.execute(
        "INSERT INTO llm_journal (job_id, attempt, role, model, schema_name, system_hash, content_hash, content_text, "
        "mode, status, output, usd, shadowed, agent, creator_id, done_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'sync', 'done', %s, %s, true, %s, %s, now()) RETURNING id",
        (job_id, attempt, role, model, schema.__name__, sys_h, c_h, text, Jsonb(out.model_dump(mode="json")), usd,
         tag.get("agent"), creator_id)).fetchone()["id"]
    seen.add(new)
    return out


def _direct(llm, role, model, fallback, **call):
    n0 = len(llm.calls)
    out = llm.answer(role, model, fallback, **call)
    return out, round(sum(u.usd for u in llm.calls[n0:]), 6)


def _read(llm, model: str, schema: type[BaseModel], row: dict) -> BaseModel:
    if row["output"] is not None:
        return schema.model_validate(row["output"])
    from .llm import Refused, _json_text
    res = row["result"] or {}
    if res.get("stop_reason") == "refusal":
        raise Refused(f"{model} declined (batched call {row['id']})")
    try:
        return schema.model_validate_json(_json_text(res.get("text") or ""))
    except ValidationError as e:
        raise RuntimeError(f"{model} returned no valid {schema.__name__} in batched call {row['id']} "
                           f"(stop_reason={res.get('stop_reason')}): {str(e)[:300]}") from None


# --- sending and collecting ----------------------------------------------------------------------------------------
def _lease(conn, seconds: float) -> str | None:
    import uuid
    holder = uuid.uuid4().hex
    got = conn.execute(
        """INSERT INTO leases (name, holder, until) VALUES (%(n)s, %(h)s, now() + make_interval(secs => %(s)s))
           ON CONFLICT (name) DO UPDATE SET holder = excluded.holder, until = excluded.until
           WHERE leases.until < now() RETURNING holder""", {"n": LEASE, "h": holder, "s": seconds}).fetchone()
    return holder if got else None


def flush(conn, policy: Policy, client, *, now: datetime | None = None) -> dict:
    """Send queued calls, collect ended batches, wake their jobs, purge old rows. One flusher at a time (a lease)."""
    c = cfg(policy)
    out = {"sent": 0, "collected": 0, "failed": 0}
    if not c.get("enabled"):
        return out
    holder = _lease(conn, float(c.get("flush_lease_seconds", 300)))
    if holder is None:
        return out
    try:
        now = now or datetime.now(timezone.utc)
        _send(conn, policy, client, c, now, out)
        _collect(conn, policy, client, c, now, out)
        conn.execute("DELETE FROM llm_journal WHERE created_at < %s AND status IN ('done', 'failed')",
                     (now - timedelta(days=float(c.get("keep_days", 14))),))
        conn.execute("DELETE FROM llm_batches WHERE status = 'collected' AND collected_at < %s",
                     (now - timedelta(days=float(c.get("keep_days", 14))),))
    finally:
        conn.execute("DELETE FROM leases WHERE name = %s AND holder = %s", (LEASE, holder))
    return out


def _wake(conn, job_ids) -> None:
    ids = sorted({str(j) for j in job_ids})
    if ids:
        conn.execute("UPDATE jobs SET run_after = now() WHERE id = ANY(%s::uuid[]) AND status = 'queued'", (ids,))


def _send(conn, policy: Policy, client, c: dict, now: datetime, out: dict) -> None:
    rows = conn.execute("SELECT id, job_id, request, created_at FROM llm_journal WHERE status = 'queued' "
                        "ORDER BY id LIMIT %s", (int(c.get("max_requests", 1000)),)).fetchall()
    if not rows:
        return
    gather = timedelta(seconds=float(c.get("gather_seconds", 30)))
    if gather and now - min(r["created_at"] for r in rows) < gather and len(rows) < int(c.get("max_requests", 1000)):
        return   # wait a little for the other jobs' calls, so they go in one batch
    try:
        batch = client.messages.batches.create(
            requests=[{"custom_id": f"mw{r['id']}", "params": r["request"]} for r in rows])
    except Exception as e:  # noqa: BLE001 - a refused batch: these calls are answered directly on their next run
        msg = f"{type(e).__name__}: {e}"[:500]
        log.warning("batch of %d calls refused: %s", len(rows), msg)
        conn.execute("UPDATE llm_journal SET status = 'failed', error = %s WHERE id = ANY(%s)",
                     (msg, [r["id"] for r in rows]))
        _wake(conn, [r["job_id"] for r in rows])
        out["failed"] += len(rows)
        return
    conn.execute("INSERT INTO llm_batches (id, requests) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                 (batch.id, len(rows)))
    conn.execute("UPDATE llm_journal SET status = 'submitted', batch_id = %s, submitted_at = now(), request = NULL "
                 "WHERE id = ANY(%s)", (batch.id, [r["id"] for r in rows]))
    out["sent"] += len(rows)


def _collect(conn, policy: Policy, client, c: dict, now: datetime, out: dict) -> None:
    poll = timedelta(seconds=float(c.get("poll_seconds", 60)))
    discount = float(c.get("discount", 0.5))
    for b in conn.execute("SELECT id, checked_at FROM llm_batches WHERE status = 'submitted' ORDER BY submitted_at"
                          ).fetchall():
        if poll and b["checked_at"] is not None and now - b["checked_at"] < poll:
            continue
        info = client.messages.batches.retrieve(b["id"])
        counts = getattr(info, "request_counts", None)
        counts = counts.model_dump() if hasattr(counts, "model_dump") else (counts or {})
        conn.execute("UPDATE llm_batches SET checked_at = %s, counts = %s WHERE id = %s",
                     (now, Jsonb(counts), b["id"]))
        if getattr(info, "processing_status", None) != "ended":
            continue
        woken = []
        for r in client.messages.batches.results(b["id"]):
            rid = int(str(r.custom_id).removeprefix("mw"))
            row = conn.execute("SELECT id, job_id, model, agent, creator_id FROM llm_journal WHERE id = %s",
                               (rid,)).fetchone()
            if row is None:
                continue
            woken.append(row["job_id"])
            res = r.result
            if getattr(res, "type", None) == "succeeded":
                msg = res.message
                usd = _charge(conn, policy, msg, row, discount)
                text = next((blk.text for blk in (msg.content or []) if getattr(blk, "type", None) == "text"), "")
                conn.execute("UPDATE llm_journal SET status = 'done', result = %s, usd = %s, done_at = now() "
                             "WHERE id = %s", (Jsonb({"text": text, "stop_reason": msg.stop_reason, "model": msg.model}),
                                               usd, rid))
                out["collected"] += 1
            else:
                err = getattr(res, "error", None)
                why = getattr(getattr(err, "error", err), "type", None) or getattr(res, "type", "unknown")
                conn.execute("UPDATE llm_journal SET status = 'failed', error = %s WHERE id = %s",
                             (f"batch result {getattr(res, 'type', '?')}: {why}", rid))
                out["failed"] += 1
        conn.execute("UPDATE llm_batches SET status = 'collected', collected_at = now() WHERE id = %s", (b["id"],))
        _wake(conn, woken)


def _charge(conn, policy: Policy, msg, row: dict, discount: float) -> float:
    """A batched answer at the batch price: the role's per-token prices times `discount` (cache reads and writes
    too), charged to the job's agent when the answer lands (whether or not the job ever runs again)."""
    u = msg.usage
    prices = policy.get("llm", "usd_per_mtok", msg.model) or policy.get("llm", "usd_per_mtok", row["model"]) or [0, 0, 0, 0]
    cr = getattr(u, "cache_read_input_tokens", 0) or 0
    cw = getattr(u, "cache_creation_input_tokens", 0) or 0
    usd = discount * (u.input_tokens * prices[0] + u.output_tokens * prices[1] + cr * prices[2] + cw * prices[3]) / 1e6
    ledger.record(conn, category="llm", provider=msg.model, quantity=u.input_tokens + u.output_tokens + cr + cw,
                  unit="token", usd=usd, creator_id=row["creator_id"], policy_version=policy.version,
                  agent=row["agent"], job_id=row["job_id"])
    return round(usd, 6)


def status(conn, policy: Policy) -> dict:
    """For `mw llm batches`: the journal by mode and status, open batches, and what batching saved."""
    by = [dict(r) for r in conn.execute(
        "SELECT mode, status, count(*) AS n, round(sum(usd)::numeric, 4) AS usd FROM llm_journal GROUP BY 1, 2 "
        "ORDER BY 1, 2").fetchall()]
    open_ = [dict(r) for r in conn.execute(
        "SELECT id, requests, counts, submitted_at, checked_at FROM llm_batches WHERE status = 'submitted' "
        "ORDER BY submitted_at").fetchall()]
    paid = sum(float(r["usd"] or 0) for r in by if r["mode"] == "batch" and r["status"] == "done")
    discount = float(cfg(policy).get("discount", 0.5))
    return {"journal": by, "open_batches": open_, "batched_usd": round(paid, 4),
            "saved_usd": round(paid * (1 / discount - 1), 4) if discount else None}
