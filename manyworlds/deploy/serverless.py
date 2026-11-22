"""Serverless GPU pools (RunPod Serverless): per-second GPUs that exist only while they work.

Rented pods bill boot, model loading and idle time; at low or bursty volume that was most of the bill (95% of the
GPU spend in one 72 h window produced nothing, 2026-11-21). A serverless endpoint runs our same worker image
(`mw serverless-worker`) with weights on a network volume, scales itself between zero and a limit, and bills
the seconds each request runs.

The queue stays the source of truth. For every job waiting in a kind a serverless pool serves, the dispatcher
sends one request to the endpoint, up to the pool's `max_concurrency` in flight; each request makes the worker
claim and run one job from Postgres (any job of those kinds, so a lost request loses nothing), then return. When a
request finishes, RunPod's execution time is booked to the ledger as `gpu_billed` at the pool's
`usd_per_second`, so the daily caps and cost per accepted output count it. Requests running past
`max_request_minutes` are cancelled. A pool serves a kind when it comes first in job_kinds.<kind>.pools (or a
job is pinned to it), and only once its `endpoint_id` is set; the autoscaler then rents no pods for that kind.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

from ..policy import Policy

API = "https://api.runpod.ai/v2"
DONE = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


def pools(policy: Policy) -> dict[str, dict]:
    """Serverless pools that are live (have an endpoint)."""
    return {name: p for name, p in (policy.get("gpu_pools", default={}) or {}).items()
            if p.get("provider") == "runpod_serverless" and p.get("endpoint_id")}


def serves(policy: Policy, kind: str, pinned: list[str] | None = None) -> str | None:
    """The live serverless pool that serves this kind's work, if any."""
    live = pools(policy)
    if pinned:
        return next((p for p in pinned if p in live), None)
    first = (policy.get("job_kinds", kind, "pools", default=[]) or [None])[0]
    return first if first in live else None


def _http(method: str, url: str, body: dict | None = None):
    from .providers import http
    return http(method, url, {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}, body)


def poll(conn, policy: Policy, *, now: datetime, http=None) -> list[dict]:
    """Settle finished requests (ledger) and cancel ones running too long."""
    http = http or _http
    from .. import ledger
    actions = []
    for r in conn.execute("SELECT * FROM serverless_requests WHERE finished_at IS NULL").fetchall():
        p = (policy.get("gpu_pools", r["pool"], default={}) or {})
        try:
            st = http("GET", f"{API}/{r['endpoint']}/status/{r['id']}") or {}
        except Exception as e:  # noqa: BLE001 - try again next tick
            actions.append({"action": "serverless_poll_failed", "request": r["id"], "error": str(e)[:200]})
            continue
        status = str(st.get("status", "UNKNOWN"))
        too_long = now - r["created_at"] > timedelta(minutes=float(p.get("max_request_minutes", 60)))
        if status not in DONE and too_long:
            try:
                http("POST", f"{API}/{r['endpoint']}/cancel/{r['id']}")
            except Exception:  # noqa: BLE001 - the next poll sees it again
                pass
            status = "CANCELLED"
        if status not in DONE:
            conn.execute("UPDATE serverless_requests SET status = %s WHERE id = %s", (status, r["id"]))
            continue
        ms = int(st.get("executionTime") or 0)
        usd = round(ms / 1000 * float(p.get("usd_per_second", 0)), 6)
        ledger.record(conn, category="gpu_billed", provider=r["pool"], quantity=ms / 1000, unit="execution_second",
                      usd=usd, policy_version=policy.version)
        conn.execute("UPDATE serverless_requests SET status = %s, finished_at = %s, execution_ms = %s, usd = %s "
                     "WHERE id = %s", (status, now, ms, usd, r["id"]))
        actions.append({"action": "serverless_done", "pool": r["pool"], "status": status, "usd": usd})
    return actions


def dispatch(conn, policy: Policy, work: list[dict], *, now: datetime, http=None) -> list[dict]:
    """One request per waiting job in serverless-served kinds, up to each pool's concurrency limit."""
    http = http or _http
    live = pools(policy)
    if not live:
        return []
    actions = []
    want: dict[str, dict] = {}
    for w in work:
        pool = serves(policy, w["kind"], w.get("pools"))
        if pool:
            d = want.setdefault(pool, {"n": 0, "kinds": set()})
            d["n"] += int(w["n"])
            d["kinds"].add(w["kind"])
    for pool, d in want.items():
        p = live[pool]
        inflight = conn.execute("SELECT count(*) AS n FROM serverless_requests WHERE pool = %s AND finished_at IS NULL",
                                (pool,)).fetchone()["n"]
        send = max(0, min(d["n"] - inflight, int(p.get("max_concurrency", 2)) - inflight))
        kinds = sorted(d["kinds"])
        for _ in range(send):
            rid = uuid.uuid4().hex[:12]
            r = http("POST", f"{API}/{p['endpoint_id']}/run",
                     {"input": {"kinds": kinds, "pool": pool, "request": rid, "max_jobs": 1}}) or {}
            conn.execute("INSERT INTO serverless_requests (id, endpoint, pool, kinds, status, created_at) "
                         "VALUES (%s, %s, %s, %s, %s, %s)",
                         (r.get("id") or rid, p["endpoint_id"], pool, kinds, r.get("status", "IN_QUEUE"), now))
        if send:
            actions.append({"action": "serverless_dispatched", "pool": pool, "requests": send, "kinds": kinds})
    return actions
