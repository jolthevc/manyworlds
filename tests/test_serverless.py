"""Serverless GPU pools: one request per waiting job up to the concurrency limit, settled into the ledger, and no
pods rented for the kinds they serve (manyworlds/deploy/serverless.py)."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

from manyworlds import queue
from manyworlds.deploy import autoscale, serverless

from test_autoscale import providers, registry  # noqa: F401

NOW = datetime.now(timezone.utc)


def _with_endpoint(policy, *, concurrency=2):
    pools = {**policy.body["gpu_pools"], "runpod_serverless_l40s": {
        "provider": "runpod_serverless", "gpu": "l40s", "endpoint_id": "ep1", "usd_per_second": 0.0005,
        "max_concurrency": concurrency, "max_request_minutes": 60}}
    kinds = {**policy.body["job_kinds"], "comfy.wan_i2v": {**policy.body["job_kinds"]["comfy.wan_i2v"],
             "pools": ["runpod_serverless_l40s", *policy.body["job_kinds"]["comfy.wan_i2v"]["pools"]]}}
    return dataclasses.replace(policy, body={**policy.body, "gpu_pools": pools, "job_kinds": kinds})


class RunPodServerless:
    def __init__(self):
        self.calls, self.status = [], {}

    def __call__(self, method, url, body=None):
        self.calls.append((method, url, body))
        if url.endswith("/run"):
            rid = f"r{len(self.calls)}"
            self.status[rid] = {"status": "IN_QUEUE"}
            return {"id": rid, "status": "IN_QUEUE"}
        if "/status/" in url:
            return self.status[url.rsplit("/", 1)[1]]
        return {}


def test_waiting_jobs_become_requests_up_to_the_limit_and_no_pods(conn, policy):
    pol = _with_endpoint(policy, concurrency=2)
    for _ in range(3):
        queue.enqueue(conn, "comfy.wan_i2v", {}, batch="bench-x")
    rp, ps = RunPodServerless(), providers()
    actions = serverless.dispatch(conn, pol, autoscale.waiting_work(conn), now=NOW, http=rp)
    assert sum(1 for m, u, b in rp.calls if u.endswith("/ep1/run")) == 2                   # the concurrency limit
    assert rp.calls[0][2]["input"]["kinds"] == ["comfy.wan_i2v"] and actions[-1]["requests"] == 2
    autoscale.scale_up(conn, pol, ps, now=NOW)
    assert not any(p.launched for p in ps.values())                                        # no pods for it
    assert serverless.dispatch(conn, pol, autoscale.waiting_work(conn), now=NOW, http=rp) == []   # still full


def test_a_finished_request_is_billed_by_its_execution_time(conn, policy):
    pol = _with_endpoint(policy)
    queue.enqueue(conn, "comfy.wan_i2v", {}, batch="bench-x")
    rp = RunPodServerless()
    serverless.dispatch(conn, pol, autoscale.waiting_work(conn), now=NOW, http=rp)
    rid = next(iter(rp.status))
    rp.status[rid] = {"status": "COMPLETED", "executionTime": 90_000}
    serverless.poll(conn, pol, now=NOW, http=rp)
    row = conn.execute("SELECT category, usd FROM cost_entries WHERE provider = 'runpod_serverless_l40s'").fetchone()
    assert row["category"] == "gpu_billed" and float(row["usd"]) == 0.045                     # 90 s x $0.0005


def test_a_request_running_too_long_is_cancelled(conn, policy):
    pol = _with_endpoint(policy)
    queue.enqueue(conn, "comfy.wan_i2v", {}, batch="bench-x")
    rp = RunPodServerless()
    serverless.dispatch(conn, pol, autoscale.waiting_work(conn), now=NOW, http=rp)
    serverless.poll(conn, pol, now=NOW + timedelta(minutes=61), http=rp)
    assert any(u.endswith(f"/cancel/{next(iter(rp.status))}") for m, u, b in rp.calls)
    assert conn.execute("SELECT status FROM serverless_requests").fetchone()["status"] == "CANCELLED"


def test_without_an_endpoint_a_serverless_pool_does_nothing(conn, policy):
    assert serverless.pools(policy) == {} and serverless.serves(policy, "comfy.wan_i2v") is None
