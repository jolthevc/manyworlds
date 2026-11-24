"""The GPU backstop (.github/workflows/gpu-enforcer.yml, `autoscale --once --no-launch`) covers every provider a
policy pool can rent from: each has an adapter in deploy.providers.all_providers and its credentials reach the
workflow, and the no-launch tick still cancels RunPod Serverless requests that run too long."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from manyworlds import queue
from manyworlds.deploy import autoscale, serverless
from manyworlds.deploy.providers import all_providers

from test_autoscale import providers, registry  # noqa: F401
from test_serverless import RunPodServerless, _with_endpoint

NOW = datetime.now(timezone.utc)
WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "gpu-enforcer.yml"
# What each provider adapter reads from the environment to see and stop its machines.
CREDENTIALS = {"salad": {"SALAD_API_KEY", "SALAD_ORG", "SALAD_PROJECT"}, "vast": {"VAST_API_KEY"},
               "runpod": {"RUNPOD_API_KEY"}, "runpod_serverless": {"RUNPOD_API_KEY"},
               "fly": {"FLY_API_TOKEN", "FLY_APP_NAME"}}
NO_MACHINE = {"local"}   # the control plane itself and hosted APIs: nothing rented, nothing to stop


def _workflow_env() -> dict:
    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["enforce"]["steps"]
    return next(s for s in steps if "autoscale" in str(s.get("run", ""))).get("env") or {}


def test_every_policy_pool_has_a_provider_the_backstop_can_stop(policy):
    adapters = set(all_providers(policy)) | {"runpod_serverless"}   # serverless: requests, cancelled by poll
    pools = policy.get("gpu_pools")
    for name, pool in pools.items():
        provider = pool.get("provider")
        assert provider in adapters | NO_MACHINE, f"pool {name}: provider {provider!r} has no backstop adapter"
    used = {p.get("provider") for p in pools.values()} - NO_MACHINE
    assert {"salad", "vast", "runpod", "fly"} <= used                       # incl. the volume pool and the finisher
    assert pools["runpod_secure_l40s_vol"]["provider"] == "runpod"


def test_the_backstop_workflow_gets_every_providers_credentials(policy):
    env = _workflow_env()
    used = {p.get("provider") for p in policy.get("gpu_pools").values()} - NO_MACHINE
    for provider in used:
        missing = CREDENTIALS[provider] - set(env)
        assert not missing, f"gpu-enforcer.yml does not pass {sorted(missing)} ({provider})"
    assert env["FLY_APP_NAME"] == "manyworlds-mw" and "--no-launch" in str(
        yaml.safe_load(WORKFLOW.read_text(encoding="utf-8")))


def test_the_fly_finisher_is_configured_for_the_backstop(policy, monkeypatch):
    monkeypatch.setenv("FLY_API_TOKEN", "test-token")
    monkeypatch.setenv("FLY_APP_NAME", "manyworlds-mw")
    fly = all_providers(policy)["fly"]
    assert fly.configured() and fly.process_groups == ["finisher"]


def test_the_no_launch_tick_cancels_serverless_requests_that_run_too_long(conn, policy, monkeypatch):
    pol = _with_endpoint(policy)
    queue.enqueue(conn, "comfy.wan_i2v", {}, batch="bench-x")
    rp = RunPodServerless()
    serverless.dispatch(conn, pol, autoscale.waiting_work(conn), now=NOW, http=rp)
    monkeypatch.setattr(serverless, "_http", rp)
    actions = autoscale.tick(conn, pol, providers(), now=NOW + timedelta(minutes=61), launch=False)
    assert any(u.endswith(f"/cancel/{next(iter(rp.status))}") for m, u, b in rp.calls)
    assert conn.execute("SELECT status FROM serverless_requests").fetchone()["status"] == "CANCELLED"
    assert any(a.get("action") == "serverless_done" and a["status"] == "CANCELLED" for a in actions)
