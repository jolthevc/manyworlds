"""RunPod REST v2 (v1 at rest.runpod.io was retired 2026-11-15; every RunPod call failed after that)."""

from __future__ import annotations

import pytest

from manyworlds.deploy import providers
from manyworlds.deploy.providers import LaunchSpec, NoCapacity, ProviderError, RunPod

SPEC = LaunchSpec(image="docker.io/u/mw-worker-talk:v2", gpu_name="NVIDIA GeForce RTX 3090", cpu=8, ram_gb=64,
                  disk_gb=200, max_usd_per_hour=0.3, env={"MW_POOL": "runpod_comm_3090"}, min_cuda=12.8)


class Recorder:
    def __init__(self, *responses):
        self.calls, self.responses = [], list(responses)

    def __call__(self, method, url, headers, body=None, timeout=60):
        self.calls.append((method, url, body))
        r = self.responses.pop(0) if self.responses else None
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def rp(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    return RunPod(["12.4", "12.8", "12.9"])


def test_a_pod_is_created_with_the_v2_body(rp, monkeypatch):
    rec = Recorder({"id": "p1", "status": "PROVISIONING", "cost": 0.22})
    monkeypatch.setattr(providers, "http", rec)
    m = rp.launch("mw-talk-3090-runpod-a", SPEC)
    method, url, body = rec.calls[0]
    assert (method, url) == ("POST", "https://api.runpod.io/v2/pods")
    assert body == {"name": "mw-talk-3090-runpod-a", "image": SPEC.image, "cloud": "COMMUNITY", "disk": 200,
                    "env": {"MW_POOL": "runpod_comm_3090"}, "ports": [],
                    "gpu": {"id": "NVIDIA GeForce RTX 3090", "count": 1, "minRamPerGpu": 64,
                            "allowedCudaVersions": ["12.8", "12.9"]}}
    assert (m.ref, m.usd_per_hour) == ("p1", 0.22)


def test_a_pod_over_the_price_limit_is_terminated_at_once(rp, monkeypatch):
    rec = Recorder({"id": "p1", "cost": 0.9}, None)
    monkeypatch.setattr(providers, "http", rec)
    with pytest.raises(NoCapacity):
        rp.launch("mw-x", SPEC)
    assert rec.calls[1][:2] == ("DELETE", "https://api.runpod.io/v2/pods/p1")


def test_no_capacity_and_no_balance_are_told_apart(rp, monkeypatch):
    monkeypatch.setattr(providers, "http", Recorder(ProviderError(
        'POST https://api.runpod.io/v2/pods -> 400: {"title":"Bad Request","detail":"There are no instances currently '
        'available for this GPU type"}')))
    with pytest.raises(NoCapacity):
        rp.launch("mw-x", SPEC)
    monkeypatch.setattr(providers, "http", Recorder(ProviderError(
        'POST https://api.runpod.io/v2/pods -> 402: {"title":"Payment Required","detail":"balance"}')))
    with pytest.raises(ProviderError, match="insufficient funds"):   # the autoscaler marks the account unfunded
        rp.launch("mw-x", SPEC)


def test_live_pods_are_listed_across_pages_without_terminated_or_foreign_ones(rp, monkeypatch):
    rec = Recorder({"pods": [{"id": "a", "name": "mw-a", "status": "RUNNING"},
                             {"id": "b", "name": "hermes", "status": "RUNNING"}],
                    "pagination": {"hasNextPage": True, "nextCursor": "c2"}},
                   {"pods": [{"id": "c", "name": "mw-c", "status": "TERMINATED"},
                             {"id": "d", "name": "mw-d", "status": "PROVISIONING"}],
                    "pagination": {"hasNextPage": False, "nextCursor": None}})
    monkeypatch.setattr(providers, "http", rec)
    assert rp.list_live() == {"mw-a": "a", "mw-d": "d"}
    assert rec.calls[1][1] == "https://api.runpod.io/v2/pods?cursor=c2"


def test_destroy_deletes_the_pod(rp, monkeypatch):
    rec = Recorder(None)
    monkeypatch.setattr(providers, "http", rec)
    rp.destroy("mw-a", "a")
    assert rec.calls == [("DELETE", "https://api.runpod.io/v2/pods/a", None)]


def test_a_worker_on_runpod_destroys_its_own_pod_through_v2(monkeypatch):
    monkeypatch.delenv("CONTAINER_ID", raising=False)
    monkeypatch.setenv("RUNPOD_POD_ID", "p9")
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    rec = Recorder(None)
    monkeypatch.setattr(providers, "http", rec)
    assert providers.self_destroy() == "runpod"
    assert rec.calls[0][:2] == ("DELETE", "https://api.runpod.io/v2/pods/p9")
