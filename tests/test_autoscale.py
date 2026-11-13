"""GPU machines must follow the queue: start when work waits, stop when it's done,
never loop on failures, and never run without a deadline."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from manyworlds import queue
from manyworlds.deploy import autoscale
from manyworlds.deploy.providers import Machine, NoCapacity, ProviderError, redact
from manyworlds.worker.runner import Worker

T0 = datetime(2026, 11, 12, 9, 0, tzinfo=timezone.utc)


class FakeProvider:
    def __init__(self, name, *, configured=True, capacity=True, price=0.15, list_fails=False):
        self.name, self._configured, self.capacity, self.price = name, configured, capacity, price
        self.list_fails = list_fails
        self.live: dict[str, str] = {}
        self.launched: list = []
        self.destroyed: list[str] = []

    def configured(self):
        return self._configured

    def launch(self, name, spec):
        if not self.capacity:
            raise NoCapacity(f"{self.name} has none")
        self.launched.append((name, spec))
        self.live[name] = f"ref-{len(self.launched)}"
        return Machine(name, self.live[name], self.price)

    def list_live(self):
        if self.list_fails:
            raise ProviderError("API down")
        return dict(self.live)

    def destroy(self, name, ref):
        self.destroyed.append(name)
        self.live.pop(name, None)


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setenv("MW_REGISTRY", "docker.io/example")
    monkeypatch.setattr(autoscale, "image_published", lambda image: True)


def providers(**kw):
    ps = {"salad": FakeProvider("salad"), "vast": FakeProvider("vast"), "runpod": FakeProvider("runpod")}
    for name, p in kw.items():
        ps[name] = p
    return ps


def test_waiting_training_starts_one_machine_in_first_pool_with_deadline(conn, policy):
    queue.enqueue(conn, "train.lora", {}, max_attempts=2)
    ps = providers()
    actions = autoscale.tick(conn, policy, ps, now=T0)
    launched = [a for a in actions if a.get("action") == "launched"]
    assert len(launched) == 1 and launched[0]["pool"] == "vast_3090"
    name, spec = ps["vast"].launched[0]
    assert spec.ram_gb == 48 and spec.env["MW_GROUP"] == name and spec.env["MW_POOL"] == "vast_3090"
    tag = policy.get("worker_images", "train", "tag")
    assert spec.image == f"docker.io/example/mw-worker-train:{tag}" and spec.gpu_name == "RTX 3090"
    g = conn.execute("SELECT * FROM gpu_groups WHERE name = %s", (name,)).fetchone()
    assert g["deadline"] == T0 + timedelta(hours=3) and float(g["max_usd"]) == 1.0
    assert g["provider_ref"] == "ref-1" and float(g["usd_per_hour"]) == 0.15
    # Next tick: already served, so nothing new starts.
    autoscale.tick(conn, policy, ps, now=T0 + timedelta(minutes=1))
    assert len(ps["vast"].launched) == 1


def test_falls_back_when_no_capacity_and_skips_unconfigured(conn, policy):
    queue.enqueue(conn, "train.lora", {})
    ps = providers(vast=FakeProvider("vast", capacity=False), runpod=FakeProvider("runpod", configured=False))
    actions = autoscale.tick(conn, policy, ps, now=T0)
    # vast_3090 none, runpod unconfigured, vast_4090 none, runpod skipped -> salad_batch_3090
    assert [a["pool"] for a in actions if a.get("action") == "launched"] == ["salad_batch_3090"]
    assert conn.execute("SELECT count(*) n FROM gpu_groups").fetchone()["n"] == 1   # failed tries left no rows


def test_reports_when_nothing_has_capacity(conn, policy):
    queue.enqueue(conn, "train.lora", {})
    ps = {k: FakeProvider(k, capacity=False) for k in ("salad", "vast", "runpod")}
    actions = autoscale.tick(conn, policy, ps, now=T0)
    assert any(a.get("action") == "no_capacity" for a in actions)


def test_machine_stops_when_queue_drains_and_is_not_restarted(conn, policy):
    job = queue.enqueue(conn, "train.lora", {})
    ps = providers()
    autoscale.tick(conn, policy, ps, now=T0)
    name = ps["vast"].launched[0][0]
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job,))
    autoscale.tick(conn, policy, ps, now=T0 + timedelta(minutes=50))      # last busy at T0+0
    autoscale.tick(conn, policy, ps, now=T0 + timedelta(minutes=70))      # idle > 20 after 45 grace
    assert ps["vast"].destroyed == [name]
    autoscale.tick(conn, policy, ps, now=T0 + timedelta(minutes=71))
    assert len(ps["vast"].launched) == 1


def test_crashed_machine_triggers_cooldown_not_a_launch_loop(conn, policy):
    queue.enqueue(conn, "train.lora", {})
    ps = providers()
    autoscale.tick(conn, policy, ps, now=T0)
    ps["vast"].live.clear()                                             # machine vanished
    actions = autoscale.tick(conn, policy, ps, now=T0 + timedelta(minutes=5))
    assert any(a.get("action") == "cooldown" for a in actions) and len(ps["vast"].launched) == 1
    cool = policy.get("autoscale", "relaunch_cooldown_minutes")
    autoscale.tick(conn, policy, ps, now=T0 + timedelta(minutes=5 + cool + 1))    # cooldown over
    assert len(ps["vast"].launched) == 2


def test_pinned_benchmark_jobs_start_their_own_pool(conn, policy):
    queue.enqueue(conn, "image.zimage", {}, pools=["runpod_comm_4090"])
    ps = providers()
    actions = autoscale.tick(conn, policy, ps, now=T0)
    assert [a["pool"] for a in actions if a.get("action") == "launched"] == ["runpod_comm_4090"]


def test_provider_outage_does_not_mark_machines_gone(conn, policy):
    queue.enqueue(conn, "train.lora", {})
    ps = providers()
    autoscale.tick(conn, policy, ps, now=T0)
    ps["vast"].list_fails = True
    actions = autoscale.tick(conn, policy, ps, now=T0 + timedelta(minutes=5))
    assert any(a.get("action") == "provider_error" for a in actions)
    assert conn.execute("SELECT count(*) n FROM gpu_groups WHERE stopped_at IS NULL").fetchone()["n"] == 1


def test_daily_budget_blocks_launches(conn, policy):
    from manyworlds import ledger
    ledger.record(conn, category="gpu_billed", provider="x", quantity=1, unit="instance_second", usd=50,
                  policy_version=policy.version)
    queue.enqueue(conn, "train.lora", {})
    ps = providers()
    actions = autoscale.tick(conn, policy, ps)
    assert ps["vast"].launched == [] and any(a.get("reason") == "daily_gpu_budget" for a in actions)


def test_claim_respects_pinned_pools(conn):
    queue.enqueue(conn, "image.zimage", {}, pools=["vast_4090"])
    assert queue.claim(conn, ["image.zimage"], "w1", 60, "salad_batch_4090") is None
    assert queue.claim(conn, ["image.zimage"], "w2", 60, "vast_4090") is not None


def test_worker_self_stop_reasons(conn, policy, store, database_url):
    conn.execute("""INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd)
                    VALUES ('mw-train-3090-vast-abcd', 'vast', 'vast_3090', 0.15, '{train.lora}', %s, 1)""",
                 (datetime.now(timezone.utc) + timedelta(hours=1),))
    w = Worker(handlers={}, store=store, policy=policy, pool="vast_3090", database_url=database_url,
               group="mw-train-3090-vast-abcd", self_destroy=lambda: "vast")
    assert w.self_stop_reason() is None
    w._idle_since = time.monotonic() - 21 * 60
    assert w.self_stop_reason() == "idle"
    w._busy = True
    assert w.self_stop_reason() is None
    assert w.self_stop_reason(now=datetime.now(timezone.utc) + timedelta(hours=2)) == "deadline"
    conn.execute("UPDATE gpu_groups SET stopped_at = now() WHERE name = 'mw-train-3090-vast-abcd'")
    assert w.self_stop_reason() == "stopped_by_autoscaler"


def test_errors_never_carry_secrets(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgres://user:hunter2hunter2@db/x")
    assert "hunter2" not in redact("400: bad env {'DATABASE_URL': 'postgres://user:hunter2hunter2@db/x'}")


def test_backstop_without_a_providers_key_does_not_mark_its_machines_gone(conn, policy):
    queue.enqueue(conn, "train.lora", {})
    autoscale.tick(conn, policy, providers(), now=T0)
    # The GitHub backstop has no Vast key: it must keep the machine, not mark it gone.
    blind = providers(vast=FakeProvider("vast", configured=False))
    autoscale.tick(conn, policy, blind, now=T0 + timedelta(minutes=5), launch=False)
    assert conn.execute("SELECT count(*) n FROM gpu_groups WHERE stopped_at IS NULL").fetchone()["n"] == 1


def test_tick_lease_blocks_concurrent_ticks_and_is_released(conn, policy):
    conn.execute("INSERT INTO leases (name, holder, until) VALUES ('autoscale.tick', 'other', now() + interval '5 min')")
    assert autoscale.tick(conn, policy, providers(), now=T0)[0]["reason"] == "another tick is running"
    conn.execute("UPDATE leases SET until = now() - interval '1 s'")          # holder crashed; lease expired
    assert all(a.get("action") != "skipped" for a in autoscale.tick(conn, policy, providers(), now=T0))
    assert conn.execute("SELECT count(*) n FROM leases").fetchone()["n"] == 0


def test_no_machine_is_rented_for_an_unpublished_image(conn, policy, monkeypatch):
    monkeypatch.setattr(autoscale, "image_published", lambda image: False)
    queue.enqueue(conn, "train.lora", {})
    ps = providers()
    actions = autoscale.tick(conn, policy, ps, now=T0)
    assert all(not p.launched for p in ps.values())
    assert conn.execute("SELECT count(*) n FROM gpu_groups").fetchone()["n"] == 0
    assert any("not published" in str(a) for a in actions)


def test_vast_search_excludes_policy_countries(policy, monkeypatch):
    from manyworlds.deploy.providers import LaunchSpec, Vast, all_providers
    sent = {}
    monkeypatch.setattr(Vast, "_req", lambda self, m, path, body=None: sent.update(body or {}) or {"offers": []})
    vast = all_providers(policy)["vast"]
    vast.offers(LaunchSpec(image="x", gpu_name="RTX 3090", cpu=4, ram_gb=24, disk_gb=60, max_usd_per_hour=0.2,
                           env={}, interruptible=False, priority="batch"))
    assert sent["geolocation"] == {"notin": ["CN"]}


def test_hosts_must_support_the_images_cuda(policy, monkeypatch):
    """RunPod hosts with older drivers ran our CUDA 12.8 image and failed every job."""
    from manyworlds.deploy.providers import LaunchSpec, RunPod, Vast, all_providers
    ps = all_providers(policy)
    spec = LaunchSpec(image="x", gpu_name="g", cpu=4, ram_gb=24, disk_gb=60, max_usd_per_hour=0.4,
                      min_cuda=float(policy.get("worker_images", "image", "min_cuda")))
    sent = {}
    monkeypatch.setattr(RunPod, "_req", lambda self, m, path, body=None: sent.update(body or {}) or
                        {"id": "p1", "costPerHr": 0.3})
    ps["runpod"].launch("mw-x", spec)
    assert sent["allowedCudaVersions"] == ["12.8", "12.9", "13.0"]
    sent.clear()
    monkeypatch.setattr(Vast, "_req", lambda self, m, path, body=None: sent.update(body or {}) or {"offers": []})
    ps["vast"].offers(spec)
    assert sent["cuda_max_good"] == {"gte": 12.8}
