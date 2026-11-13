import threading

from manyworlds import bench, ledger, queue
from manyworlds.adapters.base import Capability, ImageRequest
from manyworlds.adapters.gpu import QueueImageGen
from manyworlds.adapters.registry import Registry
from manyworlds.worker.runner import Worker, handlers_for


def make_worker(database_url, store, policy, pool="local", kinds=("fake.image", "fake.video", "fake.tts")):
    return Worker(handlers=handlers_for(list(kinds)), store=store, policy=policy, pool=pool,
                  database_url=database_url)


def test_end_to_end_assets_provenance_and_ledger(conn, database_url, store, policy):
    job_id = queue.enqueue(conn, "fake.image", {"prompt": "cafe selfie", "n": 3, "seed": 7})
    assert make_worker(database_url, store, policy).run(max_jobs=1) == 1

    job = conn.execute("SELECT status, result FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert job["status"] == "done" and len(job["result"]["asset_ids"]) == 3
    assets = conn.execute("SELECT * FROM assets WHERE job_id = %s ORDER BY uri", (job_id,)).fetchall()
    assert [a["kind"] for a in assets] == ["image"] * 3
    assert all(a["recipe"]["prompt"] == "cafe selfie" and a["bytes"] > 0 and len(a["sha256"]) == 64 for a in assets)
    cost = conn.execute("SELECT * FROM cost_entries WHERE job_id = %s", (job_id,)).fetchall()
    assert len(cost) == 1 and cost[0]["unit"] == "gpu_second" and cost[0]["policy_version"] == policy.version
    attempt = conn.execute("SELECT outcome FROM attempts WHERE job_id = %s", (job_id,)).fetchone()
    assert attempt["outcome"] == "done"


def test_failures_are_charged_and_retried_then_failed(conn, database_url, store, policy):
    job_id = queue.enqueue(conn, "fake.image", {"fail": "always"}, max_attempts=2)
    make_worker(database_url, store, policy).run(max_jobs=2)
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()["status"] == "failed"
    n = conn.execute("SELECT count(*) AS n FROM cost_entries WHERE job_id = %s", (job_id,)).fetchone()["n"]
    assert n == 2  # failed attempts still cost money


def test_gpu_cost_uses_pool_price(conn, database_url, store, policy):
    job_id = queue.enqueue(conn, "fake.image", {"sleep": 0.2})
    make_worker(database_url, store, policy, pool="salad_batch_4090").run(max_jobs=1)
    row = conn.execute("SELECT quantity, usd FROM cost_entries WHERE job_id = %s", (job_id,)).fetchone()
    expected = float(row["quantity"]) * 0.16 / 3600
    assert abs(float(row["usd"]) - expected) < 1e-9 and float(row["quantity"]) >= 0.2


def test_many_workers_share_a_batch_without_duplicates(conn, database_url, store, policy):
    batch = bench.enqueue(conn, policy, "fake.image", 24)
    workers = [make_worker(database_url, store, policy) for _ in range(4)]
    threads = [threading.Thread(target=w.run, kwargs={"idle_exit_seconds": 0.5, "poll_seconds": 0.05})
               for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    counts = {r["status"]: r["n"] for r in queue.counts(conn, batch=batch)}
    assert counts == {"done": 24}
    done_attempts = conn.execute(
        "SELECT count(*) AS n FROM attempts a JOIN jobs j ON j.id = a.job_id WHERE j.batch = %s AND a.outcome = 'done'",
        (batch,),
    ).fetchone()["n"]
    assert done_attempts == 24
    report = bench.report(conn, policy, batch)
    assert report["measured"][0]["outputs"] == 24 * 4
    assert ledger.cost_per_output(conn, batch=batch)[0]["outputs"] == 96


def test_queue_adapter_round_trip(conn, database_url, store, policy):
    gen = QueueImageGen(conn, "fake.image", poll_seconds=0.05, timeout_seconds=30)
    worker = make_worker(database_url, store, policy)
    t = threading.Thread(target=worker.run, kwargs={"max_jobs": 1})
    t.start()
    result = gen.generate(ImageRequest(prompt="mirror selfie", n=2, seed=1))
    t.join(timeout=30)
    assert len(result.asset_ids) == 2 and result.usd is not None


def test_gpu_faults_keep_attempts_and_trip_the_circuit_breaker(conn, database_url, store, policy):
    """The real SaladCloud failure: an out-of-memory node burned every job's retries in
    seconds. Machine faults must hand jobs back untouched and stop the worker."""
    import pytest

    from manyworlds.worker.runner import WorkerUnhealthy

    bad = [queue.enqueue(conn, "fake.image", {"fail": "gpu"}, max_attempts=3) for _ in range(5)]
    worker = make_worker(database_url, store, policy)
    with pytest.raises(WorkerUnhealthy):
        worker.run(poll_seconds=0.01)
    rows = conn.execute("SELECT status, attempts FROM jobs WHERE id = ANY(%s)", (bad,)).fetchall()
    assert all(r["status"] == "queued" and r["attempts"] == 0 for r in rows)
    faults = conn.execute("SELECT count(*) n FROM attempts WHERE outcome = 'worker_fault'").fetchone()["n"]
    assert faults == 3  # stopped after the third fault in a row, not after burning all 15 attempts


def test_a_success_resets_the_fault_count(conn, database_url, store, policy):
    worker = make_worker(database_url, store, policy)
    queue.enqueue(conn, "fake.image", {"fail": "gpu"})
    worker.run(max_jobs=2, poll_seconds=0.01)        # same released job faults twice
    assert worker.consecutive_faults == 2
    queue.enqueue(conn, "fake.image", {}, priority=9)  # a healthy job jumps the queue
    worker.run(max_jobs=1, poll_seconds=0.01)
    assert worker.consecutive_faults == 0


def test_job_that_faults_every_machine_eventually_fails(conn, database_url, store, policy):
    job = queue.enqueue(conn, "fake.image", {"fail": "gpu"})
    import pytest

    from manyworlds.worker.runner import WorkerUnhealthy

    for _ in range(2):  # two fresh workers, three faults each = six machine faults
        with pytest.raises(WorkerUnhealthy):
            make_worker(database_url, store, policy).run(poll_seconds=0.01)
    row = conn.execute("SELECT status, worker_faults, attempts FROM jobs WHERE id = %s", (job,)).fetchone()
    assert row["status"] == "failed" and row["worker_faults"] == 6 and row["attempts"] == 0


def test_router_prefers_cheapest_per_accepted_output():
    reg = Registry()
    reg.register(Capability("cheap-but-bad", "ImageGen", config={"est_usd_per_call": 0.0001}),
                 stats={"reviewed": 100, "accepted": 1})
    reg.register(Capability("pricier-but-good", "ImageGen", config={"est_usd_per_call": 0.001}),
                 stats={"reviewed": 100, "accepted": 60})
    assert reg.route("ImageGen").name == "pricier-but-good"


def test_worker_runs_jobs_pinned_to_its_pool_and_leaves_others(conn, database_url, store, policy):
    mine = queue.enqueue(conn, "fake.image", {}, pools=["vast_4090"])
    other = queue.enqueue(conn, "fake.image", {}, pools=["runpod_comm_4090"])
    assert make_worker(database_url, store, policy, pool="vast_4090").run(max_jobs=2, idle_exit_seconds=1) == 1
    status = lambda j: conn.execute("SELECT status FROM jobs WHERE id = %s", (j,)).fetchone()["status"]
    assert status(mine) == "done" and status(other) == "queued"
