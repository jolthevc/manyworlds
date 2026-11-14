"""Model weights come from our bucket once mirrored: one job copies them in, workers load
them from there, and nothing uses a half-finished copy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from manyworlds import queue
from manyworlds.deploy import scoring
from manyworlds.deploy.enforce import tick
from manyworlds.worker import models

from test_enforce import FakeAPI, T0, register

REPO = "Org/Model"


def fake_repo(tmp_path):
    src = tmp_path / "hf"
    (src / "transformer").mkdir(parents=True)
    (src / "model_index.json").write_text('{"x": 1}')
    (src / "transformer" / "weights.safetensors").write_bytes(b"w" * 5000)
    return src


def test_mirror_then_workers_load_from_the_bucket(store, tmp_path, monkeypatch):
    monkeypatch.setenv("MW_MODEL_DIR", str(tmp_path / "local"))
    assert models.ensure(REPO, store, tmp_path) == REPO              # not mirrored: Hugging Face
    info = models.mirror(REPO, store, tmp_path, source_dir=fake_repo(tmp_path))
    assert info["files"] == 2
    path = models.ensure(REPO, store, tmp_path)
    assert (tmp_path / "local" / "Org--Model" / "transformer" / "weights.safetensors").read_bytes() == b"w" * 5000
    assert path == str(tmp_path / "local" / "Org--Model")
    store.root.joinpath(models.manifest_key(REPO)).unlink()          # the cached copy is used from now on
    assert models.ensure(REPO, store, tmp_path) == path


def test_a_mirror_without_its_manifest_is_never_used(store, tmp_path, monkeypatch):
    monkeypatch.setenv("MW_MODEL_DIR", str(tmp_path / "local"))
    models.mirror(REPO, store, tmp_path, source_dir=fake_repo(tmp_path))
    store.root.joinpath(models.manifest_key(REPO)).unlink()          # as if the upload died before the end
    assert models.ensure(REPO, store, tmp_path) == REPO


def test_the_producer_queues_one_mirror_job_and_gives_up_after_the_limit(conn, policy, store):
    repo = policy.get("models", "mirror")[0]["repo"]
    assert [m["action"] for m in models.ensure_mirrored(conn, policy, store)] == ["queued"]
    assert models.ensure_mirrored(conn, policy, store) == []         # one is already open
    limit = policy.get("models", "max_mirror_jobs")
    for _ in range(limit - 1):
        conn.execute("UPDATE jobs SET status = 'dead' WHERE kind = 'image.mirror'")
        models.ensure_mirrored(conn, policy, store)
    conn.execute("UPDATE jobs SET status = 'dead' WHERE kind = 'image.mirror'")
    assert models.ensure_mirrored(conn, policy, store)[0]["action"] == "gave_up"
    assert conn.execute("SELECT count(*) n FROM jobs WHERE payload->>'repo' = %s", (repo,)).fetchone()["n"] == limit


def test_mirroring_gets_its_own_longer_time_limit(conn, policy):
    register(conn, "mw-image-4090-a", hours=12, max_usd=100, kinds=("image.zimage", "image.mirror"))
    conn.execute("UPDATE gpu_groups SET family = 'image' WHERE name = 'mw-image-4090-a'")
    job = queue.enqueue(conn, "image.mirror", {"repo": REPO})
    conn.execute("INSERT INTO attempts (job_id, worker_id, pool, started_at, gpu_group) "
                 "VALUES (%s, 'w1', 'salad_batch_4090', %s, 'mw-image-4090-a')", (job, T0))
    image_limit = float(policy.get("gpu_runs", "max_attempt_minutes_by_family")["default"])
    at = T0 + timedelta(minutes=image_limit + 5)
    conn.execute("UPDATE jobs SET status = 'running', worker_id = 'w1', updated_at = %s WHERE id = %s", (at, job))
    api = FakeAPI(["mw-image-4090-a"])
    tick(conn, policy, api, now=at)
    assert api.stopped == []                                  # past the image limit, inside the mirror's
    later = T0 + timedelta(minutes=float(policy.get("gpu_runs", "max_attempt_minutes_by_kind")["image.mirror"]) + 1)
    conn.execute("UPDATE jobs SET updated_at = %s WHERE id = %s", (later, job))
    actions = tick(conn, policy, api, now=later)
    assert api.stopped == ["mw-image-4090-a"] and actions[0]["reason"] == "attempt_timeout"


def test_a_machine_that_only_mirrored_is_not_a_failed_machine(conn, policy):
    now = datetime(2026, 11, 14, 12, tzinfo=timezone.utc)
    for h in (3, 2, 1):
        name = f"mw-image-3090-vast-{h}"
        conn.execute("""INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd,
                        started_at, billed_until, billed_usd, last_busy_at, stopped_at, stop_reason, family)
                        VALUES (%s, 'vast', 'vast_3090', 0.2, '{image.mirror}', %s, 1, %s, %s, 0.02, %s, %s, 'idle', 'image')""",
                     (name, now, now - timedelta(hours=h), now, now, now - timedelta(hours=h) + timedelta(minutes=20)))
        job = queue.enqueue(conn, "image.mirror", {"repo": REPO})
        conn.execute("INSERT INTO attempts (job_id, worker_id, pool, outcome, gpu_group) VALUES (%s, 'w', 'vast_3090', "
                     "'done', %s)", (job, name))
    row = scoring.score_pools(conn, policy, "image", ["vast_3090"], now)[0]
    assert row["status"] != "paused" and row["failed_machines"] == 0
