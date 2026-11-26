"""Weights persist across machine lifecycles (owner 2026-11-25): each machine records what the volume held at boot,
each job records what it reused or downloaded, and no family deletes another's weights."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds.deploy import report
from manyworlds.worker import comfy, volume, weights_log

GB = 1024 ** 3


def _file(path, size=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_a_job_notes_each_weight_it_reused_or_downloaded(tmp_path):
    weights_log.drain()
    _file(tmp_path / "comfy" / "vae" / "have.safetensors")
    got = lambda repo, name, local_dir, token, **k: _file(tmp_path / "comfy" / "vae" / name)  # noqa: E731
    comfy.ensure_weights(tmp_path / "comfy", {"vae": [("r", "have.safetensors"), ("r", "new.safetensors")]},
                         download=got, remote_size=lambda repo, name, token: 10, prune=True)
    w = weights_log.drain()
    assert w["reused"] == 1 and w["downloaded"] == 1 and {f["file"] for f in w["files"]} == {
        "vae/have.safetensors", "vae/new.safetensors"}
    comfy.ensure_weights(tmp_path / "comfy", {"vae": [("r", "have.safetensors"), ("r", "new.safetensors")]},
                         download=got, remote_size=lambda repo, name, token: 10)
    assert weights_log.drain()["downloaded"] == 0                          # the next machine downloads nothing
    assert weights_log.drain() == {}


def test_one_familys_job_leaves_every_other_familys_weights_on_the_volume(tmp_path):
    _file(tmp_path / "mw" / "svc" / "model.pt", 100)                        # the voice family
    _file(tmp_path / "mw" / "openbmb--VoxCPM2" / "w.safetensors", 100)      # a worker.models repo
    comfy.ensure_weights(tmp_path / "mw" / "comfy", {"vae": [("r", "x.safetensors")]}, prune=True,
                         download=lambda repo, name, local_dir, token, **k: _file(tmp_path / "mw" / "comfy" / "vae" / name),
                         remote_size=lambda *a: None)
    assert (tmp_path / "mw" / "svc" / "model.pt").exists()
    assert (tmp_path / "mw" / "openbmb--VoxCPM2" / "w.safetensors").exists()


def test_the_boot_inventory_lists_each_familys_folder(tmp_path, monkeypatch):
    _file(tmp_path / "mw" / "comfy" / "vae" / "a", int(0.3 * GB) // 1000)
    monkeypatch.setattr(volume, "_bytes_under", lambda p: {"mw": 3 * GB, "comfy": 2 * GB, "svc": GB}.get(p.name, 0))
    _file(tmp_path / "mw" / "svc" / "b")
    inv = volume.inventory(tmp_path)
    assert inv == {"mw": 3.0, "mw/comfy": 2.0, "mw/svc": 1.0}


def test_the_reuse_report_shows_downloads_per_machine_and_any_folder_that_shrank(conn):
    now = datetime.now(timezone.utc)
    for i, (inv, down) in enumerate([({"mw/comfy": 60.0, "mw/svc": 2.0}, 5),
                                     ({"mw/comfy": 60.0, "mw/svc": 2.0}, 0),
                                     ({"mw/comfy": 60.0}, 0)]):
        name = f"mw-talk-vol-{i}"
        start = now - timedelta(hours=5 - i)
        conn.execute(
            """INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd, family,
                                       started_at, volume_inventory, volume_free_gb)
               VALUES (%s, 'runpod', 'runpod_secure_l40s_vol', 1, '{talk.x}', %s, 1, 'talk', %s, %s, 80)""",
            (name, start + timedelta(hours=1), start, Jsonb(inv)))
        jid = conn.execute("INSERT INTO jobs (kind, status, result) VALUES ('talk.x', 'done', %s) RETURNING id",
                           (Jsonb({"weights": {"reused": 9 - down, "downloaded": down,
                                               "downloaded_gb": 4.0 * down}}),)).fetchone()["id"]
        conn.execute("INSERT INTO attempts (job_id, worker_id, pool, outcome, gpu_group, started_at) "
                     "VALUES (%s, 'w', 'p', 'done', %s, %s)", (jid, name, start))
    rows = report.weights_reuse(conn, hours=24, now=now)
    assert [r["files_downloaded"] for r in rows] == [5, 0, 0]
    assert rows[1]["shrunk_since_previous"] == {}
    assert rows[2]["shrunk_since_previous"] == {"mw/svc": [2.0, None]}       # the regression we must never see
