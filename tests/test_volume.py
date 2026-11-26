"""A full weights volume is fixed by the machines themselves: each prunes the policy's stray paths and measures free
space at boot; below the floor it gives itself up and the autoscaler routes around the volume (2026-11-24: ~47 GB of
LongCat weights filled the 150 GB volume and three machines died at their first download)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from manyworlds import db, queue
from manyworlds.deploy import autoscale
from manyworlds.worker import runner, volume

from test_autoscale import T0, providers, registry  # noqa: F401 - registry: autouse image registry stub

GB = 1024 ** 3
VOL_POOL = "runpod_secure_l40s_vol"


def _file(path, size=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_prune_deletes_listed_paths_under_the_mount_only(tmp_path):
    mount = tmp_path / "models"
    _file(mount / "longcat" / "a.safetensors")
    _file(mount / "mw" / "keep.safetensors")
    _file(tmp_path / "outside" / "x")
    out = volume.prune(mount, ["longcat", "../outside", "/etc", "missing", "mw", "."],
                       protect=[str(mount / "mw")])
    assert not (mount / "longcat").exists()
    assert (mount / "mw" / "keep.safetensors").exists()          # the shared weights cache is protected
    assert (tmp_path / "outside" / "x").exists()                 # never outside the mount
    assert [o["path"] for o in out if o.get("refused")] == ["../outside", "/etc", "mw", "."]
    assert [o for o in out if not o.get("refused")][0]["path"].endswith("longcat")


def test_free_space_is_the_lesser_of_the_filesystem_and_the_volume_size(tmp_path, monkeypatch):
    _file(tmp_path / "w", size=2 * GB // 1024)                    # ~2 MB stored
    monkeypatch.setattr(volume.shutil, "disk_usage", lambda p: type("U", (), {"free": 500 * GB})())
    assert volume.free_gb(tmp_path) == 500                        # no size: the filesystem's own figure
    assert volume.free_gb(tmp_path, size_gb=1) == pytest.approx(1, abs=0.01)


def test_check_prunes_then_measures_against_the_floor(tmp_path, monkeypatch):
    assert volume.check({}) is None                               # no volume on this machine: nothing checked
    _file(tmp_path / "longcat" / "w", size=1024)
    monkeypatch.setattr(volume.shutil, "disk_usage", lambda p: type("U", (), {"free": 5 * GB})())
    env = {"MW_VOLUME_PATH": str(tmp_path), "MW_VOLUME_PRUNE": "longcat", "MW_VOLUME_MIN_FREE_GB": "15"}
    r = volume.check(env)
    assert r["low"] and r["free_gb"] == 5 and not (tmp_path / "longcat").exists()
    assert not volume.check({**env, "MW_VOLUME_MIN_FREE_GB": "4"})["low"]


def _machine(conn, name, pool=VOL_POOL, **cols):
    conn.execute("INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd, family, "
                 "started_at) VALUES (%s, 'runpod', %s, 1.09, '{talk.infinitetalk}', %s, 1, 'talk', %s)",
                 (name, pool, T0 + timedelta(hours=2), T0 - timedelta(minutes=10)))
    for k, v in cols.items():
        conn.execute(f"UPDATE gpu_groups SET {k} = %s WHERE name = %s", (v, name))


def _start(database_url, policy, store, destroyed):
    return runner.start(["fake.image"], pool=VOL_POOL, policy=policy, store=store,
                        connect=lambda url=None, timeout=None: db.connect(database_url, timeout=timeout),
                        self_destroy=lambda: destroyed.append(1) or "runpod")


def test_a_worker_on_a_full_volume_records_it_and_gives_its_machine_up(conn, database_url, policy, store,
                                                                       tmp_path, monkeypatch):
    _machine(conn, "mw-talk-l40s-runpod-v1")
    monkeypatch.setenv("MW_GROUP", "mw-talk-l40s-runpod-v1")
    monkeypatch.setattr(runner, "gpu_memory_gb", lambda: None)
    _file(tmp_path / "longcat" / "w", size=1024)
    for k, v in autoscale.volume_env({**policy.pool(VOL_POOL)["network_volume"], "path": str(tmp_path)}).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(volume.shutil, "disk_usage", lambda p: type("U", (), {"free": 3 * GB})())
    destroyed: list = []
    with pytest.raises(SystemExit) as e:
        _start(database_url, policy, store, destroyed)
    assert e.value.code == 4 and destroyed == [1]
    assert not (tmp_path / "longcat").exists()                    # pruned before anything else
    g = conn.execute("SELECT * FROM gpu_groups WHERE name = 'mw-talk-l40s-runpod-v1'").fetchone()
    assert g["volume_low"] and float(g["volume_free_gb"]) == 3 and g["volume_checked_at"] is not None
    assert g["deadline"] <= g["volume_checked_at"] + timedelta(seconds=5)   # every enforcer now stops it

    monkeypatch.setattr(volume.shutil, "disk_usage", lambda p: type("U", (), {"free": 60 * GB})())
    w = _start(database_url, policy, store, destroyed)            # room again: the worker runs
    assert w is not None and destroyed == [1]
    assert not conn.execute("SELECT volume_low FROM gpu_groups WHERE name = 'mw-talk-l40s-runpod-v1'").fetchone()[
        "volume_low"]


def test_a_volume_machine_is_told_how_to_check_its_volume(conn, policy):
    ps = providers()
    autoscale.launch_one(conn, policy, ps, "talk.infinitetalk", VOL_POOL, T0)
    env = ps["runpod"].launched[0][1].env
    vol = policy.pool(VOL_POOL)["network_volume"]
    assert env["MW_VOLUME_PATH"] == vol["path"] and env["MW_VOLUME_PRUNE"] == ",".join(vol["prune"])
    assert float(env["MW_VOLUME_MIN_FREE_GB"]) == vol["min_free_gb"]
    assert float(env["MW_VOLUME_SIZE_GB"]) == vol["size_gb"]


def test_the_autoscaler_routes_around_a_low_volume_and_tells_the_owner(conn, policy):
    vol = policy.pool(VOL_POOL)["network_volume"]
    _machine(conn, "mw-talk-l40s-runpod-low", volume_low=True, volume_free_gb=2,
             volume_checked_at=T0 - timedelta(minutes=5), stopped_at=T0 - timedelta(minutes=4),
             stop_reason="deadline")
    queue.enqueue(conn, "talk.infinitetalk", {}, batch="bench-talk")
    ps = providers()
    actions = autoscale.tick(conn, policy, ps, now=T0)
    launched = [a for a in actions if a.get("action") == "launched"]
    # no cooldown from the machine that gave itself up, and not the volume pool
    assert launched and launched[0]["pool"] != VOL_POOL
    ev = conn.execute("SELECT * FROM events WHERE type = 'volume_low'").fetchall()
    assert len(ev) == 1 and ev[0]["payload"]["free_gb"] == 2
    # after recheck_hours one machine may try the volume again
    later = T0 + timedelta(hours=vol["recheck_hours"], minutes=1)
    assert autoscale.volume_low(conn, policy, VOL_POOL, later) is None
    assert autoscale.volume_low(conn, policy, VOL_POOL, T0)


def test_machines_that_gave_up_on_a_full_volume_do_not_trip_the_no_output_guard(conn, policy):
    n = policy.get("autoscale", "no_output_guard", "machines")
    img = policy.get("worker_images", "talk")
    for i in range(n):
        _machine(conn, f"mw-talk-l40s-runpod-g{i}", volume_low=True, volume_checked_at=T0 - timedelta(minutes=50),
                 stopped_at=T0 - timedelta(minutes=30 + i), stop_reason="never_started",
                 image=f"{img['image']}:{img['tag']}")
    assert autoscale.no_output_streak(conn, policy, "talk", T0) is None


def test_a_low_reading_does_not_block_a_volume_whose_pending_prune_frees_enough(conn, policy):
    """2026-11-26: the first v26 machine found 10.3 GB free (floor 15) because the old talk engine's 77 GB of weights
    were no longer deleted by sibling pruning; once they are on the prune list the next machine may boot and prune."""
    from psycopg.types.json import Jsonb
    _machine(conn, "mw-talk-l40s-runpod-inv", volume_low=True, volume_free_gb=10.3,
             volume_checked_at=T0 - timedelta(minutes=5), stopped_at=T0 - timedelta(minutes=4), stop_reason="deadline")
    conn.execute("UPDATE gpu_groups SET volume_inventory = %s WHERE name = 'mw-talk-l40s-runpod-inv'",
                 (Jsonb({"mw/comfy": 51.5, "mw/Wan-AI--Wan2.1-I2V-14B-480P": 76.6}),))
    assert autoscale.volume_low(conn, policy, VOL_POOL, T0) is None
    conn.execute("UPDATE gpu_groups SET volume_inventory = %s WHERE name = 'mw-talk-l40s-runpod-inv'",
                 (Jsonb({"mw/comfy": 51.5}),))
    assert autoscale.volume_low(conn, policy, VOL_POOL, T0)          # nothing listed would free space: still low
