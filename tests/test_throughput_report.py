"""What we pay for, per worker family and pool: billed $ split into boot, busy and idle, accepted outputs, first-pass
rate, retries, $ per accepted output and video second, and accepted outputs per GPU-hour (deploy/report.throughput)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from manyworlds import queue
from manyworlds.deploy import report

from test_autoscale import T0


def _machine(conn, name, pool, family, *, start_min, stop_min, usd):
    conn.execute("INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd, family, "
                 "started_at, stopped_at, billed_usd) VALUES (%s, 'runpod', %s, 1, '{}', %s, 5, %s, %s, %s, %s)",
                 (name, pool, T0 + timedelta(hours=5), family, T0 + timedelta(minutes=start_min),
                  T0 + timedelta(minutes=stop_min), usd))


def _attempt(conn, job, machine, *, start_min, end_min, outcome):
    return conn.execute(
        "INSERT INTO attempts (job_id, worker_id, pool, gpu_group, started_at, finished_at, outcome) "
        "VALUES (%s, 'w', 'p', %s, %s, %s, %s) RETURNING id",
        (job, machine, T0 + timedelta(minutes=start_min), T0 + timedelta(minutes=end_min), outcome)).fetchone()["id"]


def _asset(conn, job, kind, *, seconds=None, passed=None):
    from psycopg.types.json import Jsonb
    aid = conn.execute("INSERT INTO assets (job_id, kind, uri, sha256, bytes, meta) "
                       "VALUES (%s, %s, %s, 'x', 1, %s) RETURNING id",
                       (job, kind, f"s3://b/{job}/{kind}-{seconds}-{passed}",
                        Jsonb({"seconds": seconds} if seconds else {}))).fetchone()["id"]
    if passed is not None:
        conn.execute("INSERT INTO reviews (asset_id, gate, passed) VALUES (%s, 'g', %s)", (aid, passed))


def test_per_family_and_pool_split_billed_time_and_count_accepted_work(conn):
    # talk on the volume pool: 60 min, $1.20; boot 10 min, two attempts of 20 and 10 min, idle 20 min.
    _machine(conn, "m-talk", "runpod_secure_l40s_vol", "talk", start_min=0, stop_min=60, usd=1.20)
    a = queue.enqueue(conn, "fake.video", {})        # first pass: one done attempt, its clip accepted
    _attempt(conn, a, "m-talk", start_min=10, end_min=30, outcome="done")
    _asset(conn, a, "video", seconds=5, passed=True)
    b = queue.enqueue(conn, "fake.video", {})        # failed first on another machine, retried here, rejected
    _machine(conn, "m-talk-old", "runpod_secure_l40s", "talk", start_min=-30, stop_min=-5, usd=0.40)
    _attempt(conn, b, "m-talk-old", start_min=-25, end_min=-10, outcome="failed")
    _attempt(conn, b, "m-talk", start_min=30, end_min=40, outcome="done")
    _asset(conn, b, "video", seconds=8, passed=False)
    # an image machine that never started a job: all boot
    _machine(conn, "m-image", "vast_3090", "image", start_min=0, stop_min=15, usd=0.10)

    rows = {(r["family"], r["pool"]): r for r in report.throughput(conn, hours=24, now=T0 + timedelta(hours=2))}
    t = rows[("talk", "runpod_secure_l40s_vol")]
    assert t["billed_usd"] == pytest.approx(1.20)
    assert (t["boot_usd"], t["busy_usd"], t["idle_usd"]) == pytest.approx((0.20, 0.60, 0.40))
    assert t["accepted_outputs"] == 1 and t["accepted_video_seconds"] == 5
    assert t["attempts"] == 2 and t["retries"] == 1
    assert t["first_pass_rate"] == 1.0                 # job b's first attempt ran elsewhere
    assert t["usd_per_accepted_output"] == pytest.approx(1.20)
    assert t["usd_per_accepted_video_second"] == pytest.approx(0.24)
    assert t["accepted_per_gpu_hour"] == pytest.approx(1.0)

    old = rows[("talk", "runpod_secure_l40s")]
    assert old["first_pass_rate"] == 0.0 and old["accepted_outputs"] == 0 and old["usd_per_accepted_output"] is None
    img = rows[("image", "vast_3090")]
    assert img["boot_usd"] == pytest.approx(0.10) and img["busy_usd"] == 0 and img["first_pass_rate"] is None


def test_machines_outside_the_window_are_left_out(conn):
    _machine(conn, "m-old", "vast_3090", "image", start_min=-60 * 30, stop_min=-60 * 29, usd=1)
    assert report.throughput(conn, hours=24, now=T0) == []
