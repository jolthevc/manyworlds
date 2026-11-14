"""The autoscaler rents where an accepted output has actually cost least, failures and
idle time included, and stops renting where machines keep producing nothing."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from manyworlds import queue
from manyworlds.deploy import autoscale, scoring

from test_autoscale import providers

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
IMAGE_POOLS = ["salad_batch_3090", "vast_3090", "vast_4090"]


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setenv("MW_REGISTRY", "docker.io/example")
    monkeypatch.setattr(autoscale, "image_published", lambda image: True)


def current_image(family):
    from pathlib import Path

    from manyworlds import policy as pm
    img = pm.load(Path(__file__).resolve().parents[1] / "policy" / "default.yaml").get("worker_images", family)
    return f"{img['image']}:{img['tag']}"


def machine(conn, pool, *, billed, outputs, family="image", hours_ago=2, stopped=True, rejected=0, image=None):
    """A past machine in `pool` that billed `billed` and produced `outputs` assets."""
    name = f"mw-{family}-{pool}-{uuid.uuid4().hex[:4]}"
    started = NOW - timedelta(hours=hours_ago)
    stopped_at = started + timedelta(minutes=30) if stopped else None
    conn.execute(
        """INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd, started_at,
                                   billed_until, billed_usd, last_busy_at, stopped_at, stop_reason, family, image)
           VALUES (%s, 'vast', %s, 0.2, %s, %s, 1, %s, %s, %s, %s, %s, %s, %s, %s)""",
        (name, pool, [f"{family}.x"], started + timedelta(hours=2), started, started, billed, started,
         stopped_at, "idle" if stopped else None, family, image or current_image(family)))
    if outputs:
        job = conn.execute("INSERT INTO jobs (kind, status) VALUES (%s, 'done') RETURNING id",
                           (f"{family}.x",)).fetchone()["id"]
        conn.execute("INSERT INTO attempts (job_id, worker_id, pool, outcome, gpu_group) "
                     "VALUES (%s, 'w', %s, 'done', %s)", (job, pool, name))
        for i in range(outputs):
            a = conn.execute("INSERT INTO assets (job_id, kind, uri, sha256, bytes) VALUES (%s, 'image', %s, 'x', 1) "
                             "RETURNING id", (job, f"s3://b/{name}/{i}")).fetchone()["id"]
            if i < rejected:
                conn.execute("INSERT INTO reviews (asset_id, gate, passed) VALUES (%s, 'identity', false)", (a,))
    return name


def order(conn, policy, pools=IMAGE_POOLS, family="image", now=NOW):
    return scoring.launch_order(conn, policy, family, pools, now)


def test_cost_per_output_beats_hourly_price(conn, policy):
    # Same spend, but the Vast machine made four times as many images.
    machine(conn, "salad_batch_3090", billed=1.0, outputs=25)
    machine(conn, "vast_3090", billed=1.0, outputs=100)
    machine(conn, "vast_4090", billed=1.0, outputs=50)
    pools, rows = order(conn, policy)
    assert pools == ["vast_3090", "vast_4090", "salad_batch_3090"]
    assert rows[0]["usd_per_output"] == pytest.approx(0.01)


def test_unmeasured_pools_get_a_bounded_trial_first(conn, policy):
    machine(conn, "salad_batch_3090", billed=1.0, outputs=100)   # ranked
    machine(conn, "vast_4090", billed=0.10, outputs=4)           # still in trial
    pools, rows = order(conn, policy)
    assert pools == ["vast_3090", "vast_4090", "salad_batch_3090"]  # trials in policy order, then ranked
    machine(conn, "vast_4090", billed=0.25, outputs=4)           # trial budget spent, too few outputs
    pools, rows = order(conn, policy)
    assert pools == ["vast_3090", "salad_batch_3090", "vast_4090"]
    assert {r["pool"]: r["status"] for r in rows}["vast_4090"] == "exhausted"


def test_failures_and_idle_time_count_against_a_pool(conn, policy):
    machine(conn, "salad_batch_3090", billed=0.5, outputs=40)    # $0.0125 per image
    machine(conn, "vast_3090", billed=0.2, outputs=40)           # $0.005 per image when it works...
    machine(conn, "vast_3090", billed=0.4, outputs=0)            # ...but one machine never worked
    pools, rows = order(conn, policy, pools=["vast_3090", "salad_batch_3090"])
    assert pools == ["salad_batch_3090", "vast_3090"]            # $0.0125 beats $0.015 all-in
    by = {r["pool"]: r for r in rows}
    assert by["vast_3090"]["usd_per_output"] == pytest.approx(0.6 / 40)
    assert by["vast_3090"]["failed_machines"] == 1


def test_three_dead_machines_in_a_row_pause_the_pool_then_it_gets_another_trial(conn, policy):
    machine(conn, "salad_batch_3090", billed=1.0, outputs=100)
    for h in (3, 2, 1):
        machine(conn, "vast_3090", billed=0.01, outputs=0, hours_ago=h)
    pools, rows = order(conn, policy)
    assert "vast_3090" not in pools
    assert {r["pool"]: r["status"] for r in rows}["vast_3090"] == "paused"
    later = NOW + timedelta(hours=float(policy.get("autoscale", "scoring", "pause_hours")) + 1)
    assert "vast_3090" in order(conn, policy, now=later)[0]


def test_outputs_a_review_gate_rejected_do_not_count(conn, policy):
    machine(conn, "vast_3090", billed=1.0, outputs=100, rejected=75)
    rows = order(conn, policy, pools=["vast_3090"])[1]
    assert rows[0]["outputs"] == 25 and rows[0]["usd_per_output"] == pytest.approx(0.04)


def test_old_machines_age_out_of_the_window(conn, policy):
    days = float(policy.get("autoscale", "scoring", "window_days"))
    machine(conn, "vast_3090", billed=1.0, outputs=100, hours_ago=24 * days + 1)
    assert order(conn, policy, pools=["vast_3090"])[1][0]["status"] == "trial"


def test_autoscaler_launches_on_the_best_scored_pool(conn, policy):
    machine(conn, "salad_batch_3090", billed=1.0, outputs=25)
    machine(conn, "vast_3090", billed=1.0, outputs=100)
    machine(conn, "vast_4090", billed=1.0, outputs=50)
    for pool in set(policy.get("job_kinds", "image.zimage", "pools")) - set(IMAGE_POOLS):
        machine(conn, pool, billed=1.0, outputs=20)   # every other pool measured, and dearer
    queue.enqueue(conn, "image.zimage", {"prompt": "x"})
    ps = providers()
    actions = autoscale.scale_up(conn, policy, ps, now=NOW)
    launched = [a for a in actions if a["action"] == "launched"]
    assert launched and launched[0]["pool"] == "vast_3090"
    env = ps["vast"].launched[0][1].env
    assert env["MW_OFFLOAD_BELOW_GB"] == "22"     # pool override of the family default
    assert env["MW_GROUP"] == launched[0]["group"]


def test_pinned_work_ignores_scores_and_pauses(conn, policy):
    for h in (3, 2, 1):
        machine(conn, "vast_4090", billed=0.01, outputs=0, hours_ago=h)
    queue.enqueue(conn, "image.zimage", {"prompt": "x"}, pools=["vast_4090"])
    actions = autoscale.scale_up(conn, policy, providers(), now=NOW + timedelta(hours=1))
    assert [a["pool"] for a in actions if a["action"] == "launched"] == ["vast_4090"]


def test_every_pool_paused_launches_nothing(conn, policy):
    for pool in policy.get("job_kinds", "train.lora", "pools"):
        for h in (3, 2, 1):
            machine(conn, pool, billed=0.01, outputs=0, family="train", hours_ago=h)
    queue.enqueue(conn, "train.lora", {})
    ps = providers()
    actions = autoscale.scale_up(conn, policy, ps, now=NOW)
    assert any(a["action"] == "no_pool" for a in actions)
    assert not ps["vast"].launched and not ps["salad"].launched


def test_metering_follows_the_providers_current_price(conn, policy):
    name = machine(conn, "vast_3090", billed=0, outputs=0, stopped=False)
    autoscale.update_prices(conn, {name: 0.233})
    assert float(conn.execute("SELECT usd_per_hour FROM gpu_groups WHERE name = %s", (name,)).fetchone()[
        "usd_per_hour"]) == pytest.approx(0.233)


def test_failures_on_an_older_worker_image_do_not_pause_a_pool(conn, policy):
    # 2026-11-14: a Brotli bug in image v7 failed every machine; v8 fixed it and needed a machine.
    machine(conn, "salad_batch_3090", billed=1.0, outputs=100)
    for h in (3, 2, 1):
        machine(conn, "vast_3090", billed=0.01, outputs=0, hours_ago=h, image="mw-worker-image:old")
    pools, rows = order(conn, policy)
    assert "vast_3090" in pools
    assert {r["pool"]: r["status"] for r in rows}["vast_3090"] != "paused"
