"""Outputs go to the job kind that costs least per accepted output (manyworlds/routing.py)."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds import routing

from brain_helpers import creator

NOW = datetime.now(timezone.utc)



def _clips(conn, cid, kind, n, *, usd, accepted, group=None, gpu_usd=0.0, new_group=True, intermediate=False,
           payload=None):
    """n finished clip jobs of `kind`, `accepted` of them passing review, each with `usd` of direct cost."""
    if group and new_group:
        conn.execute("INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd, family, "
                     "billed_usd) VALUES (%s, 'runpod', 'runpod_secure_l40s', 1.09, %s, %s, 3, %s, %s)",
                     (group, [kind], NOW + timedelta(hours=1), kind.split(".")[0], gpu_usd))
    payload = Jsonb({**({"intermediate": True} if intermediate else {}), **(payload or {})})
    for i in range(n):
        job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES (%s, 'done', %s, %s) RETURNING id",
                           (kind, cid, payload)).fetchone()["id"]
        conn.execute("INSERT INTO cost_entries (job_id, category, provider, quantity, unit, usd) "
                     "VALUES (%s, 'hosted_api', 'x', 1, 'clip', %s)", (job, usd))
        if group:
            conn.execute("INSERT INTO attempts (job_id, worker_id, pool, gpu_group, outcome, gpu_seconds) "
                         "VALUES (%s, 'w', 'runpod_secure_l40s', %s, 'done', 60)", (job, group))
        a = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'video', %s, 'x', 1) "
                         "RETURNING id", (job, cid, f"s3://b/{job}.mp4")).fetchone()["id"]
        conn.execute("INSERT INTO reviews (asset_id, gate, passed, details) VALUES (%s, 'identity', %s, %s)",
                     (a, i < accepted, Jsonb({"stage": "confirm"})))


def test_before_anything_is_measured_the_policy_order_decides(conn, raw_policy):
    kind, why = routing.choose(conn, raw_policy, "story_clip", key="p1", now=NOW)
    assert kind == raw_policy.get("production", "routing", "story_clip", "kinds")[0] and "policy order" in why["why"]


def test_the_cheapest_per_accepted_clip_wins_counting_its_machines(conn, raw_policy):
    cid = creator(conn)
    _clips(conn, cid, "hosted.video", 6, usd=0.10, accepted=3)                       # $0.20 per accepted clip
    _clips(conn, cid, "comfy.wan_i2v", 6, usd=0.0, accepted=6, group="g1", gpu_usd=0.30)   # $0.05 incl. idle
    costs = routing.kind_costs(conn, ["hosted.video", "comfy.wan_i2v"], since=NOW - timedelta(days=1))
    assert costs["hosted.video"]["usd_per_accepted"] == 0.2 and costs["comfy.wan_i2v"]["usd_per_accepted"] == 0.05
    picks = [routing.choose(conn, raw_policy, "story_clip", key=f"post-{i}", now=NOW)[0] for i in range(200)]
    share = picks.count("comfy.wan_i2v") / len(picks)
    explore = raw_policy.get("production", "routing", "story_clip", "explore")
    assert 1 - explore - 0.08 < share < 1 - explore + 0.08                          # the rest explores
    assert routing.choose(conn, raw_policy, "story_clip", key="post-7", now=NOW)[0] == picks[7]   # same post, same kind


def test_motion_clips_do_not_count_in_a_kinds_cost_per_accepted_clip(conn, raw_policy):
    cid = creator(conn)
    _clips(conn, cid, "comfy.wan_i2v", 4, usd=0.0, accepted=4, group="g1", gpu_usd=0.40)   # story clips
    # Walk-and-talk's motion clips on the same kind and machine: never a post's output, never reviewed as one.
    _clips(conn, cid, "comfy.wan_i2v", 4, usd=0.05, accepted=0, group="g1", new_group=False, intermediate=True)
    costs = routing.kind_costs(conn, ["comfy.wan_i2v"], since=NOW - timedelta(days=1))["comfy.wan_i2v"]
    # half the machine's time (and none of the motion clips' direct cost) is the story clips': $0.20 / 4
    assert costs["accepted"] == 4 and costs["usd"] == 0.2 and costs["usd_per_accepted"] == 0.05


def test_set_plates_do_not_count_in_a_kinds_cost_per_accepted_image(conn, raw_policy):
    cid = creator(conn)
    _clips(conn, cid, "image.zimage", 4, usd=0.0, accepted=4, group="g1", gpu_usd=0.40)   # scene images
    # Set plates on the same kind and machine (creator.sets): never a post's output.
    _clips(conn, cid, "image.zimage", 4, usd=0.05, accepted=4, group="g1", new_group=False,
           payload={"plate": {"kind": "set", "id": "home_kitchen"}})
    costs = routing.kind_costs(conn, ["image.zimage"], since=NOW - timedelta(days=1))["image.zimage"]
    # the plates' own cost and half the machine are not the scene images': $0.20 / 4
    assert costs["accepted"] == 4 and costs["usd"] == 0.2 and costs["usd_per_accepted"] == 0.05


def test_a_kinds_machines_are_its_policy_familys(conn, raw_policy):
    cid = creator(conn)
    # image.qwen_edit runs on edit machines (job_kinds family: edit), not image ones
    _clips(conn, cid, "image.qwen_edit", 2, usd=0.0, accepted=2, group="g-edit", gpu_usd=0.40)
    conn.execute("UPDATE gpu_groups SET family = 'edit' WHERE name = 'g-edit'")
    with_family = routing.kind_costs(conn, ["image.qwen_edit"], since=NOW - timedelta(days=1), policy=raw_policy)
    assert with_family["image.qwen_edit"]["usd"] == 0.4
    assert routing.kind_costs(conn, ["image.qwen_edit"], since=NOW - timedelta(days=1))["image.qwen_edit"]["usd"] == 0


def test_production_routes_a_story_clip_when_routing_is_on(conn, raw_policy, monkeypatch):
    from manyworlds import production
    called = {}

    def fake_choose(conn_, policy_, output, *, key, now=None):
        called.update(output=output, key=key)
        return "hosted.video", {}
    monkeypatch.setattr(routing, "choose", fake_choose)
    cid = creator(conn)
    post = conn.execute("INSERT INTO posts (creator_id, format) VALUES (%s, 'story_clip') RETURNING *", (cid,)).fetchone()
    look = {k: "x" for k in ("apparent_age", "gender_presentation", "ethnicity_and_skin", "face", "eyes", "hair", "build",
                             "signature_details")}
    conn.execute("UPDATE creators SET bible = %s WHERE id = %s", (Jsonb({"appearance": look}), cid))
    pol = dataclasses.replace(raw_policy, body={**raw_policy.body, "operations": {"mode": "live"}})
    job = production._clip_job(conn, pol, dict(post), {"scenes": ["a cafe"], "script": "steam rises"},
                               [{"uri": "s3://b/a.png"}], 0)
    assert called == {"output": "story_clip", "key": str(post["id"])}
    assert conn.execute("SELECT kind FROM jobs WHERE id = %s", (job,)).fetchone()["kind"] == "hosted.video"
