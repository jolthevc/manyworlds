"""The breadth test: the test overlay, plans by capability, a dry run that creates nothing, a run that resumes itself
and only touches its own posts, the urgent test batch, one merged machine for voice + comfy + talk, and the CLI."""

from __future__ import annotations

import dataclasses
import json
import uuid
from datetime import datetime, timezone

import pytest
from psycopg.types.json import Jsonb

from manyworlds import breadth, cli, operations, policy as policy_mod, queue
from manyworlds.deploy import autoscale
from manyworlds.worker import runner

from test_autoscale import providers
from test_review import BIBLE

NOW = datetime(2026, 11, 23, 9, 0, tzinfo=timezone.utc)


def _paused(policy):
    return dataclasses.replace(policy, body={**policy.body, "operations": {**policy.body["operations"], "mode": "paused"}})


def _creator(conn, refs=3, anchors=None):
    handle = f"c{uuid.uuid4().hex[:6]}"
    conn.execute("INSERT INTO creators (handle, bible, anchors) VALUES (%s, %s, %s)",
                 (handle, Jsonb({**BIBLE, "display_name": "M"}),
                  Jsonb({"reference_images": [{"uri": f"s3://b/r{i}"} for i in range(refs)], **(anchors or {})})))
    return handle


def _count(conn, table):
    return conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"]


# ------------------------------------------------------------------------------------------------------- overlay

def test_deep_merge_merges_dicts_and_replaces_lists():
    base = {"a": {"b": 1, "c": [1, 2], "d": {"e": 1}}, "x": 1}
    assert policy_mod.deep_merge(base, {"a": {"c": [3], "d": {"f": 2}}}) == \
        {"a": {"b": 1, "c": [3], "d": {"e": 1, "f": 2}}, "x": 1}
    assert base["a"]["c"] == [1, 2]   # the base is not changed


def test_the_test_overlay_holds_the_test_settings(raw_policy):
    pol = breadth.test_policy(raw_policy)
    formats = pol.get("production", "formats")
    assert {"talking_video", "narrated_edit", "photo", "carousel", "story", "story_clip"} <= set(formats)
    assert pol.get("production", "reroute_failed_video") == {}
    assert pol.get("review", "direction", "min_score") == 0
    assert pol.get("review", "direction", "rubric") == raw_policy.get("review", "direction", "rubric")   # merged
    assert pol.get("generation", "talk", "max_take_seconds") == 30
    assert pol.get("generation", "talk", "steps") == raw_policy.get("generation", "talk", "steps")
    assert pol.get("autonomy", "publish") == "paused"
    assert pol.version != raw_policy.version and breadth.test_policy(pol).version == pol.version   # idempotent
    # production itself never reads it: the shipped policy is unchanged
    assert "talking_video" not in raw_policy.get("production", "formats")


def test_the_overlay_applies_only_to_its_batches(raw_policy):
    assert raw_policy.get("autoscale", "merge_families") is None
    over = policy_mod.for_batch(raw_policy, "test-breadth-default-abc123")
    assert "talkcomfy" in over.get("autoscale", "merge_families")
    for batch in ("post-123", "bench-talk.infinitetalk-x", "test-voice-maren", None):
        assert policy_mod.for_batch(raw_policy, batch) is raw_policy
    prefix = breadth._cfg(over)["batch_prefix"]
    assert any(prefix.startswith(p) for p in raw_policy.get("operations", "test_batch_prefixes"))
    assert any(prefix.startswith(p) for p in policy_mod.overlay_batches(raw_policy))


# --------------------------------------------------------------------------------------------------------- plans

def test_the_default_plan_parses_and_every_capability_is_registered(raw_policy):
    pol = breadth.test_policy(raw_policy)
    plan, items = breadth.load_plan(pol, "default")
    caps = breadth._cfg(pol)["capabilities"]
    assert plan["creator"] and len(items) >= 18
    for item in items:
        assert set(item.exercises) <= set(caps), item.id
        assert item.bench or item.format in breadth.POST_FORMATS, item.id   # formats are existing ones only
    covered = {c for i in items for c in i.exercises}
    for cap in ("talking", "walking", "posing", "dancing", "prop_interaction", "makeup_interaction", "camera_mirror",
                "camera_selfie", "camera_propped", "camera_pov", "camera_follow", "outfit_change", "cut_on_motion",
                "labels", "timestamps", "hooks", "voiceover_broll", "room_tone", "music", "designed_voice",
                "longcat", "hd720", "reference_conditioning", "caption_styles", "cutaway_broll", "scene_change"):
        assert cap in covered, cap
    _, two = breadth.load_plan(pol, "default", ["talk_kokoro", "photo_propped_pose"])
    assert [i.id for i in two] == ["photo_propped_pose", "talk_kokoro"]   # plan order
    with pytest.raises(ValueError):
        breadth.load_plan(pol, "default", ["nope"])
    with pytest.raises(ValueError):
        breadth.load_plan(pol, "missing")


def test_bad_items_are_refused_when_parsed():
    with pytest.raises(ValueError):
        breadth.parse_item({"id": "x", "format": "photo", "body": {"scenes": ["a"]}})            # no exercises
    with pytest.raises(ValueError):
        breadth.parse_item({"id": "x", "exercises": ["photo"], "format": "photo", "body": {}})   # no scenes
    with pytest.raises(ValueError):
        breadth.parse_item({"id": "x", "exercises": ["photo"], "format": "photo",
                            "body": {"scenes": ["a"], "shots": [{}, {}]}})                       # shots != scenes
    with pytest.raises(ValueError):
        breadth.parse_item({"id": "x", "exercises": ["dancing"], "format": "photo", "body": {"scenes": ["a"]},
                            "bench": {"kind": "comfy.wan_animate2"}})                            # both


def test_run_ids_are_stable_and_batch_safe(raw_policy):
    pol = breadth.test_policy(raw_policy)
    _, items = breadth.load_plan(pol, "default")
    a = breadth.run_id("default", items, "maren")
    assert a == breadth.run_id("default", items, "maren") and a.startswith("default-")
    assert a != breadth.run_id("default", items[:2], "maren") != breadth.run_id("default", items, "other")
    assert breadth.run_id("My_Plan", items).startswith("my-plan-")   # no LIKE wildcards in a batch
    assert breadth.batch_name(pol, a) == f"test-breadth-{a}"


def test_items_whose_capability_or_format_is_not_wired_are_skipped_with_a_reason(conn, policy):
    pol = breadth.test_policy(_paused(policy))
    handle = _creator(conn)
    cr = breadth._creator(conn, handle)
    item = breadth.parse_item({"id": "x", "exercises": ["telepathy"], "format": "photo", "body": {"scenes": ["a"]}})
    assert "not registered" in breadth.skip_reason(pol, item, cr)
    pol2 = policy_mod.from_dict(policy_mod.deep_merge(pol.body, {"breadth": {"capabilities": {
        "telepathy": {"requires": ["nothing.here.enabled"]}}}}))
    assert "not wired yet" in breadth.skip_reason(pol2, item, cr)
    later = breadth.parse_item({"id": "y", "exercises": ["photo"], "format": "grwm", "body": {"scenes": ["a"]}})
    assert "not a post format" in breadth.skip_reason(pol, later, cr)
    voice = breadth.parse_item({"id": "z", "exercises": ["talking"], "format": "talking_video",
                                "body": {"scenes": ["a"], "script": "hi"}, "options": {"voice_engine": "voxcpm"}})
    assert "kokoro" in breadth.skip_reason(pol, voice, cr)
    assert "no creator" in breadth.skip_reason(pol, voice, None)
    bad = breadth.parse_item({"id": "s", "exercises": ["photo"], "format": "photo",
                              "body": {"scenes": ["a"], "shots": [{"framing": {"device": "drone"}}]}})
    assert "not a valid shot sheet" in breadth.skip_reason(pol, bad, cr)


# -------------------------------------------------------------------------------------------------------- dry run

def test_dry_run_creates_nothing_and_estimates_from_policy(conn, policy):
    handle = _creator(conn)
    out = breadth.run(conn, _paused(policy), "default", creator=handle, dry_run=True)
    assert _count(conn, "posts") == 0 and _count(conn, "jobs") == 0 and _count(conn, "leases") == 0
    assert out["dry_run"] and out["batch"] == f"test-breadth-{out['run']}"
    assert out["priority"] == policy.get("autoscale", "batch", "urgent_priority")
    assert "talk_designed_voice" in out["skipped"]   # she still speaks with Kokoro (and captions are off in tests)
    by_id = {i["id"]: i for i in out["items"]}
    talk = by_id["talk_one_shot_30s"]["estimate"]["jobs"]
    assert any(j["kind"] == "talk.infinitetalk" and j["source"].startswith("policy") for j in talk)
    assert all(i["would"] == "create" for i in out["items"])
    assert out["estimated_usd"] > 0 and out["machines"]
    assert any(m["family"] == "talkcomfy" for m in out["machines"])   # voice, comfy and talk: one machine


# ------------------------------------------------------------------------------------------------------- the run

def test_a_run_makes_only_its_own_posts_in_an_urgent_test_batch_and_resumes(conn, policy, store):
    pol = _paused(policy)
    handle = _creator(conn)
    other = conn.execute("INSERT INTO posts (creator_id, format, status, body, scheduled_for) SELECT id, 'photo', "
                         "'planned', %s, now() FROM creators WHERE handle = %s RETURNING id",
                         (Jsonb({"scenes": ["a balcony"]}), handle)).fetchone()["id"]
    items = ["photo_propped_pose", "talk_longcat"]
    out = breadth.run(conn, pol, "default", items=items, creator=handle, store=store, once=True)
    batch = out["batch"]
    posts = conn.execute("SELECT id, status, scheduled_for, body FROM posts WHERE body->>'breadth_run' = %s",
                         (out["run"],)).fetchall()
    assert sorted(p["body"]["breadth_item"] for p in posts) == sorted(items)
    assert all(p["status"] == "producing" and p["scheduled_for"] is None for p in posts)   # never due to publish
    assert all(p["body"]["platform_ai_label"] for p in posts)
    jobs = conn.execute("SELECT batch, priority FROM jobs").fetchall()
    assert jobs and all(j["batch"] == batch for j in jobs)
    assert all(j["priority"] == breadth.urgent_priority(breadth.test_policy(pol)) for j in jobs)
    assert conn.execute("SELECT status FROM posts WHERE id = %s", (other,)).fetchone()["status"] == "planned"
    n = _count(conn, "jobs")
    again = breadth.run(conn, pol, "default", items=items, creator=handle, store=store, once=True)
    assert again["run"] == out["run"] and _count(conn, "jobs") == n
    assert conn.execute("SELECT count(*) AS n FROM posts WHERE body->>'breadth_run' = %s",
                        (out["run"],)).fetchone()["n"] == 2
    # paused machines may take the run's batch
    patterns = operations.test_batch_patterns(pol)
    job = queue.claim(conn, ["image.zimage"], "w1", 60, pool="runpod_comm_3090", batches=patterns)
    assert job is not None and job.batch == batch
    rep = breadth.report(conn, pol, out["run"], store=store, probe=False)
    assert {i["id"] for i in rep["items"]} == set(items)
    assert "photo" in rep["by_capability"] and "longcat" in rep["by_capability"]
    assert rep["skipped"]["talk_push_in"]["reason"] == "not in this run"


def test_a_live_system_refuses_the_run(conn, policy, store):
    live = dataclasses.replace(policy, body={**policy.body, "operations": {**policy.body["operations"], "mode": "live"}})
    with pytest.raises(SystemExit):
        breadth.run(conn, live, "default", items=["photo_propped_pose"], creator=_creator(conn), store=store,
                    once=True, sleep=lambda s: None)


def test_adopted_talk_jobs_get_the_item_profile_and_the_plan_pools(conn, policy):
    pol = breadth.test_policy(_paused(policy))
    handle = _creator(conn)
    cid = breadth._creator(conn, handle)["id"]
    post = conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, 'talking_video', "
                        "'producing', %s) RETURNING id", (cid, Jsonb({"scenes": ["a"], "breadth_run": "r1",
                                                                      "breadth_item": "t",
                                                                      "breadth_options": {"talk_profile": "longcat"}}))
                        ).fetchone()["id"]
    job = queue.enqueue(conn, "talk.infinitetalk", {"params": {"steps": 10, "width": 480}}, batch=f"post-{post}",
                        post_id=post, creator_id=cid)
    plan, _ = breadth.load_plan(pol, "default")
    assert breadth.adopt_jobs(conn, pol, plan, "r1", "test-breadth-r1") == 1
    j = conn.execute("SELECT batch, priority, pools, payload FROM jobs WHERE id = %s", (job,)).fetchone()
    longcat = pol.get("generation", "talk_profiles", "longcat")
    assert j["batch"] == "test-breadth-r1" and j["priority"] == breadth.urgent_priority(pol)
    assert j["pools"] == breadth._cfg(pol)["pools_by_family"]["talk"]
    assert j["payload"]["params"]["model"] == longcat["model"] and j["payload"]["params"]["width"] == 480
    assert j["payload"]["talk_profile"] == "longcat"
    assert breadth.adopt_jobs(conn, pol, plan, "r1", "test-breadth-r1") == 0   # once


def test_the_deadline_cancels_queued_jobs_and_any_retry(conn):
    a = queue.enqueue(conn, "image.zimage", {}, batch="test-breadth-r2")
    b = queue.enqueue(conn, "comfy.wan_animate2", {}, batch="test-breadth-r2-dance")
    c = queue.enqueue(conn, "image.zimage", {}, batch="test-breadth-r20")   # another run
    live = queue.enqueue(conn, "talk.infinitetalk", {}, batch="test-breadth-r2")
    lapsed = queue.enqueue(conn, "talk.infinitetalk", {}, batch="test-breadth-r2")
    conn.execute("UPDATE jobs SET status = 'running', attempts = 1, lease_until = now() + interval '5 min' WHERE id = %s", (live,))
    conn.execute("UPDATE jobs SET status = 'running', attempts = 1, lease_until = now() - interval '5 min' WHERE id = %s", (lapsed,))
    assert breadth.cancel_queued(conn, "test-breadth-r2") == 3
    rows = {r["id"]: r for r in conn.execute("SELECT id, status, attempts, max_attempts FROM jobs").fetchall()}
    assert rows[a]["status"] == rows[b]["status"] == rows[lapsed]["status"] == "failed" and rows[c]["status"] == "queued"
    assert rows[live]["status"] == "running" and rows[live]["max_attempts"] == rows[live]["attempts"]   # no retry


# ------------------------------------------------------------------------------------------------- one machine

@pytest.fixture()
def registry(monkeypatch):
    monkeypatch.setenv("MW_REGISTRY", "docker.io/example")
    monkeypatch.setattr(autoscale, "image_published", lambda image: True)


def test_test_work_across_voice_comfy_and_talk_starts_one_merged_machine(conn, policy, registry):
    pol = _paused(policy)
    pools = ["runpod_secure_l40s_vol"]
    queue.enqueue(conn, "voice.speak", {}, batch="test-breadth-r3", pools=pools, priority=1)
    queue.enqueue(conn, "comfy.wan_i2v", {}, batch="test-breadth-r3", pools=pools, priority=1)
    queue.enqueue(conn, "talk.infinitetalk", {}, batch="test-breadth-r3", pools=pools, priority=1)
    ps = providers()
    actions = autoscale.tick(conn, pol, ps, now=NOW)
    launched = [a for a in actions if a.get("action") == "launched"]
    assert len(launched) == 1 and len(ps["runpod"].launched) == 1
    name, spec = ps["runpod"].launched[0]
    assert name.startswith("mw-talkcomfy-")
    kinds = set(spec.env["MW_WORKER_KINDS"].split(","))
    assert {"voice.speak", "voice.design", "comfy.wan_i2v", "talk.infinitetalk", "comfy.wan_animate2"} <= kinds
    assert spec.env["MW_WORKER_FAMILIES"] == "voice,comfy,talk,dance"
    imgs = [pol.get("worker_images", f) for f in ("voice", "comfy", "talk", "dance")]
    newest = max((i["tag"] for i in imgs), key=autoscale._tag_order)
    assert spec.image == f"docker.io/example/mw-worker-talkcomfy:{newest}"
    assert spec.ram_gb == max(i["ram_gb"] for i in imgs)
    g = conn.execute("SELECT family, job_kinds, max_usd, max_attempt_minutes FROM gpu_groups WHERE name = %s",
                     (name,)).fetchone()
    runs = pol.get("gpu_runs")
    assert g["family"] == "talkcomfy"
    assert float(g["max_attempt_minutes"]) == max(float(runs["max_attempt_minutes_by_family"].get(
        f, runs["max_attempt_minutes_by_family"]["default"])) for f in ("voice", "comfy", "talk", "dance"))
    assert float(g["max_usd"]) == max(float(runs["max_usd_by_family"].get(f, runs["default_max_usd"]))
                                      for f in ("voice", "comfy", "talk", "dance"))
    autoscale.tick(conn, pol, ps, now=NOW)   # the merged machine serves all three: nothing else starts
    assert len(ps["runpod"].launched) == 1


def test_without_the_overlay_each_family_keeps_its_own_machine(conn, policy, registry):
    pol = _paused(policy)
    queue.enqueue(conn, "voice.speak", {}, batch="bench-voice.speak-x", pools=["runpod_secure_l40s_vol"])
    ps = providers()
    autoscale.tick(conn, pol, ps, now=NOW)
    name, spec = ps["runpod"].launched[0]
    assert name.startswith("mw-voice-") and spec.env["MW_WORKER_FAMILIES"] == "voice"
    assert set(spec.env["MW_WORKER_KINDS"].split(",")) == set(autoscale._family_kinds(pol, "voice"))
    assert autoscale.machine_family(pol, "voice.speak") == "voice"


def test_merged_images_must_share_one_image(policy):
    pol = policy_mod.from_dict(policy_mod.deep_merge(policy.body, {"autoscale": {"merge_families": {
        "mixed": ["image", "talk"]}}}))
    with pytest.raises(autoscale.ProviderError):
        autoscale.merged_image(pol, "mixed")
    assert autoscale.machine_family(pol, "talk.infinitetalk") == "mixed"


def test_a_worker_serves_its_families_kinds_from_the_environment(policy):
    assert runner.kinds_from_env(policy, {"MW_WORKER_KINDS": "talk.infinitetalk, voice.speak"}) == \
        ["talk.infinitetalk", "voice.speak"]
    kinds = runner.kinds_from_env(policy, {"MW_WORKER_FAMILIES": "voice,talk"})
    assert set(kinds) == {k for k in policy.get("job_kinds") if policy.family(k) in ("voice", "talk")}
    with pytest.raises(SystemExit):
        runner.kinds_from_env(policy, {})


# ------------------------------------------------------------------------------------------------------------ CLI

def test_cli_parses_the_breadth_commands():
    p = cli.build_parser()
    a = p.parse_args(["test", "breadth", "--plan", "default", "--dry-run", "--items", "a,b", "--hours", "6"])
    assert (a.cmd, a.test_cmd, a.plan, a.dry_run, a.items, a.hours) == ("test", "breadth", "default", True, "a,b", 6.0)
    a = p.parse_args(["test", "breadth"])
    assert a.plan == "default" and not a.dry_run and a.run is None
    a = p.parse_args(["test", "breadth-report", "default-abc123", "--no-probe"])
    assert a.test_cmd == "breadth-report" and a.run == "default-abc123" and a.no_probe


def test_cli_dry_run_prints_the_plan(conn, policy, capsys):
    handle = _creator(conn)
    args = cli.build_parser().parse_args(["test", "breadth", "--dry-run", "--creator", handle,
                                          "--items", "photo_propped_pose"])
    assert cli._test(args, conn, _paused(policy)) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] and [i["id"] for i in out["items"]] == ["photo_propped_pose"]
    assert _count(conn, "posts") == 0


def test_the_producer_never_touches_breadth_posts(conn, policy, store):
    """Once the owner goes live, unfinished breadth posts must not be produced under the base policy."""
    from manyworlds import production
    handle = _creator(conn)
    cid = conn.execute("SELECT id FROM creators WHERE handle = %s", (handle,)).fetchone()["id"]
    pids = [conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, 'photo', %s, %s) RETURNING id",
                         (cid, status, Jsonb({"breadth_run": "r1", "scenes": ["a kitchen"]}))).fetchone()["id"]
            for status in ("planned", "producing")]
    production.advance(conn, policy, now=NOW, store=store)
    assert [conn.execute("SELECT status FROM posts WHERE id = %s", (p,)).fetchone()["status"] for p in pids] == [
        "planned", "producing"]
    assert _count(conn, "jobs") == 0


def test_the_merged_machine_has_its_own_caps_and_a_small_disk_on_the_volume(policy, tmp_path):
    pol = breadth.test_policy(policy)
    img = autoscale.merged_image(pol, "talkcomfy")
    members = [pol.get("worker_images", f) for f in ("voice", "comfy", "talk", "dance")]
    assert img["volume_disk_gb"] == max(int(m["disk_gb"]) for m in members) < img["disk_gb"]
    assert autoscale._limit(pol, "talkcomfy", "max_hours_by_family", "max_hours", "default_max_hours", 2) == 6
    assert "runpod_secure_l40s_vol" not in breadth._cfg(pol)["pools_by_family"]["edit"]   # Qwen edit off the volume
    empty = tmp_path / "empty.yaml"
    empty.write_text("# nothing\n", encoding="utf-8")
    assert policy_mod.with_overlay(policy, empty).body == policy.body                      # an empty overlay is no-op
