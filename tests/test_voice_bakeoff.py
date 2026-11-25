"""The voice bake-off (manyworlds/voice_bakeoff.py): a breadth plan of voice.cast benchmarks per engine, its dry-run
estimate under the $2 cap, and the decision that flips the default engine (quality per accepted-minute cost)."""

from __future__ import annotations

import dataclasses
import uuid

from psycopg.types.json import Jsonb

from manyworlds import bench, breadth, voice_bakeoff
from manyworlds.creator import voice_lab

from test_review import BIBLE
from voice_helpers import cast_meta, finished_cast


def _paused(policy):
    return dataclasses.replace(policy, body={**policy.body, "operations": {**policy.body["operations"], "mode": "paused"}})


def _maren(conn):
    handle = f"c{uuid.uuid4().hex[:6]}"
    conn.execute("INSERT INTO creators (handle, bible, anchors) VALUES (%s, %s, %s)",
                 (handle, Jsonb({**BIBLE, "appearance": {"apparent_age": 27, "gender_presentation": "woman"}}),
                  Jsonb({"reference_images": [{"uri": f"s3://b/r{i}"} for i in range(3)]})))
    return handle


def test_the_plan_casts_each_engine_for_two_voices_within_two_dollars(conn, policy):
    handle = _maren(conn)
    out = breadth.run(conn, _paused(policy), "voicebake", creator=handle, dry_run=True)
    assert not out["skipped"], out["skipped"]
    items = {i["id"]: i for i in out["items"]}
    assert {i.split("_")[-1] for i in items} == {"indextts", "qwen3", "voxcpm"}
    assert {i.split("_")[1] for i in items} == {"maren", "other"}
    for i in items.values():
        (job,) = i["estimate"]["jobs"]
        assert job["kind"] == "voice.cast" and job["source"].startswith("policy (voice.cast:")
    assert [m["family"] for m in out["machines"]] == ["tts"]           # one machine for the whole bake-off
    plan = breadth.test_policy(policy).get("breadth", "plans", "voicebake")
    assert plan["max_usd"] <= 2.0 and out["estimated_usd"] <= plan["max_usd"]


def test_a_bakeoff_cast_carries_the_bakeoff_lines_and_is_never_a_creator_cast(conn, policy):
    handle = _maren(conn)
    cid = conn.execute("SELECT id FROM creators WHERE handle = %s", (handle,)).fetchone()["id"]
    b = bench.enqueue(conn, policy, "voice.cast", 1, batch="test-x", extra={"engine": "indextts", "creator_id": str(cid)})
    p = conn.execute("SELECT payload FROM jobs WHERE batch = %s", (b,)).fetchone()["payload"]
    assert p["purpose"] == "bakeoff" and p["fit"] == {"gender": "female", "age": 27.0}
    assert any(s["text"].startswith("Two am. I take it back.") for s in p["script"])
    assert p["n"] == policy.get("voice", "bakeoff", "candidates") and p["bakeoff"]["voice"] == "creator"
    other = bench.enqueue(conn, policy, "voice.cast", 1, batch="test-y", extra={
        "engine": "qwen3", "creator_id": str(cid), "description": "A calm man", "fit": {"gender": "male", "age": 34},
        "archetype": "educator", "label": "other"})
    q = conn.execute("SELECT payload FROM jobs WHERE batch = %s", (other,)).fetchone()["payload"]
    assert q["description"] == "A calm man" and q["fit"]["gender"] == "male"
    assert q["pace_wps"] == policy.get("voice", "judge", "pace_wps", "educator")


def test_the_decision_flips_the_default_engine(conn, policy):
    cid = conn.execute("INSERT INTO creators (handle, bible) VALUES (%s, %s) RETURNING id",
                       (f"c{uuid.uuid4().hex[:6]}", Jsonb(BIBLE))).fetchone()["id"]
    run = "vb1"
    batch = breadth.batch_name(breadth.test_policy(policy), run)
    fit = {"gender": "female", "age": 27.0}
    for engine, mos, usd in (("voxcpm", 3.4, 0.06), ("indextts", 4.3, 0.06), ("qwen3", 3.9, 0.06)):
        job = conn.execute(
            "INSERT INTO jobs (kind, status, creator_id, batch, payload) VALUES ('voice.cast', 'done', %s, %s, %s) "
            "RETURNING id", (cid, f"{batch}-cast_maren_{engine}",
                             Jsonb({"engine": engine, "purpose": "bakeoff", "fit": fit, "pace_wps": [3.6, 4.4],
                                    "bakeoff": {"engine": engine, "voice": "creator"}}))).fetchone()["id"]
        finished_cast(conn, cid, [cast_meta(0, engine=engine, mos=mos), cast_meta(1, engine=engine, mos=mos - 0.2)],
                      job_id=job)
        conn.execute("INSERT INTO cost_entries (job_id, category, provider, quantity, unit, usd) "
                     "VALUES (%s, 'gpu', 'runpod', 100, 'gpu_second', %s)", (job, usd))
    assert voice_lab.default_engine(conn, policy) == "voxcpm"
    out = voice_bakeoff.decide(conn, policy, run)
    assert out["decided"] and out["winner"] == "indextts"
    e = out["engines"]["indextts"]
    assert e["accepted_minutes"] > 0 and e["usd"] == 0.06 and e["licence_ok"] and e["gender_ok"]
    assert voice_lab.default_engine(conn, policy) == "indextts"
    d = conn.execute("SELECT agent, rationale FROM decisions WHERE kind = 'voice_engine'").fetchone()
    assert d["agent"] == "voice_bakeoff" and d["rationale"].startswith("indextts")


def test_no_decision_while_a_cast_is_running(conn, policy):
    batch = breadth.batch_name(breadth.test_policy(policy), "vb2")
    conn.execute("INSERT INTO jobs (kind, status, batch, payload) VALUES ('voice.cast', 'running', %s, %s)",
                 (f"{batch}-x", Jsonb({"engine": "indextts", "purpose": "bakeoff"})))
    out = voice_bakeoff.decide(conn, policy, "vb2")
    assert not out["decided"] and "running" in out["reason"]
    assert conn.execute("SELECT count(*) n FROM decisions WHERE kind = 'voice_engine'").fetchone()["n"] == 0
