"""Day 3: the planner writes a creator's week of posts; production turns each scene into an
image job, waits for review, picks the best confirmed image, repairs rejects and drops a post
only when repairs run out."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds import production
from manyworlds.brain import planner

from test_review import BIBLE, JudgeStub, ok

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
START = date(2026, 11, 14)


class PlanStub:
    def __init__(self, posts):
        self.posts, self.calls = posts, []

    def model_for(self, role):
        return f"stub-{role}"

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        self.calls.append(content)
        return schema(rationale="r", posts=[planner.PlannedPost(**p) for p in self.posts])


def post(day=0, fmt="photo", scenes=("on a balcony at dawn",), platform="instagram", **kw):
    return {"day": day, "platform": platform, "format": fmt, "scenes": list(scenes), "caption": "c",
            "hashtags": ["#Morning", "coffee"], "canon": "she owns a basil plant", **kw}


def creator(conn, refs=3, lora=True):
    cid = conn.execute("INSERT INTO creators (handle, bible, anchors) VALUES (%s, %s, %s) RETURNING id",
                       (f"c{uuid.uuid4().hex[:6]}", Jsonb({**BIBLE, "display_name": "M"}),
                        Jsonb({"reference_images": [{"uri": f"s3://b/r{i}"} for i in range(refs)]}))).fetchone()["id"]
    if lora:
        job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('train.lora', 'done', %s, %s) "
                           "RETURNING id", (cid, Jsonb({"trigger": "mw_m"}))).fetchone()["id"]
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) "
                     "VALUES (%s, %s, 'lora', 's3://b/m.safetensors', 'x', 1)", (job, cid))
    return cid


def test_plan_stores_valid_posts_scheduled_with_the_ai_label(conn, policy):
    cid = creator(conn)
    llm = PlanStub([post(0), post(0, "carousel", ["a", "b", "c"]), post(2, "carousel", ["only one"]),
                    post(3, "talking_head"), post(4, platform="myspace"), post(9)])
    out = planner.plan_period(conn, policy, llm, cid, start=START)
    assert out["posts"] == 3 and out["dropped_by_validation"] == 3   # 1-image carousel, bad format/platform
    rows = conn.execute("SELECT format, scheduled_for, body FROM posts WHERE creator_id = %s ORDER BY scheduled_for",
                        (cid,)).fetchall()
    hours = policy.get("planner", "post_hours_utc")
    days = policy.get("planner", "days")
    assert [r["scheduled_for"].hour for r in rows[:2]] == hours[:2]          # two posts on day 0
    assert rows[2]["scheduled_for"].date() == START + timedelta(days=days - 1)  # day 9 clamped to the last day
    assert all(r["body"]["platform_ai_label"] for r in rows)
    assert rows[0]["body"]["hashtags"] == ["morning"]          # instagram allows one by default
    assert planner.plan_period(conn, policy, llm, cid, start=START)["existing"] and len(llm.calls) == 1


def test_planner_sees_recent_posts_for_continuity(conn, policy):
    cid = creator(conn)
    planner.plan_period(conn, policy, PlanStub([post(0, canon="she hates cilantro")]), cid, start=START)
    llm = PlanStub([post(0)])
    planner.plan_period(conn, policy, llm, cid, start=START + timedelta(days=7))
    assert "she hates cilantro" in llm.calls[0]


def planned(conn, policy, cid, scenes=("on a balcony at dawn",)):
    fmt = "photo" if len(scenes) == 1 else "carousel"
    planner.plan_period(conn, policy, PlanStub([post(0, fmt, scenes)]), cid, start=START)
    return conn.execute("SELECT id FROM posts WHERE creator_id = %s", (cid,)).fetchone()["id"]


def finish(conn, store, tmp_path, policy, judge, post_id):
    """Workers finish the post's open jobs with 4 images each; the reviewer reviews them."""
    from manyworlds.review import gates
    for j in conn.execute("SELECT id FROM jobs WHERE post_id = %s AND status = 'queued'", (post_id,)).fetchall():
        conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (j["id"],))
        for i in range(4):
            p = tmp_path / f"{uuid.uuid4().hex}.png"
            p.write_bytes(b"\x89PNG " + str(i).encode())
            cid = conn.execute("SELECT creator_id FROM jobs WHERE id = %s", (j["id"],)).fetchone()["creator_id"]
            conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) VALUES (%s, %s, 'image', %s, 'x', 1)",
                         (j["id"], cid, store.put_file(p, f"p/{p.name}").uri))
        gates.review_job(conn, policy, judge(), store, j["id"], tmp_path)


def real_refs(conn, store, tmp_path, cid):
    refs = []
    for i in range(3):
        p = tmp_path / f"ref{i}.png"
        p.write_bytes(b"\x89PNG ref")
        refs.append({"uri": store.put_file(p, f"r/{cid}/{i}.png").uri})
    conn.execute("UPDATE creators SET anchors = anchors || %s WHERE id = %s", (Jsonb({"reference_images": refs}), cid))


def test_a_due_post_becomes_ready_with_the_best_confirmed_image_per_scene(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    pid = planned(conn, policy, cid, scenes=("at a bus stop", "reading on a couch"))
    assert production.advance(conn, policy, now=NOW)["started"] == 1
    jobs = conn.execute("SELECT payload FROM jobs WHERE post_id = %s ORDER BY idempotency_key", (pid,)).fetchall()
    assert len(jobs) == 2
    p = jobs[0]["payload"]
    assert p["prompt"].startswith("mw_m, ") and "thin gold wire glasses" in p["prompt"] and "bus stop" in p["prompt"]
    assert p["lora_uri"] == "s3://b/m.safetensors" and p["n"] == policy.get("production", "candidates_per_scene")
    assert production.advance(conn, policy, now=NOW)["producing"] == 1      # still generating
    # Screen: image 2 is best; confirm runs on the top two (2 and 0) and passes both.
    finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok(identity=8), ok(identity=7), ok(identity=10), ok(identity=2)]), pid)
    assert production.advance(conn, policy, now=NOW)["ready"] == 1
    row = conn.execute("SELECT status, qa_status, asset_ids, body FROM posts WHERE id = %s", (pid,)).fetchone()
    assert row["status"] == "ready" and row["qa_status"] == "passed" and len(row["asset_ids"]) == 2
    assert all(i["uri"].endswith(".png") for i in row["body"]["images"])


def test_rejected_scenes_are_regenerated_then_the_post_is_dropped(conn, policy, store, tmp_path):
    cid = creator(conn, lora=False)
    real_refs(conn, store, tmp_path, cid)
    pid = planned(conn, policy, cid)
    production.advance(conn, policy, now=NOW)
    p = conn.execute("SELECT payload FROM jobs WHERE post_id = %s", (pid,)).fetchone()["payload"]
    assert "lora_uri" not in p and p["prompt"].startswith("candid")
    max_repairs = policy.get("review", "max_repairs_per_post")
    for _ in range(max_repairs):
        finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok(identity=2)] * 4), pid)
        assert production.advance(conn, policy, now=NOW)["producing"] == 1
    seeds = {r["payload"]["seed"] for r in conn.execute("SELECT payload FROM jobs WHERE post_id = %s", (pid,))}
    assert len(seeds) == max_repairs + 1                                      # new seeds every time
    finish(conn, store, tmp_path, policy, lambda: JudgeStub([ok(identity=2)] * 4), pid)
    assert production.advance(conn, policy, now=NOW)["dropped"] == 1
    assert "no image passed review" in conn.execute("SELECT body->>'drop_reason' r FROM posts WHERE id = %s",
                                                    (pid,)).fetchone()["r"]


def test_failed_jobs_are_repaired_and_far_off_posts_wait(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    pid = planned(conn, policy, cid)
    early = NOW - timedelta(days=5)
    assert production.advance(conn, policy, now=early)["started"] == 0        # not due within lead_hours
    production.advance(conn, policy, now=NOW)
    conn.execute("UPDATE jobs SET status = 'dead' WHERE post_id = %s", (pid,))
    production.advance(conn, policy, now=NOW)
    assert conn.execute("SELECT count(*) n FROM jobs WHERE post_id = %s AND status = 'queued'", (pid,)).fetchone()["n"] == 1


def test_creators_without_reference_images_are_not_produced(conn, policy):
    cid = creator(conn, refs=0)
    planned(conn, policy, cid)
    assert production.advance(conn, policy, now=NOW)["started"] == 0


def test_salad_groups_registered_without_a_family_get_one(conn):
    from pathlib import Path

    import manyworlds
    conn.execute("INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd) "
                 "VALUES ('mw-image-salad-x', 'salad', 'salad_batch_3090', 0.1, '{image.zimage}', now(), 1)")
    text = (Path(manyworlds.__file__).parent / "migrations" / "008_posts.sql").read_text()
    conn.execute(text.split(";")[0])          # the backfill statement
    assert conn.execute("SELECT family FROM gpu_groups WHERE name = 'mw-image-salad-x'").fetchone()["family"] == "image"


def test_groups_without_job_kinds_take_their_family_from_the_name(conn):
    from pathlib import Path

    import manyworlds
    conn.execute("INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, deadline, max_usd) "
                 "VALUES ('mw-image-4090-r2', 'salad', 'salad_batch_4090', 0.1, now(), 1)")
    conn.execute((Path(manyworlds.__file__).parent / "migrations" / "009_group_family_from_name.sql").read_text())
    assert conn.execute("SELECT family FROM gpu_groups WHERE name = 'mw-image-4090-r2'").fetchone()["family"] == "image"


def test_a_job_that_lost_its_last_attempt_is_reaped_and_repaired(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    pid = planned(conn, policy, cid)
    production.advance(conn, policy, now=NOW)
    conn.execute("UPDATE jobs SET status = 'running', attempts = max_attempts, worker_id = 'w', "
                 "lease_until = now() - interval '1 minute' WHERE post_id = %s", (pid,))
    out = production.advance(conn, policy, now=NOW)
    assert out["reaped"] == 1
    assert conn.execute("SELECT count(*) n FROM jobs WHERE post_id = %s AND status = 'queued'", (pid,)).fetchone()["n"] == 1
