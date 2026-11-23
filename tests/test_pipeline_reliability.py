"""Reliability of the pipeline's loops: one broken post never stops production, a review that keeps raising ends
in a rejection (and a repair) instead of a screen paid every loop forever, a worker charges and fails a job on a
live connection, a lost lease is not a finished job, a storage error is not a missing object, and a failed job
waits for its backoff before it is retried."""

from __future__ import annotations

import pytest

from manyworlds import policy as policy_mod, production, queue
from manyworlds.review import gates, video
from manyworlds.storage import S3Store
from manyworlds.worker.runner import Worker, handlers_for

from test_production import NOW, creator, planned, real_refs
from test_review import creator_with_job
from test_video_review import clip_job, rendered


class Boom:
    """A judge whose every call raises."""

    def __init__(self):
        self.calls = 0

    def model_for(self, role):
        return f"stub-{role}"

    def structured(self, role, **kw):
        self.calls += 1
        raise RuntimeError("judge exploded")


def _with_review(policy, section: str, **over):
    body = policy.body
    return policy_mod.from_dict({**body, "review": {**body["review"], section: {**body["review"][section], **over}}})


# 1. one broken post

def test_one_broken_post_does_not_stop_the_others_and_is_dropped_after_the_limit(conn, policy, monkeypatch):
    bad, good = planned(conn, policy, creator(conn)), planned(conn, policy, creator(conn))
    assert production.advance(conn, policy, now=NOW)["started"] == 2
    real = production.advance_post

    def poisoned(conn_, policy_, post, store=None, *, llm=None):
        if post["id"] == bad:
            raise KeyError("scene_jobs")
        return real(conn_, policy_, post, store, llm=llm)

    monkeypatch.setattr(production, "advance_post", poisoned)
    out = production.advance(conn, policy, now=NOW)
    assert out["producing"] == 1 and out["advance_errors"] == 1          # the good post still moved
    row = conn.execute("SELECT status, body FROM posts WHERE id = %s", (bad,)).fetchone()
    assert row["status"] == "producing" and row["body"]["advance_errors"] == 1
    assert "KeyError" in row["body"]["advance_error"]
    limit = int(policy.get("production", "max_advance_errors"))
    for _ in range(limit - 1):
        out = production.advance(conn, policy, now=NOW)
    assert out["dropped"] == 1
    row = conn.execute("SELECT status, qa_status, body FROM posts WHERE id = %s", (bad,)).fetchone()
    assert row["status"] == "dropped" and row["qa_status"] == "failed"
    assert row["body"]["drop_reason"].startswith(f"production failed {limit} times")
    assert conn.execute("SELECT status FROM posts WHERE id = %s", (good,)).fetchone()["status"] == "producing"
    events = conn.execute("SELECT type, routed_to FROM events WHERE type = 'post_production_failed'").fetchall()
    assert len(events) == 1 and events[0]["routed_to"] == "owner"


# 2. a review that keeps raising

def test_an_image_job_whose_review_keeps_raising_counts_as_rejected(conn, policy, store, tmp_path):
    _, job = creator_with_job(conn, store, tmp_path)
    pol = _with_review(policy, "image", review_error_spacing_seconds=0)
    limit = int(pol.get("review", "image", "max_review_errors"))
    for i in range(limit):
        out = gates.run_pending(conn, pol, Boom(), store, tmp_path)
        assert out["errors"][0].get("gave_up", False) == (i == limit - 1)
    assert gates.pending_jobs(conn, pol) == []                            # no screen paid for it again
    rows = conn.execute("SELECT r.gate, r.passed, r.details FROM reviews r JOIN assets a ON a.id = r.asset_id "
                        "WHERE a.job_id = %s", (job,)).fetchall()
    assert len(rows) == 4 and all(r["gate"] == "review" and not r["passed"] and r["details"]["stage"] == "error"
                                  for r in rows)
    assert production._scene_state(conn, [str(job)]) == ("rejected", None)   # production repairs the scene


def test_a_judge_outage_uses_up_no_job_s_tries(conn, policy, store, tmp_path):
    _, job = creator_with_job(conn, store, tmp_path)
    for _ in range(int(policy.get("review", "image", "max_review_errors")) + 1):   # loop after loop, within spacing
        gates.run_pending(conn, policy, Boom(), store, tmp_path)
    assert gates.pending_jobs(conn, policy) == [job]
    assert conn.execute("SELECT review_errors FROM jobs WHERE id = %s", (job,)).fetchone()["review_errors"] == 1


def test_a_clip_whose_review_keeps_raising_counts_as_rejected(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    job = clip_job(conn, store, tmp_path, cid)
    rendered(conn, store, tmp_path, job)
    pol = _with_review(policy, "video", review_error_spacing_seconds=0)
    for _ in range(int(pol.get("review", "video", "max_review_errors"))):
        video.run_pending(conn, pol, Boom(), store, tmp_path)
    assert video.pending_jobs(conn, pol) == []
    assert production._clip_state(conn, [str(job)]) == ("rejected", None)


# 8. the clip kinds the video gate judges come from policy

def test_the_video_gate_judges_the_kinds_policy_lists(conn, policy, store, tmp_path):
    cid = creator(conn)
    real_refs(conn, store, tmp_path, cid)
    jobs = {k: clip_job(conn, store, tmp_path, cid, kind=k)
            for k in ("comfy.wan_i2v", "talk.infinitetalk", "hosted.video", "comfy.wanXi2v", "image.zimage")}
    for j in jobs.values():
        rendered(conn, store, tmp_path, j)
    pending = set(video.pending_jobs(conn, policy))
    assert pending == {jobs["comfy.wan_i2v"], jobs["talk.infinitetalk"], jobs["hosted.video"]}   # `_` is no wildcard
    only_talk = _with_review(policy, "video", kinds=["talk.*"])
    assert video.pending_jobs(conn, only_talk) == [jobs["talk.infinitetalk"]]


# 3. a worker's failure path and a lost lease

def _worker(database_url, store, policy):
    return Worker(handlers=handlers_for(["fake.image"]), store=store, policy=policy, pool="local",
                  database_url=database_url)


def test_a_job_failing_after_the_connection_dropped_is_still_charged_and_retried(conn, database_url, store, policy):
    job_id = queue.enqueue(conn, "fake.image", {"n": 1})
    worker = _worker(database_url, store, policy)

    def handler(payload, ctx):
        pid = worker.conn.execute("SELECT pg_backend_pid() p").fetchone()["p"]
        conn.execute("SELECT pg_terminate_backend(%s, 5000)", (pid,))     # the drop, then the job fails
        raise RuntimeError("download interrupted")

    worker.handlers["fake.image"] = handler
    assert worker.run(max_jobs=1) == 1
    row = conn.execute("SELECT status, error FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert row["status"] == "queued" and "download interrupted" in row["error"]
    assert conn.execute("SELECT count(*) n FROM cost_entries WHERE job_id = %s", (job_id,)).fetchone()["n"] == 1


def test_a_lease_lost_before_completion_is_not_a_finished_job(conn, database_url, store, policy):
    job_id = queue.enqueue(conn, "fake.image", {"n": 1})
    worker = _worker(database_url, store, policy)
    inner = worker.handlers["fake.image"]

    def handler(payload, ctx):
        conn.execute("UPDATE jobs SET worker_id = 'another-worker' WHERE id = %s", (job_id,))   # reclaimed meanwhile
        return inner(payload, ctx)

    worker.handlers["fake.image"] = handler
    assert worker.run(max_jobs=1) == 1
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()["status"] == "running"
    attempt = conn.execute("SELECT outcome, error FROM attempts WHERE job_id = %s", (job_id,)).fetchone()
    assert attempt["outcome"] == "lost" and "lease lost" in attempt["error"]
    assert conn.execute("SELECT count(*) n FROM cost_entries WHERE job_id = %s", (job_id,)).fetchone()["n"] == 1


# 9. retry backoff

def test_a_failed_job_waits_for_its_backoff_before_a_retry(conn):
    job_id = queue.enqueue(conn, "fake.image", {})
    queue.claim(conn, ["fake.image"], "w1", 60)
    assert queue.fail(conn, job_id, "w1", "blip", backoff_seconds=30) == "queued"
    wait = conn.execute("SELECT extract(epoch FROM run_after - now()) s FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert 25 < float(wait["s"]) <= 30                                     # 1 attempt x 30 s
    assert queue.claim(conn, ["fake.image"], "w2", 60) is None
    conn.execute("UPDATE jobs SET run_after = now() - interval '1 second' WHERE id = %s", (job_id,))
    assert queue.claim(conn, ["fake.image"], "w2", 60).id == job_id


def test_without_a_backoff_a_failed_job_is_claimable_at_once(conn):
    job_id = queue.enqueue(conn, "fake.image", {})
    queue.claim(conn, ["fake.image"], "w1", 60)
    assert queue.fail(conn, job_id, "w1", "blip") == "queued"
    assert conn.execute("SELECT run_after FROM jobs WHERE id = %s", (job_id,)).fetchone()["run_after"] is None
    assert queue.claim(conn, ["fake.image"], "w2", 60).id == job_id


# 4. storage: only "not found" is missing

class _ClientError(Exception):
    def __init__(self, code: str, status: int):
        super().__init__(code)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class _FakeS3:
    def __init__(self, error=None):
        self.error = error

    def head_object(self, Bucket, Key):
        if self.error:
            raise self.error
        return {"ContentLength": 1}


def _s3(error=None) -> S3Store:
    s = S3Store.__new__(S3Store)   # no boto3 client: a fake one
    s.bucket, s.client = "b", _FakeS3(error)
    return s


def test_s3_exists_is_false_only_when_the_object_is_not_there():
    assert _s3().exists("k")
    assert not _s3(_ClientError("404", 404)).exists("k")
    assert not _s3(_ClientError("NoSuchKey", 404)).exists("k")
    with pytest.raises(_ClientError):
        _s3(_ClientError("403", 403)).exists("k")                         # no access is not "missing"
    with pytest.raises(ConnectionError):
        _s3(ConnectionError("reset")).exists("k")
