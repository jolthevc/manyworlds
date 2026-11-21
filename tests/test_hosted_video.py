"""Story clips from a hosted video API (worker/handlers/hosted_video.py): no GPU machine, every render in the ledger,
and the hosted worker obeys the paused master switch."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from manyworlds import db, media, operations, queue
from manyworlds.worker import runner
from manyworlds.worker.handlers import hosted_video
from manyworlds.worker.runner import JobContext, NonRetryable


def _video(path: Path, seconds: int = 2) -> Path:
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc=size=160x288:rate=24:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(path)], check=True)
    return path


class FakeHost:
    """Together's video API: a job that is in progress for `polls` polls, then completed (or failed)."""

    def __init__(self, clip: Path, *, polls=2, status="completed", cost=0.1, create_error=None):
        self.clip, self.polls, self.status, self.cost, self.create_error = clip, polls, status, cost, create_error
        self.requests = []

    def request(self, method, url, key, body=None, timeout=60):
        self.requests.append((method, url, body))
        if method == "POST":
            if self.create_error:
                raise self.create_error
            return {"id": "vid-1", "status": "in_progress"}
        self.polls -= 1
        if self.polls > 0:
            return {"id": "vid-1", "status": "in_progress"}
        if self.status == "failed":
            return {"id": "vid-1", "status": "failed", "error": {"code": "bad_input", "message": "no"}, "outputs": None}
        return {"id": "vid-1", "status": "completed",
                "outputs": {"cost": self.cost, "video_url": "https://api.together.ai/shrt/x"}}

    def download(self, url, path, timeout=300):
        shutil.copy(self.clip, path)
        return path


@pytest.fixture()
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "k-test")
    h = FakeHost(_video(tmp_path / "remote.mp4"))
    monkeypatch.setattr(hosted_video, "_request", h.request)
    monkeypatch.setattr(hosted_video, "_download", h.download)
    return h


def _payload(store, tmp_path, policy):
    img = tmp_path / "start.png"
    img.write_bytes(b"\x89PNG start")
    return {"image_uri": store.put_file(img, "start.png").uri, "prompt": "steam rises from the mug", "seed": 42,
            "keyframes": 4, "params": dict(policy.get("generation", "hosted_video"))}


def _ctx(store, tmp_path):
    job = queue.Job(id="j", kind="hosted.video", payload={}, attempts=1, max_attempts=3, creator_id=None, batch=None)
    work = tmp_path / "work"
    work.mkdir()
    return JobContext(job=job, workdir=work, store=store)


def test_a_clip_is_rendered_by_the_host_from_the_approved_photo_and_charged(host, store, tmp_path, policy):
    ctx = _ctx(store, tmp_path)
    res = hosted_video.generate(_payload(store, tmp_path, policy), ctx, sleep=lambda s: None)
    kinds = [o.kind for o in res.outputs]
    assert kinds == ["video"] + ["keyframe"] * 4
    assert res.outputs[-1].meta["frame"] == res.outputs[-1].meta["of_frames"] - 1        # ends on the last frame
    method, url, body = host.requests[0]
    cfg = policy.get("generation", "hosted_video")
    assert method == "POST" and url.endswith("/v2/videos") and body["model"] == cfg["model"]
    assert body["ratio"] == "9:16" and body["seconds"] == "5" and body["seed"] == 42
    assert body["media"]["frame_images"][0]["frame"] == "first" and body["media"]["frame_images"][0]["input_image"]
    assert ctx.charges == [{"provider": f"together:{cfg['model']}", "usd": 0.1, "quantity": 1, "unit": "clip"}]


def test_a_failed_render_is_retried_and_a_rejected_request_is_not(host, store, tmp_path, policy):
    host.status = "failed"
    with pytest.raises(RuntimeError, match="bad_input"):
        hosted_video.generate(_payload(store, tmp_path, policy), _ctx(store, tmp_path), sleep=lambda s: None)
    host.create_error = NonRetryable("POST -> 400: bad model")
    with pytest.raises(NonRetryable):
        hosted_video.generate(_payload(store, tmp_path, policy), _ctx(store, tmp_path), sleep=lambda s: None)


def test_the_hosted_worker_records_the_render_in_the_ledger(conn, database_url, host, store, tmp_path, policy):
    job = queue.enqueue(conn, "hosted.video", {**_payload(store, tmp_path, policy), "params": {
        **policy.get("generation", "hosted_video"), "poll_seconds": 0}}, batch="bench-hosted")
    w = runner.start(["hosted.video"], pool="hosted", policy=policy, store=store,
                     connect=lambda url=None, timeout=None: db.connect(database_url, timeout=timeout))
    assert w.run(max_jobs=1) == 1
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job,)).fetchone()["status"] == "done"
    row = conn.execute("SELECT category, usd FROM cost_entries WHERE category = 'hosted_api'").fetchone()
    assert row and float(row["usd"]) == pytest.approx(0.1)
    assert conn.execute("SELECT count(*) n FROM assets WHERE job_id = %s", (job,)).fetchone()["n"] == 5


def test_while_paused_workers_take_only_test_and_benchmark_batches(conn, raw_policy):
    queue.enqueue(conn, "hosted.video", {}, batch="post-123")
    bench = queue.enqueue(conn, "hosted.video", {}, batch="bench-hosted-1")
    patterns = operations.test_batch_patterns(raw_policy)      # the shipped policy is paused
    assert patterns
    job = queue.claim(conn, ["hosted.video"], "w", 60, "hosted", batches=patterns)
    assert job.id == bench
    assert queue.claim(conn, ["hosted.video"], "w", 60, "hosted", batches=patterns) is None
