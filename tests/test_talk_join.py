"""The join of a long talking video's takes (breadth run 2026-11-23 `default-0229fe`: a hybrid's join timed out on the
producer's shared CPU on every loop and the post hung in producing): each ffmpeg call's timeout is sized for its input,
the join runs as a finish.join job on the finisher while production.finish_mode is queue, a join that keeps failing
drops the post with the reason, and the breadth driver counts a post whose advance keeps raising like the producer."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from psycopg.types.json import Jsonb

from manyworlds import media, production
from manyworlds.production import talk
from manyworlds.review import video
from manyworlds.worker import runner
from manyworlds.worker.handlers import finish_video

from test_talk_long import _long_post, _segment_jobs, _talking, joined  # noqa: F401 - the fixture
from test_video_review import ClipStub, clip, rendered
from test_production import NOW


def _queued(policy):
    return dataclasses.replace(policy, body={**policy.body, "production": {**policy.body["production"],
                                                                           "finish_mode": "queue"}})


def test_the_join_timeout_grows_with_the_input(policy):
    cfg = policy.get("generation", "talk_long")
    floor, per = cfg["join_timeout_s"], cfg["join_seconds_per_input_second"]
    assert talk.join_timeout(cfg, 1.0) == floor                              # short input: the floor
    assert talk.join_timeout(cfg, 100.0) == per * 100 > floor                # long input: per second of it
    assert talk.join_timeout({"timeout": cfg}, 100.0) == per * 100           # a join plan carries it
    assert talk.join_timeout({"join_timeout_s": 600}, 100.0) == 600          # no rate: the flat floor


def test_render_join_sizes_each_ffmpeg_call_for_its_input(policy, store, tmp_path, monkeypatch):
    timeouts = []

    def fake_normalize(p, out, *, timeout, **k):
        timeouts.append(("normalize", timeout))
        out.write_bytes(p.read_bytes())
        return out

    def fake_concat(paths, out, *, timeout, **k):
        timeouts.append(("concat", timeout))
        out.write_bytes(b"".join(p.read_bytes() for p in paths))
        return out
    monkeypatch.setattr(media, "normalize", fake_normalize)
    monkeypatch.setattr(media, "concat", fake_concat)
    monkeypatch.setattr(media, "duration", lambda p, **k: 40.0)
    segs = []
    for i in range(3):
        src = tmp_path / f"s{i}.mp4"
        src.write_bytes(b"take %d" % i)
        segs.append({"id": f"v{i}", "uri": store.put_file(src, f"t/s{i}.mp4").uri})
    cfg = {"join_timeout_s": 600, "join_seconds_per_input_second": 30}
    plan = {"key": "t/joined.mp4", "segments": segs, "transitions": None, "timeout": cfg,
            "normalize": {"width": 1080, "height": 1920, "fps": 30, "lufs": -14}}
    made = talk.render_join(plan, policy, store, tmp_path / "work")
    assert timeouts == [("normalize", 1200.0)] * 3 + [("concat", 3600.0)]
    assert made["uri"] == store.uri_for("t/joined.mp4") and made["segment_seconds"] == [40.0] * 3


def _ready_takes(conn, policy, store, tmp_path):
    pid = _long_post(conn, policy, store, tmp_path)
    for j in _segment_jobs(conn, pid):
        rendered(conn, store, tmp_path, j["id"])
    video.run_pending(conn, policy, ClipStub(clip()), store, tmp_path)
    return pid


def _join_jobs(conn, pid):
    return conn.execute("SELECT id, status, payload FROM jobs WHERE post_id = %s AND kind = %s ORDER BY created_at",
                        (pid, talk.JOIN_KIND)).fetchall()


def test_with_the_finish_queued_the_finisher_joins_the_takes(conn, policy, store, tmp_path, joined):  # noqa: F811
    policy = _queued(_talking(policy))
    pid = _ready_takes(conn, policy, store, tmp_path)
    assert production.advance(conn, policy, now=NOW, store=store)["producing"] == 1
    [job] = _join_jobs(conn, pid)
    assert len(joined) == 0                                                  # nothing joined on the producer
    p = job["payload"]
    assert len(p["segments"]) == 3 and p["key"].endswith("talk-joined.mp4") and "finishing" in p["policy"]
    assert runner.handlers_for([talk.JOIN_KIND])[talk.JOIN_KIND] is finish_video.join
    out = finish_video.join(p, SimpleNamespace(store=store, workdir=tmp_path / "fin"), policy=policy)
    assert len(joined) == 1 and out.outputs == []
    conn.execute("UPDATE jobs SET status = 'done', result = %s WHERE id = %s", (Jsonb(out.meta), job["id"]))
    assert production.advance(conn, policy, now=NOW, store=store)["ready"] == 1
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    asset = conn.execute("SELECT meta FROM assets WHERE uri = %s", (body["video"]["uri"],)).fetchone()
    assert body["video"]["uri"].endswith("talk-joined.mp4") and len(asset["meta"]["segments"]) == 3


def test_a_join_that_keeps_failing_drops_the_post_with_the_reason(conn, policy, store, tmp_path, joined):  # noqa: F811
    policy = _queued(_talking(policy))
    limit = int(policy.get("production", "max_job_failures_per_post"))
    pid = _ready_takes(conn, policy, store, tmp_path)
    production.advance(conn, policy, now=NOW, store=store)
    for _ in range(limit + 1):
        last = _join_jobs(conn, pid)[-1]
        conn.execute("UPDATE jobs SET status = 'failed', error = 'TimeoutExpired: ffmpeg' WHERE id = %s", (last["id"],))
        production.advance(conn, policy, now=NOW, store=store)
    row = conn.execute("SELECT status, body FROM posts WHERE id = %s", (pid,)).fetchone()
    assert row["status"] == "dropped" and "join job failed" in row["body"]["drop_reason"]
    assert len(_join_jobs(conn, pid)) == limit + 1 and "TimeoutExpired" in row["body"]["join_error"]


def test_inline_the_producer_still_joins_at_once(conn, policy, store, tmp_path, joined):  # noqa: F811
    policy = _talking(policy)                                                # conftest: finish_mode inline
    pid = _ready_takes(conn, policy, store, tmp_path)
    assert production.advance(conn, policy, now=NOW, store=store)["ready"] == 1
    assert _join_jobs(conn, pid) == [] and len(joined) == 1


def test_the_breadth_driver_drops_a_post_whose_advance_keeps_raising(conn, policy, store, monkeypatch):
    from manyworlds import breadth
    from test_breadth import _creator, _paused
    pol = _paused(policy)
    handle = _creator(conn)
    out = breadth.run(conn, pol, "default", items=["photo_propped_pose"], creator=handle, store=store, once=True)

    def boom(*a, **k):
        raise TimeoutError("ffmpeg joining segment_00.mp4 timed out")
    monkeypatch.setattr(production, "advance_post", boom)
    limit = int(breadth.test_policy(pol).get("production", "max_advance_errors"))
    for _ in range(limit):
        breadth.run(conn, pol, "default", items=["photo_propped_pose"], creator=handle, store=store, once=True)
    row = conn.execute("SELECT status, body FROM posts WHERE body->>'breadth_run' = %s", (out["run"],)).fetchone()
    assert row["status"] == "dropped" and row["body"]["advance_errors"] == limit
    assert "timed out" in row["body"]["drop_reason"]
    assert conn.execute("SELECT 1 FROM events WHERE subject->>'post_id' = %s",
                        (str(conn.execute("SELECT id FROM posts WHERE body->>'breadth_run' = %s",
                                          (out["run"],)).fetchone()["id"]),)).fetchone()
