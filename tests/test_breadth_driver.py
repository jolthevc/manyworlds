"""The breadth driver as a durable brain job (2026-11-25): queued once, one driver step per job run, deferred between
steps with no attempt charged, done when the run is over; a deploy that kills a step hands the job back at once."""

from __future__ import annotations

import pytest

from manyworlds import breadth, cli, queue
from manyworlds.brain import runtime
from manyworlds.brain.agents import breadth_driver

from brain_helpers import StubLLM
from test_breadth import _creator, _paused


REAL_WINDOW = breadth_driver._window


@pytest.fixture()
def driver_store(monkeypatch, store):
    monkeypatch.setattr(breadth_driver, "_store", lambda: store)
    monkeypatch.setattr(breadth_driver, "_window", lambda cfg: 0.0)   # one pass per job run (no real sleeps)
    return store


def _job(conn, job_id):
    return conn.execute("SELECT * FROM jobs WHERE id = %s", (job_id,)).fetchone()


def test_queueing_a_run_makes_one_driver_job_a_paused_brain_worker_takes(conn, policy):
    pol = _paused(policy)
    handle = _creator(conn)
    out = breadth.enqueue_driver(conn, pol, "default", items=["photo_propped_pose"], creator=handle)
    assert out["queued"] and out["status"] == "queued"
    job = _job(conn, out["driver_job"])
    assert job["kind"] == breadth.DRIVER_KIND and job["payload"]["test"] is True
    assert job["batch"] == breadth.driver_batch(out["batch"]) and job["payload"]["run"] == out["run"]
    # the same run again: the open driver job, not a second one
    again = breadth.enqueue_driver(conn, pol, "default", items=["photo_propped_pose"], creator=handle)
    assert again["driver_job"] == out["driver_job"] and not again["queued"]
    assert conn.execute("SELECT count(*) n FROM jobs WHERE kind = %s", (breadth.DRIVER_KIND,)).fetchone()["n"] == 1
    # the driver's own job never blocks another run, nor is it counted as the run's work
    assert breadth.open_test_work(conn, breadth.test_policy(pol), "test-breadth-other") == {}


def test_a_step_advances_the_run_then_defers_the_job_without_charging_an_attempt(conn, policy, driver_store):
    pol = _paused(policy)
    handle = _creator(conn)
    out = breadth.enqueue_driver(conn, pol, "default", items=["photo_propped_pose"], creator=handle)
    res = runtime.tick(conn, pol, StubLLM(conn=conn), worker_id="brain-1")
    defer_s = breadth._cfg(breadth.test_policy(pol))["driver_defer_seconds"]
    assert res[0]["status"] == "deferred" and res[0]["seconds"] == defer_s
    job = _job(conn, out["driver_job"])
    assert job["status"] == "queued" and job["attempts"] == 0 and job["run_after"] is not None
    assert job["result"]["run"] == out["run"] and "posts" in job["result"]
    posts = conn.execute("SELECT status FROM posts WHERE body->>'breadth_run' = %s", (out["run"],)).fetchall()
    assert [p["status"] for p in posts] == ["producing"]
    # not claimable again until its next step is due
    assert runtime.tick(conn, pol, StubLLM(conn=conn), worker_id="brain-1") == []


def test_a_finished_run_completes_the_job(conn, policy, monkeypatch, driver_store):
    pol = _paused(policy)
    out = breadth.enqueue_driver(conn, pol, "default", items=["photo_propped_pose"], creator=_creator(conn))
    monkeypatch.setattr(breadth, "run", lambda *a, **k: {"run": out["run"], "done": True, "finished": True,
                                                          "spent_usd": 1.5})
    res = runtime.tick(conn, pol, StubLLM(conn=conn), worker_id="brain-1")
    assert res[0]["status"] == "done"
    job = _job(conn, out["driver_job"])
    assert job["status"] == "done" and job["result"]["finished"] is True and job["result"]["spent_usd"] == 1.5


def test_a_busy_run_waits_and_a_refused_run_fails_once(conn, policy, monkeypatch, driver_store):
    pol = _paused(policy)
    out = breadth.enqueue_driver(conn, pol, "default", items=["photo_propped_pose"], creator=_creator(conn))

    def busy(*a, **k):
        raise breadth.DriverBusy("another driver is running r; wait for it")
    monkeypatch.setattr(breadth, "run", busy)
    assert runtime.tick(conn, pol, StubLLM(conn=conn), worker_id="brain-1")[0]["status"] == "deferred"
    assert _job(conn, out["driver_job"])["attempts"] == 0

    def refused(*a, **k):
        raise SystemExit("operations are live")
    monkeypatch.setattr(breadth, "run", refused)
    conn.execute("UPDATE jobs SET run_after = NULL WHERE id = %s", (out["driver_job"],))
    res = runtime.tick(conn, pol, StubLLM(conn=conn), worker_id="brain-1")   # SystemExit never ends the brain worker
    assert res[0]["status"] == "failed" and "live" in res[0]["error"]
    assert _job(conn, out["driver_job"])["status"] == "failed"


def test_a_new_run_beside_open_test_work_is_refused_before_anything_is_queued(conn, policy):
    pol = _paused(policy)
    queue.enqueue(conn, "talk.infinitetalk", {}, batch="test-breadth-other")
    with pytest.raises(SystemExit):
        breadth.enqueue_driver(conn, pol, "default", items=["photo_propped_pose"], creator=_creator(conn))
    assert conn.execute("SELECT count(*) n FROM jobs WHERE kind = %s", (breadth.DRIVER_KIND,)).fetchone()["n"] == 0


def test_the_cli_queues_the_driver(conn, policy, capsys):
    import json
    args = cli.build_parser().parse_args(["test", "breadth", "--queue", "--items", "photo_propped_pose",
                                          "--creator", _creator(conn)])
    assert cli._test(args, conn, _paused(policy)) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["queued"] and _job(conn, out["driver_job"])["kind"] == breadth.DRIVER_KIND


def _shared_set_run(conn, raw_policy, monkeypatch):
    """A queued maren_final run (one talking video, its shots on one shared set: the leader photo first, the
    followers waiting for it) on a test creator."""
    from test_talk_route import _maren
    monkeypatch.setenv("HF_TOKEN", "test-token")
    pol = _paused(raw_policy)
    handle = _maren(conn)
    cid = conn.execute("SELECT id FROM creators WHERE handle = %s", (handle,)).fetchone()["id"]
    out = breadth.enqueue_driver(conn, pol, "maren_final", creator=handle)
    return pol, cid, out


def _scene_jobs(conn, run):
    row = conn.execute("SELECT body FROM posts WHERE body->>'breadth_run' = %s", (run,)).fetchone()
    return row["body"]["scene_jobs"]


def test_a_photo_accepted_by_a_steps_review_starts_its_followers_in_that_step(conn, raw_policy, monkeypatch,
                                                                                driver_store):
    # 2026-11-25 (maren-final-1): the queued step advanced its post, then its review accepted the leader photo, then
    # it deferred; the eight followers waited for a next step that a long brain job (intelligence) held off. A
    # step's reviews are now followed by an advance pass in the same step.
    from test_shared_set import _accept
    pol, cid, out = _shared_set_run(conn, raw_policy, monkeypatch)
    monkeypatch.setattr(breadth, "_review", lambda *a, **k: {})
    assert runtime.tick(conn, pol, StubLLM(conn=conn), worker_id="brain-1")[0]["status"] == "deferred"
    jobs = _scene_jobs(conn, out["run"])
    assert len(jobs) > 2 and len(jobs[0]) == 1 and not any(jobs[1:])   # the leader first, its followers wait
    accepted = []

    def review(*a, **k):   # this step's review accepts the leader photo
        if not accepted:
            accepted.append(_accept(conn, jobs[0][0], cid))
        return {}
    monkeypatch.setattr(breadth, "_review", review)
    conn.execute("UPDATE jobs SET run_after = NULL WHERE id = %s", (out["driver_job"],))
    assert runtime.tick(conn, pol, StubLLM(conn=conn), worker_id="brain-1")[0]["status"] == "deferred"
    assert accepted and all(len(j) == 1 for j in _scene_jobs(conn, out["run"]))   # the followers started at once


def test_a_queued_step_drives_the_run_for_its_window_then_is_due_again_soon(conn, raw_policy, monkeypatch,
                                                                         driver_store):
    # The brain worker runs one job at a time: one claim drives several passes, loop_seconds apart, for
    # breadth.driver_step_seconds; then the job defers for breadth.driver_defer_seconds, so at its urgent priority
    # it is the next brain job claimed, ahead of a long one.
    pol, _, out = _shared_set_run(conn, raw_policy, monkeypatch)
    cfg = breadth._cfg(breadth.test_policy(pol))
    clock = {"t": 0.0}
    passes, sleeps = [], []
    real = breadth.run

    def counted(*a, **k):
        passes.append(k.get("once"))
        return real(*a, **k)

    def sleep(s):
        sleeps.append(s)
        clock["t"] += s
    monkeypatch.setattr(breadth, "run", counted)
    monkeypatch.setattr(breadth_driver, "_sleep", sleep)
    monkeypatch.setattr(breadth_driver, "_clock", lambda: clock["t"])
    monkeypatch.setattr(breadth_driver, "_window", REAL_WINDOW)
    res = runtime.tick(conn, pol, StubLLM(conn=conn), worker_id="brain-1")
    window, loop_s = float(cfg["driver_step_seconds"]), float(cfg["loop_seconds"])
    assert res[0]["status"] == "deferred" and res[0]["seconds"] == float(cfg["driver_defer_seconds"])
    assert float(cfg["driver_defer_seconds"]) < float(pol.get("brain", "runtime", "loop_seconds"))
    assert passes and all(passes) and len(passes) == int(window // loop_s) + 1   # each pass is one once=True step
    assert sleeps == [loop_s] * (len(passes) - 1)
    job = _job(conn, out["driver_job"])
    assert job["status"] == "queued" and job["attempts"] == 0 and job["result"]["passes"] == len(passes)


def test_a_single_step_still_checks_the_deadline_and_ends_the_run(conn, policy, store):
    # --once and the queued driver take one pass; a pass past the run's deadline cancels its queued work and ends it
    out = breadth.run(conn, _paused(policy), "default", items=["photo_propped_pose"], creator=_creator(conn),
                      store=store, once=True, hours=0)
    assert out["deadline_passed"] and out["finished"]
    assert conn.execute("SELECT count(*) n FROM jobs WHERE batch = %s AND status = 'queued'",
                        (out["batch"],)).fetchone()["n"] == 0
