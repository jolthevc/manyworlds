"""The finish off the producer's loop (production.finish_mode: queue): production enqueues a finish.video job with the
finish plan, waits, and records the finished file once the finisher has made it; the finisher's handler renders what
the inline finish does; the Fly adapter only starts (never creates) a stopped machine of its process group; and the
autoscaler starts the finisher only while finish jobs wait, registered (deadline, spend cap) before it starts."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from psycopg.types.json import Jsonb

from manyworlds import cli, db, production, queue
from manyworlds.deploy import autoscale
from manyworlds.deploy.providers import FlyMachines, Machine, NoCapacity
from manyworlds.worker.handlers import edit_narrated as edit
from manyworlds.worker.handlers import finish_video as handler
from manyworlds.worker.runner import JobContext, Worker

from test_edit import _edit_post, _edited, _finishing
from test_production import NOW, creator

T0 = datetime(2026, 11, 12, 9, 0, tzinfo=timezone.utc)
SCRIPT = "Thursday *soup*. More soup!"
LABELLED = {"edit": {"on_screen_text": [{"kind": "hook", "text": "Wait for the end"},
                                        {"kind": "label", "text": "Soup day"}]}}


def _mode(policy, mode: str):
    return dataclasses.replace(policy, body={**policy.body, "production": {**policy.body["production"],
                                                                           "finish_mode": mode}})


def _quiet(policy):
    """The finish as shipped (grade and captions on), without the sound mix (its room tone is seeded per post)."""
    p = _finishing(policy)
    return dataclasses.replace(p, body={**p.body, "sound": {**p.body["sound"], "enabled": False}})


@pytest.fixture()
def fake_ffmpeg(monkeypatch):
    """finishing.finish stand-in that records the text script it was given; libass here; the frame size from probe."""
    from manyworlds import captions, finishing, media, overlays
    calls = []

    def fake(video, out, *, look, grain=None, ass=None, hook_ass=None, has_voice=True, policy, timeout=None):
        script = ass or hook_ass
        calls.append(("finish", {"ass": script and Path(script).read_text(encoding="utf-8"), "look": look}))
        Path(out).write_bytes(b"finished")
        return out

    def no_burn(*a, **k):
        raise AssertionError("a second text pass")
    monkeypatch.setattr(finishing, "finish", fake)
    monkeypatch.setattr(media, "has_libass", lambda **k: True)
    monkeypatch.setattr(media, "probe", lambda p, **k: {"seconds": 3.0, "width": 1080, "height": 1920, "audio": True})
    monkeypatch.setattr(captions, "burn", no_burn)
    monkeypatch.setattr(overlays, "burn", no_burn)
    return calls


def _made(conn, store, tmp_path, src: Path, body: dict):
    """A narrated post whose (Kokoro) edit exists: the video asset with its cues, its done job, the post row."""
    cid = creator(conn)
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('image.narrated', 'done', %s, %s) "
                       "RETURNING id", (cid, Jsonb({}))).fetchone()["id"]
    meta = {"seconds": 3.0, "lead_s": edit.LEAD_S, "tail_s": edit.TAIL_S, "cues": [
        {"text": "Thursday soup.", "start_s": 0.4, "end_s": 1.4}, {"text": "More soup!", "start_s": 1.7, "end_s": 2.4}]}
    video = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                         "VALUES (%s, %s, 'video', %s, 'x', 1, %s) RETURNING id, uri",
                         (job, cid, store.put_file(src, f"v/{job}.mp4").uri, Jsonb(meta))).fetchone()
    pid = conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, 'narrated_edit', 'producing', "
                       "%s) RETURNING id", (cid, Jsonb(body))).fetchone()["id"]
    return {"id": pid, "creator_id": cid, "format": "narrated_edit", "body": body}, video


def _ctx(tmp_path, store) -> JobContext:
    work = tmp_path / "finisher"
    work.mkdir(exist_ok=True)
    return JobContext(job=None, workdir=work, store=store)


def _queued(conn, policy, store, tmp_path, src, body):
    """The queue path on its own: the producer's plan, the finisher's handler, the producer's record."""
    post, video = _made(conn, store, tmp_path, src, body)
    plan = production.finish_plan(conn, policy, post, body, video, store)
    assert json.loads(json.dumps(plan)) == plan                                    # a JSON job payload
    result = handler.finish(plan, _ctx(tmp_path, store), policy=policy)
    made = result.meta["finish"]
    assert result.outputs == [] and "path" not in made                             # the post's files, not the job's
    assert made["uri"].endswith(f"post-{post['id']}/final.mp4")
    return post, production.record_finish(conn, policy, post, body, video, made, store)


def _same(a: dict, b: dict) -> dict:
    return {k: v for k, v in a.items() if k not in ("raw", "cover")}, {k: v for k, v in b.items()
                                                                        if k not in ("raw", "cover")}


def test_the_finisher_renders_what_the_inline_finish_does(conn, policy, store, tmp_path, fake_ffmpeg):
    p = _quiet(policy)
    src = tmp_path / "raw.mp4"
    src.write_bytes(b"mp4")
    body = {"scenes": ["the kitchen"], "shots": [LABELLED], "script": SCRIPT}
    post, video = _made(conn, store, tmp_path, src, body)
    inline = production.finish_video(conn, p, post, body, video, store)
    _, queued = _queued(conn, p, store, tmp_path, src, body)
    want, got = _same(inline["meta"], queued["meta"])
    assert got == want and got["passes"] == 1 and {o["kind"] for o in got["on_screen_text"]} == {"hook", "label"}
    [(_, a), (_, b)] = fake_ffmpeg
    assert a["ass"] == b["ass"] and "Thursday" in a["ass"] and "Soup day" in a["ass"] and a["look"] == b["look"]


def test_the_finisher_really_finishes_a_clip_as_the_producer_would(conn, policy, store, tmp_path):
    """ffmpeg from the imageio-ffmpeg wheel on a 2 s clip, both ways: the same size, length, sound and record."""
    from manyworlds import media
    src = tmp_path / "edit.mp4"
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=360x640:rate=30:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=330:duration=2", "-shortest", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
    p = _quiet(policy)
    body = {"scenes": ["the kitchen"], "shots": [LABELLED], "script": SCRIPT}
    post, video = _made(conn, store, tmp_path, src, body)
    inline = production.finish_video(conn, p, post, body, video, store)
    qpost, queued = _queued(conn, p, store, tmp_path, src, body)
    want, got = _same(inline["meta"], queued["meta"])
    assert got == want
    size = p.get("finishing", "output")
    infos = [media.probe(store.get_file(row["uri"], tmp_path / f"out{i}.mp4")) for i, row in enumerate((inline, queued))]
    for info in infos:
        assert (info["width"], info["height"]) == (size["width"], size["height"]) and info["audio"]
    assert abs(infos[0]["seconds"] - infos[1]["seconds"]) < 0.1
    for row, pid in ((inline, post["id"]), (queued, qpost["id"])):   # each cover a keyframe asset of its own post
        cover = row["meta"]["cover"]
        if cover and cover["uri"].endswith("cover.jpg"):
            assert f"post-{pid}/cover.jpg" in cover["uri"]
            assert conn.execute("SELECT 1 FROM assets WHERE id = %s AND kind = 'keyframe'",
                                (cover["asset_id"],)).fetchone()


def test_the_handler_renders_under_the_producers_policy(raw_policy):
    p = _quiet(raw_policy)
    payload = {"policy": {s: p.body[s] for s in production.FINISH_SECTIONS}}
    got = handler.policy_for(payload, base=raw_policy)
    for s in production.FINISH_SECTIONS:
        assert got.body[s] == p.body[s]
    assert got.body["queue"] == raw_policy.body["queue"]                     # the rest is this image's own
    with pytest.raises(Exception, match="payload needs"):
        handler.finish({"post": {}}, None, policy=raw_policy)


def test_finishing_ships_inline_until_the_finisher_can_be_started(raw_policy, policy):
    assert production.finish_mode(raw_policy) == "inline"                     # until FLY_API_TOKEN (owner)
    assert production.finish_mode(policy) == "inline"                         # tests advance in one process
    assert raw_policy.pool("fly_finisher")["provider"] == "fly"
    assert raw_policy.get("job_kinds", production.FINISH_KIND, "pools") == ["fly_finisher"]
    from manyworlds.worker import runner
    assert runner.handlers_for([production.FINISH_KIND])[production.FINISH_KIND] is handler.finish
    assert raw_policy.family(production.FINISH_KIND) == "finish"


# --- production: the finish as a Step -------------------------------------------------------------------------------
def _run_finisher(database_url, policy, store) -> int:
    """The finisher's worker, here: one claimed finish.video job, charged to its pool."""
    w = Worker(handlers={production.FINISH_KIND: lambda payload, ctx: handler.finish(payload, ctx, policy=policy)},
               store=store, policy=policy, pool="fly_finisher", database_url=database_url)
    try:
        return w.run(max_jobs=1, idle_exit_seconds=0, poll_seconds=0)
    finally:
        w.conn.close()


def test_production_enqueues_the_finish_waits_and_records_it(conn, database_url, policy, store, tmp_path,
                                                              fake_ffmpeg):
    p, pid, cid, job = _edit_post(conn, _mode(_finishing(policy), "queue"), store, tmp_path)
    raw = _edited(conn, store, tmp_path, cid, job)
    assert production.advance(conn, p, now=NOW, store=store)["producing"] == 1
    [fj] = conn.execute("SELECT id, payload, idempotency_key, batch FROM jobs WHERE post_id = %s AND kind = %s",
                        (pid, production.FINISH_KIND)).fetchall()
    assert fj["idempotency_key"] == f"post:{pid}:finish:0" and fj["batch"] == f"post-{pid}"
    assert fj["payload"]["source_uri"] == raw and fj["payload"]["caption_words"]   # timed here, rendered there
    assert fake_ffmpeg == []                                                         # nothing encoded in the producer
    assert production.advance(conn, p, now=NOW, store=store)["producing"] == 1      # waits; no second job
    assert conn.execute("SELECT count(*) n FROM jobs WHERE kind = %s", (production.FINISH_KIND,)).fetchone()["n"] == 1
    assert _run_finisher(database_url, p, store) == 1 and len(fake_ffmpeg) == 1
    cost = conn.execute("SELECT provider, unit, usd FROM cost_entries WHERE job_id = %s AND category = 'gpu'",
                        (fj["id"],)).fetchone()
    assert cost["provider"] == "fly_finisher" and cost["unit"] == "gpu_second" and float(cost["usd"]) >= 0
    assert production.advance(conn, p, now=NOW, store=store)["ready"] == 1
    row = conn.execute("SELECT asset_ids, body FROM posts WHERE id = %s", (pid,)).fetchone()
    body = row["body"]
    assert body["video"]["uri"].endswith(f"post-{pid}/final.mp4") and body["video_raw"]["uri"] == raw
    assert str(row["asset_ids"][0]) == body["video"]["asset_id"] and body["finish_jobs"] == [str(fj["id"])]
    assert body["finish"] == {"finished": True, "look": p.get("finishing", "default_look"), "captions": "clean",
                              "placement": "lower", "passes": 1}
    asset = conn.execute("SELECT job_id, recipe FROM assets WHERE id = %s", (body["video"]["asset_id"],)).fetchone()
    assert asset["job_id"] is None and asset["recipe"]["finished_from"] == raw    # the post's asset, as inline


def test_a_failed_finish_job_is_retried_then_the_unfinished_video_ships(conn, policy, store, tmp_path):
    p, pid, cid, job = _edit_post(conn, _mode(_finishing(policy), "queue"), store, tmp_path)
    raw = _edited(conn, store, tmp_path, cid, job)
    limit = p.get("finishing", "max_failures")
    for attempt in range(limit):
        assert production.advance(conn, p, now=NOW, store=store)["producing"] == 1
        [fj] = conn.execute("SELECT id, idempotency_key FROM jobs WHERE post_id = %s AND kind = %s AND status = 'queued'",
                            (pid, production.FINISH_KIND)).fetchall()
        assert fj["idempotency_key"] == f"post:{pid}:finish:{attempt}"
        conn.execute("UPDATE jobs SET status = 'failed', error = 'ffmpeg broke' WHERE id = %s", (fj["id"],))
    assert production.advance(conn, p, now=NOW, store=store)["ready"] == 1
    body = conn.execute("SELECT body FROM posts WHERE id = %s", (pid,)).fetchone()["body"]
    assert body["video"]["uri"] == raw and "video_raw" not in body
    assert body["finish_failures"] == limit and len(body["finish_jobs"]) == limit
    assert "ffmpeg broke" in body["finish_skipped"] and body.get("finish_repairs", 0) == 0
    event = conn.execute("SELECT type, routed_to FROM events WHERE creator_id = %s", (cid,)).fetchone()
    assert event == {"type": "finish_failed", "routed_to": "owner"}


def test_inline_mode_finishes_in_the_producer_and_queues_nothing(conn, policy, store, tmp_path, fake_ffmpeg):
    p, pid, cid, job = _edit_post(conn, _mode(_finishing(policy), "inline"), store, tmp_path)
    _edited(conn, store, tmp_path, cid, job)
    assert production.advance(conn, p, now=NOW, store=store)["ready"] == 1
    assert conn.execute("SELECT 1 FROM jobs WHERE kind = %s", (production.FINISH_KIND,)).fetchone() is None
    assert len(fake_ffmpeg) == 1


# --- the Fly adapter ------------------------------------------------------------------------------------------------
class FakeFlyAPI:
    """The Machines API as /.fly/api answers it: the app's machines, metadata, start and stop."""

    def __init__(self, machines):
        self.machines = {m["id"]: m for m in machines}
        self.calls = []

    def __call__(self, method, path, body=None):
        self.calls.append((method, path, body))
        parts = path.split("/")            # ['', 'v1', 'apps', app, 'machines', id, action, key]
        assert parts[:3] == ["", "v1", "apps"] and parts[4] == "machines"
        if len(parts) == 5:
            assert method == "GET", "machines are never created here"
            return list(self.machines.values())
        m = self.machines[parts[5]]
        if len(parts) == 6:
            return m
        if parts[6] == "metadata":
            m["config"]["metadata"][parts[7]] = body["value"]
        elif parts[6] == "start":
            m["state"] = "started"
        elif parts[6] == "stop":
            m["state"] = "stopped"
        return None


def _machine(mid, group, state, **meta):
    return {"id": mid, "state": state, "config": {"metadata": {"fly_process_group": group, **meta}}}


def _spec(**kw):
    from manyworlds.deploy.providers import LaunchSpec
    return LaunchSpec(image="", gpu_name="none", cpu=0, ram_gb=0, disk_gb=0, max_usd_per_hour=0.09, **kw)


def test_the_fly_adapter_starts_only_a_stopped_finisher_and_lists_only_started_ones(tmp_path, monkeypatch):
    api = FakeFlyAPI([_machine("p1", "producer", "started"), _machine("f1", "finisher", "stopped"),
                      _machine("f2", "finisher", "started")])
    fly = FlyMachines(["finisher"], app="mw-app", transport=api)
    assert fly.configured()
    assert fly.list_live() == {"mw-fly-finisher-f2": "f2"}                 # unregistered: the enforcer stops it
    m = fly.launch("mw-finish-fly-ab12", _spec(process_group="finisher"))
    assert isinstance(m, Machine) and m.ref == "f1" and m.usd_per_hour == 0.09
    writes = [(meth, path) for meth, path, _ in api.calls if meth == "POST"]
    assert writes == [("POST", "/v1/apps/mw-app/machines/f1/metadata/mw_group"),   # named before it starts
                      ("POST", "/v1/apps/mw-app/machines/f1/start")]
    assert fly.list_live() == {"mw-finish-fly-ab12": "f1", "mw-fly-finisher-f2": "f2"}   # never the producer
    with pytest.raises(NoCapacity):                                         # no stopped finisher left: none created
        fly.launch("mw-finish-fly-cd34", _spec(process_group="finisher"))
    fly.destroy("mw-finish-fly-ab12", "f1")
    assert api.machines["f1"]["state"] == "stopped" and "mw-finish-fly-ab12" not in fly.list_live()
    # The worker on the machine reads its run, and stops its own machine.
    monkeypatch.setenv("FLY_MACHINE_ID", "f1")
    assert fly.own_group() == "mw-finish-fly-ab12"
    api.machines["f1"]["state"] = "started"
    assert fly.stop_self() == "fly" and api.machines["f1"]["state"] == "stopped"
    # Off Fly (the GitHub backstop): no socket, so not configured.
    off = FlyMachines(["finisher"], app="mw-app", socket_path=str(tmp_path / "no-socket"))
    assert not off.configured() and off.own_group() is None and off.stop_self() is None


# --- the autoscaler starts the finisher -----------------------------------------------------------------------------
class RecordingFly:
    """A Fly provider that checks the run is registered before it starts the machine."""
    name = "fly"

    def __init__(self, conn):
        self.conn, self.live, self.launched, self.destroyed = conn, {}, [], []

    def configured(self):
        return True

    def launch(self, name, spec):
        row = self.conn.execute("SELECT deadline, max_usd FROM gpu_groups WHERE name = %s", (name,)).fetchone()
        assert row is not None, "started before it was registered"
        self.launched.append((name, spec))
        self.live[name] = f"m{len(self.launched)}"
        return Machine(name, self.live[name], spec.max_usd_per_hour)

    def list_live(self):
        return dict(self.live)

    def destroy(self, name, ref):
        self.destroyed.append(name)
        self.live.pop(name, None)


def test_the_autoscaler_starts_the_finisher_only_while_finish_jobs_wait(conn, policy):
    fly = RecordingFly(conn)
    ps = {"fly": fly}
    autoscale.tick(conn, policy, ps, now=T0)
    assert fly.launched == []                                                  # nothing to finish
    queue.enqueue(conn, production.FINISH_KIND, {}, batch="post-x")
    actions = autoscale.tick(conn, policy, ps, now=T0)
    [launched] = [a for a in actions if a.get("action") == "launched"]
    name, spec = fly.launched[0]
    assert launched["pool"] == "fly_finisher" and spec.process_group == "finisher"
    g = conn.execute("SELECT * FROM gpu_groups WHERE name = %s", (name,)).fetchone()
    runs = policy.get("gpu_runs")
    assert g["deadline"] == T0 + timedelta(hours=runs["max_hours_by_family"]["finish"])
    assert float(g["max_usd"]) == runs["max_usd_by_family"]["finish"] and g["provider"] == "fly"
    assert float(g["usd_per_hour"]) == policy.pool("fly_finisher")["usd_per_hour"] and g["provider_ref"] == "m1"
    assert list(g["job_kinds"]) == [production.FINISH_KIND]
    autoscale.tick(conn, policy, ps, now=T0 + timedelta(minutes=1))
    assert len(fly.launched) == 1                                              # one finisher at a time
    # Past its deadline the enforcer stops it like any machine, and the billed time is in the ledger.
    autoscale.tick(conn, policy, ps, now=T0 + timedelta(hours=runs["max_hours_by_family"]["finish"], minutes=1))
    assert fly.destroyed == [name]
    billed = conn.execute("SELECT coalesce(sum(usd), 0) u FROM cost_entries WHERE category = 'gpu_billed' "
                          "AND provider = 'fly_finisher'").fetchone()["u"]
    assert float(billed) > 0


def test_a_finisher_run_that_did_nothing_waits_out_the_cooldown(conn, policy):
    fly = RecordingFly(conn)
    queue.enqueue(conn, production.FINISH_KIND, {}, batch="post-x")
    autoscale.tick(conn, policy, {"fly": fly}, now=T0)
    name = fly.launched[0][0]
    fly.live.clear()                                                            # it crashed at start
    autoscale.tick(conn, policy, {"fly": fly}, now=T0 + timedelta(minutes=1))   # marked gone, no job done
    actions = autoscale.tick(conn, policy, {"fly": fly}, now=T0 + timedelta(minutes=2))
    assert any(a.get("action") == "cooldown" and a["after"] == name for a in actions) and len(fly.launched) == 1


def test_finish_work_needs_no_rented_machine_or_worker_credentials(conn, policy, monkeypatch):
    monkeypatch.delenv("WORKER_DATABASE_URL")
    queue.enqueue(conn, production.FINISH_KIND, {}, batch="post-x")
    actions = autoscale.tick(conn, policy, {"fly": RecordingFly(conn)}, now=T0)
    assert not any(a.get("reason") == "worker_credentials" for a in actions)
    assert [a["pool"] for a in actions if a.get("action") == "launched"] == ["fly_finisher"]


def test_off_fly_the_backstop_keeps_the_finisher_registered_and_fails_soft(conn, policy, tmp_path):
    queue.enqueue(conn, production.FINISH_KIND, {}, batch="post-x")
    fly = RecordingFly(conn)
    autoscale.tick(conn, policy, {"fly": fly}, now=T0)
    name = fly.launched[0][0]
    off = FlyMachines(["finisher"], app="mw-app", socket_path=str(tmp_path / "no-socket"))
    actions = autoscale.tick(conn, policy, {"fly": off}, now=T0 + timedelta(hours=3), launch=False)
    assert not any(a.get("action") == "provider_error" for a in actions)
    assert [a["action"] for a in actions if a.get("group") == name] == ["stop_failed"]   # left to the Fly autoscaler
    assert conn.execute("SELECT stopped_at FROM gpu_groups WHERE name = %s", (name,)).fetchone()["stopped_at"] is None


def test_the_finisher_works_only_for_an_open_run(conn, database_url):
    connect = lambda: db.connect(database_url)  # noqa: E731
    assert not cli._open_run(None, connect) and not cli._open_run("mw-finish-fly-none", connect)
    conn.execute("INSERT INTO gpu_groups (name, provider, pool, usd_per_hour, job_kinds, deadline, max_usd) "
                 "VALUES ('mw-finish-fly-t1', 'fly', 'fly_finisher', 0.09, %s, now() + interval '1 hour', 0.25)",
                 ([production.FINISH_KIND],))
    assert cli._open_run("mw-finish-fly-t1", connect)
    conn.execute("UPDATE gpu_groups SET stopped_at = now(), stop_reason = 'idle' WHERE name = 'mw-finish-fly-t1'")
    assert not cli._open_run("mw-finish-fly-t1", connect)


def test_the_fly_adapter_uses_an_app_token_when_there_is_one(monkeypatch):
    from manyworlds.deploy import providers
    seen = {}

    def fake(token, timeout=60, host=providers.FLY_MACHINES_API):
        seen["token"] = token
        return lambda method, path, body=None: [{"id": "m1", "state": "stopped",
                                                 "config": {"metadata": {"fly_process_group": "finisher"}}}]
    monkeypatch.setattr(providers, "token_transport", fake)
    monkeypatch.setenv("FLY_API_TOKEN", "t0k")
    fly = providers.FlyMachines(["finisher"], app="mw-test", socket_path="/nonexistent")
    assert fly.configured() and fly.list_live() == {} and seen["token"] == "t0k"
