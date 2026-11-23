"""`mw voice design|candidates|pick`: the owner designs a creator's voice, listens to the candidates, and picks one
(automatically by default)."""

from __future__ import annotations

import json

import pytest

from manyworlds import cli, operations
from manyworlds.creator import voice_lab

from test_voice_lab import FAR, OTHER, anchors, creator, finished_design


def _args(*argv):
    return cli.build_parser().parse_args(list(argv))


def _handle(conn, cid):
    return conn.execute("SELECT handle FROM creators WHERE id = %s", (cid,)).fetchone()["handle"]


def test_cli_parses_the_voice_commands():
    a = _args("voice", "design", "maren", "--pools", "runpod_secure_l40s_vol,runpod_secure_l40s")
    assert (a.cmd, a.voice_cmd, a.creator, a.pools) == ("voice", "design", "maren",
                                                       "runpod_secure_l40s_vol,runpod_secure_l40s")
    assert _args("voice", "candidates", "maren").voice_cmd == "candidates"
    a = _args("voice", "pick", "maren", "--candidate", "2")
    assert a.voice_cmd == "pick" and a.candidate == 2
    assert _args("voice", "pick", "maren").candidate is None


def test_design_queues_an_urgent_test_job_pinned_to_the_pools(conn, policy, capsys):
    cid = creator(conn)
    handle = _handle(conn, cid)
    assert cli._voice(_args("voice", "design", handle, "--pools", "runpod_secure_l40s_vol"), conn, policy) == 0
    out = json.loads(capsys.readouterr().out)
    job = conn.execute("SELECT kind, batch, priority, pools FROM jobs WHERE id = %s", (out["job"],)).fetchone()
    assert job["kind"] == "voice.design" and job["batch"] == f"test-voice-{handle}"
    assert job["priority"] == policy.get("autoscale", "batch", "urgent_priority")
    assert job["pools"] == ["runpod_secure_l40s_vol"]
    # a test batch: it runs while operations are paused
    assert any(job["batch"].startswith(p.rstrip("%")) for p in operations.test_batch_patterns(
        type(policy)(body={**policy.body, "operations": {**policy.body["operations"], "mode": "paused"}},
                     version=policy.version)))


def test_candidates_list_scores_links_and_which_would_be_picked(conn, policy):
    cid = creator(conn)
    creator(conn, {"voice": {"engine": "voxcpm", "ref_uri": "s3://b/o.wav", "ref_text": "x", "embedding": OTHER}})
    job = finished_design(conn, cid, [{"wer": 0.0, "embedding": FAR, "pitch_var": 2.0, "seconds": 14.0},
                                      {"wer": 0.0, "embedding": OTHER, "pitch_var": 9.0},     # sounds like the other
                                      {"wer": 0.5, "embedding": FAR, "pitch_var": 3.0}])      # says it badly
    out = cli.voice_candidates(conn, policy, cid)
    assert out["job"] == str(job) and out["status"] == "done"
    by = {c["index"]: c for c in out["candidates"]}
    assert [by[i]["qualifies"] for i in range(3)] == [True, False, False]
    assert by[1]["distance"] == pytest.approx(0.0) and by[0]["distance"] == pytest.approx(1.0)
    assert by[0]["url"] == "s3://b/voice/00.wav" and by[0]["seconds"] == 14.0   # no store: the uri itself
    assert not any(c["current"] for c in out["candidates"])
    voice_lab.pick(conn, policy, cid, job)
    assert [c["index"] for c in cli.voice_candidates(conn, policy, cid)["candidates"] if c["current"]] == [0]


def test_candidates_without_a_design_say_what_to_do(conn, policy):
    out = cli.voice_candidates(conn, policy, creator(conn))
    assert out["job"] is None and "mw voice design" in out["note"]


def test_pick_takes_the_automatic_choice_from_the_latest_finished_design(conn, policy, capsys):
    cid = creator(conn)
    finished_design(conn, cid, [{"wer": 0.0, "embedding": FAR, "pitch_var": 2.0}])
    assert cli._voice(_args("voice", "pick", _handle(conn, cid)), conn, policy) == 0
    assert json.loads(capsys.readouterr().out)["picked"]["index"] == 0
    assert anchors(conn, cid)["voice"]["engine"] == "voxcpm"


def test_pick_by_hand_takes_the_owners_candidate(conn, policy):
    cid = creator(conn)
    finished_design(conn, cid, [{"wer": 0.0, "embedding": FAR, "pitch_var": 2.0},
                                {"wer": 0.05, "embedding": FAR, "pitch_var": 1.0}])
    assert cli._voice(_args("voice", "pick", _handle(conn, cid), "--candidate", "1"), conn, policy) == 0
    assert anchors(conn, cid)["voice"]["ref_uri"].endswith("01.wav")
    with pytest.raises(SystemExit):   # no such candidate
        cli._voice(_args("voice", "pick", _handle(conn, cid), "--candidate", "7"), conn, policy)


def test_pick_without_a_finished_design_refuses(conn, policy):
    cid = creator(conn)
    finished_design(conn, cid, [], status="running")
    with pytest.raises(SystemExit):
        cli._voice(_args("voice", "pick", _handle(conn, cid)), conn, policy)
