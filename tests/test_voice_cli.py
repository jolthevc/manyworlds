"""`mw voice cast|cast-status|quality`: a creator's voice is cast automatically. VoxCPM2's `design`, `candidates`,
`pick`, `bank` and `bank-pick` were retired with it (2026-11-26)."""

from __future__ import annotations

import json

import pytest

from manyworlds import cli, operations

from test_voice_lab import creator


def _args(*argv):
    return cli.build_parser().parse_args(list(argv))


def _handle(conn, cid):
    return conn.execute("SELECT handle FROM creators WHERE id = %s", (cid,)).fetchone()["handle"]


def test_cli_parses_the_voice_commands():
    a = _args("voice", "cast", "maren", "--engine", "indextts", "--pools", "runpod_secure_l40s_vol")
    assert (a.cmd, a.voice_cmd, a.creator, a.engine, a.pools) == ("voice", "cast", "maren", "indextts",
                                                                  "runpod_secure_l40s_vol")
    assert _args("voice", "cast-status", "maren").voice_cmd == "cast-status"
    assert _args("voice", "quality", "maren").days == 30


@pytest.mark.parametrize("retired", ["design", "candidates", "pick", "bank", "bank-pick"])
def test_the_retired_voxcpm_commands_are_gone(retired):
    with pytest.raises(SystemExit):
        _args("voice", retired, "maren")


def test_cast_queues_an_urgent_test_job_in_the_default_engine(conn, policy, capsys):
    cid = creator(conn)
    handle = _handle(conn, cid)
    assert cli._voice(_args("voice", "cast", handle, "--pools", "runpod_secure_l40s_vol"), conn, policy) == 0
    out = json.loads(capsys.readouterr().out)
    job = conn.execute("SELECT kind, batch, priority, pools, payload FROM jobs WHERE id = %s", (out["job"],)).fetchone()
    assert job["kind"] == "voice.cast" and job["batch"] == f"test-voice-{handle}"
    assert job["payload"]["engine"] == "qwen3" == policy.get("voice", "engine_gate", "until_bakeoff")
    assert job["priority"] == policy.get("autoscale", "batch", "urgent_priority")
    assert job["pools"] == ["runpod_secure_l40s_vol"]
    # a test batch: it runs while operations are paused
    assert any(job["batch"].startswith(p.rstrip("%")) for p in operations.test_batch_patterns(
        type(policy)(body={**policy.body, "operations": {**policy.body["operations"], "mode": "paused"}},
                     version=policy.version)))


def test_cast_refuses_the_retired_engine(conn, policy):
    cid = creator(conn)
    with pytest.raises(SystemExit, match="retired"):
        cli._voice(_args("voice", "cast", _handle(conn, cid), "--engine", "voxcpm"), conn, policy)
    assert conn.execute("SELECT count(*) n FROM jobs WHERE creator_id = %s", (cid,)).fetchone()["n"] == 0
