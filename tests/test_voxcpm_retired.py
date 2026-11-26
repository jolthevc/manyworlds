"""VoxCPM2 is retired (owner 2026-11-26: "the voxcpm2 sounds bad, do away with that model entirely"): no code path can
choose it, enqueue it or speak with it. Qwen3-TTS is the incumbent, IndexTTS-2.5 the challenger, and the bake-off
compares only those two. A creator whose stored voice VoxCPM2 made speaks with Kokoro until she is cast again."""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import pytest
from psycopg.types.json import Jsonb

from manyworlds import breadth, policy_schema, voices
from manyworlds.creator import voice_lab
from manyworlds.worker import runner, smoke
from manyworlds.worker.handlers import tts_engines as te
from manyworlds.worker.runner import NonRetryable

STORED = {"voice": {"engine": "voxcpm", "ref_uri": "s3://b/voice/04.wav", "ref_text": "Okay so this happened.",
                    "embedding": [1.0, 0.0, 0.0], "voice": "bf_emma"}}
ROOT = Path(__file__).resolve().parents[1]


def test_the_policy_has_only_the_two_engines_with_qwen3_the_incumbent(policy):
    assert set(policy.get("voice", "engines")) == {"qwen3", "indextts"}
    assert policy.get("voice", "engine") == "qwen3"
    assert policy.get("voice", "engine_gate", "until_bakeoff") == "qwen3"
    kinds = policy.get("job_kinds")
    assert "voice.design" not in kinds and "voice.speak" not in kinds
    assert all(policy.family(k) == "tts" for k in kinds if k.startswith("voice."))
    assert policy.get("worker_images", "voice", default=None) is None       # the family only VoxCPM2 used
    for key in ("model", "cfg_value", "inference_timesteps", "optimize", "kinds", "clone", "deliveries_spoken",
                "prompt_tail_silence_s", "delivery_bank", "candidates"):
        assert policy.get("voice", key, default=None) is None, key


def test_the_bakeoff_compares_only_qwen3_and_indextts(policy):
    plan = breadth.test_policy(policy).get("breadth", "plans", "voicebake")
    engines = {i["bench"]["extra"]["engine"] for i in plan["items"]}
    assert engines == {"qwen3", "indextts"}


def test_a_stored_voxcpm_voice_speaks_with_kokoro_until_recast():
    assert voices.engine(STORED) == "kokoro" and not voices.designed(STORED)
    assert voices.needs_recast(STORED)
    assert not voices.needs_recast({"voice": {"engine": "qwen3", "ref_uri": "s3://b/v.wav", "ref_text": "x"}})
    assert "voxcpm" not in voices.DESIGNED_ENGINES


def test_nothing_on_the_control_plane_will_choose_or_speak_with_it(conn, policy):
    with pytest.raises(ValueError, match="retired"):
        voice_lab.speak_kind(policy, "voxcpm")
    with pytest.raises(ValueError, match="retired"):
        voice_lab.speak_payload(policy, STORED["voice"], [{"text": "Hi."}])
    with pytest.raises(ValueError, match="retired"):
        voice_lab.cast_payload(policy, description="a warm woman", engine="voxcpm", seed=1)
    cid = conn.execute("INSERT INTO creators (handle, bible, anchors) VALUES (%s, %s, %s) RETURNING id",
                       (f"c{uuid.uuid4().hex[:6]}", Jsonb({"appearance": {"apparent_age": 27,
                                                                          "gender_presentation": "woman"}}),
                        Jsonb(STORED))).fetchone()["id"]
    with pytest.raises(ValueError, match="retired"):
        voice_lab.cast(conn, policy, cid, engine="voxcpm")
    assert conn.execute("SELECT count(*) n FROM jobs").fetchone()["n"] == 0
    assert voice_lab.default_engine(conn, policy) == "qwen3"
    status = voice_lab.cast_status(conn, policy, cid)
    assert status["needs_recast"] is True and status["speaks_with"] == "kokoro"


def test_production_never_enqueues_a_voice_job_for_a_stored_voxcpm_voice(conn, policy):
    from manyworlds.production import voice as pv
    cid = conn.execute("INSERT INTO creators (handle, anchors) VALUES (%s, %s) RETURNING id",
                       (f"c{uuid.uuid4().hex[:6]}", Jsonb(STORED))).fetchone()["id"]
    post = conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, 'talking_video', 'planned', "
                        "'{}') RETURNING *", (cid,)).fetchone()
    assert pv._designed_voice(conn, post) is None                            # Kokoro: no voice.speak.* job at all


def test_the_worker_has_no_voxcpm_engine_or_handler():
    assert set(te.ENGINES) == {"qwen3", "indextts"} and set(te.DESIGNERS) == {"qwen3_design"}
    with pytest.raises(NonRetryable):
        te.validate_cast({"engine": "voxcpm", "designer": "qwen3_design", "description": "a warm woman",
                          "anchor_text": "Okay.", "n": 1, "script": [{"text": "hi"}], "params": {}})
    for kind in ("voice.speak", "voice.design"):
        with pytest.raises(ValueError):
            runner.handlers_for([kind])
    assert "voice" not in smoke.CHECKS and "speech" in smoke.CHECKS
    assert not (ROOT / "manyworlds" / "worker" / "handlers" / "tts_voxcpm.py").exists()


def test_the_policy_schema_refuses_a_retired_default_engine(policy):
    body = {**policy.body, "voice": {**policy.body["voice"], "engine": "voxcpm"}}
    assert any("retired" in e for e in policy_schema.errors(body))
    gate = {**policy.body["voice"]["engine_gate"], "until_bakeoff": "voxcpm"}
    body = {**policy.body, "voice": {**policy.body["voice"], "engine_gate": gate}}
    assert any("retired" in e for e in policy_schema.errors(body))
    assert policy_schema.errors(policy.body) == []


def test_voxcpm_appears_under_manyworlds_only_in_the_retirement_note():
    """The engine's name survives in one place: voices.RETIRED_ENGINES (and its note), so a stored voice is
    recognised as retired. Prose may say VoxCPM2; nothing names the engine `voxcpm` anywhere else."""
    allowed = {ROOT / "manyworlds" / "voices.py"}
    found = [str(p.relative_to(ROOT)) for p in (ROOT / "manyworlds").rglob("*.py")
             if p not in allowed and re.search(r"\bvoxcpm\b", p.read_text(encoding="utf-8"))]
    assert found == []
    for name in ("default.yaml", "test_overlay.yaml"):
        assert not re.search(r"\bvoxcpm\b", (ROOT / "policy" / name).read_text(encoding="utf-8")), name
    docker = (ROOT / "docker" / "worker-talkcomfy.Dockerfile").read_text(encoding="utf-8")
    assert not re.search(r"voxcpm==|VOXCPM", docker)                        # not installed any more
