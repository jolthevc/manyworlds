"""The policy schema (manyworlds.policy_schema): the shipped file validates, typos in hot sections don't."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import pytest
import yaml

from manyworlds import policy as policy_mod, policy_schema
from manyworlds.brain import overrides

ROOT = Path(__file__).resolve().parents[1]


def _shipped() -> dict:
    return yaml.safe_load((ROOT / "policy" / "default.yaml").read_text(encoding="utf-8"))


def test_shipped_policy_validates():
    policy_schema.validate(_shipped())
    policy_mod.load(ROOT / "policy" / "default.yaml", strict=True)


def test_shipped_policy_with_test_overlay_validates():
    overlay = yaml.safe_load((ROOT / "policy" / "test_overlay.yaml").read_text(encoding="utf-8")) or {}
    policy_schema.validate(policy_mod.deep_merge(_shipped(), overlay))


def test_every_hot_section_is_in_the_shipped_file():
    body = _shipped()
    assert all(s in body for s in policy_schema.SECTIONS)


@pytest.mark.parametrize("path", [
    ("queue", "lease_secnds"),
    ("gpu_runs", "daily_gpu_usdd"),
    ("autoscale", "scale", "history_dayz"),
    ("autoscale", "vast", "min_reliabilty"),
    ("review", "image", "max_jobs_per_rn"),
    ("review", "final", "enforced"),
    ("production", "candidates_per_scen"),
    ("finishing", "output", "widht"),
    ("generation", "talk", "stepz"),
    ("generation", "talk_motion", "max_motion_chian"),
])
def test_a_misspelt_key_fails(path):
    body = copy.deepcopy(_shipped())
    node = body
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = 1
    with pytest.raises(policy_schema.PolicyInvalid) as err:
        policy_schema.validate(body)
    assert ".".join(path) in str(err.value)


def test_a_missing_key_fails():
    body = copy.deepcopy(_shipped())
    del body["queue"]["lease_seconds"]
    assert any(p.startswith("queue.lease_seconds") for p in policy_schema.errors(body))


def test_a_wrong_type_fails():
    body = copy.deepcopy(_shipped())
    body["queue"]["max_attempts"] = "three"
    body["autoscale"]["worker_self_stop"] = {"on": True}
    problems = policy_schema.errors(body)
    assert any(p.startswith("queue.max_attempts") for p in problems)
    assert any(p.startswith("autoscale.worker_self_stop") for p in problems)


def test_free_form_maps_take_new_names():
    body = copy.deepcopy(_shipped())
    body["gpu_runs"]["max_hours_by_family"]["newfamily"] = 1.5
    body["generation"]["talk_profiles"]["new_profile"] = {"steps": 12, "anything": "goes"}
    body["finishing"]["looks"]["new_look"] = {"contrast": 1.0}
    body["some_other_section"] = {"anything": 1}     # sections outside the schema are not checked
    assert policy_schema.errors(body) == []


def _write(tmp_path, body) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def test_load_strict_raises_on_a_typo(tmp_path):
    body = copy.deepcopy(_shipped())
    body["review"]["image"]["max_jobs_per_rn"] = 50
    with pytest.raises(policy_schema.PolicyInvalid):
        policy_mod.load(_write(tmp_path, body), strict=True)


def test_load_at_runtime_logs_and_keeps_running(tmp_path, caplog, monkeypatch):
    monkeypatch.delenv("MW_POLICY_STRICT", raising=False)
    body = copy.deepcopy(_shipped())
    body["queue"]["lease_secondz"] = 120
    with caplog.at_level(logging.ERROR, logger="manyworlds.policy"):
        pol = policy_mod.load(_write(tmp_path, body))
    assert pol.get("queue", "lease_secondz") == 120
    assert "queue.lease_secondz" in caplog.text


def test_strict_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MW_POLICY_STRICT", "1")
    body = copy.deepcopy(_shipped())
    body["production"]["loop_secs"] = 120
    with pytest.raises(policy_schema.PolicyInvalid):
        policy_mod.load(_write(tmp_path, body))


def test_an_override_that_breaks_the_schema_is_refused():
    body = copy.deepcopy(_shipped())
    body["brain"]["authority"]["observed_sections"] = ["brain", "queue"]
    pol = policy_mod.from_dict(body)
    # refused before anything touches the database
    out = overrides.propose(None, pol, "queue.max_attempts", "lots", author="test", reason="test")
    assert out["applied"] is False and out["invalid"] is True
    assert "queue.max_attempts" in out["reason"]
    out = overrides.propose(None, pol, "queue.max_atempts", 4, author="test", reason="test")
    assert out["invalid"] is True


def test_override_schema_check_ignores_unchecked_sections():
    pol = policy_mod.from_dict(_shipped())
    assert overrides.schema_problems(pol, "brain.exploration_share", 0.4) == []
    assert overrides.schema_problems(pol, "queue.max_attempts", 4) == []
