"""Rubrics are versioned files: scores under one version always mean one thing."""

from __future__ import annotations

import shutil

import pytest
import yaml

from manyworlds.brain import rubrics

NAMES = ("idea", "execution", "feed")


def test_every_rubric_has_complete_anchors():
    for name in NAMES:
        r = rubrics.load(name)
        assert r.rubric == name and r.version >= 1 and len(r.dimensions) >= 5
        for d in r.dimensions:
            assert sorted(d.anchors) == [1, 3, 5, 7, 9], (name, d.key)
            assert d.definition.strip() and all(a.strip() for a in d.anchors.values())


def test_the_idea_rubric_has_the_specs_dimensions():
    keys = [d.key for d in rubrics.load("idea").dimensions]
    assert keys == ["relatability", "entertainment", "novelty", "curiosity", "emotional_pull", "shareability",
                    "aspiration", "personality", "audience_relevance", "naturalness"]


def test_registering_twice_is_idempotent(conn):
    r = rubrics.load("idea")
    assert rubrics.register(conn, r) == r.version
    assert rubrics.register(conn, r) == r.version
    assert conn.execute("SELECT count(*) n FROM rubric_versions").fetchone()["n"] == 1


def test_editing_a_rubric_without_bumping_its_version_is_refused(conn, tmp_path):
    src = rubrics.ROOT / "idea-v1.yaml"
    shutil.copy(src, tmp_path / "idea-v1.yaml")
    rubrics.register(conn, rubrics.load("idea", root=tmp_path))
    body = yaml.safe_load((tmp_path / "idea-v1.yaml").read_text(encoding="utf-8"))
    body["dimensions"][0]["definition"] += " (edited)"
    (tmp_path / "idea-v1.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    with pytest.raises(rubrics.RubricChanged, match="idea v1"):
        rubrics.register(conn, rubrics.load("idea", root=tmp_path))


def test_the_highest_version_file_wins(tmp_path):
    for v in (1, 2):
        body = yaml.safe_load((rubrics.ROOT / "idea-v1.yaml").read_text(encoding="utf-8"))
        body["version"] = v
        (tmp_path / f"idea-v{v}.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    assert rubrics.load("idea", root=tmp_path).version == 2


def test_a_stray_draft_file_is_skipped_not_fatal(tmp_path, caplog):
    shutil.copy(rubrics.ROOT / "idea-v1.yaml", tmp_path / "idea-v1.yaml")
    (tmp_path / "idea-vdraft.yaml").write_text("not: a rubric\n", encoding="utf-8")
    assert rubrics.load("idea", root=tmp_path).version == 1
    assert "idea-vdraft.yaml" in caplog.text


def test_a_file_whose_version_disagrees_with_its_name_is_refused(tmp_path):
    body = yaml.safe_load((rubrics.ROOT / "idea-v1.yaml").read_text(encoding="utf-8"))
    body["version"] = 1
    (tmp_path / "idea-v2.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    with pytest.raises(ValueError, match="declares version 1.*v2"):
        rubrics.load("idea", root=tmp_path)


def test_rubrics_load_from_beside_mw_policy_when_it_has_them(tmp_path, monkeypatch):
    (tmp_path / "rubrics").mkdir()
    body = yaml.safe_load((rubrics.ROOT / "idea-v1.yaml").read_text(encoding="utf-8"))
    body["version"] = 7
    (tmp_path / "rubrics" / "idea-v7.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    monkeypatch.setenv("MW_POLICY", str(tmp_path / "custom.yaml"))
    assert rubrics.load("idea").version == 7
    monkeypatch.setenv("MW_POLICY", str(tmp_path / "elsewhere" / "custom.yaml"))   # no rubrics there: the repo's
    assert rubrics.policy_root() == rubrics.ROOT and rubrics.load("idea").version == 1
