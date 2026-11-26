"""A volume that filled mid-download leaves zero-byte files the Hugging Face cache treats as complete (2026-11-26:
every VoxCPM2 load on the weights volume failed with JSONDecodeError)."""

import json

import pytest

from manyworlds.worker import hf_cache


def _cache(tmp_path, *, empty_config):
    root = hf_cache.repo_dir("openbmb/VoxCPM2", tmp_path)
    (root / "blobs").mkdir(parents=True)
    (root / "snapshots" / "abc").mkdir(parents=True)
    (root / "blobs" / "cfg").write_text("" if empty_config else '{"a": 1}')
    (root / "blobs" / "w").write_bytes(b"x" * 10)
    (root / "snapshots" / "abc" / "config.json").symlink_to(root / "blobs" / "cfg")
    (root / "snapshots" / "abc" / "model.bin").symlink_to(root / "blobs" / "w")
    return root


def test_heal_removes_empty_blobs_and_their_links_and_keeps_good_files(tmp_path):
    root = _cache(tmp_path, empty_config=True)
    removed = hf_cache.heal("openbmb/VoxCPM2", tmp_path)
    assert "blobs/cfg" in removed and not (root / "snapshots" / "abc" / "config.json").is_symlink()
    assert (root / "snapshots" / "abc" / "model.bin").exists()


def test_a_load_that_still_fails_on_a_corrupt_file_forgets_the_repo_and_retries_once(tmp_path, monkeypatch):
    root = _cache(tmp_path, empty_config=False)
    monkeypatch.setattr(hf_cache, "_hub", lambda: tmp_path)
    calls = []

    def loader():
        calls.append(root.exists())
        if len(calls) == 1:
            raise json.JSONDecodeError("Expecting value", "", 0)
        return "model"
    assert hf_cache.load("openbmb/VoxCPM2", loader) == "model" and calls == [True, False]
    with pytest.raises(RuntimeError):
        hf_cache.load("openbmb/VoxCPM2", lambda: (_ for _ in ()).throw(RuntimeError("cuda")))   # not a corrupt file


def test_a_load_holds_the_repos_lock_and_never_touches_a_blob_being_written(tmp_path, monkeypatch):
    root = _cache(tmp_path, empty_config=False)
    (root / "blobs" / "next.incomplete").write_bytes(b"")          # another download, just started
    monkeypatch.setattr(hf_cache, "_hub", lambda: tmp_path)
    lock = tmp_path / ".mw-locks" / f"{root.name}.lock"
    seen = []
    assert hf_cache.load("openbmb/VoxCPM2", lambda: seen.append(lock.exists()) or "model") == "model"
    assert seen == [True] and not lock.exists()                    # held while loading, released after
    assert (root / "blobs" / "next.incomplete").exists()
