"""The worker-image smoke checks that run without a GPU or ComfyUI also run here (manyworlds.worker.smoke): the
walk-and-talk motion clip is really motion-interpolated by the bundled ffmpeg, and every graph takes a negative."""

from __future__ import annotations

import pytest

from manyworlds.worker import comfy, smoke


def test_the_interpolation_check_passes_with_the_bundled_ffmpeg(tmp_path, raw_policy):
    interp = raw_policy.get("generation", "talk", "motion_interpolate")
    assert "minterpolate" in smoke._check_interpolation(tmp_path, interp)
    assert (tmp_path / "m25_interp.mp4").stat().st_size > 0


def test_the_negative_check_passes_and_catches_a_graph_that_drops_the_negative(monkeypatch):
    assert smoke._check_negatives().startswith("every graph encodes")
    real = comfy.graph_i2v

    def dropped(**kw):
        kw.pop("negative", None)
        return real(**kw)                                  # the standard list, not the job's
    monkeypatch.setattr(comfy, "graph_i2v", dropped)
    with pytest.raises(RuntimeError, match="story clip"):
        smoke._check_negatives()


def test_the_negative_check_reports_catalogue_problems():
    with pytest.raises(RuntimeError, match="doesn't match ComfyUI's nodes"):
        smoke._check_negatives({})                         # an empty node catalogue: every node is unknown
