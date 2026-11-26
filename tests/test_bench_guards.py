"""Benchmarks refuse before queueing what would fail on a booted machine (2026-11-26: a voice.speak bench without a
reference voice failed NonRetryable after a $0.10 boot)."""

import pytest

from manyworlds import bench


def test_a_voice_bench_needs_a_reference_voice(conn, policy):
    with pytest.raises(ValueError, match="ref_uri"):
        bench.enqueue(conn, policy, "voice.speak.qwen3", 1)
    assert conn.execute("SELECT count(*) n FROM jobs").fetchone()["n"] == 0
