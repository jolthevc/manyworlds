"""Third-party creator research is discontinued in the shipped policy (owner 2026-11-27): no path can resume it."""

from __future__ import annotations

import pytest

from manyworlds.research import data_policy, external


def test_the_switch_is_off_as_shipped(raw_policy):
    assert external.allowed(raw_policy) is False
    with pytest.raises(external.Discontinued):
        external.require(raw_policy, "x")


def test_every_collection_source_is_blocked(raw_policy):
    for name in ("apify", "scrapecreators", "graph"):
        assert "blocked" in data_policy.refused(raw_policy, name, "posts")
    assert data_policy.refused(raw_policy, "claude_web", "quick") is None            # web research continues


def test_no_schedule_queues_third_party_work(conn, raw_policy):
    from manyworlds.brain import discover
    from manyworlds.research import monitor
    assert monitor.ensure_due(conn, raw_policy) is None
    assert discover.ensure_due(conn, raw_policy) is None
    assert "research_monitor" not in raw_policy.get("operations", "while_paused")
    assert conn.execute("SELECT count(*) n FROM jobs").fetchone()["n"] == 0


def test_the_scraper_call_itself_refuses():
    from manyworlds.research import apify
    with pytest.raises(external.Discontinued):
        apify.run_actor("x~y", {}, token="t", call=lambda *a, **k: pytest.fail("called Apify"))


def test_agents_and_commands_cannot_restart_it(conn, raw_policy):
    from manyworlds.brain import proposals
    from manyworlds.research import gold, propositions
    got = proposals._commission_intelligence(conn, raw_policy, {"creator_id": None, "payload": {}, "id": 1, "subject": {}})
    assert got["applied"] is None and "discontinued" in got["declined"]
    with pytest.raises(external.Discontinued):
        propositions.read(conn, raw_policy, None, ["x"])
    with pytest.raises(external.Discontinued):
        gold.run_external(conn, raw_policy, None)
