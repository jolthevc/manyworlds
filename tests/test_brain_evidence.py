"""Simulated results prove the machinery, never demand: real-mode reads can't see them."""

from __future__ import annotations

import re
from pathlib import Path

from manyworlds.brain import evidence

from brain_helpers import creator, make_post

BRAIN = Path(evidence.__file__).parent


def test_real_reads_never_see_simulated_findings(conn):
    cid = creator(conn, publish_mode="instagram")
    evidence.record_finding(conn, scope="creator", creator_id=cid, statement="sim", evidence="simulated",
                            grade="tentative")
    evidence.record_finding(conn, scope="creator", creator_id=cid, statement="real", evidence="observational",
                            grade="tentative")
    assert [f["statement"] for f in evidence.findings(conn, mode="real", creator_id=cid)] == ["real"]
    assert [f["statement"] for f in evidence.for_allocation(conn, creator_id=cid)] == ["real"]


def test_a_creator_that_left_the_sandbox_keeps_its_simulated_history_out_of_real_reads(conn):
    cid = creator(conn, publish_mode="simulated")
    evidence.record_finding(conn, scope="post", creator_id=cid, statement="old sim", evidence="simulated",
                            grade="speculative")
    conn.execute("UPDATE creators SET publish_mode = 'instagram' WHERE id = %s", (cid,))
    assert evidence.mode_for(conn, cid) == "real"
    assert evidence.findings(conn, mode=evidence.mode_for(conn, cid), creator_id=cid) == []


def test_sandbox_reads_label_simulated_rows(conn):
    cid = creator(conn)
    evidence.record_finding(conn, scope="post", creator_id=cid, statement="s", evidence="simulated",
                            grade="speculative")
    evidence.record_finding(conn, scope="post", creator_id=cid, statement="r", evidence="observational",
                            grade="speculative")
    labels = {f["statement"]: f.get("label") for f in evidence.findings(conn, mode="sandbox", creator_id=cid)}
    assert labels == {"s": "SIMULATED", "r": None}


def test_findings_filter_by_scope_and_grade(conn):
    cid = creator(conn)
    evidence.record_finding(conn, scope="creator", creator_id=cid, statement="a", evidence="experimental",
                            grade="proven", effect=0.3, interval=(0.1, 0.5), n=40)
    evidence.record_finding(conn, scope="post", creator_id=cid, statement="b", evidence="observational",
                            grade="speculative")
    got = evidence.findings(conn, mode="real", scope="creator", grades=("proven",))
    assert [(f["statement"], float(f["interval_low"]), f["n"]) for f in got] == [("a", 0.1, 40)]


def test_metrics_come_from_the_table_for_the_mode(conn):
    post = make_post(conn)
    conn.execute("INSERT INTO metric_snapshots (post_id, metrics, post_age_s) VALUES (%s, '{\"reach\": 10}', 3600)",
                 (post,))
    conn.execute("INSERT INTO simulated_metrics (post_id, metrics, post_age_s, model, panel_version) "
                 "VALUES (%s, '{\"reach\": 999}', 3600, 'm', 'v1')", (post,))
    assert [m["metrics"]["reach"] for m in evidence.metrics(conn, post, mode="real")] == [10]
    sandbox = evidence.metrics(conn, post, mode="sandbox")
    assert [(m["metrics"]["reach"], m["label"]) for m in sandbox] == [(999, "SIMULATED")]


def test_only_the_evidence_module_reads_findings_and_metrics(conn):
    pattern = re.compile(r"FROM\s+(findings|metric_snapshots|simulated_metrics)\b", re.IGNORECASE)
    offenders = [str(p.relative_to(BRAIN)) for p in BRAIN.rglob("*.py")
                 if p.name != "evidence.py" and pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
