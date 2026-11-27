"""Live end-to-end checks catch the silent failures of 2026-11-26/27 on real-shaped data (manyworlds.verify)."""

from __future__ import annotations

from psycopg.types.json import Jsonb

from manyworlds import ledger, verify


def _job(conn, kind="comfy.wan_i2v", status="done", payload=None):
    return conn.execute("INSERT INTO jobs (kind, status, payload) VALUES (%s, %s, %s) RETURNING id",
                        (kind, status, Jsonb(payload or {}))).fetchone()["id"]


def _asset(conn, job, kind, uri, sha, meta=None):
    conn.execute("INSERT INTO assets (job_id, kind, uri, sha256, bytes, meta) VALUES (%s, %s, %s, %s, 1, %s)",
                 (job, kind, uri, sha, Jsonb(meta or {})))


def test_a_clean_system_passes_every_check(conn, policy):
    out = verify.run(conn, policy)
    assert out["failing"] == [], out


def test_each_silent_failure_is_caught(conn, policy):
    _job(conn)                                                            # done, no output at all
    src = _job(conn, kind="comfy.wan_i2v")
    _asset(conn, src, "video", "s3://b/motion.mp4", "same", {"render": {"prompt_id": "p"}})
    v2v = _job(conn, kind="talk.infinitetalk", payload={"video_uri": "s3://b/motion.mp4"})
    _asset(conn, v2v, "video", "s3://b/take.mp4", "same")                # the "render" is its input, no provenance
    conn.execute("INSERT INTO reference_creators (platform, handle) VALUES ('tiktok', 'x')")
    cid = conn.execute("SELECT id FROM reference_creators").fetchone()["id"]
    pid = conn.execute("INSERT INTO reference_posts (creator_id, platform, external_id) VALUES (%s, 'tiktok', 'p1') "
                       "RETURNING id", (cid,)).fetchone()["id"]
    conn.execute("INSERT INTO reference_annotations (post_id, schema, observed) VALUES (%s, 'iia_v2', %s)",
                 (pid, Jsonb({"coded": {"pose": "unknown", "garments": []}})))   # a read that saw nothing
    ledger.record(conn, category="llm", provider="m", quantity=1, unit="call", usd=0.5, agent="research:gold")
    out = verify.run(conn, policy)
    assert {"done_without_output", "output_is_input", "render_without_provenance", "empty_reads",
            "spend_without_output"} <= set(out["failing"]), out["failing"]
    assert out["checks"]["output_is_input"]["examples"][0]["output"] == "s3://b/take.mp4"
    assert verify.alert(conn, policy, out) == len(out["failing"])
    assert verify.alert(conn, policy, out) == 0                          # once a day per check
