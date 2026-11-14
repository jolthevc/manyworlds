"""Cost per accepted output counts what it took to make and review an output, and only that:
the Brain's thinking is reported on its own line, and both stay in the overall ledger."""

from __future__ import annotations

from psycopg.types.json import Jsonb

from manyworlds import ledger, queue

from brain_helpers import creator


def _image_job(conn, cid, n_assets, accepted):
    job = queue.enqueue(conn, "image.zimage", {}, creator_id=cid)
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job,))
    for i in range(n_assets):
        a = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes) "
                         "VALUES (%s, %s, 'image', %s, 'x', 1) RETURNING id", (job, cid, f"s3://b/{job}/{i}")).fetchone()["id"]
        conn.execute("INSERT INTO reviews (asset_id, gate, passed, score, details) VALUES (%s, 'identity', %s, 9, %s)",
                     (a, i < accepted, Jsonb({})))
    return job


def test_cost_per_accepted_output_includes_review_but_not_brain_spend(conn):
    cid = creator(conn)
    job = _image_job(conn, cid, n_assets=4, accepted=2)
    ledger.record(conn, category="gpu", provider="vast_3090", quantity=60, unit="gpu_second", usd=0.004, job_id=job)
    ledger.record(conn, category="llm", provider="claude-haiku-5-5", quantity=1, unit="token", usd=0.002, job_id=job)
    brain = queue.enqueue(conn, "brain.showrunner", {})
    ledger.record(conn, category="llm", provider="claude-opus-5-5", quantity=1, unit="token", usd=0.5,
                  job_id=brain, agent="showrunner")
    [row] = ledger.cost_per_output(conn)
    assert row["kind"] == "image.zimage" and row["outputs"] == 4 and row["accepted"] == 2
    assert float(row["gpu_usd"]) == 0.004 and float(row["review_usd"]) == 0.002
    assert float(row["usd_per_accepted"]) == 0.003            # (0.004 + 0.002) / 2


def test_brain_spend_has_its_own_report_and_stays_in_the_bill(conn):
    brain = queue.enqueue(conn, "brain.showrunner", {})
    ledger.record(conn, category="llm", provider="claude-opus-5-5", quantity=1, unit="token", usd=0.5,
                  job_id=brain, agent="showrunner")
    ledger.record(conn, category="llm", provider="claude-opus-5-5", quantity=1, unit="token", usd=0.25,
                  agent="evaluator")
    ledger.record(conn, category="gpu_billed", provider="vast_3090", quantity=600, unit="instance_second", usd=0.03)
    assert {r["agent"]: float(r["usd"]) for r in ledger.brain_costs(conn)} == {"showrunner": 0.5, "evaluator": 0.25}
    sources = {(r["category"], r["source"]): float(r["usd"]) for r in ledger.totals(conn)}
    assert sources == {("llm", "brain"): 0.75, ("gpu_billed", "production"): 0.03}
    assert ledger.bill_total(conn) == 0.78


def test_review_calls_are_charged_to_the_job_they_review(conn, policy, store, tmp_path):
    from manyworlds.review import common

    class Recording:
        tags = []

        def tagged(self, agent, job_id, before_call=None):
            import contextlib
            Recording.tags.append((agent, job_id))
            return contextlib.nullcontext()
    with common.charged_to(Recording(), "job-1"):
        pass
    with common.charged_to(object(), "job-2"):       # a judge stub without tagging still works
        pass
    assert Recording.tags == [(None, "job-1")]
