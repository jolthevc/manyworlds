from manyworlds import queue


def test_claim_is_exclusive_and_ordered_by_priority(conn):
    low = queue.enqueue(conn, "fake.image", {"i": 1})
    high = queue.enqueue(conn, "fake.image", {"i": 2}, priority=5)
    a = queue.claim(conn, ["fake.image"], "w1", 60)
    b = queue.claim(conn, ["fake.image"], "w2", 60)
    assert (a.id, b.id) == (high, low)
    assert queue.claim(conn, ["fake.image"], "w3", 60) is None


def test_claim_only_takes_requested_kinds(conn):
    queue.enqueue(conn, "fake.video", {})
    assert queue.claim(conn, ["fake.image"], "w1", 60) is None
    assert queue.claim(conn, ["fake.video"], "w1", 60).kind == "fake.video"


def test_expired_lease_is_reclaimed_after_preemption(conn):
    job_id = queue.enqueue(conn, "fake.image", {})
    first = queue.claim(conn, ["fake.image"], "preempted-node", 60)
    conn.execute("UPDATE jobs SET lease_until = now() - interval '1 second' WHERE id = %s", (job_id,))
    second = queue.claim(conn, ["fake.image"], "healthy-node", 60)
    assert second.id == first.id and second.attempts == 2
    # The preempted worker can no longer heartbeat or complete it.
    assert not queue.heartbeat(conn, job_id, "preempted-node", 60)
    assert not queue.complete(conn, job_id, "preempted-node", {})
    assert queue.complete(conn, job_id, "healthy-node", {"ok": True})


def test_retry_then_fail_when_attempts_run_out(conn):
    job_id = queue.enqueue(conn, "fake.image", {}, max_attempts=2)
    queue.claim(conn, ["fake.image"], "w", 60)
    assert queue.fail(conn, job_id, "w", "boom") == "queued"
    queue.claim(conn, ["fake.image"], "w", 60)
    assert queue.fail(conn, job_id, "w", "boom again") == "failed"
    assert queue.claim(conn, ["fake.image"], "w", 60) is None


def test_non_retryable_fails_immediately(conn):
    job_id = queue.enqueue(conn, "fake.image", {}, max_attempts=3)
    queue.claim(conn, ["fake.image"], "w", 60)
    assert queue.fail(conn, job_id, "w", "bad payload", retryable=False) == "failed"


def test_reap_marks_dead_on_final_lost_lease(conn):
    job_id = queue.enqueue(conn, "fake.image", {}, max_attempts=1)
    queue.claim(conn, ["fake.image"], "w", 60)
    conn.execute("UPDATE jobs SET lease_until = now() - interval '1 second' WHERE id = %s", (job_id,))
    assert queue.reap(conn) == 1
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()["status"] == "dead"


def test_idempotency_key_dedupes(conn):
    a = queue.enqueue(conn, "fake.image", {}, idempotency_key="post-1:candidate-3")
    b = queue.enqueue(conn, "fake.image", {}, idempotency_key="post-1:candidate-3")
    assert a == b
    assert conn.execute("SELECT count(*) AS n FROM jobs").fetchone()["n"] == 1
