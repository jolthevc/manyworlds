"""Ideas become posts only once evaluated; the producer asks the showrunner for the next period,
once per creator and period."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds.brain import ideas, memory, planner, showrunner_loop

from brain_helpers import creator

START = date(2026, 11, 20)


def _creator(conn, refs=True):
    cid = creator(conn)
    conn.execute("UPDATE creators SET profile_version = 1, status = 'active', anchors = %s WHERE id = %s",
                 (Jsonb({"reference_images": [{"uri": "s3://b/r"}] if refs else []}), cid))
    return cid


def _idea(conn, cid, day, *, evaluated=True, canon="she owns a red bike", created=None):
    body = {"scenes": ["at the market"], "caption": "c", "day": day, "platform": "instagram", "hashtags": [],
            "canon": canon, "pillar": "food", "tone": "warm", "series": "", "script": "", "on_screen_text": "",
            "reasoning": "r"}
    iid = conn.execute("INSERT INTO ideas (creator_id, batch, concept, format, body, status, created_at) "
                       "VALUES (%s, %s, 'x', 'photo', %s, 'proposed', coalesce(%s, now())) RETURNING id",
                       (cid, f"{cid}:{START}", Jsonb(body), created)).fetchone()["id"]
    if evaluated:
        conn.execute("INSERT INTO evaluations (item_kind, idea_id, creator_id, rubric, rubric_version, evaluator_model, "
                     "prompt_version, scores, samples, role) VALUES ('idea', %s, %s, 'idea', 1, 'm', 'p', '{}', 3, 'evaluator')",
                     (iid, cid))
    return iid


def test_ideas_without_an_evaluation_are_not_scheduled(conn, policy):
    cid = _creator(conn)
    _idea(conn, cid, 0)
    _idea(conn, cid, 1, evaluated=False)
    out = ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc))
    assert out["scheduled"] == 0 and conn.execute("SELECT count(*) n FROM posts").fetchone()["n"] == 0


def test_an_evaluated_batch_becomes_planned_posts_in_slots(conn, policy):
    cid = _creator(conn)
    a, b = _idea(conn, cid, 0), _idea(conn, cid, 1)
    out = ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc))
    assert out["scheduled"] == 2
    posts = conn.execute("SELECT status, format, scheduled_for, body, profile_version FROM posts "
                         "ORDER BY scheduled_for").fetchall()
    assert [p["scheduled_for"] for p in posts] == planner.slot_times(policy, START, [0, 1])
    assert {p["status"] for p in posts} == {"planned"} and posts[0]["body"]["platform_ai_label"] is True
    assert {p["body"]["idea_id"] for p in posts} == {str(a), str(b)} and "reasoning" not in posts[0]["body"]
    assert {r["status"] for r in conn.execute("SELECT status FROM ideas").fetchall()} == {"scheduled"}


def test_after_max_wait_unevaluated_ideas_are_dropped_and_the_rest_scheduled(conn, policy):
    cid = _creator(conn)
    old = datetime.now(timezone.utc) - timedelta(hours=float(policy.get("brain", "showrunner", "max_wait_hours")) + 1)
    _idea(conn, cid, 0, created=old)
    stuck = _idea(conn, cid, 1, evaluated=False, created=old)
    assert ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc))["scheduled"] == 1
    assert conn.execute("SELECT status FROM ideas WHERE id = %s", (stuck,)).fetchone()["status"] == "dropped"


def test_scheduled_canon_enters_memory(conn, policy):
    cid = _creator(conn)
    _idea(conn, cid, 0, canon="her downstairs neighbour is called Ines")
    ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc))
    assert any("Ines" in f["text"] for f in memory.facts(conn, cid, limit=10))


def test_one_showrunner_job_per_creator_and_period(conn, policy):
    cid = _creator(conn)
    now = datetime(2026, 11, 19, 8, tzinfo=timezone.utc)
    showrunner_loop.ensure_batches(conn, policy, now=now)
    showrunner_loop.ensure_batches(conn, policy, now=now)
    jobs = conn.execute("SELECT kind, payload FROM jobs WHERE kind LIKE 'brain.%'").fetchall()
    # the period starts with the analyst, which hands it to the showrunner
    assert [(j["kind"], j["payload"]) for j in jobs] == [("brain.analyst", {"creator_id": str(cid), "then_plan": "2026-11-20"})]


def test_creators_without_references_get_no_batch(conn, policy):
    _creator(conn, refs=False)
    assert showrunner_loop.ensure_batches(conn, policy, now=datetime(2026, 11, 19, 8, tzinfo=timezone.utc)) == []


# --- final review fixes -------------------------------------------------------------------------

def _period_job(conn, cid, status, n=1):
    from manyworlds import queue
    j = queue.enqueue(conn, "brain.showrunner", {"creator_id": str(cid), "start": "2026-11-20"}, creator_id=cid,
                      idempotency_key=f"showrunner:{cid}:2026-11-20" + ("" if n == 1 else f":{n}"))
    conn.execute("UPDATE jobs SET status = %s WHERE id = %s", (status, j))
    return j


def test_a_failed_period_is_asked_for_again(conn, policy):
    cid = _creator(conn)
    _period_job(conn, cid, "failed")
    out = showrunner_loop.ensure_batches(conn, policy, now=datetime(2026, 11, 19, 8, tzinfo=timezone.utc))
    assert len(out) == 1 and conn.execute("SELECT count(*) n FROM jobs WHERE kind = 'brain.showrunner'").fetchone()["n"] == 2


def test_a_period_whose_ideas_were_all_dropped_is_asked_for_again(conn, policy):
    cid = _creator(conn)
    _period_job(conn, cid, "done")
    iid = _idea(conn, cid, 0, evaluated=False)
    conn.execute("UPDATE ideas SET status = 'dropped' WHERE id = %s", (iid,))
    assert len(showrunner_loop.ensure_batches(conn, policy, now=datetime(2026, 11, 19, 8, tzinfo=timezone.utc))) == 1


def test_a_period_still_being_evaluated_is_not_asked_for_again(conn, policy):
    cid = _creator(conn)
    _period_job(conn, cid, "done")
    _idea(conn, cid, 0, evaluated=False)
    assert showrunner_loop.ensure_batches(conn, policy, now=datetime(2026, 11, 19, 8, tzinfo=timezone.utc)) == []


def test_a_period_that_keeps_failing_stops_after_its_attempts_and_says_so(conn, policy):
    cid = _creator(conn)
    limit = int(policy.get("brain", "showrunner", "max_period_attempts"))
    for n in range(1, limit + 1):
        _period_job(conn, cid, "failed", n)
    out = showrunner_loop.ensure_batches(conn, policy, now=datetime(2026, 11, 19, 8, tzinfo=timezone.utc))
    assert out == [{"creator": conn.execute("SELECT handle FROM creators WHERE id = %s", (cid,)).fetchone()["handle"],
                    "start": "2026-11-20", "gave_up": True}]
    assert conn.execute("SELECT count(*) n FROM jobs WHERE kind = 'brain.showrunner'").fetchone()["n"] == limit


def test_a_concurrent_scheduler_cannot_double_post_a_batch(conn, policy, monkeypatch):
    cid = _creator(conn)
    _idea(conn, cid, 0)
    real = planner.slot_times
    nested = []

    def racing(*a, **k):        # another producer schedules the same batch mid-way
        if not nested:
            nested.append("racing")
            ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc))
        return real(*a, **k)
    monkeypatch.setattr(planner, "slot_times", racing)
    ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc))
    assert conn.execute("SELECT count(*) n FROM posts").fetchone()["n"] == 1


def test_a_keyed_canon_replaces_the_old_fact(conn, policy):
    cid = _creator(conn)
    memory.add(conn, cid, "place", "lives in Copenhagen", key="home")
    iid = _idea(conn, cid, 0, canon="she moved to Lisbon")
    conn.execute("UPDATE ideas SET body = body || '{\"canon_key\": \"home\"}' WHERE id = %s", (iid,))
    ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc))
    texts = [f["text"] for f in memory.facts(conn, cid, limit=10)]
    assert "she moved to Lisbon" in texts and "lives in Copenhagen" not in texts


def _stall(conn, cid, policy, *, hours_ago=0):
    limit = int(policy.get("brain", "showrunner", "max_period_attempts"))
    for n in range(1, limit + 1):
        j = _period_job(conn, cid, "failed", n)
        conn.execute("UPDATE jobs SET created_at = now() - make_interval(secs => %s) WHERE id = %s", (hours_ago * 3600, j))


def test_a_stalled_period_raises_one_event(conn, policy):
    cid = _creator(conn)
    _stall(conn, cid, policy)
    now = datetime(2026, 11, 19, 8, tzinfo=timezone.utc)
    showrunner_loop.ensure_batches(conn, policy, now=now)
    showrunner_loop.ensure_batches(conn, policy, now=now)
    ev = conn.execute("SELECT type, subject, routed_to FROM events").fetchall()
    assert len(ev) == 1 and ev[0]["type"] == "planning_stalled" and ev[0]["subject"]["creator_id"] == str(cid)
    assert ev[0]["routed_to"] == "owner"


def test_a_stalled_period_is_retried_after_the_cooldown(conn, policy):
    cid = _creator(conn)
    _stall(conn, cid, policy, hours_ago=float(policy.get("brain", "showrunner", "retry_after_hours")) + 1)
    out = showrunner_loop.ensure_batches(conn, policy, now=datetime(2026, 11, 19, 8, tzinfo=timezone.utc))
    assert len(out) == 1 and out[0].get("job") and not out[0].get("gave_up")


def test_a_failed_analysis_does_not_block_planning(conn, policy):
    from manyworlds import queue
    cid = _creator(conn)
    a = queue.enqueue(conn, "brain.analyst", {"creator_id": str(cid), "then_plan": "2026-11-20"}, creator_id=cid)
    conn.execute("UPDATE jobs SET status = 'failed' WHERE id = %s", (a,))
    showrunner_loop.ensure_batches(conn, policy, now=datetime(2026, 11, 19, 8, tzinfo=timezone.utc))
    sr = conn.execute("SELECT payload FROM jobs WHERE kind = 'brain.showrunner'").fetchall()
    assert [s["payload"] for s in sr] == [{"creator_id": str(cid), "start": "2026-11-20"}]


def test_a_period_whose_analysis_is_running_is_left_alone(conn, policy):
    from manyworlds import queue
    cid = _creator(conn)
    queue.enqueue(conn, "brain.analyst", {"creator_id": str(cid), "then_plan": "2026-11-20"}, creator_id=cid)
    assert showrunner_loop.ensure_batches(conn, policy, now=datetime(2026, 11, 19, 8, tzinfo=timezone.utc)) == []


def test_an_idea_whose_evaluation_failed_for_good_is_dropped_without_waiting(conn, policy):
    from manyworlds import queue
    cid = _creator(conn)
    _idea(conn, cid, 0)
    dead = _idea(conn, cid, 1, evaluated=False)
    j = queue.enqueue(conn, "brain.evaluator", {"idea_id": str(dead)}, creator_id=cid)
    conn.execute("UPDATE jobs SET status = 'failed' WHERE id = %s", (j,))
    out = ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc))
    assert out["scheduled"] == 1 and out["dropped"] == 1
    assert conn.execute("SELECT status FROM ideas WHERE id = %s", (dead,)).fetchone()["status"] == "dropped"
