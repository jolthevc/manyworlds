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
    jobs = conn.execute("SELECT payload FROM jobs WHERE kind = 'brain.showrunner'").fetchall()
    assert len(jobs) == 1 and jobs[0]["payload"] == {"creator_id": str(cid), "start": "2026-11-20"}


def test_creators_without_references_get_no_batch(conn, policy):
    _creator(conn, refs=False)
    assert showrunner_loop.ensure_batches(conn, policy, now=datetime(2026, 11, 19, 8, tzinfo=timezone.utc)) == []
