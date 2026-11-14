"""The showrunner writes a creator's next batch as ideas: from memory, recent scores and what has
been learned, exploring on purpose; every idea goes to the evaluator before it can become a post."""

from __future__ import annotations

from psycopg.types.json import Jsonb

from manyworlds import queue
from manyworlds.brain import evidence, memory, overrides
from manyworlds.brain.agents import showrunner

from brain_helpers import StubLLM, creator, make_ctx

START = "2026-11-20"


def idea(day=0, fmt="photo", scenes=("dinner on a terrace in Venice",), caption="first night, best pasta of my life",
         **kw):
    return {"day": day, "platform": "instagram", "format": fmt, "scenes": list(scenes), "caption": caption,
            "hashtags": [], "canon": "she has been to Venice", "attributes": {"setting": "Venice terrace", "mood": "warm"},
            "reasoning": "travel posts add aspiration", "exploratory": False, **kw}


class PlanLLM(StubLLM):
    def __init__(self, ideas, conn=None):
        super().__init__(conn=conn)
        self.ideas = ideas

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema)
        return schema.model_validate({"rationale": "a warm, varied few days", "ideas": self.ideas})


def _maren(conn):
    cid = creator(conn)
    conn.execute("UPDATE creators SET bible = %s, anchors = %s WHERE id = %s",
                 (Jsonb({"display_name": "Maren", "home_and_places": ["a top-floor flat above a bakery"]}),
                  Jsonb({"reference_images": [{"uri": "s3://b/r0"}]}), cid))
    memory.seed_from_profile(conn, cid)
    return cid


def _run(conn, policy, cid, llm):
    job = queue.enqueue(conn, "brain.showrunner", {"creator_id": str(cid), "start": START}, creator_id=cid)
    ctx = make_ctx(conn, overrides.effective_policy(conn, policy), job)   # as the runtime does
    ctx.llm = llm
    return showrunner.Showrunner().run(ctx), job


def test_a_batch_of_ideas_is_written_and_each_sent_for_evaluation(conn, policy):
    cid = _maren(conn)
    out, job = _run(conn, policy, cid, PlanLLM([idea(0), idea(1, "story", caption="gerald says hi")], conn=conn))
    rows = conn.execute("SELECT id, batch, status, format, body, attributes FROM ideas ORDER BY created_at").fetchall()
    assert len(rows) == 2 and {r["status"] for r in rows} == {"proposed"}
    assert rows[0]["batch"] == f"{cid}:{START}" and rows[0]["body"]["reasoning"] == "travel posts add aspiration"
    assert rows[0]["attributes"]["setting"] == "Venice terrace" and rows[1]["format"] == "story"
    evals = conn.execute("SELECT payload, parent_job_id FROM jobs WHERE kind = 'brain.evaluator'").fetchall()
    assert sorted(e["payload"]["idea_id"] for e in evals) == sorted(str(r["id"]) for r in rows)
    assert all(e["parent_job_id"] == job for e in evals) and out["ideas"] == 2


def test_invalid_ideas_are_dropped_with_reasons(conn, policy):
    cid = _maren(conn)
    out, _ = _run(conn, policy, cid, PlanLLM([idea(0), idea(1, "talking_head"), idea(2, platform="myspace"),
                                              idea(2, caption="Embrace the journey ✨✨✨")], conn=conn))
    assert out["ideas"] == 1 and len(out["dropped"]) == 3
    assert conn.execute("SELECT count(*) n FROM ideas").fetchone()["n"] == 1


def test_the_prompt_carries_memory_scores_findings_and_the_exploration_share(conn, policy):
    cid = _maren(conn)
    evidence.record_finding(conn, scope="creator", creator_id=cid, statement="steam shots get saved",
                            evidence="simulated", grade="tentative")
    overrides.propose(conn, policy, "brain.exploration_share", 0.45, author="strategist", reason="explore more")
    llm = PlanLLM([idea(0)], conn=conn)
    _run(conn, policy, cid, llm)
    prompt = str(llm.calls[0]["content"])
    assert "top-floor flat above a bakery" in prompt                    # memory
    assert "steam shots get saved" in prompt and "SIMULATED" in prompt  # sandbox findings, labelled
    assert "45%" in prompt                                              # exploration share from the effective policy


def test_a_rerun_does_not_write_the_batch_twice(conn, policy):
    cid = _maren(conn)
    job = queue.enqueue(conn, "brain.showrunner", {"creator_id": str(cid), "start": START}, creator_id=cid)
    ctx = make_ctx(conn, policy, job)
    llm = PlanLLM([idea(0), idea(1)], conn=conn)
    ctx.llm = llm
    first = showrunner.Showrunner().run(ctx)
    again = showrunner.Showrunner().run(ctx)
    assert first == again and len(llm.calls) == 1
    assert conn.execute("SELECT count(*) n FROM ideas").fetchone()["n"] == 2
