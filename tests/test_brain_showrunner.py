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


# --- final review fixes -------------------------------------------------------------------------

def test_a_creator_without_memory_is_seeded_before_planning(conn, policy):
    cid = creator(conn)
    conn.execute("UPDATE creators SET bible = %s WHERE id = %s",
                 (Jsonb({"home_and_places": ["a houseboat on the Amstel"]}), cid))
    llm = PlanLLM([idea(0)], conn=conn)
    _run(conn, policy, cid, llm)
    assert [f["text"] for f in memory.facts(conn, cid, limit=10)] == ["a houseboat on the Amstel"]


def test_days_are_named_by_weekday(conn, policy):
    cid = _maren(conn)
    llm = PlanLLM([idea(0)], conn=conn)
    _run(conn, policy, cid, llm)
    assert all(d in str(llm.calls[0]["content"]) for d in ("Friday", "Saturday", "Sunday"))   # START is a Friday


def test_a_run_that_finds_its_batch_already_written_writes_nothing(conn, policy):
    # Another worker finished this job while our LLM call was running (lost lease): don't write a second batch.
    cid = _maren(conn)
    job = queue.enqueue(conn, "brain.showrunner", {"creator_id": str(cid), "start": START}, creator_id=cid)
    ctx = make_ctx(conn, policy, job)

    class Racing(PlanLLM):
        def structured(self, *a, **k):
            out = super().structured(*a, **k)
            conn.execute("INSERT INTO decisions (kind, job_id, step, output) VALUES ('content_batch', %s, 'plan', %s)",
                         (job, Jsonb({"batch": "other", "ideas": 0})))
            return out
    ctx.llm = Racing([idea(0), idea(1)], conn=conn)
    assert showrunner.Showrunner().run(ctx) == {"batch": "other", "ideas": 0}
    assert conn.execute("SELECT count(*) n FROM ideas").fetchone()["n"] == 0


def test_an_all_invalid_batch_fails_visibly(conn, policy):
    import pytest
    from manyworlds.worker.runner import NonRetryable
    cid = _maren(conn)
    with pytest.raises(NonRetryable, match="no usable ideas"):
        _run(conn, policy, cid, PlanLLM([idea(0, "talking_head")], conn=conn))


def test_the_evaluator_is_not_told_which_ideas_are_exploratory(conn, policy):
    from manyworlds.brain.agents import evaluator
    cid = _maren(conn)
    _run(conn, policy, cid, PlanLLM([idea(0, exploratory=True)], conn=conn))
    iid = conn.execute("SELECT id FROM ideas").fetchone()["id"]
    _, _, content = evaluator.idea_content(conn, iid)
    assert "exploratory" not in str(content) and "reasoning" not in str(content)


# --- stagings (owner 2026-11-19): a format plus a setting, framing, wardrobe and script conventions -----

def _with_talking(policy):
    import dataclasses
    prod = {**policy.body["production"], "formats": [*policy.body["production"]["formats"], "talking_video"]}
    return dataclasses.replace(policy, body={**policy.body, "production": prod})


def test_stagings_are_offered_for_the_formats_production_makes(conn, policy):
    cid = _maren(conn)
    llm = PlanLLM([idea(0)], conn=conn)
    _run(conn, policy, cid, llm)
    assert "press_conference" not in str(llm.calls[0]["content"])        # no talking video made yet: no stagings
    llm = PlanLLM([idea(0)], conn=conn)
    _run(conn, _with_talking(policy), cid, llm)
    prompt = str(llm.calls[0]["content"])
    assert "press_conference" in prompt and "field_reporter" in prompt and "employer" in prompt


def test_an_ideas_staging_is_kept_for_learning_and_an_unknown_one_is_cleared(conn, policy):
    cid = _maren(conn)
    policy = _with_talking(policy)
    talk = dict(fmt="talking_video", scenes=("facing the camera outside the corner shop at night",),
                script="Right, quick report from the corner shop: the samosas are back.")
    _run(conn, policy, cid, PlanLLM([idea(0, staging="field_reporter", **talk), idea(1, staging="moon_base", **talk)],
                                    conn=conn))
    rows = conn.execute("SELECT body, attributes FROM ideas ORDER BY body->>'day'").fetchall()
    assert rows[0]["body"]["staging"] == "field_reporter" and rows[0]["attributes"]["staging"] == "field_reporter"
    assert rows[1]["body"]["staging"] == "" and "staging" not in rows[1]["attributes"]


# --- flavors: conventions for the existing pipelines (photo dump, expectation vs reality...) -------

def test_flavors_are_offered_for_formats_made_now_and_kept_for_learning(conn, policy):
    cid = _maren(conn)
    llm = PlanLLM([idea(0, "carousel", scenes=("Monday: the bakery queue", "Wednesday: the river at dusk"),
                        flavor="photo_dump"),
                   idea(1, "carousel", scenes=("what I pictured: a sunny picnic", "what happened: rain"),
                        caption="expectation vs the actual weather", flavor="tier_list")], conn=conn)
    _run(conn, policy, cid, llm)
    prompt = str(llm.calls[0]["content"])
    assert "photo_dump" in prompt and "expectation_vs_reality" in prompt
    assert "tier_list" not in prompt                                  # needs text on images: not offered yet
    rows = conn.execute("SELECT body, attributes FROM ideas ORDER BY body->>'day'").fetchall()
    assert rows[0]["body"]["flavor"] == "photo_dump" and rows[0]["attributes"]["flavor"] == "photo_dump"
    assert rows[1]["body"]["flavor"] == "" and "flavor" not in rows[1]["attributes"]
