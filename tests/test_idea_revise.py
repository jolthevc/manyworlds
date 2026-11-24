"""Critic-revise (manyworlds.brain.revise): an idea just under a feed-QA floor is rewritten once and re-evaluated
instead of dropped; far below, twice, past the batch cap, past max_wait or with the switch off, it is dropped as
before."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds.brain import ideas, revise, runtime
from manyworlds.brain.agents import evaluator, idea_reviser
from manyworlds.brain.agents.showrunner import PlannedIdea

from brain_helpers import ScoringLLM
from test_brain_schedule import START, _creator


class RevisingLLM(ScoringLLM):
    """Scores every rubric dimension `value`; answers a rewrite with a better-worded copy of the idea."""

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        if schema is idea_reviser.IdeaRevision:
            self.calls.append({"role": role, "content": content, "system": system})
            return schema(changes="said it the way she would", idea=PlannedIdea(
                day=0, platform="instagram", format="photo", scenes=["at the cafe window, rain on the glass"],
                caption="rain again. second coffee, no regrets", hashtags=[], canon="she drinks oat flat whites",
                attributes={"mood": "cosy"}))
        return super().structured(role, system=system, content=content, schema=schema)


def _floor(policy) -> float:
    return float(policy.get("brain", "qa", "min_scores")["naturalness"])


def _with_revise(policy, **changes):
    brain = policy.body["brain"]
    return dataclasses.replace(policy, body={**policy.body, "brain": {**brain, "revise": {**brain["revise"], **changes}}})


def _idea(conn, cid, day=0, *, naturalness, body=None, created=None, experiment=None):
    b = {"scenes": ["at the cafe window"], "caption": f"a quiet coffee before work {day}", "day": day,
         "platform": "instagram", "hashtags": [], "canon": "", "pillar": "", "tone": "", "series": "", "script": "",
         "on_screen_text": "", "reasoning": "r", **(body or {})}
    iid = conn.execute("INSERT INTO ideas (creator_id, batch, concept, format, body, status, experiment_id, arm, "
                       "created_at) VALUES (%s, %s, 'photo: at the cafe window', 'photo', %s, 'proposed', %s, %s, "
                       "coalesce(%s, now())) RETURNING id",
                       (cid, f"{cid}:{START}", Jsonb(b), experiment, "A" if experiment else None,
                        created)).fetchone()["id"]
    conn.execute("INSERT INTO evaluations (item_kind, idea_id, creator_id, rubric, rubric_version, evaluator_model, "
                 "prompt_version, scores, rationale, samples, role) "
                 "VALUES ('idea', %s, %s, 'idea', 1, 'm', 'p', %s, %s, 3, 'evaluator')",
                 (iid, cid, Jsonb({"naturalness": naturalness, "relatability": 6}),
                  Jsonb({"naturalness": ["the caption reads like an ad"], "_notes": ["fine moment, stiff words"]})))
    return iid


def _now():
    return datetime.now(timezone.utc)


def _jobs(conn, kind=revise.KIND):
    return conn.execute("SELECT id, status, payload FROM jobs WHERE kind = %s ORDER BY created_at", (kind,)).fetchall()


def _idea_row(conn, iid):
    return conn.execute("SELECT status, body FROM ideas WHERE id = %s", (iid,)).fetchone()


def _run_brain(conn, policy, llm):
    out = []
    for _ in range(3):
        step = runtime.tick(conn, policy, llm, worker_id="w1")
        out += step
        if not step:
            break
    return out


def test_a_near_floor_idea_is_revised_once_and_scheduled_when_the_revision_passes(conn, policy):
    cid = _creator(conn)
    floor = _floor(policy)
    orig = _idea(conn, cid, naturalness=floor - 1)
    out = ideas.schedule_ready(conn, policy, now=_now())
    assert out["revising"] == 1 and out["scheduled"] == 0
    assert conn.execute("SELECT count(*) n FROM posts").fetchone()["n"] == 0
    assert _idea_row(conn, orig)["body"]["revise_dims"] == ["naturalness"]
    # waiting on the rewrite: another pass schedules nothing and asks for nothing new
    again = ideas.schedule_ready(conn, policy, now=_now())
    assert again["scheduled"] == 0 and not again.get("revising") and len(_jobs(conn)) == 1

    llm = RevisingLLM(int(floor) + 2, conn=conn)
    ran = _run_brain(conn, policy, llm)
    assert {r["agent"] for r in ran} == {"idea_reviser", "evaluator"} and {r["status"] for r in ran} == {"done"}
    rewrite = [c for c in llm.calls if c["role"] == "idea_revise"]
    assert len(rewrite) == 1 and "the caption reads like an ad" in rewrite[0]["content"]
    assert "VOICE" in rewrite[0]["system"]   # the creator's voice rules go with the rewrite

    o = _idea_row(conn, orig)
    assert o["status"] == "revised"
    new = conn.execute("SELECT id, status, body, created_at FROM ideas WHERE body->>'revision_of' = %s",
                       (str(orig),)).fetchone()
    assert new["status"] == "proposed" and o["body"]["revised_by"] == str(new["id"])
    assert new["body"]["revision"] == 1 and new["body"]["original"]["caption"] == "a quiet coffee before work 0"
    assert "naturalness" in new["body"]["revise_reason"]
    assert new["created_at"] == conn.execute("SELECT created_at FROM ideas WHERE id = %s", (orig,)).fetchone()["created_at"]
    kinds = {d["kind"] for d in conn.execute("SELECT kind FROM decisions WHERE agent = 'idea_reviser'").fetchall()}
    assert kinds == {"idea_revision", "idea_revision_applied"}

    out = ideas.schedule_ready(conn, policy, now=_now())
    assert out["scheduled"] == 1
    post = conn.execute("SELECT caption, body FROM posts").fetchone()
    assert post["body"]["idea_id"] == str(new["id"]) and post["caption"] == "rain again. second coffee, no regrets"
    assert "original" not in post["body"] and "revision_of" not in post["body"]


def test_the_evaluator_never_sees_that_an_idea_is_a_revision(conn, policy):
    cid = _creator(conn)
    iid = _idea(conn, cid, naturalness=7, body={"revision": 1, "revision_of": "x", "original": {"caption": "OLD"},
                                                "revise_reason": "naturalness 3 below 4"})
    _, _, content = evaluator.idea_content(conn, iid)
    assert "OLD" not in content[1] and "revision" not in content[1]


def test_a_revision_that_still_fails_is_dropped_and_never_revised_again(conn, policy):
    cid = _creator(conn)
    floor = _floor(policy)
    orig = _idea(conn, cid, naturalness=floor - 1)
    assert ideas.schedule_ready(conn, policy, now=_now())["revising"] == 1
    _run_brain(conn, policy, RevisingLLM(int(floor) - 1, conn=conn))   # the rewrite scores just as low
    out = ideas.schedule_ready(conn, policy, now=_now())
    assert out["scheduled"] == 0 and not out.get("revising") and len(_jobs(conn)) == 1
    new = conn.execute("SELECT status, body FROM ideas WHERE body->>'revision_of' = %s", (str(orig),)).fetchone()
    assert new["status"] == "dropped" and "naturalness" in new["body"]["drop_reason"]
    assert _idea_row(conn, orig)["status"] == "revised"


def test_an_idea_far_below_the_floor_is_dropped_without_a_rewrite(conn, policy):
    cid = _creator(conn)
    far = _idea(conn, cid, naturalness=_floor(policy) - float(policy.get("brain", "revise", "margin")) - 1)
    out = ideas.schedule_ready(conn, policy, now=_now())
    assert not out.get("revising") and _idea_row(conn, far)["status"] == "dropped" and _jobs(conn) == []


def test_an_idea_is_never_revised_twice(conn, policy):
    cid = _creator(conn)
    floor = _floor(policy)
    revised = _idea(conn, cid, 0, naturalness=floor - 1, body={"revision": 1, "revision_of": "x"})
    # asked for once already, and that rewrite failed for good: the idea falls back to the drop
    asked = _idea(conn, cid, 1, naturalness=floor - 1, body={"revise_requested_at": "2026-11-23T00:00:00+00:00"})
    out = ideas.schedule_ready(conn, policy, now=_now())
    assert not out.get("revising") and _jobs(conn) == []
    assert _idea_row(conn, revised)["status"] == _idea_row(conn, asked)["status"] == "dropped"


def test_rewrites_per_batch_are_capped(conn, policy):
    pol = _with_revise(policy, max_per_batch=2)
    cid = _creator(conn)
    floor = _floor(pol)
    ids = [_idea(conn, cid, d, naturalness=floor - 1) for d in range(3)]
    assert ideas.schedule_ready(conn, pol, now=_now())["revising"] == 2
    asked = {j["payload"]["idea_id"] for j in _jobs(conn)}
    assert asked == {str(i) for i in ids[:2]}
    # the third gets no rewrite: once the batch's rewrites land it is dropped as before
    conn.execute("UPDATE jobs SET status = 'failed' WHERE kind = %s", (revise.KIND,))
    out = ideas.schedule_ready(conn, pol, now=_now())
    assert not out.get("revising") and len(_jobs(conn)) == 2
    assert {_idea_row(conn, i)["status"] for i in ids} == {"dropped"}


def test_only_ideas_just_under_a_floor_are_eligible(policy):
    floor, margin = _floor(policy), float(policy.get("brain", "revise", "margin"))
    near = {"naturalness": floor - margin}
    assert revise.eligible(policy, {"body": {}, "scores": near}) == ["naturalness"]
    assert revise.eligible(policy, {"body": {}, "scores": {"naturalness": floor}}) is None   # not under: kept
    assert revise.eligible(policy, {"body": {}, "scores": {"naturalness": floor - margin - 0.5}}) is None
    # experiment arms stay comparable (feed QA exempts them anyway)
    assert revise.eligible(policy, {"body": {}, "experiment_id": "e", "scores": near}) is None


def test_with_revision_off_a_near_floor_idea_is_dropped(conn, policy):
    pol = _with_revise(policy, enabled=False)
    cid = _creator(conn)
    iid = _idea(conn, cid, naturalness=_floor(pol) - 1)
    out = ideas.schedule_ready(conn, pol, now=_now())
    assert not out.get("revising") and _idea_row(conn, iid)["status"] == "dropped" and _jobs(conn) == []


def test_past_max_wait_no_rewrite_is_asked_for_and_a_pending_one_is_cancelled(conn, policy):
    cid = _creator(conn)
    floor = _floor(policy)
    old = _now() - timedelta(hours=float(policy.get("brain", "showrunner", "max_wait_hours")) + 1)
    late = _idea(conn, cid, 0, naturalness=floor - 1, created=old)
    out = ideas.schedule_ready(conn, policy, now=_now())
    assert not out.get("revising") and _idea_row(conn, late)["status"] == "dropped" and _jobs(conn) == []

    cid2 = _creator(conn)
    waiting = _idea(conn, cid2, 0, naturalness=floor - 1)
    assert ideas.schedule_ready(conn, policy, now=_now())["revising"] == 1
    later = _now() + timedelta(hours=float(policy.get("brain", "showrunner", "max_wait_hours")) + 1)
    out = ideas.schedule_ready(conn, policy, now=later)   # the rewrite never ran: the batch falls back as before
    assert out.get("cancelled") == 1 and _jobs(conn)[0]["status"] == "dead"
    assert _idea_row(conn, waiting)["status"] == "dropped"


def test_a_rewrite_that_lands_after_its_batch_moved_on_writes_nothing(conn, policy):
    cid = _creator(conn)
    orig = _idea(conn, cid, naturalness=_floor(policy) - 1)
    ideas.schedule_ready(conn, policy, now=_now())
    conn.execute("UPDATE ideas SET status = 'dropped' WHERE id = %s", (orig,))
    ran = _run_brain(conn, policy, RevisingLLM(9, conn=conn))
    assert [r["status"] for r in ran] == ["done"]
    assert conn.execute("SELECT count(*) n FROM ideas").fetchone()["n"] == 1
