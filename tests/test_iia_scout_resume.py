"""The research pilot's scout never pays twice and never eats the whole cap (production 2026-11-25: $15.64 of a $16
cap spent in scout alone; retries re-pulled 54 creators, hashtag pools took half; 0 annotations, 0 cards).
Stubbed: no Apify, no LLM, no GPU."""

from __future__ import annotations

import dataclasses
import json

import pytest

from manyworlds.brain.agents import intelligence
from manyworlds.research import pulls
from manyworlds.storage import LocalStore

from test_iia_direction import IG_POSTS, FakeApify, _agent, _ctx, _small


@pytest.fixture()
def policy(policy):
    # acquisition mechanics under test: sources approved (the shipped pauses: test_data_policy.py)
    import dataclasses
    from conftest import approve_research_sources
    pol = approve_research_sources(policy)
    intel = pol.body["intelligence"]
    agent = intel["agent"]
    # the mechanics run on two creators; the shipped floor (min_creators 3) is tested in test_review_fixes.py
    return dataclasses.replace(pol, body={**pol.body, "intelligence": {**intel, "agent": {
        **agent, "distill": {**agent["distill"], "min_creators": 1}}}})

PROFILE = "apify~instagram-profile-scraper"
POSTS = "apify~instagram-post-scraper"
HASHTAG = "apify~instagram-hashtag-scraper"


class Lost(BaseException):
    """A lost lease or a deploy killing the worker mid-scout: nothing in the agent catches it."""


def _with(policy, **agent):
    return dataclasses.replace(policy, body={**policy.body, "intelligence": {
        **policy.body["intelligence"], "agent": {**policy.body["intelligence"]["agent"], **agent}}})


def _calls(agent, actor, who=None):
    return [p for a, p in agent.fetch.calls if a == actor and (
        who is None or who in (p.get("username") or p.get("usernames") or []))]


def _scout_payload(**extra):
    return {"mode": "scout", "hypothesis": "night-out", "hashtags": ["nightout"], "platforms": ["instagram"],
            "handles": ["instagram:mira.lou", "instagram:jo.bee", "instagram:ana.k"], "creators": 4, "posts": 8,
            "usd": 5, **extra}


def test_a_retried_scout_resumes_and_pulls_no_creator_or_hashtag_twice(conn, policy, tmp_path):
    agent = _agent(LocalStore(tmp_path / "objects"))
    fake = agent.fetch
    ctx = _ctx(conn, _small(policy), _scout_payload())

    def dies_on_the_third_creator(actor, payload):
        if actor == PROFILE and payload["usernames"] == ["ana.k"]:
            raise Lost()
        return fake(actor, payload)
    agent.fetch = dies_on_the_third_creator
    with pytest.raises(Lost):
        agent.run(ctx)
    pulled = conn.execute("SELECT count(*) n FROM decisions WHERE job_id = %s AND step LIKE 'scout:instagram:%%'",
                          (ctx.job.id,)).fetchone()["n"]
    assert pulled == 2                                       # each creator recorded as its pull completed
    spent_before = conn.execute("SELECT sum(usd) u FROM cost_entries WHERE job_id = %s", (ctx.job.id,)).fetchone()["u"]

    agent.fetch = fake                                       # the job is retried on another worker
    out = agent.run(ctx)
    handles = {c["handle"] for c in out["scout"]["creators"]}
    assert {"mira.lou", "jo.bee", "ana.k", "other.creator"} <= handles
    assert len(_calls(agent, HASHTAG)) == 1                  # the pool was remembered
    for who in ("mira.lou", "jo.bee"):
        assert len(_calls(agent, PROFILE, who)) == 1 and len(_calls(agent, POSTS, who)) == 1
    spent = conn.execute("SELECT sum(usd) u FROM cost_entries WHERE job_id = %s", (ctx.job.id,)).fetchone()["u"]
    per_creator = 0.0023 + len(IG_POSTS) * 0.0023
    assert float(spent) - float(spent_before) == pytest.approx(2 * per_creator, abs=1e-6)   # only the two not pulled


def test_a_creator_pulled_recently_by_any_run_is_reused_not_pulled(conn, policy, tmp_path):
    agent = _agent(LocalStore(tmp_path / "objects"))
    first = _ctx(conn, _small(policy), _scout_payload(handles=["instagram:mira.lou"], creators=1, hashtags=[]))
    agent.run(first)
    assert len(_calls(agent, POSTS, "mira.lou")) == 1

    again = _ctx(conn, _small(policy), _scout_payload(handles=["instagram:mira.lou"], creators=1, hashtags=[]),
                 batch="test-iia-y")
    out = agent.run(again)
    assert len(_calls(agent, POSTS, "mira.lou")) == 1 and len(_calls(agent, PROFILE, "mira.lou")) == 1
    c = out["scout"]["creators"][0]
    assert c["reused"] is True and c["posts"] == len(IG_POSTS) and out["scout"]["reused"] == 1
    assert conn.execute("SELECT count(*) n FROM cost_entries WHERE job_id = %s AND provider = ANY(%s)",
                        (again.job.id, [f"apify:{PROFILE}", f"apify:{POSTS}"])).fetchone()["n"] == 0

    # older than repull_after_days: pulled again
    days = policy.get("intelligence", "agent", "repull_after_days")
    conn.execute("UPDATE reference_posts SET fetched_at = now() - make_interval(days => %s)", (int(days) + 1,))
    late = _ctx(conn, _small(policy), _scout_payload(handles=["instagram:mira.lou"], creators=1, hashtags=[]),
                batch="test-iia-z")
    agent.run(late)
    assert len(_calls(agent, POSTS, "mira.lou")) == 2


def test_a_pilots_scout_leaves_the_reserve_to_annotate_and_distill(conn, policy, tmp_path):
    agent = _agent(LocalStore(tmp_path / "objects"))
    pol = _small(policy)
    cap, reserve = 0.4, pol.get("intelligence", "agent", "reserve_share")
    assert 0 < reserve < 1
    ctx = _ctx(conn, pol, {"mode": "pilot", "hypothesis": "night-out", "hashtags": ["nightout"], "usd": cap,
                           "platforms": ["instagram"], "creators": 14, "posts": 30,
                           "handles": [f"instagram:c{i}" for i in range(14)]})
    out = agent.run(ctx)
    assert out["scout"]["held_for_reserve"] is True and out["scout"]["capped"] is False
    assert len(out["scout"]["creators"]) < 14                # it stopped early...
    scout = conn.execute("SELECT sum(usd) u FROM cost_entries WHERE job_id = %s AND agent = 'intelligence:scout'",
                         (ctx.job.id,)).fetchone()["u"]
    assert float(scout) <= cap * (1 - reserve)               # ...within its share
    assert out["annotate"]["annotated"] > 0 and "distill" in out   # and the run went on
    assert out["spent_usd"] <= cap


def test_the_candidate_pool_spends_at_most_its_budget(conn, policy):
    per = policy.get("intelligence", "agent", "items_per_hashtag")
    price = policy.get("intelligence", "agent", "usd_per_hashtag_item")["instagram"]
    seen = []

    def fetch(actor, payload):
        seen.append(payload)
        return [{**IG_POSTS[0], "id": f"{len(seen)}-{i}", "ownerUsername": f"u{i}"} for i in range(per)]
    budget = 2.5 * per * price
    pool = pulls.candidates(conn, policy, fetch, {"instagram": ["a", "b", "c", "d"]}, budget_usd=budget)
    assert len(seen) == 2 and pool
    assert float(conn.execute("SELECT sum(usd) u FROM cost_entries").fetchone()["u"]) <= budget
    assert policy.get("intelligence", "agent", "max_candidate_pool_usd") > 0


def _profile_fake(found: bool = True):
    """The profile scraper as it really answers: the profile with its latest posts embedded, or an error item."""
    class Fake(FakeApify):
        def __call__(self, actor, payload):
            if actor != PROFILE:
                return super().__call__(actor, payload)
            self.calls.append((actor, payload))
            if not found:
                return [{"error": "not_found", "errorDescription": "no such account"}]
            who = payload["usernames"][0]
            return [{"username": who, "fullName": "X", "followersCount": 42000,
                     "latestPosts": [{**p, "id": f"{who}-{p['id']}"} for p in IG_POSTS]}]
    return Fake()


@pytest.mark.parametrize("views", [False, True])
def test_instagram_posts_come_from_the_profile_item_unless_reel_views_are_needed(conn, policy, tmp_path, views):
    fetch = _profile_fake()
    r = pulls.pull_creator(conn, policy, fetch, {"platform": "instagram", "handle": "mira.lou", "role": "mid"},
                           store=None, download=None, workdir=tmp_path, posts=30, views=views)
    actors = [a for a, _ in fetch.calls]
    entries = conn.execute("SELECT provider, quantity, usd FROM cost_entries").fetchall()
    if views:
        assert POSTS in actors
    else:
        assert actors == [PROFILE] and r["posts"] == len(IG_POSTS)
        assert [(e["provider"], float(e["quantity"])) for e in entries] == [(f"apify:{PROFILE}", 1.0)]
        assert r["usd"] == pytest.approx(policy.get("intelligence", "agent", "usd_per_item")["instagram_profile"])
    assert conn.execute("SELECT count(*) n FROM reference_posts").fetchone()["n"] == len(IG_POSTS)


def test_a_profile_not_found_pulls_no_posts(conn, policy, tmp_path):
    fetch = _profile_fake(found=False)
    r = pulls.pull_creator(conn, policy, fetch, {"platform": "instagram", "handle": "gone.now"},
                           store=None, download=None, workdir=tmp_path, posts=30)
    assert [a for a, _ in fetch.calls] == [PROFILE] and r["posts"] == 0


def test_resume_annotates_and_distills_a_batchs_stored_posts_without_pulling(conn, policy, tmp_path, capsys,
                                                                              database_url, monkeypatch):
    from manyworlds import cli
    monkeypatch.setattr("manyworlds.research.external.allowed", lambda p: True)   # dormant since 2026-11-27
    agent = _agent(LocalStore(tmp_path / "objects"))
    pol = _small(policy)
    scout = _ctx(conn, pol, _scout_payload(handles=["instagram:mira.lou", "instagram:jo.bee"], creators=2,
                                           hashtags=[]), batch="test-iia-old")
    agent.run(scout)
    # the 2026-11-25 run never finished its scout step: only the posts it fetched while it ran say who it collected
    conn.execute("DELETE FROM decisions WHERE job_id = %s", (scout.job.id,))
    conn.execute("UPDATE jobs SET status = 'dead', updated_at = now() WHERE id = %s", (scout.job.id,))
    monkeypatch.setenv("DATABASE_URL", database_url)
    assert cli.main(["research", "pilot", "--resume-annotate", "test-iia-old", "--usd", "2", "--dry-run"]) == 0
    printed = capsys.readouterr().out
    dry = json.loads(printed[printed.index("{"):])
    assert dry["creators"] == 2 and dry["estimate"]["posts"] == 2 * len(IG_POSTS) and dry["estimate"]["total_usd"] > 0
    assert conn.execute("SELECT count(*) n FROM jobs WHERE payload->>'mode' = 'resume'").fetchone()["n"] == 0

    assert cli.main(["research", "pilot", "--resume-annotate", "test-iia-old", "--usd", "2"]) == 0
    job = conn.execute("SELECT id, payload, batch FROM jobs WHERE payload->>'mode' = 'resume'").fetchone()
    assert job["payload"]["test"] is True and len(job["payload"]["creator_ids"]) == 2
    from brain_helpers import make_ctx
    from test_iia_direction import IIALLM
    ctx = make_ctx(conn, pol, job["id"])
    ctx.llm = IIALLM(conn=conn)
    calls = len(agent.fetch.calls)
    out = agent.run(ctx)
    assert len(agent.fetch.calls) == calls                   # nothing pulled
    assert "scout" not in out and out["annotate"]["annotated"] == 2 * len(IG_POSTS)
    assert out["distill"]["cards"]
    assert intelligence.estimate_resume(conn, pol, job["payload"]["creator_ids"])["posts"] == 0


def test_resume_of_an_unknown_batch_queues_nothing(conn, capsys, database_url, monkeypatch):
    from manyworlds import cli
    monkeypatch.setenv("DATABASE_URL", database_url)
    assert cli.main(["research", "pilot", "--resume-annotate", "test-iia-nope", "--dry-run"]) == 2
    assert conn.execute("SELECT count(*) n FROM jobs WHERE kind = 'brain.intelligence'").fetchone()["n"] == 0
