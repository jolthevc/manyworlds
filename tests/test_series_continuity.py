"""Series continuity and comment themes: a series' episodes enter when their posts go out (never when scheduled, never
when dropped), open threads are closed by later episodes, the showrunner sees where each series stands and what the
audience keeps saying, and the comment themes cost nothing without comments and never pass a handle on."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from psycopg.types.json import Jsonb

from manyworlds import queue
from manyworlds.brain import evidence, memory, overrides, series
from manyworlds.brain.agents import comment_themes, showrunner
from manyworlds.publish import scheduler
from manyworlds.research.apify import author_hash

from brain_helpers import StubLLM, creator, make_ctx

NOW = datetime(2026, 11, 20, 18, tzinfo=timezone.utc)
BRAIN = Path(__file__).resolve().parents[1] / "manyworlds" / "brain"


def _sim(conn):
    cid = creator(conn, publish_mode="simulated")
    conn.execute("UPDATE creators SET anchors = '{\"simulation\": {\"scenario\": \"null\"}}' WHERE id = %s", (cid,))
    return cid


def _post(conn, cid, *, series_name="Night shift", status="ready", when=NOW - timedelta(hours=1), **idea_body):
    post = conn.execute("INSERT INTO posts (creator_id, platform, format, status, caption, body, scheduled_for) "
                        "VALUES (%s, 'instagram', 'photo', %s, 'c', %s, %s) RETURNING id",
                        (cid, status, Jsonb({"platform_ai_label": True, "series": series_name}), when)).fetchone()["id"]
    conn.execute("INSERT INTO ideas (creator_id, concept, format, body, status, post_id) "
                 "VALUES (%s, 'photo: x', 'photo', %s, 'scheduled', %s)",
                 (cid, Jsonb({"series": series_name, **idea_body}), post))
    return post


def _episodes(conn, cid):
    return conn.execute("SELECT number, premise, post_id FROM series_episodes WHERE creator_id = %s ORDER BY number",
                        (cid,)).fetchall()


def test_an_episode_is_recorded_when_its_post_goes_out_not_when_it_is_scheduled(conn, policy):
    cid = _sim(conn)
    later = _post(conn, cid, when=NOW + timedelta(hours=3), episode_premise="the 4am oven fails")
    due = _post(conn, cid, episode_premise="first shift alone")
    assert _episodes(conn, cid) == []                        # scheduled: no episode yet
    scheduler.publish_due(conn, policy, now=NOW)
    eps = _episodes(conn, cid)
    assert [(e["number"], e["premise"], e["post_id"]) for e in eps] == [(1, "first shift alone", due)]
    scheduler.publish_due(conn, policy, now=NOW)              # never twice
    scheduler.publish_due(conn, policy, now=NOW + timedelta(hours=4))
    assert [(e["number"], e["post_id"]) for e in _episodes(conn, cid)] == [(1, due), (2, later)]


def test_a_dropped_post_never_becomes_an_episode(conn, policy):
    cid = _sim(conn)
    _post(conn, cid, status="dropped", opens_thread="part 2 promised")
    kept = _post(conn, cid)
    scheduler.publish_due(conn, policy, now=NOW)
    assert [(e["number"], e["post_id"]) for e in _episodes(conn, cid)] == [(1, kept)]
    assert conn.execute("SELECT count(*) n FROM series_threads").fetchone()["n"] == 0


def test_a_post_retracted_after_all_takes_its_episode_and_threads_with_it(conn, policy):
    cid = _sim(conn)
    first = _post(conn, cid, opens_thread="she promised the sourdough reveal")
    scheduler.publish_due(conn, policy, now=NOW)
    tid = series.planning_view(conn, policy, cid, {})["Night shift"]["open_threads"][0]["id"]
    second = _post(conn, cid, when=NOW + timedelta(minutes=30), closes_threads=[tid])
    scheduler.publish_due(conn, policy, now=NOW + timedelta(hours=1))
    assert "open_threads" not in series.planning_view(conn, policy, cid, {})["Night shift"]
    memory.retract_post(conn, [second])
    view = series.planning_view(conn, policy, cid, {})["Night shift"]
    assert view["next_episode"] == 2 and view["open_threads"][0]["id"] == tid   # reopened
    assert [e["post_id"] for e in _episodes(conn, cid)] == [first]


def test_the_next_episode_number_follows_the_published_episodes(conn, policy):
    cid = _sim(conn)
    for i in range(3):
        _post(conn, cid, when=NOW - timedelta(hours=3 - i), episode_premise=f"night {i + 1}")
    _post(conn, cid, series_name="Sunday market")
    scheduler.publish_due(conn, policy, now=NOW)
    assert series.next_number(conn, cid, "Night shift") == 4 and series.next_number(conn, cid, "Sunday market") == 2
    view = series.planning_view(conn, policy, cid, {"Night shift": {}, "Never aired": {}})
    assert view["Night shift"]["next_episode"] == 4 and view["Never aired"] == {"next_episode": 1}
    shown = int(policy.get("brain", "series", "episodes_shown"))
    assert [e["premise"] for e in view["Night shift"]["last_episodes"]] == ["night 3", "night 2", "night 1"][:shown]


# --- the showrunner -------------------------------------------------------------------------------------------

class PlanLLM(StubLLM):
    def __init__(self, ideas, conn=None):
        super().__init__(conn=conn)
        self.ideas = ideas

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema)
        return schema.model_validate({"rationale": "r", "ideas": self.ideas})


def _idea(**kw):
    return {"day": 0, "platform": "instagram", "format": "photo", "scenes": ["flour on the counter at 4am"],
            "caption": "the oven and I are not speaking", "hashtags": [], "canon": "the oven is old", **kw}


def test_the_showrunner_sees_where_each_series_stands_and_what_the_audience_says(conn, policy):
    cid = _sim(conn)
    conn.execute("UPDATE creators SET bible = %s, anchors = anchors || %s WHERE id = %s",
                 (Jsonb({"display_name": "Maren"}), Jsonb({"reference_images": [{"uri": "s3://b/r0"}]}), cid))
    _post(conn, cid, episode_premise="the oven gives up mid-bake", opens_thread="asked followers to name the oven")
    scheduler.publish_due(conn, policy, now=NOW)
    conn.execute("INSERT INTO decisions (kind, creator_id, inputs, output, policy_version) "
                 "VALUES ('comment_themes', %s, '{}', %s, 'v')",
                 (cid, Jsonb({"themes": [{"kind": "request", "text": "people want the oven to get a name",
                                          "series": "Night shift", "count": 5},
                                         {"kind": "question", "text": "people ask where the apron is from",
                                          "count": 3}]})))
    tid = series.planning_view(conn, policy, cid, {})["Night shift"]["open_threads"][0]["id"]
    llm = PlanLLM([_idea(series="Night shift", episode=2, episode_premise="the oven gets its name",
                         closes_threads=[tid.upper()])], conn=conn)
    job = queue.enqueue(conn, "brain.showrunner", {"creator_id": str(cid), "start": "2026-11-21"}, creator_id=cid)
    ctx = make_ctx(conn, overrides.effective_policy(conn, policy), job)
    ctx.llm = llm
    showrunner.Showrunner().run(ctx)
    prompt = str(llm.calls[0]["content"])
    assert "the oven gives up mid-bake" in prompt and '"next_episode": 2' in prompt     # last episode, next number
    assert "asked followers to name the oven" in prompt and tid in prompt              # the open thread and its id
    assert "people want the oven to get a name" in prompt and "audience_asked" in prompt   # the series' request
    assert "people ask where the apron is from" in prompt                               # the creator-wide theme
    body = conn.execute("SELECT body FROM ideas WHERE batch IS NOT NULL").fetchone()["body"]
    assert (body["episode"], body["episode_premise"], body["closes_threads"]) == (2, "the oven gets its name", [tid])


def test_without_series_or_themes_the_prompt_says_nothing_about_them(conn, policy):
    cid = _sim(conn)
    assert showrunner._continuity_text(conn, policy, cid, None) == ""


# --- comment themes -------------------------------------------------------------------------------------------

class ThemeLLM(StubLLM):
    def __init__(self, themes):
        super().__init__()
        self.themes = themes

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema)
        return schema.model_validate({"themes": self.themes})


def _real_with_comments(conn, texts, *, ours=()):
    cid = creator(conn, publish_mode="instagram")
    conn.execute("UPDATE creators SET status = 'active' WHERE id = %s", (cid,))
    conn.execute("INSERT INTO accounts (creator_id, platform, handle, external_id, status) "
                 "VALUES (%s, 'instagram', 'maren.bakes', '100', 'connected')", (cid,))
    other = creator(conn)
    conn.execute("INSERT INTO accounts (creator_id, platform, handle, external_id, status) "
                 "VALUES (%s, 'instagram', 'other.creator', '555', 'connected')", (other,))
    post = conn.execute("INSERT INTO posts (creator_id, format, status, caption, body) VALUES (%s, 'photo', "
                        "'published', 'the 4am croissant, thanks @bakery_friend', %s) RETURNING id",
                        (cid, Jsonb({"series": "Night shift"}))).fetchone()["id"]
    pub = conn.execute("INSERT INTO publications (post_id, platform, external_id, simulated, published_at) "
                       "VALUES (%s, 'instagram', 'm1', false, now()) RETURNING id", (post,)).fetchone()["id"]
    rows = [(t, author_hash(f"fan{i}"), "new", None) for i, t in enumerate(texts)]
    rows += [(t, author_hash("other.creator"), "new", None) for t in ours]   # one of ours, by hash
    rows += [("from our account, by reason", author_hash("x"), "skipped", "one of our accounts")]
    for i, (text, who, status, reason) in enumerate(rows):
        conn.execute("INSERT INTO comments (platform, external_id, publication_id, creator_id, author_hash, text, "
                     "commented_at, status, reason) VALUES ('instagram', %s, %s, %s, %s, %s, now(), %s, %s)",
                     (f"k{i}", pub, cid, who, text, status, reason))
    return cid


def _run_themes(conn, policy, cid, llm):
    job = queue.enqueue(conn, "brain.comment_themes", {"creator_id": str(cid)}, creator_id=cid)
    ctx = make_ctx(conn, policy, job)
    ctx.llm = llm
    return comment_themes.CommentThemes().run(ctx)


def test_no_comments_no_job_and_no_call(conn, policy):
    cid = creator(conn, publish_mode="instagram")
    conn.execute("UPDATE creators SET status = 'active' WHERE id = %s", (cid,))
    assert comment_themes.ensure(conn, policy, now=NOW) == []
    llm = ThemeLLM([])
    assert _run_themes(conn, policy, cid, llm) == {"comments": 0, "themes": 0} and llm.calls == []
    assert comment_themes.latest(conn, cid) is None


def test_themes_are_summarised_without_handles_and_without_our_own_accounts(conn, policy):
    cid = _real_with_comments(conn, ["@fan_handle_one look, part 2 please!", "where is the apron from? www.shop.example",
                                     "part 2 when"], ours=["our creator says hi"])
    jobs = comment_themes.ensure(conn, policy, now=datetime.now(timezone.utc))
    assert len(jobs) == 1
    llm = ThemeLLM([{"kind": "request", "text": "a part 2, as @fan_handle_one asked", "series": "Night shift",
                     "count": 2},
                    {"kind": "question", "text": "where the apron is from", "count": 1}])
    out = _run_themes(conn, policy, cid, llm)
    sent = str(llm.calls[0]["content"])
    assert llm.calls[0]["role"] == "comment_themes" and out["comments"] == 3
    assert "fan_handle_one" not in sent and "bakery_friend" not in sent and "www.shop" not in sent
    assert "our creator says hi" not in sent and "by reason" not in sent           # our accounts never count
    assert not re.search(r"[0-9a-f]{24}", sent)                                    # not even as hashes
    stored = str(comment_themes.latest(conn, cid)["output"])
    assert "fan_handle_one" not in stored and "part 2" in stored
    assert comment_themes.for_planning(conn, policy, cid)[0]["series"] == "Night shift"
    assert comment_themes.ensure(conn, policy, now=datetime.now(timezone.utc)) == []   # summarised: nothing new


def test_only_the_evidence_module_reads_comments_inside_the_brain(conn):
    pattern = re.compile(r"(FROM|JOIN)\s+comments\b", re.IGNORECASE)
    offenders = [str(p.relative_to(BRAIN)) for p in BRAIN.rglob("*.py")
                 if p.name != "evidence.py" and pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
    assert evidence.scrub("hi @a.b_c see https://x.example/y") == "hi @someone see [link]"
