"""Influencer intelligence into creative direction: pattern cards retrieved by archetype, format and style reach the
showrunner, the director, talk production and casting, are recorded on ideas and posts, learn from production
acceptance and real outcomes, and are written by the IIA from Apify pulls (stubbed here: no Apify, no LLM, no GPU)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from psycopg.types.json import Jsonb

from manyworlds import cuts, queue
from manyworlds.brain import overrides, patterns, proposals
from manyworlds.brain.agents import intelligence, showrunner
from manyworlds.research import annotation, references
from manyworlds.storage import LocalStore

from brain_helpers import StubLLM, creator as bare_creator, make_ctx
from test_brain_showrunner import START, PlanLLM, _maren, idea
from test_director import CANAL, DirectorLLM, text

FASHION = {"archetype": "quiet-luxury fashion in a small city flat", "niche": "fashion",
           "one_line": "outfits that cost less than they look"}
BEAUTY = {"archetype": "get ready with me beauty storyteller", "niche": "beauty grwm", "one_line": "stories while she gets ready"}


def _who(conn, cid, bible):
    conn.execute("UPDATE creators SET bible = bible || %s WHERE id = %s", (Jsonb(bible), cid))


# --- retrieval ----------------------------------------------------------------------------------------------------
def test_seed_cards_are_retrieved_by_archetype_and_format_and_marked_illustrative(conn, policy):
    got = patterns.retrieve(conn, policy, stage="director", who=FASHION, fmt="carousel")
    assert got and all(c["medium"] in ("carousel", "sequence", "photo") for c in got)
    assert len(got) <= policy.get("intelligence", "retrieval", "k", "director")
    assert all("fashion_style" in c["best_fit"] for c in got)
    talk = patterns.retrieve(conn, policy, stage="director", who=BEAUTY, fmt="talking_video")
    assert talk and {c["medium"] for c in talk} == {"video"}
    assert "grwm_confession" in [c["key"] for c in talk]
    lines = patterns.brief(talk)
    assert "ILLUSTRATIVE" in lines and "http" not in lines


def test_researched_cards_and_accepted_ones_outrank_illustrative(conn, policy):
    patterns.seed(conn, policy)
    illus = {"status": "illustrative", "sources": []}
    researched = {"status": "draft", "sources": [{"post_id": "p", "url": "https://x"}], "evidence": {"n_posts": 12}}
    assert patterns.weight(policy, researched) > patterns.weight(policy, illus)
    good = {**illus, "acceptance": {"tried": 10, "accepted": 10}}
    bad = {**illus, "acceptance": {"tried": 10, "accepted": 1}}
    assert patterns.weight(policy, good) > patterns.weight(policy, illus) > patterns.weight(policy, bad)
    won = {**illus, "outcomes": {"n": 5, "median_ratio": 1.8}}
    assert patterns.weight(policy, won) > patterns.weight(policy, illus)
    # a researched card of the same fit leads the retrieval
    conn.execute("UPDATE pattern_cards SET status = 'draft', sources = %s, evidence = %s WHERE key = 'mirror_phone_cover'",
                 (Jsonb([{"post_id": "p1", "creator_id": "c1", "url": "https://www.instagram.com/p/x/"}]),
                  Jsonb({"n_posts": 20, "n_creators": 6})))
    top = patterns.retrieve(conn, policy, stage="director", who=FASHION, fmt="photo")
    assert top[0]["key"] == "mirror_phone_cover"
    assert "researched: 20 posts from 6 creators" in patterns.brief(top[:1])
    assert "instagram.com" not in patterns.brief(top)                       # never the source links


def test_a_retired_card_is_never_retrieved(conn, policy):
    patterns.seed(conn, policy)
    conn.execute("UPDATE pattern_cards SET status = 'retired' WHERE key = 'mirror_phone_cover'")
    assert "mirror_phone_cover" not in [c["key"] for c in patterns.cards(conn, policy)]


# --- showrunner ---------------------------------------------------------------------------------------------------
def _run_showrunner(conn, policy, cid, ideas_):
    llm = PlanLLM(ideas_, conn=conn)
    job = queue.enqueue(conn, "brain.showrunner", {"creator_id": str(cid), "start": START}, creator_id=cid)
    ctx = make_ctx(conn, overrides.effective_policy(conn, policy), job)
    ctx.llm = llm
    showrunner.Showrunner().run(ctx)
    return llm


def test_the_showrunner_is_offered_cards_and_an_idea_takes_its_cards_recipe(conn, policy):
    cid = _maren(conn)
    _who(conn, cid, FASHION)
    dump = idea(0, "carousel", scenes=("dinner with friends, flash", "her ring on a glass", "the table from above"), pattern="dinner_flash_friend")
    llm = _run_showrunner(conn, policy, cid, [dump, idea(1)])
    prompt = llm.calls[0]["content"]
    assert "Pattern cards from the creative pattern library" in prompt and "dinner_flash_friend" in prompt
    assert "ILLUSTRATIVE" in prompt and "http" not in prompt.split("Pattern cards")[1].split("Her ")[0]
    rows = {r["format"]: r["body"] for r in conn.execute("SELECT format, body FROM ideas").fetchall()}
    used = {u["key"]: u for u in rows["carousel"]["pattern_cards"]}
    assert used["dinner_flash_friend"]["applied"] is True and used["dinner_flash_friend"]["stage"] == "showrunner"
    assert all(k == "dinner_flash_friend" or not u["applied"] for k, u in used.items())
    assert not any(u["applied"] for u in rows["photo"].get("pattern_cards") or [])
    decision = conn.execute("SELECT inputs FROM decisions WHERE kind = 'content_batch'").fetchone()["inputs"]
    assert "dinner_flash_friend" in decision["pattern_cards"]


def test_a_talking_idea_takes_its_cards_grammar_and_clamped_overrides(conn, policy):
    cards = {c["key"]: c for c in patterns.cards(conn, policy)}
    moves = policy.get("moves", "library")
    body = showrunner.apply_pattern(policy, {}, "talking_video", cards, "grwm_confession", moves)
    assert body["edit_grammar"] == "story_time"                               # the card's grammar
    assert body["edit_overrides"] == {"punch_every_s": [2.0, 3.5], "cutaway_share": 0.25, "hook": "text_and_line"}
    assert cuts.choose(policy, body, BEAUTY) == "story_time"
    used = {u["key"]: u for u in body["pattern_cards"]}
    assert used["grwm_confession"]["applied"] and {u["stage"] for u in used.values()} == {"showrunner"}


def test_a_card_of_the_wrong_medium_is_ignored(conn, policy):
    body = showrunner.apply_pattern(policy, {}, "photo", {c["key"]: c for c in patterns.cards(conn, policy)},
                                    "grwm_confession", {})
    assert "edit_grammar" not in body and not any(u["applied"] for u in body["pattern_cards"])


# --- director, scheduling and talk production ---------------------------------------------------------------------
def _real_creator(conn, bible):
    cid = bare_creator(conn)
    conn.execute("UPDATE creators SET profile_version = 1, status = 'active', anchors = %s, bible = %s WHERE id = %s",
                 (Jsonb({"reference_images": [{"uri": "s3://b/r"}]}), Jsonb(bible), cid))
    return cid


def test_the_director_brief_carries_cards_and_the_post_records_which_were_used(conn, policy):
    from manyworlds.brain import ideas
    cid = _real_creator(conn, FASHION)
    iid = conn.execute(
        "INSERT INTO ideas (creator_id, batch, concept, format, body, status) VALUES (%s, %s, 'x', 'photo', %s, "
        "'proposed') RETURNING id",
        (cid, f"{cid}:{START}", Jsonb({"scenes": ["the lift mirror on the way out"], "caption": "c", "day": 0,
                                       "platform": "instagram", "script": "", "reasoning": "r",
                                       "pattern_cards": [{"key": "mirror_phone_cover", "id": None, "version": 0,
                                                          "stage": "showrunner", "applied": True}]}))).fetchone()["id"]
    conn.execute("INSERT INTO evaluations (item_kind, idea_id, creator_id, rubric, rubric_version, evaluator_model, "
                 "prompt_version, scores, samples, role) VALUES ('idea', %s, %s, 'idea', 1, 'm', 'p', '{}', 3, 'evaluator')",
                 (iid, cid))
    llm = DirectorLLM({"shots": [{**CANAL, "shot_type": "elevator_mirror"}]}, conn=conn)
    ideas.schedule_ready(conn, policy, now=datetime.now(timezone.utc), llm=llm)
    brief = text(llm.calls[0]["content"])
    assert "Pattern cards for this post" in brief
    assert brief.index("- mirror_phone_cover") < brief.index("Why it works")  # the idea's own card comes first
    assert "phone over the face" in brief                                     # its pose dimension
    body = conn.execute("SELECT body FROM posts").fetchone()["body"]
    used = {(u["key"], u["stage"]): u for u in body["pattern_cards"]}
    assert used[("mirror_phone_cover", "showrunner")]["applied"] is True        # carried from the idea
    assert used[("mirror_phone_cover", "director")]["applied"] is True          # its shot type was shot


def test_a_talking_posts_card_sets_the_performance_band_and_the_punch_in_rhythm(conn, policy):
    from manyworlds.brain import director
    cid = _real_creator(conn, BEAUTY)
    body = {"scenes": ["at the bathroom mirror"], "script": "So, last night happened.", "edit_grammar": "story_time",
            "edit_overrides": {"punch_every_s": [2.0, 3.5], "cutaway_share": 0.25, "hook": "text_and_line"},
            "pattern_cards": [{"key": "grwm_confession", "stage": "showrunner", "applied": True}]}
    llm = DirectorLLM({"shots": [{"set": {"location": "bathroom"}, "shot_type": "bathroom_grwm"}]}, conn=conn)
    director.direct(conn, policy, llm, body, cid, fmt="talking_video")
    brief = text(llm.calls[0]["content"])
    assert "Performance and edit from grwm_confession" in brief
    assert "3.8-4.3 words a second" in brief and "65% of the time" in brief and "Cutaways about 25%" in brief
    g = cuts.grammar_for(policy, "story_time", body)
    assert g["punch_in"]["every_s"] == [2.0, 3.5]
    assert cuts.grammar(policy, "story_time")["punch_in"]["every_s"] != [2.0, 3.5]   # the policy is untouched
    assert cuts.grammar_for(policy, "long_take", body).get("punch_in") is None       # no punch-ins stays none


def test_talk_overrides_are_clamped_to_policy_bounds(policy):
    card = {"recipe": {"edit": {"punch_every_s": [0.2, 9.0], "cutaway_share": 0.9, "hook": "line"}}}
    out = patterns.talk_overrides(policy, card)
    lo, hi = policy.get("intelligence", "retrieval", "talk_bounds", "punch_every_s")
    assert out["punch_every_s"] == [lo, hi]
    assert out["cutaway_share"] == policy.get("intelligence", "retrieval", "talk_bounds", "cutaway_share")[1]


# --- casting ------------------------------------------------------------------------------------------------------
def test_casting_and_the_style_director_see_aesthetic_references(conn, policy):
    from manyworlds.creator import lab

    class Bible(StubLLM):
        def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
            super().structured(role, system=system, content=content, schema=schema)
            raise RuntimeError("stop after the prompt")
    llm = Bible()
    with pytest.raises(RuntimeError):
        lab.write_bible(conn, policy, llm, {"archetype": "fashion and style", "summary": "outfits in a small flat"})
    prompt = llm.calls[0]["content"]
    assert "Aesthetic and performance references" in prompt and "dinner_flash_friend" in prompt
    assert [c["key"] for c in lab.casting_cards(conn, policy, {"archetype": "fashion and style"})]

    from manyworlds.brain.agents import style_director
    cards = patterns.retrieve(conn, policy, stage="style", who=FASHION)
    text_ = style_director._prompt(policy, {"persona": {}}, FASHION, "style", datetime.now(timezone.utc), cards)
    assert text_.startswith("Aesthetic references from the creative pattern library")


# --- outcome feedback ---------------------------------------------------------------------------------------------
def _post_with(conn, cid, status, key="mirror_phone_cover", applied=True, why=None):
    body = {"pattern_cards": [{"key": key, "stage": "director", "applied": applied}]}
    if why:
        body["drop_reason"] = why
    return conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, 'photo', %s, %s) "
                        "RETURNING id", (cid, status, Jsonb(body))).fetchone()["id"]


def _published(conn, post_id, reach, policy):
    age = int(policy.get("brain", "analysis", "target_age_s"))
    conn.execute("INSERT INTO publications (post_id, platform, external_id, published_at, simulated) "
                 "VALUES (%s, 'instagram', %s, now() - interval '5 days', false)", (post_id, str(post_id)))
    conn.execute("INSERT INTO metric_snapshots (post_id, metrics, post_age_s) VALUES (%s, %s, %s)",
                 (post_id, Jsonb({"reach": reach}), age))


def test_feedback_counts_acceptance_and_real_outcomes_only(conn, policy):
    real = bare_creator(conn, publish_mode="instagram")
    sandbox = bare_creator(conn)
    a = _post_with(conn, real, "published")
    b = _post_with(conn, real, "ready")
    _post_with(conn, real, "dropped", why="identity: face mismatch")
    _post_with(conn, real, "producing")                                        # not judged yet
    _post_with(conn, real, "ready", applied=False)                            # offered, not used
    other = [_post_with(conn, real, "published", key="dinner_flash_friend") for _ in range(3)]
    s = _post_with(conn, sandbox, "published")
    for pid, reach in [(a, 300), (b, 300), (other[0], 100), (other[1], 100), (other[2], 100)]:
        _published(conn, pid, reach, policy)
    conn.execute("INSERT INTO publications (post_id, platform, external_id, published_at, simulated) "
                 "VALUES (%s, 'simulated', 'sim', now() - interval '5 days', true)", (s,))
    out = patterns.feedback(conn, policy)
    assert out["cards"] >= 9
    row = conn.execute("SELECT acceptance, outcomes FROM pattern_cards WHERE key = 'mirror_phone_cover'").fetchone()
    assert row["acceptance"]["tried"] == 4 and row["acceptance"]["accepted"] == 3   # sandbox's one included: renders
    assert row["acceptance"]["drop_reasons"] == {"identity": 1}
    assert row["outcomes"]["n"] == 2 and row["outcomes"]["median_ratio"] == 3.0    # 300 over her median of 100
    again = patterns.feedback(conn, policy)                                         # idempotent
    assert conn.execute("SELECT acceptance FROM pattern_cards WHERE key = 'mirror_phone_cover'").fetchone()[
        "acceptance"] == row["acceptance"] and again["cards"] == out["cards"]


def test_the_chief_of_staff_runs_the_feedback(conn, policy):
    from manyworlds.brain import chief_of_staff
    out = chief_of_staff.tick(conn, policy)
    assert out["pattern_feedback"]["cards"] >= 9


# --- requests and the agent ---------------------------------------------------------------------------------------
def test_an_intelligence_request_commissions_the_agent_capped(conn, policy):
    cid = bare_creator(conn)
    r = proposals.file(conn, policy, kind="intelligence", from_role="talent_manager", creator_id=cid,
                       subject={"question": "what do GRWM storytellers do in their first 2 seconds?", "usd": 50,
                                "handles": ["instagram:mira.lou"]},
                       argument="her talking posts stall at 2 s")
    assert r["status"] == "accepted"
    job = conn.execute("SELECT payload, creator_id FROM jobs WHERE kind = 'brain.intelligence'").fetchone()
    assert job["payload"]["usd"] == policy.get("intelligence", "agent", "max_usd_per_commission")
    assert job["payload"]["handles"] == ["instagram:mira.lou"] and job["payload"]["mode"] == "pilot"
    again = proposals.file(conn, policy, kind="intelligence", from_role="talent_manager", creator_id=cid,
                           subject={"question": "another"}, argument="x")
    assert "refused" in again                                                  # one a week per creator


IG_POSTS = [{"id": f"p{i}", "shortCode": f"s{i}", "url": f"https://www.instagram.com/p/s{i}/",
             "type": "Video" if i % 2 else "Image", "caption": f"post {i}", "likesCount": 100 * (i + 1),
             "commentsCount": 5, "videoViewCount": 1000 * (i + 1) if i % 2 else None,
             "displayUrl": f"https://cdn.example/{i}.jpg", "videoUrl": f"https://cdn.example/{i}.mp4" if i % 2 else None,
             "timestamp": (datetime.now(timezone.utc) - timedelta(days=i)).isoformat(), "ownerUsername": "mira.lou"}
            for i in range(8)]


class FakeApify:
    def __init__(self):
        self.calls = []

    def __call__(self, actor, payload):
        self.calls.append((actor, payload))
        if actor == "apify~instagram-profile-scraper":
            return [{"username": "mira.lou", "fullName": "Mira Lou", "biography": "night outs", "followersCount": 42000}]
        if actor == "apify~instagram-post-scraper":
            who = payload["username"][0]
            return [{**p, "id": f"{who}-{p['id']}", "ownerUsername": who} for p in IG_POSTS]
        if actor == "apify~instagram-hashtag-scraper":
            return [{**IG_POSTS[0], "id": "h1", "ownerUsername": "other.creator", "ownerFollowersCount": 9000}]
        return []


class IIALLM(StubLLM):
    """Answers the IIA's calls: the plan, the reads (alternating situations) and a card draft."""

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema)
        if schema is intelligence.ScoutPlan:
            return schema(hashtags_instagram=["nightout"], why="w")
        if schema is intelligence.ReadBatch:
            n = sum(1 for b in content if b.get("type") == "text" and str(b.get("text", "")).startswith("Post "))
            return schema(reads=[intelligence.PostRead(post=j, annotation=annotation.Annotation(
                observed=annotation.Observed(visual=annotation.Visual(situation="dinner", device="friend",
                                                                      nearest_shot_type="flash_night")),
                mechanism=[annotation.Note(text="being there " * 15)], confidence=0.7)) for j in range(1, n + 1)])
        if schema is intelligence.CardDraft:
            return schema(key="Dinner Flash Table", medium="carousel", situation="a dinner with friends",
                          shot_types=["flash_night", "not_a_shot"], pose="seated", camera="friend's phone",
                          expression="laughing", styling="evening", lighting="flash", social_energy="group",
                          movement="still", mechanism="being there", audience_role=["aspiration"],
                          best_fit=["fashion_style", "nope"], grade="flash_digicam", move="bogus_move")
        return schema.model_construct()


def _agent(store):
    a = intelligence.Intelligence()
    a.fetch = FakeApify()

    def download(url, dest, max_bytes):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\xff\xd8 fake " + url.encode())
        return dest
    a.download = download
    a.store = lambda policy: store
    return a


def _small(policy):
    """Clusters of three posts are enough for the fixtures (policy: six). Every video is copied here (policy: only
    selected posts'; test_research_providers covers that), so each video is measured."""
    import dataclasses
    agent = {**policy.body["intelligence"]["agent"], "videos_for": "all",
             "distill": {**policy.body["intelligence"]["agent"]["distill"], "min_posts": 3}}
    return dataclasses.replace(policy, body={**policy.body, "intelligence": {**policy.body["intelligence"],
                                                                             "agent": agent}})


def _ctx(conn, policy, payload, batch="test-iia-x"):
    job = queue.enqueue(conn, "brain.intelligence", payload, batch=batch)
    ctx = make_ctx(conn, policy, job)
    ctx.llm = IIALLM(conn=conn)
    return ctx


def test_the_pilot_scouts_annotates_and_distills_into_a_draft_card(conn, policy, tmp_path):
    store = LocalStore(tmp_path / "objects")
    agent = _agent(store)
    policy = _small(policy)
    ctx = _ctx(conn, policy, {"mode": "pilot", "hypothesis": "aspirational yet relatable night-out", "usd": 5,
                              "handles": ["instagram:mira.lou"], "platforms": ["instagram"], "creators": 2,
                              "posts": 8})
    out = agent.run(ctx)
    # scout: the named creator and one from the hashtag pool, pulled with their posts and research media
    handles = {r["handle"] for r in conn.execute("SELECT handle FROM reference_creators").fetchall()}
    assert {"mira.lou", "other.creator"} <= handles
    posts = conn.execute("SELECT media, metrics, media_purge_after FROM reference_posts p JOIN reference_creators c "
                         "ON c.id = p.creator_id WHERE c.handle = 'mira.lou'").fetchall()
    assert len(posts) == 8 and all(p["media"] and p["media_purge_after"] for p in posts)
    assert all(references.is_research_uri(m["uri"]) for p in posts for m in p["media"])
    assert all("over_own_median" in p["metrics"] for p in posts)
    actors = [a for a, _ in agent.fetch.calls]
    assert "apify~instagram-post-scraper" in actors and "apify~instagram-profile-scraper" in actors
    post_call = next(p for a, p in agent.fetch.calls if a == "apify~instagram-post-scraper")
    assert post_call["username"] == ["mira.lou"] and post_call["resultsLimit"] == 8
    data = conn.execute("SELECT sum(usd) u, min(agent) a FROM cost_entries WHERE category = 'data' AND job_id = %s",
                        (ctx.job.id,)).fetchone()
    assert data["u"] > 0 and data["a"] == "intelligence:scout"
    # annotate: videos queued for measurement, every post read and stored with notes clipped
    assert conn.execute("SELECT count(*) n FROM jobs WHERE kind = 'finish.measure_video'").fetchone()["n"] == 8   # 4 videos each
    reads = conn.execute("SELECT mechanism, evidence FROM reference_annotations WHERE schema = %s",
                         (annotation.ANNOTATION_SCHEMA,)).fetchall()
    assert len(reads) >= 8
    assert all(len(r["mechanism"][0]["text"].split()) <= annotation.NOTE_WORDS for r in reads)
    # distill: a draft card in our names, unknown names dropped, sources are the high-traction posts
    card = conn.execute("SELECT * FROM pattern_cards WHERE key = 'dinner_flash_table'").fetchone()
    assert card["status"] == "draft" and card["body"]["dims"]["shot_types"] == ["flash_night"]
    assert card["body"]["best_fit"] == ["fashion_style"] and "move" not in card["body"]["recipe"]
    assert card["sources"] and all(s["post_id"] and s["url"] for s in card["sources"])
    assert card["evidence"]["n_posts"] == len(card["sources"]) and patterns.counts_as_evidence(
        {"status": card["status"], "sources": card["sources"]})
    assert out["capped"] is False and out["spent_usd"] <= 5
    # retrieval now offers it, weighted as evidence
    got = patterns.retrieve(conn, policy, stage="director", who=FASHION, fmt="carousel")
    assert "dinner_flash_table" in [c["key"] for c in got]
    # a rerun pulls nothing again
    n = len(agent.fetch.calls)
    agent.run(ctx)
    assert len(agent.fetch.calls) == n


def test_a_run_stops_at_its_cap(conn, policy, tmp_path):
    agent = _agent(LocalStore(tmp_path / "objects"))
    ctx = _ctx(conn, policy, {"mode": "pilot", "hypothesis": "h", "usd": 0.01, "hashtags": ["nightout"],
                              "handles": ["instagram:mira.lou"], "platforms": ["instagram"], "creators": 1,
                              "posts": 30})
    out = agent.run(ctx)
    assert out["capped"] is True and "annotate" not in out
    assert conn.execute("SELECT count(*) n FROM reference_creators").fetchone()["n"] == 0
    assert out["spent_usd"] <= 0.01


def test_forgotten_accounts_are_never_pulled(conn, policy, tmp_path):
    references.forget(conn, None, "mira.lou")
    agent = _agent(LocalStore(tmp_path / "objects"))
    ctx = _ctx(conn, policy, {"mode": "scout", "hypothesis": "h", "hashtags": ["x"], "handles": ["instagram:mira.lou"],
                              "platforms": ["instagram"], "creators": 2, "posts": 5, "usd": 5})
    out = agent.run(ctx)
    assert "mira.lou" not in {c["handle"] for c in out["scout"]["creators"]}
    assert not any(a == "apify~instagram-post-scraper" and p["username"] == ["mira.lou"]
                   for a, p in agent.fetch.calls)


def test_the_estimate_adds_up_from_policy_prices(policy):
    est = intelligence.estimate(policy, creators=40, posts=30, platforms=["instagram", "tiktok"])
    price = policy.get("intelligence", "agent", "usd_per_item")
    # 2026-11-25: Instagram posts come from the profile scraper's embedded latest posts (one item, 12 posts)
    assert policy.get("intelligence", "agent", "instagram_posts_from") == "profile"
    ig = 20 * price["instagram_profile"]
    tt = 20 * 30 * (price["tiktok_profile"] + price["tiktok_video_download"])
    assert est["lines"]["creator_pulls"] == round(ig + tt, 2)
    assert est["total_usd"] == round(sum(est["lines"].values()), 2) and est["posts"] == 20 * 12 + 20 * 30
    assert est["lines"]["candidate_pool"] <= policy.get("intelligence", "agent", "max_candidate_pool_usd")


def test_mw_research_pilot_dry_run_queues_nothing(conn, policy, capsys, database_url, monkeypatch):
    from manyworlds import cli
    monkeypatch.setenv("DATABASE_URL", database_url)
    rc = cli.main(["research", "pilot", "--hypothesis", "aspirational yet relatable night-out", "--usd", "16",
                   "--dry-run"])
    assert rc == 0
    printed = capsys.readouterr().out
    out = json.loads(printed[printed.index("{"):])
    assert out["dry_run"] is True and out["estimate"]["total_usd"] > 0 and out["estimate"]["cap_usd"] == 16
    assert conn.execute("SELECT count(*) n FROM jobs WHERE kind = 'brain.intelligence'").fetchone()["n"] == 0


def test_research_media_defaults_to_the_main_bucket_under_its_prefix(policy, monkeypatch, tmp_path):
    monkeypatch.delenv(references.STORE_ENV, raising=False)
    monkeypatch.setenv("STORAGE_URL", (tmp_path / "main").resolve().as_uri())
    store = references.research_store(policy)
    f = tmp_path / "x.jpg"
    f.write_bytes(b"x")
    uri = store.put_file(f, references.media_key("p", "x.jpg")).uri
    assert references.is_research_uri(uri)                                  # still refused in production jobs
    with pytest.raises(references.ResearchMediaRefused):
        references.check_job("image.zimage", {"ref": uri})
    import dataclasses
    off = dataclasses.replace(policy, body={**policy.body, "intelligence": {**policy.body["intelligence"],
                                                                            "storage": {"fallback": "none"}}})
    assert references.research_store(off) is None


def test_a_queued_pilot_is_test_work_so_a_paused_brain_worker_takes_it(conn, policy, capsys, database_url, monkeypatch):
    # 2026-11-25: the first pilot sat queued: while paused, workers claim only payload.test jobs.
    from manyworlds import cli
    monkeypatch.setenv("DATABASE_URL", database_url)
    assert cli.main(["research", "pilot", "--hypothesis", "aspirational yet relatable night-out", "--usd", "16"]) == 0
    job = conn.execute("SELECT payload FROM jobs WHERE kind = 'brain.intelligence'").fetchone()
    assert job["payload"]["test"] is True
