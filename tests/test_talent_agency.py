"""The talent agency (spec 2026-11-25-talent-agency): strategy in policy, the capability card, the commercial registry,
the brand bible, casting with auditions, the portfolio map, growth metrics, and desirability judged in context with a
per-creator register that the learning loop cannot raise on engagement. Stubs only: no model, Apify or GPU."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from psycopg.types.json import Jsonb

from manyworlds import policy as policy_mod, queue
from manyworlds.brain import capability_card, casting, commerce, evidence, launches, portfolio_map
from manyworlds.brain.agents import talent_manager
from manyworlds.creator import dossier, lab, profile
from manyworlds.publish import comments, instagram, metrics
from manyworlds.review import common as review_common, gates, video

from brain_helpers import StubLLM, make_ctx

NOW = datetime(2026, 11, 25, 12, tzinfo=timezone.utc)


def _with(policy, path: list[str], value):
    body = dict(policy.body)
    node = body
    for k in path[:-1]:
        node[k] = dict(node.get(k) or {})
        node = node[k]
    node[path[-1]] = value
    return policy_mod.from_dict(body)


def _creator(conn, policy, handle, *, prof=None, bible=None, publish_mode="simulated"):
    cid = conn.execute("INSERT INTO creators (handle, status, bible, publish_mode) VALUES (%s, 'active', %s, %s) "
                       "RETURNING id", (handle, Jsonb(bible or {"appearance": {"apparent_age": 30,
                                                                                "gender_presentation": "woman"}}),
                                        publish_mode)).fetchone()["id"]
    if prof is not None:
        base = {"identity": {"apparent_age": 30}, "look": {"hair": "bob", "wardrobe": "knits", "accessories": "watch",
                                                          "makeup": "none", "signature_item": "glasses"},
                "persona": {"traits": ["dry"], "vibe": "quiet", "ethos": "e", "humour": "h", "caption_voice": "c",
                            "speaking_voice": "s", "never": ["x"]}, "dials": {}}
        profile.save(conn, policy, cid, {**base, **prof}, author="test", reason="seed")
    return cid


def _post(conn, cid, fmt="photo", status="ready", usd=0.0):
    pid = conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, %s, %s, '{}') RETURNING id",
                       (cid, fmt, status)).fetchone()["id"]
    if usd:
        conn.execute("INSERT INTO cost_entries (post_id, creator_id, category, provider, quantity, unit, usd) "
                     "VALUES (%s, %s, 'llm', 'test', 1, 'token', %s)", (pid, cid, usd))
    return pid


# --- strategy, capability card, commerce ---------------------------------------------------------------------------
def test_the_strategy_is_policy_with_a_rubric_a_to_e_and_a_feasibility_gate(policy):
    s = policy.get("strategy")
    crit = s["rubric"]["criteria"]
    assert list(crit) == ["desirability", "surface_area", "commercial", "feasibility", "distinctiveness"]
    assert abs(sum(c["weight"] for c in crit.values()) - 1) < 1e-9 and crit["feasibility"]["hard_gate"]
    assert s["focus"] in s["verticals"] and s["verticals"]["education"]["status"] == "deferred"
    assert "Reels" in s["content"]                       # the discovery layer is in the content guidance


def test_the_capability_card_uses_policy_until_measured_then_the_posts_and_the_ledger(conn, policy):
    card = capability_card.card(conn, policy)
    assert card["photo"]["source"] == "policy" and card["talking_video"]["status"] == "unproven"
    cid = _creator(conn, policy, "maker")
    n = int(policy.get("strategy", "capability_card", "min_tried"))
    for i in range(n):
        _post(conn, cid, "photo", "ready" if i % 4 else "dropped", usd=0.03)
    card = capability_card.refresh(conn, policy)
    accepted = sum(1 for i in range(n) if i % 4)
    assert card["photo"]["source"] == "measured" and card["photo"]["reliability"] == round(accepted / n, 3)
    assert card["photo"]["usd_per_accepted"] == round(0.03 * n / accepted, 4)
    assert capability_card.card(conn, policy)["photo"]["source"] == "measured"      # stored, read back
    assert capability_card.feasibility(card, {"photo": 1.0}) == round(10 * card["photo"]["reliability"], 2)
    assert capability_card.feasibility(card, {"talking_video": 1}) < capability_card.feasibility(card, {"carousel": 1})


def test_the_commercial_registry_is_seeded_as_estimates_and_research_is_never_overwritten(conn, policy):
    cats = commerce.categories(conn, policy)
    assert {c["source"] for c in cats} == {"estimate"} and cats[0]["index"] >= cats[-1]["index"]
    by = {c["key"]: c for c in cats}
    assert by["automotive"]["fit"] < by["mens_fashion"]["fit"]                   # cars are hard to render
    assert commerce.record_research(conn, "watches", fields={"influencer_activity": "high"},
                                    evidence={"sponsored_share": 0.1})
    commerce.seed(conn, policy)
    w = {c["key"]: c for c in commerce.categories(conn, policy)}["watches"]
    assert w["source"] == "research" and w["influencer_activity"] == "high" and w["evidence"]["sponsored_share"] == 0.1


# --- brand bible ---------------------------------------------------------------------------------------------------
def test_brand_sections_are_checked_by_code(policy):
    from test_talent_manager import SECTION_OUT
    world = dossier.clean(policy, "brand_world", dossier.BrandWorld.model_validate(SECTION_OUT["BrandWorld"]), {})
    assert [r["format"] for r in world["brand_world"]["repertoire"]] == ["photo"]      # not on the capability card
    com = dossier.clean(policy, "brand_commerce", dossier.BrandCommerce.model_validate(SECTION_OUT["BrandCommerce"]), {})
    assert com["brand_commerce"]["categories"] == ["home_interiors"]                  # unknown category dropped
    bad = {**SECTION_OUT["BrandCore"], "presentation": {**SECTION_OUT["BrandCore"]["presentation"], "register": "x"}}
    with pytest.raises(ValueError):
        dossier.clean(policy, "brand_core", dossier.BrandCore.model_validate(bad), {})
    view = dossier.planning_view({"brand_core": SECTION_OUT["BrandCore"], "brand_commerce": com["brand_commerce"]})
    assert view["brand_core"]["presentation"]["register"] == "natural" and view["brand_commerce"]["categories"]


def test_the_brand_core_changes_only_with_evidence_and_never_raises_the_register_on_engagement(conn, policy):
    from test_talent_manager import SECTION_OUT
    core = SECTION_OUT["BrandCore"]
    cid = _creator(conn, policy, "steady", prof={"brand_core": core})
    assert "real posts" in talent_manager.brand_core_gate(conn, policy, cid)
    prof = {"brand_core": core}
    up2 = {**core["presentation"], "register": "sultry_editorial"}                  # natural (1) -> 3
    up1 = {**core["presentation"], "register": "glamorous"}                         # natural (1) -> 2
    assert "brand fit" in talent_manager.presentation_gate(policy, prof, "brand_core.presentation", up1, "")
    assert "levels" in talent_manager.presentation_gate(policy, prof, "brand_core.presentation", up2,
                                                        "fits a new evening-wear direction")
    assert talent_manager.presentation_gate(policy, prof, "brand_core.presentation", up1,
                                            "her brand moved to evening wear for a fragrance positioning") is None
    assert talent_manager.presentation_gate(policy, prof, "brand_core", {**core, "values": ["x"]}, "") is None


def test_onboarding_builds_the_brand_bible_and_the_review_sees_brand_and_business(conn, policy):
    from test_talent_manager import DossierLLM, _creator as tm_creator, _run
    cid = tm_creator(conn, policy)
    llm = DossierLLM(conn=conn)
    out, _ = _run(conn, policy, cid, "onboard", llm)
    assert out["onboard"]["built"] == list(dossier.SECTIONS)
    prof = profile.current(conn, cid)[1]
    assert set(dossier.BRAND_SECTIONS) <= set(prof)
    review_prompt = str(llm.calls[-1]["content"])
    for text in ("capability card", "Her commercial categories", "What is working", "presentation register",
                 "Desirability:"):
        assert text in review_prompt


# --- portfolio map -------------------------------------------------------------------------------------------------
def test_the_portfolio_map_shows_coverage_overlaps_and_ranked_gaps(conn, policy):
    a = {"archetypes": ["lifestyle_slow"], "audience": {"age_range": "25-34", "gender": "women"},
         "brand_commerce": {"categories": ["womens_fashion", "home_interiors"]},
         "positioning": {"why_follow": "calm rituals at night"}}
    b = {**a, "brand_commerce": {"categories": ["womens_fashion"]}}
    _creator(conn, policy, "alma", prof=a)
    _creator(conn, policy, "bea", prof=b)
    conn.execute("INSERT INTO creators (handle, status, bible, anchors) VALUES ('clone', 'active', '{}', %s)",
                 (Jsonb({"simulation": {"audition": "x"}}),))
    m = portfolio_map.build(conn, policy)
    assert [c["handle"] for c in m["creators"]] == ["alma", "bea"]                  # auditions and clones left out
    assert m["coverage"]["cells"]["female 25-34"] == ["alma", "bea"]
    assert m["overlaps"] and m["overlaps"][0]["categories"] == ["womens_fashion"]
    assert "male 35-54" in m["gaps"]["cells"]
    gaps = [g["key"] for g in m["gaps"]["categories"]]
    assert "womens_fashion" not in gaps and "automotive" not in gaps[:2]           # cars rank low: hard to render
    assert "underexposed to the" in portfolio_map.text(m)


def test_the_ceo_prompt_carries_the_strategy_and_the_map(conn, policy):
    from manyworlds.brain.agents import strategist
    job = queue.enqueue(conn, "brain.strategist", {"day": "2026-11-25"})
    text = strategist._strategy_text(make_ctx(conn, policy, job))
    assert "Portfolio map" in text and "capability card" in text and "Commercial categories" in text


# --- casting -------------------------------------------------------------------------------------------------------
def _concept(i: int, **kw) -> dict:
    from test_talent_manager import SECTION_OUT
    return {"display_name": f"Candidate {i}", "handle": f"cand{i}", "one_line": f"concept {i}",
            "archetype": "lifestyle_slow", "desire": f"desire {i}", "why_follow": f"reason {i}",
            "signature": f"signature {i}", "apparent_age": 34, "gender_presentation": "man", "look": "tall",
            "styling": "tailoring", "environments": ["terrace"],
            "audience": {"core": "c", "age_range": "35-54", "gender": "men", "regions": "EU"},
            "brand_core": SECTION_OUT["BrandCore"],
            "strategy": {"commercial_categories": ["mens_fashion", "travel_hospitality"],   # nested (casting.flatten)
                         "content_mix": kw.pop("content_mix", {"photo": 0.6, "carousel": 0.3, "narrated_edit": 0.1})},
            **kw}


def _scores(i: int, total: float, **kw) -> dict:
    s = {k: {"score": total, "why": "w"} for k in casting.CRITERIA}
    return {"candidate": i, **s, "guardrail_ok": True, "guardrail_note": "", "generic_risk": "", **kw}


def _bible(handle: str) -> dict:
    return {"handle": handle, "display_name": handle.title(), "archetype": "lifestyle_slow", "niche": "style",
            "one_line": "a man of taste", "appearance": {"apparent_age": 38, "gender_presentation": "man",
                                                        "ethnicity_and_skin": "olive", "face": "square",
                                                        "eyes": "grey", "hair": "silver", "build": "tall",
                                                        "signature_details": "a steel watch"},
            "style_and_wardrobe": "tailoring", "home_and_places": "Milan", "personality": "dry",
            "interests": ["sailing"], "opinions_and_running_jokes": ["espresso"], "caption_voice": "short",
            "posting_habits": "mornings", "never": ["explicit"], "disclosure_line": "AI-generated character",
            "kokoro_voice": "am_michael", "voice_notes": "low", "reference_prompts": [f"ref {i}" for i in range(6)]}


class CastingLLM(StubLLM):
    def __init__(self, conn, concepts, scores):
        super().__init__(conn=conn)
        self.concepts, self.scores = concepts, scores

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema)
        self.calls[-1]["system"] = system
        if schema.__name__ == "Concepts":
            return schema.model_validate({"concepts": self.concepts})
        if schema.__name__ == "Scores":
            return schema.model_validate({"scores": self.scores, "note": "n"})
        if schema.__name__ == "AuditionDraft":
            return schema.model_validate({"bible": _bible(f"aud{len(self.calls)}"),
                                          "feed": [f"scene {i}" for i in range(9)]})
        raise AssertionError(schema.__name__)


def _thesis(conn):
    return conn.execute("INSERT INTO theses (summary, status, body) VALUES ('sophisticated man 35-55', 'proposed', %s) "
                        "RETURNING id", (Jsonb({"summary": "sophisticated male lifestyle"}),)).fetchone()["id"]


def test_casting_scores_concepts_applies_floors_gates_and_restrictions_and_shortlists_the_best(conn, policy):
    t = _thesis(conn)
    concepts = [_concept(1), _concept(2), _concept(3, content_mix={"talking_video": 1.0}), _concept(4)]
    scores = [_scores(1, 8), _scores(2, 7), _scores(3, 9), _scores(4, 9, guardrail_ok=False, guardrail_note="explicit")]
    llm = CastingLLM(conn, concepts, scores)
    out = casting.cast(conn, policy, llm, t, run="r1")
    by = {a["arm"]: a for a in out["auditions"]}
    assert by["A"]["stage"] == "shortlisted" and by["B"]["stage"] == "shortlisted"
    assert any("feasibility" in w for w in by["C"]["why_not"])                    # talking video: under the gate
    assert any("restricted" in w for w in by["D"]["why_not"])
    assert [c["role"] for c in llm.calls] == ["casting", "casting_judge"]          # two calls for four concepts
    assert "Desirability:" in str(llm.calls[1]["content"])
    assert casting.cast(conn, policy, llm, t, run="r1")["again"] and len(llm.calls) == 2


def _audition_to_feed(conn, policy, llm, monkeypatch, run="r2"):
    t = _thesis(conn)
    casting.cast(conn, policy, llm, t, run=run)
    out = casting.advance(conn, policy, run=run, llm=llm)                          # bibles, sandbox creators, packs
    assert len([m for m in out["moved"] if m.get("stage") == "references"]) == 2
    auds = conn.execute("SELECT * FROM auditions WHERE run = %s AND stage = 'references' ORDER BY arm",
                        (run,)).fetchall()
    for a in auds:
        anchors = conn.execute("SELECT anchors FROM creators WHERE id = %s", (a["creator_id"],)).fetchone()["anchors"]
        assert anchors["simulation"]["audition"] == str(a["id"])                   # never a portfolio creator
        jobs = conn.execute("SELECT batch, idempotency_key FROM jobs WHERE creator_id = %s", (a["creator_id"],)
                            ).fetchall()
        assert jobs and all(j["batch"].endswith(".refpack") and ":audition-ref:" in j["idempotency_key"]
                            for j in jobs)
    conn.execute("UPDATE jobs SET status = 'done' WHERE batch LIKE %s", (f"test-casting-{run}-%",))

    def anchors(conn_, policy_, llm_, store, creator_id, workdir, *, key="ref"):
        assert key == "audition-ref"
        conn_.execute("UPDATE creators SET anchors = anchors || %s WHERE id = %s",
                      (Jsonb({"reference_images": [{"uri": "s3://b/a.png"}]}), creator_id))
        return {"selected": 1}
    monkeypatch.setattr(lab, "select_anchors", anchors)
    casting.advance(conn, policy, run=run, llm=llm, store=None, workdir="/tmp")
    return t, conn.execute("SELECT * FROM auditions WHERE run = %s AND stage = 'feed' ORDER BY arm", (run,)).fetchall()


def test_auditions_render_a_mini_feed_as_test_posts_and_the_winner_is_picked_and_promoted(conn, policy, monkeypatch):
    llm = CastingLLM(conn, [_concept(1), _concept(2)], [_scores(1, 8), _scores(2, 7)])
    t, auds = _audition_to_feed(conn, policy, llm, monkeypatch)
    assert len(auds) == 2
    for a in auds:
        posts = casting._posts(conn, a["id"])
        assert len(posts) == 9 and all(p["body"]["breadth_run"] == f"casting-{a['run']}" for p in posts)
        assert all(p["scheduled_for"] is None for p in posts)                       # never published
        jobs = conn.execute("SELECT DISTINCT batch FROM jobs WHERE post_id = ANY(%s) AND status = 'queued'",
                            ([p["id"] for p in posts],)).fetchall()
        assert {j["batch"] for j in jobs} == {casting.batch(policy, a["run"], a["arm"])}
    a_posts, b_posts = casting._posts(conn, auds[0]["id"]), casting._posts(conn, auds[1]["id"])
    for i, p in enumerate(a_posts):        # A: 4 of 9 accepted (under the minimum); B: 8 of 9
        conn.execute("UPDATE posts SET status = %s WHERE id = %s", ("ready" if i < 4 else "dropped", p["id"]))
    for i, p in enumerate(b_posts):
        conn.execute("UPDATE posts SET status = %s WHERE id = %s", ("ready" if i < 8 else "dropped", p["id"]))
    out = casting.advance(conn, policy, run="r2")
    assert out["picked"] and out["picked"][0]["winner"] == str(auds[1]["id"])
    decision = conn.execute("SELECT output FROM decisions WHERE kind = 'casting_pick'").fetchone()["output"]
    assert decision["winner"] == "B"
    win = casting.promote(conn, policy, auds[1]["id"], t)
    c = conn.execute("SELECT anchors, bible FROM creators WHERE id = %s", (win["creator_id"],)).fetchone()
    assert "simulation" not in c["anchors"] and c["bible"]["casting"]["brand_core"]["presentation"]["register"]
    assert conn.execute("SELECT creator_id FROM theses WHERE id = %s", (t,)).fetchone()["creator_id"] == \
        uuid.UUID(win["creator_id"])
    assert conn.execute("SELECT status FROM creators WHERE id = %s", (auds[0]["creator_id"],)
                        ).fetchone()["status"] == "retired"


def test_a_launch_with_casting_starts_at_casting_and_goes_on_at_references_with_the_winner(conn, policy, monkeypatch):
    t = _thesis(conn)
    conn.execute("INSERT INTO approvals (kind, subject, reason, requested_by, status, decided_by) "
                 "VALUES ('launch', %s, 'test', 'strategist', 'approved', 'owner')",
                 (Jsonb({"thesis_id": str(t), "cohort": "c"}),))
    launches.advance(conn, policy)
    launch = conn.execute("SELECT * FROM launches").fetchone()
    assert launch["step"] == "casting"
    job = conn.execute("SELECT id, payload FROM jobs WHERE kind = 'brain.creator_lab'").fetchone()
    assert job["payload"]["step"] == "casting"
    from manyworlds.brain.agents import creator_lab
    ctx = make_ctx(conn, policy, job["id"])
    ctx.llm = CastingLLM(conn, [_concept(1)], [_scores(1, 8)])
    creator_lab.CreatorLab().run(ctx)
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job["id"],))
    run = casting.launch_run(launch)
    aud = conn.execute("SELECT * FROM auditions WHERE run = %s", (run,)).fetchone()
    assert aud["stage"] == "shortlisted" and str(aud["launch_id"]) == str(launch["id"])
    launches.advance(conn, policy)                                         # the audition's bible is a lab job
    j2 = conn.execute("SELECT id FROM jobs WHERE kind = 'brain.creator_lab' AND payload->>'step' = 'audition'"
                      ).fetchone()
    ctx = make_ctx(conn, policy, j2["id"])
    ctx.llm = CastingLLM(conn, [], [])
    creator_lab.CreatorLab().run(ctx)
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (j2["id"],))
    conn.execute("UPDATE auditions SET stage = 'judged', scores = scores || %s WHERE id = %s",
                 (Jsonb({"audition": {"acceptance": 0.9, "usd_per_accepted": 0.04}}), aud["id"]))
    out = launches.advance(conn, policy)
    launch = conn.execute("SELECT * FROM launches").fetchone()
    assert out["advanced"] == 1 and launch["step"] == "references" and launch["creator_id"] is not None


def test_casting_dry_run_estimate_comes_from_policy(policy):
    e = casting.estimate(policy)
    assert e["auditions"] == policy.get("brain", "casting", "audition_top") and e["total_usd"] > 0


# --- IIA commerce mode ---------------------------------------------------------------------------------------------
def _commerce_job(conn, policy, **payload):
    from manyworlds.brain.agents import intelligence
    job = queue.enqueue(conn, "brain.intelligence", {"mode": "commerce", **payload})
    ctx = make_ctx(conn, policy, job)
    agent = intelligence.Intelligence()
    fetched = []

    def fetch(actor, payload):
        fetched.append(payload)
        return [{"id": "1", "caption": "new watch #ad", "ownerUsername": "x"},
                {"id": "2", "caption": "sunday", "ownerUsername": "y"}]
    agent.fetch = fetch
    return agent.run(ctx), fetched


def test_the_commerce_mode_only_plans_while_off_and_refreshes_categories_when_on(conn, policy):
    out, fetched = _commerce_job(conn, policy, categories=["watches"])
    assert out["commerce"]["ran"] is False and not fetched and out["commerce"]["plan"][0]["key"] == "watches"
    on = _with(policy, ["intelligence", "commerce", "enabled"], True)
    out, fetched = _commerce_job(conn, on, categories=["watches"], usd=1.0)
    got = out["commerce"]["categories"][0]
    assert got["sponsored_share"] == 0.5 and got["influencer_activity"] == "high" and len(fetched) == 2
    row = {c["key"]: c for c in commerce.categories(conn, on)}["watches"]
    assert row["source"] == "research"
    assert conn.execute("SELECT count(*) n FROM cost_entries WHERE agent = 'intelligence:commerce'").fetchone()["n"] == 2


# --- metrics and evidence ------------------------------------------------------------------------------------------
def test_account_insights_are_collected_daily_and_growth_reads_them(conn, policy, monkeypatch):
    cid = _creator(conn, policy, "live", publish_mode="instagram")
    conn.execute("INSERT INTO accounts (creator_id, platform, status, external_id) VALUES (%s, 'instagram', "
                 "'connected', 'ig1')", (cid,))
    monkeypatch.setattr(instagram, "account", lambda conn_, c: instagram.Creds("ig1", "tok", "graph.facebook.com"))
    calls = []

    def graph(method, path, params=None, token=None):
        calls.append(params.get("metric") or params.get("fields"))
        if params.get("fields"):
            return {"followers_count": 1200}
        if params["metric"] == "reach":
            return {"data": [{"name": "reach", "total_value": {"value": 5000, "breakdowns": [{"results": [
                {"dimension_values": ["FOLLOWER"], "value": 1000},
                {"dimension_values": ["NON_FOLLOWER"], "value": 4000}]}]}}]}
        if params["metric"] == "follows_and_unfollows":
            return {"data": [{"name": "follows_and_unfollows", "total_value": {"breakdowns": [{"results": [
                {"dimension_values": ["FOLLOWER"], "value": 20}, {"dimension_values": ["NON_FOLLOWER"], "value": 3}]}]}}]}
        raise instagram.GraphError("400 unsupported")
    monkeypatch.setattr(instagram, "graph", graph)
    out = metrics.collect_account(conn, policy, now=NOW)
    assert out == {"accounts": 1, "errors": 1}                                   # profile_views refused: the rest kept
    assert metrics.collect_account(conn, policy, now=NOW)["accounts"] == 0       # once a day
    conn.execute("UPDATE account_insights SET day = current_date")
    g = evidence.growth(conn, cid, "real", days=28)
    assert g["follows_per_1k_non_follower_reach"] == 5.0 and g["followers_end"] == 1200
    assert evidence.growth(conn, cid, "sandbox") is None


def test_attribution_groups_outcomes_by_format_shot_type_and_pattern_card(conn, policy):
    cid = _creator(conn, policy, "attr", prof={"archetypes": ["lifestyle_slow"]}, publish_mode="instagram")
    for i, (fmt, st, reach) in enumerate((("photo", "back_at_sunset", 1000), ("carousel", "table_from_above", 3000),
                                          ("photo", "back_at_sunset", 2000))):
        pid = conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, %s, 'published', %s) "
                           "RETURNING id", (cid, fmt, Jsonb({"shots": [{"shot_type": st}], "pattern_cards": [
                               {"key": "golden_hour", "applied": True}]}))).fetchone()["id"]
        conn.execute("INSERT INTO publications (post_id, platform, external_id, ai_label, simulated, published_at) "
                     "VALUES (%s, 'instagram', %s, true, false, %s)", (pid, f"m{i}", NOW - timedelta(days=5)))
        conn.execute("INSERT INTO metric_snapshots (post_id, captured_at, simulated, metrics, post_age_s) VALUES "
                     "(%s, %s, false, %s, 300000)", (pid, NOW, Jsonb({"reach": reach, "saves": reach // 100,
                                                                        "follows": 2})))
    a = evidence.attribution(conn, [cid], "real", policy=policy)
    assert a["format"]["photo"]["n"] == 2 and a["format"]["carousel"]["lift"] == 1.5
    assert a["shot_type"]["back_at_sunset"]["saves_per_1k"] == 10.0
    assert a["pattern_card"]["golden_hour"]["n"] == 3 and a["archetype"]["lifestyle_slow"]["n"] == 3


# --- desirability, judged in context -------------------------------------------------------------------------------
def _image_review(**kw):
    base = {"index": 0, "identity": 9, "realism": 8, "defects": [], "looks_under_min_age": False, "sexualized": False,
            "real_brand_marks": False, "notes": "fine"}
    return gates.CandidateReview.model_validate({**base, **kw})


@pytest.mark.parametrize("notes", ["a bikini on a beach holiday, smiling at the sea",
                                   "a shirtless man swimming laps in an outdoor pool",
                                   "an evening dress with a deep neckline at a gala"])
def test_swimwear_shirtless_and_revealing_fashion_pass_the_safety_gate_in_context(notes):
    r = _image_review(notes=notes, defects=["cleavage visible"] if "neckline" in notes else [])
    assert gates.decide(r, {})["safety"] is True                  # clothing alone never fails the gate


@pytest.mark.parametrize("case", ["explicit sexual activity", "implied nudity: a cardigan with nothing under it",
                                  "fetish-oriented outfit and pose", "a caption soliciting sexual contact"])
def test_explicit_nude_fetish_and_soliciting_content_fails_the_safety_gate(case):
    r = _image_review(sexualized=True, notes=case)
    assert gates.decide(r, {})["safety"] is False
    clip = video.ClipReview.model_validate({"identity": 9, "motion": 8, "realism": 8, "defects": [],
                                            "looks_under_min_age": False, "sexualized": True,
                                            "real_brand_marks": False, "notes": case})
    assert video.decide(clip, {})["safety"] is False


def test_the_safety_rubric_is_contextual_and_the_same_for_images_and_clips(policy):
    rule = review_common.SAFETY_RULE
    for allowed in ("swimwear", "shirtless", "cleavage", "revealing fashion", "NOT sexualized"):
        assert allowed in rule
    for kept in ("staged nudity simulation", "fetish", "solicitation", "VISIBLE", "Never speculate"):
        assert kept in rule
    assert gates.CandidateReview.model_fields["sexualized"].description == rule
    assert video.ClipReview.model_fields["sexualized"].description == rule
    assert "never by clothing alone" in gates.SYSTEM and "never by clothing alone" in video.SYSTEM
    excluded = " ".join(policy.get("scope", "excluded_topics"))
    assert "sexualized content" not in excluded and "swimwear" not in excluded and "nudity" in excluded
    assert "flirt" in comments.SYSTEM and "girlfriend" in comments.SYSTEM
    assert "never sexualize" not in lab.SYSTEM.lower()
    assert set(policy.get("strategy", "platform_guidelines")) >= {"instagram", "tiktok"}


def test_her_presentation_register_bounds_the_showrunner_and_director(conn, policy):
    from test_talent_manager import SECTION_OUT
    line = dossier.presentation_line({"brand_core": SECTION_OUT["BrandCore"]})
    assert "natural" in line and "never justify" in line and "don't sanitise" in line
    assert dossier.presentation_line({}) == ""
    text = dossier.desirability_text(policy)
    assert "Vogue" in text and "never by clothing alone" in text and "sultry_editorial" in text


# --- the desirability range: an end-to-end matrix of prompts, negatives and QA (owner 2026-11-25) --------------------
CONTRADICTIONS = ("never sexualize", "be modest", "dress modestly", "fully clothed", "no swimwear", "no cleavage",
                  "cover up", "no skin", "thirst trap", "suggestive pose", "no shirtless", "no lingerie",
                  "appears to wear nothing")

MATRIX = [   # (case, what the judge saw, the creator's register, distribution risk the judge reports)
    ("luxury swimwear at a resort", "an elegant one-piece swimsuit by a resort pool", "glamorous", 4),
    ("low-cut and backless evening dress", "a backless black evening dress, low neckline, at a gala", "glamorous", 3),
    ("fitted fashion and activewear", "fitted matching activewear in a pilates studio", "sporty", 1),
    ("shirtless male beach and fitness", "a shirtless man walking out of the sea in swim shorts", "natural", 3),
    ("glamorous fashion editorial", "a glamour editorial, silk gown, flash, magnetic gaze", "sultry_editorial", 3),
    ("confident magnetic pose", "a confident direct gaze, hand on hip, editorial light", "refined", 1),
    ("bathrobe relaxed morning", "a white hotel bathrobe by the window, coffee on the table", "natural", 1),
    ("covered underwear fashion campaign", "a tasteful covered lingerie campaign, studio, editorial",
     "sultry_editorial", 6),
    ("fragrance and jewelry luxury ad", "an unbranded perfume bottle near the collarbone, gold necklace", "glamorous", 1),
    ("beauty manicure pedicure grooming", "close-up of a fresh manicure on a coffee cup; a groomed beard", "refined", 0),
    ("styled outfit with coordinated jewelry", "camel coat, gold necklace and earrings, leather bag", "refined", 0),
    ("an oversized shirt worn as a dress", "an oversized white shirt worn as a dress, covered, at home", "natural", 2),
]
NEGATIVES = [("explicit sexual content", {"sexualized": True}), ("a staged nudity simulation", {"sexualized": True}),
             ("fetish-oriented content", {"sexualized": True}), ("sexual solicitation", {"sexualized": True}),
             ("could read as a minor", {"looks_under_min_age": True})]


def _prompt_texts(policy) -> dict[str, str]:
    from manyworlds.brain import aesthetics, director
    from manyworlds.brain.agents import showrunner
    from manyworlds.worker import comfy
    return {"casting": casting.SYSTEM + casting.JUDGE, "lab": lab.SYSTEM, "dossier": dossier.SYSTEM,
            "persona_guide": str(policy.get("creator_lab", "persona_guide")),
            "desirability": dossier.desirability_text(policy),
            "style_rules": str(aesthetics.library(policy)["rules"]), "showrunner": showrunner._strategy_line(policy),
            "director": director.static_brief(policy, "photo"), "negative_comfy": comfy.NEGATIVE,
            "negatives_policy": str(policy.get("direction", "negatives")),
            "safety": review_common.SAFETY_RULE + review_common.SAFETY_SYSTEM + gates.SYSTEM + video.SYSTEM,
            "comments": comments.SYSTEM, "excluded": " ".join(policy.get("scope", "excluded_topics"))}


def test_no_prompt_or_negative_prompt_carries_a_contradictory_restriction(policy):
    for name, text in _prompt_texts(policy).items():
        low = text.lower()
        assert not [c for c in CONTRADICTIONS if c in low], name
    excluded = _prompt_texts(policy)["excluded"]
    assert "dating" not in excluded and "romance" not in excluded            # fictional adult relationships are fine
    assert "simulated romantic relationships with followers" in excluded


@pytest.mark.parametrize("case,seen,register,risk", MATRIX)
def test_the_range_passes_safety_and_brand_and_distribution_never_rejects(conn, policy, store, tmp_path, case, seen,
                                                                          register, risk):
    from test_review import JudgeStub, creator_with_job, ok
    from test_talent_manager import SECTION_OUT
    cid, job = creator_with_job(conn, store, tmp_path, n_images=1)
    core = {**SECTION_OUT["BrandCore"], "presentation": {**SECTION_OUT["BrandCore"]["presentation"],
                                                         "register": register}}
    base = {"identity": {"apparent_age": 29}, "look": {"hair": "bob", "wardrobe": "", "accessories": "",
                                                      "makeup": "", "signature_item": ""},
            "persona": {"traits": [], "vibe": "", "ethos": "", "humour": "", "caption_voice": "", "speaking_voice": "",
                        "never": []}, "dials": {}, "brand_core": core}
    profile.save(conn, policy, cid, base, author="test", reason="seed")
    judge = JudgeStub([ok(notes=seen, brand_alignment=8, distribution_risk=risk,
                          distribution_label="mainstream" if risk < 5 else "limit boosting")])
    res = gates.review_job(conn, policy, judge, store, job, tmp_path)[0]
    assert res["accepted"] and res["safety"], case
    shown = " ".join(str(b.get("text", "")) for b in judge.calls[0]["content"])
    assert "HER PRESENTATION" in shown and register in shown                    # one call, all three judgments
    dist = conn.execute("SELECT passed, details FROM reviews WHERE gate = 'distribution'").fetchone()
    assert dist["passed"] and dist["details"]["report_only"]                     # a report, never a reject
    assert len(judge.calls) <= 2                                                 # screen (+ a confirm): no new call


@pytest.mark.parametrize("case,flags", NEGATIVES)
def test_explicit_staged_nude_fetish_soliciting_and_minor_content_is_rejected(conn, policy, store, tmp_path, case,
                                                                               flags):
    from test_review import JudgeStub, creator_with_job, ok
    _, job = creator_with_job(conn, store, tmp_path, n_images=1)
    res = gates.review_job(conn, policy, JudgeStub([ok(notes=case, **flags)]), store, job, tmp_path)[0]
    assert not res["accepted"] and not res["safety"]


def test_a_clear_brand_misfit_only_ranks_and_minor_detail_defects_are_only_reported(conn, policy, store, tmp_path):
    """Generous publishing (owner 2026-11-26): an off-register image is flagged and ranks lower, never rejected; a
    major detail defect (a material artifact) still fails realism."""
    from test_review import JudgeStub, creator_with_job, ok
    _, job = creator_with_job(conn, store, tmp_path, n_images=3)
    judge = JudgeStub([ok(brand_alignment=1), ok(detail_defects=["minor: a ring slightly soft"]),
                       ok(detail_defects=["major: melted nails on the visible hand"])])
    res = gates.review_job(conn, policy, judge, store, job, tmp_path)
    assert [r["accepted"] for r in res] == [True, True, False]
    assert res[1]["realism"] and not res[2]["realism"]
    brand = conn.execute("SELECT passed, score, details FROM reviews WHERE gate = 'brand'").fetchall()
    assert brand and all(b["passed"] for b in brand)
    assert any(b["details"]["off_register"] and b["details"]["rank_only"] for b in brand)


def test_simulated_intimacy_with_followers_is_refused_but_charm_is_asked_for():
    s = comments.SYSTEM
    assert "warmly acknowledge compliments" in s and "banter" in s and "never cold or robotic" in s
    assert "romantic relationship" in s and "girlfriend or boyfriend" in s


def test_personal_style_reaches_the_director_and_its_details_map_to_commercial_categories(policy):
    from test_talent_manager import SECTION_OUT
    prof = {"brand_world": SECTION_OUT["BrandWorld"]}
    line = dossier.styling_line(prof)
    assert "signature elements recur subtly" in line and "round tortoiseshell glasses" in line
    assert "linen sheets" in line
    adj = commerce.adjacent(policy, [prof["brand_world"], "a fresh manicure", "a briefcase", "a gold necklace"])
    assert {"beauty_skincare", "luxury_accessories", "jewelry", "watches", "eyewear", "home_interiors"} <= set(adj)
    assert "personal_style" in dossier.brand_view(prof)["brand_world"]


def test_the_range_breadth_plan_is_photos_only_within_two_dollars(conn, policy):
    import dataclasses
    from manyworlds import breadth
    from test_breadth import _creator as breadth_creator
    pol = dataclasses.replace(policy, body={**policy.body, "operations": {**policy.body["operations"],
                                                                         "mode": "paused"}})
    her = breadth_creator(conn)
    conn.execute("UPDATE creators SET handle = 'range.marco' WHERE handle = %s", (breadth_creator(conn),))
    plan, items = breadth.load_plan(breadth.test_policy(pol), "range")
    assert float(plan["max_usd"]) <= 2 and len(items) == 12 and {i.format for i in items} == {"photo"}
    assert any(i.creator == "range.marco" for i in items)
    out = breadth.run(conn, pol, "range", creator=her, dry_run=True)
    assert out["dry_run"] and not out["skipped"] and len(out["items"]) == 12
    assert 0 < out["estimated_usd"] <= float(plan["max_usd"])
    assert conn.execute("SELECT count(*) n FROM posts").fetchone()["n"] == 0     # a dry run makes nothing
