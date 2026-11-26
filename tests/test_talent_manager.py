"""The agency layer: an archetype library every planner reads, a dossier built section by section, and a talent
manager per creator that owns the dossier, briefs the showrunner and reports to the strategist."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds import queue
from manyworlds.brain import archetypes, portfolio
from manyworlds.brain.agents import showrunner, talent_manager
from manyworlds.creator import dossier, profile

from brain_helpers import StubLLM, make_ctx

SECTION_OUT = {
    "ArchetypePick": {"primary": "lifestyle_slow", "secondary": "made_up_archetype", "why": "routines"},
    "Positioning": {"why_follow": "a calm night routine after late shifts", "the_gap": "night owls get no calm content",
                    "point_of_view": "small rituals beat big plans", "signature_angle": "steam and a lamp at 2am",
                    "distinct_from_siblings": "quiet, not loud", "comparable_patterns": ["slow living diaries"]},
    "AudienceInsight": {"who_exactly": "a 27-year-old nurse on nights", "desires": ["calm"], "frustrations": ["noise"],
                        "when_they_scroll": "after a shift", "what_they_save": ["routines"],
                        "what_they_share": ["relatable lines"], "words_they_use": ["post-shift"],
                        "turn_offs": ["hustle talk"]},
    "Story": {"backstory": "moved cities for work", "values_in_action": ["cooks for her neighbour"],
              "quirks": ["names her plants"], "routines": ["tea at 2am"], "small_goals": ["finish a scarf"],
              "recurring_people": []},
    "ContentSystem": {"mix": {"photo": 2, "carousel": 1, "story": 1, "video": 4}, "talk_share": 1.7,
                      "series": [{"name": "2am tea", "premise": "the same ritual each night", "archetype": "storyteller",
                                  "capabilities": ["devices.pov", "edits.not_a_thing"], "per_week": 30,
                                  "hook": "a time stamp", "why_it_works": "calm"}],
                      "hooks": ["a time stamp"], "never_makes": ["hauls"]},
    # the Talent & Brand Bible sections (2026-11-25)
    "BrandCore": {"why_exist": "calm for night workers", "desire": "a life with quiet rituals", "represents": "rest",
                  "why_people_follow": "the 2am calm", "value_compounds": "a ritual people return to",
                  "values": ["care"], "ethos": "small rituals beat big plans", "aspirations": ["a garden"],
                  "personality": ["dry"], "appearance_traits": ["calm eyes"], "audience_relationship": "friend",
                  "narrative": "moved cities for work",
                  "presentation": {"register": "natural", "attractiveness": "quiet warmth",
                                   "wardrobe_range": "knits, linen, a swimsuit at the lake", "photographic_style": "soft",
                                   "never": ["explicit"]}},
    "BrandWorld": {"style": "soft knits", "environments": ["kitchen at night"], "hobbies": ["tea"],
                   "routines": ["tea at 2am"], "visual_language": "lamp light",
                   "repertoire": [{"name": "2am tea", "format": "photo", "why": "calm"},
                                  {"name": "dance", "format": "hologram", "why": "x"}],
                   "personal_style": {"finish": "effortless but intentional", "grooming": "clean skin",
                                      "hair_and_makeup": "low bun, bare skin", "nails": "short, bare",
                                      "jewelry_and_accessories": ["thin gold ring", "steel watch"],
                                      "eyewear": "round tortoiseshell glasses", "footwear": "soft loafers",
                                      "silhouettes_materials_colours": "knits, linen, oat and navy",
                                      "possessions": ["enamel kettle", "linen notebook"],
                                      "seasonal_and_occasion": "wool in winter, linen in summer",
                                      "signature_elements": ["thin gold ring", "round glasses"],
                                      "varies": "outfits and places by season"},
                   "world_details": ["a steaming mug", "linen sheets", "a lamp", "fresh flowers"]},
    "BrandCommerce": {"categories": ["home_interiors", "not_a_category"], "why_categories": "her kitchen",
                      "product_integration": "a kettle on the counter", "will_do": ["tea"], "wont_do": ["hauls"],
                      "will_wear": ["knits"], "wont_wear": ["logos"], "will_promote": ["tea"], "wont_promote": ["diets"]},
    # the performance profile (2026-11-26, manyworlds.performance)
    "PerformanceProfile": {"baseline": [{"control": "warmth", "level": 2}, {"control": "energy", "level": 1}],
                           "natural_range": [{"control": "energy", "low": 0, "high": 2}],
                           "signature": {"resting_expression": "calm", "smile": "slow", "gaze_habits": "down",
                                         "posture": "curled", "walk": "slow", "hands": "around a mug",
                                         "vocal_cadence": "soft", "laugh": "quiet", "mannerisms": ["sighs"]},
                           "opening": "already mid-sip", "closing": "a small nod",
                           "context_shifts": [{"context": "quiet_morning",
                                               "offsets": [{"control": "energy", "level": -1}]}]},
}


class DossierLLM(StubLLM):
    """Returns a fixed answer per schema; a CareerReview from `review`."""

    def __init__(self, conn=None, review=None):
        super().__init__(conn=conn)
        self.review = review or {}

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema)
        if schema.__name__ == "CareerReview":
            base = {"state": "early", "diagnosis": "", "working": [], "not_working": [], "focus": ["rituals"],
                    "series_next": ["2am tea"], "capabilities_to_try": ["devices.pov", "edits.nope"], "stop": [],
                    "hypotheses": [], "brief_for_showrunner": "more 2am tea", "profile_changes": [], "requests": [],
                    "predictions": [],
                    "conviction": {"probability": 0.4, "low": 0.2, "high": 0.6, "horizon_weeks": 8,
                                   "case": "a clear ritual angle nobody owns"}}
            return schema.model_validate({**base, **self.review})
        return schema.model_validate(SECTION_OUT[schema.__name__])


def _creator(conn, policy, handle="nightowl", **extra):
    cid = conn.execute("INSERT INTO creators (handle, status, bible) VALUES (%s, 'active', %s) RETURNING id",
                       (handle, Jsonb({"handle": handle, "niche": "cozy night routines",
                                       "appearance": {"apparent_age": 27}}))).fetchone()["id"]
    body = {"identity": {"apparent_age": 27}, "look": {"hair": "bob", "wardrobe": "knits", "accessories": "watch",
                                                      "makeup": "none", "signature_item": "glasses"},
            "persona": {"traits": ["dry"], "vibe": "quiet", "ethos": "e", "humour": "h", "caption_voice": "c",
                        "speaking_voice": "s", "never": ["glam"]},
            "content": {"pillars": [{"name": "night rituals", "about": "a", "why_it_lasts": "b"}]},
            "niche": "cozy night routines", "dials": {}, **extra}
    profile.save(conn, policy, cid, body, author="test", reason="seed")
    return cid


def _run(conn, policy, cid, mode, llm):
    job = queue.enqueue(conn, "brain.talent_manager", {"creator_id": str(cid), "mode": mode}, creator_id=cid)
    ctx = make_ctx(conn, policy, job)
    ctx.llm = llm
    return talent_manager.TalentManager().run(ctx), job


def test_every_archetype_names_real_capabilities_and_matches_by_words(policy):
    assert archetypes.unknown_capabilities(policy) == []
    assert set(archetypes.names(policy)) >= {"fashion_style", "beauty_grwm", "lifestyle_slow", "storyteller"}
    assert archetypes.for_creator(policy, {"niche": "fitness and running"}) == ["movement"]   # not fashion ("fit")
    assert archetypes.for_creator(policy, {"archetypes": ["storyteller"], "niche": "makeup"}) == ["storyteller"]
    assert archetypes.communicates(policy, {"archetype": "retired lighthouse keeper"}) == [
        archetypes.library(policy)["default_communicates"]]
    for name in archetypes.names(policy):
        mix = archetypes.get(policy, name)["mix"]
        assert abs(sum(mix.values()) - 1) < 1e-6, name


def test_onboarding_builds_every_section_once_with_code_checks(conn, policy):
    cid = _creator(conn, policy)
    llm = DossierLLM(conn=conn)
    out, job = _run(conn, policy, cid, "onboard", llm)
    assert out["onboard"]["built"] == list(dossier.SECTIONS)
    version, prof = profile.current(conn, cid)
    assert prof["archetypes"] == ["lifestyle_slow"]                           # unknown archetype dropped
    s = prof["content_system"]
    assert abs(sum(s["mix"].values()) - 1) < 0.01 and s["talk_share"] == 1.0
    assert s["series"][0]["capabilities"] == ["devices.pov"]             # unknown capability dropped
    assert s["series"][0]["archetype"] == "lifestyle_slow" and s["series"][0]["per_week"] == 7.0
    authors = {r["author"] for r in conn.execute("SELECT author FROM creator_profiles WHERE creator_id = %s "
                                                 "AND version > 1", (cid,)).fetchall()}
    assert authors == {"talent_manager"}
    brief = talent_manager.latest_brief(conn, cid)["output"]
    assert brief["capabilities_to_try"] == ["devices.pov"]
    calls = len(llm.calls)
    talent_manager.TalentManager().run(_ctx_again(conn, policy, job, llm))
    assert len(llm.calls) == calls                                            # a rerun asks nothing again


def _ctx_again(conn, policy, job, llm):
    ctx = make_ctx(conn, policy, job)
    ctx.llm = llm
    return ctx


def test_a_positioning_too_close_to_a_sibling_is_rewritten_once(conn, policy):
    _creator(conn, policy, handle="sibling", positioning=SECTION_OUT["Positioning"],
             content_system=SECTION_OUT["ContentSystem"])
    cid = _creator(conn, policy)
    llm = DossierLLM(conn=conn)
    out, _ = _run(conn, policy, cid, "onboard", llm)
    diff = out["onboard"]["differentiation"]
    assert diff.get("rewritten") and diff["closest"] == "sibling"
    assert sum("too much" in str(c["content"]) for c in llm.calls) == 1


def test_a_review_changes_only_what_it_may_and_asks_the_owner_for_the_rest(conn, policy):
    cid = _creator(conn, policy)
    llm = DossierLLM(conn=conn, review={"profile_changes": [
        {"path": "look.wardrobe", "value": json.dumps("long cardigans"), "reason": "warmer"},
        {"path": "persona.never", "value": json.dumps([]), "reason": "loosen"},
        {"path": "identity.face", "value": json.dumps("new"), "reason": "x"}],
        "requests": [{"kind": "escalate", "subject": json.dumps({"ask": "a second set"}), "argument": "repetition"},
                     {"kind": "tier_change", "subject": json.dumps({"tier": "standard"}), "argument": "video works"}],
        "predictions": [{"claim": "reach doubles", "metric": "median_reach", "comparator": "above", "threshold": 500,
                         "probability": 0.6, "horizon_days": 21}]})
    out, _ = _run(conn, policy, cid, "review", llm)
    r = out["review"]
    assert (r["applied"], r["needs_owner"], r["requests"], r["predictions"]) == (1, 2, 3, 1)
    prof = profile.current(conn, cid)[1]
    assert prof["look"]["wardrobe"] == "long cardigans" and prof["persona"]["never"] == ["glam"]
    reqs = {(x["kind"], x["to_role"]) for x in conn.execute("SELECT kind, to_role FROM proposals").fetchall()}
    assert reqs == {("escalate", "owner"), ("tier_change", "strategist")}
    assert conn.execute("SELECT count(*) n FROM events WHERE type = 'request_for_owner'").fetchone()["n"] == 2
    conv = conn.execute("SELECT mean, low, high FROM convictions WHERE subject_id = %s", (cid,)).fetchone()
    assert (float(conv["mean"]), float(conv["low"]), float(conv["high"])) == (0.4, 0.2, 0.6)
    snap = portfolio.snapshot(conn, policy)
    mine = next(c for c in snap["creators"] if c["creator_id"] == str(cid))
    assert mine["talent"]["conviction"]["case"] == "a clear ritual angle nobody owns"   # the CEO reads the case


def test_the_showrunner_plans_from_the_brief_and_the_playbook(conn, policy):
    cid = _creator(conn, policy, archetypes=["lifestyle_slow"])
    _run(conn, policy, cid, "review", DossierLLM(conn=conn))
    job = queue.enqueue(conn, "brain.showrunner", {"creator_id": str(cid), "start": "2026-11-23"}, creator_id=cid)
    prompt, meta = showrunner._prompt(make_ctx(conn, policy, job), cid, datetime(2026, 11, 23).date(), 3)
    assert "more 2am tea" in prompt and "talent manager's brief" in prompt
    assert archetypes.get(policy, "lifestyle_slow")["audience_promise"] in prompt
    assert meta["talent_brief_id"]


def test_reviews_are_queued_on_the_stage_cadence_for_active_creators_only_when_enabled(conn, policy):
    cid = _creator(conn, policy)
    conn.execute("INSERT INTO creators (handle, status) VALUES ('draftie', 'draft')")
    days = policy.get("brain", "talent_manager", "review_days_by_stage", "exploring")
    t0 = datetime(2026, 11, 23, 9, tzinfo=timezone.utc)
    t0 = datetime.fromtimestamp((t0.timestamp() // (days * 86400)) * days * 86400, tz=timezone.utc)   # a period start
    first = talent_manager.ensure_reviews(conn, policy, now=t0)
    again = talent_manager.ensure_reviews(conn, policy, now=t0.replace(hour=15))
    assert len(first) == 1 and first == again                                 # one per creator per period
    later = talent_manager.ensure_reviews(conn, policy, now=t0 + timedelta(days=days))
    assert later != first                                                     # the next period: a new review
    off = dataclasses.replace(policy, body={**policy.body, "brain": {
        **policy.body["brain"], "talent_manager": {**policy.body["brain"]["talent_manager"], "enabled": False}}})
    assert talent_manager.ensure_reviews(conn, off, now=t0) == []
    assert conn.execute("SELECT creator_id FROM jobs WHERE kind = 'brain.talent_manager'").fetchone()["creator_id"] == cid
