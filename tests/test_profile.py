"""Creators are versioned profiles: every section can change (except the face), each change
is a version with a reason, the change flows into new photos, and the planner works from the
profile's dials, pillars and video series."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from manyworlds import production
from manyworlds.brain import planner
from manyworlds.creator import lab, profile

from test_production import PlanStub, creator, post

START = date(2026, 11, 15)


def draft(**dials):
    return profile.ProfileDraft(
        look=profile.Look(hair="black bob with a clip", wardrobe="oversized knits", accessories="plain watch",
                          makeup="none", signature_item="thin round gold wire glasses"),
        persona=profile.Persona(traits=["dry", "warm"], vibe="quiet", ethos="small things matter", humour="deadpan",
                                caption_voice="lowercase, short", speaking_voice="low, unhurried", never=["glam"]),
        audience=profile.Audience(age_range="22-34", gender="mostly women", regions="UK, US", income="lower-middle",
                                  ethnicity_mix="broad", life_stage="renters living alone", what_they_get="calm"),
        pillars=[profile.Pillar(name="night rituals", about="a", why_it_lasts="b"),
                 profile.Pillar(name="cheap food", about="a", why_it_lasts="b")],
        video_series=[profile.VideoSeries(name="kettle ratings", premise="p", format="narrated_edit",
                                          what_it_says="w", pillar="night rituals")],
        arc=profile.Arc(long_arc="l", current_chapter="c", open_threads=["the scarf"]),
        commercialization=profile.Commercialization(product_lines=["candles"], signature_hook="her glasses",
                                                    stages=["world", "products"], shared_lines=["candles"],
                                                    rules=["never named"]),
        tone_mix=[profile.Weight(name="cozy", weight=3), profile.Weight(name="deadpan", weight=1)],
        cadence=[profile.Cadence(platform="tiktok", posts_per_day=dials.get("tiktok", 3)),
                 profile.Cadence(platform="instagram", posts_per_day=2), profile.Cadence(platform="myspace", posts_per_day=9)],
        images_per_video=dials.get("ipv", 3))


class DraftStub:
    def __init__(self, d):
        self.d = d

    def model_for(self, role):
        return "stub"

    def structured(self, role, **kw):
        return self.d


def built(conn, policy, **dials):
    cid = creator(conn)
    profile.build(conn, policy, DraftStub(draft(**dials)), cid)
    return cid


def test_build_keeps_the_face_and_clamps_the_dials(conn, policy):
    cid = built(conn, policy, tiktok=40)
    v, body = profile.current(conn, cid)
    assert v == 1 and body["identity"]["eyes"] == "brown eyes"
    lim = policy.get("creators", "dials_limits")
    assert body["dials"]["posts_per_day"] == {"tiktok": lim["posts_per_day_max"], "instagram": 2}   # no myspace
    assert body["dials"]["tone_mix"] == {"cozy": 0.75, "deadpan": 0.25}
    bible = conn.execute("SELECT bible, profile_version FROM creators WHERE id = %s", (cid,)).fetchone()
    assert bible["profile_version"] == 1
    assert bible["bible"]["appearance"]["signature_details"].startswith("thin round gold wire glasses")


def test_a_haircut_is_a_new_version_and_reaches_new_photos(conn, policy):
    cid = built(conn, policy)
    v = profile.set_path(conn, policy, cid, "look.hair", "short copper pixie cut", author="human", reason="test haircut")
    assert v == 2
    bible = conn.execute("SELECT bible FROM creators WHERE id = %s", (cid,)).fetchone()["bible"]
    assert "short copper pixie cut" in lab.photo_prompt(bible, "at a bus stop")
    assert "glasses" in lab.photo_prompt(bible, "at a bus stop")
    rows = conn.execute("SELECT version, author, reason FROM creator_profiles WHERE creator_id = %s ORDER BY version",
                        (cid,)).fetchall()
    assert [(r["author"], r["reason"]) for r in rows] == [("brain", "initial profile from bible and thesis"),
                                                           ("human", "test haircut")]


def test_the_face_cannot_change_and_every_change_needs_a_reason(conn, policy):
    cid = built(conn, policy)
    with pytest.raises(ValueError):
        profile.set_path(conn, policy, cid, "identity.eyes", "blue eyes", author="human", reason="x")
    with pytest.raises(ValueError):
        profile.set_path(conn, policy, cid, "look.hair", "x", author="human", reason=" ")


def test_the_planner_works_from_the_profile(conn, policy):
    cid = built(conn, policy)
    llm = PlanStub([
        post(0, pillar="night rituals", tone="cozy"),
        post(0, "narrated_edit", ["a", "b", "c"], platform="tiktok", series="kettle ratings", script="six out of ten."),
        post(1, "narrated_edit", ["a", "b", "c"], platform="tiktok", series="morning yoga"),     # not her series
        post(1, caption="embracing the little things in life \u2014 so cozy"),                    # reads as AI
        post(2, platform="tiktok", hashtags=["a", "b", "c", "d"]),
    ])
    out = planner.plan_period(conn, policy, llm, cid, start=START)
    assert out["posts"] == 3 and {d["why"][:5] for d in out["dropped"]} == {"video", "reads"}
    days = policy.get("planner", "days")
    assert f"Plan exactly {5 * days} posts" in llm.calls[0] and f"{(5 * days) // 4} of them are videos" in llm.calls[0]
    rows = conn.execute("SELECT format, platform, profile_version, body FROM posts WHERE creator_id = %s "
                        "ORDER BY scheduled_for", (cid,)).fetchall()
    assert all(r["profile_version"] == 1 for r in rows)
    assert rows[0]["body"]["pillar"] == "night rituals" and rows[1]["body"]["series"] == "kettle ratings"
    assert rows[2]["body"]["hashtags"] == ["a", "b"]                       # tiktok allows two by default


def test_the_producer_keeps_creators_planned_ahead(conn, policy):
    cid = built(conn, policy)
    now = datetime(2026, 11, 14, 12, tzinfo=timezone.utc)
    llm = PlanStub([post(0), post(2)])
    out = planner.ensure_plans(conn, policy, llm, now=now)
    assert out == [{"creator": out[0]["creator"], "start": "2026-11-15", "posts": 2}]
    assert planner.ensure_plans(conn, policy, llm, now=now) == []          # planned through the 17th
    later = planner.ensure_plans(conn, policy, llm, now=now + timedelta(days=2))
    assert later[0]["start"] == "2026-11-18" and len(llm.calls) == 2


def test_video_posts_wait_until_production_makes_their_format(conn, policy):
    cid = built(conn, policy)
    planner.plan_period(conn, policy, PlanStub([post(0, "story_clip", ["x"], series="kettle ratings")]), cid,
                        start=START)
    assert production.advance(conn, policy, now=datetime(2026, 11, 15, tzinfo=timezone.utc))["started"] == 0
