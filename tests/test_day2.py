"""Day 2: research, Brain discovery and Creator Lab, with Claude and Apify stubbed."""

from __future__ import annotations

import pytest

from manyworlds.brain import discover as disc
from manyworlds.creator import lab
from manyworlds.research import signals
from manyworlds.research.apify import ApifySource, author_hash, normalize_instagram, normalize_tiktok

IG_ITEM = {"id": "ig1", "type": "Sidecar", "ownerUsername": "Someone.Real", "ownerFollowersCount": 20000,
           "caption": "sunday reset ☕️ #sundayreset #slowliving", "hashtags": ["sundayreset", "slowliving"],
           "likesCount": 900, "commentsCount": 40, "timestamp": "2026-11-01T10:00:00.000Z"}
TT_ITEM = {"id": "tt1", "text": "pov: your 7am walk", "authorMeta": {"name": "walker", "fans": 50000},
           "hashtags": [{"name": "morningroutine"}], "diggCount": 5000, "commentCount": 120,
           "playCount": 80000, "shareCount": 300, "createTime": 1762000000, "videoMeta": {"duration": 14}}


class FakeLLM:
    """Returns canned, schema-valid objects by output type, and records prompts."""

    def __init__(self, min_age_ok=True):
        self.prompts = []
        self.min_age_ok = min_age_ok

    def model_for(self, role):
        return "fake-model"

    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        self.prompts.append((role, schema.__name__, content))
        if schema is disc.ResearchPlan:
            return schema(rationale="r", plans=[
                {"platform": "instagram", "hashtags": ["#SundayReset", "slowliving"]},
                {"platform": "tiktok", "hashtags": ["morningroutine"]}])
        if schema is disc.ThesisSet:
            return schema(theses=[disc.Thesis(
                title=f"t{i}", archetype="cozy routines", audience="25-34 urban", why_now="median ER 0.05",
                format_mix=["photo dump"], differentiation="d", risks="r", production_fit="good",
                score=float(i)) for i in range(3)])
        if schema is disc.ThesisChoice:
            return schema(chosen_index=0, reasoning="highest learning per dollar")
        if schema is lab.CreatorBible:
            return schema(
                handle="@Mara.Leigh!", display_name="Mara", archetype="cozy routines", niche="lifestyle",
                one_line="a florist who over-plans her sundays",
                appearance=lab.Appearance(apparent_age=27 if self.min_age_ok else 17, gender_presentation="woman",
                                          ethnicity_and_skin="olive skin", face="oval face, freckles",
                                          eyes="hazel", hair="dark wavy shoulder-length", build="slim",
                                          signature_details="gold hoops"),
                style_and_wardrobe="linen", home_and_places="small flat", personality="dry humor",
                interests=["flowers"], opinions_and_running_jokes=["hates oat milk"], caption_voice="lowercase",
                posting_habits="mornings", never=["politics"], disclosure_line="florist, mostly",
                kokoro_voice="not_a_voice", voice_notes="soft",
                reference_prompts=[f"scene {i}" for i in range(12)])
        raise AssertionError(f"unexpected schema {schema}")


def test_normalizers_hash_authors_and_map_formats():
    ig = normalize_instagram(IG_ITEM, "sundayreset")
    tt = normalize_tiktok(TT_ITEM, "morningroutine")
    assert ig["media_type"] == "carousel" and tt["media_type"] == "video"
    assert ig["author_hash"] == author_hash("someone.real") and "Someone" not in str(ig["author_hash"])
    assert tt["views"] == 80000 and tt["hashtags"] == ["morningroutine"] and tt["posted_at"].year == 2025


def test_discover_end_to_end_with_stubs(conn, policy):
    calls = []

    def fetch(actor, payload):
        calls.append((actor, payload))
        return [IG_ITEM] if "instagram" in actor else [TT_ITEM]

    llm = FakeLLM()
    result = disc.discover(conn, policy, llm, ApifySource(conn, policy, fetch=fetch))
    assert [p["hashtags"] for p in result["plan"]["plans"]][0] == ["sundayreset", "slowliving"]
    assert len(calls) == 3 and len(result["theses"]) == 3
    assert conn.execute("SELECT count(*) n FROM research_items").fetchone()["n"] == 2
    data_cost = conn.execute("SELECT sum(usd) u FROM cost_entries WHERE category = 'data'").fetchone()["u"]
    assert data_cost > 0
    summary = signals.summarize(conn)
    assert summary["total_posts"] == 2 and "Someone" not in str(summary)
    choice = disc.choose_thesis(conn, policy, llm)
    assert choice["title"] == "t2"  # highest score sorted first, index 0
    kinds = {r["kind"] for r in conn.execute("SELECT kind FROM decisions").fetchall()}
    assert kinds == {"research_plan", "thesis_choice"}


def test_failed_pull_is_recorded_not_fatal(conn, policy):
    def boom(actor, payload):
        raise RuntimeError("actor timed out")

    r = ApifySource(conn, policy, fetch=boom).pull("instagram", "x", 10)
    assert r["items"] == 0 and "timed out" in r["error"]
    assert conn.execute("SELECT error FROM research_runs").fetchone()["error"] == "actor timed out"


def test_launch_creator_bible_guardrails_and_reference_pack(conn, policy):
    llm = FakeLLM()
    thesis_id = conn.execute("INSERT INTO theses (summary, status) VALUES ('t', 'chosen') RETURNING id").fetchone()["id"]
    bible = lab.write_bible(conn, policy, llm, {"id": thesis_id, "title": "t"})
    assert bible.handle == "mara.leigh"
    assert bible.kokoro_voice == "af_heart"                      # invalid voice replaced
    assert "AI-generated character" in bible.disclosure_line      # disclosure enforced
    row = lab.create_creator(conn, bible, thesis_id)
    batch = lab.enqueue_reference_pack(conn, policy, row["id"])
    lab.enqueue_reference_pack(conn, policy, row["id"])            # idempotent
    jobs = conn.execute("SELECT payload FROM jobs WHERE batch = %s", (batch,)).fetchall()
    assert len(jobs) == 12 and all("hazel" in j["payload"]["prompt"] for j in jobs)
    assert conn.execute("SELECT status FROM theses WHERE id = %s", (thesis_id,)).fetchone()["status"] == "launched"
    with pytest.raises(ValueError):
        lab.enqueue_lora_training(conn, policy, row["id"])          # no approved references yet


def test_underage_bible_is_rejected(conn, policy):
    with pytest.raises(ValueError):
        lab.write_bible(conn, policy, FakeLLM(min_age_ok=False), {"title": "t"})
