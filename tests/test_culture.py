"""Culture intelligence beyond social media (owner 2026-11-25; manyworlds.research.culture): publishers' feeds kept as
source references only, signals in our words with a code-given kind, read by style, casting and the showrunner."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from manyworlds.research import culture, data_policy

NOW = datetime(2026, 11, 25, 12, tzinfo=timezone.utc)

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>Feed</title>
<item><title>Butter yellow takes over the autumn runways</title><link>https://pub.example/a</link>
<pubDate>Tue, 24 Nov 2026 10:00:00 GMT</pubDate><description>&lt;p&gt;Long article text here.&lt;/p&gt;</description></item>
<item><title>Quiet hotels in Lisbon</title><link>https://pub.example/b</link>
<pubDate>Mon, 23 Nov 2026 10:00:00 GMT</pubDate><description>More text.</description></item>
<item><title>Old story</title><link>https://pub.example/old</link><pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Butter yellow kitchens</title>
<link href="https://other.example/c"/><updated>2026-11-24T09:00:00Z</updated><summary>Text.</summary></entry></feed>"""


def _sources(policy, **status):
    body = dict(policy.body)
    src = dict(body["data_policy"]["sources"])
    for name, st in status.items():
        src[name] = {**src[name], "status": st}
    body["data_policy"] = {**body["data_policy"], "sources": src}
    return dataclasses.replace(policy, body=body)


def _feeds(policy, feeds):
    # direct publisher feeds are paused as shipped (their terms); a test approves them, GDELT off
    policy = _sources(policy, feeds="approved", gdelt="paused")
    body = {**policy.body, "intelligence": {**policy.body["intelligence"], "culture": {
        **policy.body["intelligence"]["culture"], "feeds": feeds}}}
    return dataclasses.replace(policy, body=body)


def test_items_are_kept_as_references_never_their_text(conn, policy):
    pol = _feeds(policy, [{"name": "pub", "category": "fashion", "url": "u1"},
                          {"name": "other", "category": "interiors", "url": "u2"}])
    new, report = culture.collect(conn, pol, get=lambda u: RSS if u == "u1" else ATOM, now=NOW)
    assert report == {"pub": 2, "other": 1} and {i["url"] for i in new} == {"https://pub.example/a",
                                                                              "https://pub.example/b",
                                                                              "https://other.example/c"}
    assert new[0]["summary"].startswith("Long article")                      # for this run's distill only
    cols = {r["column_name"] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'culture_items'")}
    assert cols == {"url", "feed", "category", "title", "published_at", "fetched_at"}
    again, _ = culture.collect(conn, pol, get=lambda u: RSS if u == "u1" else ATOM, now=NOW)
    assert again == []                                                        # deduped by link


def test_a_paused_source_is_never_read(conn, policy):
    assert policy.get("data_policy", "sources", "feeds", "status") == "paused"   # publisher terms unconfirmed
    with pytest.raises(data_policy.SourcePaused):
        culture.collect(conn, _sources(policy, gdelt="paused"), get=lambda u: RSS, now=NOW)


GDELT_JSON = (b'{"articles": [{"url": "https://www.elle.com/a", "title": "Swim caps are the hat of the season", '
              b'"seendate": "20261124T201500Z", "domain": "elle.com"}, {"url": "https://www.harpersbazaar.com/b", '
              b'"title": "Spring 2027 trends", "seendate": "20261124T024500Z", "domain": "harpersbazaar.com"}]}')


def test_gdelt_headlines_are_read_politely_and_keep_their_publication(conn, policy):
    asked, slept = [], []
    new, report = culture.collect(conn, policy, get=lambda u: asked.append(u) or GDELT_JSON, now=NOW,
                                  sleep=slept.append)
    queries = policy.get("intelligence", "culture", "gdelt", "queries")
    assert len(asked) == len(queries) and all("api.gdeltproject.org" in u for u in asked)
    assert len(slept) == len(queries) - 1 and set(slept) == {policy.get("intelligence", "culture", "gdelt",
                                                                          "min_interval_s")}
    assert {i["feed"] for i in new} == {"elle.com", "harpersbazaar.com"}     # the publication is the source
    assert len(new) == 2                                                      # deduped across queries
    with pytest.raises(RuntimeError):
        culture.parse_gdelt(b"Please limit requests to one every 5 seconds")


class LLM:
    def __init__(self, signals, model="claude-haiku-5-5"):
        self.signals, self.model, self.calls = signals, model, []

    def model_for(self, role):
        return self.model

    def structured(self, role, *, system, content, schema, effort="low", creator_id=None):
        self.calls.append(content)
        return schema.model_validate({"signals": self.signals})


def _items():
    return [{"url": f"https://p{k}.example/x", "title": t, "summary": "s", "feed": f, "category": c,
             "published_at": NOW}
            for k, (t, f, c) in enumerate([("Butter yellow on runways", "vogue", "fashion"),
                                           ("Butter yellow kitchens", "ad", "interiors"),
                                           ("A hotel opens", "cnt", "travel")])]


def test_signals_get_their_kind_from_code_and_reach_the_agents(conn, policy):
    sig = [{"category": "fashion", "signal": "Butter yellow is the colour of the season across clothes and rooms",
            "items": [1, 2], "confidence": 0.8,
            "implications": {"style": "a butter-yellow knit for her capsule", "casting": None,
                             "commerce": "homeware and knitwear", "content": "a colour-story carousel"}},
           {"category": "travel", "signal": "Small quiet hotels over resorts", "items": [3], "confidence": 0.9,
            "implications": {"style": None, "casting": "a slow-travel personality", "commerce": None,
                             "content": "a hotel morning"}}]
    made = culture.distill(conn, policy, LLM(sig), _items())
    assert [(m["kind"], m["n_sources"]) for m in made] == [("corroborated", 2), ("single_source", 1)]
    conf = conn.execute("SELECT confidence FROM culture_signals WHERE kind = 'single_source'").fetchone()["confidence"]
    assert float(conf) <= policy.get("intelligence", "culture", "single_source_confidence_max")
    style = culture.brief(conn, policy, use="style")
    assert "butter-yellow knit" in style and "corroborated, 2 publication(s)" in style and "hotel" not in style
    assert "slow-travel" in culture.brief(conn, policy, use="casting")


def test_no_research_text_to_an_unapproved_host(conn, policy):
    assert culture.distill(conn, policy, LLM([], model="glm-5.3"), _items()) == []


def test_the_culture_run_is_queued_once_a_day_while_paused_with_leave(conn, raw_policy):
    assert raw_policy.get("intelligence", "culture", "enabled") is True
    job = culture.ensure_due(conn, raw_policy, now=NOW)
    row = conn.execute("SELECT payload, batch FROM jobs WHERE id = %s", (job,)).fetchone()
    assert row["payload"]["mode"] == "culture" and row["payload"]["test"] and row["batch"].startswith("research-")
    culture.ensure_due(conn, raw_policy, now=NOW)
    assert conn.execute("SELECT count(*) n FROM jobs").fetchone()["n"] == 1


def test_the_research_budget_is_the_owners_dollar_a_day(raw_policy):
    total = raw_policy.get("intelligence", "culture", "daily_usd") + raw_policy.get("intelligence", "monitor",
                                                                                     "daily_usd")
    assert total == pytest.approx(1.0)
