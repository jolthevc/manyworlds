"""The research desk (owner 2026-11-25; manyworlds.research.desk, llm.web_research): shared research on Claude's web
search and fetch; findings reused while fresh; claims classified with a code check; budget-bound; read by agents."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace as NS

import pytest

from manyworlds.llm import LLM
from manyworlds.research import desk


def _usage(searches=2, fetches=0):
    return NS(input_tokens=2000, output_tokens=300, cache_read_input_tokens=0, cache_creation_input_tokens=0,
              server_tool_use=NS(web_search_requests=searches, web_fetch_requests=fetches))


def _cite(url, title):
    return NS(url=url, title=title, cited_text="x" * 400, type="web_search_result_location")


class WebClient:
    """Stands in for the Anthropic client: one paused turn, then the answer with citations."""
    def __init__(self):
        self.calls = []
        outer = self

        class messages:
            @staticmethod
            def create(**kw):
                outer.calls.append(kw)
                if len(outer.calls) == 1:
                    return NS(model=kw["model"], usage=_usage(1), stop_reason="pause_turn", stop_details=None,
                              content=[NS(type="server_tool_use")])
                return NS(model=kw["model"], usage=_usage(2), stop_reason="end_turn", stop_details=None, content=[
                    NS(type="text", text="Butter yellow is everywhere. ", citations=[
                        _cite("https://www.vogue.com/a", "Vogue"), _cite("https://www.elle.com/b", "Elle")]),
                    NS(type="text", text="Some say it peaks in spring.", citations=[
                        _cite("https://blog.example/c", "A blog")])])
        self.messages = messages


def test_web_research_follows_pause_turn_charges_searches_and_keeps_only_sources(conn, policy):
    client = WebClient()
    llm = LLM(conn, policy, client=client)
    out = llm.web_research("research_quick", system="s", question="what's moving?", max_searches=3)
    assert len(client.calls) == 2 and client.calls[1]["messages"][-1]["role"] == "assistant"
    tools = client.calls[0]["tools"]
    assert tools[0]["name"] == "web_search" and tools[0]["max_uses"] == 3 and len(tools) == 1   # no fetch on quick
    assert tools[0]["type"] == policy.get("llm", "web", "search_tool")
    assert out["searches"] == 3 and out["text"].startswith("Butter yellow")
    assert [s["url"] for s in out["sources"]] == ["https://www.vogue.com/a", "https://www.elle.com/b",
                                                  "https://blog.example/c"]
    assert all(len(s["cited_text"]) <= 150 for s in out["sources"])
    fee = conn.execute("SELECT sum(usd) u FROM cost_entries WHERE unit = 'search'").fetchone()["u"]
    assert float(fee) == pytest.approx(3 * policy.get("llm", "web", "usd_per_search"))


def test_a_verified_fact_needs_two_independent_sources():
    sources = [{"url": "https://www.vogue.com/a"}, {"url": "https://vogue.com/b"}, {"url": "https://elle.com/c"}]
    claims = [desk.Claim(text="one outlet twice", kind="verified_fact", sources=[1, 2]),
              desk.Claim(text="two outlets", kind="verified_fact", sources=[1, 3]),
              desk.Claim(text="a taste", kind="opinion", sources=[9])]
    out = desk.classify(claims, sources)
    assert [c["kind"] for c in out] == ["reported_claim", "verified_fact", "opinion"]
    assert out[2]["sources"] == []                                              # unknown source numbers dropped


class DeskLLM:
    def __init__(self, conn):
        self.conn, self.web_calls = conn, 0

    def tagged(self, agent, job_id):
        import contextlib
        return contextlib.nullcontext()

    def model_for(self, role):
        return "claude-haiku-5-5"

    def web_research(self, role, **kw):
        self.web_calls += 1
        return {"text": "answer", "model": "claude-haiku-5-5", "searches": 2, "fetches": 0,
                "sources": [{"url": "https://www.vogue.com/a", "title": "Vogue", "cited_text": "c"},
                            {"url": "https://elle.com/b", "title": "Elle", "cited_text": "d"}]}

    def structured(self, role, *, system, content, schema, effort="low", creator_id=None):
        return schema.model_validate({"summary": "Butter yellow leads", "claims": [
            {"text": "butter yellow across collections", "kind": "verified_fact", "sources": [1, 2],
             "confidence": 0.8}], "implications": {"style": "a butter-yellow knit", "casting": "",
                                                   "commerce": "knitwear", "content": ""}})


def test_a_fresh_finding_is_reused_across_agents_and_reaches_the_brief(conn, policy):
    llm = DeskLLM(conn)
    first = desk.ask(conn, policy, llm, "What is moving in fashion?", requester="style_director", tags=["fashion"])
    again = desk.ask(conn, policy, llm, "what is moving in FASHION", requester="casting")
    assert not first["reused"] and again["reused"] and llm.web_calls == 1
    assert first["claims"][0]["kind"] == "verified_fact" and first["sources"][0]["url"].startswith("https://")
    style = desk.brief(conn, policy, use="style")
    assert "butter-yellow knit" in style and "1 verified fact" in style
    assert desk.brief(conn, policy, use="casting") == ""                       # nothing to say for casting


def test_the_desk_stops_at_its_daily_budget(conn, policy):
    from manyworlds import ledger
    ledger.record(conn, category="llm", provider="x", quantity=1, unit="token",
                  usd=policy.get("intelligence", "desk", "daily_usd"), agent="research:desk")
    out = desk.ask(conn, policy, DeskLLM(conn), "anything new?")
    assert out == {"skipped": "desk daily budget reached"}


def test_a_paused_source_or_a_disabled_desk_does_nothing(conn, policy):
    body = dict(policy.body)
    body["data_policy"] = {**body["data_policy"], "sources": {**body["data_policy"]["sources"],
                                                              "claude_web": {"status": "paused", "why": "review"}}}
    assert "review" in desk.ask(conn, dataclasses.replace(policy, body=body), DeskLLM(conn), "q")["skipped"]


def test_watched_topics_are_asked_once_per_cadence(conn, policy):
    llm = DeskLLM(conn)
    n = len(policy.get("intelligence", "desk", "topics"))
    out = desk.run_topics(conn, policy, llm)
    assert len(out["asked"]) + len(out["skipped"]) == n
    asked = llm.web_calls
    desk.run_topics(conn, policy, llm)
    assert llm.web_calls == asked                                              # nothing due again yet


def test_a_question_that_found_no_sources_is_not_stored_and_is_asked_again(conn, policy):
    class Failing(DeskLLM):
        def web_research(self, role, **kw):
            self.web_calls += 1
            return {"text": "I could not search.", "model": "claude-haiku-5-5", "searches": 0, "fetches": 0,
                    "sources": []}
    llm = Failing(conn)
    assert desk.ask(conn, policy, llm, "anything new?")["skipped"] == "research failed: no sources"
    assert conn.execute("SELECT count(*) n FROM research_findings").fetchone()["n"] == 0
    desk.ask(conn, policy, llm, "anything new?")
    assert llm.web_calls == 2
