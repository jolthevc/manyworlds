"""Roles route to Claude or to open-weight models; failures fall back; candidates are shadow-tested (llm.py)."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from manyworlds import policy as policy_mod, shadow
from manyworlds.llm import LLM, OpenModelError, image_block


class Verdict(BaseModel):
    score: float
    passed: bool
    label: str
    note: str = ""


class FakeAnthropic:
    def __init__(self, answer=None):
        self.calls = []
        usage = type("U", (), {"input_tokens": 1000, "output_tokens": 100,
                               "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})
        outer = self

        class messages:
            @staticmethod
            def parse(**kw):
                outer.calls.append(kw)
                out = answer or kw["output_format"](score=7, passed=True, label="cozy")
                return type("R", (), {"model": kw["model"], "usage": usage, "stop_reason": "end_turn",
                                      "parsed_output": out})
        self.messages = messages


class FakeHost:
    """An OpenAI-compatible host: answers with the queued replies, in order."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def __call__(self, url, body, key, timeout):
        self.requests.append({"url": url, "body": body, "key": key})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return {"choices": [{"message": {"content": r if isinstance(r, str) else json.dumps(r)}}],
                "usage": {"prompt_tokens": 2000, "completion_tokens": 500}}


@pytest.fixture()
def open_policy(policy, monkeypatch):
    monkeypatch.setenv("TEST_HOST_KEY", "k-123456789")
    llm = {**policy.body["llm"],
           "hosts": {"testhost": {"base_url": "https://open.example/v1", "key_env": "TEST_HOST_KEY"}},
           "open_models": {"open-small": {"host": "testhost", "id": "org/open-small", "usd_per_mtok": [0.1, 0.4]},
                           "open-vision": {"host": "testhost", "id": "org/open-vision", "usd_per_mtok": [0.2, 0.8],
                                           "vision": True}},
           "roles": {**policy.body["llm"]["roles"],
                     "evaluator": {"model": "open-small", "fallback": "claude-sonnet-5-5"},
                     "review": "open-vision"}}
    return policy_mod.from_dict({**policy.body, "llm": llm})


def _usd(conn, provider):
    return float(conn.execute("SELECT coalesce(sum(usd), 0) AS u FROM cost_entries WHERE provider = %s",
                              (provider,)).fetchone()["u"])


def test_a_role_on_an_open_model_is_answered_and_charged(conn, open_policy):
    host = FakeHost({"score": 6.5, "passed": True, "label": "cozy"})
    llm = LLM(conn, open_policy, client=FakeAnthropic(), post=host)
    out = llm.structured("evaluator", system="judge", content="an idea", schema=Verdict)
    assert out == Verdict(score=6.5, passed=True, label="cozy")
    req = host.requests[0]
    assert req["url"] == "https://open.example/v1/chat/completions" and req["key"] == "k-123456789"
    assert req["body"]["model"] == "org/open-small" and "k-123456789" not in json.dumps(req["body"])
    assert req["body"]["response_format"]["json_schema"]["name"] == "Verdict"
    assert _usd(conn, "open-small") == pytest.approx((2000 * 0.1 + 500 * 0.4) / 1e6)
    assert llm.model_for("evaluator") == "open-small"


def test_invalid_output_is_retried_then_falls_back_to_claude(conn, open_policy):
    host = FakeHost("not json", '{"score": "high"}')
    claude = FakeAnthropic()
    llm = LLM(conn, open_policy, client=claude, post=host)
    out = llm.structured("evaluator", system="judge", content="an idea", schema=Verdict)
    assert out.label == "cozy" and len(host.requests) == 2 and claude.calls[0]["model"] == "claude-sonnet-5-5"
    assert _usd(conn, "open-small") > 0          # failed attempts are charged too


def test_a_host_error_falls_back(conn, open_policy):
    llm = LLM(conn, open_policy, client=FakeAnthropic(), post=FakeHost(OpenModelError("503")))
    assert llm.structured("evaluator", system="s", content="c", schema=Verdict).label == "cozy"


def test_no_fallback_means_the_error_surfaces(conn, open_policy):
    llm = LLM(conn, open_policy, client=FakeAnthropic(), post=FakeHost(OpenModelError("503")))
    with pytest.raises(OpenModelError):
        llm.structured("review", system="s", content="c", schema=Verdict)


def test_a_missing_key_is_an_open_model_error(conn, open_policy, monkeypatch):
    monkeypatch.delenv("TEST_HOST_KEY")
    llm = LLM(conn, open_policy, client=FakeAnthropic(), post=FakeHost())
    assert llm.structured("evaluator", system="s", content="c", schema=Verdict).label == "cozy"   # fell back


def test_images_go_as_data_urls_to_vision_models_only(conn, open_policy):
    host = FakeHost({"score": 8, "passed": True, "label": "ok"})
    llm = LLM(conn, open_policy, client=FakeAnthropic(), post=host)
    content = [image_block(b"\x89PNG", "image/png"), {"type": "text", "text": "is it on brand?"}]
    llm.structured("review", system="s", content=content, schema=Verdict)
    parts = host.requests[0]["body"]["messages"][1]["content"]
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,") and parts[1]["text"] == "is it on brand?"
    blind = policy_mod.from_dict({**open_policy.body, "llm": {
        **open_policy.body["llm"], "roles": {**open_policy.body["llm"]["roles"], "review": "open-small"}}})
    with pytest.raises(OpenModelError, match="not a vision model"):
        LLM(conn, blind, post=FakeHost()).structured("review", system="s", content=content, schema=Verdict)


def test_fenced_json_is_accepted(conn, open_policy):
    host = FakeHost('```json\n{"score": 5, "passed": false, "label": "meh"}\n```')
    llm = LLM(conn, open_policy, client=FakeAnthropic(), post=host)
    assert llm.structured("evaluator", system="s", content="c", schema=Verdict).label == "meh"


def test_claude_roles_are_unchanged(conn, policy):
    claude = FakeAnthropic()
    llm = LLM(conn, policy, client=claude)
    llm.structured("brain", system="s", content="c", schema=Verdict)
    assert claude.calls[0]["model"] == "claude-opus-5-5" and claude.calls[0]["max_tokens"] == 16000


def _shadowed(open_policy, share=1.0, cap=1.0):
    llm = {**open_policy.body["llm"], "roles": {**open_policy.body["llm"]["roles"], "evaluator": "claude-sonnet-5-5"},
           "shadow": {"evaluator": {"candidate": "open-small", "share": share}}, "shadow_usd_per_day": cap}
    return policy_mod.from_dict({**open_policy.body, "llm": llm})


def test_a_shadow_candidate_is_recorded_but_never_changes_the_answer(conn, open_policy):
    host = FakeHost({"score": 6.0, "passed": False, "label": "Cozy"})
    llm = LLM(conn, _shadowed(open_policy), client=FakeAnthropic(), post=host)
    with llm.tagged("evaluator", None):
        out = llm.structured("evaluator", system="s", content="c", schema=Verdict)
    assert out.score == 7 and out.passed is True                        # Claude's answer
    row = conn.execute("SELECT * FROM llm_shadow").fetchone()
    assert row["primary_model"] == "claude-sonnet-5-5" and row["candidate_output"]["score"] == 6.0
    assert row["agent"] == "evaluator" and float(row["candidate_usd"]) > 0
    shadow_agent = conn.execute("SELECT agent FROM cost_entries WHERE provider = 'open-small'").fetchone()["agent"]
    assert shadow_agent == "shadow"                                     # outside the evaluator's budget
    [r] = shadow.report(conn, "evaluator")
    assert r["valid_share"] == 1.0 and r["fields"]["score"]["mean_abs_diff"] == 1.0
    assert r["fields"]["passed"]["agreement"] == 0.0 and r["fields"]["label"]["agreement"] == 1.0
    assert "note" in r["fields"] and r["cost_ratio"] is not None


def test_a_failing_candidate_is_recorded_as_an_error(conn, open_policy):
    llm = LLM(conn, _shadowed(open_policy), client=FakeAnthropic(), post=FakeHost(OpenModelError("host down")))
    assert llm.structured("evaluator", system="s", content="c", schema=Verdict).score == 7
    [r] = shadow.report(conn, "evaluator")
    assert r["valid_share"] == 0.0 and "host down" in r["errors"][0]


def test_shadow_spend_is_capped_per_day(conn, open_policy):
    host = FakeHost(*[{"score": 6, "passed": True, "label": "x"}] * 3)
    llm = LLM(conn, _shadowed(open_policy, cap=0.0), client=FakeAnthropic(), post=host)
    llm.structured("evaluator", system="s", content="c", schema=Verdict)
    assert host.requests == [] and conn.execute("SELECT count(*) n FROM llm_shadow").fetchone()["n"] == 0


def test_compare_matches_fields_inside_lists():
    a = {"scores": [{"dim": "hook", "value": 6}, {"dim": "voice", "value": 8}]}
    b = {"scores": [{"dim": "hook", "value": 7}, {"dim": "voice", "value": 8}]}
    c = shadow.compare(a, b)
    assert c["scores[0].value"] == {"kind": "number", "diff": 1.0, "within": True}
    assert c["scores[1].dim"] == {"kind": "label", "agree": True}


def test_a_role_can_shadow_several_candidates_and_never_its_own_model(conn, open_policy):
    llm_cfg = {**open_policy.body["llm"], "roles": {**open_policy.body["llm"]["roles"], "evaluator": "claude-sonnet-5-5"},
               "shadow": {"evaluator": {"candidates": ["open-small", "claude-sonnet-5-5", "open-vision"], "share": 1.0}}}
    pol = policy_mod.from_dict({**open_policy.body, "llm": llm_cfg})
    host = FakeHost(*[{"score": 6, "passed": True, "label": "x"}] * 40)
    llm = LLM(conn, pol, client=FakeAnthropic(), post=host)
    for _ in range(40):
        llm.structured("evaluator", system="s", content="c", schema=Verdict)
    tried = {r["candidate_model"] for r in conn.execute("SELECT candidate_model FROM llm_shadow")}
    assert tried == {"open-small", "open-vision"}


def test_open_model_requests_send_a_user_agent(monkeypatch):
    # Together's firewall answers urllib's default agent with 403 (found on the live host, 2026-11-21).
    import urllib.request

    from manyworlds import llm as llm_mod
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def urlopen(req, timeout):
        seen.update(req.headers)
        return Resp()
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    llm_mod._post_json("https://open.example/v1/chat/completions", {}, "k", 5)
    assert seen.get("User-agent") == llm_mod.USER_AGENT


def test_every_shipped_shadow_candidate_is_a_priced_model(raw_policy):
    llm_cfg = raw_policy.body["llm"]
    for role, cfg in (llm_cfg.get("shadow") or {}).items():
        assert role in llm_cfg["roles"], role
        for c in cfg.get("candidates") or [cfg.get("candidate")]:
            assert c in llm_cfg["open_models"] or c in llm_cfg["usd_per_mtok"], (role, c)
            if role in ("evaluator", "review", "judge") and c in llm_cfg["open_models"]:
                assert llm_cfg["open_models"][c].get("vision"), (role, c)   # these roles send images


def test_a_trial_sends_every_call_to_every_candidate_until_its_budget_is_spent(conn, open_policy):
    llm_cfg = {**open_policy.body["llm"], "roles": {**open_policy.body["llm"]["roles"], "evaluator": "claude-sonnet-5-5"},
               "shadow": {"evaluator": {"candidates": ["open-small", "open-vision"], "share": 0.0}}}
    pol = policy_mod.from_dict({**open_policy.body, "llm": llm_cfg})
    host = FakeHost(*[{"score": 6, "passed": True, "label": "x"}] * 10)
    llm = LLM(conn, pol, client=FakeAnthropic(), post=host)
    llm.trial_usd = 0.002            # each candidate call costs $0.0004 (open-small) or $0.0008 (open-vision)
    for _ in range(4):
        llm.structured("evaluator", system="s", content="c", schema=Verdict)
    rows = [r["candidate_model"] for r in conn.execute("SELECT candidate_model FROM llm_shadow ORDER BY id")]
    assert rows[:2] == ["open-small", "open-vision"]            # both, on the same call, though share is 0
    assert llm.trial_spent >= llm.trial_usd and len(rows) < 8     # stopped once the trial budget was spent


def test_a_streamed_answer_is_read_like_a_whole_one(conn, open_policy):
    # Qwen on Together only answers streamed (found live 2026-11-21).
    from manyworlds import llm as llm_mod
    events = [b'data: {"choices":[{"delta":{"content":"{\\"score\\": 6, "}}]}\n', b"\n",
              b'data: {"choices":[{"delta":{"content":"\\"passed\\": true, \\"label\\": \\"x\\"}"}}]}\n',
              b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20}}\n', b"data: [DONE]\n"]
    r = llm_mod._read_stream(events)
    assert r["usage"]["completion_tokens"] == 20 and Verdict.model_validate_json(r["choices"][0]["message"]["content"])
    llm_cfg = {**open_policy.body["llm"], "open_models": {**open_policy.body["llm"]["open_models"],
               "open-small": {**open_policy.body["llm"]["open_models"]["open-small"], "stream": True}}}
    host = FakeHost({"score": 6, "passed": True, "label": "x"})
    LLM(conn, policy_mod.from_dict({**open_policy.body, "llm": llm_cfg}), client=FakeAnthropic(), post=host) \
        .structured("evaluator", system="s", content="c", schema=Verdict)
    assert host.requests[0]["body"]["stream"] is True and host.requests[0]["body"]["stream_options"]["include_usage"]
