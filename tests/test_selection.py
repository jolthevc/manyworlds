"""Model selection: the cheapest candidate that does the job is proposed to the owner, and an approved switch
applies to every service's model router (manyworlds/selection.py)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from manyworlds import policy as policy_mod, selection
from manyworlds.brain import overrides
from manyworlds.deploy import autoscale
from manyworlds.llm import LLM

from brain_helpers import creator


def _shadow(conn, role, candidate, n, *, primary="claude-sonnet-5-5", p_out=None, c_out=None, p_usd=0.02,
            c_usd=0.001, judged=None, creator_id=None):
    for _ in range(n):
        conn.execute(
            "INSERT INTO llm_shadow (role, primary_model, candidate_model, primary_output, candidate_output, "
            "primary_usd, candidate_usd, judged, creator_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (role, primary, candidate, Jsonb(p_out or {"novelty": {"score": 6}, "fit": True}),
             Jsonb(c_out or {"novelty": {"score": 6}, "fit": True}), p_usd, c_usd,
             Jsonb(judged) if judged else None, creator_id))


def test_the_cheapest_candidate_that_agrees_is_proposed_and_an_approval_switches_the_role(conn, policy):
    need = policy.get("llm", "selection", "roles", "evaluator", "min_calls")
    _shadow(conn, "evaluator", "claude-haiku-5-5", need)
    _shadow(conn, "evaluator", "nemotron-3.5-vl", need, c_usd=0.0005, c_out={"novelty": {"score": 2}, "fit": False})
    rec = selection.recommend(conn, policy, "evaluator")
    assert rec["pick"]["candidate"] == "claude-haiku-5-5"            # nemotron is cheaper but disagrees
    nemo = next(c for c in rec["candidates"] if c["candidate"] == "nemotron-3.5-vl")
    assert not nemo["qualifies"] and nemo["why_not"]
    out = selection.propose(conn, policy, "evaluator", rec)
    assert out["proposed"] and "approval_id" in out
    assert not selection.propose(conn, policy, "evaluator", rec)["proposed"]      # one pending proposal at a time
    llm = LLM(conn, policy)
    assert llm.model_for("evaluator") == policy.get("llm", "roles", "evaluator")
    overrides.decide(conn, out["approval_id"], approved=True, by="owner")
    assert LLM(conn, policy).model_for("evaluator") == "claude-haiku-5-5"      # every service, not only the Brain


def test_an_open_model_switch_keeps_the_old_model_as_its_fallback(conn, policy):
    _shadow(conn, "evaluator", "nemotron-3.5-vl", policy.get("llm", "selection", "roles", "evaluator", "min_calls"))
    out = selection.propose(conn, policy, "evaluator", selection.recommend(conn, policy, "evaluator"))
    overrides.decide(conn, out["approval_id"], approved=True, by="owner")
    assert LLM(conn, policy).route("evaluator") == ("nemotron-3.5-vl", policy.get("llm", "roles", "evaluator"))


def test_a_rejected_switch_is_not_proposed_again_during_the_cooldown(conn, policy):
    _shadow(conn, "evaluator", "claude-haiku-5-5", policy.get("llm", "selection", "roles", "evaluator", "min_calls"))
    rec = selection.recommend(conn, policy, "evaluator")
    out = selection.propose(conn, policy, "evaluator", rec)
    overrides.decide(conn, out["approval_id"], approved=False, by="owner")
    assert selection.propose(conn, policy, "evaluator", rec)["reason"] == "already rejected"
    later = datetime.now(timezone.utc) + timedelta(days=policy.get("llm", "selection", "rejected_cooldown_days") + 1)
    assert selection.propose(conn, policy, "evaluator", rec, now=later)["proposed"]


def test_too_few_calls_or_too_small_a_saving_proposes_nothing(conn, policy):
    _shadow(conn, "evaluator", "claude-haiku-5-5", 3)
    _shadow(conn, "evaluator", "qwen4-max", 40, c_usd=0.019)                     # agrees, but barely cheaper
    rec = selection.recommend(conn, policy, "evaluator")
    assert rec["pick"] is None
    assert all(c["why_not"] for c in rec["candidates"])


def test_a_generative_candidate_qualifies_on_judged_quality_not_on_agreement(conn, policy):
    rc = policy.get("llm", "selection", "roles", "showrunner")
    cid = creator(conn)
    as_good = {"primary": {"score": 6.0, "rule_breaks": 0.1}, "candidate": {"score": 5.9, "rule_breaks": 0.0}}
    worse = {"primary": {"score": 6.0, "rule_breaks": 0.1}, "candidate": {"score": 5.0, "rule_breaks": 0.3}}
    different = {"ideas": [{"caption": "totally different idea"}]}
    n = max(rc["min_judged"], rc["min_calls"])
    _shadow(conn, "showrunner", "glm-5.3", n, primary="claude-opus-5-5", c_out=different,
            p_out={"ideas": [{"caption": "an idea"}]}, judged=as_good, creator_id=cid)
    _shadow(conn, "showrunner", "glm-5.3-flash", n, primary="claude-opus-5-5", c_out=different,
            p_out={"ideas": [{"caption": "an idea"}]}, judged=worse, c_usd=0.0001, creator_id=cid)
    rec = selection.recommend(conn, policy, "showrunner")
    assert rec["pick"]["candidate"] == "glm-5.3"
    flash = next(c for c in rec["candidates"] if c["candidate"] == "glm-5.3-flash")
    assert any("scores" in w for w in flash["why_not"]) and any("voice rules" in w for w in flash["why_not"])


def test_selection_runs_once_a_day_on_the_always_on_loop_and_spends_nothing(conn, policy):
    _shadow(conn, "evaluator", "claude-haiku-5-5", policy.get("llm", "selection", "roles", "evaluator", "min_calls"))
    now = datetime.now(timezone.utc)
    assert autoscale.model_selection(conn, policy, now=now) == [{"action": "model_selection", "proposed": ["evaluator"]}]
    assert autoscale.model_selection(conn, policy, now=now + timedelta(hours=1)) == []
    assert conn.execute("SELECT count(*) n FROM cost_entries").fetchone()["n"] == 0


class _Scorer:
    """An Anthropic client whose evaluator scores every dimension `value`."""

    def __init__(self, value):
        self.value, self.calls = value, 0
        usage = type("U", (), {"input_tokens": 1000, "output_tokens": 50, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 0})
        outer = self

        class messages:
            @staticmethod
            def parse(**kw):
                outer.calls += 1
                schema = kw["output_format"]
                out = schema(**{k: {"score": outer.value, "why": "x"} for k in schema.model_fields if k != "overall_note"})
                return type("R", (), {"model": kw["model"], "usage": usage, "stop_reason": "end_turn",
                                      "parsed_output": out})
        self.messages = messages


def test_judging_scores_both_answers_blind_within_its_budget(conn, policy):
    cid = creator(conn)
    idea = {"format": "photo", "scenes": ["a cafe"], "caption": "the good croissant, out of spite"}
    _shadow(conn, "showrunner", "glm-5.3", 3, primary="claude-opus-5-5", creator_id=cid,
            p_out={"ideas": [idea]}, c_out={"ideas": [{**idea, "caption": "embrace the journey"}]})
    client = _Scorer(6)
    llm = LLM(conn, policy, client=client)
    out = selection.judge(conn, policy, llm, "showrunner", usd=1.0)
    assert out["judged"] == 3 and client.calls == 6
    j = conn.execute("SELECT judged FROM llm_shadow WHERE judged IS NOT NULL LIMIT 1").fetchone()["judged"]
    assert j["primary"]["score"] == j["candidate"]["score"] == 6
    assert j["primary"]["rule_breaks"] == 0 and j["candidate"]["rule_breaks"] == 1     # "embrace", "journey"
    agents = {r["agent"] for r in conn.execute("SELECT agent FROM cost_entries")}
    assert agents == {"shadow"} and llm.shadowing                                    # charged as shadow, restored
    assert selection.judge(conn, policy, LLM(conn, policy, client=_Scorer(6)), "showrunner", usd=0.0)["judged"] == 0


class _Preferer:
    """An Anthropic client whose judge always prefers the answer containing `mark`."""

    def __init__(self, mark):
        self.mark, self.prompts = mark, []
        usage = type("U", (), {"input_tokens": 2000, "output_tokens": 80, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 0})
        outer = self

        class messages:
            @staticmethod
            def parse(**kw):
                text = kw["messages"][0]["content"]
                outer.prompts.append(text)
                a = text.split("=== Answer A ===")[1].split("=== Answer B ===")[0]
                better = "A" if outer.mark in a else "B"
                return type("R", (), {"model": kw["model"], "usage": usage, "stop_reason": "end_turn",
                                      "parsed_output": kw["output_format"](reason="r", better=better)})
        self.messages = messages


def _requested(conn):
    conn.execute("UPDATE llm_shadow SET request = 'write the shot sheet for a cafe photo'")


def test_an_open_ended_role_is_judged_blind_in_both_orders(conn, policy):
    _shadow(conn, "talent_manager", "qwen4-max", 6, p_out={"brief": "PRIMARY plan"}, c_out={"brief": "cand plan"},
            p_usd=0.03, c_usd=0.006)
    _requested(conn)
    client = _Preferer("PRIMARY")
    out = selection.judge(conn, policy, LLM(conn, policy, client=client), "talent_manager", usd=5.0)
    assert out["judged"] == 6
    firsts = {p.split("=== Answer A ===")[1][:40] for p in client.prompts}
    assert len(firsts) == 2                                   # the primary is shown first in some, second in others
    assert all("write the shot sheet" in p for p in client.prompts)
    prefs = {r["judged"]["preference"] for r in conn.execute("SELECT judged FROM llm_shadow").fetchall()}
    assert prefs == {"primary"}                               # unscrambled back to the right side every time


def test_a_preference_role_switches_only_when_the_candidate_is_rarely_worse(conn, policy):
    rc = policy.get("llm", "selection", "roles", "talent_manager")
    n = max(rc["min_calls"], rc["min_judged"])
    _shadow(conn, "talent_manager", "qwen4-max", n, p_usd=0.03, c_usd=0.006, judged={"preference": "primary"})
    rec = selection.recommend(conn, policy, "talent_manager")
    assert rec["pick"] is None and "no worse" in " ".join(rec["candidates"][0]["why_not"])
    conn.execute("TRUNCATE llm_shadow")
    _shadow(conn, "talent_manager", "qwen4-max", n - 1, p_usd=0.03, c_usd=0.006, judged={"preference": "tie"})
    _shadow(conn, "talent_manager", "qwen4-max", 1, p_usd=0.03, c_usd=0.006, judged={"preference": "primary"})
    pick = selection.recommend(conn, policy, "talent_manager")["pick"]
    assert pick["candidate"] == "qwen4-max" and pick["not_worse"] >= rc["min_not_worse"]


def test_shadow_rows_keep_the_request_for_judging(conn, policy):
    from pydantic import BaseModel

    class Out(BaseModel):
        brief: str = "x"

    pol = policy_mod.from_dict({**policy.body, "llm": {**policy.body["llm"], "shadow": {
        "talent_manager": {"candidates": ["claude-haiku-5-5"], "share": 1.0}}}})
    usage = type("U", (), {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 0,
                           "cache_creation_input_tokens": 0})

    class C:
        class messages:
            @staticmethod
            def parse(**kw):
                return type("R", (), {"model": kw["model"], "usage": usage, "stop_reason": "end_turn",
                                      "parsed_output": kw["output_format"]()})
    LLM(conn, pol, client=C()).structured("talent_manager", system="be her manager", content="her week", schema=Out)
    req = conn.execute("SELECT request FROM llm_shadow").fetchone()["request"]
    assert "be her manager" in req and "her week" in req
