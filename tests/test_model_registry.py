"""Models are replaceable resources behind tasks (owner 2026-11-27). Every model a role, fallback or shadow test
names is in the registry with its licence, data terms and status; every task the code names is declared with a
parent role; a task routes as its parent until it is given its own model, and then only that task moves."""

from __future__ import annotations

import re
from pathlib import Path

from manyworlds import llm as llm_mod
from manyworlds import policy as policy_mod
from manyworlds.creator import dossier

ROOT = Path(__file__).resolve().parents[1]


def _models_named(raw_policy) -> set[str]:
    out = set()
    for spec in (raw_policy.get("llm", "roles") or {}).values():
        if isinstance(spec, dict):
            out |= {str(spec["model"])} | ({str(spec["fallback"])} if spec.get("fallback") else set())
        else:
            out.add(str(spec))
    for cfg in (raw_policy.get("llm", "shadow") or {}).values():
        out |= {str(c) for c in (cfg.get("candidates") or [cfg.get("candidate")]) if c}
    return out


def test_every_model_in_use_is_registered_with_licence_terms_and_status(raw_policy):
    reg = raw_policy.get("llm", "registry") or {}
    missing = sorted(m for m in _models_named(raw_policy) if m not in reg)
    assert missing == [], f"add these models to llm.registry: {missing}"
    for name, e in reg.items():
        assert e.get("licence") and e.get("data_terms") and e.get("family"), name
        assert e.get("status") in ("approved", "candidate", "retired"), name
        for ev in e.get("evidence") or []:
            assert ev.get("kind") in ("vendor", "benchmark", "community", "own") and ev.get("claim"), name
    holders = {str(s["model"] if isinstance(s, dict) else s) for s in (raw_policy.get("llm", "roles") or {}).values()}
    assert not [m for m in holders if reg[m]["status"] != "approved"], "a role's primary model must be approved"


def test_every_task_the_code_names_is_declared_with_a_parent_role(raw_policy):
    tasks = raw_policy.get("llm", "tasks") or {}
    roles = raw_policy.get("llm", "roles") or {}
    assert all(parent in roles for parent in tasks.values())
    named = set()
    for f in (ROOT / "manyworlds").rglob("*.py"):
        named |= set(re.findall(r'as_task\([^,]+,\s*"([a-z0-9_]+)"\)', f.read_text(encoding="utf-8")))
    rubrics = {p.stem.rsplit("-", 1)[0] for p in (ROOT / "policy" / "rubrics").glob("*-v*.yaml")}
    named |= {f"evaluate_{r}" for r in rubrics} | {f"dossier_{s}" for s in dossier.SCHEMAS}
    named |= {f"{kind}_{raw_policy.get('review', part, key)}" for kind, part in (("image", "image"), ("clip", "video"))
              for key in ("role", "confirm_role")}
    assert sorted(named - set(tasks)) == []


def test_a_task_routes_as_its_role_until_it_has_its_own_model(policy):
    llm = llm_mod.Claude(None, policy)
    director = llm.route("director")
    with llm_mod.as_task(llm, "director_edit"):
        assert llm.route("director") == director                    # unconfigured: exactly as before
    moved = policy_mod.from_dict({**policy.body, "llm": {**policy.body["llm"], "roles": {
        **policy.body["llm"]["roles"], "director_edit": {"model": "glm-5.3", "fallback": "claude-sonnet-5-5"}}}})
    llm = llm_mod.Claude(None, moved)
    with llm_mod.as_task(llm, "director_edit"):
        assert llm.route("director") == ("glm-5.3", "claude-sonnet-5-5")
    with llm_mod.as_task(llm, "director_shots"):
        assert llm.route("director") == director                    # only that task moved
    assert llm.route("director") == director
    with llm_mod.as_task(object(), "director_edit"):                 # a stub without task routing is left alone
        pass


def test_an_open_model_gets_its_own_system_prefix_and_sampling(conn, policy):
    seen = {}

    def post(url, body, key, timeout):
        seen.update(body)
        return {"choices": [{"message": {"content": '{"answer": "ok"}'}}], "usage": {"prompt_tokens": 1,
                                                                                    "completion_tokens": 1}}
    from pydantic import BaseModel

    class Out(BaseModel):
        answer: str

    om = policy.body["llm"]["open_models"]
    tuned = policy_mod.from_dict({**policy.body, "llm": {**policy.body["llm"], "open_models": {
        **om, "glm-5.3": {**om["glm-5.3"], "system_prefix": "Write like a person, not an assistant.",
                          "temperature": 0.8, "min_p": 0.05}}}})
    import os
    os.environ.setdefault(tuned.get("llm", "hosts", om["glm-5.3"]["host"])["key_env"], "test-key")
    llm = llm_mod.Claude(conn, tuned, post=post)
    llm._open("glm-5.3", system="You are the showrunner.", content="hi", schema=Out, creator_id=None)
    assert seen["messages"][0]["content"].startswith("Write like a person") and "showrunner" in seen["messages"][0]["content"]
    assert seen["temperature"] == 0.8 and seen["min_p"] == 0.05


def test_json_mode_prompt_sends_the_schema_in_the_prompt_and_no_grammar(conn, policy):
    seen = {}

    def post(url, body, key, timeout):
        seen.update(body)
        return {"choices": [{"message": {"content": 'Here it is: {"answer": "ok"}',
                                         "reasoning_content": "thinking {not json}"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    from pydantic import BaseModel

    class Out(BaseModel):
        answer: str

    import os
    assert policy.get("llm", "open_models", "glm-5.3", "json_mode") == "prompt"
    os.environ.setdefault(policy.get("llm", "hosts", "together")["key_env"], "test-key")
    got = llm_mod.Claude(conn, policy, post=post)._open("glm-5.3", system="s", content="write it", schema=Out,
                                                         creator_id=None)
    assert got.answer == "ok" and "response_format" not in seen
    assert "JSON schema" in seen["messages"][1]["content"] and "write it" in seen["messages"][1]["content"]


def test_a_split_sends_a_share_of_a_tasks_real_calls_to_a_candidate_and_the_ledger_says_so(conn, policy):
    from pydantic import BaseModel

    class Out(BaseModel):
        answer: str

    def post(url, body, key, timeout):
        assert body.get("reasoning") == {"effort": "low"}               # the model's extra request parameters
        return {"choices": [{"message": {"content": '{"answer": "from glm"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    import os
    os.environ.setdefault(policy.get("llm", "hosts", "together")["key_env"], "test-key")
    llm_body = policy.body["llm"]
    om = llm_body["open_models"]
    split = policy_mod.from_dict({**policy.body, "llm": {**llm_body, "splits": {"director_edit": {
        "candidate": "glm-5.3", "share": 1.0}}, "open_models": {**om, "glm-5.3": {**om["glm-5.3"],
                                                                                "extra_body": {"reasoning": {"effort": "low"}}}}}})
    llm = llm_mod.Claude(conn, split, post=post)
    with llm.tagged("director", None), llm_mod.as_task(llm, "director_edit"):
        got = llm.structured("director", system="s", content="c", schema=Out)
    assert got.answer == "from glm"
    row = conn.execute("SELECT provider, task, route FROM cost_entries WHERE agent = 'director' "
                       "ORDER BY id DESC LIMIT 1").fetchone()
    assert (row["provider"], row["task"], row["route"]) == ("glm-5.3", "director_edit", "split")
    assert llm.model_for("director") == llm_mod.Claude(conn, policy).model_for("director")   # routing unchanged
