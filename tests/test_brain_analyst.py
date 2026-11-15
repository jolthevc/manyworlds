"""The analyst turns outcomes into evidence-qualified hypotheses and guidance, and the showrunner's
next batch records which of them it used."""

from __future__ import annotations

from manyworlds import queue
from manyworlds.brain import evidence, overrides
from manyworlds.brain.agents import analyst, showrunner

from brain_helpers import StubLLM, make_ctx, sandbox_history


class AnalystLLM(StubLLM):
    def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
        super().structured(role, system=system, content=content, schema=schema)
        if schema is analyst.Analysis:
            return schema.model_validate({
                "hypotheses": [{"statement": "Posts that show more of her personality reach more people.", "direction": "positive",
                                "supports": ["dimension:personality"], "how_to_confirm": "alternate high/low"}],
                "guidance": {"more_of": ["personality-led moments"], "less_of": ["story clips"],
                             "explore": ["day-off outdoor scenes"], "notes": "n is modest; keep a third exploratory"}})
        return schema.model_validate({"rationale": "r", "ideas": [
            {"day": 0, "platform": "instagram", "format": "photo", "scenes": ["s"], "caption": "c", "hashtags": [],
             "canon": "x", "reasoning": "personality posts do well (finding)"}]})


def _run(conn, policy, kind, payload, llm):
    job = queue.enqueue(conn, f"brain.{kind}", payload, creator_id=payload["creator_id"])
    ctx = make_ctx(conn, overrides.effective_policy(conn, policy), job)
    ctx.llm = llm
    agent = analyst.Analyst() if kind == "analyst" else showrunner.Showrunner()
    return agent.run(ctx), job


def test_findings_from_a_sandbox_creator_are_simulated_and_labelled(conn, policy):
    cid = sandbox_history(conn, policy, 90)
    out, job = _run(conn, policy, "analyst", {"creator_id": str(cid)}, AnalystLLM(conn=conn))
    found = evidence.findings(conn, mode="sandbox", creator_id=cid)
    assert found and all(f["label"] == "SIMULATED" and f["evidence"] == "simulated" for f in found)
    assert evidence.findings(conn, mode="real", creator_id=cid) == []
    f = found[0]
    assert f["statement"].startswith("Posts that show more of her personality") and f["grade"] == "tentative"
    assert f["n"] and f["interval_low"] is not None and float(f["effect"]) > 0
    d = conn.execute("SELECT output FROM decisions WHERE job_id = %s AND kind = 'analysis'", (job,)).fetchone()["output"]
    assert d["finding_ids"] == [str(f["id"])] and d["guidance"]["less_of"] == ["story clips"]


def test_insufficient_evidence_records_no_findings_and_asks_to_explore(conn, policy):
    cid = sandbox_history(conn, policy, 5)
    llm = AnalystLLM(conn=conn)
    out, _ = _run(conn, policy, "analyst", {"creator_id": str(cid)}, llm)
    assert out["insufficient"] and llm.calls == [] and evidence.findings(conn, mode="sandbox", creator_id=cid) == []
    assert "explore" in out["guidance"]["notes"].lower()


def test_the_llm_sees_numbers_not_raw_posts(conn, policy):
    cid = sandbox_history(conn, policy, 40)
    llm = AnalystLLM(conn=conn)
    _run(conn, policy, "analyst", {"creator_id": str(cid)}, llm)
    seen = str(llm.calls[0]["content"])
    assert "SECRET CAPTION TEXT" not in seen and "dimension" in seen and "SIMULATED" in seen


def test_a_new_analysis_supersedes_the_last(conn, policy):
    cid = sandbox_history(conn, policy, 90)
    _run(conn, policy, "analyst", {"creator_id": str(cid)}, AnalystLLM(conn=conn))
    _run(conn, policy, "analyst", {"creator_id": str(cid)}, AnalystLLM(conn=conn))
    assert len(evidence.findings(conn, mode="sandbox", creator_id=cid)) == 1


def test_the_next_batch_records_the_findings_and_guidance_it_used(conn, policy):
    from psycopg.types.json import Jsonb
    cid = sandbox_history(conn, policy, 90)
    conn.execute("UPDATE creators SET anchors = %s WHERE id = %s", (Jsonb({"reference_images": [{"uri": "s3://r"}]}), cid))
    _, ajob = _run(conn, policy, "analyst", {"creator_id": str(cid)}, AnalystLLM(conn=conn))
    analysis = conn.execute("SELECT id, output FROM decisions WHERE job_id = %s", (ajob,)).fetchone()
    llm = AnalystLLM(conn=conn)
    _, sjob = _run(conn, policy, "showrunner", {"creator_id": str(cid), "start": "2026-11-20"}, llm)
    prompt = str(llm.calls[0]["content"])
    assert "personality-led moments" in prompt and "Posts that show more of her personality" in prompt
    inputs = conn.execute("SELECT inputs FROM decisions WHERE job_id = %s AND step = 'plan'", (sjob,)).fetchone()["inputs"]
    assert inputs["analysis_decision_id"] == str(analysis["id"]) and inputs["finding_ids"] == analysis["output"]["finding_ids"]


def test_the_analyst_hands_the_period_to_the_showrunner(conn, policy):
    cid = sandbox_history(conn, policy, 5)
    _, job = _run(conn, policy, "analyst", {"creator_id": str(cid), "then_plan": "2026-11-20"}, AnalystLLM(conn=conn))
    nxt = conn.execute("SELECT payload, parent_job_id FROM jobs WHERE kind = 'brain.showrunner'").fetchone()
    assert nxt["payload"]["start"] == "2026-11-20" and nxt["parent_job_id"] == job


def test_simulated_guidance_never_reaches_a_real_plan(conn, policy):
    cid = sandbox_history(conn, policy, 90)
    _run(conn, policy, "analyst", {"creator_id": str(cid)}, AnalystLLM(conn=conn))
    conn.execute("UPDATE creators SET publish_mode = 'instagram' WHERE id = %s", (cid,))
    assert analyst.latest(conn, cid) is None


def test_a_hypothesis_that_contradicts_its_effect_is_not_recorded(conn, policy):
    class Wrong(AnalystLLM):
        def structured(self, role, *, system, content, schema, effort="high", creator_id=None):
            out = super().structured(role, system=system, content=content, schema=schema)
            if schema is analyst.Analysis:
                out.hypotheses[0].direction = "negative"
            return out
    cid = sandbox_history(conn, policy, 90)
    out, _ = _run(conn, policy, "analyst", {"creator_id": str(cid)}, Wrong(conn=conn))
    assert out["finding_ids"] == []
