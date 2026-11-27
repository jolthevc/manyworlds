"""Benchmark models on Manyworlds' own creative tasks (owner 2026-11-27: the best economically viable model per
task, chosen on our assignments, not leaderboards).

A scenario (policy/benchmarks/<file>.yaml) is one assignment given identically to every candidate model, written
for one of our creators (her bible and voice rules as the system prompt). Each candidate answers `samples` times;
each answer is compared blind with the incumbent's answer of the same sample number, as A and B in an order fixed
by a hash (half each way), by every judge whose model family is neither contestant's (no model judges its own
family's work). The trial goes in the ledger (analysis_runs, method model_bench) with every answer, verdict, cost
and latency, so the evidence lives with the rest of the Brain's. Spend stops at `usd`; `dry_run` only estimates.

Text answers here are the cheap first gate: the finished media is the real benchmark (a model that writes better
direction must also make better renders), which needs a render run and the owner's yes.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .policy import Policy

def _root() -> Path:
    for candidate in (Path(__file__).resolve().parents[1] / "policy" / "benchmarks",
                      Path.cwd() / "policy" / "benchmarks", Path("/app/policy/benchmarks")):
        if candidate.is_dir():
            return candidate
    return Path.cwd() / "policy" / "benchmarks"


MIN_JUDGES = 2   # eligible judges a pair should have; fewer is flagged in the plan


class Piece(BaseModel):
    text: str = Field(description="the answer to the assignment, as it would be used")


class Verdict(BaseModel):
    reason: str
    better: Literal["A", "B", "tie"]


JUDGE_SYSTEM = (
    "You compare two answers to the same creative assignment for a studio that makes original social content with "
    "fictional, openly AI-generated personalities. Judge only by the criteria given, as a demanding creative "
    "director would: what would actually delight an audience, sound human and fit this character. Length and "
    "polish are not quality in themselves; generic, safe or 'AI-sounding' writing loses to specific, surprising "
    "work. Say 'tie' when neither is clearly better. Give your reason first, in two or three sentences.")


def load(name: str = "creative-v1") -> dict:
    import yaml
    return yaml.safe_load((_root() / f"{name}.yaml").read_text(encoding="utf-8"))


def family(policy: Policy, model: str) -> str:
    e = (policy.get("llm", "registry", model, default={}) or {})
    return str(e.get("family") or model.split("-")[0])


def judges_for(policy: Policy, judges: list[str], a: str, b: str) -> list[str]:
    """The judges allowed to compare a and b: none from either contestant's family."""
    banned = {family(policy, a), family(policy, b)}
    return [j for j in judges if family(policy, j) not in banned]


def _system(bible: dict, voice: str) -> str:
    return ("You write for a fictional creator, an openly AI-generated personality, as her writers' room would. "
            "Her bible (who she is):\n" + json.dumps(bible, default=str)[:6000] +
            (f"\n\nHer voice rules:\n{voice}" if voice else ""))


def _price(policy: Policy, model: str) -> tuple[float, float]:
    om = policy.get("llm", "open_models", model, default=None)
    if om:
        p = om.get("usd_per_mtok") or [0, 0]
        return float(p[0]), float(p[1])
    p = policy.get("llm", "usd_per_mtok", model, default=[0, 0]) or [0, 0]
    return float(p[0]), float(p[1])


def estimate(policy: Policy, scenarios: list[str], models: list[str], baseline: str, judges: list[str],
             samples: int, *, in_tokens: int = 2500, out_tokens: int = 400, judge_in: int = 1500,
             judge_out: int = 150) -> float:
    """Rough dollars for a run: every contestant's answers plus every allowed judgement."""
    usd = 0.0
    for m in {*models, baseline}:
        pi, po = _price(policy, m)
        usd += len(scenarios) * samples * (in_tokens * pi + out_tokens * po) / 1e6
    for m in models:
        for j in judges_for(policy, judges, m, baseline):
            pi, po = _price(policy, j)
            usd += len(scenarios) * samples * (judge_in * pi + judge_out * po) / 1e6
    return round(usd, 3)


def run(conn, policy: Policy, llm, *, creator_id, scenarios: list[str] | None, models: list[str], baseline: str,
        judges: list[str], samples: int = 2, usd: float = 1.0, bench: str = "creative-v1",
        dry_run: bool = False) -> dict:
    from .brain import inquiry
    spec = load(bench)
    chosen = scenarios or list(spec["scenarios"])
    est = estimate(policy, chosen, models, baseline, judges, samples)
    plan = {"bench": bench, "scenarios": chosen, "models": models, "baseline": baseline, "samples": samples,
            "judges": {m: judges_for(policy, judges, m, baseline) for m in models}, "estimate_usd": est,
            "cap_usd": usd}
    # one judge per pair is one opinion: on 2026-11-27 two judges split 3-9 and 8-5 on the same candidate
    plan["under_judged"] = sorted(m for m, js in plan["judges"].items() if len(js) < MIN_JUDGES)
    if dry_run:
        return plan
    c = conn.execute("SELECT bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
    from .brain import planner
    system = _system(c["bible"] or {}, planner.voice_rules(policy) if hasattr(planner, "voice_rules") else "")
    spent0, answers, verdicts = llm.spent, {}, []
    was = getattr(llm, "shadowing", None)
    if was is not None:
        llm.shadowing = False
    try:
        with llm.tagged("model_bench", None):
            for sid in chosen:
                sc = spec["scenarios"][sid]
                for m in [baseline, *models]:
                    for k in range(samples):
                        if llm.spent - spent0 >= usd:
                            break
                        t0 = time.monotonic()
                        n0 = len(llm.calls)
                        try:
                            got = llm._call(m, system=system, content=sc["brief"], schema=Piece, effort="medium",
                                            creator_id=creator_id).text
                            err = None
                        except Exception as e:  # noqa: BLE001 - a candidate's failure is a result
                            got, err = None, f"{type(e).__name__}: {str(e)[:300]}"
                        answers[(sid, m, k)] = {"text": got, "error": err,
                                                "seconds": round(time.monotonic() - t0, 2),
                                                "usd": round(sum(u.usd for u in llm.calls[n0:]), 6)}
                for m in models:
                    for k in range(samples):
                        a_, b_ = answers.get((sid, baseline, k)), answers.get((sid, m, k))
                        if not a_ or not b_ or not a_["text"] or not b_["text"]:
                            continue
                        flip = int(hashlib.sha256(f"{sid}:{m}:{k}".encode()).hexdigest(), 16) % 2 == 1
                        first, second = (b_, a_) if flip else (a_, b_)
                        for j in judges_for(policy, judges, m, baseline):
                            if llm.spent - spent0 >= usd:
                                break
                            content = (f"The assignment:\n{sc['brief']}\nCriteria: {sc['criteria']}\n\n"
                                       f"=== Answer A ===\n{first['text']}\n\n=== Answer B ===\n{second['text']}")
                            try:
                                v = llm._call(j, system=JUDGE_SYSTEM, content=content, schema=Verdict,
                                              effort="medium", creator_id=creator_id)
                            except Exception as e:  # noqa: BLE001
                                verdicts.append({"scenario": sid, "model": m, "sample": k, "judge": j,
                                                 "error": f"{type(e).__name__}: {str(e)[:200]}"})
                                continue
                            won = ("tie" if v.better == "tie" else
                                   (("candidate" if v.better == "A" else "baseline") if flip
                                    else ("baseline" if v.better == "A" else "candidate")))
                            verdicts.append({"scenario": sid, "model": m, "sample": k, "judge": j, "winner": won,
                                             "reason": v.reason[:500]})
    finally:
        if was is not None:
            llm.shadowing = was
    summary = {}
    for m in models:
        vs = [v for v in verdicts if v["model"] == m and "winner" in v]
        cost = [a["usd"] for (s, mm, k), a in answers.items() if mm == m and a["text"]]
        summary[m] = {"judgements": len(vs), "wins": sum(v["winner"] == "candidate" for v in vs),
                      "losses": sum(v["winner"] == "baseline" for v in vs), "ties": sum(v["winner"] == "tie" for v in vs),
                      "failed_answers": sum(1 for (s, mm, k), a in answers.items() if mm == m and not a["text"]),
                      "usd_per_answer": round(sum(cost) / len(cost), 5) if cost else None}
    base_cost = [a["usd"] for (s, mm, k), a in answers.items() if mm == baseline and a["text"]]
    result = {**plan, "spent_usd": round(llm.spent - spent0, 4), "summary": summary,
              "baseline_usd_per_answer": round(sum(base_cost) / len(base_cost), 5) if base_cost else None,
              "answers": [{"scenario": s, "model": m, "sample": k, **a} for (s, m, k), a in answers.items()],
              "verdicts": verdicts}
    inquiry.record(conn, policy, requester="model_bench", method="model_bench", spec=plan, result=result,
                   data={"answers": len(answers), "verdicts": len(verdicts)})
    return result
