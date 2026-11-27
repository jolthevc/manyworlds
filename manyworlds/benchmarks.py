"""Benchmark models on Manyworlds' own creative tasks (owner 2026-11-27: the best economically viable model per
task, chosen on our assignments, not leaderboards; evaluation rigorous and independent of the owner).

A scenario (policy/benchmarks/<file>.yaml) is one assignment given identically to every candidate model, written
for one of our creators (her bible as the system prompt). Each model answers `samples` times: those are the
independent creative samples. Each candidate's sample k is compared blind with the incumbent's sample k, in an A/B
order fixed by a hash (half each way), by every judge in the pool, Claude included (owner: excluding a family is a
bias of its own). What the trial reports, kept apart:

    per sample   one vote per independent sample (the judges' majority): the unit of creative evidence
    per judge    every verdict, and each judge family's verdicts on the pairs where its own family competes
    self-preference  how much more often a judge picks its own family's answer than the other judges do on the
                 same pairs (a judge with a positive bias is reported, and the per-sample votes are also given
                 without self-family judges)
    length       how often the longer answer won a decisive verdict (verbosity is not quality)
    dimensions   1-5 scores per answer on the bench's creative dimensions (originality, wit, voice, charisma,
                 plausibility...): evidence, never a gate

Spend: every call is reserved at its estimate before it is made; the run stops when actual plus reserved would
pass the cap, so the cap holds (2026-11-27: a run without reservations went $0.01 over). The result records the
plan's estimate, the committed (reserved) and the actual cost. Text answers are the cheap first gate: the
finished media is the real benchmark, measured downstream (llm.splits, the ledger's task and model).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
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


MIN_JUDGES = 2   # judges with no stake (neither contestant's family) a pair should have; fewer is flagged

DIMENSIONS = ("originality", "delight", "wit", "voice", "nuance", "naturalness", "charisma", "memorability",
              "aspiration", "distinctiveness", "taste", "surprise", "plausibility")


class Piece(BaseModel):
    text: str = Field(description="the answer to the assignment, as it would be used")


class Scores(BaseModel):
    """1-5 each (0 when the dimension does not apply to the assignment)."""
    originality: int = 0
    delight: int = 0
    wit: int = 0
    voice: int = Field(0, description="sounds like this specific character")
    nuance: int = Field(0, description="emotional nuance and subtext")
    naturalness: int = 0
    charisma: int = 0
    memorability: int = 0
    aspiration: int = 0
    distinctiveness: int = 0
    taste: int = 0
    surprise: int = Field(0, description="unexpected but coherent choices")
    plausibility: int = Field(0, description="physically and directorially achievable")


class Verdict(BaseModel):
    reason: str
    better: Literal["A", "B", "tie"]
    a: Scores = Field(default_factory=Scores)
    b: Scores = Field(default_factory=Scores)


JUDGE_SYSTEM = (
    "You compare two answers to the same creative assignment for a studio that makes original social content with "
    "fictional, openly AI-generated personalities. Judge as a demanding creative director: what would actually "
    "delight an audience, sound human, fit this character and be worth following. Length, polish and a familiar "
    "writing style are not quality in themselves; generic, safe or 'AI-sounding' work loses to specific, original, "
    "compelling work, and an unexpected choice that coheres beats a predictable one. You do not know which system "
    "wrote which answer. Say 'tie' when neither is clearly better. Give your reason first, in two or three "
    "sentences, then score each answer 1-5 on every dimension (0 when one does not apply).")


def load(name: str = "creative-v1") -> dict:
    import yaml
    return yaml.safe_load((_root() / f"{name}.yaml").read_text(encoding="utf-8"))


def family(policy: Policy, model: str) -> str:
    e = (policy.get("llm", "registry", model, default={}) or {})
    return str(e.get("family") or model.split("-")[0])


def judges_for(policy: Policy, judges: list[str], a: str, b: str) -> list[str]:
    """The judges with no stake in comparing a and b (neither contestant's family)."""
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


# rough tokens per call, by kind; observed means in the run replace them as calls finish
OUT_TOKENS = {"answer": 1500, "judge": 900}   # answers include hidden reasoning; verdicts include 26 scores


def call_estimate(policy: Policy, model: str, chars_in: int, kind: str, observed_out: dict | None = None) -> float:
    pi, po = _price(policy, model)
    out = (observed_out or {}).get((model, kind)) or OUT_TOKENS[kind]
    return (chars_in / 4 * pi + out * po) / 1e6


def estimate(policy: Policy, scenarios: list[str], models: list[str], baseline: str, judges: list[str],
             samples: int, *, chars_in: int = 9000, judge_chars: int = 7000) -> float:
    """The plan's dollars: every contestant's answers plus every verdict (all judges on every pair)."""
    usd = 0.0
    for m in {*models, baseline}:
        usd += len(scenarios) * samples * call_estimate(policy, m, chars_in, "answer")
    for _m in models:
        for j in judges:
            usd += len(scenarios) * samples * call_estimate(policy, j, judge_chars, "judge")
    return round(usd, 3)


def _words(t: str | None) -> int:
    return len((t or "").split())


def summarise(policy: Policy, models: list[str], baseline: str, answers: dict, verdicts: list[dict]) -> dict:
    """Per sample (one majority vote per independent sample, with and without self-family judges), per judge,
    self-preference by judge family, length, dimension means, cost per answer."""
    out: dict = {}
    for m in models:
        vs = [v for v in verdicts if v["model"] == m and "winner" in v]
        samples: dict = defaultdict(list)
        for v in vs:
            samples[(v["scenario"], v["sample"])].append(v)

        def vote(group, *, neutral_only=False):
            ws = Counter(v["winner"] for v in group if not (neutral_only and v["stake"]))
            if not ws:
                return None
            top, n = ws.most_common(1)[0]
            return "tie" if list(ws.values()).count(n) > 1 else top
        per_sample = Counter(vote(g) for g in samples.values())
        neutral = Counter(vote(g, neutral_only=True) for g in samples.values())
        neutral.pop(None, None)
        cost = [a["usd"] for (s, mm, k), a in answers.items() if mm == m and a["text"]]
        dims = {d: round(sum(v["scores"]["candidate"][d] for v in vs) / len(vs), 2) for d in DIMENSIONS} if vs else {}
        base_dims = {d: round(sum(v["scores"]["baseline"][d] for v in vs) / len(vs), 2) for d in DIMENSIONS} if vs else {}
        by_scenario = defaultdict(Counter)
        for (s, k), g in samples.items():
            by_scenario[s][vote(g)] += 1
        out[m] = {"samples": {k: per_sample.get(k, 0) for k in ("candidate", "baseline", "tie")},
                  "samples_neutral_judges": {k: neutral.get(k, 0) for k in ("candidate", "baseline", "tie")},
                  "by_scenario": {s: {k: c.get(k, 0) for k in ("candidate", "baseline", "tie")}
                                  for s, c in sorted(by_scenario.items())},
                  "judgements": len(vs), "by_judge": {j: dict(Counter(v["winner"] for v in vs if v["judge"] == j))
                                                      for j in sorted({v["judge"] for v in vs})},
                  "dimensions": dims, "baseline_dimensions": base_dims,
                  "failed_answers": sum(1 for (s, mm, k), a in answers.items() if mm == m and not a["text"]),
                  "usd_per_answer": round(sum(cost) / len(cost), 5) if cost else None}
    # self-preference: on pairs where a judge's family competes, how often it picks that side, against how often
    # the judges with no stake pick the same side on the same pairs
    bias: dict = {}
    for fam in sorted({family(policy, v["judge"]) for v in verdicts if "winner" in v}):
        own, others = [], []
        for v in verdicts:
            if "winner" not in v or v["winner"] == "tie":
                continue
            side = ("candidate" if family(policy, v["model"]) == fam else
                    "baseline" if family(policy, baseline) == fam else None)
            if side is None:
                continue
            (own if family(policy, v["judge"]) == fam else others if not v["stake"] else []).append(v["winner"] == side)
        if own and others:
            bias[fam] = {"own_family_pick_rate": round(sum(own) / len(own), 2),
                         "neutral_judges_pick_rate": round(sum(others) / len(others), 2),
                         "self_preference": round(sum(own) / len(own) - sum(others) / len(others), 2),
                         "n_own": len(own), "n_neutral": len(others)}
    decisive = [v for v in verdicts if v.get("winner") in ("candidate", "baseline") and v["words"]["candidate"]
                != v["words"]["baseline"]]
    longer = sum((v["winner"] == "candidate") == (v["words"]["candidate"] > v["words"]["baseline"]) for v in decisive)
    return {"models": out, "self_preference": bias,
            "length": {"decisive_verdicts": len(decisive),
                       "longer_answer_won": round(longer / len(decisive), 2) if decisive else None}}


def run(conn, policy: Policy, llm, *, creator_id, scenarios: list[str] | None, models: list[str], baseline: str,
        judges: list[str], samples: int = 2, usd: float = 1.0, bench: str = "creative-v1",
        dry_run: bool = False) -> dict:
    from .brain import inquiry
    spec = load(bench)
    chosen = scenarios or list(spec["scenarios"])
    plan = {"bench": bench, "scenarios": chosen, "models": models, "baseline": baseline, "samples": samples,
            "judges": judges, "neutral_judges": {m: judges_for(policy, judges, m, baseline) for m in models},
            "estimate_usd": estimate(policy, chosen, models, baseline, judges, samples), "cap_usd": usd}
    plan["under_judged"] = sorted(m for m, js in plan["neutral_judges"].items() if len(js) < MIN_JUDGES)
    if dry_run:
        return plan
    c = conn.execute("SELECT bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
    system = _system(c["bible"] or {}, "")
    spent0, answers, verdicts = llm.spent, {}, []
    committed, stopped = 0.0, None
    observed: dict = defaultdict(list)

    def means():
        return {k: sum(v) / len(v) for k, v in observed.items() if v}

    def reserve(model, chars, kind) -> float | None:
        """The call's estimate, if actual spend plus it stays under the cap; else None (the run stops)."""
        nonlocal committed, stopped
        est = call_estimate(policy, model, chars, kind, means())
        if (llm.spent - spent0) + est > usd:
            stopped = stopped or f"cap: ${usd} (spent ${round(llm.spent - spent0, 4)}, next call ~${round(est, 4)})"
            return None
        committed += est
        return est

    def call(model, *, system_, content, schema, kind):
        n0, t0 = len(llm.calls), time.monotonic()
        try:
            got, err = llm._call(model, system=system_, content=content, schema=schema, effort="medium",
                                 creator_id=creator_id), None
        except Exception as e:  # noqa: BLE001 - a model's failure is a result
            got, err = None, f"{type(e).__name__}: {str(e)[:300]}"
        new = llm.calls[n0:]
        out_tok = sum(getattr(u, "output_tokens", 0) or 0 for u in new)
        if out_tok:
            observed[(model, kind)].append(out_tok)
        return got, err, round(time.monotonic() - t0, 2), round(sum(u.usd for u in new), 6), out_tok

    was = getattr(llm, "shadowing", None)
    if was is not None:
        llm.shadowing = False
    try:
        with llm.tagged("model_bench", None):
            for sid in chosen:
                sc = spec["scenarios"][sid]
                for m in [baseline, *models]:
                    for k in range(samples):
                        if stopped or reserve(m, len(system) + len(sc["brief"]), "answer") is None:
                            break
                        got, err, secs, cost, tok = call(m, system_=system, content=sc["brief"], schema=Piece,
                                                         kind="answer")
                        answers[(sid, m, k)] = {"text": got.text if got else None, "error": err, "seconds": secs,
                                                "usd": cost, "output_tokens": tok}
                for m in models:
                    for k in range(samples):
                        a_, b_ = answers.get((sid, baseline, k)), answers.get((sid, m, k))
                        if not a_ or not b_ or not a_["text"] or not b_["text"]:
                            continue
                        flip = int(hashlib.sha256(f"{sid}:{m}:{k}".encode()).hexdigest(), 16) % 2 == 1
                        first, second = (b_, a_) if flip else (a_, b_)
                        content = (f"The assignment:\n{sc['brief']}\nCriteria: {sc['criteria']}\n\n"
                                   f"=== Answer A ===\n{first['text']}\n\n=== Answer B ===\n{second['text']}")
                        for j in judges:
                            if stopped or reserve(j, len(JUDGE_SYSTEM) + len(content), "judge") is None:
                                break
                            v, err, secs, cost, _ = call(j, system_=JUDGE_SYSTEM, content=content, schema=Verdict,
                                                         kind="judge")
                            stake = family(policy, j) in {family(policy, m), family(policy, baseline)}
                            if v is None:
                                verdicts.append({"scenario": sid, "model": m, "sample": k, "judge": j,
                                                 "stake": stake, "error": err})
                                continue
                            cand, base = (v.a, v.b) if flip else (v.b, v.a)
                            won = ("tie" if v.better == "tie" else
                                   (("candidate" if v.better == "A" else "baseline") if flip
                                    else ("baseline" if v.better == "A" else "candidate")))
                            verdicts.append({"scenario": sid, "model": m, "sample": k, "judge": j, "stake": stake,
                                             "winner": won, "reason": v.reason[:500], "usd": cost,
                                             "words": {"candidate": _words(b_["text"]), "baseline": _words(a_["text"])},
                                             "scores": {"candidate": cand.model_dump(), "baseline": base.model_dump()}})
    finally:
        if was is not None:
            llm.shadowing = was
    tested = {s for (s, _m, _k) in answers}
    judged = {v["scenario"] for v in verdicts if "winner" in v}
    result = {**plan, **summarise(policy, models, baseline, answers, verdicts),
              "cost": {"estimated_usd": plan["estimate_usd"], "committed_usd": round(committed, 4),
                       "actual_usd": round(llm.spent - spent0, 4)},
              "stopped": stopped, "untested_scenarios": [s for s in chosen if s not in judged],
              "unanswered_scenarios": [s for s in chosen if s not in tested],
              "answers": [{"scenario": s, "model": m, "sample": k, **a} for (s, m, k), a in answers.items()],
              "verdicts": verdicts}
    inquiry.record(conn, policy, requester="model_bench", method="model_bench", spec=plan, result=result,
                   data={"answers": len(answers), "verdicts": len(verdicts)})
    return result


def evidence(conn, policy: Policy, *, bench: str | None = None) -> dict:
    """Every model_bench trial in the ledger, pooled per model and scenario task: independent samples won, lost
    and tied against each baseline (the unit of evidence), and the scenarios never judged. What the registry's
    own evidence and task-level promotions rest on."""
    spec_tasks: dict = {}
    rows = conn.execute("SELECT result FROM analysis_runs WHERE method = 'model_bench' ORDER BY created_at").fetchall()
    pooled: dict = defaultdict(lambda: defaultdict(Counter))
    untested: set = set()
    for r in rows:
        res = r["result"] or {}
        if bench and res.get("bench") != bench:
            continue
        b = res.get("bench") or "creative-v1"
        if b not in spec_tasks:
            try:
                spec_tasks[b] = {k: v.get("task") for k, v in load(b)["scenarios"].items()}
            except Exception:  # noqa: BLE001 - a retired bench file
                spec_tasks[b] = {}
        for m, s in (res.get("models") or {}).items():
            for sid, c in (s.get("by_scenario") or {}).items():
                task = spec_tasks[b].get(sid, sid)
                pooled[(m, res.get("baseline"))][f"{task}:{sid}"].update(c)
        untested |= set(res.get("untested_scenarios") or [])
    return {"models": {f"{m} vs {base}": {k: dict(c) for k, c in tasks.items()} for (m, base), tasks in pooled.items()},
            "never_judged": sorted(untested - {k.split(":", 1)[1] for t in pooled.values() for k in t})}
