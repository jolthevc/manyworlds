"""The voice bake-off (research 2026-11-24-expressive-voice §5, §6): which engine becomes the default, decided by code.

    mw test breadth --plan voicebake --dry-run    # what it makes and what it should cost; creates nothing
    mw test breadth --plan voicebake              # only with the owner's go for that run (CLAUDE.md)
    mw voice bakeoff-decide <run>                 # (the driver also runs it at the end of the run)

The plan (policy/test_overlay.yaml `breadth.plans.voicebake`) is one `voice.cast` benchmark per engine and voice
description: Maren's bible and one other archetype's description. Each cast designs `voice.bakeoff.candidates`
references and speaks the casting script plus the bake-off lines (`voice.bakeoff.lines`, "Two am. I take it back."
among them) in that engine, every line measured on the worker (worker/handlers/voice_judge.py). Nobody listens.

`decide` scores each engine (manyworlds.voice_eval.bakeoff_winner): per cast, the best qualifying candidate's
composite (what production would get), averaged over the engine's casts; accepted minutes are the passing lines'
audio; cost is the GPU time the ledger attributes to its jobs. The winner is the highest composite per accepted
minute cost (cost dampened by policy voice.bakeoff.cost_exponent), licence cleared, no gender miss; the incumbent
stays unless beaten by voice.bakeoff.min_gain. The result is a decision row (kind voice_engine), which
voice_lab.default_engine reads: that row is the flag that switches the default engine.
"""

from __future__ import annotations

from statistics import mean

from psycopg.types.json import Jsonb

from .creator import voice_lab
from .policy import Policy


def _cfg(policy: Policy) -> dict:
    return policy.get("voice", "bakeoff", default={}) or {}


def script(policy: Policy) -> list[dict]:
    """The casting script followed by the bake-off lines."""
    return [*(policy.get("voice", "casting", "script", default=[]) or []), *(_cfg(policy).get("lines") or [])]


def _words(text: str) -> int:
    return len(str(text or "").split())


def expected_job(policy: Policy, bench: dict) -> dict:
    """A voice.cast bench item for the breadth estimate: its audio seconds (candidates x (reference + every line at
    the expected pace)), priced per output second by the engine's rate (breadth.estimate.seconds_per_output_second
    `voice.cast:<engine>`: generation, transcription, judging and repairs included)."""
    extra = bench.get("extra") or {}
    engine = extra.get("engine") or policy.get("voice", "engine", default="indextts")
    n = int(extra.get("n") or _cfg(policy).get("candidates", 2))
    wps = float(_cfg(policy).get("estimate_wps", 3.8))
    lines = script(policy)
    anchor = _words(policy.get("voice", "anchor_text", default=""))
    seconds = n * (anchor + sum(_words(x.get("text")) for x in lines)) / wps
    return {"kind": "voice.cast", "n": int(bench.get("jobs", 1)), "output_seconds": round(seconds, 1),
            "rate": f"voice.cast:{engine}"}


def bench_payload(conn, policy: Policy, extra: dict, i: int) -> tuple[dict, object]:
    """One bake-off cast (bench.enqueue): the creator's bible description (or `extra.description` for another
    archetype, with `extra.fit` {gender, age} and `extra.archetype`), the engine named in `extra.engine`. Never
    stored as anyone's voice (purpose bakeoff). -> (payload, creator_id)."""
    from . import voice_eval
    from .brain import archetypes
    cid = extra.get("creator_id")
    bible = {}
    if cid:
        row = conn.execute("SELECT bible FROM creators WHERE id = %s", (cid,)).fetchone()
        bible = (row["bible"] if row else None) or {}
    description = extra.get("description") or voice_lab.describe(bible)
    if not description:
        raise ValueError("a bake-off cast needs a creator with a bible or --extra description")
    engine = str(extra.get("engine") or policy.get("voice", "engine", default="indextts"))
    if engine not in (policy.get("voice", "engines", default={}) or {}):
        raise ValueError(f"no policy voice.engines.{engine}")
    payload = voice_lab.cast_payload(policy, description=description, engine=engine,
                                     seed=int(extra.get("seed", 9100 + i)), n=int(extra.get("n") or
                                                                                  _cfg(policy).get("candidates", 2)),
                                     script=script(policy))
    fit = extra.get("fit") or voice_lab.fit_of(bible)
    named = [extra["archetype"]] if extra.get("archetype") else archetypes.for_creator(policy, bible)
    payload.update(purpose="bakeoff", fit=fit, pace_wps=list(voice_eval.pace_band(policy, named)),
                   bakeoff={"engine": engine, "voice": extra.get("label") or ("creator" if cid and not
                                                                              extra.get("description") else "other")})
    if extra.get("breadth"):
        payload["breadth"] = extra["breadth"]
    return payload, cid


def _gpu_usd(conn, job_id) -> float:
    return float(conn.execute("SELECT coalesce(sum(usd), 0) AS u FROM cost_entries WHERE job_id = %s "
                              "AND category = 'gpu'", (job_id,)).fetchone()["u"])


def scores(conn, policy: Policy, batch_like: str) -> dict:
    """Per engine, from its finished casts in the run: {composite, pass_share, accepted_minutes, usd, gender_ok,
    licence_ok, casts [{voice, best, qualified}]}."""
    from . import voice_eval
    jobs = conn.execute("SELECT id, status, payload FROM jobs WHERE kind = %s AND (batch = %s OR batch LIKE %s) "
                        "AND payload->>'purpose' = 'bakeoff' ORDER BY created_at",
                        (voice_lab.cast_kind(policy), batch_like, f"{batch_like}-%")).fetchall()
    judge = policy.get("voice", "judge", default={}) or {}
    engines: dict[str, dict] = {}
    for j in jobs:
        p = j["payload"] or {}
        e = engines.setdefault(p.get("engine"), {"casts": [], "seconds": 0.0, "usd": 0.0, "gender_ok": True})
        e["usd"] += _gpu_usd(conn, j["id"])
        if j["status"] != "done":
            e["casts"].append({"voice": (p.get("bakeoff") or {}).get("voice"), "status": j["status"], "best": None})
            continue
        band = tuple(p.get("pace_wps") or voice_eval.pace_band(policy, None))
        metas = [a["meta"] or {} for a in conn.execute(
            "SELECT meta FROM assets WHERE job_id = %s AND kind = 'audio' ORDER BY uri", (j["id"],)).fetchall()]
        summaries = voice_lab.judge_cast(policy, metas, fit=p.get("fit") or {}, band=band)
        for m in metas:
            for line in m.get("lines") or []:
                if voice_eval.line_verdict(line, judge, band)["passed"]:
                    e["seconds"] += float(line.get("seconds") or 0)
        ok = [s for s in summaries if s["qualifies"]]
        best = max(ok, key=lambda s: s["composite"]) if ok else None
        e["gender_ok"] = e["gender_ok"] and any(s["gender_ok"] for s in summaries)
        e["casts"].append({"voice": (p.get("bakeoff") or {}).get("voice"), "status": "done",
                           "best": None if best is None else {k: best[k] for k in (
                               "composite", "pass_share", "line_composite", "fit", "mean_mos", "mean_wer",
                               "delivery_spread_st")},
                           "qualified": len(ok), "candidates": len(summaries),
                           "reasons": [s["reasons"] for s in summaries if not s["qualifies"]]})
    out = {}
    for name, e in engines.items():
        bests = [c["best"] for c in e["casts"]]
        out[name] = {"composite": round(mean((b or {}).get("composite", 0.0) for b in bests), 4) if bests else 0.0,
                     "pass_share": round(mean((b or {}).get("pass_share", 0.0) for b in bests), 4) if bests else 0.0,
                     "accepted_minutes": round(e["seconds"] / 60, 3), "usd": round(e["usd"], 4),
                     "gender_ok": e["gender_ok"] and all(bests),
                     "licence_ok": bool(policy.get("voice", "engines", name, "licence_ok", default=False)),
                     "casts": e["casts"]}
    return out


def decide(conn, policy: Policy, run: str) -> dict:
    """Score the run's casts and record the winner as the default engine (a decision row of kind
    voice.engine_gate.decision_kind; nothing is recorded while any cast is unfinished or none qualifies)."""
    from . import breadth, voice_eval
    batch = breadth.batch_name(breadth.test_policy(policy), run)
    per_engine = scores(conn, policy, batch)
    if not per_engine:
        return {"run": run, "decided": False, "reason": "no bake-off casts in this run"}
    waiting = [n for n, e in per_engine.items() for c in e["casts"] if c["status"] in ("queued", "running")]
    if waiting:
        return {"run": run, "decided": False, "reason": f"casts still running for {sorted(set(waiting))}",
                "engines": per_engine}
    cfg = {**_cfg(policy), "incumbent": policy.get("voice", "engine_gate", "until_bakeoff", default="voxcpm")}
    result = voice_eval.bakeoff_winner(per_engine, cfg)
    out = {"run": run, "decided": result["winner"] is not None, **result, "engines": per_engine}
    if result["winner"] is None:
        return {**out, "reason": "no engine qualified"}
    kind = str(policy.get("voice", "engine_gate", "decision_kind", default="voice_engine"))
    conn.execute(
        """INSERT INTO decisions (kind, inputs, output, policy_version, agent, step, rationale)
           VALUES (%s, %s, %s, %s, 'voice_bakeoff', %s, %s)""",
        (kind, Jsonb({"run": run, "engines": per_engine, "rule": {k: cfg.get(k) for k in (
            "cost_exponent", "min_gain", "min_pass_share", "incumbent")}}),
         Jsonb({"winner": result["winner"], "ranked": result["ranked"], "refused": result["refused"]}),
         policy.version, f"voice_engine:{run}",
         f"{result['winner']}: highest composite per accepted-minute cost among cleared engines"))
    return out
