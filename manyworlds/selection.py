"""Model selection: the best model for each job at the lowest cost, chosen from evidence (owner 2026-11-21).

Every role's calls are shadowed by candidate models (manyworlds.llm, llm.shadow). This turns those comparisons
into proposals: switch role X to model Y. A proposal is an approval the owner decides (it reaches the phone
through manyworlds.notify); approving it writes `llm.roles.<role>`, which every service's LLM router reads, with
the old model kept as the fallback. A rejected proposal is not raised again for `rejected_cooldown_days`.

Two ways a candidate qualifies (policy `llm.selection.roles.<role>.mode`):

  agreement  judgment roles (scoring, review, strategy): its answers match the role's model closely enough
             (numbers within `max_mean_abs_diff` on average, labels agreeing at least `min_agreement`)
  quality    generative roles (the showrunner): different answers are fine if they are as good. The evaluator
             scores both answers' items blind on a rubric (`judge`); the candidate qualifies when its mean
             score is at most `max_score_drop` below the primary's and it breaks the voice rules no more often

Either way it must answer validly (`min_valid`), have enough calls (`min_calls`), and cost at most
`max_cost_ratio` of the role's model. The cheapest qualifying candidate is proposed.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from statistics import mean

from psycopg.types.json import Jsonb

from . import shadow
from .policy import Policy

AUTHOR = "selection"


def _cfg(policy: Policy) -> dict:
    return policy.get("llm", "selection", default={}) or {}


def _role_cfg(policy: Policy, role: str) -> dict | None:
    cfg = _cfg(policy)
    rc = (cfg.get("roles") or {}).get(role)
    return {**{k: v for k, v in cfg.items() if k != "roles"}, **rc} if rc else None


def _judged(conn, role: str, primary: str, candidate: str, since: datetime) -> list[dict]:
    return [r["judged"] for r in conn.execute(
        "SELECT judged FROM llm_shadow WHERE role = %s AND primary_model = %s AND candidate_model = %s "
        "AND judged IS NOT NULL AND created_at >= %s", (role, primary, candidate, since)).fetchall()]


def assess(conn, policy: Policy, role: str, r: dict, *, since: datetime) -> dict:
    """Whether one candidate (a `shadow.report` row) qualifies for the role, and why not."""
    rc = _role_cfg(policy, role) or {}
    why = []
    if r["calls"] < int(rc.get("min_calls", 30)):
        why.append(f"{r['calls']} calls < {rc.get('min_calls', 30)}")
    if r["valid_share"] < float(rc.get("min_valid", 0.95)):
        why.append(f"valid {r['valid_share']} < {rc.get('min_valid', 0.95)}")
    if r["cost_ratio"] is None or r["cost_ratio"] > float(rc.get("max_cost_ratio", 0.7)):
        why.append(f"cost ratio {r['cost_ratio']} > {rc.get('max_cost_ratio', 0.7)}")
    evidence: dict = {}
    if rc.get("mode") == "quality":
        judged = _judged(conn, role, r["primary"], r["candidate"], since)
        evidence["judged"] = len(judged)
        if len(judged) < int(rc.get("min_judged", 5)):
            why.append(f"{len(judged)} judged < {rc.get('min_judged', 5)}")
        else:
            p, c = mean(j["primary"]["score"] for j in judged), mean(j["candidate"]["score"] for j in judged)
            pb, cb = mean(j["primary"]["rule_breaks"] for j in judged), mean(j["candidate"]["rule_breaks"] for j in judged)
            evidence.update(primary_score=round(p, 2), candidate_score=round(c, 2), primary_breaks=round(pb, 2),
                            candidate_breaks=round(cb, 2))
            if c < p - float(rc.get("max_score_drop", 0.3)):
                why.append(f"scores {c:.2f} vs {p:.2f}")
            if cb > pb:
                why.append(f"breaks the voice rules more ({cb:.2f} vs {pb:.2f} per answer)")
    else:
        fields = r["fields"].values()
        diffs = [f["mean_abs_diff"] for f in fields if "mean_abs_diff" in f]
        agree = [f["agreement"] for f in fields if "agreement" in f]
        evidence.update(mean_abs_diff=round(mean(diffs), 3) if diffs else None,
                        agreement=round(mean(agree), 3) if agree else None)
        if diffs and mean(diffs) > float(rc.get("max_mean_abs_diff", 1.0)):
            why.append(f"scores differ by {mean(diffs):.2f} on average")
        if agree and mean(agree) < float(rc.get("min_agreement", 0.85)):
            why.append(f"agrees {mean(agree):.2f} of the time")
        if not diffs and not agree:
            why.append("nothing comparable in the answers")
    return {"candidate": r["candidate"], "primary": r["primary"], "qualifies": not why, "why_not": why,
            "cost_ratio": r["cost_ratio"], "calls": r["calls"], "valid_share": r["valid_share"],
            "primary_usd_per_call": r["primary_usd_per_call"], "candidate_usd_per_call": r["candidate_usd_per_call"],
            **evidence}


def recommend(conn, policy: Policy, role: str, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    rc = _role_cfg(policy, role)
    if not rc:
        return {"role": role, "pick": None, "candidates": [], "note": "no llm.selection entry for this role"}
    since = now - timedelta(days=float(rc.get("days", 14)))
    rows = shadow.report(conn, role, days=float(rc.get("days", 14)), tolerance=float(rc.get("tolerance", 1.0)), now=now)
    assessed = sorted((assess(conn, policy, role, r, since=since) for r in rows),
                      key=lambda a: (not a["qualifies"], a["cost_ratio"] if a["cost_ratio"] is not None else 9e9))
    pick = assessed[0] if assessed and assessed[0]["qualifies"] else None
    return {"role": role, "pick": pick, "candidates": assessed}


def propose(conn, policy: Policy, role: str, rec: dict, *, now: datetime | None = None) -> dict:
    """File the pick as an approval, unless one is pending or was rejected within the cooldown."""
    from .brain import overrides
    now = now or datetime.now(timezone.utc)
    pick = rec.get("pick")
    if not pick:
        return {"role": role, "proposed": False, "reason": "no qualifying candidate"}
    path = f"llm.roles.{role}"
    current = policy.get("llm", "roles", role)
    fallback = current["fallback"] if isinstance(current, dict) else current
    value = {"model": pick["candidate"], "fallback": fallback} if policy.get("llm", "open_models", pick["candidate"]) \
        else pick["candidate"]
    cooldown = now - timedelta(days=float(_cfg(policy).get("rejected_cooldown_days", 14)))
    seen = conn.execute(
        "SELECT status FROM approvals WHERE kind = 'policy_override' AND subject->>'path' = %s "
        "AND subject->'value' = %s AND (status = 'pending' OR (status = 'rejected' AND decided_at > %s))",
        (path, Jsonb(value), cooldown)).fetchone()
    if seen:
        return {"role": role, "proposed": False, "reason": f"already {seen['status']}"}
    saving = (1 - float(pick["cost_ratio"])) * 100
    reason = (f"Switch {role} from {pick['primary']} to {pick['candidate']}: {saving:.0f}% cheaper per call "
              f"(${pick['candidate_usd_per_call']:.4f} vs ${pick['primary_usd_per_call']:.4f}) over {pick['calls']} "
              f"shadowed calls; evidence {json.dumps({k: v for k, v in pick.items() if k not in ('why_not',)})}")
    out = overrides.propose(conn, policy, path, value, author=AUTHOR, reason=reason[:2000])
    return {"role": role, "proposed": "approval_id" in out, **out}


def run(conn, policy: Policy, *, now: datetime | None = None) -> list[dict]:
    """Every role with a selection entry: recommend, and propose the pick. Reads stored evidence only (no spend)."""
    return [propose(conn, policy, role, recommend(conn, policy, role, now=now), now=now)
            for role in (_cfg(policy).get("roles") or {})]


def due(conn, policy: Policy, *, now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    last = conn.execute("SELECT max(created_at) AS t FROM decisions WHERE kind = 'model_selection'").fetchone()["t"]
    return last is None or now - last >= timedelta(hours=float(_cfg(policy).get("every_hours", 24)))


def record(conn, policy: Policy, results: list[dict]) -> None:
    conn.execute("INSERT INTO decisions (kind, inputs, output, policy_version, agent) VALUES "
                 "('model_selection', %s, %s, %s, %s)",
                 (Jsonb({"roles": list((_cfg(policy).get("roles") or {}))}), Jsonb({"results": results}, dumps=_dumps),
                  policy.version, AUTHOR))


def _dumps(obj) -> str:
    return json.dumps(obj, default=str)


# --- judging generative answers ---------------------------------------------------------------------------------
def judge(conn, policy: Policy, llm, role: str, *, usd: float) -> dict:
    """Score both answers of unjudged shadow rows of a quality-mode role blind with the evaluator, until `usd` is
    spent. Each item (`items`: the list field holding the ideas) is scored on `rubric`; voice-rule breaks
    (planner.caption_issues) are counted too. The scores land in llm_shadow.judged. The judging calls are
    charged as shadow spend and are never shadowed themselves."""
    from .brain import rubrics
    from .llm import SHADOW_AGENT
    rc = _role_cfg(policy, role) or {}
    if rc.get("mode") != "quality":
        return {"role": role, "judged": 0, "note": "not a quality-mode role"}
    rubric = rubrics.load(rc.get("rubric", "idea"))
    rows = conn.execute("SELECT id, creator_id, primary_output, candidate_output FROM llm_shadow WHERE role = %s "
                        "AND candidate_output IS NOT NULL AND judged IS NULL AND creator_id IS NOT NULL "
                        "ORDER BY created_at DESC", (role,)).fetchall()
    spent0, done, was = llm.spent, 0, llm.shadowing
    llm.shadowing = False
    try:
        with llm.tagged(SHADOW_AGENT, None):
            for row in rows:
                if llm.spent - spent0 >= usd:
                    break
                sides = {side: _judge_answer(conn, policy, llm, rubric, rc, row["creator_id"], row[f"{side}_output"])
                         for side in ("primary", "candidate")}
                conn.execute("UPDATE llm_shadow SET judged = %s WHERE id = %s", (Jsonb(sides), row["id"]))
                done += 1
    finally:
        llm.shadowing = was
    return {"role": role, "judged": done, "usd": round(llm.spent - spent0, 4)}


def _judge_answer(conn, policy: Policy, llm, rubric, rc: dict, creator_id, answer: dict | None) -> dict:
    """Mean rubric score of an answer's items (each laid out as a live idea is) and its share of rule breaks."""
    from .brain import evaluation, ideas, planner
    from .brain.agents.evaluator import creator_summary
    _, who = creator_summary(conn, creator_id)
    items = ((answer or {}).get(rc.get("items", "ideas")) or [])[:int(rc.get("max_items", 6))]
    scores, breaks = [], 0
    for it in items:
        fmt = it.get("format", "photo")
        plan = {k: v for k, v in it.items() if k not in ("reasoning", "exploratory", "day", "platform", "canon_key",
                                                         "experiment")}
        text = evaluation.IDEA_TEMPLATE.format(format=fmt, concept=ideas.concept_text(fmt, it, it.get("caption")),
                                               plan=json.dumps(plan, ensure_ascii=False))
        s = evaluation.score(llm, policy, rubric, [who, text], role="evaluator", samples=1, creator_id=creator_id)
        vals = [v for v in s.medians.values() if v is not None]
        if vals:
            scores.append(mean(vals))
        breaks += bool(planner.caption_issues(policy, f"{it.get('caption', '')} {it.get('script', '')} "
                                                      f"{it.get('on_screen_text', '')}"))
    return {"score": round(mean(scores), 3) if scores else 0.0, "items": len(scores),
            "rule_breaks": round(breaks / max(1, len(items)), 3)}
