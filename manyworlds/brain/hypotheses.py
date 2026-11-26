"""Hypotheses, not permanent beliefs (owner 2026-11-26; docs/2026-11-26-learning-harness.md section 4.3-4.4).

Every finding and pattern card is a hypothesis with a record (migration 055): where it came from (`origin`: post,
creator, research, experiment), the contexts it applies to (`applies_to` tags), the level it holds at (`level`: post,
character, archetype, vertical, portfolio, production), the creators where it held (`replications`) and failed
(`contradictions`), when it was last tested and when it is due for a retest (`retest_after`; NULL: policy
brain.hypotheses.retest_days by grade after `last_tested`). Code, no model:

    weight(policy, f, now)        grade x origin x replication x decay: our own controlled experiments outrank every
                                  other kind of evidence; past its retest date a finding's weight decays until retested
    view(policy, f, now)          what a planning prompt shows of a finding: level, replications, age, weight, and
                                  "due for retest" for a stale one (a well-supported finding stays revisable)
    on_experiment(conn, policy, exp, result)
                                  cross-portfolio propagation: a creator's proven experiment opens a speculative
                                  archetype-level hypothesis for comparable creators; a comparable creator's own test
                                  of it adds a replication (raising its grade and, across archetypes, its level) or a
                                  contradiction (lowering it, and closing it once contradictions dominate)
    hypotheses_for(conn, policy, creator_id)
                                  the open archetype-level hypotheses a creator may test: offered to her showrunner and
                                  her experiment flow as suggestions, never applied as rules

Comparability (policy brain.hypotheses.propagation.comparable): creators sharing their primary archetype, or any
archetype. Shared intelligence only ever reaches another creator as an experiment she may run, so it cannot
homogenise the portfolio. Every read and write of the findings tables goes through manyworlds.brain.evidence.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..policy import Policy

LEVELS = ("post", "character", "archetype", "vertical", "portfolio", "production")
ORIGINS = ("post", "creator", "research", "experiment")
SCOPE_LEVEL = {"post": "post", "creator": "character", "cluster": "vertical", "market": "vertical",
               "genre": "vertical", "portfolio": "portfolio", "archetype": "archetype"}


def _cfg(policy: Policy | None) -> dict:
    return (policy.get("brain", "hypotheses", default={}) if policy is not None else {}) or {}


def _prop(policy: Policy | None) -> dict:
    return _cfg(policy).get("propagation") or {}


def defaults(scope: str, evidence_kind: str, sources=()) -> dict:
    """The origin and level a finding has when its writer names none: an experiment's (experimental evidence, or a
    simulated one drawn from an experiment), research, a single post, else the creator's own observations."""
    from_experiment = any(isinstance(s, dict) and s.get("experiment_id") for s in sources or ())
    origin = ("experiment" if evidence_kind == "experimental" or from_experiment else
              "research" if evidence_kind == "research" else "post" if scope == "post" else "creator")
    return {"origin": origin, "level": SCOPE_LEVEL.get(scope, "vertical")}


def _ts(v) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, str):
        v = datetime.fromisoformat(v)
    return v if v.tzinfo else v.replace(tzinfo=timezone.utc)


def due(policy: Policy | None, f: dict) -> datetime | None:
    """When the finding is due for a retest: its own retest_after, else last tested (or created) + retest_days."""
    if f.get("retest_after"):
        return _ts(f["retest_after"])
    base = _ts(f.get("last_tested") or f.get("created_at"))
    days = (_cfg(policy).get("retest_days") or {}).get(f.get("grade"))
    return base + timedelta(days=float(days)) if base is not None and days is not None else None


def decay(policy: Policy | None, f: dict, now: datetime) -> float:
    """1 until the retest date; then halves every decay.half_life_days, down to decay.floor."""
    d = due(policy, f)
    if d is None or now <= d:
        return 1.0
    c = _cfg(policy).get("decay") or {}
    half, floor = float(c.get("half_life_days", 60)), float(c.get("floor", 0.25))
    return round(max(floor, 0.5 ** (((now - d).total_seconds() / 86400.0) / half)), 4)


def due_for_retest(policy: Policy | None, f: dict, now: datetime) -> bool:
    d = due(policy, f)
    return d is not None and now > d


def _count(xs) -> int:
    return len([x for x in (xs or []) if x])


def replication_factor(policy: Policy | None, f: dict) -> float:
    """(1 + replication_bonus x held, at most 1 + max_replication_bonus) / (1 + contradiction_weight x failed): each
    creator where it held raises the weight, each where it failed lowers it more."""
    c = _cfg(policy)
    held, failed = _count(f.get("replications")), _count(f.get("contradictions"))
    up = min(float(c.get("max_replication_bonus", 1.0)), float(c.get("replication_bonus", 0.25)) * held)
    return round((1.0 + up) / (1.0 + float(c.get("contradiction_weight", 1.5)) * failed), 4)


def weight(policy: Policy | None, f: dict, now: datetime | None = None) -> float:
    """Grade x origin x replication x decay (policy brain.hypotheses)."""
    now = now or datetime.now(timezone.utc)
    c = _cfg(policy)
    g = float((c.get("grade_weight") or {}).get(f.get("grade"), 1.0))
    o = float((c.get("origin_weight") or {}).get(f.get("origin") or "creator", 1.0))
    return round(g * o * replication_factor(policy, f) * decay(policy, f, now), 4)


def age_days(f: dict, now: datetime) -> int | None:
    t = _ts(f.get("last_tested") or f.get("created_at"))
    return max(0, int((now - t).total_seconds() // 86400)) if t is not None else None


def view(policy: Policy | None, f: dict, now: datetime | None = None) -> dict:
    """A finding as planning prompts show it: the statement and grade, its level, origin, where it held and failed,
    its age since last tested, its weight and, past its retest date, "due for retest" (still revisable)."""
    now = now or datetime.now(timezone.utc)
    out = {"finding": f["statement"], "grade": f["grade"], "level": f.get("level") or SCOPE_LEVEL.get(f.get("scope")),
           "origin": f.get("origin"), "replicated_in": _count(f.get("replications")),
           "contradicted_in": _count(f.get("contradictions")), "age_days": age_days(f, now),
           "weight": weight(policy, f, now)}
    if due_for_retest(policy, f, now):
        out["due_for_retest"] = True
    if f.get("label"):
        out["label"] = f["label"]
    return out


def views(policy: Policy | None, rows: list[dict], now: datetime | None = None) -> list[dict]:
    """Every finding's view, highest weight first (ties: newest first, as recorded)."""
    now = now or datetime.now(timezone.utc)
    vs = [view(policy, f, now) for f in rows]
    return sorted(vs, key=lambda v: -v["weight"])


# --- comparability and propagation ----------------------------------------------------------------------------------
def creator_archetypes(conn, policy: Policy | None, creator_id) -> list[str]:
    """The creator's archetypes, primary first (her latest profile, else her bible; manyworlds.brain.archetypes)."""
    from . import archetypes
    row = conn.execute("SELECT body FROM creator_profiles WHERE creator_id = %s ORDER BY version DESC LIMIT 1",
                       (creator_id,)).fetchone()
    who = (row or {}).get("body")
    if not who:
        c = conn.execute("SELECT bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
        who = (c or {}).get("bible") or {}
    return archetypes.for_creator(policy, who)


def comparable(policy: Policy | None, mine: list[str], theirs: list[str]) -> bool:
    """Whether two creators are comparable (policy brain.hypotheses.propagation.comparable): the same primary
    archetype, or (any_archetype) any archetype in common."""
    if not mine or not theirs:
        return False
    if str(_prop(policy).get("comparable", "any_archetype")) == "primary_archetype":
        return mine[0] == theirs[0]
    return bool(set(mine) & set(theirs))


def comparable_creators(conn, policy: Policy | None, creator_id) -> list[str]:
    """Other creators (not retired) comparable to this one."""
    mine = creator_archetypes(conn, policy, creator_id)
    out = []
    for r in conn.execute("SELECT id FROM creators WHERE id <> %s AND status <> 'retired' ORDER BY created_at",
                          (creator_id,)).fetchall():
        if comparable(policy, mine, creator_archetypes(conn, policy, r["id"])):
            out.append(str(r["id"]))
    return out


def _applies(policy: Policy | None, h: dict, mine: list[str]) -> bool:
    a = h.get("applies_to") or {}
    return comparable(policy, mine, [x for x in [a.get("archetype"), *(a.get("archetypes") or [])] if x])


def _tested_by(h: dict) -> set[str]:
    return {str(x.get("creator_id")) for x in (h.get("replications") or []) + (h.get("contradictions") or [])
            if isinstance(x, dict)}


def hypotheses_for(conn, policy: Policy | None, creator_id) -> list[dict]:
    """The open archetype-level hypotheses this creator may test: comparable to her, not yet tested on her, in her
    evidence mode, highest weight first. Suggestions only, never rules."""
    from . import evidence
    if not _prop(policy).get("enabled", True):
        return []
    mine = creator_archetypes(conn, policy, creator_id)
    mode = evidence.mode_for(conn, creator_id)
    now = datetime.now(timezone.utc)
    rows = [h for h in evidence.archetype_hypotheses(conn, mode=mode)
            if _applies(policy, h, mine) and str(creator_id) not in _tested_by(h)]
    return sorted(rows, key=lambda h: -weight(policy, h, now))


def standing(policy: Policy | None, h: dict) -> dict:
    """A hypothesis' grade, level and status from its replications and contradictions: tentative once it held in
    tentative_after creators (more than it failed), portfolio level once it held across portfolio_after_archetypes
    archetypes, closed (`contradicted`) once contradictions outnumber replications by retire_after_contradictions.
    Never proven: proof for a creator is her own experiment."""
    p = _prop(policy)
    held = [x for x in (h.get("replications") or []) if isinstance(x, dict)]
    failed = [x for x in (h.get("contradictions") or []) if isinstance(x, dict)]
    grade = "tentative" if len(held) >= int(p.get("tentative_after", 2)) and len(held) > len(failed) else "speculative"
    kinds = {x.get("archetype") for x in held if x.get("archetype")}
    level = "portfolio" if len(kinds) >= int(p.get("portfolio_after_archetypes", 2)) else "archetype"
    status = "contradicted" if len(failed) - len(held) >= int(p.get("retire_after_contradictions", 2)) else "open"
    return {"grade": grade, "level": level, "status": status}


def _sign(x) -> int:
    x = float(x or 0)
    return (x > 0) - (x < 0)


def on_experiment(conn, policy: Policy, exp: dict, result: dict) -> dict | None:
    """After a content experiment concluded: if it tested an archetype-level hypothesis (its source finding), record
    the creator as a replication (the same direction, significant) or a contradiction (no difference, or the other
    way); else, if it proved something (a proven finding was written), record it on the open hypothesis for the same
    effect that applies to her, or open one at archetype level for comparable creators. None when propagation is off
    or there is nothing to record."""
    from . import evidence
    if not _prop(policy).get("enabled", True) or not exp.get("effect_key"):
        return None
    cid, key = str(exp["creator_id"]), exp["effect_key"]
    mode = evidence.mode_for(conn, cid)
    mine = creator_archetypes(conn, policy, cid)
    entry = {"creator_id": cid, "archetype": mine[0] if mine else None, "experiment_id": str(exp["id"]),
             "effect": result.get("effect"), "at": datetime.now(timezone.utc).isoformat()}
    source = evidence.finding(conn, exp["source_finding"]) if exp.get("source_finding") else None
    target = source if source and source.get("scope") == "archetype" else None
    if target is None:
        if result.get("status") == "no_difference" or not result.get("finding_id"):
            return None
        target = next((h for h in evidence.archetype_hypotheses(conn, mode=mode, effect_key=key)
                       if _applies(policy, h, mine)), None)
    if target is not None:
        if cid in _tested_by(target):
            return None
        held = result.get("status") != "no_difference" and _sign(result.get("effect")) == _sign(target.get("effect"))
        reps = list(target.get("replications") or []) + ([entry] if held else [])
        cons = list(target.get("contradictions") or []) + ([] if held else [entry])
        new = standing(policy, {**target, "replications": reps, "contradictions": cons})
        evidence.record_test(conn, target["id"], replications=reps, contradictions=cons, **new)
        return {"hypothesis_id": str(target["id"]), "held": held, **new}
    if not mine:
        return None
    statement = (f"Hypothesis for {mine[0]} creators, from one creator's experiment (a suggestion to test, not a "
                 f"rule): {exp.get('hypothesis') or key}")
    sources = [{"effects": [key]}, {"finding_id": str(result["finding_id"])}, {"creator_id": cid},
               {"experiment_id": str(exp["id"])}] + ([{"table": "simulated_metrics"}] if mode == "sandbox" else [])
    hid = evidence.record_finding(
        conn, scope="archetype", statement=statement, evidence="simulated" if mode == "sandbox" else "experimental",
        grade="speculative", effect=result.get("effect"),
        interval=(result.get("low"), result.get("high")) if result.get("low") is not None else None,
        n=(result.get("n_a") or 0) + (result.get("n_b") or 0), sources=sources, origin="experiment",
        level="archetype", applies_to={"archetype": mine[0], "archetypes": mine, "effect_key": key,
                                       "domain": "content"},
        replications=[entry], parent_id=result["finding_id"])
    return {"hypothesis_id": str(hid), "opened": True}


def planning_views(conn, policy: Policy | None, creator_id, now: datetime | None = None) -> list[dict]:
    """The hypotheses offered to a creator's showrunner (at most propagation.shown), as views with the experiment
    they would test."""
    now = now or datetime.now(timezone.utc)
    shown = int(_prop(policy).get("shown", 5))
    out = []
    for h in hypotheses_for(conn, policy, creator_id)[:shown]:
        out.append({**view(policy, h, now), "tests": (h.get("applies_to") or {}).get("effect_key")})
    return out


def casting_views(conn, policy: Policy | None, now: datetime | None = None) -> list[dict]:
    """Every open archetype-level hypothesis with real evidence, for casting: what has held for which archetype and
    in how many creators (suggestions about archetypes, never proof about a new creator)."""
    from . import evidence
    now = now or datetime.now(timezone.utc)
    rows = evidence.archetype_hypotheses(conn, mode="real")
    vs = [{**view(policy, h, now), "archetype": (h.get("applies_to") or {}).get("archetype")} for h in rows]
    return sorted(vs, key=lambda v: (-v["replicated_in"], -v["weight"]))
