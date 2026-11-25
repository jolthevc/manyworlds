"""The creative pattern library: photo and video patterns in one vocabulary (policy/patterns, then pattern_cards).

A pattern card extends the aesthetics library's shot vocabulary with the dimensions that make a picture or a video
work: situation x shot types x pose x camera x expression x styling x lighting x social energy x movement, and for
video the move, the talking-edit grammar, deliveries, pace, eye contact and mannerisms. Each says why it works
(mechanism), for whom (audience role, best-fit archetypes), how we make it (recipe in our own names), its failure
modes and guardrails, and later its evidence (source links to the public posts it was learned from; one creator may
be enough), our production acceptance and our real outcomes.

    library(policy)        the seed file's cards (policy `intelligence.patterns.file`), keyed by name
    validate(policy, lib)  every name a card uses checked against what the pipeline renders: problems, proposals
    seed(conn, policy)     the seed file's cards into pattern_cards (status illustrative), once per key and version
    signature_elements()   what a creator's cards let recur (the aesthetics library's `signature` rule)

Illustrative cards are craft knowledge, not research: they never count as evidence (`counts_as_evidence`).
    cards(conn, policy)    the live library: pattern_cards' latest versions plus seed cards without a row
    retrieve(...)          the top-k cards for a stage (showrunner, director, casting, style) by fit x weight
    brief(cards)           compact prompt lines per card (never source links)
    usage(...)             what an idea or post records in body.pattern_cards (offered, applied), for attribution
    talk_overrides/rules   a video card's talk-edit parameters (clamped) and performance band for the director
    feedback(conn, policy) acceptance (production) and real outcomes (evidence.py) per card, recomputed

Retrieval (spec section 4, task 11) reaches the showrunner (which card an idea follows: format, hook, move, edit
grammar), the director (shot types, pose, camera, expression, styling, light, movement, edit), talk production
(grammar and clamped punch-in rhythm, cutaway share, hook) and casting (the creator lab's bible, the style
director). Weight = evidence (illustrative cards count half) x production acceptance x real outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import get_args

import yaml
from psycopg.types.json import Jsonb

from ..policy import Policy

MEDIA = ("photo", "carousel", "video", "sequence")
STATUSES = ("illustrative", "draft", "active", "retired")
AUDIENCE_ROLES = ("attraction", "intimacy", "humour", "aspiration", "utility")
DIMENSIONS = ("pose", "camera", "expression", "styling", "lighting", "social_energy", "movement")
DEFAULT_FILE = "patterns-v0.yaml"


def _find_root() -> Path:
    for candidate in (Path(__file__).resolve().parents[2] / "policy" / "patterns",
                      Path.cwd() / "policy" / "patterns", Path("/app/policy/patterns")):
        if candidate.exists():
            return candidate
    return Path.cwd() / "policy" / "patterns"


ROOT = _find_root()


@lru_cache(maxsize=8)
def _load(path: str, digest: str) -> dict:   # digest keys the cache on content, so an edited file is re-read
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def library(policy: Policy | None = None) -> dict:
    name = (policy.get("intelligence", "patterns", "file", default=DEFAULT_FILE) if policy else DEFAULT_FILE) \
        or DEFAULT_FILE
    path = ROOT / name
    if not path.exists():
        return {"version": None, "cards": {}}
    lib = _load(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    return {"version": lib.get("version") or name, "cards": dict(lib.get("cards") or {})}


def counts_as_evidence(card: dict) -> bool:
    return card.get("status") in ("draft", "active") and bool(card.get("sources"))


def two_proportion_p(p1: float, n1: int, p2: float, n2: int) -> float | None:
    """Two-sided p of a difference between two shares (pooled z test); None when a side is empty."""
    if n1 <= 0 or n2 <= 0:
        return None
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0 if p1 == p2 else 0.0
    z = abs(p1 - p2) / se
    return round(math.erfc(z / math.sqrt(2)), 4)


def _bcfg(policy: Policy | None) -> dict:
    return _rcfg(policy).get("basis") or {}


def basis(policy: Policy | None, card: dict) -> dict:
    """What kind of knowledge a card is, from code, never from a model (owner 2026-11-25): {kind, ...}.
    validated: our own posts' real outcomes back it; correlation: its strongest high-vs-low traction attribute
    difference is significant (intelligence.retrieval.basis); observed: researched, not significant; judgment: craft
    knowledge with no research behind it (illustrative)."""
    c, w = _bcfg(policy), (_rcfg(policy).get("weights") or {})
    out = card.get("outcomes") or {}
    if int(out.get("n") or 0) >= int(w.get("min_outcomes", 3)) and out.get("median_ratio") is not None:
        return {"kind": "validated", "n": int(out["n"]), "median_ratio": out["median_ratio"]}
    if not counts_as_evidence(card):
        return {"kind": "judgment"}
    ev = card.get("evidence") or {}
    best = None
    for t in ev.get("tests") or []:
        if (t.get("p") is not None and t["p"] <= float(c.get("max_p", 0.1))
                and min(int(t.get("n_high") or 0), int(t.get("n_low") or 0)) >= int(c.get("min_group", 5))):
            if best is None or t["p"] < best["p"]:
                best = t
    if best:
        return {"kind": "correlation", **{k: best[k] for k in ("attribute", "high", "low", "p", "n_high", "n_low")}}
    return {"kind": "observed", "n_posts": int(ev.get("n_posts") or len(card.get("sources") or []))}


def _names(policy: Policy | None) -> dict[str, set[str]]:
    from .. import direction
    from . import aesthetics, archetypes
    p = policy
    talk_edit = (p.get("generation", "talk_edit", "grammars", default={}) if p else {}) or {}
    deliveries = (p.get("voice", "delivery_bank", "deliveries", default={}) if p else {}) or {}
    refs = (p.get("dance", "references", default=[]) if p else []) or []
    moves = (p.get("moves", "library", default={}) if p else {}) or {}
    devices = {d for d in get_args(get_args(direction.FramingSpec.model_fields["device"].annotation)[0])}
    return {"shot_types": set(aesthetics.names(p)), "grades": set(aesthetics.grades(p)),
            "archetypes": set(archetypes.names(p)), "moves": set(moves), "grammars": set(talk_edit),
            "deliveries": set(deliveries), "motion_moves": {str(r.get("move")) for r in refs if r.get("move")},
            "devices": devices}


def _hook_kinds() -> set[str]:
    from ..research import annotation
    return set(get_args(annotation.HookKind))


def validate(policy: Policy | None, lib: dict | None = None) -> dict:
    """{"problems": [...], "proposed_shot_types": {card: [names]}}. A problem is a name the pipeline doesn't know
    (or a card that breaks the library's rules); a proposed shot type is a new one the card defines for the aesthetics
    library to take, reported and never used until it does."""
    lib = lib if lib is not None else library(policy)
    known = _names(policy)
    problems: list[str] = []
    proposed: dict[str, list[str]] = {}
    for key, card in (lib.get("cards") or {}).items():
        where = f"card {key}"
        card = card or {}
        if card.get("medium") not in MEDIA:
            problems.append(f"{where}: medium {card.get('medium')!r} is not one of {MEDIA}")
        if card.get("status") not in STATUSES:
            problems.append(f"{where}: status {card.get('status')!r} is not one of {STATUSES}")
        if card.get("status") == "illustrative" and card.get("sources"):
            problems.append(f"{where}: an illustrative card has no sources (it is craft knowledge, not research)")
        for field in ("mechanism", "situation"):
            if not str(card.get(field) or "").strip():
                problems.append(f"{where}: no {field}")
        dims = card.get("dims") or {}
        new = set((card.get("proposed_shot_types") or {}).keys())
        for s in dims.get("shot_types") or []:
            if s in new:
                proposed.setdefault(key, []).append(s)
            elif s not in known["shot_types"]:
                problems.append(f"{where}: shot type {s!r} is not in the aesthetics library (propose it under "
                                f"proposed_shot_types)")
        if not dims.get("shot_types"):
            problems.append(f"{where}: no shot types")
        for d in DIMENSIONS:
            if d not in dims:
                problems.append(f"{where}: dimension {d} missing")
        for r in card.get("audience_role") or []:
            if r not in AUDIENCE_ROLES:
                problems.append(f"{where}: audience role {r!r} is not one of {AUDIENCE_ROLES}")
        for a in card.get("best_fit") or []:
            if a not in known["archetypes"]:
                problems.append(f"{where}: archetype {a!r} is not in the archetype library")
        recipe = card.get("recipe") or {}
        checks = (("grade", "grades"), ("move", "moves"), ("grammar", "grammars"), ("device", "devices"))
        for field, names in checks:
            if recipe.get(field) and recipe[field] not in known[names]:
                problems.append(f"{where}: recipe {field} {recipe[field]!r} is unknown")
        for d in recipe.get("deliveries") or []:
            if d not in known["deliveries"]:
                problems.append(f"{where}: delivery {d!r} is not in voice.delivery_bank.deliveries")
        for m in recipe.get("motion_moves") or []:
            if m not in known["motion_moves"]:
                problems.append(f"{where}: motion move {m!r} has no licensed reference in dance.references")
        edit = recipe.get("edit") or {}
        for k in edit:
            if k not in EDIT_KEYS:
                problems.append(f"{where}: recipe edit {k!r} is not one of {EDIT_KEYS}")
        if edit.get("hook") and edit["hook"] not in _hook_kinds():
            problems.append(f"{where}: recipe edit hook {edit['hook']!r} is unknown")
        if card.get("medium") in ("video",) and not (recipe.get("move") or recipe.get("grammar")):
            problems.append(f"{where}: a video card names a move or a talking-edit grammar")
    return {"problems": problems, "proposed_shot_types": proposed}


def seed(conn, policy: Policy | None) -> int:
    """The seed file's cards into pattern_cards at version 1 (illustrative, no sources); an existing key is left as
    research has it. Returns how many were added."""
    lib = library(policy)
    n = 0
    for key, card in lib["cards"].items():
        body = {k: v for k, v in card.items() if k not in ("medium", "status", "signature_ok", "sources")}
        row = conn.execute(
            """INSERT INTO pattern_cards (key, version, status, medium, signature_ok, body, evidence)
               VALUES (%s, 1, %s, %s, %s, %s, %s) ON CONFLICT (key, version) DO NOTHING RETURNING id""",
            (key, card.get("status") or "illustrative", card["medium"], bool(card.get("signature_ok")), Jsonb(body),
             Jsonb({"seed": lib["version"], "n_posts": 0, "n_creators": 0}))).fetchone()
        n += bool(row)
    return n


def signature_elements(cards: list[dict]) -> list[str]:
    """The cards whose element may recur (signature_ok): the director may repeat their pose, set or outfit across
    posts and carousels, outside the variety rules (aesthetics rules.signature)."""
    return [str(c.get("key") or c.get("name")) for c in cards if c.get("signature_ok")]


# --- retrieval into creative direction ----------------------------------------------------------------------------
EDIT_KEYS = ("punch_every_s", "cutaway_share", "hook")


def _rcfg(policy: Policy | None) -> dict:
    return (policy.get("intelligence", "retrieval", default={}) if policy else {}) or {}


def _as_card(key: str, card: dict, **extra) -> dict:
    return {**(card or {}), "key": key, "id": None, "version": 0, "sources": list((card or {}).get("sources") or []),
            "evidence": {}, "acceptance": {}, "outcomes": {}, **extra}


def cards(conn, policy: Policy | None) -> list[dict]:
    """The live library: each key's latest version in pattern_cards (a retired one hides the key), plus the seed
    file's cards that have no row yet (id None, version 0). Each card is its body plus key, id, version, status,
    medium, signature_ok, sources, evidence, acceptance and outcomes."""
    out: dict[str, dict | None] = {}
    if conn is not None:
        for r in conn.execute("SELECT DISTINCT ON (key) * FROM pattern_cards ORDER BY key, version DESC").fetchall():
            if r["status"] == "retired":
                out[r["key"]] = None
                continue
            out[r["key"]] = {**(r["body"] or {}), "key": r["key"], "id": str(r["id"]), "version": r["version"],
                             "status": r["status"], "medium": r["medium"], "signature_ok": r["signature_ok"],
                             "sources": r["sources"] or [], "evidence": r["evidence"] or {},
                             "acceptance": r["acceptance"] or {}, "outcomes": r["outcomes"] or {}}
    for key, card in library(policy)["cards"].items():
        if key not in out:
            out[key] = _as_card(key, card)
    return [c for c in out.values() if c]


def media_for(policy: Policy | None, fmt: str | None) -> list[str] | None:
    """The card media that fit a post format (policy intelligence.retrieval.format_media); None: any."""
    if not fmt:
        return None
    m = (_rcfg(policy).get("format_media") or {}).get(fmt)
    return list(m) if m else None


def weight(policy: Policy | None, card: dict) -> float:
    """Evidence x production acceptance x real outcomes (policy intelligence.retrieval.weights)."""
    w = _rcfg(policy).get("weights") or {}
    if counts_as_evidence(card):
        n = int((card.get("evidence") or {}).get("n_posts") or len(card.get("sources") or []))
        ev = min(float(w.get("evidence_max", 3.0)),
                 float(w.get("evidence_base", 1.0)) + float(w.get("evidence_per_log_post", 0.4)) * math.log1p(n))
    else:
        ev = float(w.get("illustrative", 0.5))
    acc = card.get("acceptance") or {}
    pa, pd = float(w.get("prior_accepted", 2)), float(w.get("prior_dropped", 1))
    tried, accepted = float(acc.get("tried") or 0), float(acc.get("accepted") or 0)
    acceptance = ((accepted + pa) / (tried + pa + pd)) / (pa / (pa + pd)) if pa > 0 else 1.0
    out = card.get("outcomes") or {}
    lo, hi = (w.get("outcome_clip") or [0.5, 2.0])[:2]
    outcome = 1.0
    if int(out.get("n") or 0) >= int(w.get("min_outcomes", 3)) and out.get("median_ratio") is not None:
        outcome = max(float(lo), min(float(hi), float(out["median_ratio"])))
    kind = float((_bcfg(policy).get("weight") or {}).get(basis(policy, card)["kind"], 1.0))
    return round(ev * acceptance * outcome * kind, 4)


def fit(policy: Policy | None, card: dict, archetype_names: list[str], media: list[str] | None,
        shot_types: list[str] | None) -> float:
    """How well a card fits: her archetypes in its best_fit, its medium among the format's, her shot families in its
    shot types (policy intelligence.retrieval.fit). A medium the format can't show is no fit at all (0)."""
    f = _rcfg(policy).get("fit") or {}
    if media and card.get("medium") not in media:
        return 0.0
    score = 0.0
    if set(card.get("best_fit") or []) & set(archetype_names or []):
        score += float(f.get("archetype", 2.0))
    if media:
        score += float(f.get("medium", 1.0))
    st = list((card.get("dims") or {}).get("shot_types") or [])
    if shot_types and st:
        score += float(f.get("shot_types", 1.0)) * len(set(st) & set(shot_types)) / len(st)
    return score


def retrieve(conn, policy: Policy | None, *, stage: str, who: dict | None = None, fmt: str | None = None,
             media: list[str] | None = None, shot_types: list[str] | None = None, pinned=(), k: int | None = None
             ) -> list[dict]:
    """The top-k cards for a creative stage (policy intelligence.retrieval.k.<stage>), by fit x weight; cards named in
    `pinned` (an idea's chosen card) come first whatever their score. Each card carries `_score`."""
    cfg = _rcfg(policy)
    if not cfg.get("enabled", True):
        return []
    from . import archetypes
    k = int(k if k is not None else (cfg.get("k") or {}).get(stage, 3))
    if k <= 0:
        return []
    names = archetypes.for_creator(policy, who) if who else []
    media = media if media is not None else media_for(policy, fmt)
    floor = float((cfg.get("fit") or {}).get("min", 1.0))
    pinned = [p for p in pinned if p]
    scored = []
    for c in cards(conn, policy):
        f = fit(policy, c, names, media, shot_types)
        if c["key"] in pinned or f >= floor:
            scored.append({**c, "_score": round(f * weight(policy, c), 4)})
    scored.sort(key=lambda c: (c["key"] not in pinned, -c["_score"], c["key"]))
    return scored[:max(k, len([c for c in scored if c["key"] in pinned]))]


def _evidence_line(card: dict, policy: Policy | None = None) -> str:
    b = basis(policy, card)
    if b["kind"] == "judgment":
        line = "ILLUSTRATIVE JUDGMENT: craft knowledge, unproven"
    else:
        ev = card.get("evidence") or {}
        line = (f"researched: {ev.get('n_posts') or len(card.get('sources') or [])} posts from "
                f"{ev.get('n_creators') or '?'} creators")
        if b["kind"] == "correlation":
            line = (f"CORRELATION (not proof of cause): {b['attribute']} in {round(b['high'] * 100)}% of high-traction "
                    f"vs {round(b['low'] * 100)}% of low-traction posts (p={b['p']}, n={b['n_high']}/{b['n_low']}); "
                    + line)
        elif b["kind"] == "observed":
            line = "OBSERVED (no significant difference yet): " + line
        else:
            line = "VALIDATED on our own posts: " + line
    acc = card.get("acceptance") or {}
    if acc.get("tried"):
        line += f"; our renders {acc.get('accepted', 0)}/{acc['tried']} accepted"
    out = card.get("outcomes") or {}
    if out.get("n"):
        line += f"; real outcome {out.get('median_ratio')}x her median over {out['n']} posts"
    return line


def brief(cards_: list[dict], *, detail: str = "full") -> str:
    """Compact lines per card for a prompt: what it is, why it works, its dimensions and recipe in our names, what
    breaks it. Never its source links. detail "plan": the showrunner's view (shot types and recipe only)."""
    lines = []
    for c in cards_:
        dims = c.get("dims") or {}
        lines.append(f"- {c['key']} ({c.get('medium')}; {_evidence_line(c)}): {c.get('situation', '')}. "
                     f"Why it works: {c.get('mechanism', '')} For: {', '.join(c.get('audience_role') or [])}.")
        recipe = dict(c.get("recipe") or {})
        if detail == "plan":
            lines.append(f"  shots: {', '.join(dims.get('shot_types') or [])}; recipe: "
                         f"{json.dumps(recipe, separators=(',', ':'))}")
            continue
        dim_text = "; ".join(f"{d} {dims[d]}" for d in DIMENSIONS if dims.get(d))
        lines.append(f"  shots: {', '.join(dims.get('shot_types') or [])}; {dim_text}")
        lines.append(f"  recipe: {json.dumps(recipe, separators=(',', ':'))}; avoid: "
                     f"{'; '.join(c.get('failure_modes') or [])}; guardrails: {c.get('guardrails', '')}")
    return "\n".join(lines)


def used_in(card: dict, shots: list[dict] | None = None, body: dict | None = None) -> bool:
    """Whether a piece of direction used the card: a shot of one of its shot types, or the idea's move or grammar."""
    st = set((card.get("dims") or {}).get("shot_types") or [])
    if shots and any((s or {}).get("shot_type") in st for s in shots):
        return True
    recipe = card.get("recipe") or {}
    body = body or {}
    return bool((recipe.get("move") and body.get("move") == recipe["move"])
                or (recipe.get("grammar") and body.get("edit_grammar") == recipe["grammar"]))


def usage(cards_: list[dict], stage: str, *, shots: list[dict] | None = None, body: dict | None = None,
          chosen=()) -> list[dict]:
    """What an idea or post records (body.pattern_cards): each card offered to the stage, its id and version, and
    whether the direction applied it (chosen by name, or its shot types, move or grammar used)."""
    return [{"key": c["key"], "id": c.get("id"), "version": c.get("version"), "stage": stage,
             "applied": c["key"] in chosen or used_in(c, shots, body)} for c in cards_]


def merge_usage(old: list[dict] | None, new: list[dict]) -> list[dict]:
    seen = {(u.get("key"), u.get("stage")): u for u in old or []}
    for u in new:
        seen[(u["key"], u["stage"])] = u
    return list(seen.values())


def chosen(body: dict | None) -> list[str]:
    """The cards an idea applied (by the showrunner or director), in order."""
    return [u["key"] for u in (body or {}).get("pattern_cards") or [] if u.get("applied")]


def card(conn, policy: Policy | None, key: str | None) -> dict | None:
    return next((c for c in cards(conn, policy) if c["key"] == key), None) if key else None


def talk_overrides(policy: Policy | None, card_: dict | None) -> dict:
    """A video card's talk-edit overrides (recipe.edit), clamped to intelligence.retrieval.talk_bounds."""
    edit = ((card_ or {}).get("recipe") or {}).get("edit") or {}
    bounds = _rcfg(policy).get("talk_bounds") or {}
    out: dict = {}
    every = edit.get("punch_every_s")
    if isinstance(every, (list, tuple)) and len(every) == 2:
        lo, hi = (bounds.get("punch_every_s") or [1.5, 5.0])[:2]
        a, b = sorted(max(float(lo), min(float(hi), float(x))) for x in every)
        out["punch_every_s"] = [a, b]
    if edit.get("cutaway_share") is not None:
        lo, hi = (bounds.get("cutaway_share") or [0.0, 0.5])[:2]
        out["cutaway_share"] = round(max(float(lo), min(float(hi), float(edit["cutaway_share"]))), 2)
    if edit.get("hook"):
        out["hook"] = str(edit["hook"])
    return out


def talk_rules(card_: dict | None, overrides: dict | None = None) -> list[str]:
    """A video card's performance band and edit, as director rules for a talking post."""
    if not card_:
        return []
    r, o = card_.get("recipe") or {}, overrides or {}
    rules = []
    if r.get("pace_wps"):
        rules.append(f"Lines are written for about {r['pace_wps'][0]}-{r['pace_wps'][-1]} words a second: short "
                     f"spoken sentences, no long build-up.")
    if r.get("eye_contact") is not None:
        rules.append(f"She looks into the lens about {round(float(r['eye_contact']) * 100)}% of the time "
                     f"(eyeline 'lens'); otherwise at what her hands do.")
    if r.get("deliveries"):
        rules.append(f"Deliveries: {', '.join(r['deliveries'])}.")
    if r.get("mannerisms"):
        rules.append(f"Mannerisms, once each: {', '.join(r['mannerisms'])}.")
    if o.get("cutaway_share") is not None:
        rules.append(f"Cutaways about {round(o['cutaway_share'] * 100)}% of the running time.")
    if o.get("hook"):
        rules.append(f"The hook in the first 2 seconds is carried by: {o['hook'].replace('_', ' ')}.")
    return rules


# --- outcome feedback ---------------------------------------------------------------------------------------------
def feedback(conn, policy: Policy) -> dict:
    """Every card's production acceptance and real outcomes, recomputed from the posts that applied it
    (body.pattern_cards). Acceptance: posts by status (intelligence.feedback.accepted_statuses / dropped_statuses).
    Outcomes: through evidence.post_outcomes in `real` mode only (a sandbox creator's simulated outcomes never reach a
    card): each post's reach over its creator's median reach. Idempotent: a rerun writes the same numbers."""
    from . import evidence
    cfg = policy.get("intelligence", "feedback", default={}) or {}
    if not cfg.get("enabled", True):
        return {}
    seed(conn, policy)
    ok, bad = set(cfg.get("accepted_statuses") or []), set(cfg.get("dropped_statuses") or [])
    n_recent = int(cfg.get("recent_posts", 200))
    by_creator: dict[str, dict] = {}

    def creator_outcomes(cid: str) -> dict:
        if cid not in by_creator:
            got = evidence.post_outcomes(conn, cid, n_recent, "real", policy=policy) \
                if evidence.mode_for(conn, cid) == "real" else {}
            reaches = sorted(float(o["reach"]) for o in got.values() if o.get("reach") is not None)
            by_creator[cid] = {"posts": got, "median": reaches[len(reaches) // 2] if reaches else None}
        return by_creator[cid]

    res = {"cards": 0, "posts": 0}
    for c in conn.execute("SELECT DISTINCT ON (key) id, key FROM pattern_cards ORDER BY key, version DESC").fetchall():
        posts = conn.execute(
            "SELECT id, creator_id, status, body->>'drop_reason' AS why FROM posts WHERE body->'pattern_cards' "
            "@> jsonb_build_array(jsonb_build_object('key', %s::text, 'applied', true))", (c["key"],)).fetchall()
        acc = {"tried": 0, "accepted": 0, "dropped": 0, "drop_reasons": {}}
        ratios = []
        for p in posts:
            if p["status"] in ok:
                acc["accepted"] += 1
            elif p["status"] in bad:
                acc["dropped"] += 1
                why = str(p["why"] or "unknown").split(":")[0][:60]
                acc["drop_reasons"][why] = acc["drop_reasons"].get(why, 0) + 1
            else:
                continue                       # still in production: not judged yet
            acc["tried"] += 1
            co = creator_outcomes(str(p["creator_id"]))
            o = co["posts"].get(str(p["id"]))
            if o and co["median"]:
                ratios.append(float(o["reach"]) / co["median"])
        ratios.sort()
        out = {"n": len(ratios), "median_ratio": round(ratios[len(ratios) // 2], 3) if ratios else None,
               "mode": "real"}
        conn.execute("UPDATE pattern_cards SET acceptance = %s, outcomes = %s, updated_at = now() WHERE id = %s",
                     (Jsonb(acc), Jsonb(out), c["id"]))
        res["cards"] += 1
        res["posts"] += acc["tried"]
    return res
