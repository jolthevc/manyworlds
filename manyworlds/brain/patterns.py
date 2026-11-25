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
Retrieval into the director, showrunner and talk production is a later step (spec section 4, task 11).
"""

from __future__ import annotations

import hashlib
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
