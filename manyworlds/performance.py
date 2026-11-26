"""Performance direction: one control language for directing virtual talent, and its translation per model.

The library (policy/performance/<file>, named by policy `performance.file`; docs/2026-11-26-performance-direction.md)
holds ~14 controls (energy, confidence, playfulness, sass, ... candidness, polish), each with 5 levels (0 none ... 4
extreme) and, per level, observable behaviour per channel (face, gaze, body, walk, voice, camera). Every agent reads
and writes the same names: the talent manager's performance profile (creator.dossier section `performance`), the
director's shot sheet (direction.PerformanceSpec.intent / controls), the judges' observations, calibration.

    library(policy)                         the library, validated (`problems` says what is wrong with one)
    apply_context(baseline, context, ...)   a creator's baseline levels under a context shift (outfit_check, luxury...)
    compose(intent, controls, target, ...)  the performance text for one target model: the intent sentence first, then
                                            only the most salient controls' behaviours, a few phrases per channel, as
                                            one coherent performance (conflicts resolved by context, level 4 only with
                                            a reason, calibration offsets applied)
    edges(profile, policy)                  natural beginnings and endings for talk and i2v (never a blank stare)
    voice_params(controls, engine, policy)  IndexTTS emotion strength and energy, pause scale by pace, a Qwen3 hook
    closeness(requested, observed)          1 = observed exactly as requested, 0 = four levels off everywhere
    leakage(requested, observed, library)   held controls that moved without being expected to (`leaks`)

Observation rides on the review calls we already make (review.common.performance_*): the judge returns observed levels
for the shot's directed controls plus appeal, charisma, expression and naturalness. They rank candidates among those
that pass the defect gates (selection points, a variety penalty), never reject, and are stored per asset in
`performance_observations` (requested vs observed). Calibration (`refresh_calibration`, the chief of staff's tick)
adjusts a model's level offsets only from reliable observations (the library's `calibration` floors, judge agreement
measured first; policy `performance.calibration.enabled`), one step per refresh, each change a decision row.
"""

from __future__ import annotations

import hashlib
import math
import re
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from psycopg.types.json import Jsonb

# The control names, static so a strict structured-output schema can enumerate them (direction.ControlLevel). A test
# keeps them equal to the shipped library's; a new control is added to both.
CONTROL_NAMES = ("energy", "confidence", "playfulness", "sass", "flamboyance", "elegance", "warmth",
                 "facial_expressiveness", "gesture", "movement", "vocal_expressiveness", "pace", "candidness", "polish")
CHANNELS = ("face", "gaze", "body", "walk", "voice", "camera")
LEVELS = (0, 1, 2, 3, 4)
DEFAULT_FILE = "performance-v1.yaml"
# Words that make a phrase a negation (a negation in a prompt tends to summon the thing): the library is positive.
_NEGATION = re.compile(r"\b(no|not|never|without|nothing|none|neither|nor)\b|n't\b", re.IGNORECASE)


def _find_root() -> Path:
    for candidate in (Path(__file__).resolve().parents[1] / "policy" / "performance",
                      Path.cwd() / "policy" / "performance", Path("/app/policy/performance")):
        if candidate.exists():
            return candidate
    return Path.cwd() / "policy" / "performance"


ROOT = _find_root()


def _get(policy: Any, *path: str, default: Any = None) -> Any:
    """A policy value from a Policy or a plain dict (direction's tests pass dicts)."""
    if policy is None:
        return default
    if hasattr(policy, "get") and not isinstance(policy, dict):
        return policy.get(*path, default=default)
    node: Any = policy
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


@lru_cache(maxsize=8)
def _load(path: str, digest: str) -> dict:   # digest keys the cache on content, so an edited file is re-read
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def library(policy: Any = None) -> dict:
    """The performance library named by policy `performance.file` (the default file without a policy). A missing file
    is an empty library: nothing is compiled, observed or calibrated."""
    name = _get(policy, "performance", "file", default=DEFAULT_FILE) or DEFAULT_FILE
    path = ROOT / str(name)
    if not path.exists():
        return {"version": None, "controls": {}, "targets": {}, "contexts": {}, "compose": {}, "edges": {},
                "observe": {}, "calibration": {}, "voice": {}, "energy_alias": {}}
    lib = dict(_load(str(path), hashlib.sha256(path.read_bytes()).hexdigest()))
    lib.setdefault("version", str(name))
    for key in ("controls", "targets", "contexts", "compose", "edges", "observe", "calibration", "voice",
                "energy_alias"):
        lib[key] = lib.get(key) or {}
    return lib


def problems(lib: dict) -> list[str]:
    """What is wrong with a library, one line each (empty when it is sound): control names, levels 0-4, known channels,
    positive phrases, contexts/conflicts/favours naming known controls, targets naming known channels."""
    out: list[str] = []
    controls = lib.get("controls") or {}
    for name, c in controls.items():
        for lvl, chans in ((c or {}).get("levels") or {}).items():
            if not isinstance(lvl, int) or lvl not in LEVELS:
                out.append(f"controls.{name}: level {lvl!r} is not one of {list(LEVELS)}")
            for ch, phrase in (chans or {}).items():
                if ch not in CHANNELS:
                    out.append(f"controls.{name}.{lvl}: unknown channel {ch!r}")
                if _NEGATION.search(str(phrase)):
                    out.append(f"controls.{name}.{lvl}.{ch}: a negation ({phrase!r}); phrase it positively")
        for leak in (c or {}).get("leaks") or []:
            if leak not in controls:
                out.append(f"controls.{name}.leaks: unknown control {leak!r}")
        if not (c or {}).get("bipolar") and 0 in ((c or {}).get("levels") or {}):
            out.append(f"controls.{name}: level 0 of a control that isn't bipolar says nothing; leave it out")
    for target, chans in (lib.get("targets") or {}).items():
        for ch in chans or []:
            if ch not in CHANNELS:
                out.append(f"targets.{target}: unknown channel {ch!r}")
    for ctx, offsets in (lib.get("contexts") or {}).items():
        for name in offsets or {}:
            if name not in controls:
                out.append(f"contexts.{ctx}: unknown control {name!r}")
    comp = lib.get("compose") or {}
    for i, c in enumerate(comp.get("conflicts") or []):
        for name in c.get("pair") or []:
            if name not in controls:
                out.append(f"compose.conflicts[{i}]: unknown control {name!r}")
    for ctx, names in (comp.get("favours") or {}).items():
        for name in names or []:
            if name not in controls:
                out.append(f"compose.favours.{ctx}: unknown control {name!r}")
    for target, e in (lib.get("edges") or {}).items():
        for key in ("opening", "closing"):
            if _NEGATION.search(str((e or {}).get(key) or "")):
                out.append(f"edges.{target}.{key}: a negation; phrase it positively")
    return out


def names(policy: Any = None) -> list[str]:
    return list(library(policy)["controls"])


def _level(value) -> int | None:
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(LEVELS[0], min(LEVELS[-1], n))


def levels_of(controls, lib: dict | None = None) -> dict[str, int]:
    """Controls as {name: level}, from a list of {control, level} (the shot sheet's strict form) or a dict; unknown
    names dropped (when a library is given), levels clamped to 0-4, the last of a repeated name wins."""
    known = set((lib or {}).get("controls") or ()) or set(CONTROL_NAMES)
    out: dict[str, int] = {}
    items = controls.items() if isinstance(controls, dict) else [
        ((c.get("control") if isinstance(c, dict) else getattr(c, "control", None)),
         (c.get("level") if isinstance(c, dict) else getattr(c, "level", None))) for c in (controls or [])]
    for name, lvl in items:
        n = _level(lvl)
        if name in known and n is not None:
            out[str(name)] = n
    return out


def energy_level(word: str | None, policy: Any = None) -> int | None:
    """The energy control's level for a shot sheet's legacy energy word (low/medium/high)."""
    return (library(policy)["energy_alias"] or {}).get(str(word or "")) if word else None


def energy_word(level: int | None, policy: Any = None) -> str:
    """The legacy energy word nearest a level (what direction's idle-movement line reads)."""
    alias = library(policy)["energy_alias"] or {}
    if level is None or not alias:
        return ""
    return min(alias, key=lambda w: (abs(int(alias[w]) - int(level)), -int(alias[w])))


def context_offsets(context, policy: Any = None, *, profile: dict | None = None) -> dict[str, int]:
    """A context's level offsets: a dict as given, else the creator's own (profile contexts) by name, else the
    library's; each clamped to the library's compose.max_context_offset."""
    lib = library(policy)
    cap = int((lib["compose"] or {}).get("max_context_offset", 2))
    if isinstance(context, dict):
        raw = context
    elif context:
        raw = ((profile or {}).get("contexts") or {}).get(str(context)) or (lib["contexts"] or {}).get(str(context)) or {}
    else:
        raw = {}
    out = {}
    for name, off in (raw or {}).items():
        try:
            out[str(name)] = max(-cap, min(cap, int(off)))
        except (TypeError, ValueError):
            continue
    return out


def apply_context(baseline: dict | None, context, policy: Any = None, *, profile: dict | None = None) -> dict[str, int]:
    """A creator's baseline levels shifted by a context (a name or {control: offset}), clamped to 0-4. A control the
    baseline doesn't name starts from the library's neutral level when the context moves it."""
    lib = library(policy)
    neutral = int((lib["compose"] or {}).get("neutral_level", 2))
    out = dict(levels_of(baseline or {}, lib))
    for name, off in context_offsets(context, policy, profile=profile).items():
        out[name] = max(LEVELS[0], min(LEVELS[-1], out.get(name, neutral) + off))
    return out


def _phrase(lib: dict, control: str, level: int, channel: str) -> str:
    return str(((((lib["controls"].get(control) or {}).get("levels") or {}).get(level) or {}).get(channel)) or "").strip()


def compose(intent: str, controls, target: str, policy: Any = None, *, baseline: dict | None = None,
            context=None, profile: dict | None = None, extreme_reason: str = "", channels: list | None = None) -> dict:
    """The performance for one target model (photo, i2v, talk, voice), as one coherent text:

        {text, intent, used: [{control, requested, level}], phrases: {channel: [...]}, notes: [...]}

    `controls` are the shot's levels ({name: level} or [{control, level}]); `baseline` the creator's levels (else her
    profile's `baseline`), `context` the format's shift (a name, or offsets), `profile` her performance profile
    (natural_range, contexts, and `calibration`: {target: {control: {level: offset}}} from refresh_calibration).
    Steps (the library's `compose`): level 4 only with `extreme_reason`; levels kept inside her natural range;
    conflicting pairs resolved (the context's favourite wins, else the higher level, ties to the first); the
    controls ranked by salience (distance from her baseline under the context, then from neutral) and the top
    max_controls kept; calibration offsets applied; at most per_channel phrases per channel, the most salient first;
    the intent sentence, then the target's channels in order."""
    lib = library(policy)
    comp = lib["compose"] or {}
    profile = profile or {}
    notes: list[str] = []
    want = levels_of(controls, lib)
    requested = dict(want)
    cap4 = int(comp.get("max_level_without_reason", 3))
    if not str(extreme_reason or "").strip():
        for name, lvl in list(want.items()):
            if lvl > cap4:
                want[name] = cap4
                notes.append(f"{name} {lvl} -> {cap4}: level {lvl} needs a stated reason")
    for name, rng in (profile.get("natural_range") or {}).items():
        if name in want and isinstance(rng, (list, tuple)) and len(rng) == 2:
            lo, hi = int(rng[0]), int(rng[1])
            if not lo <= want[name] <= hi:
                clamped = max(lo, min(hi, want[name]))
                notes.append(f"{name} {want[name]} -> {clamped}: outside her natural range {lo}-{hi}")
                want[name] = clamped
    ctx_name = context if isinstance(context, str) else ""
    favoured = list((comp.get("favours") or {}).get(ctx_name) or [])
    for c in comp.get("conflicts") or []:
        pair = [p for p in (c.get("pair") or []) if p in want]
        if len(pair) != 2:
            continue
        a, b = pair
        at, cap = int(c.get("at", 3)), int(c.get("cap", 2))
        if want[a] < at or want[b] < at:
            continue
        if a in favoured or b in favoured:
            winner = a if (a in favoured and (b not in favoured or favoured.index(a) <= favoured.index(b))) else b
            why = f"the {ctx_name} context favours {winner}"
        else:
            winner = a if want[a] >= want[b] else b
            why = f"{winner} is the stronger request"
        loser = b if winner == a else a
        notes.append(f"{loser} {want[loser]} -> {cap}: conflicts with {winner} ({why})")
        want[loser] = cap
    base = apply_context(baseline if baseline is not None else profile.get("baseline"), context, policy,
                         profile=profile)
    neutral = int(comp.get("neutral_level", 2))
    order = list(lib["controls"])

    def salience(name: str) -> tuple:
        lvl = want[name]
        return (-abs(lvl - base.get(name, neutral)), -abs(lvl - neutral), order.index(name) if name in order else 99)

    def speaks(name: str) -> bool:
        lvl = want[name]
        if lvl == 0 and not (lib["controls"].get(name) or {}).get("bipolar"):
            return False
        return lvl != base.get(name, neutral) or lvl != neutral
    ranked = sorted((n for n in want if speaks(n)), key=salience)
    kept = ranked[:int(comp.get("max_controls", 4))]
    dropped = [n for n in ranked[len(kept):]]
    if dropped:
        notes.append(f"less salient, left to the intent: {', '.join(dropped)}")
    offsets = ((profile.get("calibration") or {}).get(target) or {})
    channels = list(channels if channels is not None else ((lib["targets"] or {}).get(target) or []))
    per_channel = int(comp.get("per_channel", 2))
    phrases: dict[str, list[str]] = {ch: [] for ch in channels}
    used = []
    for name in kept:
        lvl = want[name]
        off = int(((offsets.get(name) or {}).get(str(lvl)) or (offsets.get(name) or {}).get(lvl) or 0))
        compiled = max(LEVELS[0], min(LEVELS[-1], lvl + off))
        used.append({"control": name, "requested": requested.get(name, lvl), "level": lvl, "compiled": compiled})
        for ch in channels:
            text = _phrase(lib, name, compiled, ch)
            if text and len(phrases[ch]) < per_channel and text not in phrases[ch]:
                phrases[ch].append(text)
    labels = comp.get("channel_labels") or {}
    intent = " ".join(str(intent or "").split()).rstrip(".")
    parts = [f"{labels.get(ch, ch)}: {', '.join(phrases[ch])}" for ch in channels if phrases[ch]]
    text = "; ".join(x for x in [intent, *parts] if x)
    return {"text": text, "intent": intent, "used": used, "phrases": {k: v for k, v in phrases.items() if v},
            "notes": notes, "version": lib.get("version")}


def edges(profile: dict | None, policy: Any = None, target: str = "talk") -> dict[str, str]:
    """How a talking (talk) or moving (i2v) clip begins and ends: the creator's signature entrance and exit
    (performance profile `opening`/`closing`), else the library's defaults for the target. Never a start from, or
    a return to, a blank neutral stare."""
    lib = library(policy)
    p = profile or {}
    default = (lib["edges"] or {}).get(target) or {}
    return {k: str(p.get(k) or "").strip() or str(default.get(k) or "").strip() for k in ("opening", "closing")}


def voice_params(controls, engine: str | None, policy: Any = None) -> dict:
    """How a line's voice-channel controls reach a voice engine:

        {alpha, boost: {emotion: add}, pause_scale, instruct}

    IndexTTS: `alpha` (emo_alpha) from vocal_expressiveness, `boost` added to the emotion vector per energy level above
    `energy_from`; every engine: `pause_scale` for the shot's pauses from pace; `instruct`, the voice phrases, a hook
    for an engine that takes an instruction (only used where policy voice.engines.<engine>.instruct is on). Controls
    the line doesn't set leave the engine's own defaults (None / {} / 1.0)."""
    lib = library(policy)
    voice = lib["voice"] or {}
    lv = levels_of(controls, lib)
    out: dict = {"alpha": None, "boost": {}, "pause_scale": 1.0, "instruct": ""}
    pace = voice.get("pause_scale_by_pace") or {}
    if "pace" in lv and (lv["pace"] in pace or str(lv["pace"]) in pace):
        out["pause_scale"] = float(pace.get(lv["pace"], pace.get(str(lv["pace"]), 1.0)))
    if engine == "indextts":
        ix = voice.get("indextts") or {}
        alpha = ix.get("emo_alpha_by_level") or {}
        if "vocal_expressiveness" in lv:
            v = alpha.get(lv["vocal_expressiveness"], alpha.get(str(lv["vocal_expressiveness"])))
            out["alpha"] = float(v) if v is not None else None
        above = lv.get("energy", 0) - int(ix.get("energy_from", 2))
        if "energy" in lv and above > 0:
            out["boost"] = {k: round(float(v) * above, 4) for k, v in (ix.get("energy_boost") or {}).items()}
    phrases = compose("", lv, "voice", policy)["phrases"].get("voice") or []
    out["instruct"] = ", ".join(phrases)
    return out


def apply_voice(emotion: dict | None, params: dict, engine: str | None, policy: Any = None) -> dict | None:
    """An engine's emotion control ({name, vector, alpha}) with voice_params applied: alpha replaced, the energy boost
    added to the named weights (IndexTTS vector order from the library). A boost on a line with no vector (neutral)
    starts from zeros. None stays None when there is nothing to apply."""
    if engine != "indextts":
        return emotion
    order = list(((library(policy)["voice"] or {}).get("indextts") or {}).get("vector_order") or [])
    boost = {k: v for k, v in (params.get("boost") or {}).items() if k in order and v}
    if emotion is None and not boost:
        return None
    out = dict(emotion or {"name": "energy", "vector": [0.0] * len(order), "alpha": 1.0})
    vec = [float(x) for x in (out.get("vector") or [0.0] * len(order))]
    for k, v in boost.items():
        i = order.index(k)
        if i < len(vec):
            vec[i] = round(vec[i] + float(v), 4)
    out["vector"] = vec
    if params.get("alpha") is not None:
        out["alpha"] = float(params["alpha"])
    return out


def closeness(requested: dict | None, observed: dict | None) -> float | None:
    """How close the observed levels are to the requested ones, over the controls both name: 1 - mean |difference|
    / 4. None when they share no control."""
    req, obs = levels_of(requested or {}), levels_of(observed or {})
    common = [k for k in req if k in obs]
    if not common:
        return None
    return round(1 - sum(abs(req[k] - obs[k]) for k in common) / (len(common) * (LEVELS[-1] - LEVELS[0])), 3)


def leakage(requested: dict | None, observed: dict | None, lib: dict | None = None, *, moved=None,
            baseline: dict | None = None, threshold: int = 1) -> dict:
    """Held controls that moved: for each requested control that is not one of `moved` (default: those away from the
    baseline, else the one furthest from neutral), its observed level off by `threshold` or more is leakage, unless a
    moved control lists it in `leaks` (expected to move with it). {leaks: [{control, requested, observed, shift}],
    tolerated: [...], held: n, score: share of held controls that leaked}."""
    lib = lib if lib is not None else library()
    req, obs = levels_of(requested or {}, lib), levels_of(observed or {}, lib)
    neutral = int((lib.get("compose") or {}).get("neutral_level", 2))
    if moved is None:
        if baseline:
            moved = [k for k in req if req[k] != levels_of(baseline, lib).get(k, neutral)]
        else:
            moved = [max(req, key=lambda k: abs(req[k] - neutral))] if req else []
    moved = list(moved)
    expected = {x for m in moved for x in ((lib.get("controls") or {}).get(m) or {}).get("leaks") or []}
    held = [k for k in req if k not in moved and k in obs]
    leaks, tolerated = [], []
    for k in held:
        shift = obs[k] - req[k]
        if abs(shift) >= threshold:
            (tolerated if k in expected else leaks).append({"control": k, "requested": req[k], "observed": obs[k],
                                                           "shift": shift})
    return {"leaks": leaks, "tolerated": tolerated, "held": len(held),
            "score": round(len(leaks) / len(held), 3) if held else None}


# --- prompts the agents read ----------------------------------------------------------------------------------------

def vocabulary(policy: Any = None) -> dict:
    """The control language for a prompt: {controls: {name: means}, levels, contexts}. Static for a library version."""
    lib = library(policy)
    return {"controls": {n: str((c or {}).get("means") or "") for n, c in lib["controls"].items()},
            "levels": {str(k): v for k, v in (lib.get("levels") or {}).items()},
            "contexts": lib["contexts"]}


def profile_view(profile: dict | None) -> dict:
    """A creator's performance profile as planners and the director read it (compact, empty without one)."""
    p = profile or {}
    keys = ("baseline", "natural_range", "signature", "opening", "closing", "contexts")
    return {k: p[k] for k in keys if p.get(k)}


def observe_controls(shot_controls, policy: Any = None, *, baseline: dict | None = None) -> list[str]:
    """The directed controls a judge is asked to observe for a shot: at most observe.max_controls, the most salient
    first (distance from the baseline, then from neutral)."""
    lib = library(policy)
    lv = levels_of(shot_controls, lib)
    neutral = int((lib["compose"] or {}).get("neutral_level", 2))
    base = levels_of(baseline or {}, lib)
    order = list(lib["controls"])
    ranked = sorted(lv, key=lambda n: (-abs(lv[n] - base.get(n, neutral)), -abs(lv[n] - neutral),
                                       order.index(n) if n in order else 99))
    return ranked[:int((lib["observe"] or {}).get("max_controls", 6))]


def selection_points(scores: dict, close: float | None, policy: Any = None) -> float | None:
    """A candidate's performance points among those that pass the defect gates: observe.selection_weights over its
    scores (1-10) and its closeness (0-1, counted on the same 10-point scale). None without any score."""
    w = (library(policy)["observe"] or {}).get("selection_weights") or {}
    total, any_ = 0.0, False
    for k, weight in w.items():
        v = (close * 10 if close is not None else None) if k == "closeness" else scores.get(k)
        if v is None:
            continue
        total += float(weight) * float(v)
        any_ = True
    return round(total, 3) if any_ else None


_WORDS = re.compile(r"[a-z']+")
_STOP = frozenset({"a", "an", "the", "and", "or", "with", "of", "to", "in", "on", "at", "her", "she", "is", "one"})


def _word_set(text: str) -> set[str]:
    return {w for w in _WORDS.findall(str(text or "").lower()) if w not in _STOP}


def similar(a: dict, b: dict, policy: Any = None) -> bool:
    """Whether two judged frames repeat each other (variety): their pose/expression descriptors overlap by
    observe.variety_similarity or more (Jaccard over words), or both have the same non-empty observed levels."""
    sim = float((library(policy)["observe"] or {}).get("variety_similarity", 0.6))
    wa, wb = _word_set(a.get("descriptor") or ""), _word_set(b.get("descriptor") or "")
    if wa and wb and len(wa & wb) / len(wa | wb) >= sim:
        return True
    oa, ob = levels_of(a.get("observed") or {}), levels_of(b.get("observed") or {})
    return bool(oa) and oa == ob


def variety_penalty(policy: Any = None) -> float:
    return float((library(policy)["observe"] or {}).get("variety_penalty", 0.0))


# --- observations ----------------------------------------------------------------------------------------------------

def record_observations(conn, rows: list[dict]) -> int:
    """Store requested vs observed per asset and review stage (performance_observations); a stage already stored for
    an asset is left as it is (idempotent). Returns how many were written."""
    n = 0
    for r in rows:
        cur = conn.execute(
            """INSERT INTO performance_observations (asset_id, post_id, creator_id, job_id, model, judge, stage, target,
                   intent, requested, compiled, observed, scores, descriptor, phrases, closeness, points,
                   policy_version, library_version)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (asset_id, stage) DO NOTHING""",
            (r["asset_id"], r.get("post_id"), r.get("creator_id"), r.get("job_id"), r.get("model"), r.get("judge"),
             r.get("stage") or "screen", r.get("target"), r.get("intent") or "", Jsonb(r.get("requested") or {}),
             Jsonb(r.get("compiled") or {}), Jsonb(r.get("observed") or {}), Jsonb(r.get("scores") or {}),
             r.get("descriptor") or "", Jsonb(r.get("phrases") or {}), r.get("closeness"), r.get("points"),
             r.get("policy_version"), r.get("library_version")))
        n += cur.rowcount or 0
    return n


def observations_for(conn, asset_ids: list) -> dict[str, dict]:
    """The newest observation per asset (confirm over screen): {asset_id: {descriptor, observed, scores}}."""
    if not asset_ids:
        return {}
    rows = conn.execute(
        """SELECT DISTINCT ON (asset_id) asset_id, descriptor, observed, scores FROM performance_observations
           WHERE asset_id = ANY(%s::uuid[]) ORDER BY asset_id, (stage = 'confirm') DESC, created_at DESC""",
        ([str(a) for a in asset_ids],)).fetchall()
    return {str(r["asset_id"]): dict(r) for r in rows}


# --- calibration -----------------------------------------------------------------------------------------------------

def _cfg(policy: Any) -> dict:
    return _get(policy, "performance", default={}) or {}


def _family(model: str | None) -> str:
    """A judge's model family ('claude-haiku-5-5' -> claude, 'qwen4-max' -> qwen): a second judge must be another."""
    m = re.match(r"[a-z]+", str(model or "").lower())
    return m.group(0) if m else ""


def judge_agreement(conn, policy: Any = None) -> dict[str, dict]:
    """Per judge model: {agreement, pairs, source}. From the owner's own ratings (decisions of kind
    `performance_rating`: output {asset_id, observed: {control: level}}) where there are any, else from a judge of
    another model family that observed the same assets. agreement = share of paired control observations within the
    library's agreement_tolerance. A judge with fewer than min_agreement_pairs pairs has agreement None."""
    cal = library(policy)["calibration"] or {}
    tol = int(cal.get("agreement_tolerance", 0))
    min_pairs = int(cal.get("min_agreement_pairs", 20))
    obs = conn.execute("SELECT asset_id, judge, observed FROM performance_observations "
                       "WHERE observed <> '{}'::jsonb").fetchall()
    by_asset: dict[str, list] = {}
    for r in obs:
        by_asset.setdefault(str(r["asset_id"]), []).append(r)
    owner = conn.execute("SELECT output FROM decisions WHERE kind = 'performance_rating'").fetchall()
    rated = {str((r["output"] or {}).get("asset_id")): levels_of((r["output"] or {}).get("observed") or {})
             for r in owner if (r["output"] or {}).get("asset_id")}
    owner_pairs: dict[str, list] = {}
    second_pairs: dict[str, list] = {}

    def pair(into: dict, judge: str, a: dict, b: dict):
        into.setdefault(judge, []).extend(abs(a[k] - b[k]) <= tol for k in a if k in b)
    for aid, rows in by_asset.items():
        for r in rows:
            mine = levels_of(r["observed"] or {})
            if aid in rated:
                pair(owner_pairs, r["judge"], mine, rated[aid])
            for other in rows:
                if other is not r and _family(other["judge"]) != _family(r["judge"]):
                    pair(second_pairs, r["judge"], mine, levels_of(other["observed"] or {}))
    stats = {}
    for judge in set(owner_pairs) | set(second_pairs):
        # the owner's own ratings outrank a second judge once there are enough of them
        use, source = ((owner_pairs[judge], "owner") if len(owner_pairs.get(judge) or []) >= min_pairs
                       else (second_pairs.get(judge) or owner_pairs.get(judge) or [],
                             "second_judge" if second_pairs.get(judge) else "owner"))
        stats[judge] = {"agree": sum(use), "pairs": len(use), "source": source}
    out = {}
    for judge, s in stats.items():
        ok = s["pairs"] >= min_pairs
        out[judge] = {"agreement": round(s["agree"] / s["pairs"], 3) if ok and s["pairs"] else None,
                      "pairs": s["pairs"], "source": s["source"]}
    return out


def calibration_cells(conn, policy: Any = None, *, judge: str | None = None, creator_id=None) -> list[dict]:
    """Requested vs observed per (model, target, control, requested level) from performance_observations (one judge's
    when named): {model, target, control, level, n, posts, mean_shift, low, high} (shift = observed - requested;
    low/high its interval, mean +/- z standard errors)."""
    z = float((library(policy)["calibration"] or {}).get("z", 1.96))
    rows = conn.execute(
        """SELECT o.model, o.target, o.post_id, o.job_id, o.asset_id, o.judge, o.requested, o.observed, o.compiled
           FROM performance_observations o
           WHERE (%(j)s::text IS NULL OR o.judge = %(j)s) AND (%(c)s::uuid IS NULL OR o.creator_id = %(c)s::uuid)
             AND o.observed <> '{}'::jsonb""", {"j": judge, "c": str(creator_id) if creator_id else None}).fetchall()
    cells: dict[tuple, dict] = {}
    for r in rows:
        req, obs = levels_of(r["requested"] or {}), levels_of(r["observed"] or {})
        for k, lvl in req.items():
            if k not in obs:
                continue
            c = cells.setdefault((r["model"], r["target"], k, lvl), {"shifts": [], "posts": set()})
            c["shifts"].append(obs[k] - lvl)
            c["posts"].add(str(r["post_id"] or r["job_id"] or r["asset_id"]))   # a post, else its job
    out = []
    for (model, target, control, lvl), c in sorted(cells.items(), key=lambda x: tuple(str(v) for v in x[0])):
        xs = c["shifts"]
        n = len(xs)
        mean = sum(xs) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
        half = z * sd / math.sqrt(n) if n > 1 else float("inf")
        out.append({"model": model, "target": target, "control": control, "level": lvl, "n": n,
                    "posts": len(c["posts"]), "mean_shift": round(mean, 3),
                    "low": round(mean - half, 3) if math.isfinite(half) else None,
                    "high": round(mean + half, 3) if math.isfinite(half) else None})
    return out


def reliable(cell: dict, policy: Any = None) -> str | None:
    """Why a cell may not adjust its offset (None: it may): too few samples or posts, too small a shift, an interval
    that includes zero."""
    cal = library(policy)["calibration"] or {}
    if cell["n"] < int(cal.get("min_samples", 12)):
        return f"{cell['n']} samples, needs {int(cal.get('min_samples', 12))}"
    if cell["posts"] < int(cal.get("min_assets", 6)):
        return f"{cell['posts']} posts, needs {int(cal.get('min_assets', 6))}"
    if abs(cell["mean_shift"]) < float(cal.get("min_shift", 0.75)):
        return f"mean shift {cell['mean_shift']}, under {float(cal.get('min_shift', 0.75))}"
    if cell["low"] is None or cell["high"] is None or cell["low"] <= 0 <= cell["high"]:
        return f"interval [{cell['low']}, {cell['high']}] includes zero"
    return None


def offsets(conn, policy: Any = None, *, model: str | None = None) -> dict:
    """The current calibration offsets {target: {control: {level: offset}}} (one model's when named)."""
    rows = conn.execute("SELECT model, target, control, level, level_offset FROM performance_calibration "
                        "WHERE level_offset <> 0 AND (%s::text IS NULL OR model = %s)", (model, model)).fetchall()
    out: dict = {}
    for r in rows:
        out.setdefault(r["target"], {}).setdefault(r["control"], {})[str(r["level"])] = int(r["level_offset"])
    return out


def profile_offsets(conn, policy: Any = None) -> dict:
    """The offsets a shot is directed under, per target for the model policy performance.models names for it:
    {target: {control: {level: offset}}}."""
    out = {}
    for target, model in (_cfg(policy).get("models") or {}).items():
        got = offsets(conn, policy, model=str(model)).get(target)
        if got:
            out[target] = got
    return out


def refresh_calibration(conn, policy: Any = None) -> dict:
    """One calibration step (the chief of staff's tick), while policy performance.calibration.enabled: for each cell
    of the most reliable judge (agreement at least min_judge_agreement), a cell that passes `reliable` moves its
    offset one level against the shift (a model that undershoots is asked one higher), never past max_offset; every
    change is a decision row (kind performance_calibration). Returns {enabled, judge, agreement, changed, held,
    reason}; nothing changes until agreement is measured, and the reason says so."""
    if not (_cfg(policy).get("calibration") or {}).get("enabled"):
        return {"enabled": False, "changed": [], "reason": "performance.calibration.enabled is off"}
    cal = library(policy)["calibration"] or {}
    floor = float(cal.get("min_judge_agreement", 0.6))
    agree = judge_agreement(conn, policy)
    # the best-agreeing judge; between equals, the one with the most requested-vs-observed samples (a second judge
    # sampled only for agreement has none)
    good = sorted(((a["agreement"], sum(c["n"] for c in calibration_cells(conn, policy, judge=j)), j)
                   for j, a in agree.items() if a["agreement"] is not None and a["agreement"] >= floor), reverse=True)
    if not good:
        return {"enabled": True, "changed": [], "agreement": agree,
                "reason": f"no judge has measured agreement >= {floor} (owner ratings or a second judge family)"}
    score, _, judge = good[0]
    cap = int(cal.get("max_offset", 1))
    changed, held = [], []
    from .policy import Policy
    version = policy.version if isinstance(policy, Policy) else None
    for cell in calibration_cells(conn, policy, judge=judge):
        why = reliable(cell, policy)
        if why:
            held.append({**cell, "reason": why})
            continue
        cur = conn.execute("SELECT level_offset FROM performance_calibration WHERE model = %s AND target = %s "
                           "AND control = %s AND level = %s",
                           (cell["model"], cell["target"], cell["control"], cell["level"])).fetchone()
        before = int(cur["level_offset"]) if cur else 0
        step = 1 if cell["mean_shift"] < 0 else -1          # undershoot: ask higher
        after = max(-cap, min(cap, before + step))
        if after == before:
            held.append({**cell, "reason": f"offset already at the cap {before}"})
            continue
        conn.execute(
            """INSERT INTO performance_calibration (model, target, control, level, level_offset, judge, stats,
                                                   updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, now()) ON CONFLICT (model, target, control, level) DO UPDATE SET
                   level_offset = EXCLUDED.level_offset, judge = EXCLUDED.judge, stats = EXCLUDED.stats,
                   updated_at = now()""",
            (cell["model"], cell["target"], cell["control"], cell["level"], after, judge, Jsonb(cell)))
        conn.execute("INSERT INTO decisions (kind, inputs, output, rationale, policy_version) VALUES "
                     "('performance_calibration', %s, %s, %s, %s)",
                     (Jsonb({**cell, "judge": judge, "agreement": score}), Jsonb({"before": before, "after": after}),
                      f"{cell['model']} {cell['target']}: {cell['control']} asked at {cell['level']} came out "
                      f"{cell['mean_shift']:+} on average over {cell['n']} observations; offset {before} -> {after}",
                      version))
        changed.append({**cell, "before": before, "after": after})
    return {"enabled": True, "judge": judge, "agreement": score, "changed": changed, "held": len(held)}


def report(conn, policy: Any = None, *, creator_id=None) -> dict:
    """`mw performance report`: requested vs observed per cell, the judges' agreement, the current offsets and why
    calibration does or doesn't adjust."""
    agree = judge_agreement(conn, policy)
    cells = calibration_cells(conn, policy, creator_id=creator_id)
    for c in cells:
        c["calibrates"] = reliable(c, policy) or "reliable"
    scores = conn.execute(
        """SELECT target, count(*) n, avg((scores->>'appeal')::numeric) appeal,
                  avg((scores->>'naturalness')::numeric) naturalness, avg(closeness) closeness
           FROM performance_observations WHERE (%s::uuid IS NULL OR creator_id = %s::uuid) GROUP BY target""",
        (str(creator_id) if creator_id else None, str(creator_id) if creator_id else None)).fetchall()
    cal_on = bool((_cfg(policy).get("calibration") or {}).get("enabled"))
    floor = float((library(policy)["calibration"] or {}).get("min_judge_agreement", 0.6))
    measured = [j for j, a in agree.items() if a["agreement"] is not None and a["agreement"] >= floor]
    state = ("off (performance.calibration.enabled)" if not cal_on else
             "on" if measured else f"on, waiting: no judge with measured agreement >= {floor}")
    return {"library": library(policy).get("version"), "calibration": state, "judges": agree,
            "offsets": offsets(conn, policy),
            "by_target": [{k: (float(v) if isinstance(v, Decimal) else v) for k, v in dict(r).items()}
                          for r in scores],
            "cells": cells}
