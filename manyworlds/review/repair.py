"""Targeted repair: why a scene's images were rejected, and what the next attempt changes because of it.

Diagnosis reads the review rows of the rejected job (manyworlds.review.gates writes one per gate per image):
the gate that failed most is the cause, except that any safety failure makes the cause `safety`. Each cause
has its own fix (policy `review.repair`):
  identity  the likeness LoRA gets stronger by `lora_scale_step` per identity repair, up to `lora_scale_max`,
            and the prompt asks for a clearly visible face with the signature details
  scene     the prompt leads with the scene, so the setting and action can't be lost
  realism   the prompt asks for what renders well (relaxed hands, plain surfaces) and the job's negative_prompt
            names what renders badly (text, logos, warped hands): hints never put a negation in a positive prompt
  safety    never regenerated: the post is dropped
A new seed always comes with it. Repairs still count against `review.max_repairs_per_post`.
"""

from __future__ import annotations

from collections import Counter

from ..policy import Policy

ORDER = ("safety", "identity", "scene", "realism")   # ties go to the earlier cause


def reviewed(conn, job_id) -> bool:
    return bool(conn.execute("SELECT 1 FROM reviews r JOIN assets a ON a.id = r.asset_id WHERE a.job_id = %s LIMIT 1",
                             (job_id,)).fetchone())


def diagnose(conn, job_id) -> dict:
    """Rows the judge never filled in (it skipped a candidate; the gate fails closed with details.error) say
    nothing about the image, so they are ignored. Only an apparent minor or sexualised content is `safety`:
    a stray brand mark usually goes with a new seed, so it counts as `realism` (whose hint asks for no logos)."""
    rows = [r for r in conn.execute(
        "SELECT r.gate, r.passed, r.details FROM reviews r JOIN assets a ON a.id = r.asset_id "
        "WHERE a.job_id = %s AND a.kind = 'image'", (job_id,)).fetchall() if not (r["details"] or {}).get("error")]
    failed: Counter = Counter()
    for r in rows:
        if r["passed"]:
            continue
        d = r["details"] or {}
        if r["gate"] == "safety":
            failed["safety" if (d.get("looks_under_min_age") or d.get("sexualized")) else "realism"] += 1
        else:
            # another place, or a face in a cutaway: lead with the scene; extra hands: a realism repair
            failed[{"set_match": "scene", "framing": "scene", "anatomy": "realism"}.get(r["gate"], r["gate"])] += 1
    if not failed:
        return {"cause": None, "failed": {}, "defects": []}
    cause = "safety" if failed.get("safety") else max(
        failed, key=lambda g: (failed[g], -ORDER.index(g) if g in ORDER else -len(ORDER)))
    defects = Counter(d for r in rows if not r["passed"] for d in ((r["details"] or {}).get("defects") or []))
    return {"cause": cause, "failed": dict(failed), "defects": [d for d, _ in defects.most_common(3)]}


NEGATIONS = ("no ", "not ", "without ", "avoid ", "never ", "don't ", "do not ")


def split_hint(hint) -> tuple[str, list[str]]:
    """(positive text, things to avoid) of one repair hint (policy `review.repair.hints.<cause>`). A hint is a string
    (positive wording) or {positive, avoid: [...]}. A negation never stays in a positive prompt (a model with no
    negative channel tends to draw what it is told not to, manyworlds.direction): a clause of a string hint that
    starts with one ("no text or logos") moves to the avoid list."""
    if isinstance(hint, dict):
        avoid = hint.get("avoid") or []
        text, avoid = str(hint.get("positive") or ""), [avoid] if isinstance(avoid, str) else list(avoid)
    else:
        text, avoid = str(hint or ""), []
    kept = []
    for clause in text.split(","):
        bare = clause.strip().lstrip(".").strip()
        neg = next((n for n in NEGATIONS if bare.lower().startswith(n)), None)
        if neg:
            avoid.append(bare[len(neg):].strip().rstrip("."))
        else:
            kept.append(clause)
    return ",".join(kept), [str(a).strip() for a in avoid if str(a).strip()]


def with_negative(payload: dict, avoid: list[str], policy: Policy | None = None, target: str = "photo") -> dict:
    """`payload` with `avoid` added to its `negative_prompt`, each item once. A target whose model has no negative
    channel (Z-Image at guidance 0: every repaired scene photo) also gets the positive stand-ins of those items in
    its prompt (direction.positive_instead_of), since it ignores the field; a target with one starts from its
    standard negative list when the payload has none, so the repair adds to it instead of replacing it."""
    if not avoid:
        return payload
    from .. import direction
    if policy is not None and not direction.has_negative_channel(target, policy):
        instead = {**(policy.get("direction", "positive_instead_of", default={}) or {}),
                   **(policy.get("review", "repair", "stand_ins", default={}) or {})}
        tail = direction._capitalised(direction._join(list(dict.fromkeys(
            str(instead[a]).strip() for a in avoid if str(instead.get(a) or "").strip()))))
        if tail and tail not in str(payload.get("prompt") or ""):
            payload = {**payload, "prompt": f"{str(payload.get('prompt') or '').rstrip()}. {tail}"}
    current = str(payload.get("negative_prompt") or "").strip()
    if not current and policy is not None and direction.has_negative_channel(target, policy):
        current = direction._negative_text([], target, policy)
    have = {x.strip().lower() for x in current.split(",") if x.strip()}
    extra = [a for a in dict.fromkeys(avoid) if a.lower() not in have]
    if not extra:
        return payload
    return {**payload, "negative_prompt": ", ".join(x for x in (current, *extra) if x)}


def plan(policy: Policy, cause: str, *, scene: str, identity_repairs: int, target: str = "photo"):
    """(adjust, action): a function that changes the next job's payload, and a description of it. Hints add only
    positive wording to the prompt; what to avoid goes to the job's `negative_prompt`."""
    cfg = policy.get("review", "repair", default={}) or {}
    hints = cfg.get("hints") or {}
    if cause == "identity":
        step, cap = float(cfg.get("lora_scale_step", 0.1)), float(cfg.get("lora_scale_max", 1.4))
        text, avoid = split_hint(hints.get("identity", ""))

        def adjust(p: dict) -> dict:
            out = with_negative({**p, "prompt": p["prompt"] + text}, avoid, policy, target)
            if "lora_scale" in p:
                out["lora_scale"] = round(min(cap, float(p["lora_scale"]) + step * (identity_repairs + 1)), 3)
            return out
        return adjust, f"stronger likeness (lora_scale +{step} per repair, max {cap}) and a face-visible prompt"
    if cause == "scene":
        text, avoid = split_hint(hints.get("scene", ""))
        lead = text.format(scene=scene)
        return (lambda p: with_negative({**p, "prompt": lead + p["prompt"]}, avoid, policy, target)), "prompt leads with the scene"
    if cause in ("realism", "hands", "teeth"):   # hands / teeth: the finer checks (review.detail), when enforced
        text, avoid = split_hint(hints.get(cause, hints.get("realism", "")))
        return (lambda p: with_negative({**p, "prompt": p["prompt"] + text}, avoid, policy, target)), "anatomy and text hints"
    return None, "new seed"
