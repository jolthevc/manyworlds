"""Targeted repair: why a scene's images were rejected, and what the next attempt changes because of it.

Diagnosis reads the review rows of the rejected job (manyworlds.review.gates writes one per gate per image):
the gate that failed most is the cause, except that any safety failure makes the cause `safety`. Each cause
has its own fix (policy `review.repair`):
  identity  the likeness LoRA gets stronger by `lora_scale_step` per identity repair, up to `lora_scale_max`,
            and the prompt asks for a clearly visible face with the signature details
  scene     the prompt leads with the scene, so the setting and action can't be lost
  realism   the prompt steers away from what renders badly (hands, text)
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
            failed[r["gate"]] += 1
    if not failed:
        return {"cause": None, "failed": {}, "defects": []}
    cause = "safety" if failed.get("safety") else max(
        failed, key=lambda g: (failed[g], -ORDER.index(g) if g in ORDER else -len(ORDER)))
    defects = Counter(d for r in rows if not r["passed"] for d in ((r["details"] or {}).get("defects") or []))
    return {"cause": cause, "failed": dict(failed), "defects": [d for d, _ in defects.most_common(3)]}


def plan(policy: Policy, cause: str, *, scene: str, identity_repairs: int):
    """(adjust, action): a function that changes the next job's payload, and a description of it."""
    cfg = policy.get("review", "repair", default={}) or {}
    hints = cfg.get("hints") or {}
    if cause == "identity":
        step, cap = float(cfg.get("lora_scale_step", 0.1)), float(cfg.get("lora_scale_max", 1.4))

        def adjust(p: dict) -> dict:
            out = {**p, "prompt": p["prompt"] + hints.get("identity", "")}
            if "lora_scale" in p:
                out["lora_scale"] = round(min(cap, float(p["lora_scale"]) + step * (identity_repairs + 1)), 3)
            return out
        return adjust, f"stronger likeness (lora_scale +{step} per repair, max {cap}) and a face-visible prompt"
    if cause == "scene":
        lead = hints.get("scene", "").format(scene=scene)
        return (lambda p: {**p, "prompt": lead + p["prompt"]}), "prompt leads with the scene"
    if cause in ("realism", "hands", "teeth"):   # hands / teeth: the finer checks (review.detail), when enforced
        hint = hints.get(cause, hints.get("realism", ""))
        return (lambda p: {**p, "prompt": p["prompt"] + hint}), "anatomy and text hints"
    return None, "new seed"
