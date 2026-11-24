"""Dance and motion transfer rules on the control plane (plan docs/superpowers/plans/2026-11-22-dance-motion-transfer.md).

Switched by policy `dance.enabled`. Two rules a dance job must pass before it is enqueued, and one check its output
must pass (the video gate, manyworlds.review.video, runs it on every motion-transfer clip):

  reference        the driving clip comes from `dance.references` (empty by default) and must be ours or licensed:
                   an owner in `dance.allowed_owners`, a licence string, a source, and a source that is not a
                   platform we would have downloaded a trend video from (`dance.blocked_sources`). Choreography can
                   be protected separately from its music, so a downloaded trend video is never a reference.
  strip audio      a reference's own sound never reaches the output (the graph in worker/comfy_animate.py takes only
                   its frames and frame rate).
  identity_check   per-keyframe identity scores (1-10, the video gate's identity row against the creator's
                   reference photos): the worst keyframe at `dance.identity.min_score` or more (like
                   review.video.min_identity) and the mean at `min_mean` or more, over `keyframes` keyframes
                   (gate `dance_identity`). Animate-2's card claims identity preservation but publishes no numbers,
                   so this gate decides.
"""

from __future__ import annotations

from urllib.parse import urlparse

from .policy import Policy


class RefusedReference(ValueError):
    """A driving clip that is not ours or licensed (or not in the catalogue)."""


def cfg(policy: Policy | None) -> dict:
    return (policy.get("dance", default={}) or {}) if policy is not None else {}


def enabled(policy: Policy | None) -> bool:
    return bool(cfg(policy).get("enabled"))


def _host(source: str) -> str:
    host = urlparse(source if "://" in source else f"https://{source}").hostname or ""
    return host.lower().removeprefix("www.")


def reference(ref_id: str, policy: Policy | None, sent: dict | None = None) -> dict:
    """The catalogue entry for a driving clip, or RefusedReference with the reason. `sent` is the entry the producer
    put in the job (its dance.references, newer than a worker image's baked policy): used when this policy doesn't
    list the id, and checked by the same rules (owner, licence, source), so a new clip needs no image rebuild."""
    c = cfg(policy)
    entries = {str(e.get("id")): dict(e) for e in (c.get("references") or []) if e and e.get("id")}
    entry = entries.get(str(ref_id))
    if entry is None and sent and str(sent.get("id")) == str(ref_id):
        entry = dict(sent)
    if entry is None:
        raise RefusedReference(f"reference {ref_id!r} is not in dance.references")
    if not str(entry.get("uri") or "").strip():
        raise RefusedReference(f"reference {ref_id!r} has no uri")
    owner = str(entry.get("owner") or "").strip().lower()
    allowed = {str(o).strip().lower() for o in (c.get("allowed_owners") or [])}
    if owner not in allowed:
        raise RefusedReference(f"reference {ref_id!r}: owner {owner or 'none'!r} is not one of {sorted(allowed)}")
    if not str(entry.get("licence") or "").strip():
        raise RefusedReference(f"reference {ref_id!r} has no licence")
    source = str(entry.get("source") or "").strip()
    if not source:
        raise RefusedReference(f"reference {ref_id!r} has no source")
    host = _host(source)
    for blocked in c.get("blocked_sources") or []:
        b = str(blocked).strip().lower().removeprefix("www.")
        if host == b or host.endswith("." + b):
            raise RefusedReference(f"reference {ref_id!r} comes from {host}: never a downloaded platform video")
    return entry


def identity_check(scores: list[float], policy: Policy | None) -> dict:
    """{passed, min, mean, failed_frames}: every keyframe's identity score at `dance.identity.min_score` or more,
    and their mean at `dance.identity.min_mean` or more. No keyframes fails."""
    ic = cfg(policy).get("identity") or {}
    floor, mean_floor = float(ic.get("min_score", 6)), float(ic.get("min_mean", 7))
    sims = [float(s) for s in scores or []]
    if not sims:
        return {"passed": False, "min": None, "mean": None, "failed_frames": [], "reason": "no keyframes"}
    failed = [i for i, s in enumerate(sims) if s < floor]
    mean = sum(sims) / len(sims)
    return {"passed": not failed and mean >= mean_floor, "min": min(sims), "mean": round(mean, 4),
            "failed_frames": failed}
