"""Cheap checks of a new creator against the real creators her brief names (owner 2026-11-25).

Emulating successful creators is the point, and a brief may name one creator as its reference. What we never do is
impersonate a real person, make a confusingly similar name, handle or face, or copy a specific post. Two checks,
all code, no model call:

    names(candidate, references, policy)        name and handle string similarity
    near_duplicates(image, cited_images, policy) a 64-bit difference hash against the posts the brief cited

No face comparison (owner 2026-11-25, data_policy red): it would need face-identity templates of real people. A
look-alike is prevented upstream instead: no real person's media ever reaches a generation job (research.references
guard, assets_not_research), and faces come from the creator's own designed bible and LoRA.

Flags are warnings for the talent manager. An obvious impersonation (a name or handle past `name_block`, policy
`intelligence.checks`) blocks.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from ..policy import Policy


def _cfg(policy: Policy | None) -> dict:
    return (policy.get("intelligence", "checks", default={}) or {}) if policy is not None else {}


def normalise(s: str | None) -> str:
    """Lowercase letters and digits only, accents folded, a leading @ and separators gone ('Mia.Lou_' -> 'mialou')."""
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def similarity(a: str | None, b: str | None) -> float:
    a, b = normalise(a), normalise(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ratio = SequenceMatcher(None, a, b).ratio()
    contained = (a in b or b in a) and min(len(a), len(b)) >= 4
    return round(max(ratio, 0.95 if contained else 0.0), 3)


def _verdict(score: float, warn: float, block: float) -> str | None:
    return "block" if score >= block else "warn" if score >= warn else None


def names(candidate: dict, references: list[dict], policy: Policy | None) -> list[dict]:
    """candidate {name, handle}; references [{id, display_name, handle}]. Every pair at or past the warning level."""
    c = _cfg(policy)
    warn, block = float(c.get("name_warn", 0.75)), float(c.get("name_block", 0.92))
    flags = []
    for r in references:
        for mine, theirs in (("name", "display_name"), ("handle", "handle"), ("name", "handle"), ("handle", "display_name")):
            score = similarity(candidate.get(mine), r.get(theirs))
            v = _verdict(score, warn, block)
            if v:
                flags.append({"check": "name", "verdict": v, "score": score, "reference": str(r.get("id")),
                              "fields": [mine, theirs]})
    return flags


def dhash(path: str | Path, size: int = 8) -> int:
    """A difference hash: the image greyed and shrunk to (size+1) x size, one bit per left/right brightness step."""
    from PIL import Image
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        px = list(g.tobytes())
    bits = 0
    for row in range(size):
        for col in range(size):
            left, right = px[row * (size + 1) + col], px[row * (size + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def near_duplicates(image: str | Path, cited: dict[str, str | Path], policy: Policy | None) -> list[dict]:
    """A generated image against the cited posts' research images: within `near_duplicate_bits` is a warning."""
    limit = int(_cfg(policy).get("near_duplicate_bits", 6))
    mine = dhash(image)
    flags = []
    for ref, path in cited.items():
        bits = bin(mine ^ dhash(path)).count("1")
        if bits <= limit:
            flags.append({"check": "near_duplicate", "verdict": "warn", "bits": bits, "reference": str(ref)})
    return flags


def check(candidate: dict, references: list[dict], policy: Policy | None) -> dict:
    """The casting check: {blocked, warnings, blocks, ran}. Blocked means an obvious impersonation."""
    flags = names(candidate, references, policy)
    blocks = [f for f in flags if f["verdict"] == "block"]
    return {"blocked": bool(blocks), "blocks": blocks, "warnings": [f for f in flags if f["verdict"] == "warn"],
            "ran": ["names"]}
