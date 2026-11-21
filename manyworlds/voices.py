"""A voice of their own for every creator: a fixed blend of Kokoro's stock voices, distinct from every sibling.

Kokoro has a few dozen stock voices, so creators picking from the list end up sharing them. A voice pack is a
tensor, and weighted sums of packs are new voices that keep Kokoro's quality. Each creator gets one blend at
creation, written as a spec ("af_bella:0.55+af_nova:0.3+af_sky:0.15"): the base voice the creator lab chose
leads it, the others come from the same accent and gender group, and the weights are drawn from the creator's
handle, so the blend is reproducible. A blend closer than `min_distance` (L1 over the weights) to another
creator's is redrawn. A stopgap until a voice-design model is benchmarked (docs/research/2026-11-21-model-landscape.md).

`design` and `parse` are pure; `voice_arg` (torch) runs on the worker that speaks.
"""

from __future__ import annotations

import hashlib
import random


def parse(spec: str) -> list[tuple[str, float]]:
    """'af_bella:0.6+af_heart:0.4' -> [('af_bella', 0.6), ('af_heart', 0.4)]; a plain name is weight 1."""
    out = []
    for part in (spec or "").split("+"):
        name, _, w = part.strip().partition(":")
        if name:
            out.append((name.strip(), float(w) if w else 1.0))
    total = sum(w for _, w in out) or 1.0
    return [(n, w / total) for n, w in out]


def distance(a: str, b: str) -> float:
    wa, wb = dict(parse(a)), dict(parse(b))
    return sum(abs(wa.get(k, 0) - wb.get(k, 0)) for k in set(wa) | set(wb))


def design(base: str, catalog: list[str], taken: list[str], *, seed: str, mix: int = 3, lead: float = 0.5,
           min_distance: float = 0.5, tries: int = 200) -> str:
    """A blend led by `base`, mixed with voices of its group (same two-letter prefix: accent and gender), at least
    `min_distance` from every blend in `taken`. Falls back to the farthest one found."""
    group = [v for v in catalog if v[:2] == base[:2] and v != base]
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:12], 16))
    best, best_d = base, -1.0
    for _ in range(tries):
        others = rng.sample(group, min(len(group), max(0, mix - 1)))
        head = rng.uniform(lead, 0.8) if others else 1.0
        rest = [rng.random() + 0.1 for _ in others]
        weights = [head] + [(1 - head) * r / sum(rest) for r in rest] if others else [1.0]
        spec = "+".join(f"{n}:{w:.2f}" for n, w in zip([base, *others], weights))
        d = min((distance(spec, t) for t in taken), default=2.0)
        if d >= min_distance:
            return spec
        if d > best_d:
            best, best_d = spec, d
    return best


def voice_arg(pipe, spec: str):
    """What Kokoro's pipeline takes as `voice`: the name for a single stock voice, else the blended pack."""
    parts = parse(spec)
    if len(parts) == 1:
        return parts[0][0]
    return sum(pipe.load_single_voice(name) * w for name, w in parts)
