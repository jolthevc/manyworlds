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
import re


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


# --- designed voices (VoxCPM2, worker/handlers/tts_voxcpm.py, creator/voice_lab.py) -------------------------------
# VoxCPM2 reads an instruction in parentheses at the start of the text: a voice description when designing, a
# delivery ("slightly faster, cheerful tone") on a cloned voice. The instruction is never spoken, so it never
# belongs in the text a transcript is checked against.
_LEADING_INSTRUCTION = re.compile(r"^\s*(?:\([^()]*\)\s*)+")


def split_instruction(text: str) -> tuple[str, str]:
    """'(softly, amused) I knew it.' -> ('softly, amused', 'I knew it.'): leading parenthetical instructions and
    the words to be spoken."""
    text = text or ""
    m = _LEADING_INSTRUCTION.match(text)
    if not m:
        return "", text.strip()
    inner = [p.strip() for p in re.findall(r"\(([^()]*)\)", m.group(0)) if p.strip()]
    return ", ".join(inner), text[m.end():].strip()


def strip_delivery(text: str) -> str:
    """The words to be spoken: `text` without leading parenthetical instructions."""
    return split_instruction(text)[1]


def with_instruction(instruction: str | None, text: str) -> str:
    """What VoxCPM2 is given: '(instruction)text', or the bare words when there is no instruction. Parentheses
    inside the instruction would end it early, so they become commas."""
    words = strip_delivery(text)
    instruction = re.sub(r"\s*[()]\s*", ", ", (instruction or "").strip().strip("()")).strip(" ,")
    return f"({instruction}){words}" if instruction else words


# --- natural speech: clauses and the pauses between them (plan 2026-11-22-video-quality-autonomous, item 7) ---------
# The model's own pauses are not relied on (docs/research/2026-11-22-video-quality-wave2-3.md section 6): a line is
# spoken clause by clause and the silences between them are ours. Split points: a comma, semicolon or colon (so a
# clause that opens with "and", "but" or "so" after a comma starts its own piece), a dash, an ellipsis, and a
# sentence end. Never inside a number ("1,000", "3.5", "5:30", "10–20": no space after the mark) or after an
# abbreviation ("Dr.", "e.g.", "U.S.", "a.m.", an initial, or a full stop followed by a lower-case word).
_CLOSERS = "\"'”’)\\]"
_BOUNDARY = re.compile(
    rf"(?P<ellipsis>(?:\.{{3,}}|…)[{_CLOSERS}]*)(?=\s|$)"
    rf"|(?P<sentence>[.!?]+[{_CLOSERS}]*)(?=\s|$)"
    rf"|(?P<clause>[,;:][{_CLOSERS}]*)(?=\s|$)"
    r"|(?P<dash>\s+(?:--?|[–—])\s+|\s*(?:--|[–—])\s*)")
_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "approx", "inc", "ltd",
                            "dept", "mt", "ft", "e.g", "i.e", "a.m", "p.m"})
_WORD = re.compile(r"[A-Za-z0-9'’]+")


def _word_count(text: str) -> int:
    return len(_WORD.findall(text or ""))


def _after_abbreviation(text: str, m: re.Match) -> bool:
    """Is this full stop part of an abbreviation rather than a sentence end?"""
    if m.group(0).rstrip(_CLOSERS.replace("\\", "")) != ".":
        return False                           # "!", "?" or "?!" always end a sentence
    if text[m.end():].lstrip()[:1].islower():
        return True                            # "approx. five", "etc. and then"
    before = re.search(r"(\S+)$", text[:m.start()])
    token = (before.group(1) if before else "").lstrip("(\"'“‘[")
    return (token.lower() in _ABBREVIATIONS or bool(re.fullmatch(r"[A-Z]", token))
            or bool(re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", token)))


def _number_range(text: str, m: re.Match) -> bool:
    """An unspaced dash between digits ("10–20") is a range, not a break."""
    return (m.group(0) == m.group(0).strip() and m.start() > 0 and m.end() < len(text)
            and text[m.start() - 1].isdigit() and text[m.end()].isdigit())


def segments(text: str, *, min_words: int = 1) -> list[tuple[str, str]]:
    """A line's clauses with the boundary after each: [(clause, kind)], kind 'clause' (a short pause follows),
    'sentence' (a longer one) or 'end' (the line's own pause follows). Clause text keeps its punctuation, so a
    model still hears a comma as a continuation; a dash is kept as " —". A clause under `min_words` words joins
    the next one (or the one before, at a sentence end), since a one-word take is hard to voice and to score."""
    text = (text or "").strip()
    raw: list[list[str]] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        kind = m.lastgroup
        if kind == "sentence" and _after_abbreviation(text, m):
            continue
        if kind == "dash" and _number_range(text, m):
            continue
        if kind == "ellipsis":
            kind = "clause" if text[m.end():].lstrip()[:1].islower() else "sentence"
        piece = (text[start:m.start()].rstrip() + " —") if kind == "dash" else text[start:m.end()]
        start = m.end()
        kind = "clause" if kind == "dash" else kind
        if _word_count(piece):
            raw.append([piece.strip(), kind])
        elif raw and kind == "sentence":
            raw[-1][1] = "sentence"            # a stray mark ends the clause before it
    tail = text[start:].strip()
    if _word_count(tail):
        raw.append([tail, "end"])
    if not raw:
        return []
    raw[-1][1] = "end"
    merged: list[list[str]] = []
    carry = ""
    for piece, kind in raw:                    # a short clause joins the next one
        piece = f"{carry} {piece}" if carry else piece
        carry = ""
        if kind == "clause" and _word_count(piece) < min_words:
            carry = piece
            continue
        merged.append([piece, kind])
    out: list[list[str]] = []
    for piece, kind in merged:                 # a short clause at a sentence end joins the one before
        if out and out[-1][1] == "clause" and _word_count(piece) < min_words:
            out[-1] = [f"{out[-1][0]} {piece}", kind]
        else:
            out.append([piece, kind])
    return [(p, k) for p, k in out]


def clauses(text: str, *, min_words: int = 1) -> list[str]:
    """'Okay, so I went — and it rained.' -> ['Okay,', 'so I went —', 'and it rained.'] (see `segments`)."""
    return [p for p, _ in segments(text, min_words=min_words)]


def gap_s(rng, span: tuple[float, float]) -> float:
    """One pause length in seconds, drawn from [lo, hi] with the caller's seeded RNG: reproducible for a seed, but
    never the same length every time (a metronomic pause sounds read, not spoken)."""
    lo, hi = span
    return round(lo if hi <= lo else rng.uniform(lo, hi), 4)


def engine(anchors: dict | None) -> str:
    """The creator's voice engine: 'voxcpm' once a designed voice is stored (anchors.voice with a reference clip),
    otherwise 'kokoro' (a stock voice or blend), so creators made before designed voices keep speaking."""
    v = (anchors or {}).get("voice") or {}
    return "voxcpm" if v.get("engine") == "voxcpm" and v.get("ref_uri") and v.get("ref_text") else "kokoro"


def cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine similarity of two speaker embeddings (0 = same voice, 1 = unrelated, up to 2)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if not na or not nb:
        return 1.0
    return round(1.0 - dot / (na * nb), 4)


def nearest_delivery(delivery: str | None, bank: dict[str, list[str]]) -> str | None:
    """Which delivery-bank clip a line's delivery asks for (policy voice.delivery_bank): `bank` maps each delivery
    name to its keywords. The name whose keywords the delivery mentions most wins; on a tie, the one mentioned
    first ("flat, tired, then a small laugh" is deadpan before amused). Keywords match whole words or phrases,
    case-insensitively. None when no keyword is mentioned: the line keeps the base clip."""
    text = " ".join(re.findall(r"[a-z']+", (delivery or "").lower()))
    if not text:
        return None
    best, best_key = None, None
    for name, keywords in bank.items():
        hits = [m.start() for k in keywords or [] if str(k).strip()
                for m in [re.search(rf"\b{re.escape(' '.join(str(k).lower().split()))}\b", text)] if m]
        if not hits:
            continue
        key = (-len(hits), min(hits))
        if best_key is None or key < best_key:
            best, best_key = name, key
    return best
