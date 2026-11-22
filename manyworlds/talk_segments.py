"""Long talking videos in segments: a script of any length is split at sentence boundaries into pieces short
enough to render reliably (policy generation.talk_long.max_segment_seconds), each rendered as its own talk job.

Length is estimated from words at the creator's speaking rate (`words_per_second`); a sentence longer than one
segment is split at commas, then between words, so no segment is ever over the limit.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:—])\s+")


def _words(text: str) -> int:
    return len(text.split())


def sentences(text: str) -> list[str]:
    """The text's sentences, in order (whitespace collapsed)."""
    return [s for s in _SENTENCE_END.split(" ".join((text or "").split())) if s]


def estimate_seconds(text: str, words_per_second: float) -> float:
    return _words(text) / float(words_per_second)


def split_script(script: str, *, max_seconds: float, words_per_second: float) -> list[str]:
    """The script as segments of at most `max_seconds` of speech, cut between sentences where possible."""
    text = " ".join((script or "").split())
    if not text:
        return []
    max_words = max(1, int(max_seconds * words_per_second))
    if _words(text) <= max_words:
        return [text]
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        if _words(sentence) <= max_words:
            pieces.append(sentence)
            continue
        for clause in _CLAUSE_END.split(sentence):
            words = clause.split()
            while len(words) > max_words:
                pieces.append(" ".join(words[:max_words]))
                words = words[max_words:]
            if words:
                pieces.append(" ".join(words))
    segments, current = [], ""
    for piece in pieces:
        joined = f"{current} {piece}".strip()
        if current and _words(joined) > max_words:
            segments.append(current)
            current = piece
        else:
            current = joined
    if current:
        segments.append(current)
    return segments
