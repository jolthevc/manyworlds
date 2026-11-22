"""Does the voice say the script? Speech is transcribed by a hosted Whisper and compared with what was meant.

Owner 2026-11-21: voices are judged automatically, never by ear. A talking clip whose voice line garbles, drops or
invents words fails the `intelligibility` gate before any paid video review. The word error rate (WER) is
the edit distance between the script's words and the transcript's, over the script's length, after both are
lower-cased and stripped of punctuation. The transcription is charged to the ledger (`hosted_api`).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from . import ledger
from .llm import USER_AGENT

HOSTS = {"together": {"url": "https://api.together.xyz/v1/audio/transcriptions", "key_env": "TOGETHER_API_KEY"}}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower().replace("’", "'"))


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return round(prev[-1] / len(ref), 3)


def _multipart(fields: dict, file_field: str, path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
             for k, v in fields.items()]
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{path.name}\"\r\n"
                 f"Content-Type: audio/wav\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _request(path: Path, cfg: dict, extra: dict, *, post=None):
    """POST the audio to the host's OpenAI-compatible transcription endpoint; the parsed JSON (or what `post`
    returns: a stand-in for tests, called as post(url, body, content_type, key))."""
    host = HOSTS[cfg.get("host", "together")]
    key = os.environ.get(host["key_env"])
    if not key:
        raise RuntimeError(f"{host['key_env']} is not set")
    body, ctype = _multipart({"model": cfg.get("model", "openai/whisper-large-v3"), "language": cfg.get("language", "en"),
                              **extra}, "file", path)
    if post:
        return post(host["url"], body, ctype, key)
    req = urllib.request.Request(host["url"], data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": ctype,
                                          "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=float(cfg.get("timeout_seconds", 120))) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"transcription -> {e.code}: {e.read().decode(errors='replace')[:300]}") from None


def transcribe(path: Path, cfg: dict, *, post=None) -> str:
    out = _request(path, cfg, {}, post=post)
    return out.get("text", "") if isinstance(out, dict) else str(out or "")


WORD_FIELDS = {"response_format": "verbose_json", "timestamp_granularities": "word"}


def parse_words(out) -> list[dict]:
    """A verbose_json response's `words` as [{word, start, end}] in seconds. Raises when it has none."""
    items = out.get("words") if isinstance(out, dict) else None
    words = [{"word": str(w.get("word", "")).strip(), "start": float(w["start"]), "end": float(w["end"])}
             for w in items or [] if isinstance(w, dict) and w.get("start") is not None and w.get("end") is not None]
    if not words:
        raise RuntimeError("the transcription has no word timestamps")
    return words


def transcribe_words(path: Path, cfg: dict, *, post=None) -> list[dict]:
    """Word timestamps: [{word, start, end}] in seconds. Together's transcription endpoint (OpenAI-compatible) gives
    them with response_format=verbose_json and timestamp_granularities=word (a single granularity is sent as a
    plain form field); the response's `words` hold {word, start, end}. Raises when the host returns none."""
    return parse_words(_request(path, cfg, WORD_FIELDS, post=post))


def record_cost(conn, policy, cfg: dict, seconds: float, *, job_id=None, creator_id=None) -> None:
    """A transcription's cost (`hosted_api`, per audio minute)."""
    ledger.record(conn, category="hosted_api", provider=f"{cfg.get('host', 'together')}:{cfg.get('model')}",
                  quantity=round(seconds / 60, 3), unit="audio_minute",
                  usd=round(seconds / 60 * float(cfg.get("usd_per_minute", 0.0015)), 6), job_id=job_id,
                  creator_id=creator_id, policy_version=policy.version)


def timed_words(conn, policy, path: Path, seconds: float, *, job_id=None, creator_id=None, post=None) -> list[dict]:
    """Transcribe a voice line with word timestamps (review.video.intelligibility's host and model) and charge it
    (also when the answer has no words: the host billed the call)."""
    cfg = policy.get("review", "video", "intelligibility", default={}) or {}
    out = _request(path, cfg, WORD_FIELDS, post=post)
    record_cost(conn, policy, cfg, seconds, job_id=job_id, creator_id=creator_id)
    return parse_words(out)


def check(conn, policy, path: Path, script: str, seconds: float, *, job_id=None, creator_id=None, post=None) -> dict:
    """Transcribe a voice line and score it against its script: {wer, transcript, passed}."""
    cfg = policy.get("review", "video", "intelligibility", default={}) or {}
    text = transcribe(path, cfg, post=post)
    record_cost(conn, policy, cfg, seconds, job_id=job_id, creator_id=creator_id)
    return score(policy, script, text)


def score(policy, script: str, text: str) -> dict:
    """A transcript scored against its script: {wer, transcript, passed} (review.video.intelligibility.max_wer)."""
    cfg = policy.get("review", "video", "intelligibility", default={}) or {}
    w = wer(script, text)
    return {"wer": w, "transcript": text[:2000], "passed": w <= float(cfg.get("max_wer", 0.15))}
