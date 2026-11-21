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


def transcribe(path: Path, cfg: dict, *, post=None) -> str:
    host = HOSTS[cfg.get("host", "together")]
    key = os.environ.get(host["key_env"])
    if not key:
        raise RuntimeError(f"{host['key_env']} is not set")
    body, ctype = _multipart({"model": cfg.get("model", "openai/whisper-large-v3"), "language": cfg.get("language", "en")},
                             "file", path)
    if post:
        return post(host["url"], body, ctype, key)
    req = urllib.request.Request(host["url"], data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": ctype,
                                          "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=float(cfg.get("timeout_seconds", 120))) as r:
            return json.loads(r.read()).get("text", "")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"transcription -> {e.code}: {e.read().decode(errors='replace')[:300]}") from None


def check(conn, policy, path: Path, script: str, seconds: float, *, job_id=None, creator_id=None, post=None) -> dict:
    """Transcribe a voice line and score it against its script: {wer, transcript, passed}."""
    cfg = policy.get("review", "video", "intelligibility", default={}) or {}
    text = transcribe(path, cfg, post=post)
    ledger.record(conn, category="hosted_api", provider=f"{cfg.get('host', 'together')}:{cfg.get('model')}",
                  quantity=round(seconds / 60, 3), unit="audio_minute",
                  usd=round(seconds / 60 * float(cfg.get("usd_per_minute", 0.0015)), 6), job_id=job_id,
                  creator_id=creator_id, policy_version=policy.version)
    w = wer(script, text)
    return {"wer": w, "transcript": text[:2000], "passed": w <= float(cfg.get("max_wer", 0.15))}
