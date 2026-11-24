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


# Normalisation (2026-11-23 breadth run: "six p.m. Coffee number two" heard as "6 p.m. Coffee No. 2" scored WER 0.5
# and the clip was dropped). Script and transcript are both put in one spelling before they are compared: numbers
# as digits (number words, ordinals and plurals of number words included), "No."/"#" before a number as "number",
# times as "<n> pm", contractions expanded. These are spelling rules of English, not tunables.
_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000}
_ORDINALS = {"first": "one", "second": "two", "third": "three", "fourth": "four", "fifth": "five", "sixth": "six",
             "seventh": "seven", "eighth": "eight", "ninth": "nine", "tenth": "ten", "eleventh": "eleven",
             "twelfth": "twelve", "thirteenth": "thirteen", "fourteenth": "fourteen", "fifteenth": "fifteen",
             "sixteenth": "sixteen", "seventeenth": "seventeen", "eighteenth": "eighteen", "nineteenth": "nineteen",
             "twentieth": "twenty", "thirtieth": "thirty", "fortieth": "forty", "fiftieth": "fifty",
             "sixtieth": "sixty", "seventieth": "seventy", "eightieth": "eighty", "ninetieth": "ninety",
             "hundredth": "hundred", "thousandth": "thousand"}
_PLURALS = {"ones": "one", "twos": "two", "threes": "three", "fours": "four", "fives": "five", "sixes": "six",
            "sevens": "seven", "eights": "eight", "nines": "nine", "tens": "ten"}
_CONTRACTIONS = {"won't": "will not", "can't": "can not", "cannot": "can not", "shan't": "shall not",
                 "ain't": "is not", "let's": "let us", "y'all": "you all", "gonna": "going to", "wanna": "want to",
                 "gotta": "got to", "okay": "ok", "o'clock": ""}
_IS_S = {"it's", "that's", "what's", "he's", "she's", "there's", "here's", "who's", "where's", "how's", "when's",
         "why's"}
_SUFFIXES = (("n't", " not"), ("'m", " am"), ("'re", " are"), ("'ve", " have"), ("'ll", " will"), ("'d", " would"))
_NUMBER_WORD = "|".join(sorted({*_UNITS, *_TENS, *_ORDINALS}, key=len, reverse=True))


def _suffix(n: int) -> str:
    return "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _number_base(t: str) -> tuple[str, bool]:
    """A token as its cardinal number word (or itself) and whether it was an ordinal."""
    if t in _ORDINALS:
        return _ORDINALS[t], True
    return _PLURALS.get(t, t), False


def _spell_numbers(tokens: list[str]) -> list[str]:
    """Runs of number words as one number in digits ("twenty one" -> 21, "two hundred and five" -> 205, "third" ->
    3rd, "ones" -> 1); a number word that cannot continue the run starts a new one ("six thirty" -> 6 30)."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        total, cur, used, ordinal, last, j = 0, 0, 0, False, None, i
        while j < len(tokens):
            t = tokens[j]
            if t == "and" and last == "scale" and j + 1 < len(tokens) and \
                    _number_base(tokens[j + 1])[0] in {**_UNITS, **_TENS}:
                j += 1
                continue
            base, is_ord = _number_base(t)
            if base in _UNITS and last != "unit" and not (last == "tens" and (_UNITS[base] >= 10 or _UNITS[base] == 0)):
                cur, last = cur + _UNITS[base], "unit"
            elif base in _TENS and last in (None, "scale"):
                cur, last = cur + _TENS[base], "tens"
            elif base in _SCALES and used:
                if _SCALES[base] == 100:
                    cur = (cur or 1) * 100
                else:
                    total, cur = total + (cur or 1) * _SCALES[base], 0
                last = "scale"
            else:
                break
            used += 1
            j += 1
            if is_ord:
                ordinal = True
                break
        if not used:
            out.append(tokens[i])
            i += 1
            continue
        n = total + cur
        out.append(f"{n}{_suffix(n)}" if ordinal else str(n))
        i = j
    return out


def _expand(t: str) -> list[str]:
    """One token with its contraction spelled out ("i'm" -> i am, "don't" -> do not, "it's" -> it is)."""
    if t in _CONTRACTIONS:
        return _CONTRACTIONS[t].split()
    if t in _IS_S:
        return [t[:-2], "is"]
    for end, full in _SUFFIXES:
        if t.endswith(end) and len(t) > len(end):
            return (t[:-len(end)] + full).split()
    t = t.strip("'")
    if t.endswith("'s"):
        t = t[:-2] + "s"            # possessive: Maren's -> marens on both sides
    return [t] if t else []


def normalise(text: str) -> str:
    """The text in the one spelling WER compares: lower case, no punctuation, numbers in digits, contractions
    expanded (see the rules above)."""
    t = (text or "").lower().replace("’", "'").replace("‘", "'")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)                                       # 1,000 -> 1000
    t = re.sub(rf"(?:\bno\.|\bnum\.|\bnr\.?|#)\s*(?=\d|(?:{_NUMBER_WORD})\b)", " number ", t)   # No. 2, #2
    t = re.sub(r"\bno\s+(?=\d)", "number ", t)                                       # no 2 (before a digit only)
    t = re.sub(r"\b([ap])\.\s?m\b\.?", r"\1m", t)                                    # p.m. -> pm
    t = re.sub(r"\b([ap])\s+m\b", r"\1m", t)                                         # p m -> pm
    t = re.sub(r"(\d)\s*(am|pm)\b", r"\1 \2", t)                                     # 6pm -> 6 pm
    t = re.sub(r"\b(\d{1,2}):00\b", r"\1", t)                                        # 6:00 -> 6
    t = t.replace("&", " and ").replace("%", " percent ")
    tokens = [w for raw in re.findall(r"[a-z0-9']+", t) for w in _expand(raw)]
    return " ".join(_spell_numbers(tokens))


def words(text: str) -> list[str]:
    return normalise(text).split()


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
    """A transcript scored against its script: {wer, transcript, passed, max_wer} (review.video.intelligibility.max_wer;
    a line shorter than short_line_words is judged by short_line_max_wer, see max_wer_for)."""
    cfg = policy.get("review", "video", "intelligibility", default={}) or {}
    w, limit = wer(script, text), max_wer_for(cfg, len(words(script)))
    return {"wer": w, "transcript": text[:2000], "passed": w <= limit, "max_wer": limit}


def max_wer_for(cfg: dict, n_words: int) -> float:
    """The WER a line of n_words may have. On a very short line one misheard word is a third or half of it ("Look
    one" heard "Look ones" is 0.5), so a line under `short_line_words` words gets `short_line_max_wer` instead of
    `max_wer` (2026-11-23 breadth run)."""
    limit = float(cfg.get("max_wer", 0.15))
    short = int(cfg.get("short_line_words") or 0)
    if short and 0 < n_words < short and cfg.get("short_line_max_wer") is not None:
        return max(limit, float(cfg["short_line_max_wer"]))
    return limit
