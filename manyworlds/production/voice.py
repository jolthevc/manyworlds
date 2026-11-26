"""The designed voice: each spoken piece's voice.speak.<engine> job (a Step of its own) and the lines it says."""

from __future__ import annotations

from typing import Any

from .. import queue
from ..policy import Policy
from .common import _cfg, _seed, piece_key, unmarked
from .steps import Step, _status, advance_step


# The designed voice (creator/voice_lab.py, worker/handlers/tts_engines.py): for a creator whose voice engine is
# designed (qwen3, indextts), each spoken piece is a job of that engine's kind first (voice.speak.<engine>,
# voice_lab.speak_kind), a Step of its own (retried like any other on failure; its
# takes are chosen by WER on the worker), and the piece's job takes its wav. A rejected talk clip is remade over the
# same voice line, so a line whose worker transcript already shows the intelligibility gate would reject it is voiced
# again first (`gate_misses`, policy voice.gate_repairs).
def _designed_voice(conn, post: dict) -> dict | None:
    """The creator's designed voice (anchors.voice), or None while they speak with Kokoro."""
    from .. import voices
    anchors = conn.execute("SELECT anchors FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["anchors"]
    return (anchors or {}).get("voice") if voices.designed(anchors) else None


def _sentence_lines(text: str) -> list[dict]:
    from ..talk_segments import sentences
    return [{"text": s} for s in sentences(text)]


def voice_lines(body: dict, segment: int | None = None) -> list[dict]:
    """What one take of a talking video says, line by line, for voice.speak.*: [{text, delivery, pause_after_s}].
    A directed take (body.talk_shots) speaks its shots' lines with their deliveries and pauses, cut to this take's
    words (a take split for length gets only its part; a line cut in two keeps its pause on the second part).
    Otherwise, or when the take's words don't follow the shots' lines, one line per sentence, no delivery.
    *Emphasis* asterisks are never spoken: they are taken out here (the finish stage reads them from the script)."""
    out = []
    for line in _marked_voice_lines(body, segment):
        text = " ".join(unmarked(line["text"]).split())
        if text:
            out.append({**line, "text": text})
    return out


def _marked_voice_lines(body: dict, segment: int | None = None) -> list[dict]:
    from itertools import groupby

    takes = body.get("talk_segments") or [body.get("talk_script") or body.get("script", "")]
    k = segment or 0
    text = takes[k] if k < len(takes) else ""
    if body.get("talk_shots") and body.get("shots"):
        from ..direction import spoken_lines
        lines = [line for shot in body["shots"] for line in spoken_lines(shot)]
        stream = [(w, li, wi == len(line["text"].split()) - 1)
                  for li, line in enumerate(lines) for wi, w in enumerate(line["text"].split())]
        start, words = sum(len(t.split()) for t in takes[:k]), text.split()
        part = stream[start:start + len(words)]
        if words and [w for w, _, _ in part] == words:
            out = []
            for li, group in groupby(part, key=lambda x: x[1]):
                group = list(group)
                out.append({"text": " ".join(w for w, _, _ in group), "delivery": lines[li]["delivery"] or None,
                            "pause_after_s": lines[li]["pause_after_s"] if group[-1][2] else None,
                            # the shot's performance levels (voice_lab.speak_payload: emotion strength, pauses)
                            **({"controls": lines[li]["controls"]} if lines[li].get("controls") else {})})
            return out
    return _sentence_lines(text)


def gate_misses(policy: Policy, meta: dict | None) -> dict | None:
    """Would the intelligibility gate reject this voice line? The voice worker already transcribed its best take
    with the gate's own model (meta.wer, over the words it was asked to say), so this costs nothing. The gate scores
    the same wav (review.video._intelligibility reads the clip's audio_uri) against review.video.intelligibility
    max_wer (short_line_max_wer for a short line), which is tighter than the worker's own line ceiling
    (voice.line_max_wer, failing only above it): a line that lost its first words ("Two am. I take it back..." heard
    as "and take it back...", WER 0.25, breadth rerun 2026-11-24) passed the worker and then failed every talk take
    made over it. None when it passes, the gate is not enforced, or the worker gave no score."""
    from .. import speech
    gate = policy.get("review", "video", "intelligibility", default={}) or {}
    meta = meta or {}
    if not gate.get("enforce") or meta.get("wer") is None:
        return None
    lines = [line for line in meta.get("lines") or [] if isinstance(line, dict)]
    said = " ".join(str(line.get("text", "")) for line in lines)
    limit = speech.max_wer_for(gate, len(speech.words(said))) if said else float(gate.get("max_wer", 0.15))
    if float(meta["wer"]) <= limit:
        return None
    heard = " ".join(str(line.get("transcript", "")) for line in lines)
    return {"wer": float(meta["wer"]), "max_wer": limit, "heard": heard[:300]}


def _voice_state(conn, policy: Policy, job_ids: list[str]) -> tuple[str, Any]:
    """Done with an audio asset is chosen, unless the worker's own transcript already shows the intelligibility gate
    would reject it (`gate_misses`): then it is rejected and voiced again with new seeds, before any talk clip is
    made over it."""
    status = _status(conn, job_ids[-1])
    if status in ("queued", "running"):
        return "waiting", None
    audio = conn.execute("SELECT id, uri, meta FROM assets WHERE job_id = %s AND kind = 'audio' ORDER BY uri LIMIT 1",
                         (job_ids[-1],)).fetchone()
    if status != "done" or not audio:
        return "failed", None
    return ("rejected", None) if gate_misses(policy, audio["meta"]) else ("chosen", audio)


def _voice_job(conn, policy: Policy, post: dict, voice: dict, lines: list[dict], attempt: int, *,
               segment: int | None = None):
    from ..creator import voice_lab
    payload = voice_lab.speak_payload(policy, voice, lines, seed=_seed(post, 800 + (segment or 0) * 1000, attempt))
    base = piece_key(post)   # shared across posts with one body.shared_key: one voice line for all of them
    key = f"{base}:voice:{attempt}" if segment is None else f"{base}:voice:{segment}:{attempt}"
    kind = voice_lab.speak_kind(policy, voice.get("engine"))
    return queue.enqueue(conn, kind, payload, creator_id=post["creator_id"],
                         batch=f"post-{post['id']}", post_id=post["id"], idempotency_key=key)


def _advance_voice(conn, policy: Policy, post: dict, body: dict, voice: dict | None, jobs: list[str],
                   lines: list[dict], *, segment: int | None = None, label: str = "") -> tuple[str, Any]:
    """('chosen', audio asset) once the piece's voice line exists; ('chosen', None) when there is nothing to voice
    here (a Kokoro creator, or no words), so the piece is made as before."""
    if voice is None or not lines:
        return "chosen", None
    step = Step(jobs, lambda j: _voice_state(conn, policy, j),
                lambda attempt: _voice_job(conn, policy, post, voice, lines, attempt, segment=segment),
                ("voice_repairs", int(policy.get("voice", "gate_repairs", default=2))),("follow_up_failures", int(_cfg(policy).get("max_job_failures_per_post", 6))),
                "voice", label)
    return advance_step(body, step)


def _with_audio(payload: dict, audio: dict | None, *, captions: bool = False) -> dict:
    """The piece speaks the finished voice line (its wav) instead of TTS on its own worker; an edit also gets
    each line's timing, so its captions appear exactly while they are said."""
    if audio:
        payload["audio_uri"] = audio["uri"]
        if captions:
            payload["cues"] = [{k: line.get(k) for k in ("text", "start_s", "end_s")}
                               for line in (audio.get("meta") or {}).get("lines") or []]
    return payload
