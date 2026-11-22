"""A designed voice for every creator (VoxCPM2, worker/handlers/tts_voxcpm.py; policy `voice`).

    design(conn, policy, creator_id) -> job_id     candidates from the bible's voice notes and appearance
    pick(conn, policy, creator_id, job) -> dict    one candidate, chosen automatically, becomes anchors.voice

The choice is never made by ear (owner 2026-11-21): the candidate must say the anchor passage (word error rate within
`voice.max_wer`), must sound unlike every other creator (speaker-embedding cosine distance at least
`voice.min_distance` from each of their anchors), and among those the lowest WER wins, then the liveliest (highest
pitch variance: least monotone). The pick is a decision row (kind `voice_pick`).

A creator without a designed voice keeps speaking with Kokoro (`voices.engine`), and a designed voice keeps the
creator's Kokoro voice in anchors.voice.voice, so nothing that still reads it changes until it reads the new one.
"""

from __future__ import annotations

import hashlib
import re

from psycopg.types.json import Jsonb

from .. import queue, voices
from ..policy import Policy


def _cfg(policy: Policy) -> dict:
    return policy.get("voice", default={}) or {}


def _kind(policy: Policy, what: str) -> str:
    return (_cfg(policy).get("kinds") or {}).get(what, f"voice.{what}")


def worker_params(policy: Policy) -> dict:
    """What the voice worker needs from policy: the model, its sampling, and the speech gate's transcriber."""
    cfg, heard = _cfg(policy), policy.get("review", "video", "intelligibility", default={}) or {}
    params = {k: cfg[k] for k in ("model", "cfg_value", "inference_timesteps", "optimize", "embedding", "pitch_hz",
                                  "pause_s") if k in cfg}
    params.update(asr_model=heard.get("model", "openai/whisper-large-v3"), language=heard.get("language", "en"))
    return params


def describe(bible: dict) -> str:
    """The voice description VoxCPM2 designs from: who is speaking (apparent age, gender presentation) and how they
    sound (the bible's voice notes). No parentheses: they delimit the description in the model's text."""
    a = (bible or {}).get("appearance") or {}
    who = " ".join(str(x).strip() for x in (f"{a['apparent_age']}-year-old" if a.get("apparent_age") else "",
                                            a.get("gender_presentation") or "") if str(x).strip())
    notes = (bible or {}).get("voice_notes") or ""
    parts = [f"A {who} speaking to a phone camera" if who else "", notes.strip()]
    text = re.sub(r"\s*[()]\s*", ", ", ". ".join(p.rstrip(". ") for p in parts if p))
    return re.sub(r"\s*,(\s*,)+", ",", text).strip(" ,")


def design(conn, policy: Policy, creator_id, *, batch: str | None = None, pools: list[str] | None = None):
    """Enqueue `voice.candidates` voice candidates for the creator, reading `voice.anchor_text`. The same
    description and passage give the same job (idempotent); a changed bible designs again."""
    cfg = _cfg(policy)
    row = conn.execute("SELECT handle, bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
    if not row:
        raise ValueError(f"no creator {creator_id}")
    description, text = describe(row["bible"] or {}), (cfg.get("anchor_text") or "").strip()
    if not description or not text:
        raise ValueError("a voice needs the bible's voice notes or appearance, and policy voice.anchor_text")
    digest = hashlib.sha256(f"{creator_id}|{description}|{text}".encode()).hexdigest()
    payload = {"description": description, "text": text, "n": int(cfg.get("candidates", 4)),
               "seed": int(digest[:8], 16) % (1 << 31), "params": worker_params(policy)}
    return queue.enqueue(conn, _kind(policy, "design"), payload, creator_id=creator_id, batch=batch, pools=pools,
                         idempotency_key=f"{creator_id}:voice:design:{digest[:12]}",
                         max_attempts=int(policy.get("queue", "max_attempts", default=3)))


def rank(candidates: list[dict], others: list[list[float]], *, max_wer: float, min_distance: float) -> list[dict]:
    """The candidates that qualify, best first. Each is {wer, embedding, pitch_var, ...}; `others` are the
    embeddings of every other creator's voice. Adds `distance` (to the nearest other voice) to each."""
    ok = []
    for c in candidates:
        d = min((voices.cosine_distance(c.get("embedding") or [], o) for o in others if o), default=2.0)
        c = {**c, "distance": d}
        if c.get("wer") is not None and float(c["wer"]) <= max_wer and d >= min_distance:
            ok.append(c)
    return sorted(ok, key=lambda c: (float(c["wer"]), -float(c.get("pitch_var") or 0.0), int(c.get("index") or 0)))


def pick(conn, policy: Policy, creator_id, job) -> dict:
    """Choose the creator's voice from a finished design job and store it as anchors.voice:
    {engine: voxcpm, ref_uri, ref_text, description, embedding, voice (their Kokoro voice, kept)}."""
    cfg = _cfg(policy)
    job_id = job.get("id") if isinstance(job, dict) else getattr(job, "id", job)
    j = conn.execute("SELECT id, kind, status, payload FROM jobs WHERE id = %s", (job_id,)).fetchone()
    if not j:
        raise ValueError(f"no job {job_id}")
    if j["status"] != "done":
        return {"picked": None, "reason": f"design job is {j['status']}"}
    assets = conn.execute("SELECT id, uri, meta FROM assets WHERE job_id = %s AND kind = 'audio' ORDER BY uri",
                          (j["id"],)).fetchall()
    candidates = [{**(a["meta"] or {}), "asset_id": str(a["id"]), "uri": a["uri"]} for a in assets]
    others = [r["e"] for r in conn.execute(
        "SELECT anchors->'voice'->'embedding' AS e FROM creators WHERE id <> %s "
        "AND anchors->'voice'->>'engine' = 'voxcpm'", (creator_id,)).fetchall() if r["e"]]
    max_wer, min_distance = float(cfg.get("max_wer", 0.1)), float(cfg.get("min_distance", 0.35))
    ranked = rank(candidates, others, max_wer=max_wer, min_distance=min_distance)
    summary = [{k: c.get(k) for k in ("index", "uri", "wer", "pitch_var", "seconds", "transcript")} for c in candidates]
    inputs = {"job_id": str(j["id"]), "candidates": summary, "other_voices": len(others),
              "max_wer": max_wer, "min_distance": min_distance}
    if not ranked:
        out = {"picked": None, "reason": "no candidate said the passage within max_wer and sounded unlike the others"}
        _decide(conn, policy, creator_id, j["id"], inputs, out, out["reason"])
        return out
    best = ranked[0]
    payload = j["payload"] or {}
    row = conn.execute("SELECT anchors FROM creators WHERE id = %s", (creator_id,)).fetchone()
    old = ((row["anchors"] if row else None) or {}).get("voice") or {}
    voice = {"engine": "voxcpm", "ref_uri": best["uri"], "ref_text": voices.strip_delivery(payload.get("text", "")),
             "description": payload.get("description", ""), "embedding": best.get("embedding") or [],
             "asset_id": best["asset_id"], "design_job_id": str(j["id"]),
             **{k: old[k] for k in ("voice", "speed") if k in old}}   # Kokoro stays the fallback
    conn.execute("UPDATE creators SET anchors = coalesce(anchors, '{}'::jsonb) || jsonb_build_object('voice', %s::jsonb) "
                 "WHERE id = %s", (Jsonb(voice), creator_id))
    out = {"picked": {k: best.get(k) for k in ("index", "uri", "wer", "pitch_var", "distance", "seconds")},
           "ranked": [c["index"] for c in ranked]}
    _decide(conn, policy, creator_id, j["id"], inputs, out,
            f"lowest WER ({best['wer']}) within {max_wer}, {best['distance']} from the nearest other voice, "
            f"pitch variance {best.get('pitch_var')}")
    return out


def _decide(conn, policy: Policy, creator_id, job_id, inputs: dict, output: dict, rationale: str) -> None:
    conn.execute(
        """INSERT INTO decisions (kind, creator_id, inputs, output, policy_version, agent, job_id, step, rationale)
           VALUES ('voice_pick', %s, %s, %s, %s, 'voice_lab', %s, 'voice_pick', %s)
           ON CONFLICT (job_id, step) WHERE job_id IS NOT NULL DO NOTHING""",
        (creator_id, Jsonb(inputs), Jsonb(output), policy.version, job_id, rationale))


def speak_payload(policy: Policy, voice: dict, lines: list[dict], *, seed: int | None = None) -> dict:
    """A voice.speak payload for a creator's designed voice (anchors.voice): lines [{text, delivery,
    pause_after_s}], each generated `voice.takes` times."""
    return {"ref_uri": voice["ref_uri"], "ref_text": voice["ref_text"],
            "lines": [{k: line[k] for k in ("text", "delivery", "pause_after_s") if line.get(k) is not None}
                      for line in lines],
            "takes": int(_cfg(policy).get("takes", 1)), "seed": seed, "params": worker_params(policy)}
