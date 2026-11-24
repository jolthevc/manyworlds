"""A designed voice for every creator (VoxCPM2, worker/handlers/tts_voxcpm.py; policy `voice`).

    design(conn, policy, creator_id) -> job_id     candidates from the bible's voice notes and appearance
    candidates(conn, policy, job) -> list           each candidate's uri and scores, for listening
    pick(conn, policy, creator_id, job, candidate=None) -> dict
                                                    one candidate, chosen automatically (or by index), becomes
                                                    anchors.voice
    bank(conn, policy, creator_id) -> {delivery: job_id}
                                                    one clip of her voice per policy delivery (voice.delivery_bank)
    pick_bank(conn, policy, creator_id) -> dict     the clips that pass become anchors.voice.bank; speak_payload
                                                    then gives a line the clip its delivery names

The automatic choice is never made by ear (owner 2026-11-21): the candidate must say the anchor passage (word error
rate within `voice.max_wer`), must sound unlike every other creator (speaker-embedding cosine distance at least
`voice.min_distance` from each of their anchors), and among those the lowest WER wins, then the liveliest (highest
pitch variance: least monotone). The owner may override it with a candidate's index after listening (the voice
CLI), never with one too close to another creator's voice. The pick is a decision row (kind `voice_pick`).

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


# Natural speech (worker/handlers/tts_voxcpm.speech_config): whole lines or clauses (`synthesis`; `split_clauses`
# is its older spelling), how lines are cloned (`clone`), the gaps between clauses and sentences, trimming and
# fades, per-clause WER retries, the line ceiling, and the experimental breaths and fillers.
NATURAL_KEYS = ("synthesis", "split_clauses", "clone", "clause_min_words", "clause_gap_s", "sentence_gap_s", "trim",
                "fade_ms", "max_wer", "clause_retry_wer_factor", "line_max_wer", "breaths", "disfluency",
                "prompt_tail_silence_s")


def worker_params(policy: Policy) -> dict:
    """What the voice worker needs from policy: the model, its sampling, how lines become natural speech, and the
    speech gate's transcriber."""
    cfg, heard = _cfg(policy), policy.get("review", "video", "intelligibility", default={}) or {}
    params = {k: cfg[k] for k in ("model", "cfg_value", "inference_timesteps", "optimize", "embedding", "pitch_hz",
                                  "pause_s", *NATURAL_KEYS) if k in cfg}
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


def _job(conn, job):
    job_id = job.get("id") if isinstance(job, dict) else getattr(job, "id", job)
    j = conn.execute("SELECT id, kind, status, payload, creator_id FROM jobs WHERE id = %s", (job_id,)).fetchone()
    if not j:
        raise ValueError(f"no job {job_id}")
    return j


def _other_voices(conn, creator_id) -> list[list[float]]:
    """The speaker embeddings of every other creator's designed voice."""
    return [r["e"] for r in conn.execute(
        "SELECT anchors->'voice'->'embedding' AS e FROM creators WHERE id IS DISTINCT FROM %s "
        "AND anchors->'voice'->>'engine' = 'voxcpm'", (creator_id,)).fetchall() if r["e"]]


def _scored(conn, policy: Policy, j, creator_id) -> tuple[list[dict], int]:
    """Each candidate of a design job ({index, uri, wer, pitch_var, seconds, transcript, embedding, asset_id}) with
    its `distance` to the nearest other creator's voice and whether it `qualifies` (within max_wer, at least
    min_distance away), in output order; and how many other voices there are."""
    cfg = _cfg(policy)
    max_wer, min_distance = float(cfg.get("max_wer", 0.1)), float(cfg.get("min_distance", 0.35))
    assets = conn.execute("SELECT id, uri, meta FROM assets WHERE job_id = %s AND kind = 'audio' ORDER BY uri",
                          (j["id"],)).fetchall()
    others = _other_voices(conn, creator_id)
    qualified = {int(c.get("index") or 0) for c in rank(
        [{**(a["meta"] or {})} for a in assets], others, max_wer=max_wer, min_distance=min_distance)}
    out = []
    for a in assets:
        c = {**(a["meta"] or {}), "asset_id": str(a["id"]), "uri": a["uri"]}
        c["distance"] = min((voices.cosine_distance(c.get("embedding") or [], o) for o in others if o), default=2.0)
        c["qualifies"] = int(c.get("index") or 0) in qualified
        out.append(c)
    return out, len(others)


def candidates(conn, policy: Policy, job) -> list[dict]:
    """A design job's candidates for listening (the voice CLI): {index, uri, wer, distance, pitch_var, seconds,
    transcript, qualifies}, best first as `pick` would rank them, then the ones that do not qualify. Empty while
    the job is not done."""
    j = _job(conn, job)
    if j["status"] != "done":
        return []
    scored, _ = _scored(conn, policy, j, j["creator_id"])
    keys = ("index", "uri", "wer", "distance", "pitch_var", "seconds", "transcript", "qualifies")
    order = lambda c: (not c["qualifies"], float(c["wer"]) if c.get("wer") is not None else float("inf"),  # noqa: E731
                       -float(c.get("pitch_var") or 0.0), int(c.get("index") or 0))
    return [{k: c.get(k) for k in keys} for c in sorted(scored, key=order)]


def pick(conn, policy: Policy, creator_id, job, candidate: int | None = None) -> dict:
    """Choose the creator's voice from a finished design job and store it as anchors.voice:
    {engine: voxcpm, ref_uri, ref_text, description, embedding, voice (their Kokoro voice, kept)}.

    `candidate` (a candidate's index) overrides the automatic choice, for the owner after listening
    (`candidates`). It may be over max_wer (the owner heard it), but never closer than min_distance to another
    creator's voice: creators must not sound alike (no coordinated network, CLAUDE.md)."""
    cfg = _cfg(policy)
    j = _job(conn, job)
    if j["status"] != "done":
        return {"picked": None, "reason": f"design job is {j['status']}"}
    scored, n_others = _scored(conn, policy, j, creator_id)
    max_wer, min_distance = float(cfg.get("max_wer", 0.1)), float(cfg.get("min_distance", 0.35))
    ranked = rank(scored, _other_voices(conn, creator_id), max_wer=max_wer, min_distance=min_distance)
    summary = [{k: c.get(k) for k in ("index", "uri", "wer", "pitch_var", "seconds", "transcript")} for c in scored]
    inputs = {"job_id": str(j["id"]), "candidates": summary, "other_voices": n_others,
              "max_wer": max_wer, "min_distance": min_distance, "candidate": candidate}
    if candidate is not None:
        chosen = next((c for c in scored if int(c.get("index") or 0) == int(candidate)), None)
        if chosen is None:
            raise ValueError(f"design job {j['id']} has no candidate {candidate}; it has "
                             f"{[c.get('index') for c in scored]}")
        if chosen["distance"] < min_distance:
            out = {"picked": None, "reason": f"candidate {candidate} is {chosen['distance']} from another creator's "
                                             f"voice, closer than min_distance {min_distance}"}
            return out             # not a decision: nothing changes, the owner can choose another
        ranked = [chosen]
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
           "ranked": [c["index"] for c in ranked], "override": candidate is not None}
    if candidate is None:
        why = (f"lowest WER ({best['wer']}) within {max_wer}, {best['distance']} from the nearest other voice, "
               f"pitch variance {best.get('pitch_var')}")
    else:
        why = (f"the owner's choice after listening: candidate {candidate} (WER {best.get('wer')}, "
               f"{best['distance']} from the nearest other voice)")
    _decide(conn, policy, creator_id, j["id"], inputs, out, why,
            step="voice_pick" if candidate is None else f"voice_pick:candidate:{int(candidate)}")
    return out


def _decide(conn, policy: Policy, creator_id, job_id, inputs: dict, output: dict, rationale: str, *,
            step: str = "voice_pick", kind: str = "voice_pick") -> None:
    conn.execute(
        """INSERT INTO decisions (kind, creator_id, inputs, output, policy_version, agent, job_id, step, rationale)
           VALUES (%s, %s, %s, %s, %s, 'voice_lab', %s, %s, %s)
           ON CONFLICT (job_id, step) WHERE job_id IS NOT NULL DO NOTHING""",
        (kind, creator_id, Jsonb(inputs), Jsonb(output), policy.version, job_id, step, rationale))


def speak_kind(policy: Policy) -> str:
    """The job kind that speaks lines in a designed voice (policy voice.kinds.speak)."""
    return _kind(policy, "speak")


def _deliveries_spoken(policy: Policy, params: dict) -> bool:
    """Whether a line's delivery goes to the model as its instruction: per clone mode (policy
    voice.deliveries_spoken: {ultimate: false, controllable: true})."""
    modes = _cfg(policy).get("deliveries_spoken") or {}
    mode = str(params.get("clone") or "ultimate")
    return bool(modes.get(mode, mode != "ultimate"))


def _line(line: dict, keep_delivery: bool) -> dict:
    out = {k: line[k] for k in ("text", "delivery", "pause_after_s") if line.get(k) is not None}
    if not keep_delivery:
        out.pop("delivery", None)
        out["text"] = voices.split_instruction(out.get("text", ""))[1]
    return out


def speak_payload(policy: Policy, voice: dict, lines: list[dict], *, seed: int | None = None,
                  compare: bool = False) -> dict:
    """A voice.speak payload for a creator's designed voice (anchors.voice): lines [{text, delivery,
    pause_after_s}], each generated `voice.takes` times. `compare` (the breadth test) also makes the other
    synthesis (whole line / clauses) as a second output, for listening side by side.

    Deliveries reach the model only where it follows them (`deliveries_spoken`): with ultimate cloning VoxCPM
    continues the reference transcript and reads the delivery's parenthetical aloud ("A quiet, sincere, one thing
    nobody tells you...": breadth rerun 2026-11-24, three of three directed lines), so the delivery and any inline
    parenthetical are left out there; the line keeps its delivery in the post. Instead, a line whose delivery
    names one of the creator's delivery-bank clips continues from that clip (`prompt`: {ref_uri, ref_text}, its
    name in `bank`), so the delivery comes from the audio it continues, never from words in the text."""
    params = worker_params(policy)
    keep = _deliveries_spoken(policy, params)
    out = []
    for line in lines:
        spoken = _line(line, keep)
        inline = voices.split_instruction(line.get("text") or "")[0]
        chosen = None if keep else bank_prompt(policy, voice, line.get("delivery") or inline)
        if chosen:
            spoken["prompt"] = {"ref_uri": chosen[1]["ref_uri"], "ref_text": chosen[1]["ref_text"]}
            spoken["bank"] = chosen[0]
        out.append(spoken)
    payload = {"ref_uri": voice["ref_uri"], "ref_text": voice["ref_text"], "lines": out,
               "takes": int(_cfg(policy).get("takes", 1)), "seed": seed, "params": params}
    if compare:
        payload["compare"] = True
    return payload


def _bank_cfg(policy: Policy) -> dict:
    return _cfg(policy).get("delivery_bank") or {}


def bank_prompt(policy: Policy, voice: dict, delivery: str | None) -> tuple[str, dict] | None:
    """(name, clip) of the creator's delivery-bank clip a line's delivery asks for (voices.nearest_delivery over
    the policy keywords of the deliveries she has a kept clip for), or None: the base clip."""
    cfg, kept = _bank_cfg(policy), (voice or {}).get("bank") or {}
    if not cfg.get("enabled") or not kept or not (delivery or "").strip():
        return None
    keywords = {name: (spec or {}).get("keywords") or [] for name, spec in (cfg.get("deliveries") or {}).items()
                if (kept.get(name) or {}).get("ref_uri") and (kept.get(name) or {}).get("ref_text")}
    name = voices.nearest_delivery(delivery, keywords)
    return (name, kept[name]) if name else None


# --- the delivery bank (policy voice.delivery_bank; research 2026-11-24-expressive-voice §2.2) -------------------
def bank(conn, policy: Policy, creator_id, *, batch: str | None = None, pools: list[str] | None = None) -> dict:
    """Enqueue one voice.speak job per policy delivery: the delivery's text in the creator's designed voice, made
    with controllable cloning (the instruction is followed there, not read aloud; the base clip is the timbre
    reference), `delivery_bank.takes` takes with the best by WER kept, and its speaker embedding (payload
    `embed`) for `pick_bank`. Idempotent per voice, instruction and text. -> {delivery: job id}."""
    cfg = _bank_cfg(policy)
    row = conn.execute("SELECT anchors FROM creators WHERE id = %s", (creator_id,)).fetchone()
    if not row:
        raise ValueError(f"no creator {creator_id}")
    if voices.engine(row["anchors"]) != "voxcpm":
        raise ValueError("a delivery bank needs a designed voice first: mw voice design, then mw voice pick")
    voice = row["anchors"]["voice"]
    deliveries = cfg.get("deliveries") or {}
    if not deliveries:
        raise ValueError("policy voice.delivery_bank.deliveries is empty")
    params = {**worker_params(policy), "clone": "controllable"}
    jobs = {}
    for name, spec in deliveries.items():
        instruction, text = (spec.get("instruction") or "").strip(), (spec.get("text") or "").strip()
        if not instruction or not text:
            raise ValueError(f"delivery_bank.deliveries.{name} needs an instruction and a text")
        digest = hashlib.sha256(f"{voice['ref_uri']}|{instruction}|{text}".encode()).hexdigest()
        payload = {"ref_uri": voice["ref_uri"], "ref_text": voice["ref_text"],
                   "lines": [{"text": text, "delivery": instruction}], "takes": int(cfg.get("takes", 1)),
                   "seed": int(digest[:8], 16) % (1 << 31), "params": params, "embed": True, "bank": name}
        jobs[name] = queue.enqueue(conn, speak_kind(policy), payload, creator_id=creator_id, batch=batch, pools=pools,
                                   idempotency_key=f"{creator_id}:voice:bank:{name}:{digest[:12]}",
                                   max_attempts=int(policy.get("queue", "max_attempts", default=3)))
    return jobs


def leaked(instruction: str, text: str, heard: str, ignore: list[str] | tuple = ()) -> list[str]:
    """Words of the instruction the transcript has but the text does not (the delivery was read aloud), apart from
    the `ignore` words (policy delivery_bank.leak_ignore: "a", "and"... a misheard filler is not a leak)."""
    from ..speech import words
    said, script, skip = set(words(heard)), set(words(text)), {str(w).lower() for w in ignore}
    return [w for w in dict.fromkeys(words(instruction)) if w in said and w not in script and w not in skip]


def review_clip(policy: Policy, spec: dict, meta: dict, base: list[float], others: list[list[float]]) -> dict:
    """One bank clip judged: {kept, reasons, wer, leaked, distance (to her base clip), nearest_other, transcript}."""
    cfg = _bank_cfg(policy)
    max_wer, max_distance = float(cfg.get("max_wer", 0.1)), float(cfg.get("max_distance", 0.3))
    min_distance = float(_cfg(policy).get("min_distance", 0.35))
    heard = " ".join(str(line.get("transcript", "")) for line in meta.get("lines") or [] if isinstance(line, dict))
    embedding = meta.get("embedding") or []
    others = [o for o in others if o]
    out = {"wer": meta.get("wer"),
           "leaked": leaked(spec.get("instruction", ""), spec.get("text", ""), heard, cfg.get("leak_ignore") or ()),
           "distance": voices.cosine_distance(embedding, base) if embedding and base else None,
           "nearest_other": min((voices.cosine_distance(embedding, o) for o in others), default=None)
           if embedding else None, "transcript": heard[:300]}
    reasons = []
    if out["wer"] is None or float(out["wer"]) > max_wer:
        reasons.append(f"WER {out['wer']} over {max_wer}")
    if out["leaked"]:
        reasons.append(f"read its instruction aloud: {', '.join(out['leaked'])}")
    if out["distance"] is None:
        reasons.append("no speaker embedding to compare with her base clip")
    elif out["distance"] > max_distance:
        reasons.append(f"{out['distance']} from her base clip, over {max_distance}")
    if out["nearest_other"] is not None and out["nearest_other"] < min_distance:
        reasons.append(f"{out['nearest_other']} from another creator's voice, under {min_distance}")
    return {**out, "kept": not reasons, "reasons": reasons}


def pick_bank(conn, policy: Policy, creator_id) -> dict:
    """Review the creator's finished bank jobs (`bank`) and store the clips that pass as anchors.voice.bank
    {delivery: {ref_uri, ref_text, instruction, embedding, wer, distance, asset_id, job_id}}. A delivery whose
    latest clip fails keeps no entry from that job (its lines continue from the base clip). Each verdict is a
    decision row (kind `voice_bank`). -> {kept, rejected: {delivery: reasons}, waiting}."""
    cfg = _bank_cfg(policy)
    row = conn.execute("SELECT anchors FROM creators WHERE id = %s", (creator_id,)).fetchone()
    if not row or voices.engine(row["anchors"]) != "voxcpm":
        return {"kept": [], "rejected": {}, "waiting": [], "reason": "no designed voice"}
    voice = row["anchors"]["voice"]
    base, others = voice.get("embedding") or [], _other_voices(conn, creator_id)
    stored = dict(voice.get("bank") or {})
    out: dict = {"kept": [], "rejected": {}, "waiting": []}
    for name, spec in (cfg.get("deliveries") or {}).items():
        job = conn.execute(
            "SELECT id, status FROM jobs WHERE creator_id = %s AND kind = %s AND payload->>'bank' = %s "
            "AND payload->>'ref_uri' = %s ORDER BY created_at DESC LIMIT 1",
            (creator_id, speak_kind(policy), name, voice["ref_uri"])).fetchone()
        if not job or job["status"] in ("queued", "running"):
            out["waiting"].append(name)
            continue
        audio = conn.execute("SELECT id, uri, meta FROM assets WHERE job_id = %s AND kind = 'audio' "
                             "ORDER BY uri LIMIT 1", (job["id"],)).fetchone()
        verdict = (review_clip(policy, spec, audio["meta"] or {}, base, others) if audio
                   else {"kept": False, "reasons": [f"the job is {job['status']} with no clip"]})
        if verdict["kept"]:
            stored[name] = {"ref_uri": audio["uri"], "ref_text": voices.strip_delivery(spec["text"]),
                            "instruction": spec["instruction"], "embedding": (audio["meta"] or {}).get("embedding"),
                            "wer": verdict["wer"], "distance": verdict["distance"], "asset_id": str(audio["id"]),
                            "job_id": str(job["id"])}
            out["kept"].append(name)
        else:
            if (stored.get(name) or {}).get("job_id") == str(job["id"]):
                stored.pop(name)
            out["rejected"][name] = verdict["reasons"]
        _decide(conn, policy, creator_id, job["id"],
                {"delivery": name, "instruction": spec.get("instruction"), "text": spec.get("text")},
                {**{k: v for k, v in verdict.items() if k != "transcript"}, "delivery": name},
                "kept" if verdict["kept"] else "; ".join(verdict["reasons"]), step=f"voice_bank:{name}",
                kind="voice_bank")
    conn.execute("UPDATE creators SET anchors = jsonb_set(anchors, '{voice,bank}', %s::jsonb) WHERE id = %s",
                 (Jsonb(stored), creator_id))
    return out
