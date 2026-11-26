"""A designed voice for every creator (policy `voice`), cast automatically (owner 2026-11-25: no human gates).

    cast(conn, policy, creator_id) -> job_id       voice.cast: candidates designed from the bible, each speaking the
                                                    casting script in the creator's engine (Qwen3-TTS by default
                                                    until the bake-off says otherwise: default_engine), judged on
                                                    the worker
    pick_cast(conn, policy, creator_id, job)        the best qualifying candidate becomes anchors.voice (voice_eval)
    settle_casts(conn, policy)                      the producer judges every finished cast by itself
    speak_payload(policy, voice, lines)             voice.speak.<engine>, with the per-line check (voice.line_check)

The engines are Qwen3-TTS (the incumbent, owner 2026-11-26: its emotion and inflection) and IndexTTS-2.5 (the
challenger). VoxCPM2 and its design / pick / delivery-bank flow were retired 2026-11-26 by the owner: a creator whose
stored voice a retired engine made speaks with Kokoro (`voices.engine`) until she is cast again
(`voices.needs_recast`); nothing here will speak with it (`speak_kind` and `speak_payload` refuse it).

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


# Natural speech (worker/handlers/speech_audio.speech_config): whole lines or clauses (`synthesis`; `split_clauses`
# is its older spelling), the gaps between clauses and sentences, trimming and fades, per-clause WER retries, the
# line ceiling, and the experimental breaths and fillers.
NATURAL_KEYS = ("synthesis", "split_clauses", "clause_min_words", "clause_gap_s", "sentence_gap_s", "trim",
                "fade_ms", "max_wer", "clause_retry_wer_factor", "line_max_wer", "breaths", "disfluency")


def worker_params(policy: Policy) -> dict:
    """What the voice worker needs from policy: the embedder, the pitch tracker's range, how lines become natural
    speech, and the speech gate's transcriber."""
    cfg, heard = _cfg(policy), policy.get("review", "video", "intelligibility", default={}) or {}
    params = {k: cfg[k] for k in ("embedding", "pitch_hz", "pause_s", *NATURAL_KEYS) if k in cfg}
    params.update(asr_model=heard.get("model", "openai/whisper-large-v3"), language=heard.get("language", "en"))
    return params


def describe(bible: dict) -> str:
    """The voice description a cast designs from: who is speaking (apparent age, gender presentation) and how they
    sound (the bible's voice notes). No parentheses: a script line's leading parenthetical is its delivery."""
    a = (bible or {}).get("appearance") or {}
    who = " ".join(str(x).strip() for x in (f"{a['apparent_age']}-year-old" if a.get("apparent_age") else "",
                                            a.get("gender_presentation") or "") if str(x).strip())
    notes = (bible or {}).get("voice_notes") or ""
    parts = [f"A {who} speaking to a phone camera" if who else "", notes.strip()]
    text = re.sub(r"\s*[()]\s*", ", ", ". ".join(p.rstrip(". ") for p in parts if p))
    return re.sub(r"\s*,(\s*,)+", ",", text).strip(" ,")


def _job(conn, job):
    job_id = job.get("id") if isinstance(job, dict) else getattr(job, "id", job)
    j = conn.execute("SELECT id, kind, status, payload, creator_id FROM jobs WHERE id = %s", (job_id,)).fetchone()
    if not j:
        raise ValueError(f"no job {job_id}")
    return j


def _decide(conn, policy: Policy, creator_id, job_id, inputs: dict, output: dict, rationale: str, *,
            step: str = "voice_cast", kind: str = "voice_cast") -> None:
    conn.execute(
        """INSERT INTO decisions (kind, creator_id, inputs, output, policy_version, agent, job_id, step, rationale)
           VALUES (%s, %s, %s, %s, %s, 'voice_lab', %s, %s, %s)
           ON CONFLICT (job_id, step) WHERE job_id IS NOT NULL DO NOTHING""",
        (kind, creator_id, Jsonb(inputs), Jsonb(output), policy.version, job_id, step, rationale))


# --- engines and automatic casting (owner 2026-11-25: no human selection gates) --------------------------------------
# A creator's voice is cast, not picked by ear: `cast` asks the worker (voice.cast, worker/handlers/tts_engines.py)
# for `voice.casting.candidates` reference voices designed from the bible (gender, age, voice notes), each speaking the
# casting script (neutral, warm, excited, quiet, a question) in the creator's engine, every line measured there;
# `pick_cast` judges them (manyworlds.voice_eval: fit to the bible with a hard gender gate, naturalness,
# expressiveness within and across lines, clarity, consistency, pacing in her archetype's band; at least
# voice.min_distance from every other creator, a floor, never maximised) and stores the best as anchors.voice. The
# producer settles finished casts on its own (`settle_casts`); a launch casts as one of its steps.
def engine_cfg(policy: Policy, engine: str | None) -> dict:
    return ((_cfg(policy).get("engines") or {}).get(engine) or {}) if engine else {}


def _usable(policy: Policy, engine: str | None) -> str:
    """The engine, if it is one this policy can speak with; a retired or unknown engine is refused with the way out."""
    if engine in voices.RETIRED_ENGINES:
        raise ValueError(f"voice engine {engine!r} was retired (2026-11-26): this voice must be cast again "
                         f"(mw voice cast <creator>); until then the creator speaks with Kokoro")
    if not engine or not engine_cfg(policy, engine):
        raise ValueError(f"no policy voice.engines.{engine} (known: {sorted((_cfg(policy).get('engines') or {}))})")
    return str(engine)


def speak_kind(policy: Policy, engine: str | None) -> str:
    """The job kind that speaks lines in a designed voice: the engine's own kind (policy voice.engines.<engine>.kind,
    voice.speak.qwen3 or voice.speak.indextts). A retired engine (VoxCPM2) is refused, never spoken with."""
    kind = engine_cfg(policy, _usable(policy, engine)).get("kind")
    if not kind:
        raise ValueError(f"policy voice.engines.{engine}.kind is not set")
    return str(kind)


def default_engine(conn, policy: Policy) -> str:
    """The engine new casts use: the bake-off's recorded winner (a decision row of kind voice.engine_gate
    .decision_kind), else policy voice.engine, unless the gate still waits for the bake-off (requires_bakeoff): then
    until_bakeoff (Qwen3-TTS, owner 2026-11-26)."""
    cfg = _cfg(policy)
    gate = cfg.get("engine_gate") or {}
    known = set((cfg.get("engines") or {}).keys())
    row = conn.execute("SELECT output FROM decisions WHERE kind = %s ORDER BY created_at DESC LIMIT 1",
                       (gate.get("decision_kind", "voice_engine"),)).fetchone()
    winner = ((row["output"] if row else None) or {}).get("winner")
    if winner in known:
        return str(winner)
    if gate.get("requires_bakeoff") and gate.get("until_bakeoff"):
        return _usable(policy, gate["until_bakeoff"])
    return _usable(policy, cfg.get("engine"))


def line_check_params(policy: Policy, voice: dict | None = None) -> dict:
    """The per-line check the worker runs (policy voice.line_check over voice.judge), with the creator's pace band
    (anchors.voice.pace_wps, set at casting; else the default band)."""
    from .. import voice_eval
    cfg = _cfg(policy)
    band = (voice or {}).get("pace_wps") or list(voice_eval.pace_band(policy, None))
    return {**(cfg.get("judge") or {}), **(cfg.get("line_check") or {}), "pace_wps": [float(x) for x in band]}


def engine_params(policy: Policy, engines: list[str], voice: dict | None = None) -> dict:
    """worker_params plus what the engines, the line check and the judges need."""
    params = worker_params(policy)
    params["line_check"] = line_check_params(policy, voice)
    params["judge_models"] = dict(_cfg(policy).get("judge_models") or {})
    params["engines"] = {e: {k: v for k, v in engine_cfg(policy, _usable(policy, e)).items()
                             if k not in ("emotions", "licence")} for e in engines}
    return params


def _deliveries(policy: Policy) -> dict:
    """The named deliveries a line can ask for (policy voice.deliveries: {name: {keywords}})."""
    return _cfg(policy).get("deliveries") or {}


def emotion_for(policy: Policy, engine: str | None, delivery: str | None) -> dict | None:
    """An engine's own delivery control for a line ({name, vector, alpha}), or None (the reference's own tone). The
    delivery is matched to a name by the policy deliveries' keywords (voices.nearest_delivery); the engine's
    `emotions` table gives its vector. Never text: nothing here reaches the spoken words."""
    table = engine_cfg(policy, engine).get("emotions") or {}
    if not table:
        return None
    keywords = {name: (_deliveries(policy).get(name) or {}).get("keywords") or [] for name in table}
    name = voices.nearest_delivery(delivery, keywords) or ("neutral" if "neutral" in table else None)
    spec = table.get(name) if name else None
    if not spec or not spec.get("vector"):
        return None
    return {"name": name, "vector": [float(x) for x in spec["vector"]],
            "alpha": float(spec.get("alpha", engine_cfg(policy, engine).get("emo_alpha", 1.0)))}


def speak_payload(policy: Policy, voice: dict, lines: list[dict], *, seed: int | None = None) -> dict:
    """A voice.speak.<engine> payload for a creator's designed voice (anchors.voice): the words only (an inline
    parenthetical becomes the delivery), each line's emotion from its delivery, the creator's reference clip as the
    timbre, `takes` per line, and the per-line check (policy voice.line_check). A voice a retired engine made is
    refused (ValueError): it must be cast again.

    A line with performance controls (the shot's levels, direction.spoken_lines `controls`; manyworlds.performance
    .voice_params): IndexTTS's emotion strength follows vocal_expressiveness and energy adds to its happy and
    surprised weights; the shot's pauses scale with pace; an engine whose policy says `instruct: true` also gets the
    voice phrases as its instruction (Qwen3-TTS Base takes none, so as shipped nothing is sent)."""
    from .. import performance
    engine = _usable(policy, voice.get("engine"))
    out = []
    for line in lines:
        inline, words = voices.split_instruction(line.get("text") or "")
        delivery = (line.get("delivery") or "").strip() or inline
        spoken: dict = {"text": words}
        params = performance.voice_params(line.get("controls") or {}, engine, policy) if line.get("controls") else None
        if line.get("pause_after_s") is not None:
            spoken["pause_after_s"] = (round(float(line["pause_after_s"]) * params["pause_scale"], 3)
                                       if params else line["pause_after_s"])
        if delivery:
            spoken["delivery"] = delivery
        emotion = emotion_for(policy, engine, delivery)
        if params:
            emotion = performance.apply_voice(emotion, params, engine, policy)
            if params.get("instruct") and engine_cfg(policy, engine).get("instruct"):
                spoken["instruct"] = params["instruct"]
        if emotion:
            spoken["emotion"] = emotion
        out.append(spoken)
    return {"ref_uri": voice["ref_uri"], "ref_text": voice["ref_text"], "lines": out,
            "takes": int(engine_cfg(policy, engine).get("takes", _cfg(policy).get("takes", 1))), "seed": seed,
            "params": engine_params(policy, [engine], voice)}


def _casting(policy: Policy) -> dict:
    return _cfg(policy).get("casting") or {}


def cast_kind(policy: Policy) -> str:
    return str(_casting(policy).get("kind", "voice.cast"))


def fit_of(bible: dict) -> dict:
    """What the voice must fit: {gender, age} from the bible's appearance."""
    from .. import voice_eval
    a = (bible or {}).get("appearance") or {}
    age = a.get("apparent_age")
    return {"gender": voice_eval.gender_of(a.get("gender_presentation")),
            "age": float(age) if isinstance(age, (int, float)) and not isinstance(age, bool) else None}


def cast_payload(policy: Policy, *, description: str, engine: str, seed: int, n: int | None = None,
                 script: list[dict] | None = None) -> dict:
    """The voice.cast payload: candidates designed from `description` reading voice.anchor_text, each speaking the
    casting script (each line's emotion as the engine takes it)."""
    c = _casting(policy)
    engine = _usable(policy, engine)
    designer = str(c.get("designer") or "qwen3_design")
    lines = [{"name": spec.get("name"), "text": spec["text"],
              "emotion": emotion_for(policy, engine, spec.get("delivery"))} for spec in script or c.get("script") or []]
    engines = sorted({engine, "qwen3"} if designer == "qwen3_design" else {engine})
    return {"engine": engine, "designer": designer, "description": description,
            "anchor_text": (_cfg(policy).get("anchor_text") or "").strip(),
            "n": int(n or c.get("candidates", 4)), "seed": seed, "script": lines,
            "params": {**engine_params(policy, engines), "listen_gap_s": float(c.get("listen_gap_s", 0.6))}}


def cast(conn, policy: Policy, creator_id, *, engine: str | None = None, attempt: int = 0, batch: str | None = None,
         pools: list[str] | None = None):
    """Enqueue the creator's casting (voice.cast). Idempotent per bible description, engine and attempt; a later
    attempt (after a cast where no candidate qualified) draws new seeds."""
    row = conn.execute("SELECT handle, bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
    if not row:
        raise ValueError(f"no creator {creator_id}")
    engine = _usable(policy, engine or default_engine(conn, policy))
    description = describe(row["bible"] or {})
    if not description or not (_cfg(policy).get("anchor_text") or "").strip():
        raise ValueError("casting needs the bible's voice notes or appearance, and policy voice.anchor_text")
    digest = hashlib.sha256(f"{creator_id}|{description}|{engine}|{attempt}".encode()).hexdigest()
    payload = cast_payload(policy, description=description, engine=engine, seed=int(digest[:8], 16) % (1 << 30))
    payload["purpose"] = "creator"          # settle_casts stores only these (a bake-off cast never replaces a voice)
    return queue.enqueue(conn, cast_kind(policy), payload, creator_id=creator_id, batch=batch, pools=pools,
                         idempotency_key=f"{creator_id}:voice:cast:{digest[:12]}",
                         max_attempts=int(policy.get("queue", "max_attempts", default=3)))


def _designed_others(conn, creator_id) -> list[list[float]]:
    return [r["e"] for r in conn.execute(
        "SELECT anchors->'voice'->'embedding' AS e FROM creators WHERE id IS DISTINCT FROM %s "
        "AND anchors->'voice'->>'engine' = ANY(%s)", (creator_id, list(voices.DESIGNED_ENGINES))).fetchall() if r["e"]]


def judge_cast(policy: Policy, metas: list[dict], *, fit: dict, band: tuple[float, float]) -> list[dict]:
    """Each candidate's summary (voice_eval.cast_summary), plus a reason when its reference clip did not say the
    anchor text (a mis-said reference makes every cloned line drift)."""
    from .. import voice_eval
    judge = _cfg(policy).get("judge") or {}
    max_ref_wer = float(_casting(policy).get("max_ref_wer", 0.1))
    out = []
    for m in metas:
        s = voice_eval.cast_summary(m.get("lines") or [], judge, band, fit, m.get("who") or {})
        if m.get("ref_wer") is None or float(m["ref_wer"]) > max_ref_wer:
            s = {**s, "qualifies": False,
                 "reasons": [*s["reasons"], f"reference WER {m.get('ref_wer')} over {max_ref_wer}"]}
        out.append(s)
    return out


def pick_cast(conn, policy: Policy, creator_id, job) -> dict:
    """Judge a finished cast job and store the best qualifying candidate as anchors.voice {engine, ref_uri, ref_text,
    description, embedding, scores, fit, pace_wps, asset_id, cast_job_id, voice (Kokoro kept)}. A decision row
    (kind voice_cast) records every candidate's scores and why. -> {picked, ranked, reason}."""
    from .. import voice_eval
    from ..brain import archetypes
    j = _job(conn, job)
    if j["status"] != "done":
        return {"picked": None, "reason": f"cast job is {j['status']}"}
    row = conn.execute("SELECT bible, anchors FROM creators WHERE id = %s", (creator_id,)).fetchone()
    bible = (row["bible"] if row else None) or {}
    fit, band = fit_of(bible), voice_eval.pace_band(policy, archetypes.for_creator(policy, bible))
    assets = conn.execute("SELECT id, uri, meta FROM assets WHERE job_id = %s AND kind = 'audio' ORDER BY uri",
                          (j["id"],)).fetchall()
    metas = [a["meta"] or {} for a in assets]
    summaries = judge_cast(policy, metas, fit=fit, band=band)
    cands = [{"index": int(m.get("index", i)), "embedding": m.get("embedding") or [], "summary": s, "asset": a}
             for i, (a, m, s) in enumerate(zip(assets, metas, summaries))]
    min_distance = float(_cfg(policy).get("min_distance", 0.35))
    ranked = voice_eval.rank_cast(cands, _designed_others(conn, creator_id), min_distance)
    payload = j["payload"] or {}
    inputs = {"job_id": str(j["id"]), "engine": payload.get("engine"), "fit": fit, "pace_wps": list(band),
              "min_distance": min_distance,
              "candidates": [{"index": c["index"], **{k: v for k, v in c["summary"].items() if k != "lines"}}
                             for c in cands]}
    if not ranked:
        out = {"picked": None, "reason": "no candidate qualified: " + "; ".join(
            f"{c['index']}: {', '.join(c['summary']['reasons']) or 'too close to another creator'}" for c in cands)}
        _decide(conn, policy, creator_id, j["id"], inputs, out, out["reason"][:500], step="voice_cast",
                kind="voice_cast")
        return out
    best = ranked[0]
    meta = best["asset"]["meta"] or {}
    old = ((row["anchors"] if row else None) or {}).get("voice") or {}
    summary = {k: v for k, v in best["summary"].items() if k != "lines"}
    voice = {"engine": meta.get("engine") or payload.get("engine"), "ref_uri": best["asset"]["uri"],
             "ref_text": voices.strip_delivery(meta.get("ref_text") or payload.get("anchor_text", "")),
             "description": payload.get("description", ""), "embedding": best["embedding"],
             "asset_id": str(best["asset"]["id"]), "cast_job_id": str(j["id"]), "scores": summary, "fit": fit,
             "pace_wps": list(band), **{k: old[k] for k in ("voice", "speed") if k in old}}
    conn.execute("UPDATE creators SET anchors = coalesce(anchors, '{}'::jsonb) || jsonb_build_object('voice', %s::jsonb) "
                 "WHERE id = %s", (Jsonb(voice), creator_id))
    out = {"picked": {"index": best["index"], "uri": best["asset"]["uri"], "composite": summary["composite"],
                      "distance": best["distance"], "engine": voice["engine"]},
           "ranked": [c["index"] for c in ranked]}
    why = (f"{voice['engine']} candidate {best['index']}: composite {summary['composite']} (lines "
           f"{summary['line_composite']}, fit {summary['fit']}, pass share {summary['pass_share']}), "
           f"{best['distance']} from the nearest other voice (floor {min_distance})")
    _decide(conn, policy, creator_id, j["id"], inputs, out, why, step="voice_cast", kind="voice_cast")
    return out


def settle_casts(conn, policy: Policy) -> dict:
    """Pick from every finished creator cast not judged yet (the producer loop): no person in between."""
    rows = conn.execute(
        """SELECT j.id, j.creator_id FROM jobs j WHERE j.kind = %s AND j.status = 'done'
           AND j.payload->>'purpose' = 'creator' AND j.creator_id IS NOT NULL
           AND NOT EXISTS (SELECT 1 FROM decisions d WHERE d.job_id = j.id AND d.step = 'voice_cast')
           ORDER BY j.created_at""", (cast_kind(policy),)).fetchall()
    picked = sum(pick_cast(conn, policy, r["creator_id"], r["id"]).get("picked") is not None for r in rows)
    return {"judged": len(rows), "picked": picked}


def cast_status(conn, policy: Policy, creator_id) -> dict:
    """Her latest cast: the job, the verdict (decision voice_cast: every candidate's scores, the pick and why) and the
    voice she speaks with now."""
    job = conn.execute("SELECT id, status, batch, payload->>'engine' AS engine FROM jobs WHERE creator_id = %s "
                       "AND kind = %s ORDER BY created_at DESC LIMIT 1", (creator_id, cast_kind(policy))).fetchone()
    anchors = conn.execute("SELECT anchors FROM creators WHERE id = %s", (creator_id,)).fetchone()["anchors"] or {}
    voice = anchors.get("voice") or {}
    current = {k: voice.get(k) for k in ("engine", "ref_uri", "cast_job_id", "scores", "pace_wps")}
    recast = {"needs_recast": True, "speaks_with": "kokoro"} if voices.needs_recast(anchors) else {}
    if not job:
        return {"job": None, "voice": current, **recast, "note": "no cast yet: mw voice cast <creator>"}
    d = conn.execute("SELECT inputs, output, rationale FROM decisions WHERE job_id = %s AND step = 'voice_cast'",
                     (job["id"],)).fetchone()
    return {"job": str(job["id"]), "status": job["status"], "batch": job["batch"], "engine": job["engine"],
            "verdict": ({"output": d["output"], "why": d["rationale"], "candidates": d["inputs"].get("candidates")}
                        if d else None), "voice": current, **recast}


def quality_stats(conn, policy: Policy, creator_id=None, *, days: int = 30) -> dict:
    """Quality per accepted minute of voice lines over the last `days` (the line check's meta on every
    voice.speak.<engine> asset, and the GPU time the ledger attributes to those jobs): per engine {lines, passed,
    accepted_minutes, minutes, composite, repairs, gpu_usd, usd_per_accepted_minute}."""
    rows = conn.execute(
        """SELECT j.id, j.kind, a.meta FROM jobs j JOIN assets a ON a.job_id = j.id AND a.kind = 'audio'
           WHERE j.kind LIKE 'voice.speak.%%' AND j.status = 'done'
             AND j.created_at > now() - make_interval(days => %s) AND (%s::uuid IS NULL OR j.creator_id = %s::uuid)""",
        (int(days), creator_id, creator_id)).fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        meta = r["meta"] or {}
        e = out.setdefault(meta.get("engine") or r["kind"].rsplit(".", 1)[-1],
                           {"lines": 0, "passed": 0, "accepted_seconds": 0.0, "seconds": 0.0, "composite_sum": 0.0,
                            "repairs": 0, "jobs": set()})
        e["jobs"].add(r["id"])
        for line in meta.get("lines") or []:
            j = line.get("judge")
            if not j:
                continue
            secs = max(0.0, float(line.get("end_s") or 0) - float(line.get("start_s") or 0))
            e["lines"] += 1
            e["passed"] += bool(j.get("passed"))
            e["accepted_seconds"] += secs if j.get("passed") else 0.0
            e["seconds"] += secs
            e["composite_sum"] += float(j.get("composite") or 0)
            e["repairs"] += int(j.get("repairs") or 0)
    stats = {}
    for name, e in out.items():
        usd = float(conn.execute("SELECT coalesce(sum(usd), 0) AS u FROM cost_entries WHERE category = 'gpu' "
                                 "AND job_id = ANY(%s::uuid[])", ([str(x) for x in e["jobs"]],)).fetchone()["u"])
        acc = e["accepted_seconds"] / 60
        stats[name] = {"lines": e["lines"], "passed": e["passed"], "accepted_minutes": round(acc, 3),
                       "minutes": round(e["seconds"] / 60, 3), "repairs": e["repairs"],
                       "composite": round(e["composite_sum"] / e["lines"], 4) if e["lines"] else None,
                       "gpu_usd": round(usd, 4), "usd_per_accepted_minute": round(usd / acc, 4) if acc else None}
    return stats
