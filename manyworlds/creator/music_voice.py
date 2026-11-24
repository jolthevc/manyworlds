"""An artist's singing voice (policy music.voice, OFF; spec 2026-11-23-music-department.md section 5): the same voice
on every song, while style and delivery change per song.

    advance(conn, policy, creator_id, candidate=None) -> dict    one step of the flow, idempotent; call it again
                                                                  until `stage` is `ready`
  seed       music.voice.creation.seed_candidates short ACE-Step songs from the artist's vocal tags (music.song
             with `embed`: the worker stores each vocal stem's voice embedding).
  pick       the seed timbre: among candidates whose song passed its check, the one farthest from every other
             artist's voice (at least creation.min_distance away), best lyric coverage first. The owner may pick a
             candidate by index after listening, never one too close to another artist. Stored as voice version 1
             (adapter soulx_svc: the seed's vocal stem is the prompt clip, its embedding the centroid). A decision
             row (kind music_voice_pick).
  dataset    songs across her registers (creation.registers: keys, tags, tempo), songs_per_register each, and every
             vocal converted zero-shot toward the seed (music.convert_voice, soulx_svc).
  train      the most similar dataset_keep_share of the converted vocals (similarity to the seed's centroid) train
             her RVC model (music.train_voice), once there are at least min_train_clips.
  store      the trained model becomes the next version (adapter rvc, centroid of the training clips, the seed's
             prompt clip kept), refused like every rvc job while the adapter's licence is unverified.

Each version is a music_voices row (versions only grow); the current one is copied into creators.anchors.music_voice,
which production and the bench read. The flow's own job lists live in creators.anchors.music_voice_lab.
Original voices only: everything starts from our own ACE-Step vocals; no real singer's recording is ever an input.
"""

from __future__ import annotations

import hashlib
import statistics

from psycopg.types.json import Jsonb

from .. import music, queue
from .. import singing_voice as sv
from ..policy import Policy

# Original lyrics for voice material (written for this; no real song): open vowels, held notes and plain words, so
# the dataset covers sustained and syllabic singing in every register.
LYRICS = [
    "[Verse]\nMorning comes in slow and golden\nI hold the light the way you showed me\n"
    "[Chorus]\nOh, oh, carry me home\nAcross the water, all alone\nOh, oh, carry me home",
    "[Verse]\nPaper boats along the gutter\nEvery street a little river\n"
    "[Chorus]\nAnd I sing, la la la\nTo the rain, la la la\nUntil the evening finds us",
    "[Verse]\nCount the windows on the high road\nOne for every name I know\n"
    "[Chorus]\nHold on, hold on, the night is long\nI keep the tune, I keep it strong",
]


def _creation(policy: Policy) -> dict:
    return sv.cfg(policy).get("creation") or {}


def _row(conn, creator_id) -> dict:
    row = conn.execute("SELECT id, handle, bible, anchors FROM creators WHERE id = %s", (creator_id,)).fetchone()
    if not row:
        raise ValueError(f"no creator {creator_id}")
    return row


def current(conn, creator_id) -> dict | None:
    """The artist's current voice asset (anchors.music_voice), or None."""
    row = conn.execute("SELECT anchors->'music_voice' AS v FROM creators WHERE id = %s", (creator_id,)).fetchone()
    return (row or {}).get("v") or None


def other_centroids(conn, creator_id) -> list[list[float]]:
    """Every other artist's current voice centroid."""
    return [r["c"] for r in conn.execute(
        "SELECT anchors->'music_voice'->'centroid' AS c FROM creators WHERE id IS DISTINCT FROM %s "
        "AND anchors ? 'music_voice'", (creator_id,)).fetchall() if r["c"]]


def vocal_tags(policy: Policy, bible: dict) -> str:
    """Her vocal and genre words: the bible's music.vocal_tags, else creation.default_tags."""
    m = (bible or {}).get("music") or {}
    return music.tags_text(m.get("vocal_tags") or m.get("tags") or _creation(policy).get("default_tags") or "")


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def _song(conn, policy: Policy, creator_id, spec: dict, *, seed: int, key: str, batch: str | None):
    payload = music.song_payload(policy, spec, seed=seed, check=True, embed=sv.similarity_settings(policy))
    return queue.enqueue(conn, music.kind(policy), payload, creator_id=creator_id, batch=batch, idempotency_key=key,
                         max_attempts=int(policy.get("queue", "max_attempts", default=3)))


def seed_jobs(conn, policy: Policy, creator_id, *, batch: str | None = None) -> list[str]:
    """The seed candidates: short songs in her vocal tags, one seed each (the same bible gives the same jobs)."""
    c = _creation(policy)
    row = _row(conn, creator_id)
    tags = vocal_tags(policy, row["bible"] or {})
    if not tags:
        raise ValueError("no vocal tags: the bible's music.vocal_tags or music.voice.creation.default_tags")
    digest = hashlib.sha256(f"{creator_id}|{tags}".encode()).hexdigest()[:12]
    out = []
    for i in range(int(c.get("seed_candidates", 8))):
        spec = {"title": f"voice seed {i + 1}", "tags": tags, "lyrics": LYRICS[i % len(LYRICS)],
                "duration_s": float(c.get("seed_seconds", 30))}
        out.append(str(_song(conn, policy, creator_id, spec, seed=_seed(creator_id, digest, i),
                             key=f"{creator_id}:music_voice:seed:{digest}:{i}", batch=batch)))
    return out


def _jobs_done(conn, job_ids: list[str]) -> tuple[bool, int]:
    """(every job finished, how many are done)."""
    rows = conn.execute("SELECT status FROM jobs WHERE id = ANY(%s::uuid[])", (list(job_ids),)).fetchall()
    finished = all(r["status"] in ("done", "failed", "cancelled", "dead") for r in rows) and len(rows) == len(job_ids)
    return finished, sum(1 for r in rows if r["status"] == "done")


def _stems(conn, job_id) -> dict:
    """A done music.song job's outputs by role: {role: {uri, meta}}."""
    rows = conn.execute("SELECT a.uri, a.meta FROM assets a JOIN jobs j ON j.id = a.job_id WHERE a.job_id = %s "
                        "AND j.status = 'done' AND a.kind = 'audio' ORDER BY a.uri", (job_id,)).fetchall()
    return {str((r["meta"] or {}).get("role")): {"uri": r["uri"], "meta": r["meta"] or {}} for r in rows}


def candidates(conn, policy: Policy, creator_id, job_ids: list[str]) -> list[dict]:
    """Each seed candidate {index, job_id, vocals_uri, embedding, passed, line_coverage, distance, qualifies}, in
    job order. `distance` is 1 - cosine to the nearest other artist (2.0 with none)."""
    min_distance = float(_creation(policy).get("min_distance", 0.35))
    others = other_centroids(conn, creator_id)
    out = []
    for i, job in enumerate(job_ids):
        st = _stems(conn, job)
        v, mix = st.get("vocals"), st.get("mix")
        emb = (v or {}).get("meta", {}).get("embedding") or []
        check = (mix or {}).get("meta", {}).get("check") or {}
        distance = min((1 - sv.cosine(emb, o) for o in others), default=2.0) if emb else None
        c = {"index": i, "job_id": str(job), "vocals_uri": (v or {}).get("uri"), "embedding": emb,
             "passed": check.get("passed"), "line_coverage": check.get("line_coverage"),
             "distance": round(distance, 4) if distance is not None else None}
        c["qualifies"] = bool(v and emb and c["passed"] is not False and distance is not None
                              and distance >= min_distance)
        out.append(c)
    return out


def rank(cands: list[dict]) -> list[dict]:
    """Qualifying candidates, best first: lyric coverage, then distance from the other artists, then index."""
    return sorted((c for c in cands if c["qualifies"]),
                  key=lambda c: (-(c.get("line_coverage") or 0.0), -(c.get("distance") or 0.0), c["index"]))


def store(conn, policy: Policy, creator_id, *, adapter: str, centroid: list[float], prompt_uri: str,
          model_uri: str | None = None, index_uri: str | None = None, source: dict | None = None,
          calibration: dict | None = None) -> dict:
    """A new version of her voice asset: a music_voices row (the next version) and anchors.music_voice."""
    model = str(sv.similarity_settings(policy).get("model") or "speechbrain/spkrec-ecapa-voxceleb")
    row = conn.execute(
        """INSERT INTO music_voices (creator_id, version, adapter, centroid, embedding_model, prompt_uri, model_uri,
                                     index_uri, calibration, source, policy_version)
           SELECT %s, coalesce(max(version), 0) + 1, %s, %s, %s, %s, %s, %s, %s, %s, %s
           FROM music_voices WHERE creator_id = %s
           RETURNING id, version""",
        (creator_id, adapter, Jsonb(centroid), model, prompt_uri, model_uri, index_uri, Jsonb(calibration or {}),
         Jsonb(source or {}), policy.version, creator_id)).fetchone()
    asset = {"id": str(row["id"]), "version": int(row["version"]), "adapter": adapter, "centroid": centroid,
             "embedding_model": model, "prompt_uri": prompt_uri, "model_uri": model_uri, "index_uri": index_uri}
    conn.execute("UPDATE creators SET anchors = coalesce(anchors, '{}'::jsonb) || jsonb_build_object('music_voice', "
                 "%s::jsonb) WHERE id = %s", (Jsonb(asset), creator_id))
    return asset


def pick(conn, policy: Policy, creator_id, job_ids: list[str], candidate: int | None = None) -> dict:
    """Choose the seed timbre and store it as her first voice version (see the module docstring)."""
    cands = candidates(conn, policy, creator_id, job_ids)
    ranked = rank(cands)
    min_distance = float(_creation(policy).get("min_distance", 0.35))
    if candidate is not None:
        chosen = next((c for c in cands if c["index"] == int(candidate)), None)
        if chosen is None:
            raise ValueError(f"no seed candidate {candidate}; there are {len(cands)}")
        if chosen["distance"] is None or chosen["distance"] < min_distance or not chosen["vocals_uri"]:
            return {"picked": None, "reason": f"candidate {candidate} is {chosen['distance']} from another artist's "
                                              f"voice (min_distance {min_distance}) or has no vocal stem"}
        ranked = [chosen]
    summary = [{k: c.get(k) for k in ("index", "job_id", "passed", "line_coverage", "distance", "qualifies")}
               for c in cands]
    if not ranked:
        out = {"picked": None, "reason": "no seed candidate passed its check and sounded unlike the other artists"}
        _decide(conn, policy, creator_id, job_ids[0] if job_ids else None, {"candidates": summary}, out, out["reason"],
                step="music_voice_pick")
        return out
    best = ranked[0]
    asset = store(conn, policy, creator_id, adapter="soulx_svc", centroid=sv.centroid([best["embedding"]]),
                  prompt_uri=best["vocals_uri"], source={"seed_job": best["job_id"], "candidate": best["index"],
                                                         "seed_jobs": list(job_ids)})
    out = {"picked": {k: best.get(k) for k in ("index", "job_id", "line_coverage", "distance")},
           "version": asset["version"], "override": candidate is not None}
    why = (f"the owner's choice after listening: candidate {candidate}" if candidate is not None else
           f"best lyric coverage ({best.get('line_coverage')}) among candidates {best['distance']} or more from the "
           f"nearest other artist")
    _decide(conn, policy, creator_id, best["job_id"], {"candidates": summary, "min_distance": min_distance}, out, why,
            step="music_voice_pick" if candidate is None else f"music_voice_pick:candidate:{int(candidate)}")
    return out


def _decide(conn, policy: Policy, creator_id, job_id, inputs: dict, output: dict, rationale: str, *, step: str):
    conn.execute(
        """INSERT INTO decisions (kind, creator_id, inputs, output, policy_version, agent, job_id, step, rationale)
           VALUES ('music_voice_pick', %s, %s, %s, %s, 'music_voice', %s, %s, %s)
           ON CONFLICT (job_id, step) WHERE job_id IS NOT NULL DO NOTHING""",
        (creator_id, Jsonb(inputs), Jsonb(output), policy.version, job_id, step, rationale))


def dataset_jobs(conn, policy: Policy, creator_id, asset: dict, *, batch: str | None = None) -> list[str]:
    """Songs across her registers for the training set (keys, tags and tempo per register; same jobs on a rerun)."""
    c = _creation(policy)
    tags = vocal_tags(policy, _row(conn, creator_id)["bible"] or {})
    out, n = [], int(c.get("songs_per_register", 7))
    for reg, r in sorted((c.get("registers") or {}).items()):
        keys = list(r.get("keys") or ["C major"])
        lo, hi = (r.get("bpm") or [90, 110])
        for k in range(n):
            bpm = int(round(float(lo) + (float(hi) - float(lo)) * (k / max(1, n - 1))))
            spec = {"title": f"voice {reg} {k + 1}", "tags": f"{tags}, {r.get('tags') or ''}".strip(", "),
                    "lyrics": LYRICS[k % len(LYRICS)], "key": keys[k % len(keys)], "bpm": bpm,
                    "duration_s": float(c.get("dataset_seconds", 60))}
            out.append(str(_song(conn, policy, creator_id, spec, seed=_seed(creator_id, asset["version"], reg, k),
                                 key=f"{creator_id}:music_voice:v{asset['version']}:data:{reg}:{k}", batch=batch)))
    return out


def conversions(conn, policy: Policy, creator_id, asset: dict, song_jobs: list[str], *,
                batch: str | None = None) -> list[str]:
    """Each finished dataset song's vocal converted zero-shot toward the seed (soulx_svc); a song without a vocal
    stem is skipped. One job per song (idempotent)."""
    out = []
    for job in song_jobs:
        st = _stems(conn, job)
        if "vocals" not in st or "instrumental" not in st:
            continue
        payload = sv.convert_payload(policy, asset, vocals_uri=st["vocals"]["uri"],
                                     instrumental_uri=st["instrumental"]["uri"], seed=_seed(job, "convert"),
                                     adapter="soulx_svc")
        out.append(str(queue.enqueue(conn, sv.kind(policy, "convert"), payload, creator_id=creator_id, batch=batch,
                                     idempotency_key=f"{creator_id}:music_voice:v{asset['version']}:convert:{job}",
                                     max_attempts=int(policy.get("queue", "max_attempts", default=3)))))
    return out


def converted(conn, job_ids: list[str]) -> list[dict]:
    """The converted vocals of done conversion jobs: [{job_id, uri, similarity, embedding, seconds, line_coverage}]."""
    rows = conn.execute(
        "SELECT a.job_id, a.uri, a.meta FROM assets a JOIN jobs j ON j.id = a.job_id WHERE a.job_id = ANY(%s::uuid[]) "
        "AND j.status = 'done' AND a.meta->>'role' = 'vocals' ORDER BY j.created_at, a.uri", (list(job_ids),)).fetchall()
    return [{"job_id": str(r["job_id"]), "uri": r["uri"], **{k: (r["meta"] or {}).get(k) for k in
            ("similarity", "embedding", "seconds", "line_coverage", "voice_version")}} for r in rows]


def keep_for_training(clips: list[dict], keep_share: float) -> list[dict]:
    """The most similar `keep_share` of the clips (pure; a clip without a score is dropped), most similar first."""
    scored = sorted((c for c in clips if c.get("similarity") is not None), key=lambda c: (-c["similarity"], c["uri"]))
    n = max(1, int(round(len(scored) * float(keep_share)))) if scored else 0
    return scored[:n]


def train_job(conn, policy: Policy, creator_id, asset: dict, clips: list[dict], *, batch: str | None = None):
    """The RVC training job for the kept clips (sv.train_payload refuses while rvc's licence is unverified)."""
    uris = [c["uri"] for c in clips]
    digest = hashlib.sha256("|".join(uris).encode()).hexdigest()[:12]
    payload = sv.train_payload(policy, uris, seed=_seed(creator_id, digest), name=f"{str(creator_id)[:8]}_{digest}")
    return str(queue.enqueue(conn, sv.kind(policy, "train"), payload, creator_id=creator_id, batch=batch,
                             idempotency_key=f"{creator_id}:music_voice:v{asset['version']}:train:{digest}",
                             max_attempts=int(policy.get("queue", "max_attempts", default=3))))


def store_trained(conn, policy: Policy, creator_id, asset: dict, job_id, clips: list[dict]) -> dict:
    """A finished training job's model and index as her next voice version (centroid of the training clips)."""
    rows = {r["kind"]: r for r in conn.execute(
        "SELECT a.kind, a.uri, a.meta FROM assets a JOIN jobs j ON j.id = a.job_id WHERE a.job_id = %s "
        "AND j.status = 'done'", (job_id,)).fetchall()}
    if "voice_model" not in rows:
        raise ValueError(f"training job {job_id} has no model")
    center = (rows["voice_model"]["meta"] or {}).get("centroid") or sv.centroid(c["embedding"] for c in clips
                                                                                if c.get("embedding"))
    same = [c["similarity"] for c in clips if c.get("similarity") is not None]
    cal = {"training_similarity_median": round(statistics.median(same), 4) if same else None, "clips": len(clips)}
    return store(conn, policy, creator_id, adapter="rvc", centroid=center, prompt_uri=asset["prompt_uri"],
                 model_uri=rows["voice_model"]["uri"], index_uri=(rows.get("voice_index") or {}).get("uri"),
                 source={"train_job": str(job_id), "from_version": asset["version"],
                         "clips": [c["uri"] for c in clips]}, calibration=cal)


def _lab(conn, creator_id) -> dict:
    row = conn.execute("SELECT anchors->'music_voice_lab' AS l FROM creators WHERE id = %s", (creator_id,)).fetchone()
    return dict((row or {}).get("l") or {})


def _save_lab(conn, creator_id, lab: dict) -> None:
    conn.execute("UPDATE creators SET anchors = coalesce(anchors, '{}'::jsonb) || jsonb_build_object("
                 "'music_voice_lab', %s::jsonb) WHERE id = %s", (Jsonb(lab), creator_id))


def advance(conn, policy: Policy, creator_id, *, candidate: int | None = None, batch: str | None = None) -> dict:
    """One step of the voice flow (module docstring): {stage, ...}. Safe to call again at any time."""
    lab = _lab(conn, creator_id)
    try:
        return _advance(conn, policy, creator_id, lab, candidate=candidate, batch=batch)
    finally:
        _save_lab(conn, creator_id, lab)


def _advance(conn, policy: Policy, creator_id, lab: dict, *, candidate, batch) -> dict:
    c = _creation(policy)
    asset = current(conn, creator_id)
    if not asset:
        if not lab.get("seed_jobs"):
            lab["seed_jobs"] = seed_jobs(conn, policy, creator_id, batch=batch)
            return {"stage": "seed", "jobs": lab["seed_jobs"]}
        finished, done = _jobs_done(conn, lab["seed_jobs"])
        if not finished:
            return {"stage": "seed", "waiting": len(lab["seed_jobs"]) - done}
        out = pick(conn, policy, creator_id, lab["seed_jobs"], candidate)
        return {"stage": "pick", **out}
    if asset.get("model_uri"):
        return {"stage": "ready", "version": asset["version"], "adapter": asset["adapter"]}
    if not lab.get("dataset_jobs"):
        lab["dataset_jobs"] = dataset_jobs(conn, policy, creator_id, asset, batch=batch)
        return {"stage": "dataset", "jobs": len(lab["dataset_jobs"])}
    finished, _ = _jobs_done(conn, lab["dataset_jobs"])
    if not finished:
        return {"stage": "dataset", "waiting": True}
    if not lab.get("convert_jobs"):
        lab["convert_jobs"] = conversions(conn, policy, creator_id, asset, lab["dataset_jobs"], batch=batch)
        return {"stage": "convert", "jobs": len(lab["convert_jobs"])}
    finished, _ = _jobs_done(conn, lab["convert_jobs"])
    if not finished:
        return {"stage": "convert", "waiting": True}
    kept = keep_for_training(converted(conn, lab["convert_jobs"]), float(c.get("dataset_keep_share", 0.8)))
    minutes = round(sum(float(k.get("seconds") or 0) for k in kept) / 60, 1)
    if not lab.get("train_job"):
        try:
            lab["train_job"] = train_job(conn, policy, creator_id, asset, kept, batch=batch)
        except sv.VoiceRefused as e:
            return {"stage": "train", "refused": str(e), "clips": len(kept), "minutes": minutes}
        return {"stage": "train", "job": lab["train_job"], "clips": len(kept), "minutes": minutes}
    finished, done = _jobs_done(conn, [lab["train_job"]])
    if not finished:
        return {"stage": "train", "waiting": True}
    if not done:
        return {"stage": "train", "failed": lab["train_job"]}
    new = store_trained(conn, policy, creator_id, asset, lab["train_job"], kept)
    return {"stage": "ready", "version": new["version"], "adapter": new["adapter"]}


# --- calibration -------------------------------------------------------------------------------------------------

def calibration(conn, policy: Policy, *, kind: str | None = None) -> dict:
    """Same-artist vs other-artist spread over every converted vocal so far (music.convert_voice outputs with an
    embedding): each vocal against its own artist's current centroid (same) and every other artist's (other);
    singing_voice.calibrate proposes the gate's min_similarity (None while they overlap)."""
    rows = conn.execute(
        "SELECT j.creator_id, a.meta->'embedding' AS e FROM assets a JOIN jobs j ON j.id = a.job_id "
        "WHERE j.kind = %s AND j.status = 'done' AND a.meta->>'role' = 'vocals' AND a.meta ? 'embedding'",
        (kind or sv.kind(policy, "convert"),)).fetchall()
    cents = {str(r["id"]): r["c"] for r in conn.execute(
        "SELECT id, anchors->'music_voice'->'centroid' AS c FROM creators WHERE anchors ? 'music_voice'").fetchall()
        if r["c"]}
    return sv.calibrate(*spread([(str(r["creator_id"]), r["e"]) for r in rows if r["e"]], cents))


def spread(vocals: list[tuple[str, list[float]]], centroids: dict[str, list[float]]) -> tuple[list[float], list[float]]:
    """(same-artist scores, other-artist scores) for (artist, embedding) pairs against every artist's centroid."""
    same, other = [], []
    for artist, emb in vocals:
        for who, cent in centroids.items():
            (same if who == artist else other).append(round(sv.cosine(emb, cent), 4))
    return same, other
