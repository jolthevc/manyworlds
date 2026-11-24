"""Songs as rows: the unit the music department learns from (spec 2026-11-23-music-department.md, section 1).

    from_job(conn, job_id, ...)      the song a done music.song job made (idempotent: one row per job)
    for_post(conn, post, body, ...)  production's hook: the post's song row, linked (posts.song_id)
    revise(conn, parent_id, ...)     a revision: a new row naming its parent, the version one higher
    register_batch(conn, batch)      a benchmark batch's songs (no artist)
    store_features / add_features    the measured anatomy, versioned (song_features, song_measure.VERSION)
    measure_song / measure_pending   download the mix and vocal stem, measure them (song_measure), store
    latest_features(conn, song_ids)  {song_id: features} at the current measurement version, else the newest
    feedback_for(conn, role)         the critic's open fixes addressed to a role (the roles branch reads these)

Measurement runs on the control plane after music.song completes: the producer's tick measures songs that have
none at the current version (production.core.advance, policy music.measure.enabled), at most
music.measure.per_tick a tick; `mw music measure` does it on demand (a benchmark batch). A song whose measurement
fails is retried up to music.measure.max_errors times, then left with its error (the song itself is never held up:
measurements are evidence, not a gate).
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from . import music, song_measure
from .policy import Policy

log = logging.getLogger("manyworlds.songs")


def _media(conn, job_id) -> dict[str, dict]:
    rows = conn.execute("SELECT id, uri, meta FROM assets WHERE job_id = %s AND kind = 'audio' ORDER BY uri",
                        (job_id,)).fetchall()
    return {str((r["meta"] or {}).get("role")): {"asset_id": str(r["id"]), "uri": r["uri"], "meta": r["meta"] or {}}
            for r in rows if (r["meta"] or {}).get("role") in music.ROLES}


def from_job(conn, job_id, *, creator_id=None, brief: dict | None = None, genre: str | None = None,
             subgenre: str | None = None, parent_id=None) -> uuid.UUID | None:
    """The song row for a done music.song job (made once; a rerun finds it). None when the job made no mix."""
    row = conn.execute("SELECT id FROM songs WHERE job_id = %s", (job_id,)).fetchone()
    if row:
        return row["id"]
    job = conn.execute("SELECT id, payload, batch, creator_id, status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    if not job or job["status"] != "done":
        return None
    media = _media(conn, job_id)
    if "mix" not in media:
        return None
    p = job["payload"] or {}
    brief = dict(brief or {})
    version = 1
    if parent_id:
        version = int(conn.execute("SELECT version FROM songs WHERE id = %s", (parent_id,)).fetchone()["version"]) + 1
    spec = {k: p.get(k) for k in ("tags", "bpm", "key", "time_signature", "language", "duration_s", "instrumental",
                                  "seed") if k in p}
    spec["engine"] = (p.get("engine") or {}).get("engine")
    return conn.execute(
        """INSERT INTO songs (creator_id, title, version, parent_id, genre, subgenre, brief, spec, lyrics, job_id, batch,
                              mix_asset_id, mix_uri, vocal_uri, instrumental_uri, worker_check)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (job_id) DO UPDATE SET updated_at = now() RETURNING id""",
        (creator_id or job["creator_id"], str(p.get("title") or ""), version, parent_id,
         genre or brief.get("genre"), subgenre or brief.get("subgenre"), Jsonb(brief), Jsonb(spec),
         str(p.get("lyrics") or ""), job_id, job["batch"], media["mix"]["asset_id"], media["mix"]["uri"],
         (media.get("vocals") or {}).get("uri"), (media.get("instrumental") or {}).get("uri"),
         Jsonb(media["mix"]["meta"].get("check") or {}))).fetchone()["id"]


def for_post(conn, post: dict, body: dict, job_id) -> uuid.UUID | None:
    """Production's hook (production.song, when the song is chosen): the post's song row, the post linked to it
    (posts.song_id) and its id kept in body.song_id. The brief is the post's song spec minus the words."""
    spec = body.get("song") or {}
    brief = {k: v for k, v in spec.items() if k not in ("lyrics",)}
    sid = from_job(conn, job_id, creator_id=post.get("creator_id"), brief=brief,
                   genre=spec.get("genre"), subgenre=spec.get("subgenre"))
    if sid is None:
        return None
    conn.execute("UPDATE posts SET song_id = %s WHERE id = %s", (sid, post["id"]))
    body["song_id"] = str(sid)
    return sid


def link_post(conn, post_id, song_id) -> None:
    """A post that uses an existing song (the song under a later post)."""
    conn.execute("UPDATE posts SET song_id = %s, updated_at = now() WHERE id = %s", (song_id, post_id))


def revise(conn, parent_id, job_id, *, brief: dict | None = None) -> uuid.UUID | None:
    """A revision of `parent_id` made by `job_id`: same artist, genre and release lineage, version + 1."""
    parent = conn.execute("SELECT creator_id, genre, subgenre, brief FROM songs WHERE id = %s", (parent_id,)).fetchone()
    if parent is None:
        raise KeyError(f"no song {parent_id}")
    return from_job(conn, job_id, creator_id=parent["creator_id"], brief={**(parent["brief"] or {}), **(brief or {})},
                    genre=parent["genre"], subgenre=parent["subgenre"], parent_id=parent_id)


def register_batch(conn, batch: str, *, kind: str = music.KIND) -> list[uuid.UUID]:
    """Every done song job of a batch as a song row (a benchmark: no artist)."""
    jobs = conn.execute("SELECT id FROM jobs WHERE batch = %s AND kind = %s AND status = 'done' ORDER BY created_at",
                        (batch, kind)).fetchall()
    return [s for s in (from_job(conn, j["id"]) for j in jobs) if s is not None]


def place(conn, song_id, release_id, track_number: int) -> None:
    conn.execute("UPDATE songs SET release_id = %s, track_number = %s, updated_at = now() WHERE id = %s",
                 (release_id, int(track_number), song_id))


def new_release(conn, creator_id, kind: str, title: str) -> uuid.UUID:
    return conn.execute("INSERT INTO releases (creator_id, kind, title) VALUES (%s, %s, %s) RETURNING id",
                        (creator_id, kind, title)).fetchone()["id"]


# --- features ------------------------------------------------------------------------------------------------------

def store_features(conn, song_id, features: dict, *, detail: dict | None = None, version: str = song_measure.VERSION,
                   policy_version: str | None = None) -> uuid.UUID:
    """The song's features at a measurement version (a re-measure at the same version replaces the names it
    measures; features merged in from other sources (add_features) stay)."""
    return conn.execute(
        """INSERT INTO song_features (song_id, version, policy_version, features, detail) VALUES (%s, %s, %s, %s, %s)
           ON CONFLICT (song_id, version) DO UPDATE SET features = song_features.features || EXCLUDED.features,
                  detail = EXCLUDED.detail,
                  policy_version = EXCLUDED.policy_version, created_at = now() RETURNING id""",
        (song_id, version, policy_version, Jsonb(features), Jsonb(detail or {}))).fetchone()["id"]


def add_features(conn, song_id, features: dict, *, version: str = song_measure.VERSION) -> None:
    """Merge more features into the current record (scores from another source: aesthetics, voice similarity,
    the critic); a song not yet measured gets a record holding only these."""
    conn.execute(
        """INSERT INTO song_features (song_id, version, features) VALUES (%s, %s, %s)
           ON CONFLICT (song_id, version) DO UPDATE SET features = song_features.features || EXCLUDED.features""",
        (song_id, version, Jsonb(features)))


def latest_features(conn, song_ids, *, version: str = song_measure.VERSION) -> dict[str, dict]:
    """{song_id: features}: the record at `version` when there is one, else the newest."""
    ids = [str(s) for s in song_ids]
    if not ids:
        return {}
    rows = conn.execute(
        """SELECT DISTINCT ON (song_id) song_id, features FROM song_features WHERE song_id = ANY(%s::uuid[])
           ORDER BY song_id, (version = %s) DESC, created_at DESC""", (ids, version)).fetchall()
    return {str(r["song_id"]): dict(r["features"] or {}) for r in rows}


def get(conn, song_id) -> dict | None:
    return conn.execute("SELECT * FROM songs WHERE id = %s", (song_id,)).fetchone()


def _cfg(policy: Policy) -> dict:
    return song_measure.cfg(policy)


MEASURE_KIND = "finish.measure_song"   # the finisher's measurement job (worker/handlers/finish_video.py)


def measure_mode(policy: Policy) -> str:
    """'queue' (a finish.measure_song job on the finisher: 4 GB, dedicated CPUs) or 'inline' (here, in the calling
    process: tests and big machines). 2026-11-24: a 2-minute 48 kHz stereo song was OOM-killed on the producer
    (512 MB) measuring inline."""
    return str(_cfg(policy).get("mode") or "queue")


def measure_payload(policy: Policy, song: dict) -> dict:
    """Everything the measurement needs, JSON-ready (the finish.measure_song payload): no database on the finisher."""
    return {"song_id": str(song["id"]), "mix_uri": song["mix_uri"], "vocal_uri": song["vocal_uri"],
            "lyrics": song["lyrics"], "title": song["title"], "policy": {"music": policy.body.get("music") or {}},
            "policy_version": policy.version}


def measure_files(payload: dict, policy: Policy, store, workdir: Path | None = None) -> dict:
    """Download the mix and vocal stem, measure them (song_measure.measure), delete the files. Returns the measure."""
    if workdir:
        Path(workdir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mw-song-", dir=str(workdir) if workdir else None) as tmp:
        mix = store.get_file(payload["mix_uri"], Path(tmp) / ("mix" + Path(payload["mix_uri"]).suffix))
        vocal = (store.get_file(payload["vocal_uri"], Path(tmp) / ("vocals" + Path(payload["vocal_uri"]).suffix))
                 if payload.get("vocal_uri") else None)
        out = song_measure.measure(str(mix), str(vocal) if vocal else None, payload.get("lyrics") or "",
                                   policy=policy, title=payload.get("title") or "")
    out["policy_version"] = payload.get("policy_version")
    return out


def record_measurement(conn, song_id, out: dict, *, policy_version: str | None) -> None:
    """A measurement stored as the song's features (with the worker's lyric line coverage), its error cleared."""
    song = get(conn, song_id)
    feats = dict(out["features"])
    check = (song or {}).get("worker_check") or {}
    if check.get("line_coverage") is not None:
        feats["line_coverage"] = check["line_coverage"]
    store_features(conn, song_id, feats, detail=out.get("detail") or {},
                   version=out.get("version") or song_measure.VERSION, policy_version=policy_version)
    conn.execute("UPDATE songs SET measure_error = NULL, updated_at = now() WHERE id = %s", (song_id,))


def _measure_failed(conn, song_id, error: str) -> None:
    conn.execute("UPDATE songs SET measure_error = %s, measure_errors = measure_errors + 1, updated_at = now() "
                 "WHERE id = %s", (error[:500], song_id))


def measure_song(conn, policy: Policy, store, song_id) -> dict:
    """Measure one song here (inline) and store its features at the current version. Returns the measure."""
    song = get(conn, song_id)
    if song is None or not song["mix_uri"]:
        raise KeyError(f"no song {song_id} with a mix")
    out = measure_files(measure_payload(policy, song), policy, store)
    record_measurement(conn, song_id, out, policy_version=policy.version)
    return out


def pending(conn, policy: Policy, *, limit: int | None = None) -> list:
    c = _cfg(policy)
    return [r["id"] for r in conn.execute(
        """SELECT s.id FROM songs s WHERE s.mix_uri IS NOT NULL AND s.measure_errors < %s
             AND NOT EXISTS (SELECT 1 FROM song_features f WHERE f.song_id = s.id AND f.version = %s
                               AND f.features ? 'duration_s')
           ORDER BY s.created_at LIMIT %s""",
        (int(c.get("max_errors", 3)), song_measure.VERSION,
         int(limit if limit is not None else c.get("per_tick", 2)))).fetchall()]


def _measure_key(song_id, errors: int) -> str:
    return f"song:{song_id}:measure:{song_measure.VERSION}:{errors}"


def enqueue_measure(conn, policy: Policy, song_id):
    """The song's finish.measure_song job: one per song, measurement version and failed attempt (idempotent), in
    the song's own batch (a benchmark's songs are measured while operations are paused, like the benchmark)."""
    from . import queue
    song = get(conn, song_id)
    return queue.enqueue(conn, MEASURE_KIND, measure_payload(policy, song), creator_id=song["creator_id"],
                         batch=song["batch"] or f"song-{song_id}",
                         idempotency_key=_measure_key(song_id, song["measure_errors"]))


def collect_measurements(conn) -> dict[str, int]:
    """Finished finish.measure_song jobs into the songs (each once): a done job's measure stored, a failed or dead
    job counted against its song (measure_errors; music.measure.max_errors of them and the song is left)."""
    out = {"measured": 0, "failed": 0}
    for j in conn.execute(
            "SELECT id, status, result, error, payload->>'song_id' AS song_id FROM jobs WHERE kind = %s "
            "AND status IN ('done', 'failed', 'dead') AND NOT coalesce((result->>'recorded')::boolean, false)",
            (MEASURE_KIND,)).fetchall():
        res = j["result"] or {}
        if j["status"] == "done" and res.get("song_features"):
            m = res["song_features"]
            record_measurement(conn, j["song_id"], m, policy_version=m.get("policy_version"))
            out["measured"] += 1
        else:
            _measure_failed(conn, j["song_id"], f"{MEASURE_KIND} {j['status']}: {j['error'] or 'no result'}")
            out["failed"] += 1
        conn.execute("UPDATE jobs SET result = coalesce(result, '{}'::jsonb) || jsonb_build_object('recorded', true) "
                     "WHERE id = %s", (j["id"],))
    return out


def measure_pending(conn, policy: Policy, store=None, *, limit: int | None = None,
                    inline: bool | None = None) -> dict[str, Any]:
    """Songs without features at the current version, measured: queued as finish.measure_song jobs on the finisher
    (measure_mode queue; finished ones are collected here, every tick) or measured here (inline). One song's failure
    is recorded on it and never stops the others."""
    if not _cfg(policy).get("enabled", False):
        return {"measured": 0}
    inline = measure_mode(policy) == "inline" if inline is None else inline
    if not inline:
        out = {**collect_measurements(conn), "queued": 0}
        n = limit if limit is not None else int(_cfg(policy).get("queue_per_tick", 20))
        for sid in pending(conn, policy, limit=n):
            waiting = conn.execute("SELECT 1 FROM jobs WHERE kind = %s AND payload->>'song_id' = %s "
                                   "AND status IN ('queued', 'running')", (MEASURE_KIND, str(sid))).fetchone()
            if not waiting:
                enqueue_measure(conn, policy, sid)
                out["queued"] += 1
        return out if any(out.values()) else {"measured": 0}
    ids = pending(conn, policy, limit=limit)
    if not ids:
        return {"measured": 0}
    if store is None:
        from . import storage
        store = storage.from_url()
    out = {"measured": 0, "failed": 0}
    for sid in ids:
        try:
            measure_song(conn, policy, store, sid)
            out["measured"] += 1
        except Exception as e:  # noqa: BLE001 - a song that can't be measured stays unmeasured, with its reason
            log.exception("measuring song %s failed", sid)
            _measure_failed(conn, sid, f"{type(e).__name__}: {e}")
            out["failed"] += 1
    return out


# --- the critic's fixes ----------------------------------------------------------------------------------------------

def feedback_for(conn, role: str, *, song_id=None, status: str = "open") -> list[dict]:
    """The critic's fixes addressed to `role` (songwriter, lyricist, melody, producer, vocal_director, mix_master),
    oldest first, optionally for one song: what the roles branch reads before it changes a song."""
    rows = conn.execute(
        "SELECT id, song_id, problem, change, evidence, created_at FROM song_feedback WHERE role = %s AND status = %s "
        "AND (%s::uuid IS NULL OR song_id = %s) ORDER BY created_at", (role, status, song_id, song_id)).fetchall()
    return [{**r, "id": str(r["id"]), "song_id": str(r["song_id"])} for r in rows]


def close_feedback(conn, feedback_id, *, status: str = "done") -> None:
    conn.execute("UPDATE song_feedback SET status = %s WHERE id = %s", (status, feedback_id))
