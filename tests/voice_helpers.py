"""Stand-ins for what the voice worker measures (worker/handlers/voice_judge.py), for control-plane tests."""

from __future__ import annotations

from psycopg.types.json import Jsonb


def line(**over) -> dict:
    """One casting line as the worker measures it: clear, natural, moving, on pace, her voice."""
    base = {"name": "neutral", "text": "Okay, so here's the plan for today.", "wer": 0.0, "words": 7, "seconds": 1.8,
            "wps": 3.9, "mos": 4.0, "f0_std_st": 2.5, "f0_range_st": 7.0, "f0_median_st": 24.0, "f0_median_hz": 220.0,
            "energy_std_db": 6.0, "ref_distance": 0.2}
    return {**base, **over}


def lines(n: int = 5, **over) -> list[dict]:
    """`n` lines whose median pitch moves between deliveries (neutral, warm, excited, quiet...)."""
    return [line(name=f"l{i}", f0_median_st=22.0 + 1.5 * i, wps=3.7 + 0.15 * (i % 3), **over) for i in range(n)]


def cast_meta(index: int, *, gender: str = "female", p: float = 0.95, age: float = 27.0, embedding=None,
              engine: str = "indextts", **line_over) -> dict:
    return {"index": index, "seed": 100 + index, "role": "cast_reference", "engine": engine,
            "designer": "qwen3_design", "ref_text": "Okay, so this morning did not go to plan.", "ref_wer": 0.0,
            "transcript": "okay so this morning did not go to plan", "embedding": embedding or [1.0, 0.0, 0.0],
            "who": {"gender_p": {gender: p, ("male" if gender == "female" else "female"): round(1 - p, 4)},
                    "age": age}, "lines": lines(**line_over), "seconds": 14.0, "sample_rate": 24000}


def finished_cast(conn, cid, metas: list[dict], *, engine: str = "indextts", purpose: str = "creator",
                  batch: str | None = None, status: str = "done", job_id=None):
    """A voice.cast job (new, or `job_id`) finished with one audio asset per candidate meta."""
    if job_id is None:
        job_id = conn.execute(
            "INSERT INTO jobs (kind, status, creator_id, batch, payload) VALUES ('voice.cast', %s, %s, %s, %s) "
            "RETURNING id", (status, cid, batch, Jsonb({"engine": engine, "purpose": purpose,
                                                        "description": "a warm woman", "anchor_text": "Okay."}))
        ).fetchone()["id"]
    else:
        conn.execute("UPDATE jobs SET status = %s WHERE id = %s", (status, job_id))
    for m in metas:
        conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                     "VALUES (%s, %s, 'audio', %s, 'x', 1, %s)",
                     (job_id, cid, f"s3://b/cast/{job_id}/{m['index']:02d}.wav", Jsonb(m)))
    return job_id
