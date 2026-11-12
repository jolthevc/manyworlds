"""Day-1 benchmark: measure real seconds per output on each GPU pool and project
cost per 1,000 outputs. These measurements replace the research estimates in the
proposal, and feed the router's cost model."""

from __future__ import annotations

import statistics
import uuid
from typing import Any

from . import ledger, queue
from .policy import Policy

# Varied, everyday creator-feed prompts so the benchmark isn't tuned to one easy scene.
PHOTO_PROMPTS = [
    "candid phone photo of a young woman laughing at a cafe table, window light, slightly blurry background",
    "mirror selfie in a small apartment hallway, oversized hoodie, phone covering part of face, warm lamp light",
    "outfit photo on a city sidewalk, full body, overcast daylight, shot on iPhone, slightly tilted framing",
    "close-up selfie in a car, golden hour light through the window, natural skin texture",
    "photo of a man cooking pasta in a home kitchen, steam, messy counter, flash photography",
    "beach sunset photo from behind, wind in hair, imperfect horizon, phone camera grain",
    "gym mirror selfie, fluorescent lighting, water bottle in hand, casual pose",
    "dinner table photo with friends' hands in frame, candlelight, shallow depth of field",
]
SPEECH_LINES = [
    "Okay so I finally tried that new coffee place everyone keeps talking about, and honestly? Worth it.",
    "Quick update: the apartment is still a mess, but the plants are thriving, so that's a win.",
    "If you asked me five years ago where I'd be now, I would not have guessed any of this.",
]


def enqueue(conn, policy: Policy, kind: str, jobs: int, *, batch: str | None = None,
            extra: dict[str, Any] | None = None, pools: list[str] | None = None) -> str:
    batch = batch or f"bench-{kind}-{uuid.uuid4().hex[:6]}"
    gen = policy.get("generation", default={})
    for i in range(jobs):
        if kind.endswith("image") or kind.startswith("image."):
            img = gen.get("image", {})
            payload = {"prompt": PHOTO_PROMPTS[i % len(PHOTO_PROMPTS)], "n": policy.get(
                "job_kinds", kind, "outputs_per_job", default=4), "width": img.get("width", 896),
                "height": img.get("height", 1152), "seed": 1000 + i,
                "params": {"steps": img.get("steps", 9), "guidance": img.get("guidance", 0.0)}}
        elif kind.endswith("video") or kind.startswith("video."):
            vid = gen.get("video", {})
            payload = {"prompt": "she turns toward the camera and smiles, handheld phone footage",
                       "width": vid.get("width", 480), "height": vid.get("height", 832),
                       "frames": vid.get("frames", 81), "seed": 2000 + i,
                       "params": {"steps": vid.get("steps", 4), "guidance": vid.get("guidance", 1.0),
                                  "fps": vid.get("fps", 16)}}
        else:
            payload = {"text": SPEECH_LINES[i % len(SPEECH_LINES)], "voice": "af_heart"}
        payload.update(extra or {})
        queue.enqueue(conn, kind, payload, batch=batch, pools=pools, max_attempts=int(policy.get("queue", "max_attempts", default=3)))
    return batch


def report(conn, policy: Policy, batch: str) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT j.kind, a.pool, a.gpu_type, a.gpu_seconds,
               (SELECT count(*) FROM assets s WHERE s.job_id = j.id) AS outputs
        FROM attempts a JOIN jobs j ON j.id = a.job_id
        WHERE j.batch = %s AND a.outcome = 'done'
        """,
        (batch,),
    ).fetchall()
    out: dict[str, Any] = {"batch": batch, "jobs": queue.counts(conn, batch=batch),
                           "spend": ledger.totals(conn, batch=batch), "measured": []}
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["kind"], r["pool"], r["gpu_type"]), []).append(r)
    for (kind, pool, gpu), rs in sorted(groups.items()):
        per_output = [float(r["gpu_seconds"]) / max(r["outputs"], 1) for r in rs]
        med = statistics.median(per_output)
        p90 = sorted(per_output)[max(0, int(len(per_output) * 0.9) - 1)]
        # Project to every pool with the same GPU type (prices differ, speed shouldn't).
        projections = {
            name: round(med * float(p["usd_per_hour"]) / 3.6, 4)
            for name, p in policy.get("gpu_pools", default={}).items()
            if p.get("gpu") == gpu
        }
        out["measured"].append({
            "kind": kind, "pool": pool, "gpu": gpu, "jobs": len(rs),
            "outputs": sum(r["outputs"] for r in rs),
            "sec_per_output_median": round(med, 3), "sec_per_output_p90": round(p90, 3),
            "usd_per_1k_outputs_by_pool": projections,
        })
    return out
