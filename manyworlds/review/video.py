"""Review gate for story clips: does the creator stay herself, does the motion hold up.

The video worker stores evenly spaced keyframes beside each clip (video.wan_i2v). The judge
sees the creator's reference photos, the approved start photo the clip was made from, and
the keyframes in order, and scores the clip as a whole. Every clip gets one `reviews` row
per gate, written against the video asset:

  identity  the face in the WORST keyframe still matches the references (image-to-video
            models drift: on 2026-11-14 a 480p Wan clip slid into a different face by 4 s)
  motion    the body and objects stay physically whole between frames: no morphing hands,
            merging limbs, things appearing or vanishing
  realism   each frame reads as phone video, not a render
  safety    as for images: adult-looking, not sexualized, no real brand marks
  scene     (when the planner wrote one) the shot shows the scene and the movement intended

Keyframes can't show smoothness or timing; they catch the failures seen so far (face drift,
warping), which is what a viewer notices first. Like images there are two stages: a cheap
screen, then a stronger confirm only for clips the screen passed. Thresholds and roles are in
policy `review.video`.
"""

from __future__ import annotations

from pathlib import Path

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from ..creator.lab import appearance_prompt
from ..llm import StructuredLLM, image_block
from ..policy import Policy
from ..storage import ObjectStore
from . import common

SYSTEM = """You are the quality gate for an AI creator studio whose creators are openly labeled as AI.
You review short image-to-video clips of one fictional person from keyframes, strictly. A clip is only
as good as its worst frame. Score what you see, not what the prompt intended."""

GATES = ("identity", "motion", "realism", "safety")


class ClipReview(BaseModel):
    identity: int = Field(description="10 = in EVERY keyframe unmistakably the same person as the REFERENCE photos "
                          "(face shape and features, age, skin, hair, signature details); score the worst keyframe. "
                          "0 = a different person somewhere in the clip")
    drift_from: int | None = Field(default=None, description="the first keyframe number where the face stops matching "
                                   "the start photo, or null if it never does")
    motion: int = Field(description="10 = body, hands and objects stay whole and plausible from frame to frame; deduct "
                        "for morphing or extra fingers, merging limbs, objects appearing, vanishing or melting")
    realism: int = Field(description="10 = every keyframe reads as ordinary phone video; deduct for plastic skin, "
                         "smearing, rendering artifacts, garbled text")
    defects: list[str] = Field(description="concrete visible problems with the keyframe number, empty if none")
    looks_under_min_age: bool
    sexualized: bool
    real_brand_marks: bool = Field(description="a legible real-world brand logo or trademark is visible")
    scene: int | None = Field(default=None, description="only when an INTENDED SHOT is given: 10 = shows that "
                              "setting and that movement; 5 = right setting, movement wrong or absent; 0 = different")
    notes: str = Field(description="one sentence")


def gates_for(scene: str | None) -> tuple[str, ...]:
    return GATES + ("scene",) if scene else GATES


def decide(r: ClipReview, cfg: dict, scene: str | None = None) -> dict[str, bool]:
    verdict = {"identity": r.identity >= int(cfg.get("min_identity", 7)),
               "motion": r.motion >= int(cfg.get("min_motion", 6)),
               "realism": r.realism >= int(cfg.get("min_realism", 6)),
               "safety": not (r.looks_under_min_age or r.sexualized or r.real_brand_marks)}
    if scene:
        verdict["scene"] = r.scene is None or r.scene >= int(cfg.get("min_scene", 0))
    return verdict


def _media_type(uri: str) -> str:
    return "image/jpeg" if uri.lower().endswith((".jpg", ".jpeg")) else "image/png"


def _judge(llm: StructuredLLM, policy: Policy, role: str, effort: str, creator, refs: list[bytes],
           start: bytes | None, frames: list[tuple[bytes, str]], scene: str | None) -> ClipReview:
    min_age = int(policy.get("scope", "creator_min_age", default=21))
    content: list = [{"type": "text", "text": f"The person: {appearance_prompt(creator['bible'])}. "
                                              f"Minimum age for this creator: {min_age}."}]
    for i, b in enumerate(refs):
        content += [{"type": "text", "text": f"REFERENCE {i}:"}, image_block(b)]
    content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
    if start:
        content += [{"type": "text", "text": "START PHOTO (approved; the clip begins from it):"}, image_block(start)]
    for k, (b, media) in enumerate(frames):
        content += [{"type": "text", "text": f"KEYFRAME {k + 1} of {len(frames)}:"}, image_block(b, media)]
    if scene:
        content.append({"type": "text", "text": f"INTENDED SHOT (score `scene` against it): {scene}"})
    content.append({"type": "text", "text": "Review the clip as a whole from these keyframes."})
    r = llm.structured(role, system=SYSTEM, content=content, schema=ClipReview, effort=effort,
                       creator_id=creator["id"])
    for k in ("identity", "motion", "realism"):
        setattr(r, k, max(0, min(10, getattr(r, k))))
    r.scene = max(0, min(10, r.scene)) if (scene and r.scene is not None) else None
    return r


def review_job(conn, policy: Policy, llm: StructuredLLM, store: ObjectStore, job_id, workdir: Path,
               *, write: bool = True) -> list[dict]:
    """Review the clip of one finished video job. Returns one dict per video asset (normally one)."""
    cfg = policy.get("review", "video", default={})
    stages = common.stages(cfg)
    job = conn.execute("SELECT creator_id, payload FROM jobs WHERE id = %s", (job_id,)).fetchone()
    payload = job["payload"] or {}
    scene = payload.get("scene")
    gates = gates_for(scene)
    creator = conn.execute("SELECT id, bible, anchors FROM creators WHERE id = %s", (job["creator_id"],)).fetchone()
    videos = conn.execute("SELECT id, uri FROM assets WHERE job_id = %s AND kind = 'video' ORDER BY uri",
                          (job_id,)).fetchall()
    keys = conn.execute("SELECT uri FROM assets WHERE job_id = %s AND kind = 'keyframe' ORDER BY uri",
                        (job_id,)).fetchall()
    if not videos or not keys:
        return []
    workdir = Path(workdir)
    refs = common.reference_images(creator, store, workdir, int(cfg.get("reference_images", 3)))
    start = (store.get_file(payload["image_uri"], workdir / "start.png").read_bytes()
             if payload.get("image_uri") else None)
    frames = [(store.get_file(k["uri"], workdir / f"key_{i}{Path(k['uri']).suffix}").read_bytes(), _media_type(k["uri"]))
              for i, k in enumerate(keys)]

    rows, out = [], []
    for v in videos:
        res = {"asset_id": str(v["id"]), "uri": v["uri"], "accepted": True, "stages": [], **{g: True for g in gates}}
        for role, effort, stage in stages:
            with common.charged_to(llm, job_id):   # review counts in this clip's cost
                r = _judge(llm, policy, role, effort, creator, refs, start, frames, scene)
            verdict = decide(r, cfg, scene)
            scores = {"identity": r.identity, "motion": r.motion, "realism": r.realism, "scene": r.scene}
            details = {**r.model_dump(), "model": llm.model_for(role), "stage": stage}
            rows += [(v["id"], g, verdict[g], scores.get(g), Jsonb(details)) for g in gates]
            res["stages"].append({"stage": stage, "model": details["model"], **scores, "drift_from": r.drift_from,
                                  "defects": r.defects, "passed": all(verdict.values())})
            res.update({g: res[g] and verdict[g] for g in gates})
            res["accepted"] = all(res[g] for g in gates)
            if not res["accepted"]:
                break   # rejects stay cheap: no confirm for a clip the screen failed
        res["confirmed"] = res["accepted"] and len(res["stages"]) == len(stages)
        out.append(res)
    if write:
        common.write_rows(conn, rows)
    return out


def pending_jobs(conn, policy: Policy, *, batch: str | None = None, limit: int | None = None) -> list:
    """Finished clip jobs (with keyframes) for a creator with reference images, not yet reviewed."""
    limit = limit or int(policy.get("review", "video", "max_jobs_per_run", default=10))
    return [r["id"] for r in conn.execute(
        """SELECT j.id FROM jobs j JOIN creators c ON c.id = j.creator_id
           WHERE j.status = 'done' AND j.kind LIKE 'video.%%' AND j.kind <> 'video.mirror'
             AND jsonb_array_length(coalesce(c.anchors->'reference_images', '[]'::jsonb)) > 0
             AND (%(batch)s::text IS NULL OR j.batch = %(batch)s)
             AND EXISTS (SELECT 1 FROM assets k WHERE k.job_id = j.id AND k.kind = 'keyframe')
             AND EXISTS (SELECT 1 FROM assets a WHERE a.job_id = j.id AND a.kind = 'video'
                           AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id = a.id))
           ORDER BY j.updated_at LIMIT %(limit)s""", {"batch": batch, "limit": limit}).fetchall()]


def run_pending(conn, policy: Policy, llm: StructuredLLM, store: ObjectStore, workdir: Path,
                *, batch: str | None = None) -> dict:
    return common.run_jobs(pending_jobs(conn, policy, batch=batch),
                           lambda job, d: review_job(conn, policy, llm, store, job, d), workdir, prefix="video-")
