"""Review gates for generated images: is it the creator, is it believable, is it safe.

One vision call per job reviews all of that job's images against the creator's reference
images (cached across calls, so each extra image costs only its own tokens). Every
asset gets one `reviews` row per gate:

  identity  same person as the references (face, hair, signature details such as glasses)
  realism   reads as an ordinary phone photo; no anatomical, text or rendering defects
  safety    looks at least policy creator_min_age, not sexualized, no real brand marks

An asset is accepted only if every gate passes. Thresholds, the model role and limits
are in policy `review`. Cost per accepted output (manyworlds.deploy.scoring) counts
only accepted assets, so pools and settings that make rejects are ranked down.
"""

from __future__ import annotations

from pathlib import Path

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from ..creator.lab import appearance_prompt
from ..llm import StructuredLLM, image_block
from ..policy import Policy
from ..storage import ObjectStore

SYSTEM = """You are the quality gate for an AI creator studio whose creators are openly labeled as AI.
You compare generated photos against reference photos of one fictional person and judge each candidate
strictly and independently. Score what you see, not what the prompt intended."""

GATES = ("identity", "realism", "safety")


class CandidateReview(BaseModel):
    index: int = Field(description="0-based candidate index")
    identity: int = Field(description="10 = unmistakably the same person as the REFERENCE photos: "
                          "face shape and features, apparent age, skin, hair, and signature details; "
                          "0 = a different person. A missing signature detail (e.g. glasses) costs at least 3")
    realism: int = Field(description="10 = indistinguishable from an ordinary smartphone photo; "
                         "deduct for warped hands or anatomy, garbled text, plastic skin, rendering artifacts")
    defects: list[str] = Field(description="concrete visible problems, empty if none")
    looks_under_min_age: bool = Field(description="the person could plausibly be read as younger than the stated minimum age")
    sexualized: bool
    real_brand_marks: bool = Field(description="a legible real-world brand logo or trademark is visible")
    notes: str = Field(description="one sentence")


class ImageReviews(BaseModel):
    reviews: list[CandidateReview]


def _reference_images(creator: dict, store: ObjectStore, workdir: Path, n: int) -> list[bytes]:
    refs = (creator["anchors"] or {}).get("reference_images", [])[:n]
    return [store.get_file(r["uri"], workdir / f"ref_{i}.png").read_bytes() for i, r in enumerate(refs)]


def decide(r: CandidateReview, cfg: dict) -> dict[str, bool]:
    return {"identity": r.identity >= int(cfg.get("min_identity", 7)),
            "realism": r.realism >= int(cfg.get("min_realism", 6)),
            "safety": not (r.looks_under_min_age or r.sexualized or r.real_brand_marks)}


def _judge(llm: StructuredLLM, policy: Policy, cfg: dict, role: str, effort: str, creator, refs: list[bytes],
           candidates: list[bytes]) -> tuple[str, list[CandidateReview | None]]:
    """One vision call: the creator's references (cached prefix), then the candidates."""
    min_age = int(policy.get("scope", "creator_min_age", default=21))
    content: list = [{"type": "text", "text": f"The person: {appearance_prompt(creator['bible'])}. "
                                              f"Minimum age for this creator: {min_age}."}]
    for i, b in enumerate(refs):
        content += [{"type": "text", "text": f"REFERENCE {i}:"}, image_block(b)]
    content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
    for i, b in enumerate(candidates):
        content += [{"type": "text", "text": f"Candidate {i}:"}, image_block(b)]
    content.append({"type": "text", "text": f"Review all {len(candidates)} candidates, one entry each."})
    out = llm.structured(role, system=SYSTEM, content=content, schema=ImageReviews, effort=effort,
                         creator_id=creator["id"])
    by_index = {r.index: r for r in out.reviews}
    reviews = []
    for i in range(len(candidates)):
        r = by_index.get(i)
        if r is not None:
            r.identity, r.realism = max(0, min(10, r.identity)), max(0, min(10, r.realism))
        reviews.append(r)
    return llm.model_for(role), reviews


def review_job(conn, policy: Policy, llm: StructuredLLM, store: ObjectStore, job_id, workdir: Path,
               *, role: str | None = None, write: bool = True) -> list[dict]:
    """Review every image asset of one finished job. Returns one dict per asset.

    Two stages: a cheap screen (policy review.image.role) on every image, then a stronger
    confirm (confirm_role) only on the images the screen passed. Rejects stay cheap;
    anything accepted has passed both. Each stage writes its own rows per gate, so an
    asset is accepted only when no row failed."""
    cfg = policy.get("review", "image", default={})
    stages = [(role or cfg.get("role", "judge"), cfg.get("effort", "low"), "screen")]
    if cfg.get("confirm_role") and not role:
        stages.append((cfg["confirm_role"], cfg.get("confirm_effort", "low"), "confirm"))
    job = conn.execute("SELECT creator_id FROM jobs WHERE id = %s", (job_id,)).fetchone()
    creator = conn.execute("SELECT id, bible, anchors FROM creators WHERE id = %s", (job["creator_id"],)).fetchone()
    assets = conn.execute("SELECT id, uri FROM assets WHERE job_id = %s AND kind = 'image' ORDER BY uri",
                          (job_id,)).fetchall()
    if not assets:
        return []
    workdir = Path(workdir)
    refs = _reference_images(creator, store, workdir, int(cfg.get("reference_images", 3)))
    if not refs:
        raise ValueError(f"creator {creator['id']} has no reference images to review against")
    images = {a["id"]: store.get_file(a["uri"], workdir / f"cand_{i}.png").read_bytes() for i, a in enumerate(assets)}
    results = {a["id"]: {"asset_id": str(a["id"]), "uri": a["uri"], "accepted": True, "stages": [],
                         **{g: True for g in GATES}, "identity_score": None, "realism_score": None, "defects": []}
               for a in assets}

    remaining = list(assets)
    for stage_role, effort, stage in stages:
        if not remaining:
            break
        model, reviews = _judge(llm, policy, cfg, stage_role, effort, creator, refs, [images[a["id"]] for a in remaining])
        still = []
        for a, r in zip(remaining, reviews):
            if r is None:  # the judge skipped it: fail closed, it can be re-reviewed
                verdict, details, scores = {g: False for g in GATES}, {"error": "no review returned"}, {}
            else:
                verdict, details = decide(r, cfg), r.model_dump()
                scores = {"identity": r.identity, "realism": r.realism}
            details.update(model=model, stage=stage)
            if write:
                for g in GATES:
                    conn.execute("INSERT INTO reviews (asset_id, gate, passed, score, details) VALUES (%s, %s, %s, %s, %s)",
                                 (a["id"], g, verdict[g], scores.get(g), Jsonb(details)))
            res = results[a["id"]]
            res["stages"].append({"stage": stage, "model": model, **scores, "passed": all(verdict.values())})
            res.update({g: res[g] and verdict[g] for g in GATES}, identity_score=scores.get("identity"),
                       realism_score=scores.get("realism"), defects=details.get("defects", []))
            res["accepted"] = all(res[g] for g in GATES)
            if res["accepted"]:
                still.append(a)
        remaining = still
    return [results[a["id"]] for a in assets]


def pending_jobs(conn, policy: Policy, *, batch: str | None = None, limit: int | None = None) -> list:
    """Finished image jobs for a creator with reference images, whose images have no review yet.
    Reference-pack jobs are skipped: they are what the references are chosen from."""
    cfg = policy.get("review", "image", default={})
    limit = limit or int(cfg.get("max_jobs_per_run", 50))
    return [r["id"] for r in conn.execute(
        """SELECT j.id FROM jobs j JOIN creators c ON c.id = j.creator_id
           WHERE j.status = 'done' AND j.kind LIKE 'image.%%'
             AND coalesce(j.batch, '') NOT LIKE 'refpack-%%'
             AND jsonb_array_length(coalesce(c.anchors->'reference_images', '[]'::jsonb)) > 0
             AND (%(batch)s::text IS NULL OR j.batch = %(batch)s)
             AND EXISTS (SELECT 1 FROM assets a WHERE a.job_id = j.id AND a.kind = 'image'
                           AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id = a.id))
           ORDER BY j.updated_at LIMIT %(limit)s""", {"batch": batch, "limit": limit}).fetchall()]


def run_pending(conn, policy: Policy, llm: StructuredLLM, store: ObjectStore, workdir: Path,
                *, batch: str | None = None) -> dict:
    accepted = rejected = 0
    errors = []
    for job_id in pending_jobs(conn, policy, batch=batch):
        jd = Path(workdir) / str(job_id)
        jd.mkdir(parents=True, exist_ok=True)
        try:
            for r in review_job(conn, policy, llm, store, job_id, jd):
                accepted += r["accepted"]
                rejected += not r["accepted"]
        except Exception as e:  # noqa: BLE001 - one bad job must not stop the others
            errors.append({"job": str(job_id), "error": str(e)[:300]})
    return {"accepted": accepted, "rejected": rejected, "errors": errors}
