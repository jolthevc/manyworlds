"""Review gates for generated images: is it the creator, is it believable, is it safe.

One vision call per job reviews all of that job's images against the creator's reference
images (cached across calls, so each extra image costs only its own tokens). Every
asset gets one `reviews` row per gate:

  identity  same person as the references (face, hair, signature details such as glasses)
  realism   reads as an ordinary phone photo; no anatomical, text or rendering defects
  safety    looks at least policy creator_min_age, not sexualized, no real brand marks
  scene     (post images only) shows the scene the planner wrote: setting, action, styling
  direction (directed scenes: the job carries a shot sheet, manyworlds.direction) follows the shot and its first
            beat, on rubric policy/rubrics/direction-v<n>.yaml; below review.direction.min_score it is rejected
  hands, teeth, consistency
            finer checks (manyworlds.review.detail, policy review.detail): a row each when the judge scores them;
            they reject only when enforced

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
from . import common, detail

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
    scene: int | None = Field(default=None, description="only when an INTENDED SCENE is given: 10 = shows it "
                              "(setting, action, clothing and props as written); 5 = right setting, details wrong; "
                              "0 = a different scene. null when no scene is given")
    direction: int | None = Field(default=None, description="only when a DIRECTED SHOT is given: 1-10 on the "
                                  "DIRECTION rubric given with it (does the photo follow the shot and its first "
                                  "beat); null when no directed shot is given")
    hands: int | None = Field(default=None, description="only when DETAIL CHECKS ask for it: 1-10 for visible "
                              "hands; null when no hand is visible")
    teeth: int | None = Field(default=None, description="only when DETAIL CHECKS ask for it: 1-10 for visible "
                              "teeth; null when no teeth show")
    consistency: int | None = Field(default=None, description="only when SET / OUTFIT REFERENCE images are given: "
                                    "1-10, the same set and clothes as the references; null otherwise")
    notes: str = Field(description="one sentence")


class ImageReviews(BaseModel):
    reviews: list[CandidateReview]


def gates_for(scene: str | None, shot: dict | None = None) -> tuple[str, ...]:
    return GATES + (("scene",) if scene else ()) + (("direction",) if shot else ())


def decide(r: CandidateReview, cfg: dict, scene: str | None = None, *, shot: dict | None = None,
           policy: Policy | None = None) -> dict[str, bool]:
    verdict = {"identity": r.identity >= int(cfg.get("min_identity", 7)),
               "realism": r.realism >= int(cfg.get("min_realism", 6)),
               "safety": not (r.looks_under_min_age or r.sexualized or r.real_brand_marks)}
    if scene:  # a judge that leaves it out passes: fit is a preference, not a safety line
        verdict["scene"] = r.scene is None or r.scene >= int(cfg.get("min_scene", 0))
    if shot:
        verdict["direction"] = common.direction_passes(r.direction, policy)
    return verdict


def _judge(llm: StructuredLLM, policy: Policy, cfg: dict, role: str, effort: str, creator, refs: list[bytes],
           candidates: list[bytes], scene: str | None = None,
           shot: dict | None = None, extra: tuple[list, list[str]] = ([], [])) -> tuple[str, list[CandidateReview | None]]:
    """One vision call: the creator's references (cached prefix), then the candidates."""
    min_age = int(policy.get("scope", "creator_min_age", default=21))
    content: list = [{"type": "text", "text": f"The person: {appearance_prompt(creator['bible'])}. "
                                              f"Minimum age for this creator: {min_age}."}]
    for i, b in enumerate(refs):
        content += [{"type": "text", "text": f"REFERENCE {i}:"}, image_block(b)]
    content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
    for i, b in enumerate(candidates):
        content += [{"type": "text", "text": f"Candidate {i}:"}, image_block(b)]
    if scene:
        content.append({"type": "text", "text": f"INTENDED SCENE (score `scene` against it): {scene}"})
    if shot:
        content.append({"type": "text", "text": "DIRECTED SHOT (score `direction` 1-10 against it, per candidate): "
                                                + common.direction_brief(shot, policy, still=True)})
    ref_content, ref_labels = extra   # set / outfit references (review.detail consistency)
    content += ref_content
    ask = detail.judge_prompt(policy, consistency=ref_labels, still=True)
    if ask:
        content.append({"type": "text", "text": ask})
    content.append({"type": "text", "text": f"Review all {len(candidates)} candidates, one entry each."})
    out = llm.structured(role, system=SYSTEM, content=content, schema=ImageReviews, effort=effort,
                         creator_id=creator["id"])
    by_index = {r.index: r for r in out.reviews}
    reviews = []
    for i in range(len(candidates)):
        r = by_index.get(i)
        if r is not None:
            r.identity, r.realism = max(0, min(10, r.identity)), max(0, min(10, r.realism))
            r.scene = max(0, min(10, r.scene)) if (scene and r.scene is not None) else None
            r.direction = max(1, min(10, r.direction)) if (shot and r.direction is not None) else None
            detail.clamp_scores(r, policy, consistency=bool(ref_labels))
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
    stages = common.stages(cfg, role)
    job = conn.execute("SELECT creator_id, payload->>'scene' AS scene, payload->'shot' AS shot, "
                       "payload->'references' AS refs FROM jobs WHERE id = %s", (job_id,)).fetchone()
    shot = job["shot"] if isinstance(job["shot"], dict) else None   # a directed scene (manyworlds.direction)
    scene, gates = job["scene"], gates_for(job["scene"], shot)
    creator = conn.execute("SELECT id, bible, anchors FROM creators WHERE id = %s", (job["creator_id"],)).fetchone()
    assets = conn.execute("SELECT id, uri FROM assets WHERE job_id = %s AND kind = 'image' ORDER BY uri",
                          (job_id,)).fetchall()
    if not assets:
        return []
    workdir = Path(workdir)
    refs = common.reference_images(creator, store, workdir, int(cfg.get("reference_images", 3)))
    images = {a["id"]: store.get_file(a["uri"], workdir / f"cand_{i}.png").read_bytes() for i, a in enumerate(assets)}
    # Set and outfit references for a shot that names them (review.detail consistency); none: no consistency row.
    extra = detail.reference_blocks(detail.references_for({"references": job["refs"] or []}, shot, policy),
                                    store, workdir, image_block)
    results = {a["id"]: {"asset_id": str(a["id"]), "uri": a["uri"], "accepted": True, "stages": [],
                         **{g: True for g in gates}, "identity_score": None, "realism_score": None, "scene_score": None,
                         "defects": []}
               for a in assets}

    remaining, rows = list(assets), []
    for stage_role, effort, stage in stages:
        if not remaining:
            break
        with common.charged_to(llm, job_id):   # review counts in this output's cost
            model, reviews = _judge(llm, policy, cfg, stage_role, effort, creator, refs,
                                    [images[a["id"]] for a in remaining], scene, shot, extra)
        still = []
        for a, r in zip(remaining, reviews):
            checks = {}
            if r is None:  # the judge skipped it: fail closed, it can be re-reviewed
                verdict, details, scores = {g: False for g in gates}, {"error": "no review returned"}, {}
            else:
                verdict = decide(r, cfg, scene, shot=shot, policy=policy)
                details = r.model_dump(exclude=detail.unasked(r, shot))
                scores = {"identity": r.identity, "realism": r.realism, "scene": r.scene}
                if shot:
                    scores["direction"] = r.direction
                # finer checks: a row each when scored; they reject only when enforced (review.detail)
                checks = detail.judged_verdicts(policy, r, consistency=bool(extra[1]))
            details.update(model=model, stage=stage)
            rows += [(a["id"], g, verdict[g], scores.get(g), Jsonb(details)) for g in gates]
            rows += [(a["id"], g, ok, getattr(r, g), Jsonb({**details, **info})) for g, (ok, info) in checks.items()]
            res = results[a["id"]]
            res["stages"].append({"stage": stage, "model": model, **scores,
                                  **{g: getattr(r, g) for g in checks},
                                  "passed": all(verdict.values()) and all(ok for ok, _ in checks.values())})
            res.update({g: res[g] and verdict[g] for g in gates}, identity_score=scores.get("identity"),
                       realism_score=scores.get("realism"), scene_score=scores.get("scene"), defects=details.get("defects", []))
            res.update({g: res.get(g, True) and ok for g, (ok, _) in checks.items()})
            res["accepted"] = all(res[g] for g in gates) and all(res.get(g, True) for g in detail.JUDGED)
            if res["accepted"]:
                still.append(a)
        # Only the best screen passes are worth a confirm: a post uses its top candidates.
        # The rest stay accepted on the screen alone ("confirmed": False).
        top = int(cfg.get("confirm_top_per_job", 0)) or len(still)
        still.sort(key=lambda a: -sum(results[a["id"]][f"{k}_score"] or 0 for k in ("identity", "realism", "scene")))
        remaining = still[:top]
    if write:
        common.write_rows(conn, rows)
    for r in results.values():
        r["confirmed"] = r["accepted"] and len(r["stages"]) == len(stages)
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
    return common.run_jobs(pending_jobs(conn, policy, batch=batch),
                           lambda job, d: review_job(conn, policy, llm, store, job, d), workdir)
