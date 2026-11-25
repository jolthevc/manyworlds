"""Review gates for generated images: is it the creator, is it believable, is it safe.

One vision call per job reviews all of that job's images against the creator's reference
images (cached across calls, so each extra image costs only its own tokens). Every
asset gets one `reviews` row per gate:

  identity  same person as the references (face, hair, signature details such as glasses); not asked of set
            plates and outfit references (payload `plate`, manyworlds.creator.sets), which have nobody in them
  realism   reads as an ordinary phone photo; no anatomical, text or rendering defects
  safety    looks at least policy creator_min_age, no explicit, nude, implied-nude, fetish or soliciting content (judged in
            context, never by clothing alone: common.SAFETY_RULE), no real brand marks
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
strictly and independently. Score what you see, not what the prompt intended.
""" + common.SAFETY_SYSTEM

GATES = ("identity", "realism", "safety")


class CandidateReview(BaseModel):
    index: int = Field(description="0-based candidate index")
    identity: int = Field(description="is this the same PERSON as the REFERENCE photos, judged by the face: "
                          "10 = unmistakably her (face shape and features, apparent age, skin); 0 = a different "
                          "person. Clothes, hats, hairstyle, props, pose, framing and setting never lower it: "
                          "those are the scene, not the person. A visible signature detail that is wrong costs 1-2")
    realism: int = Field(description="10 = indistinguishable from an ordinary smartphone photo; "
                         "deduct for warped hands or anatomy, garbled text, plastic skin, rendering artifacts")
    defects: list[str] = Field(description="concrete visible problems, empty if none")
    looks_under_min_age: bool = Field(description="the person could plausibly be read as younger than the stated minimum age")
    sexualized: bool = Field(description=common.SAFETY_RULE)
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
    # owner 2026-11-25: brand alignment and distribution in the same call (common.BRAND_RULE, DISTRIBUTION_RULE), and
    # styling/detail defects with a severity (common.DETAIL_RULE): only major ones reject (as realism)
    brand_alignment: int | None = Field(default=None, description=common.BRAND_RULE)
    distribution_risk: int | None = Field(default=None, description=common.DISTRIBUTION_RULE)
    distribution_label: str | None = Field(default=None, description="only with distribution_risk: a few words, "
                                           "e.g. 'mainstream', 'swimwear: fine for followers, boost with care'")
    detail_defects: list[str] = Field(default_factory=list, description=common.DETAIL_RULE)
    # 2026-11-25 (maren-final-3): a photo with her hands on a phone AND two more hands holding a note passed, the judge
    # wrote "a phone is in her other hand"; two cutaways of hands and a counter showed her face. Counted, not inferred.
    extra_body_parts: bool = Field(default=False, description="count the hands and arms you can see for each person: "
                                   "true when a person has more than two hands or two arms, or a hand or arm belongs "
                                   "to nobody in the picture (for example hands holding a phone plus two more hands "
                                   "holding something else)")
    face_visible: bool | None = Field(default=None, description="true when any human face is in the picture, even "
                                      "partly or out of focus")
    person_visible: bool | None = Field(default=None, description="true when any part of a person (face, hand, arm, "
                                        "body) is in the picture")
    notes: str = Field(description="one sentence")


class ImageReviews(BaseModel):
    reviews: list[CandidateReview]


def gates_for(scene: str | None, shot: dict | None = None, *, plate: bool = False) -> tuple[str, ...]:
    """The gates an image is judged on. A set plate or outfit reference (payload `plate`, manyworlds.creator.sets)
    has nobody in it, so it has no identity gate: an identity row would always fail and count as a reject."""
    face = common.judges_identity(shot, plate=plate)   # a plate has nobody; a POV or cutaway shot shows no face
    base = tuple(g for g in GATES if face or g != "identity")
    return base + (("scene",) if scene else ()) + (("direction",) if shot else ())


def set_match(policy: Policy, shared: dict | None, r, has_references: bool) -> tuple[bool, dict] | None:
    """For a scene photo of a shared set (payload `shared_set`, creator.sets.shared_leader: the post's first
    accepted photo of that set is its SET REFERENCE), whether it shows the same place: the judge's `consistency`
    score at policy sets.shared_set.min_consistency or more. Always enforced: perf_compare 2026-11-25 had one bar
    shot and eight in a living room or kitchen, every one passed because scene fit only scores. None when the photo
    has no shared set, no reference was shown or the judge gave no score."""
    if not shared or not has_references or getattr(r, "consistency", None) is None:
        return None
    bar = float((policy.get("sets", "shared_set", default={}) or {}).get("min_consistency", 0))
    ok = float(r.consistency) >= bar
    return ok, {"would_pass": ok, "enforced": True, "bar": bar, "shared_set": shared}


BODY_CHECKS = ("anatomy", "framing")


def body_checks(policy: Policy | None, r, shot: dict | None, plate: bool) -> dict[str, tuple[bool, dict]]:
    """Defects in the picture itself, enforced (review.image.body_checks): `anatomy`, a person with extra hands or
    arms; `framing`, a face in a shot that must show none (a cutaway, hands or object shot: it would sit silent under
    her voice) or anyone at all in a set plate (followers are drawn into it). A field the judge left out checks
    nothing."""
    c = (policy.get("review", "image", "body_checks", default={}) or {}) if policy is not None else {}
    out: dict[str, tuple[bool, dict]] = {}
    if c.get("anatomy", True) and getattr(r, "extra_body_parts", False):   # a row only when found
        out["anatomy"] = (False, {"would_pass": False, "enforced": True})
    if c.get("framing", True) and not common.judges_identity(shot, plate=plate):
        seen = getattr(r, "person_visible" if plate else "face_visible", None)
        if seen is not None:
            ok = not seen
            out["framing"] = (ok, {"would_pass": ok, "enforced": True,
                                   "expects": "nobody" if plate else "no face"})
    return out


def _check_score(r, gate: str):
    return getattr(r, {"set_match": "consistency", **common.CONTEXT_SCORES}.get(gate, gate), None)


def decide(r: CandidateReview, cfg: dict, scene: str | None = None, *, shot: dict | None = None,
           policy: Policy | None = None, plate: bool = False) -> dict[str, bool]:
    verdict = {"identity": r.identity >= int(cfg.get("min_identity", 7)),
               "realism": r.realism >= int(cfg.get("min_realism", 6)) and not common.major_details(r, cfg),
               "safety": not (r.looks_under_min_age or r.sexualized or r.real_brand_marks)}
    if not common.judges_identity(shot, plate=plate):
        del verdict["identity"]
    if scene:  # a judge that leaves it out passes: fit is a preference, not a safety line
        verdict["scene"] = r.scene is None or r.scene >= int(cfg.get("min_scene", 0))
    if shot:
        verdict["direction"] = common.direction_passes(r.direction, policy)
    return verdict


def _judge(llm: StructuredLLM, policy: Policy, cfg: dict, role: str, effort: str, creator, refs: list[bytes],
           candidates: list[bytes], scene: str | None = None,
           shot: dict | None = None, extra: tuple[list, list[str]] = ([], []),
           presentation: dict | None = None) -> tuple[str, list[CandidateReview | None]]:
    """One vision call: the creator's references (cached prefix), then the candidates. With her `presentation`
    (brand_core.presentation) the same call scores brand alignment and distribution risk."""
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
    if presentation:
        content.append({"type": "text", "text": common.presentation_prompt(policy, presentation)})
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
    asset is accepted only when no row failed.

    A job in a test-overlay batch (operations.test_overlay.batches, e.g. test-breadth-*) is judged under that
    overlay (policy.for_batch: the breadth test's direction gate is report-only), wherever it is reviewed."""
    job = conn.execute("SELECT creator_id, batch, payload->>'scene' AS scene, payload->'shot' AS shot, "
                       "payload->'references' AS refs, payload->'plate' AS plate, payload->'shared_set' AS shared_set "
                       "FROM jobs WHERE id = %s",
                       (job_id,)).fetchone()
    policy = common.policy_for(policy, job.get("batch"))
    cfg = policy.get("review", "image", default={})
    stages = common.stages(cfg, role)
    shot = job["shot"] if isinstance(job["shot"], dict) else None   # a directed scene (manyworlds.direction)
    plate = bool(job["plate"])   # a set plate or outfit reference (creator.sets): nobody in it, no identity gate
    scene, gates = job["scene"], gates_for(job["scene"], shot, plate=plate)
    creator = conn.execute("SELECT id, bible, anchors FROM creators WHERE id = %s", (job["creator_id"],)).fetchone()
    presentation = None if plate else common.presentation_of(conn, job["creator_id"])
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
                                    [images[a["id"]] for a in remaining], scene, shot, extra,
                                    presentation=presentation)
        still = []
        for a, r in zip(remaining, reviews):
            checks = {}
            if r is None:  # the judge skipped it: fail closed, it can be re-reviewed
                verdict, details, scores = {g: False for g in gates}, {"error": "no review returned"}, {}
            else:
                verdict = decide(r, cfg, scene, shot=shot, policy=policy, plate=plate)
                details = r.model_dump(exclude=detail.unasked(r, shot))
                scores = {"identity": None if plate else r.identity, "realism": r.realism, "scene": r.scene}
                if shot:
                    scores["direction"] = r.direction
                # finer checks: a row each when scored; they reject only when enforced (review.detail)
                checks = detail.judged_verdicts(policy, r, consistency=bool(extra[1]))
                checks.update(common.context_verdicts(cfg, r))   # brand alignment, distribution (report only)
                same = set_match(policy, job.get("shared_set"), r, bool(extra[1]))
                if same is not None:   # a shared set (creator.sets.shared_leader): another place rejects
                    checks["set_match"] = same
                checks.update(body_checks(policy, r, shot, plate))   # extra hands; a face where none belongs
            details.update(model=model, stage=stage)
            rows += [(a["id"], g, verdict[g], scores.get(g), Jsonb(details)) for g in gates]
            rows += [(a["id"], g, ok, _check_score(r, g), Jsonb({**details, **info})) for g, (ok, info) in checks.items()]
            res = results[a["id"]]
            res["stages"].append({"stage": stage, "model": model, **scores,
                                  **{g: _check_score(r, g) for g in checks},
                                  "passed": all(verdict.values()) and all(ok for ok, _ in checks.values())})
            res.update({g: res[g] and verdict[g] for g in gates}, identity_score=scores.get("identity"),
                       realism_score=scores.get("realism"), scene_score=scores.get("scene"), defects=details.get("defects", []))
            res.update({g: res.get(g, True) and ok for g, (ok, _) in checks.items()})
            res["accepted"] = all(res[g] for g in gates) and all(res.get(g, True)
                                                                 for g in (*detail.JUDGED, "set_match", "brand",
                                                                           *BODY_CHECKS))
            if res["accepted"]:
                still.append(a)
        # A screen pass that clears every bar by `clear_pass_margin` needs no paid confirm (as for clips): only
        # close calls are worth the stronger model.
        if stage == "screen" and len(stages) > 1 and cfg.get("clear_pass_margin") is not None:
            m = int(cfg["clear_pass_margin"])
            face = common.judges_identity(shot, plate=plate)   # a faceless shot's identity score judges nothing
            for a in list(still):
                res = results[a["id"]]
                identity_clear = (not face or res["identity_score"] is None
                                  or res["identity_score"] >= int(cfg.get("min_identity", 7)) + m)
                if identity_clear and (res["realism_score"] or 0) >= int(cfg.get("min_realism", 6)) + m:
                    res["clear_pass"] = True
                    still.remove(a)
        # Only the best screen passes are worth a confirm: a post uses its top candidates.
        # The rest stay accepted on the screen alone ("confirmed": False).
        top = int(cfg.get("confirm_top_per_job", 0)) or len(still)
        still.sort(key=lambda a: -sum(results[a["id"]][f"{k}_score"] or 0 for k in ("identity", "realism", "scene")))
        remaining = still[:top]
    # A clear pass is stored on its screen rows, so production can choose it without a confirm row.
    clear = {str(i) for i, r in results.items() if r.get("clear_pass")}
    rows = [(aid, g, ok, sc, Jsonb({**j.obj, "clear_pass": True})
             if str(aid) in clear and j.obj.get("stage") == "screen" else j) for aid, g, ok, sc, j in rows]
    if write:
        common.write_rows(conn, rows)
    for r in results.values():
        r["confirmed"] = r["accepted"] and (len(r["stages"]) == len(stages) or bool(r.get("clear_pass")))
    return [results[a["id"]] for a in assets]


def pending_jobs(conn, policy: Policy, *, batch: str | None = None, limit: int | None = None) -> list:
    """Finished image jobs for a creator with reference images, whose images have no review yet.
    Reference-pack jobs are skipped: they are what the references are chosen from (a launch's refpack-<creator>
    batch, a casting audition's <batch>.refpack)."""
    cfg = policy.get("review", "image", default={})
    limit = limit or int(cfg.get("max_jobs_per_run", 50))
    return [r["id"] for r in conn.execute(
        """SELECT j.id FROM jobs j JOIN creators c ON c.id = j.creator_id
           WHERE j.status = 'done' AND j.kind LIKE 'image.%%'
             AND coalesce(j.batch, '') NOT LIKE 'refpack-%%' AND coalesce(j.batch, '') NOT LIKE '%%.refpack'
             AND jsonb_array_length(coalesce(c.anchors->'reference_images', '[]'::jsonb)) > 0
             AND (%(batch)s::text IS NULL OR j.batch = %(batch)s)
             AND EXISTS (SELECT 1 FROM assets a WHERE a.job_id = j.id AND a.kind = 'image'
                           AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id = a.id))
           ORDER BY j.updated_at LIMIT %(limit)s""", {"batch": batch, "limit": limit}).fetchall()]


def run_pending(conn, policy: Policy, llm: StructuredLLM, store: ObjectStore, workdir: Path,
                *, batch: str | None = None) -> dict:
    return common.run_jobs(pending_jobs(conn, policy, batch=batch),
                           lambda job, d: review_job(conn, policy, llm, store, job, d), workdir,
                           on_error=lambda job, e: common.give_up(conn, policy, job, e, asset_kind="image",
                                                                  section="image"))
