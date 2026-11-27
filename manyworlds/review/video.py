"""Review gate for story clips: does the creator stay herself, does the motion hold up.

The video worker stores evenly spaced keyframes beside each clip (comfy.wan_i2v, talk.*). The judge
sees the creator's reference photos, the approved start photo the clip was made from, and
the keyframes in order, and scores the clip as a whole. Every clip gets one `reviews` row
per gate, written against the video asset:

  identity  the face in the WORST keyframe still matches the references (image-to-video
            models drift: on 2026-11-14 a 480p Wan clip slid into a different face by 4 s)
  motion    the body and objects stay physically whole between frames: no morphing hands,
            merging limbs, things appearing or vanishing
  realism   each frame reads as phone video, not a render
  safety    as for images: adult-looking, nothing explicit, nude, implied-nude, fetish or soliciting (common.SAFETY_RULE,
            in context), no real brand marks
  scene     (when the planner wrote one) the shot shows the scene and the movement intended
  direction (when the job carries a shot sheet, manyworlds.direction) the clip follows the shot and its beats,
            on rubric policy/rubrics/direction-v<n>.yaml; below review.direction.min_score it is rejected
  dance_identity
            (motion transfer: comfy.wan_animate2, or a payload `reference` motion clip) identity per keyframe over
            `dance.identity.keyframes` keyframes, decided by manyworlds.dance.identity_check (worst and mean)
  performance (policy performance.observe) rank only, always passes: observed levels of the directed controls,
            appeal, charisma, expression and naturalness, stored per clip (performance_observations)
  finer checks (manyworlds.review.detail, policy review.detail; each rejects only when enforced):
    lip_closure, background  free, before the judge: lips closing on b/p/m (talking clips with a mouth series),
                             background edge flicker across the keyframes
    hands, teeth, consistency  scored by the judge when visible / when set or outfit references are given

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
from . import common, detail

SYSTEM = """You are the quality gate for an AI creator studio whose creators are openly labeled as AI.
You review short image-to-video clips of one fictional person from keyframes, strictly. A clip is only
as good as its worst frame. Score what you see, not what the prompt intended.
""" + common.SAFETY_SYSTEM

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
    sexualized: bool = Field(description=common.SAFETY_RULE)
    real_brand_marks: bool = Field(description="a legible real-world brand logo or trademark is visible")
    scene: int | None = Field(default=None, description="only when an INTENDED SHOT is given: 10 = shows that "
                              "setting and that movement; 5 = right setting, movement wrong or absent; 0 = different")
    direction: int | None = Field(default=None, description="only when a DIRECTED SHOT is given: 1-10 on the "
                                  "DIRECTION rubric given with it (does the clip follow the shot and its beats); "
                                  "null when no directed shot is given")
    hands: int | None = Field(default=None, description="only when DETAIL CHECKS ask for it: 1-10 for visible "
                              "hands in the worst keyframe; null when no hand is visible")
    teeth: int | None = Field(default=None, description="only when DETAIL CHECKS ask for it: 1-10 for visible "
                              "teeth in the worst keyframe; null when no teeth show")
    consistency: int | None = Field(default=None, description="only when SET / OUTFIT REFERENCE images are given: "
                                    "1-10, the same set and clothes as the references; null otherwise")
    identity_per_frame: list[int] | None = Field(default=None, description="only when PER-KEYFRAME IDENTITY is "
                                                "asked: one 1-10 identity score per keyframe, in keyframe order; "
                                                "null otherwise")
    # owner 2026-11-25: the same three judgments as images in the same call (review.common); clips report them
    brand_alignment: int | None = Field(default=None, description=common.BRAND_RULE)
    distribution_risk: int | None = Field(default=None, description=common.DISTRIBUTION_RULE)
    distribution_label: str | None = Field(default=None, description="only with distribution_risk: a few words")
    detail_defects: list[str] = Field(default_factory=list, description=common.DETAIL_RULE)
    # owner 2026-11-26 (manyworlds.performance): rank only, never a reason to reject (common.performance_prompt)
    # One string, parsed in code (common.parse_read): six more properties per candidate made the API refuse the image
    # judge ("Schema is too complex", 2026-11-26); the strict grammar's size grows with every property.
    performance_read: str = Field(default="", description=common.READ_RULE)
    notes: str = Field(description="one sentence")


def gates_for(scene: str | None, shot: dict | None = None) -> tuple[str, ...]:
    """A POV or cutaway clip (direction.faceless) shows no face, so it is not judged on identity."""
    base = tuple(g for g in GATES if g != "identity" or common.judges_identity(shot))
    return base + (("scene",) if scene else ()) + (("direction",) if shot else ())


MOTION_TRANSFER_KINDS = ("comfy.wan_animate2",)


def is_motion_transfer(kind: str | None, payload: dict | None) -> bool:
    """A clip driven by someone else's motion (dance and motion transfer): its kind, or a `reference` motion clip
    in the payload (manyworlds.dance.reference; `references`, plural, are set and outfit images, not motion)."""
    return kind in MOTION_TRANSFER_KINDS or bool((payload or {}).get("reference"))


def _identity_frames(policy: Policy, store: ObjectStore, video_uri: str, frames: list[tuple[bytes, str]],
                     workdir: Path) -> list[tuple[bytes, str]]:
    """The keyframes the dance identity gate scores: `dance.identity.keyframes` spread over the clip. The worker's
    own keyframes when it stored that many; else cut from the clip here (manyworlds.media.keyframes, plain ffmpeg);
    the stored ones when the clip can't be read."""
    from .. import dance, media
    n = int((dance.cfg(policy).get("identity") or {}).get("keyframes", len(frames)) or len(frames))
    if len(frames) >= n:
        return frames
    try:
        clip = store.get_file(video_uri, workdir / f"clip{Path(video_uri).suffix or '.mp4'}")
        d = workdir / "identity"
        d.mkdir(parents=True, exist_ok=True)
        return [(p.read_bytes(), "image/jpeg") for p, _, _ in media.keyframes(clip, d, n)]
    except Exception:  # noqa: BLE001 - an unreadable clip is judged on the worker's keyframes
        return frames


def decide(r: ClipReview, cfg: dict, scene: str | None = None, *, shot: dict | None = None,
           policy: Policy | None = None) -> dict[str, bool]:
    verdict = {"identity": r.identity >= int(cfg.get("min_identity", 7)),
               "motion": r.motion >= int(cfg.get("min_motion", 6)),
               "realism": r.realism >= int(cfg.get("min_realism", 6)) and not common.major_details(r, cfg),
               "safety": not (r.looks_under_min_age or r.sexualized or r.real_brand_marks)}
    if scene:
        verdict["scene"] = r.scene is None or r.scene >= int(cfg.get("min_scene", 0))
    if shot:
        verdict["direction"] = common.direction_passes(r.direction, policy)
    if not common.judges_identity(shot):
        del verdict["identity"]
    return verdict


def clear_pass(r: ClipReview, cfg: dict, shot: dict | None = None) -> bool:
    """The screen passed the clip on every scored bar by at least `clear_pass_margin` points: the paid confirm
    (about $0.05 a clip, as much as or more than a render) only checks the close calls. Off when unset. A clip
    with no face (common.judges_identity) is not held back by an identity score that judges nothing."""
    margin = cfg.get("clear_pass_margin")
    if margin is None:
        return False
    m = int(margin)
    identity_clear = not common.judges_identity(shot) or r.identity >= int(cfg.get("min_identity", 7)) + m
    return (identity_clear and r.motion >= int(cfg.get("min_motion", 6)) + m
            and r.realism >= int(cfg.get("min_realism", 6)) + m)


def _media_type(uri: str) -> str:
    return "image/jpeg" if uri.lower().endswith((".jpg", ".jpeg")) else "image/png"


def _judge(llm: StructuredLLM, policy: Policy, role: str, effort: str, creator, refs: list[bytes],
           start: bytes | None, frames: list[tuple[bytes, str]], scene: str | None,
           shot: dict | None = None, extra: tuple[list, list[str]] = ([], []),
           crops: tuple[list, set[str]] = ([], set()), *, per_frame: bool = False) -> ClipReview:
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
    if shot:
        content.append({"type": "text", "text": "DIRECTED SHOT (score `direction` 1-10 against it): "
                                                + common.direction_brief(shot, policy, still=False)})
    ref_content, ref_labels = extra   # set / outfit references (review.detail consistency)
    content += ref_content
    content += crops[0]               # hand and mouth close-ups (review.detail.crops)
    ask = detail.judge_prompt(policy, consistency=ref_labels, still=False, crops=crops[1])
    if ask:
        content.append({"type": "text", "text": ask})
    if per_frame:   # motion transfer: the dance identity gate needs every keyframe's own score
        content.append({"type": "text", "text": f"PER-KEYFRAME IDENTITY: also give `identity_per_frame`, one 1-10 "
                                                f"score per keyframe ({len(frames)} numbers, in order) for how "
                                                f"clearly that keyframe shows the REFERENCE person."})
    perf = common.performance_prompt(shot, policy, still=False)
    if perf:
        content.append({"type": "text", "text": perf})
    content.append({"type": "text", "text": "Review the clip as a whole from these keyframes."})
    r = llm.structured(role, system=SYSTEM, content=content, schema=ClipReview, effort=effort,
                       creator_id=creator["id"])
    for k in ("identity", "motion", "realism"):
        setattr(r, k, max(0, min(10, getattr(r, k))))
    r.identity_per_frame = ([max(1, min(10, int(s))) for s in r.identity_per_frame]
                            if (per_frame and r.identity_per_frame) else None)
    r.scene = max(0, min(10, r.scene)) if (scene and r.scene is not None) else None
    r.direction = max(1, min(10, r.direction)) if (shot and r.direction is not None) else None
    detail.clamp_scores(r, policy, consistency=bool(ref_labels))
    return r


def review_job(conn, policy: Policy, llm: StructuredLLM, store: ObjectStore, job_id, workdir: Path,
               *, write: bool = True) -> list[dict]:
    """Review the clip of one finished video job. Returns one dict per video asset (normally one).

    A job in a test-overlay batch (test-breadth-*) is judged under the overlay (common.policy_for). A motion-transfer
    clip (comfy.wan_animate2, or any job driven by a `reference` motion clip) also gets the `dance_identity` gate:
    the judge scores identity per keyframe (`dance.identity.keyframes` of them, cut from the clip when the worker
    stored fewer) and manyworlds.dance.identity_check decides."""
    job = conn.execute("SELECT creator_id, kind, post_id, batch, payload FROM jobs WHERE id = %s",
                       (job_id,)).fetchone()
    policy = common.policy_for(policy, job.get("batch"))
    cfg = policy.get("review", "video", default={})
    stages = common.stages(cfg)
    payload = job["payload"] or {}
    scene = payload.get("scene")
    shot = payload.get("shot") if isinstance(payload.get("shot"), dict) else None   # a directed clip
    motion_ref = is_motion_transfer(job.get("kind"), payload)
    gates = gates_for(scene, shot) + (("dance_identity",) if motion_ref else ())
    creator = conn.execute("SELECT id, bible, anchors FROM creators WHERE id = %s", (job["creator_id"],)).fetchone()
    videos = conn.execute("SELECT id, uri, meta FROM assets WHERE job_id = %s AND kind = 'video' ORDER BY uri",
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
    if motion_ref:
        frames = _identity_frames(policy, store, videos[0]["uri"], frames, workdir)

    heard = _intelligibility(conn, policy, store, job_id, payload, job["creator_id"], workdir)
    extra = detail.reference_blocks(detail.references_for(payload, shot, policy), store, workdir, image_block)
    crop_rows = conn.execute("SELECT uri, meta FROM assets WHERE job_id = %s AND kind = 'crop' ORDER BY uri",
                             (job_id,)).fetchall() if detail.check(policy, "crops").get("enabled") else []
    crops = detail.crop_blocks(crop_rows, store, workdir, image_block, policy)
    warp = detail.background_flicker([b for b, _ in frames], policy)   # free, so measured before any paid judge

    rows, out, observations = [], [], []
    target = "talk" if str(job.get("kind") or "").startswith("talk.") else "i2v"
    for v in videos:
        res = {"asset_id": str(v["id"]), "uri": v["uri"], "accepted": True, "stages": [], **{g: True for g in gates}}
        if heard is not None:   # the voice says the script (checked before anything paid per clip)
            rows.append((v["id"], "intelligibility", heard["passed"], heard["wer"], Jsonb(heard)))
            res["intelligibility"] = heard["passed"]
            if not heard["passed"]:
                res.update(accepted=False, confirmed=False)
                out.append(res)
                continue
        sync = (v["meta"] or {}).get("sync")
        if sync is not None:
            null = _sync_null(conn, cfg, store, job_id, payload, v, sync, workdir)
            if null is not None:   # the score against chance: recorded, and a gate only with null.min_margin set
                sync = {**sync, "null": null}
            res["sync"] = sync
        synced = lip_sync_ok(cfg, sync)
        if synced is False:   # the whole clip failed: score the frames its voice covers (the join drops the rest)
            voiced = _voiced_sync(conn, cfg, store, job_id, payload, v, sync, workdir)
            if voiced is not None:
                sync = {**sync, "voiced": voiced}
                res["sync"] = sync
                synced = lip_sync_ok(cfg, sync)
        if synced is not None:   # checked first: it is free, and a clip out of sync never needs a paid judge
            rows.append((v["id"], "lip_sync", synced, sync_value(cfg, sync), Jsonb(_without_series(sync))))
            res["lip_sync"] = synced
            if not synced:
                res.update(accepted=False, confirmed=False)
                out.append(res)
                continue
        # Free finer checks (review.detail): lip closure on b/p/m, background flicker. Enforced, a fail rejects here.
        free = _free_checks(conn, policy, payload, sync, warp)
        for g, (ok, value, info) in free.items():
            rows.append((v["id"], g, ok, value, Jsonb(info)))
            res[g] = ok
        if not all(ok for ok, _, _ in free.values()):
            res.update(accepted=False, confirmed=False)
            out.append(res)
            continue
        for role, effort, stage in stages:
            with common.charged_to(llm, job_id):   # review counts in this clip's cost
                r = _judge(llm, policy, role, effort, creator, refs, start, frames, scene, shot, extra, crops,
                           per_frame=motion_ref)
            verdict = decide(r, cfg, scene, shot=shot, policy=policy)
            checks = detail.judged_verdicts(policy, r, consistency=bool(extra[1]))   # hands, teeth, consistency
            scores = {"identity": r.identity, "motion": r.motion, "realism": r.realism, "scene": r.scene}
            if shot:
                scores["direction"] = r.direction
            held = None
            if motion_ref:   # motion transfer: every keyframe holds her face (manyworlds.dance.identity_check)
                from .. import dance
                per = r.identity_per_frame or []
                held = dance.identity_check(per if len(per) >= len(frames) else [], policy)   # fails closed
                if per and len(per) < len(frames):
                    held["reason"] = f"{len(per)} identity scores for {len(frames)} keyframes"
                verdict["dance_identity"], scores["dance_identity"] = held["passed"], held["min"]
            passed = all(verdict.values()) and all(ok for ok, _ in checks.values())
            clear = stage == "screen" and len(stages) > 1 and passed and clear_pass(r, cfg, shot)
            exclude = (detail.unasked(r, shot) or set()) | (set() if motion_ref else {"identity_per_frame"})
            details = {**r.model_dump(exclude=exclude), "model": llm.model_for(role),
                       "stage": stage, "clear_pass": clear, **({"dance_identity": held} if held else {})}
            rows += [(v["id"], g, verdict[g], scores.get(g), Jsonb(details)) for g in gates]
            rows += [(v["id"], g, ok, getattr(r, g), Jsonb({**details, **info})) for g, (ok, info) in checks.items()]
            perf = common.performance_result(r, shot, policy, target=target)
            if perf is not None:   # rank only: never part of the verdict
                if perf["points"] is not None:
                    rows.append(common.performance_row(v["id"], perf, details))
                observations.append({**perf, "asset_id": v["id"], "post_id": job.get("post_id"),
                                     "creator_id": job["creator_id"], "job_id": job_id, "model": job.get("kind"),
                                     "judge": details["model"], "stage": stage,
                                     "policy_version": getattr(policy, "version", None)})
            res["stages"].append({"stage": stage, "model": details["model"], **scores,
                                  **{g: getattr(r, g) for g in checks}, "drift_from": r.drift_from,
                                  "defects": r.defects, "passed": passed})
            res.update({g: res[g] and verdict[g] for g in gates})
            res.update({g: res.get(g, True) and ok for g, (ok, _) in checks.items()})
            res["accepted"] = all(res[g] for g in gates) and all(res.get(g, True) for g in detail.JUDGED)
            if not res["accepted"]:
                break   # rejects stay cheap: no confirm for a clip the screen failed
            if clear:
                res["clear_pass"] = True
                break   # passes cheaply: the screen cleared every bar by the margin, so no paid confirm
        res["confirmed"] = res["accepted"] and (len(res["stages"]) == len(stages) or bool(res.get("clear_pass")))
        out.append(res)
    if write:
        common.write_rows(conn, rows)
        from .. import performance
        performance.record_observations(conn, observations)
        if motion_ref:
            for res in out:
                if res.get("dance_identity") is False:
                    res["purged"] = purge_other_identity(conn, policy, store, job_id, res)
    return out


def purge_other_identity(conn, policy: Policy, store: ObjectStore, job_id, res: dict) -> bool:
    """A borrowed-motion clip that shows someone else is never kept (policy dance.identity.purge, on): perf_compare
    2026-11-25 and the dance tests of 2026-11-24 returned the stock clip's real performer, identity 0-1 of 10. Its
    video, keyframes and crops are deleted from the store and marked deleted, and the job records why. Only when the
    worst keyframe is under dance.identity.min_score (the floor), not for a near miss on the mean."""
    ident = policy.get("dance", "identity", default={}) or {}
    if not ident.get("purge", True):
        return False
    worst = next((s.get("dance_identity") for s in reversed(res.get("stages") or [])
                  if s.get("dance_identity") is not None), None)
    if worst is not None and float(worst) >= float(ident.get("min_score", 6)):
        return False
    rows = conn.execute("SELECT id, uri FROM assets WHERE job_id = %s AND kind IN ('video', 'keyframe', 'crop') "
                        "AND deleted_at IS NULL", (job_id,)).fetchall()
    for a in rows:
        try:
            store.delete(a["uri"])
        except Exception:  # noqa: BLE001 - marked deleted below; the retention sweep retries the object
            pass
    conn.execute("UPDATE assets SET deleted_at = now() WHERE job_id = %s AND kind IN ('video', 'keyframe', 'crop') "
                 "AND deleted_at IS NULL", (job_id,))
    conn.execute("UPDATE jobs SET error = %s WHERE id = %s",
                 (f"identity guard: the motion clip shows someone else (worst keyframe {worst}); deleted", job_id))
    return True


def _intelligibility(conn, policy: Policy, store: ObjectStore, job_id, payload: dict, creator_id, workdir: Path):
    """For a talking clip: does its voice line say the script (manyworlds.speech)? None when not enforced, or the
    clip has no voice line and script of its own (a story clip)."""
    cfg = policy.get("review", "video", "intelligibility", default={}) or {}
    if not cfg.get("enforce") or not (payload.get("script") or "").strip():
        return None
    audio = conn.execute("SELECT uri, meta FROM assets WHERE job_id = %s AND kind = 'audio' ORDER BY uri LIMIT 1",
                         (job_id,)).fetchone()
    if not audio and payload.get("audio_uri"):   # a designed voice's line, spoken by its own voice.speak job
        audio = conn.execute("SELECT uri, meta FROM assets WHERE uri = %s AND kind = 'audio' LIMIT 1",
                             (payload["audio_uri"],)).fetchone()
    if not audio:
        return None
    from .. import speech
    cached = (audio["meta"] or {}).get("words")   # word timestamps production.voice_words already paid for
    if cached:
        text = " ".join(str(w.get("word", "")).strip() for w in cached if isinstance(w, dict))
        return {**speech.score(policy, payload["script"], text), "cached_words": True}
    path = store.get_file(audio["uri"], workdir / f"voice{Path(audio['uri']).suffix or '.wav'}")
    return speech.check(conn, policy, path, payload["script"], float((audio["meta"] or {}).get("seconds") or 0),
                        job_id=job_id, creator_id=creator_id)


def _without_series(sync: dict) -> dict:
    """A sync score for a review row: the per-frame mouth series stays on the asset, not in every row."""
    return {k: v for k, v in sync.items() if k != "openness"}


def _voice_lines(conn, payload: dict, script: str) -> list[dict] | None:
    """The designed voice's per-line timings (voice.speak meta `lines`) when that line is this clip's own audio
    and says this clip's script; else None (word times are then spread over the script)."""
    if not payload.get("audio_uri"):
        return None
    row = conn.execute("SELECT meta FROM assets WHERE uri = %s AND kind = 'audio' LIMIT 1",
                       (payload["audio_uri"],)).fetchone()
    lines = ((row or {}).get("meta") or {}).get("lines") if row else None
    if not lines or not all(isinstance(ln, dict) and "start_s" in ln and "end_s" in ln for ln in lines):
        return None
    said = " ".join(str(ln.get("text", "")) for ln in lines).split()
    return lines if said == script.split() else None


def _free_checks(conn, policy: Policy, payload: dict, sync, warp) -> dict[str, tuple[bool, float, dict]]:
    """{gate: (passed, score, details)} for the finer checks that cost nothing (review.detail): lip closure on
    b/p/m for a talking clip with a mouth series, and background flicker across the keyframes. Off, or without
    the signal: no entry."""
    out = {}
    script = (payload.get("script") or "").strip()
    if script and isinstance(sync, dict):
        m = detail.lip_closure(sync, policy, script=script, lines=_voice_lines(conn, payload, script))
        v = detail.verdict(policy, "lip_closure", m.get("share")) if m else None
        if v is not None:
            out["lip_closure"] = (v[0], m["share"], {**m, **v[1]})
    if warp is not None:
        v = detail.verdict(policy, "background", warp["flicker"], lower_is_better=True)
        if v is not None:
            out["background"] = (v[0], warp["flicker"], {**warp, **v[1]})
    return out


def lip_sync_ok(cfg: dict, sync: dict | None) -> bool | None:
    """Whether a talking clip's mouth follows its voice (worker/lipsync.py), or None when the score isn't
    enforced (policy `review.video.lip_sync.enforce`, off until calibrated by eye) or there is none to judge."""
    ls = cfg.get("lip_sync") or {}
    if not ls.get("enforce") or not sync or "error" in sync:
        return None
    face = float(sync.get("face_share") or 0) >= float(ls.get("min_face_share", 0.8))
    whole = sync.get("corr") is not None and float(sync["corr"]) >= float(ls.get("min_corr", 0.3))
    floor = (ls.get("null") or {}).get("min_margin")   # beats chance by this much (report-only while unset)
    if floor is not None and sync.get("null") and float(sync["null"]["margin"]) < float(floor):
        return False
    return face and (whole or _voiced_ok(ls, sync.get("voiced")))


def _voiced_ok(ls: dict, voiced: dict | None) -> bool:
    """The voiced span's score passes (lipsync.voiced_score): the same min_corr, a mouth that moves over it
    (voiced_span.min_open_std) and a face in it."""
    vs = ls.get("voiced_span") or {}
    if not vs.get("enabled") or not voiced or voiced.get("corr") is None:
        return False
    return (float(voiced["corr"]) >= float(ls.get("min_corr", 0.3))
            and float(voiced.get("open_std") or 0) >= float(vs.get("min_open_std", 0.0))
            and float(voiced.get("face_share") or 0) >= float(ls.get("min_face_share", 0.8)))


def sync_value(cfg: dict, sync: dict) -> float | None:
    """The lip_sync row's score: the voiced span's when it decided the pass, else the whole clip's."""
    ls = cfg.get("lip_sync") or {}
    whole = sync.get("corr") is not None and float(sync["corr"]) >= float(ls.get("min_corr", 0.3))
    if not whole and _voiced_ok(ls, sync.get("voiced")):
        return sync["voiced"]["corr"]
    return sync.get("corr")


def _line_audio(conn, job_id, payload: dict, video_uri: str) -> str:
    """Where the clip's voice line is: the job's own audio output (a Kokoro line, a cut window), else the designed
    voice's line it was given, else the clip's own sound track."""
    row = conn.execute("SELECT uri FROM assets WHERE job_id = %s AND kind = 'audio' ORDER BY uri LIMIT 1",
                       (job_id,)).fetchone()
    return row["uri"] if row else (payload.get("audio_uri") or video_uri)


def _decode_mono(path: Path, rate: int):
    """The file's sound as mono float samples at `rate` (ffmpeg; a video's audio track too)."""
    import subprocess

    import numpy as np

    from .. import media
    r = subprocess.run([media.ffmpeg(), "-v", "error", "-i", str(path), "-vn", "-f", "f32le", "-ac", "1",
                        "-ar", str(rate), "-"], capture_output=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or b"")[-300:].decode(errors="replace"))
    return np.frombuffer(r.stdout, dtype=np.float32)


def _voiced_sync(conn, cfg: dict, store: ObjectStore, job_id, payload: dict, video: dict, sync: dict | None,
                 workdir: Path) -> dict | None:
    """The sync score over the clip's voiced frames (lipsync.voiced_score), from its stored mouth series and its
    voice line; None when off, when the clip has no series or no frames past its line, or the audio can't be read
    (the whole-clip verdict stands)."""
    import numpy as np

    from ..worker import lipsync
    ls = cfg.get("lip_sync") or {}
    vs = ls.get("voiced_span") or {}
    series = (sync or {}).get("openness")
    if not vs.get("enabled") or not series:
        return None
    got = _clip_loudness(conn, store, job_id, payload, video, sync, int(vs.get("audio_rate", 16000)), workdir)
    if got is None:
        return None
    openness, loud, voiced = got
    return lipsync.voiced_score(openness, loud, voiced, max_offset_frames=int(vs.get("max_offset_frames", 5)),
                                min_frames=int(vs.get("min_frames", 25)), tail_frames=int(vs.get("tail_frames", 0)))


def _clip_loudness(conn, store: ObjectStore, job_id, payload: dict, video: dict, sync: dict, rate: int,
                   workdir: Path):
    """(the clip's mouth series, its voice line's loudness per frame on the clip's clock, the frames the line spans
    from the clip's start) or None when there is no series or the line can't be read. The line starts after the
    take's lead-in (meta.lead_s, worker talk_comfy.rolls): unshifted, a rolled take's line sat 7-8 frames early,
    past the lag search, and the voiced-span score could not pass a good take (review 2026-11-26)."""
    import numpy as np

    from ..worker import lipsync
    series = (sync or {}).get("openness")
    if not series:
        return None
    fps = float(sync.get("fps") or 25)
    uri = _line_audio(conn, job_id, payload, video["uri"])
    try:
        samples = _decode_mono(store.get_file(uri, Path(workdir) / f"line_{video['id']}{Path(uri).suffix}"), rate)
    except Exception:  # noqa: BLE001 - without the line the whole-clip score decides
        return None
    if not len(samples):
        return None
    lead = float(((video.get("meta") or {}).get("lead_s")) or 0)
    if lead > 0 and uri != video["uri"]:   # the clip's own track already carries its lead-in
        samples = np.concatenate([np.zeros(int(round(lead * rate)), dtype=samples.dtype), samples])
    openness = np.array([np.nan if x is None else float(x) for x in series], dtype=float)
    return openness, lipsync.loudness(samples, rate, fps, len(openness)), int(np.ceil(len(samples) / rate * fps))


def _sync_null(conn, cfg: dict, store: ObjectStore, job_id, payload: dict, video: dict, sync: dict | None,
               workdir: Path) -> dict | None:
    """The clip's sync against chance (lipsync.null_margin, policy review.video.lip_sync.null), or None when off or
    unmeasurable."""
    nc = (cfg.get("lip_sync") or {}).get("null") or {}
    if not nc.get("enabled") or not sync or sync.get("corr") is None:
        return None
    got = _clip_loudness(conn, store, job_id, payload, video, sync, int(nc.get("audio_rate", 16000)), workdir)
    if got is None:
        return None
    from ..worker import lipsync
    openness, loud, _ = got
    return lipsync.null_margin(openness, loud, float(sync["corr"]), min_lag=int(nc["min_lag_frames"]),
                               max_lag=int(nc["max_lag_frames"]), percentile=float(nc["percentile"]))


def pending_jobs(conn, policy: Policy, *, batch: str | None = None, limit: int | None = None) -> list:
    """Finished clip jobs, story clips and talking clips (with keyframes) for a creator with reference images,
    not yet reviewed. Keyframes show identity and artefacts, not whether the mouth matches the audio. An
    intermediate clip (payload.intermediate: walk-and-talk's silent motion clip) is not reviewed: the lip-synced take
    made from it is, except a borrowed-motion clip (Wan-Animate-2): it is judged on her identity before any lip sync
    (production.motion._borrowed_state)."""
    limit = limit or int(policy.get("review", "video", "max_jobs_per_run", default=10))
    return [r["id"] for r in conn.execute(
        """SELECT j.id FROM jobs j JOIN creators c ON c.id = j.creator_id
           WHERE j.status = 'done' AND j.kind LIKE ANY(%(kinds)s::text[])
             AND j.kind NOT LIKE '%%.mirror'
             AND (coalesce(j.payload->>'intermediate', '') <> 'true' OR j.kind = ANY(%(motion)s::text[])
                  OR j.payload ? 'reference')
             AND jsonb_array_length(coalesce(c.anchors->'reference_images', '[]'::jsonb)) > 0
             AND (%(batch)s::text IS NULL OR j.batch = %(batch)s)
             AND EXISTS (SELECT 1 FROM assets k WHERE k.job_id = j.id AND k.kind = 'keyframe')
             AND EXISTS (SELECT 1 FROM assets a WHERE a.job_id = j.id AND a.kind = 'video'
                           AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id = a.id))
           ORDER BY j.updated_at LIMIT %(limit)s""",
        {"batch": batch, "limit": limit, "kinds": reviewed_kind_patterns(policy),
         "motion": list(MOTION_TRANSFER_KINDS)}).fetchall()]


def _like_literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def reviewed_kind_patterns(policy: Policy) -> list[str]:
    """The clip job kinds the video gate judges, as LIKE patterns: policy review.video.kinds, where an entry ending
    in `.*` is a kind prefix (`talk.*`: every talk kind) and any other is one kind, matched exactly."""
    kinds = policy.get("review", "video", "kinds", default=None) or [
        "video.*", "talk.*", "hosted.video", "comfy.wan_i2v", "comfy.wan_animate2"]
    return [_like_literal(k[:-1]) + "%" if k.endswith(".*") else _like_literal(k) for k in kinds]


def run_pending(conn, policy: Policy, llm: StructuredLLM, store: ObjectStore, workdir: Path,
                *, batch: str | None = None) -> dict:
    return common.run_jobs(pending_jobs(conn, policy, batch=batch),
                           lambda job, d: review_job(conn, policy, llm, store, job, d), workdir, prefix="video-",
                           on_error=lambda job, e: common.give_up(conn, policy, job, e, asset_kind="video",
                                                                  section="video"))
