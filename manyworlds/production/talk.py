"""Talking video: the lip-synced talk job (from a photo, or V2V over a motion clip), the hybrid's cutaways, and a
long script's segments, each its own job, joined with each shot's transition in (on the finisher, finish.join, while
production.finish_mode is queue); repairs of a take that failed lip sync (talk_repair)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from psycopg.types.json import Jsonb

from .. import queue
from ..policy import Policy
from .common import SUNG, _cfg, _seed, _shot, _with_crops, _with_prompts, _with_shot, unmarked
from .finish import captions_burned
from .motion import _advance_motion, _last_frame, _motion_cfg, _needs_motion, talk_v2v
from .steps import DropPost, Step, _clip_state, _edit_state, advance_step
from .takes import _hybrid_cfg, _photo_per_shot, _take_index, _take_photo, _take_script, _take_shot, _talk_long_cfg
from .voice import _advance_voice, _designed_voice, _with_audio, voice_lines


def _talk_job(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], attempt: int, *,
              segment: int | None = None, start_uri: str | None = None, audio: dict | None = None,
              motion: dict | None = None):
    """A talking video: the creator says the script to camera from the approved photo, lip-synced,
    in their own voice (Kokoro on the talk worker, or the designed voice's finished line, `audio`: the worker then
    speaks nothing, and `script` stays for the judge). For a long script, one segment of it. A directed post's take
    (body.talk_shots) uses its shot's scene, prompt and, when every shot has its own photo, that photo. With
    `motion` (the take's silent motion clip, walk-and-talk), the clip is lip-synced instead of the photo
    (`video_uri`, generation.talk_motion.v2v settings), when the talk engine is ComfyUI (`talk_v2v`)."""
    row = conn.execute("SELECT bible, anchors FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()
    voice = (row["anchors"] or {}).get("voice") or {}
    params = dict(policy.get("generation", "talk", default={}) or {})
    default_voice = params.pop("voice", "af_heart")
    params.pop("max_take_seconds", None)   # production's take limit, not a render setting
    i = _take_index(body, segment)
    scene, shot = body["scenes"][i], _shot(body, i)
    from ..direction import talk_profile_for
    profile = talk_profile_for(shot, policy)   # 720p close-ups (direction.hd_close_ups; off as shipped)
    if profile:
        params.update(policy.get("generation", "talk_profiles", profile, default={}) or {})
    motion = motion if motion and talk_v2v(policy) else None   # the research-script engine has no V2V
    if motion:
        params.update(_motion_cfg(policy).get("v2v") or {})
    prior = ((body.get("talk_segment_jobs") or [])[segment] if segment is not None
             and segment < len(body.get("talk_segment_jobs") or []) else body.get("talk_jobs") or [])
    photo = start_uri or _take_photo(body, chosen, i)["uri"]
    options = RepairOptions(
        photos=(lambda used: _other_photos(conn, body, i, used)) if not start_uri else None,
        cutaway=(lambda: _cutaway_photo(policy, body, chosen, segment)) if segment is not None else None)
    repair = talk_repair(conn, policy, [str(j) for j in prior], params, v2v=bool(motion), options=options)
    if repair and repair.get("knob") == "cutaway":   # the line goes under a cutaway shot, in the same voice
        body["talk_modes"][segment] = "narrated"
        body.setdefault("rerouted", {})[str(segment)] = {"image_uri": repair["image_uri"],
                                                         "reason": repair.get("reason", "")}
        return _cutaway_job(conn, policy, post, body, chosen, attempt, segment=segment, audio=audio)
    script = unmarked(_take_script(body, segment))
    seed = _seed(post, 700 + (segment or 0) * 1000, attempt)
    payload = {"image_uri": (repair or {}).get("image_uri") or photo, "script": script,
               "voice": voice.get("voice", default_voice), "speed": float(voice.get("speed", 1.0)),
               # the video judge scores against it (a music_video take sings its song's words)
               "scene": f"{scene} {'Singing' if post['format'] in SUNG else 'Speaking'} to camera: {script}",
               "keyframes": int(policy.get("review", "video", "keyframes", default=4)), "seed": seed, "params": params}
    _with_prompts(payload, "talk", row["bible"], scene, shot=shot, policy=policy)
    _with_shot(payload, shot)
    _with_crops(payload, policy)
    _with_audio(payload, audio)
    if motion:
        payload["video_uri"] = motion["uri"]
    if repair:
        payload["repair"] = repair
    if segment is not None:
        payload["segment"] = {"index": segment, "of": len(body["talk_segments"])}
    key = f"post:{post['id']}:talk:{attempt}" if segment is None else f"post:{post['id']}:talk:{segment}:{attempt}"
    return queue.enqueue(conn, _cfg(policy).get("talk_kind", "talk.infinitetalk"), payload,
                         creator_id=post["creator_id"], batch=f"post-{post['id']}", post_id=post["id"],
                         idempotency_key=key)


def _take_history(conn, job_ids: list[str]) -> list[dict]:
    """A take's earlier talk jobs, oldest first: {id, start_step, repair, image_uri, passed, corr, failed}. passed and
    corr come from the lip_sync review row of its video (None when it has none: a failed job, or the gate not
    enforced); failed is {gate: score} of every gate its video failed (the latest row per gate)."""
    if not job_ids:
        return []
    rows = {str(r["id"]): dict(r) for r in conn.execute(
        """SELECT j.id, j.payload->'params'->>'start_step' AS start_step, j.payload->'repair' AS repair,
                  j.payload->>'image_uri' AS image_uri,
                  (SELECT r.passed FROM assets a JOIN reviews r ON r.asset_id = a.id
                    WHERE a.job_id = j.id AND a.kind = 'video' AND r.gate = 'lip_sync' ORDER BY r.created_at DESC
                    LIMIT 1) AS passed,
                  (SELECT r.score FROM assets a JOIN reviews r ON r.asset_id = a.id
                    WHERE a.job_id = j.id AND a.kind = 'video' AND r.gate = 'lip_sync' ORDER BY r.created_at DESC
                    LIMIT 1) AS corr
           FROM jobs j WHERE j.id = ANY(%s::uuid[])""", (job_ids,)).fetchall()}
    for r in rows.values():
        r["failed"] = {}
    for f in conn.execute(
            """SELECT DISTINCT ON (a.job_id, r.gate) a.job_id, r.gate, r.passed, r.score
               FROM assets a JOIN reviews r ON r.asset_id = a.id
               WHERE a.job_id = ANY(%s::uuid[]) AND a.kind = 'video'
               ORDER BY a.job_id, r.gate, r.created_at DESC""", (job_ids,)).fetchall():
        if f["passed"] is False and str(f["job_id"]) in rows:
            rows[str(f["job_id"])]["failed"][f["gate"]] = None if f["score"] is None else float(f["score"])
    return [rows[j] for j in job_ids if j in rows]


@dataclass
class RepairOptions:
    """What a take's repair may change besides the render: `photos(used)` lists the scene's other accepted photos
    (uris not in `used`), `cutaway()` the faceless photo a line could go under instead ({image_uri, reason}, or
    None when the post and its grammar don't allow it)."""
    photos: Callable[[set], list[str]] | None = None
    cutaway: Callable[[], dict | None] | None = None


def _repair_cfg(policy: Policy) -> dict:
    return policy.get("review", "video", "talk_repair", default={}) or {}


def _cause(failed: dict, gates: dict) -> str | None:
    """The gate a take's repair answers: the first failed gate with an escalation (policy order), else any."""
    for g in gates:
        if g in failed:
            return g
    return next(iter(sorted(failed)), None)


def talk_repair(conn, policy: Policy, prior: list[str], params: dict, *, v2v: bool,
                options: RepairOptions | None = None) -> dict | None:
    """What a take's next talk job changes, given its earlier jobs (`prior`, oldest first); sets params (start_step,
    audio guidance) and returns the record kept in the job's payload.repair: {cause, knob, start_step, dead_knobs,
    applied, image_uri}. None for a first take. Knob `cutaway` means the line goes under a faceless shot instead (the
    caller reroutes it; image_uri is that shot's photo).

    Breadth run 2026-11-23: V2V takes (walk-and-talk) failed lip_sync with the mouth never redrawn, and every repair
    changed only the seed and scored the same. A lip_sync failure on a V2V take redraws more: start_step one lower
    per repair (generation.talk_motion.v2v_repair.start_step_step) down to start_step_floor.

    perf_compare 2026-11-25: segment 8 failed lip_sync three times at -0.46, -0.34, -0.43, each repair only a new
    seed. Each gate has an escalation (policy review.video.talk_repair.gates.<gate>.steps, after start_step for a V2V
    lip-sync take): the first knob not dead is used. A knob whose repair failed the same gate as the attempt before
    with a score within the gate's tolerance (start_step: v2v_repair.same_score_tolerance) is dead: it is not tried
    again (dead_knobs says why), and the next one changes something material: lip_sync raises the audio guidance
    (talk_repair.audio_guidance), then puts the line under a cutaway when the grammar allows it; identity and realism
    start from another accepted photo of the scene. An escalation, once applied, is kept by later repairs. When every
    knob is dead the post is dropped with the reason instead of rendering the same clip again."""
    hist = _take_history(conn, prior)
    if not hist:
        return None
    options = options or RepairOptions()
    rcfg = _repair_cfg(policy)
    gates = rcfg.get("gates") or {}
    vcfg = _motion_cfg(policy).get("v2v_repair") or {}
    last = hist[-1]
    record_before = last["repair"] or {}
    dead = dict(record_before.get("dead_knobs") or {})
    applied = list(record_before.get("applied") or [])
    image = record_before.get("image_uri")
    start = last["start_step"] if last["start_step"] is not None else params.get("start_step")
    v2v = v2v and start is not None
    if v2v:
        params["start_step"] = int(start)
    if "audio_guidance" in applied:
        params.update(rcfg.get("audio_guidance") or {})
    out = {"cause": None, "knob": "seed", "start_step": params.get("start_step"), "dead_knobs": dead,
           "applied": applied, **({"image_uri": image} if image else {})}
    failed = _with_lip_sync(last)
    cause = _cause(failed, gates)
    if cause is None:      # the job failed, or its clip was rejected by no gate: a new seed
        return out
    out["cause"] = cause
    knob_before = str(record_before.get("knob") or "seed")
    before = _with_lip_sync(hist[-2]) if len(hist) >= 2 else {}
    tol = float(vcfg.get("same_score_tolerance", 0.0)) if knob_before == "start_step" else \
        float((gates.get(cause) or {}).get("tolerance", 0.0))
    if failed.get(cause) is not None and before.get(cause) is not None \
            and abs(float(failed[cause]) - float(before[cause])) <= tol:
        label = cause.replace("_", " ")
        dead[knob_before] = (f"attempt {len(hist)} changed {knob_before} and scored {label} {float(failed[cause]):.3f},"
                             f" as attempt {len(hist) - 1} did ({float(before[cause]):.3f})")
    steps = list((gates.get(cause) or {}).get("steps") or ["seed"])
    if v2v and cause == "lip_sync":
        steps = ["start_step", *[s for s in steps if s != "start_step"]]
    for knob in steps:
        if knob in dead:
            continue
        if knob == "start_step":
            floor, step = int(vcfg.get("start_step_floor", 0)), int(vcfg.get("start_step_step", 1))
            if not v2v or int(start) <= floor:
                continue
            params["start_step"] = max(floor, int(start) - step)
        elif knob == "audio_guidance":
            guidance = rcfg.get("audio_guidance") or {}
            if not guidance:
                continue
            params.update(guidance)
            if knob not in applied:
                applied = [*applied, knob]
        elif knob == "next_photo" and image and knob_before == "next_photo":
            pass   # the last new photo did better than the attempt before it: keep it, with a new seed
        elif knob == "next_photo":
            used = {h.get("image_uri") for h in hist if h.get("image_uri")}
            others = options.photos(used) if options.photos else []
            if not others:
                continue
            out["image_uri"] = others[0]
        elif knob == "cutaway":
            under = options.cutaway() if options.cutaway else None
            if not under:
                continue
            return {**out, "knob": knob, "image_uri": under["image_uri"], "reason": under.get("reason", ""),
                    "applied": applied}
        elif knob != "seed":
            continue
        return {**out, "knob": knob, "start_step": params.get("start_step"), "applied": applied}
    raise DropPost(f"{cause.replace('_', ' ')} failed and no repair changes it: "
                   + "; ".join(f"{k}: {v}" for k, v in dead.items()))


def _with_lip_sync(h: dict) -> dict:
    """An attempt's failed gates, with its lip_sync row's score when that row failed."""
    failed = dict(h.get("failed") or {})
    if h.get("passed") is False and "lip_sync" not in failed:
        failed["lip_sync"] = None if h.get("corr") is None else float(h["corr"])
    return failed


def _other_photos(conn, body: dict, i: int, used: set) -> list[str]:
    """The scene's other photos that passed every review row (best first), none a take of it has used yet."""
    jobs = [str(j) for j in ((body.get("scene_jobs") or [])[i] if i < len(body.get("scene_jobs") or []) else [])]
    if not jobs:
        return []
    rows = conn.execute(
        """SELECT a.uri, coalesce(sum(r.score) FILTER (WHERE r.details->>'stage' = 'confirm'),
                                  sum(r.score) FILTER (WHERE r.details->>'stage' = 'screen')) AS score
           FROM assets a JOIN reviews r ON r.asset_id = a.id
           WHERE a.job_id = ANY(%s::uuid[]) AND a.kind = 'image'
           GROUP BY a.uri HAVING bool_and(r.passed)
           ORDER BY score DESC NULLS LAST, a.uri""", (jobs,)).fetchall()
    return [r["uri"] for r in rows if r["uri"] not in used]


def _cutaway_photo(policy: Policy, body: dict, chosen: list[dict], segment: int) -> dict | None:
    """The faceless photo a failed on-camera line can go under ({image_uri, reason}), or None: the post's edit
    grammar must lip-sync on camera only, the line may not be one the policy keeps on camera
    (review.video.talk_repair.cutaway.keep_on_camera: first = the hook, last = the sign-off), another take must stay
    on camera, and the post must have a faceless shot with its own photo (the nearest one is used)."""
    from .. import cuts
    from ..direction import faceless
    if str(cuts.grammar(policy, body.get("edit_grammar")).get("lip_sync") or "all") != "on_camera":
        return None
    modes = list(body.get("talk_modes") or [])
    n = len(body.get("talk_segments") or [])
    if not modes or len(modes) != n:
        return None
    keep = set((_repair_cfg(policy).get("cutaway") or {}).get("keep_on_camera") or [])
    if ("first" in keep and segment == 0) or ("last" in keep and segment == n - 1):
        return None
    if not any(m == "talk" and k != segment for k, m in enumerate(modes)):
        return None
    if not _photo_per_shot(body, chosen):
        return None
    here = _take_shot(body, segment)
    shots = body.get("shots") or []
    near = sorted((k for k, s in enumerate(shots) if faceless(s) and k < len(chosen)), key=lambda k: abs(k - here))
    if not near:
        return None
    return {"image_uri": chosen[near[0]]["uri"], "reason": f"lip sync failed on camera; the line goes under shot {near[0]}"}


def _cutaway_job(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], attempt: int, *, segment: int,
                 audio: dict | None = None):
    """A cutaway segment of a hybrid talking video: the post's other photos, moving gently, under the creator's
    voice saying this part of the script (the narrated-edit worker; no lip-sync render). With `audio`, the
    designed voice's finished line instead of Kokoro. A line rerouted by a lip-sync repair (body.rerouted) goes
    under the faceless photo the repair chose."""
    anchors = conn.execute("SELECT anchors FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["anchors"]
    out = _hybrid_cfg(policy).get("output", {}) or {}
    rerouted = (body.get("rerouted") or {}).get(str(segment))
    if rerouted:
        images = [{"uri": rerouted["image_uri"]}]
    elif body.get("talk_shots") and _photo_per_shot(body, chosen):   # a directed cutaway: its own shot's photo
        images = [chosen[_take_shot(body, segment)]]
    else:
        images = chosen[1:] or chosen
    payload = {"images": [a["uri"] for a in images], "script": unmarked(body["talk_segments"][segment]),
               "voice": (anchors.get("voice") or {}).get("voice", "af_heart"), "on_screen_text": "",
               **(_cfg(policy).get("edit", {}) or {}), **{k: out[k] for k in ("width", "height", "fps") if k in out}}
    _with_audio(payload, audio, captions=True)
    if captions_burned(policy, "talking_video"):   # the joined video's captions are burned in at the finish stage
        payload["captions"] = False
    return queue.enqueue(conn, "image.narrated", payload, creator_id=post["creator_id"], batch=f"post-{post['id']}",
                         post_id=post["id"], idempotency_key=f"post:{post['id']}:cutaway:{segment}:{attempt}")


def join_transitions(policy: Policy, body: dict, n: int) -> list[str] | None:
    """The transition at each of the n - 1 joins of a directed post's takes: where the shot changes, the next shot's
    `edit.transition_in` as direction.transition_kind reads it (empty text: a cut); within one shot (a take split for
    length), a cut. None when every join is a cut, or the post has no shots (a plain join)."""
    if not body.get("shots") or n < 2:
        return None
    from ..direction import transition_kind
    out = []
    for i in range(1, n):
        k = _take_shot(body, i)
        text = str((((_shot(body, k) or {}).get("edit") or {}).get("transition_in")) or "").strip()
        out.append(transition_kind(text, policy) if text and k != _take_shot(body, i - 1) else "cut")
    return out if any(t != "cut" for t in out) else None


# The join of a long talking video's takes is three halves, like the finish (production.finish), so its encodes can
# leave the producer: join_plan (the database half, JSON-ready: the finish.join job's payload), render_join (ffmpeg,
# no database: normalise each take to the hybrid's output when needed, join with the transitions, upload) and
# record_join (the joined asset row). Breadth run 2026-11-23: a hybrid's three takes normalised to 1080x1920 with
# motion interpolation on the producer's shared CPU timed out (600 s) on every loop and the post hung for hours.
# While production.finish_mode is queue the join is a finish.join job on the finisher (a Step: a join that keeps
# failing drops the post with the reason); inline, the producer joins. Each ffmpeg call's timeout is sized for its
# input (join_timeout).
JOIN_KIND = "finish.join"     # the finisher's join job (worker/handlers/finish_video.py)
JOIN_SECTIONS = ("finishing",)   # the policy the join reads (interpolation, transitions, output flags)


def _join_key(post: dict) -> str:
    return f"assets/{post['creator_id']}/post-{post['id']}/talk-joined.mp4"


def join_timeout(plan_or_cfg: dict, seconds: float | None) -> float:
    """An ffmpeg call's timeout for `seconds` of input: generation.talk_long.join_seconds_per_input_second per
    second of it, never under join_timeout_s."""
    cfg = plan_or_cfg.get("timeout", plan_or_cfg)
    floor = float(cfg.get("join_timeout_s", 600))
    per = float(cfg.get("join_seconds_per_input_second") or 0)
    return max(floor, per * float(seconds or 0))


def _existing_join(conn, store, post: dict) -> dict | None:
    return conn.execute("SELECT id, uri FROM assets WHERE uri = %s AND kind = 'video'",
                        (store.uri_for(_join_key(post)),)).fetchone()


def join_plan(policy: Policy, post: dict, videos: list[dict], body: dict) -> dict:
    """Everything render_join needs, JSON-ready (the finish.join payload)."""
    normalize = bool(body.get("talk_modes") or _hybrid_cfg(policy).get("normalize_all"))
    out_cfg = _hybrid_cfg(policy).get("output", {}) or {}
    long_cfg = _talk_long_cfg(policy)
    return {"post_id": str(post["id"]), "key": _join_key(post),
            "segments": [{"id": str(v["id"]), "uri": v["uri"]} for v in videos],
            "transitions": join_transitions(policy, body, len(videos)),
            "normalize": ({"width": int(out_cfg.get("width", 720)), "height": int(out_cfg.get("height", 1280)),
                           "fps": int(out_cfg.get("fps", 30)), "lufs": float(out_cfg.get("lufs", -14))}
                          if normalize else None),
            "timeout": {k: long_cfg.get(k) for k in ("join_timeout_s", "join_seconds_per_input_second")},
            "join": long_cfg.get("join", "cut"),
            "policy": {s: policy.get(s, default=None) for s in JOIN_SECTIONS}}


def render_join(plan: dict, policy: Policy, store, workdir) -> dict:
    """The takes joined and uploaded to plan.key: {uri, sha256, bytes, segment_seconds}. No database."""
    from pathlib import Path

    from .. import media
    d = Path(workdir)
    d.mkdir(parents=True, exist_ok=True)
    paths = [store.get_file(s["uri"], d / f"segment_{i:02d}.mp4") for i, s in enumerate(plan["segments"])]
    lengths = [media.duration(p) for p in paths]
    n = plan.get("normalize")
    if n:
        paths = [media.normalize(p, p.with_name(f"norm_{p.name}"), width=int(n["width"]), height=int(n["height"]),
                                 fps=int(n["fps"]), lufs=float(n["lufs"]), timeout=join_timeout(plan, s),
                                 policy=policy) for p, s in zip(paths, lengths)]
    seconds = [media.duration(p) for p in paths]   # where each segment starts in the joined video (captions)
    total = sum(s or 0 for s in (seconds if n else lengths))
    out = media.concat(paths, d / "joined.mp4", timeout=join_timeout(plan, total),
                       transitions=plan.get("transitions"), policy=policy)
    stored = store.put_file(out, plan["key"])
    return {"uri": stored.uri, "sha256": stored.sha256, "bytes": stored.bytes, "segment_seconds": seconds}


def record_join(conn, post: dict, plan: dict, made: dict) -> dict:
    """The joined video's asset row (once: a rerun finds it)."""
    existing = conn.execute("SELECT id, uri FROM assets WHERE uri = %s AND kind = 'video'", (made["uri"],)).fetchone()
    if existing:
        return existing
    segs = plan["segments"]
    return conn.execute(
        """INSERT INTO assets (creator_id, kind, uri, sha256, bytes, meta, recipe)
           VALUES (%s, 'video', %s, %s, %s, %s, %s) RETURNING id, uri""",
        (post["creator_id"], made["uri"], made["sha256"], made["bytes"],
         Jsonb({"lip_sync": True, "segments": [s["id"] for s in segs], "segment_seconds": made["segment_seconds"]}),
         Jsonb({"joined": [s["uri"] for s in segs], "join": plan.get("join", "cut"),
                "transitions": plan.get("transitions")}))).fetchone()


def join_talk(conn, policy: Policy, post: dict, videos: list[dict], store=None, *, body: dict | None = None) -> dict:
    """The post's segments joined into one video asset here, with each shot's transition in (join_transitions).
    Idempotent: a rerun reuses the joined asset."""
    import tempfile

    from .. import storage
    store = store or storage.from_url()
    body = body if body is not None else (post.get("body") or {})
    existing = _existing_join(conn, store, post)
    if existing:
        return existing
    plan = join_plan(policy, post, videos, body)
    with tempfile.TemporaryDirectory(prefix="mw-join-") as d:
        made = render_join(plan, policy, store, d)
    return record_join(conn, post, plan, made)


def _join_job(conn, policy: Policy, post: dict, body: dict, videos: list[dict], attempt: int):
    return queue.enqueue(conn, JOIN_KIND, join_plan(policy, post, videos, body), creator_id=post["creator_id"],
                         batch=f"post-{post['id']}", post_id=post["id"],
                         idempotency_key=f"post:{post['id']}:join:{attempt}")


def _join_state(conn, job_ids: list[str], body: dict) -> tuple[str, Any]:
    job = conn.execute("SELECT status, result, error, payload FROM jobs WHERE id = %s", (job_ids[-1],)).fetchone()
    if job["status"] in ("queued", "running"):
        return "waiting", None
    made = (job["result"] or {}).get("join")
    if job["status"] == "done" and made:
        return "chosen", (job["payload"], made)
    body["join_error"] = str(job["error"] or f"the join job ended {job['status']}")[:500]
    return "failed", None


def advance_join(conn, policy: Policy, post: dict, body: dict, videos: list[dict], store=None) -> tuple[str, Any]:
    """('chosen', the joined asset), ('waiting', None) while the finisher joins, or ('dropped', None) once
    production.max_job_failures_per_post join jobs have failed (body.drop_reason says why). Inline
    (production.finish_mode inline) the producer joins at once."""
    from .. import storage
    from .finish import finish_mode
    store = store or storage.from_url()
    existing = _existing_join(conn, store, post)
    if existing:
        return "chosen", existing
    if finish_mode(policy) != "queue":
        return "chosen", join_talk(conn, policy, post, videos, store, body=body)
    step = Step(body.setdefault("join_jobs", []), lambda j: _join_state(conn, j, body),
                lambda attempt: _join_job(conn, policy, post, body, videos, attempt),
                ("join_repairs", 0), ("join_failures", int(_cfg(policy).get("max_job_failures_per_post", 6))),
                "join")
    state, done = advance_step(body, step)
    if state != "chosen":
        return state, None
    plan, made = done
    return "chosen", record_join(conn, post, plan, made)


def _advance_long_talk(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], store=None):
    """A script longer than one segment: every segment is a talk job of its own, judged by the video gate and
    repaired on its own; `cut` renders them all from the approved photo at once, `chain` starts each from the
    last frame of the one before. Once all are chosen they are joined into the post's video."""
    chain = _talk_long_cfg(policy).get("join", "cut") == "chain"
    jobs = body.setdefault("talk_segment_jobs", [[] for _ in body["talk_segments"]])
    repairs = int(policy.get("review", "video", "max_repairs_per_post", default=2))
    failures = int(_cfg(policy).get("max_job_failures_per_post", 6))
    results: list[tuple[str, Any]] = []
    modes = body.get("talk_modes") or ["talk"] * len(body["talk_segments"])
    voice = _designed_voice(conn, post)
    voice_jobs = body.setdefault("voice_segment_jobs", [[] for _ in body["talk_segments"]]) if voice else None
    for i in range(len(body["talk_segments"])):
        # The designed voice says this segment first (every segment's line at once, even while a chain waits).
        # A segment started before the voice was designed keeps going as it began.
        voiced = voice if voice and (voice_jobs[i] or not jobs[i]) else None
        vstate, audio = _advance_voice(conn, policy, post, body, voiced, voice_jobs[i] if voiced else [],
                                       voice_lines(body, i) if voiced else [], segment=i, label=f"segment {i}: ")
        if vstate == "dropped":
            return "dropped", None
        # Walk-and-talk: the take's motion clip is made alongside its voice line. A segment started without one
        # keeps going as it began.
        motion_jobs = body.get("motion_segment_jobs")
        wants_motion = modes[i] != "narrated" and (bool(motion_jobs and motion_jobs[i])
                                                    or (not jobs[i] and _needs_motion(policy, post, body, i)))
        if vstate != "chosen" and not wants_motion:
            results.append(("waiting", None))
            continue
        if modes[i] == "narrated":   # a cutaway over the other photos (hybrid): not a lip-sync render
            step = Step(jobs[i], lambda j: _edit_state(conn, j),
                        lambda attempt, i=i, a=audio: _cutaway_job(conn, policy, post, body, chosen, attempt,
                                                                   segment=i, audio=a),
                        (f"segment_repairs_{i}", repairs), ("follow_up_failures", failures), "edit", f"segment {i}: ")
            results.append(advance_step(body, step))
            if results[-1][0] == "dropped":
                return "dropped", None
            continue
        # A directed post chains only within one shot (a take split for length): a new shot is a new scene.
        follows = chain and i > 0 and modes[i - 1] == "talk" and _take_shot(body, i) == _take_shot(body, i - 1)
        if follows and results[-1][0] != "chosen":
            results.append(("waiting", None))   # starts from the previous segment's last frame
            continue
        start_uri = _last_frame(conn, results[-1][1]) if follows else None
        mstate, motion = "chosen", None
        if wants_motion:
            motion_jobs = body.setdefault("motion_segment_jobs", [[] for _ in body["talk_segments"]])
            mstate, motion = _advance_motion(conn, policy, post, body, chosen, motion_jobs[i], segment=i,
                                             start_uri=start_uri, audio=audio, label=f"segment {i}: ", store=store)
            if mstate == "dropped":
                return "dropped", None
        if vstate != "chosen" or mstate != "chosen":
            results.append(("waiting", None))   # the take needs its voice line and its motion clip
            continue
        step = Step(jobs[i], lambda j: _clip_state(conn, j),
                    lambda attempt, i=i, s=start_uri, a=audio, m=motion: _talk_job(
                        conn, policy, post, body, chosen, attempt, segment=i, start_uri=s, audio=a, motion=m),
                    (f"segment_repairs_{i}", repairs), ("follow_up_failures", failures), "clip", f"segment {i}: ")
        results.append(advance_step(body, step))
        if results[-1][0] == "dropped":
            return "dropped", None
    if any(state != "chosen" for state, _ in results):
        return "waiting", None
    return advance_join(conn, policy, post, body, [video for _, video in results], store)
