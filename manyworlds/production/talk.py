"""Talking video: the lip-synced talk job (from a photo, or V2V over a motion clip), the hybrid's cutaways, and a
long script's segments, each its own job, joined with each shot's transition in."""

from __future__ import annotations

from typing import Any

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
    repair = talk_repair(conn, policy, [str(j) for j in prior], params, v2v=bool(motion))
    script = unmarked(_take_script(body, segment))
    seed = _seed(post, 700 + (segment or 0) * 1000, attempt)
    payload = {"image_uri": start_uri or _take_photo(body, chosen, i)["uri"], "script": script,
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
    """A take's earlier talk jobs, oldest first: {id, start_step, repair, passed, corr} (passed and corr from the
    lip_sync review row of its video; None when it has none: a failed job, or the gate not enforced)."""
    if not job_ids:
        return []
    rows = {str(r["id"]): r for r in conn.execute(
        """SELECT j.id, j.payload->'params'->>'start_step' AS start_step, j.payload->'repair' AS repair,
                  (SELECT r.passed FROM assets a JOIN reviews r ON r.asset_id = a.id
                    WHERE a.job_id = j.id AND a.kind = 'video' AND r.gate = 'lip_sync' ORDER BY r.created_at DESC
                    LIMIT 1) AS passed,
                  (SELECT r.score FROM assets a JOIN reviews r ON r.asset_id = a.id
                    WHERE a.job_id = j.id AND a.kind = 'video' AND r.gate = 'lip_sync' ORDER BY r.created_at DESC
                    LIMIT 1) AS corr
           FROM jobs j WHERE j.id = ANY(%s::uuid[])""", (job_ids,)).fetchall()}
    return [dict(rows[j]) for j in job_ids if j in rows]


def talk_repair(conn, policy: Policy, prior: list[str], params: dict, *, v2v: bool) -> dict | None:
    """What a take's next talk job changes, given its earlier jobs (`prior`, oldest first); sets params.start_step
    and returns the record kept in the job's payload.repair: {cause, knob, start_step, dead_knobs}. None for a first
    take.

    Breadth run 2026-11-23: V2V takes (walk-and-talk) failed lip_sync with the mouth never redrawn, and every repair
    changed only the seed and scored the same. A lip_sync failure on a V2V take now redraws more: start_step one lower
    per repair (generation.talk_motion.v2v_repair.start_step_step) down to start_step_floor; an earlier escalation
    is kept by later repairs. A knob whose repair scored the same lip sync as the attempt before
    (within same_score_tolerance) is dead: it is not tried again (dead_knobs says why), and when every knob is dead
    the post is dropped with the reason (lip sync is a media defect) instead of rendering the same clip again."""
    hist = _take_history(conn, prior)
    if not hist:
        return None
    cfg = _motion_cfg(policy).get("v2v_repair") or {}
    last = hist[-1]
    record_before = last["repair"] or {}
    dead = dict(record_before.get("dead_knobs") or {})
    start = last["start_step"] if last["start_step"] is not None else params.get("start_step")
    v2v = v2v and start is not None
    if v2v:
        params["start_step"] = int(start)
    out = {"cause": None, "knob": "seed", "start_step": params.get("start_step"), "dead_knobs": dead}
    if last["passed"] is not False:      # rejected by another gate, or the job failed: a new seed
        return out
    out["cause"] = "lip_sync"
    tol = float(cfg.get("same_score_tolerance", 0.0))
    if len(hist) >= 2 and hist[-2]["passed"] is False and last["corr"] is not None and hist[-2]["corr"] is not None \
            and abs(float(last["corr"]) - float(hist[-2]["corr"])) <= tol:
        knob = str(record_before.get("knob") or "seed")
        dead[knob] = (f"attempt {len(hist)} changed {knob} and scored lip sync {float(last['corr']):.3f}, as attempt "
                      f"{len(hist) - 1} did ({float(hist[-2]['corr']):.3f})")
    for knob in (["start_step"] if v2v else []) + ["seed"]:
        if knob in dead:
            continue
        if knob == "start_step":
            floor, step = int(cfg.get("start_step_floor", 0)), int(cfg.get("start_step_step", 1))
            if int(start) <= floor:
                continue
            params["start_step"] = max(floor, int(start) - step)
        return {**out, "knob": knob, "start_step": params.get("start_step")}
    raise DropPost("lip sync failed and no repair changes it: " + "; ".join(f"{k}: {v}" for k, v in dead.items()))


def _cutaway_job(conn, policy: Policy, post: dict, body: dict, chosen: list[dict], attempt: int, *, segment: int,
                 audio: dict | None = None):
    """A cutaway segment of a hybrid talking video: the post's other photos, moving gently, under the creator's
    voice saying this part of the script (the narrated-edit worker; no lip-sync render). With `audio`, the
    designed voice's finished line instead of Kokoro."""
    anchors = conn.execute("SELECT anchors FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["anchors"]
    out = _hybrid_cfg(policy).get("output", {}) or {}
    if body.get("talk_shots") and _photo_per_shot(body, chosen):   # a directed cutaway: its own shot's photo
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


def join_talk(conn, policy: Policy, post: dict, videos: list[dict], store=None, *, body: dict | None = None) -> dict:
    """The post's segments joined into one video asset, with each shot's transition in (join_transitions).
    Idempotent: a rerun reuses the joined asset."""
    import tempfile
    from pathlib import Path

    from .. import media, storage
    store = store or storage.from_url()
    body = body if body is not None else (post.get("body") or {})
    key = f"assets/{post['creator_id']}/post-{post['id']}/talk-joined.mp4"
    uri = store.uri_for(key)
    existing = conn.execute("SELECT id, uri FROM assets WHERE uri = %s AND kind = 'video'", (uri,)).fetchone()
    if existing:
        return existing
    timeout = float(_talk_long_cfg(policy).get("join_timeout_s", 600))
    transitions = join_transitions(policy, body, len(videos))
    with tempfile.TemporaryDirectory(prefix="mw-join-") as d:
        paths = [store.get_file(v["uri"], Path(d) / f"segment_{i:02d}.mp4") for i, v in enumerate(videos)]
        if body.get("talk_modes") or _hybrid_cfg(policy).get("normalize_all"):
            out_cfg = _hybrid_cfg(policy).get("output", {}) or {}
            paths = [media.normalize(p, p.with_name(f"norm_{p.name}"), width=int(out_cfg.get("width", 720)),
                                     height=int(out_cfg.get("height", 1280)), fps=int(out_cfg.get("fps", 30)),
                                     lufs=float(out_cfg.get("lufs", -14)), timeout=timeout, policy=policy)
                     for p in paths]
        seconds = [media.duration(p) for p in paths]   # where each segment starts in the joined video (captions)
        out = media.concat(paths, Path(d) / "joined.mp4", timeout=timeout, transitions=transitions, policy=policy)
        stored = store.put_file(out, key)
    return conn.execute(
        """INSERT INTO assets (creator_id, kind, uri, sha256, bytes, meta, recipe)
           VALUES (%s, 'video', %s, %s, %s, %s, %s) RETURNING id, uri""",
        (post["creator_id"], stored.uri, stored.sha256, stored.bytes,
         Jsonb({"lip_sync": True, "segments": [str(v["id"]) for v in videos], "segment_seconds": seconds}),
         Jsonb({"joined": [v["uri"] for v in videos], "join": _talk_long_cfg(policy).get("join", "cut"),
                "transitions": transitions}))).fetchone()


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
    return "chosen", join_talk(conn, policy, post, [video for _, video in results], store, body=body)
