"""The finish stage: the AI upscale (a GPU Step), then one pass of sound mix, grade, grain, loudness and burned-in
captions and on-screen text over a made video, the cover frame, and counting a finish that keeps failing."""

from __future__ import annotations

from typing import Any, Callable

from psycopg.types.json import Jsonb

from .. import queue
from ..policy import Policy
from .captioning import caption_plan
from .common import FOLLOW_UP_FORMATS, SPOKEN, _cfg, _pieces, _shot, log
from .mixing import _mix_sound, sound_plan
from .steps import Step, _edit_state, advance_step
from .takes import _take_index


# --- finish: one pass of grade, grain, loudness and burned-in captions over a made video ------------------------
# Once a video post's video exists (joined, for a long talking video), it is finished on the control plane before
# the post is ready: manyworlds.finishing (look, grain, two-pass loudness) and manyworlds.captions (the spoken words,
# timed from the voice lines). With libass in this ffmpeg the captions' ASS rides in finishing's own filter chain
# (after the grain, so text gets none): one re-encode. Without it, finishing then the caption overlay: two. A rerun
# reuses post-<id>/final.mp4. A finish that keeps failing (finishing.max_failures) ships the unfinished video and
# tells the owner.
def _switched_on(policy: Policy, block: str, fmt: str) -> bool:
    cfg = policy.get(block, default={}) or {}
    formats = cfg.get("formats")
    return bool(cfg.get("enabled")) and fmt in FOLLOW_UP_FORMATS and (formats is None or fmt in formats)


def finishes(policy: Policy, fmt: str) -> bool:
    """Whether a video post of this format is graded, grained and loudness-normalised (finishing.enabled, .formats)."""
    return _switched_on(policy, "finishing", fmt)


def captions_burned(policy: Policy, fmt: str) -> bool:
    """Whether a video post of this format gets burned-in captions (burned_captions.enabled, .formats)."""
    return _switched_on(policy, "burned_captions", fmt)


def _piece_keyframes(conn, store, body: dict, pieces: list[dict], workdir) -> dict:
    """The pieces' keyframes (else the post's chosen photos), fetched: {local path: {id, uri}}."""
    from pathlib import Path
    job_ids = [str(p["job"]["id"]) for p in pieces if p["job"]]
    frames = conn.execute("SELECT id, uri FROM assets WHERE job_id = ANY(%s::uuid[]) AND kind = 'keyframe' ORDER BY uri",
                          (job_ids,)).fetchall() if job_ids else []
    frames = frames or [{"id": im.get("asset_id"), "uri": im["uri"]} for im in body.get("images") or []]
    local = {}
    for i, f in enumerate(frames):
        local[store.get_file(f["uri"], Path(workdir) / f"frame_{i:03d}{Path(f['uri']).suffix or '.jpg'}")] = f
    return local


def _cover(conn, policy: Policy, store, body: dict, pieces: list[dict], workdir) -> dict | None:
    """The cover frame (finishing.pick_cover) among the pieces' keyframes, else the post's chosen photos: the fallback
    when the finished file's own frames can't be read (_final_cover)."""
    from .. import finishing
    local = _piece_keyframes(conn, store, body, pieces, workdir)
    if not local:
        return None
    best = local[finishing.pick_cover(list(local), policy=policy)]
    return {"asset_id": str(best["id"]) if best["id"] else None, "uri": best["uri"]}


def text_windows(plan, ovs: list) -> list[tuple[float, float]]:
    """(start, end) of every caption chunk and on-screen text: the times a cover frame would show text."""
    return [(float(c.start), float(c.end)) for c in (plan.chunks if plan is not None else ())] + \
           [(float(o.start), float(o.end)) for o in ovs]


def _final_cover(conn, policy: Policy, store, post: dict, final, workdir, *,
                 avoid: list[tuple[float, float]] | None = None) -> dict:
    """The cover picked from the finished video's own frames (finishing.cover_from_video, as pick_cover on a video),
    so it shows the post as published (graded, upscaled), passing over frames with text on screen (`avoid`); stored as
    post-<id>/cover.jpg. Raises when the frames can't be read."""
    from pathlib import Path

    from .. import finishing
    d = Path(workdir) / "cover"
    d.mkdir(parents=True, exist_ok=True)
    picked = finishing.cover_from_video(Path(final), workdir=d, policy=policy, avoid=avoid)
    stored = store.put_file(Path(picked["frame"]), f"assets/{post['creator_id']}/post-{post['id']}/cover.jpg")
    meta = Jsonb({"cover": True, "t": picked.get("t")})
    row = conn.execute("UPDATE assets SET sha256 = %s, bytes = %s, meta = %s WHERE uri = %s AND kind = 'keyframe' "
                       "AND job_id IS NULL RETURNING id, uri", (stored.sha256, stored.bytes, meta, stored.uri)).fetchone()
    row = row or conn.execute(   # (a rerun of a finish that failed after its cover updates the same asset)
        """INSERT INTO assets (creator_id, kind, uri, sha256, bytes, meta, recipe)
           VALUES (%s, 'keyframe', %s, %s, %s, %s, %s) RETURNING id, uri""",
        (post["creator_id"], stored.uri, stored.sha256, stored.bytes, meta,
         Jsonb({"cover_of": f"post-{post['id']}/final.mp4"}))).fetchone()
    return {"asset_id": str(row["id"]), "uri": row["uri"]}


def face_box(frames: Callable[[], Any], *, face=None) -> tuple | None:
    """One box round the faces in the video's keyframes (`frames()`: their local paths, fetched only when a face
    reader is here; `face`: finishing.mediapipe_face), as captions.face_box_union, so captions and labels stay off
    the face wherever it moves. None when this host has no MediaPipe (the control plane today) or no frame shows one."""
    import importlib.util

    from .. import captions, finishing
    if face is None:
        if importlib.util.find_spec("mediapipe") is None:
            return None
        face = finishing.mediapipe_face
    return captions.face_box_union((face(p) or {}).get("box") for p in frames())


# On-screen text (manyworlds.overlays): each shot's edit.on_screen_text items (and the opening shot's edit.hook as a
# hook item), timed in the finished video from where that shot plays in it.
def shot_spans(conn, post: dict, body: dict, video: dict, duration: float | None) -> list[tuple[int, float, float | None]]:
    """(shot index, start_s, end_s) for each shot that plays in the made video. A joined talking video: each take's
    shot over its segments (the join's measured lengths, else each piece's own). One piece: its take's shot over the
    whole (a story clip's shot 0); a narrated edit's shots share it evenly, one photo each."""
    shots = body.get("shots") or []
    if not shots:
        return []
    row, pieces = _pieces(conn, video)
    measured = (row["meta"] or {}).get("segment_seconds") or []
    if len(body.get("talk_segments") or []) == len(pieces) > 1:
        spans: dict[int, list] = {}
        t = 0.0
        for i, piece in enumerate(pieces):
            length = measured[i] if i < len(measured) and measured[i] is not None else (piece["meta"] or {}).get("seconds")
            end = None if length is None else t + float(length)
            k = _take_index(body, i)
            if k in spans:
                spans[k][1] = end
            else:
                spans[k] = [t, end]
            if end is None:
                break   # later shots can't be placed
            t = end
        return [(k, a, b) for k, (a, b) in spans.items()]
    if post["format"] == "narrated_edit" and len(shots) > 1 and duration:
        step = float(duration) / len(shots)
        return [(i, i * step, (i + 1) * step) for i in range(len(shots))]
    return [(_take_index(body, None) if body.get("scenes") else 0, 0.0, duration)]


def overlay_items(conn, post: dict, body: dict, video: dict, duration: float | None) -> list[dict]:
    """The post's on-screen text items in the finished video's time (overlays.items_for_shot per shot). Only
    `edit.on_screen_text` is drawn: `edit.hook` describes what grabs attention (often a visual note), and the
    director writes hook TEXT as a hook item."""
    from .. import overlays
    items: list[dict] = []
    for k, a, b in shot_spans(conn, post, body, video, duration):
        edit = (_shot(body, k) or {}).get("edit") or {}
        items += overlays.items_for_shot(edit, a, b)
    return items


def _burn_text(video, out, workdir, plan, ovs: list, policy: Policy):
    """Captions and on-screen texts burned over `video` in one pass (no libass: overlays.burn draws both; captions
    alone: captions.burn)."""
    from .. import captions, overlays
    if ovs:
        return overlays.burn(video, out, workdir=workdir, plan=plan, overlays=ovs, policy=policy)
    return captions.burn(video, plan, out, workdir=workdir, policy=policy)


def _text_fonts(plan, ovs: list, policy: Policy):
    """The fonts directory of the text script, after checking every style's font is there (text in a fallback font
    is worse than none)."""
    from .. import captions
    fonts = plan.fonts_dir if plan is not None else (ovs[0].fonts_dir if ovs else captions.fonts_dir(policy))
    for style in ([plan.style] if plan is not None else []) + [ov.style for ov in ovs]:
        if fonts is None or not (fonts / str(style.get("file", ""))).exists():
            raise FileNotFoundError(f"text font {style.get('file')!r} not found ({fonts})")
    return fonts


def join_times(conn, video: dict) -> list[float]:
    """Where the joins are in a joined video (seconds): after each segment but the last, by the join's measured
    lengths; [] for a single piece, or when a length is unknown from there on."""
    row = conn.execute("SELECT meta FROM assets WHERE id = %s", (video["id"],)).fetchone()
    out, t = [], 0.0
    for s in ((row["meta"] or {}).get("segment_seconds") or [])[:-1] if row else []:
        if s is None:
            break
        t += float(s)
        out.append(round(t, 3))
    return out


def text_spans(plan, ovs: list) -> list[dict]:
    """The text on screen for the final review: each caption chunk (kind caption) and on-screen text (hook, else
    label), {text, start_s, end_s, kind}."""
    spans = [{"text": " ".join(w.text for w in c.words), "start_s": float(c.start), "end_s": float(c.end),
              "kind": "caption"} for c in (plan.chunks if plan is not None else ())]
    return spans + [{"text": o.text, "start_s": float(o.start), "end_s": float(o.end),
                     "kind": "hook" if o.kind == "hook" else "label"} for o in ovs]


def finish_video(conn, policy: Policy, post: dict, body: dict, video: dict, store=None, *,
                 source: dict | None = None, llm=None) -> dict:
    """The post's finished video asset (or `video` itself when nothing applies to its format). `source`: the file to
    finish when it is not `video` itself (the AI upscaler's output; captions and shots are still timed from `video`'s
    pieces). With `llm` (the judge), the finished file gets the final review once (manyworlds.review.final), its
    result in body.final_review. Idempotent: the finished file is assets/<creator>/post-<id>/final.mp4, and a rerun
    reuses its asset.

    In order: the sound mix (sound_plan, its own pass; a mix that fails is recorded and the video goes on unmixed),
    then the captions plan (style, word timings, placement off the face) and the shots' on-screen texts
    (overlay_items -> overlays.overlay_events) in ONE ASS script (overlays.write_script) that rides in finishing's
    filter chain where this ffmpeg has libass, else one Pillow burn pass of both; the cover from the finished file."""
    import dataclasses
    import tempfile
    from pathlib import Path

    from .. import captions, finishing, media, overlays, storage
    fmt = post["format"]
    grade, burn = finishes(policy, fmt), captions_burned(policy, fmt)
    if not (grade or burn):
        return video
    store = store or storage.from_url()
    key = f"assets/{post['creator_id']}/post-{post['id']}/final.mp4"
    existing = conn.execute("SELECT id, uri, meta FROM assets WHERE uri = %s AND kind = 'video'",
                            (store.uri_for(key),)).fetchone()
    if existing:
        return existing
    _, pieces = _pieces(conn, video)
    look, sound_error = None, None
    with tempfile.TemporaryDirectory(prefix="mw-finish-") as tmp:
        d = Path(tmp)
        raw = store.get_file((source or video)["uri"], d / "raw.mp4")
        trim_s = float(finishing.cfg(policy).get("trim_head_s", 0) or 0)
        if trim_s > 0 and fmt not in SPOKEN:
            # An image-to-video clip opens on its still start photo for a beat: the first frame must already move.
            try:
                raw = media.trim_start(raw, d / "trimmed.mp4",
                                       frames=round(trim_s * float(media.probe(raw).get("fps") or 16)))
            except Exception as e:  # noqa: BLE001 - an untrimmed clip beats no clip
                log.warning("could not trim the opening of post %s: %s", post["id"], str(e)[:200])
        cues, used, mixed, passes = sound_plan(conn, policy, post, body, video), None, False, 0
        if cues:   # room tone, polish, effects, music under the voice (policy sound): its own pass, before the finish
            try:
                raw, used = _mix_sound(policy, store, cues, raw, d)
                mixed = True
                passes += 1
            except Exception as e:  # noqa: BLE001 - the video ships unmixed rather than not at all; the error is kept
                sound_error = f"{type(e).__name__}: {e}"[:500]
        plan, ovs, width, height = None, [], None, None
        if burn or body.get("shots"):
            info = media.probe(raw)
            if not (info["width"] and info["height"]):
                raise RuntimeError(f"can't read the frame size of {video['uri']}")
            # The finish stage upscales a smaller video to finishing.output first, so its text is laid out at that size.
            width, height = (finishing.finished_size(info["width"], info["height"], policy) if grade
                             else (info["width"], info["height"]))
            items = overlay_items(conn, post, body, video, info.get("seconds"))
            face = (face_box(lambda: list(_piece_keyframes(conn, store, body, pieces, d / "faces")))
                    if burn or items else None)
            if burn:
                plan = caption_plan(conn, policy, post, body, video, width=width, height=height, store=store,
                                    face_box=face)
            if items:
                ovs = overlays.overlay_events(items, (width, height), policy, duration=info.get("seconds"),
                                              face_box=face,   # labels stay off the burned captions too
                                              avoid_bands=[captions.plan_band(plan)] if plan else ())
        text = plan is not None or bool(ovs)
        if not grade and not text and not mixed:
            return video   # captions only, and nothing is said or shown
        out, cover = d / "final.mp4", None
        if grade:
            handle = conn.execute("SELECT handle FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["handle"]
            shot = _shot(body, _take_index(body, 0 if body.get("talk_segments") else None)) or {}
            look = finishing.look_for(policy, creator=handle, set_id=(shot.get("set") or {}).get("set_id"))
            one_pass = text and media.has_libass()
            ass, fpolicy = None, policy
            if one_pass:   # captions and labels ride in finishing's filter chain, after the grain: one script
                fonts = _text_fonts(plan, ovs, policy)
                ass = overlays.write_script(d / "text.ass", plan, ovs, policy, size=(width, height))
                fpolicy = dataclasses.replace(policy, body={
                    **policy.body, "finishing": {**finishing.cfg(policy), "fonts_dir": str(fonts)}})
            # A format nobody speaks in (a story clip) is levelled as a bed, not loudnormed to the voice target.
            graded = finishing.finish(raw, out if not text or one_pass else d / "graded.mp4", look=look,
                                      ass=ass, has_voice=fmt in SPOKEN, policy=fpolicy)
            passes += 1
            if text and not one_pass:   # no libass here: the Pillow drawings, a second pass
                _burn_text(graded, out, d / "captions", plan, ovs, policy)
                passes += 1
            try:
                cover = _final_cover(conn, policy, store, post, out, d, avoid=text_windows(plan, ovs))
            except Exception:  # noqa: BLE001 - unreadable frames: the pieces' keyframes, else the photos
                try:
                    cover = _cover(conn, policy, store, body, pieces, d)
                except Exception:  # noqa: BLE001 - a post without a picked cover still ships (the publisher has a fallback)
                    cover = None
        elif text:
            _burn_text(raw, out, d / "captions", plan, ovs, policy)
            passes += 1
        else:   # captions only and nothing said, but the sound was mixed
            import shutil
            shutil.copyfile(raw, out)
        stored = store.put_file(out, key)
        meta = {"finished": grade, "look": look, "captions": plan.style_name if plan else None,
                "placement": plan.placement if plan else None, "passes": passes, "raw": str(video["id"]), "cover": cover,
                **({"sound": used} if used else {}), **({"sound_error": sound_error} if sound_error else {}),
                **({"on_screen_text": [{"kind": o.kind, "text": o.text, "position": o.position} for o in ovs]}
                   if ovs else {}),
                **({"sound_design": {"room_tone": getattr(cues.get("room_tone_synth"), "preset", None),
                                     "room_size": cues.get("room_size"), "mic": cues.get("mic")}} if mixed else {})}
        row = conn.execute(
            """INSERT INTO assets (creator_id, kind, uri, sha256, bytes, meta, recipe)
               VALUES (%s, 'video', %s, %s, %s, %s, %s) RETURNING id, uri, meta""",
            (post["creator_id"], stored.uri, stored.sha256, stored.bytes, Jsonb(meta),
             Jsonb({"finished_from": (source or video)["uri"], "policy_version": policy.version}))).fetchone()
        # judged after the asset exists, so its final_* review rows land on it (review/final.py _rows)
        if llm is not None:   # the finished file judged once (review.final; body.final_review, updated in place)
            from ..review.final import review_final
            try:
                review_final(conn, policy, llm, post, out, joins=join_times(conn, video),
                             text_spans=text_spans(plan, ovs))
            except Exception as e:  # noqa: BLE001 - a judge outage never holds a finished post; the error is kept
                body["final_review"] = {"error": f"{type(e).__name__}: {e}"[:500], "passed": True}
    return row


# The AI upscaler (finishing.upscaler, off until benchmarked): when it names a job kind, a made video is first upscaled
# on a GPU worker (a Step of its own, not reviewed: done with a video is enough), and the finish works on that file.
# One that keeps failing (production.max_job_failures_per_post) is given up: the finish pass's lanczos scales instead.
def _upscale_job(conn, policy: Policy, post: dict, video: dict, spec: dict, attempt: int):
    payload = {"video_uri": video["uri"], "model": spec.get("model"), "scale": spec.get("scale"),
               "params": {k: v for k, v in spec.items() if k not in ("enabled", "kind", "model", "scale")},
               "intermediate": True}   # the finish's source, not a post's output
    return queue.enqueue(conn, str(spec["kind"]), payload, creator_id=post["creator_id"], batch=f"post-{post['id']}",
                         post_id=post["id"], idempotency_key=f"post:{post['id']}:upscale:{attempt}")


def _advance_upscale(conn, policy: Policy, post: dict, body: dict, video: dict) -> tuple[str, dict | None]:
    """('chosen', the upscaled video) once made, ('waiting', None) while it is, ('chosen', None) when there is nothing
    to upscale (the upscaler off, a format not finished) or it was given up."""
    from .. import finishing
    spec = finishing.upscaler(policy)
    if not spec or not finishes(policy, post["format"]) or body.get("upscale_skipped"):
        return "chosen", None
    step = Step(body.setdefault("upscale_jobs", []), lambda j: _edit_state(conn, j),
                lambda attempt: _upscale_job(conn, policy, post, video, spec, attempt),
                ("upscale_repairs", 0), ("follow_up_failures", int(_cfg(policy).get("max_job_failures_per_post", 6))),
                "upscale")
    state, up = advance_step(body, step)
    if state == "dropped":
        body["upscale_skipped"] = body.pop("drop_reason", "the upscale failed")
        return "chosen", None
    return state, up


def _finish_failed(conn, policy: Policy, post: dict, body: dict, error: Exception) -> bool:
    """Count a failed finish. True once finishing.max_failures is used up: the post then ships its unfinished video
    and the owner hears of it; until then it waits for the next tick."""
    from .. import notify
    body["finish_failures"] = int(body.get("finish_failures", 0)) + 1
    body["finish_error"] = f"{type(error).__name__}: {error}"[:500]
    if body["finish_failures"] < int(policy.get("finishing", "max_failures", default=3)):
        return False
    body["finish_skipped"] = body["finish_error"]
    notify.owner_event(conn, "finish_failed", subject={"post_id": str(post["id"])},
                       payload={"format": post["format"], "error": body["finish_error"],
                                "failures": body["finish_failures"]},
                       creator_id=post["creator_id"], dedupe_key=f"finish_failed:{post['id']}")
    return True
