"""The finish stage: the AI upscale (a GPU Step), then one pass of sound mix, grade, grain, loudness and burned-in
captions and on-screen text over a made video, the cover frame, and counting a finish that keeps failing.

The finish is three halves, so the heavy one can leave the control plane (policy production.finish_mode):
  finish_plan    the database half: everything the finish reads about the post (its pieces, keyframes, handle, caption
                 style and timed words), JSON-ready; it is the finish.video job's payload
  render_finish  the ffmpeg half, no database: trim, mix, grade, captions and on-screen text, the cover frame; uploads
                 post-<id>/final.mp4 and post-<id>/cover.jpg (fixed keys: a rerun overwrites them)
  record_finish  the database half again: the cover and finished video's asset rows, and the final review
`queue` (shipped): render_finish runs on the finisher (worker/handlers/finish_video.py, a Fly machine the autoscaler
starts) and production waits for its job like any other Step (_advance_finish). `inline`: finish_video runs all three
in the producer, as before."""

from __future__ import annotations

import json
from typing import Any, Callable

from psycopg.types.json import Jsonb

from .. import queue
from ..policy import Policy
from .captioning import caption_style, caption_words, captions_from_words
from .common import FOLLOW_UP_FORMATS, SPOKEN, _cfg, _pieces, _shot, log
from .mixing import _mix_sound, _sound_plan_from
from .steps import Step, _edit_state, advance_step
from .takes import _take_index

FINISH_KIND = "finish.video"   # the finisher's job kind (worker/handlers/finish_video.py)
# The policy sections the finish reads, carried in its payload: the finisher renders under the producer's settings
# (runtime overrides and test policies included), whatever release its own image is.
FINISH_SECTIONS = ("finishing", "burned_captions", "on_screen_text", "sound")


# --- finish: one pass of grade, grain, loudness and burned-in captions over a made video ------------------------
# Once a video post's video exists (joined, for a long talking video), it is finished before the post is ready:
# manyworlds.finishing (look, grain, two-pass loudness) and manyworlds.captions (the spoken words, timed from the
# voice lines). With libass in this ffmpeg the captions' ASS rides in finishing's own filter chain (after the grain,
# so text gets none): one re-encode. Without it, finishing then the caption overlay: two. A rerun reuses
# post-<id>/final.mp4. A finish that keeps failing (finishing.max_failures) ships the unfinished video and tells the
# owner.
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


def finish_mode(policy: Policy) -> str:
    """Where the finish renders (production.finish_mode): 'queue' (a finish.video job on the finisher) or 'inline'
    (in the producer's own tick)."""
    return str(_cfg(policy).get("finish_mode") or "inline")


def _final_key(post: dict) -> str:
    return f"assets/{post['creator_id']}/post-{post['id']}/final.mp4"


def _existing_final(conn, store, post: dict) -> dict | None:
    """The post's finished video asset when a finish already recorded it (a rerun reuses it)."""
    return conn.execute("SELECT id, uri, meta FROM assets WHERE uri = %s AND kind = 'video'",
                        (store.uri_for(_final_key(post)),)).fetchone()


def _fetch_frames(store, frames: list[dict], workdir) -> dict:
    """These keyframes ({id, uri}), fetched: {local path: {id, uri}}."""
    from pathlib import Path
    local = {}
    for i, f in enumerate(frames):
        local[store.get_file(f["uri"], Path(workdir) / f"frame_{i:03d}{Path(f['uri']).suffix or '.jpg'}")] = f
    return local


def _keyframe_cover(policy: Policy, store, frames: list[dict], workdir) -> dict | None:
    """The cover frame (finishing.pick_cover) among the pieces' keyframes, else the post's chosen photos (`frames`):
    the fallback when the finished file's own frames can't be read."""
    from .. import finishing
    local = _fetch_frames(store, frames, workdir)
    if not local:
        return None
    best = local[finishing.pick_cover(list(local), policy=policy)]
    return {"asset_id": str(best["id"]) if best["id"] else None, "uri": best["uri"]}


def text_windows(plan, ovs: list) -> list[tuple[float, float]]:
    """(start, end) of every caption chunk and on-screen text: the times a cover frame would show text."""
    return [(float(c.start), float(c.end)) for c in (plan.chunks if plan is not None else ())] + \
           [(float(o.start), float(o.end)) for o in ovs]


def _cover_upload(policy: Policy, store, final, key: str, workdir, *,
                  avoid: list[tuple[float, float]] | None = None) -> dict:
    """The cover picked from the finished video's own frames (finishing.cover_from_video, as pick_cover on a video),
    so it shows the post as published (graded, upscaled), passing over frames with text on screen (`avoid`); uploaded
    to `key` (post-<id>/cover.jpg). Raises when the frames can't be read. Its asset row is record_finish's."""
    from pathlib import Path

    from .. import finishing
    d = Path(workdir) / "cover"
    d.mkdir(parents=True, exist_ok=True)
    picked = finishing.cover_from_video(Path(final), workdir=d, policy=policy, avoid=avoid)
    stored = store.put_file(Path(picked["frame"]), key)
    return {"upload": {"uri": stored.uri, "sha256": stored.sha256, "bytes": stored.bytes, "t": picked.get("t")}}


def _cover_row(conn, post: dict, up: dict) -> dict:
    """The uploaded cover's keyframe asset (updated in place on a rerun): {asset_id, uri}."""
    meta = Jsonb({"cover": True, "t": up.get("t")})
    row = conn.execute("UPDATE assets SET sha256 = %s, bytes = %s, meta = %s WHERE uri = %s AND kind = 'keyframe' "
                       "AND job_id IS NULL RETURNING id, uri", (up["sha256"], up["bytes"], meta, up["uri"])).fetchone()
    row = row or conn.execute(   # (a rerun of a finish that failed after its cover updates the same asset)
        """INSERT INTO assets (creator_id, kind, uri, sha256, bytes, meta, recipe)
           VALUES (%s, 'keyframe', %s, %s, %s, %s, %s) RETURNING id, uri""",
        (post["creator_id"], up["uri"], up["sha256"], up["bytes"], meta,
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
    whole (a story clip's shot 0); a narrated edit's shots where their photos really play (narrated_spans), evenly
    shared only when the edit recorded no spans (older posts)."""
    if not body.get("shots"):
        return []
    row, pieces = _pieces(conn, video)
    return _spans_from(post["format"], body, row["meta"] or {}, pieces, duration)


def narrated_spans(body: dict, meta: dict, n: int, duration: float | None) -> list[tuple[int, float, float]] | None:
    """A narrated edit's shots over its photos' real spans: body.photo_spans (production keeps the chosen edit's),
    else the video's own meta `photo_spans` (worker.handlers.edit_narrated), one photo per shot in order, each end
    held inside `duration`. None when neither is there or they don't hold one span per shot."""
    spans = body.get("photo_spans") or (meta or {}).get("photo_spans")
    if not isinstance(spans, list) or len(spans) != n:
        return None
    out = []
    try:
        for i, s in enumerate(spans):
            a, b = float(s["start_s"]), float(s["end_s"])
            if duration:
                a, b = min(a, float(duration)), min(b, float(duration))
            out.append((i, a, b))
    except (KeyError, TypeError, ValueError):   # a malformed record: the even share instead
        return None
    return out


def _spans_from(fmt: str, body: dict, meta: dict, pieces: list[dict],
                duration: float | None) -> list[tuple[int, float, float | None]]:
    """shot_spans from the made video's asset meta and its pieces (no database: the finisher runs it too)."""
    shots = body.get("shots") or []
    if not shots:
        return []
    measured = (meta or {}).get("segment_seconds") or []
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
    real = narrated_spans(body, meta, len(shots), duration) if fmt == "narrated_edit" and len(shots) > 1 else None
    if real:
        return real
    if fmt == "narrated_edit" and len(shots) > 1 and duration:   # an older edit without photo_spans: an even share
        step = float(duration) / len(shots)
        return [(i, i * step, (i + 1) * step) for i in range(len(shots))]
    return [(_take_index(body, None) if body.get("scenes") else 0, 0.0, duration)]


def overlay_items(conn, post: dict, body: dict, video: dict, duration: float | None) -> list[dict]:
    """The post's on-screen text items in the finished video's time (overlays.items_for_shot per shot). Only
    `edit.on_screen_text` is drawn: `edit.hook` describes what grabs attention (often a visual note), and the
    director writes hook TEXT as a hook item."""
    if not body.get("shots"):
        return []
    row, pieces = _pieces(conn, video)
    return _overlay_items_from(post["format"], body, row["meta"] or {}, pieces, duration)


def _overlay_items_from(fmt: str, body: dict, meta: dict, pieces: list[dict], duration: float | None) -> list[dict]:
    """overlay_items from the made video's asset meta and its pieces (no database)."""
    from .. import overlays
    items: list[dict] = []
    for k, a, b in _spans_from(fmt, body, meta, pieces, duration):
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


def _jsonable(value):
    """`value` as JSON has it (ids and times as text): what a job payload or result holds."""
    return json.loads(json.dumps(value, default=str))


def finish_plan(conn, policy: Policy, post: dict, body: dict, video: dict, store=None, *,
                source: dict | None = None) -> dict:
    """Everything the finish reads from the database, JSON-ready: the finish.video payload (render_finish's input).
    The post, its body, the made video and the file to finish (`source`: the AI upscaler's output, else the video),
    the video's pieces (each with its job) and keyframes (else the chosen photos), the creator's handle (the look),
    the caption style and the spoken words already timed (caption_words: word timestamps are transcribed and cached
    here, on the control plane), the output keys, and the finish's policy sections (FINISH_SECTIONS)."""
    import dataclasses

    fmt = post["format"]
    grade, burn = finishes(policy, fmt), captions_burned(policy, fmt)
    row, pieces = _pieces(conn, video)
    job_ids = [str(p["job"]["id"]) for p in pieces if p["job"]]
    frames = conn.execute("SELECT id, uri FROM assets WHERE job_id = ANY(%s::uuid[]) AND kind = 'keyframe' ORDER BY uri",
                          (job_ids,)).fetchall() if job_ids else []
    frames = frames or [{"id": im.get("asset_id"), "uri": im["uri"]} for im in body.get("images") or []]
    handle = (conn.execute("SELECT handle FROM creators WHERE id = %s", (post["creator_id"],)).fetchone()["handle"]
              if grade else None)
    style, words = None, None
    if burn:
        style = caption_style(conn, policy, post, body)
        if style is not None:
            timed = caption_words(conn, policy, body, video, store=store, fmt=fmt)
            words = None if timed is None else [dataclasses.asdict(w) for w in timed]
    key = _final_key(post)
    return _jsonable({
        "post": {"id": post["id"], "creator_id": post["creator_id"], "format": fmt},
        "body": body,
        "video": {"id": video["id"], "uri": video["uri"]},
        "source_uri": (source or video)["uri"],
        "video_meta": row["meta"] or {},
        "pieces": [{"id": p["id"], "uri": p["uri"], "meta": p["meta"] or {},
                    "job": ({"id": p["job"]["id"], "kind": p["job"]["kind"], "payload": p["job"]["payload"]}
                            if p["job"] else None)} for p in pieces],
        "keyframes": [{"id": f["id"], "uri": f["uri"]} for f in frames],
        "grade": grade, "burn": burn, "handle": handle,
        "caption_style": style, "caption_words": words,
        "final_key": key, "cover_key": key.rsplit("/", 1)[0] + "/cover.jpg",
        "policy": {s: policy.body.get(s) for s in FINISH_SECTIONS}, "policy_version": policy.version,
    })


def render_finish(plan: dict, policy: Policy, store, workdir) -> dict | None:
    """The finish itself, from a finish_plan and with no database: the finished file uploaded to plan.final_key and
    what was done ({uri, sha256, bytes, meta, cover, text_spans, finished_from, path}), or None when nothing applies
    (captions only, and nothing is said or shown). Run by finish_video in the producer (inline) or by the finisher's
    finish.video handler (queue); a rerun renders again and overwrites the same keys.

    In order: the sound mix (sound_plan, its own pass; a mix that fails is recorded and the video goes on unmixed),
    then the captions plan (style, word timings, placement off the face) and the shots' on-screen texts
    (overlay_items -> overlays.overlay_events) in ONE ASS script (overlays.write_script) that rides in finishing's
    filter chain where this ffmpeg has libass, else one Pillow burn pass of both; the cover from the finished file
    (uploaded; `cover.upload`), else the best keyframe or photo."""
    import dataclasses
    import shutil
    from pathlib import Path

    from .. import captions, finishing, media, overlays
    d = Path(workdir)
    d.mkdir(parents=True, exist_ok=True)
    post, body, fmt = plan["post"], plan["body"], plan["post"]["format"]
    grade, burn = bool(plan["grade"]), bool(plan["burn"])
    meta_v, pieces = plan.get("video_meta") or {}, plan.get("pieces") or []
    look, sound_error = None, None
    raw = store.get_file(plan["source_uri"], d / "raw.mp4")
    trim_s = float(finishing.cfg(policy).get("trim_head_s", 0) or 0)
    if trim_s > 0 and fmt not in SPOKEN:
        # An image-to-video clip opens on its still start photo for a beat: the first frame must already move.
        try:
            raw = media.trim_start(raw, d / "trimmed.mp4",
                                   frames=round(trim_s * float(media.probe(raw).get("fps") or 16)))
        except Exception as e:  # noqa: BLE001 - an untrimmed clip beats no clip
            log.warning("could not trim the opening of post %s: %s", post["id"], str(e)[:200])
    cues = _sound_plan_from(policy, post, body, lambda: (meta_v, pieces))
    used, mixed, passes = None, False, 0
    if cues:   # room tone, polish, effects, music under the voice (policy sound): its own pass, before the finish
        try:
            raw, used = _mix_sound(policy, store, cues, raw, d)
            mixed = True
            passes += 1
        except Exception as e:  # noqa: BLE001 - the video ships unmixed rather than not at all; the error is kept
            sound_error = f"{type(e).__name__}: {e}"[:500]
    cap, ovs, width, height = None, [], None, None
    if burn or body.get("shots"):
        info = media.probe(raw)
        if not (info["width"] and info["height"]):
            raise RuntimeError(f"can't read the frame size of {plan['video']['uri']}")
        # The finish stage upscales a smaller video to finishing.output first, so its text is laid out at that size.
        width, height = (finishing.finished_size(info["width"], info["height"], policy) if grade
                         else (info["width"], info["height"]))
        items = _overlay_items_from(fmt, body, meta_v, pieces, info.get("seconds"))
        face = (face_box(lambda: list(_fetch_frames(store, plan.get("keyframes") or [], d / "faces")))
                if burn or items else None)
        if burn:
            words = [captions.Word(**w) for w in plan.get("caption_words") or []]
            cap = captions_from_words(policy, body, words, style=plan.get("caption_style"), width=width,
                                      height=height, face_box=face)
        if items:
            ovs = overlays.overlay_events(items, (width, height), policy, duration=info.get("seconds"),
                                          face_box=face,   # labels stay off the burned captions too
                                          avoid_bands=[captions.plan_band(cap)] if cap else ())
    text = cap is not None or bool(ovs)
    if not grade and not text and not mixed:
        return None   # captions only, and nothing is said or shown
    out, cover = d / "final.mp4", None
    if grade:
        shot = _shot(body, _take_index(body, 0 if body.get("talk_segments") else None)) or {}
        look = finishing.look_for(policy, creator=plan.get("handle"), set_id=(shot.get("set") or {}).get("set_id"))
        one_pass = text and media.has_libass()
        ass, fpolicy = None, policy
        if one_pass:   # captions and labels ride in finishing's filter chain, after the grain: one script
            fonts = _text_fonts(cap, ovs, policy)
            ass = overlays.write_script(d / "text.ass", cap, ovs, policy, size=(width, height))
            fpolicy = dataclasses.replace(policy, body={
                **policy.body, "finishing": {**finishing.cfg(policy), "fonts_dir": str(fonts)}})
        # A format nobody speaks in (a story clip) is levelled as a bed, not loudnormed to the voice target.
        graded = finishing.finish(raw, out if not text or one_pass else d / "graded.mp4", look=look,
                                  ass=ass, has_voice=fmt in SPOKEN, policy=fpolicy)
        passes += 1
        if text and not one_pass:   # no libass here: the Pillow drawings, a second pass
            _burn_text(graded, out, d / "captions", cap, ovs, policy)
            passes += 1
        try:
            cover = _cover_upload(policy, store, out, plan["cover_key"], d, avoid=text_windows(cap, ovs))
        except Exception:  # noqa: BLE001 - unreadable frames: the pieces' keyframes, else the photos
            try:
                cover = _keyframe_cover(policy, store, plan.get("keyframes") or [], d / "covers")
            except Exception:  # noqa: BLE001 - a post without a picked cover still ships (the publisher has a fallback)
                cover = None
    elif text:
        _burn_text(raw, out, d / "captions", cap, ovs, policy)
        passes += 1
    else:   # captions only and nothing said, but the sound was mixed
        shutil.copyfile(raw, out)
    stored = store.put_file(out, plan["final_key"])
    meta = {"finished": grade, "look": look, "captions": cap.style_name if cap else None,
            "placement": cap.placement if cap else None, "passes": passes, "raw": str(plan["video"]["id"]),
            **({"sound": used} if used else {}), **({"sound_error": sound_error} if sound_error else {}),
            **({"on_screen_text": [{"kind": o.kind, "text": o.text, "position": o.position} for o in ovs]}
               if ovs else {}),
            **({"sound_design": {"room_tone": getattr(cues.get("room_tone_synth"), "preset", None),
                                 "room_size": cues.get("room_size"), "mic": cues.get("mic")}} if mixed else {})}
    return {"uri": stored.uri, "sha256": stored.sha256, "bytes": stored.bytes, "meta": _jsonable(meta),
            "cover": cover, "text_spans": text_spans(cap, ovs), "finished_from": plan["source_uri"],
            "path": str(out)}


def record_finish(conn, policy: Policy, post: dict, body: dict, video: dict, made: dict | None, store=None, *,
                  llm=None, local=None) -> dict:
    """The finished video's asset (or `video` itself when render_finish made nothing): its cover's keyframe asset
    (an uploaded cover), then the video asset at post-<id>/final.mp4 (an existing one is returned as it is). With
    `llm` (the judge), the finished file (`local`, else fetched) gets the final review once (manyworlds.review.final),
    its result in body.final_review. A talking post's finished file also gets the CPU lip-sync score
    (manyworlds.review.sync, report-only, policy review.sync), its result in body.sync_review."""
    import tempfile
    from pathlib import Path

    from .. import storage
    if not made or not made.get("uri"):
        return video
    existing = conn.execute("SELECT id, uri, meta FROM assets WHERE uri = %s AND kind = 'video'",
                            (made["uri"],)).fetchone()
    if existing:
        return existing
    cover = made.get("cover")
    if cover and cover.get("upload"):
        cover = _cover_row(conn, post, cover["upload"])
    row = conn.execute(
        """INSERT INTO assets (creator_id, kind, uri, sha256, bytes, meta, recipe)
           VALUES (%s, 'video', %s, %s, %s, %s, %s) RETURNING id, uri, meta""",
        (post["creator_id"], made["uri"], made["sha256"], made["bytes"], Jsonb({**made["meta"], "cover": cover}),
         Jsonb({"finished_from": made.get("finished_from"), "policy_version": policy.version}))).fetchone()
    # judged after the asset exists, so its final_* review rows land on it (review/final.py _rows)
    from ..review import sync as sync_review
    scored = sync_review.applies(policy, post)   # the lip-sync score (review.sync; CPU, free, report-only)
    if llm is None and not scored:
        return row
    try:
        with tempfile.TemporaryDirectory(prefix="mw-final-") as tmp:
            path = Path(local) if local else (store or storage.from_url()).get_file(made["uri"],
                                                                                  Path(tmp) / "final.mp4")
            if scored:   # never raises: a file it can't score records why in body.sync_review
                sync_review.review_sync(conn, policy, {**post, "body": body}, path)
            if llm is not None:   # the finished file judged once (review.final; body.final_review, updated in place)
                from ..review.final import review_final
                try:
                    review_final(conn, policy, llm, post, path, joins=join_times(conn, video),
                                 text_spans=made.get("text_spans") or [])
                except Exception as e:  # noqa: BLE001 - a judge outage never holds a finished post; the error is kept
                    body["final_review"] = {"error": f"{type(e).__name__}: {e}"[:500], "passed": True}
    except Exception as e:  # noqa: BLE001 - the finished file can't be fetched: the post ships, the error is kept
        err = {"error": f"{type(e).__name__}: {e}"[:500], "passed": True}
        if llm is not None:
            body["final_review"] = err
        if scored:
            body["sync_review"] = err
    return row


def finish_video(conn, policy: Policy, post: dict, body: dict, video: dict, store=None, *,
                 source: dict | None = None, llm=None) -> dict:
    """The post's finished video asset (or `video` itself when nothing applies to its format), made here in the
    producer (production.finish_mode inline): finish_plan, render_finish and record_finish in one go. `source`: the
    file to finish when it is not `video` itself (the AI upscaler's output; captions and shots are still timed from
    `video`'s pieces). With `llm` (the judge), the finished file gets the final review once (manyworlds.review.final),
    its result in body.final_review. Idempotent: the finished file is assets/<creator>/post-<id>/final.mp4, and a
    rerun reuses its asset."""
    import tempfile
    from pathlib import Path

    from .. import storage
    fmt = post["format"]
    if not (finishes(policy, fmt) or captions_burned(policy, fmt)):
        return video
    store = store or storage.from_url()
    existing = _existing_final(conn, store, post)
    if existing:
        return existing
    plan = finish_plan(conn, policy, post, body, video, store, source=source)
    with tempfile.TemporaryDirectory(prefix="mw-finish-") as tmp:
        made = render_finish(plan, policy, store, Path(tmp))
        return record_finish(conn, policy, post, body, video, made, store, llm=llm,
                             local=made.get("path") if made else None)


# --- the finish as a queue job (production.finish_mode: queue) ---------------------------------------------------
# The finish's encodes (minutes of 1080x1920 libx264 per post) block the producer's loop and outgrow its small VM, so
# they run as a finish.video job on the finisher (a Fly machine of this app that the autoscaler starts while finish
# jobs wait: deploy/autoscale.ensure_finisher). The job is a Step: enqueued once per attempt with the plan as its
# payload; done -> its result recorded here (record_finish); failed -> a new attempt, until finishing.max_failures
# jobs have failed, when the unfinished video ships and the owner hears of it, as with the inline finish.
def _finish_job(conn, policy: Policy, post: dict, body: dict, video: dict, store, source: dict | None, attempt: int):
    plan = finish_plan(conn, policy, {**post, "body": body}, body, video, store, source=source)
    return queue.enqueue(conn, FINISH_KIND, plan, creator_id=post["creator_id"], batch=f"post-{post['id']}",
                         post_id=post["id"], idempotency_key=f"post:{post['id']}:finish:{attempt}")


def _finish_state(conn, job_ids: list[str], body: dict) -> tuple[str, Any]:
    """waiting while the finish job is queued or running; chosen with what it made once done ({} when nothing
    applied); failed otherwise (its error kept in body.finish_error)."""
    job = conn.execute("SELECT status, result, error FROM jobs WHERE id = %s", (job_ids[-1],)).fetchone()
    if job["status"] in ("queued", "running"):
        return "waiting", None
    if job["status"] == "done":
        return "chosen", (job["result"] or {}).get("finish") or {}
    body["finish_error"] = str(job["error"] or f"the finish job ended {job['status']}")[:500]
    return "failed", None


def _advance_finish(conn, policy: Policy, post: dict, body: dict, video: dict, store=None, *,
                    source: dict | None = None, llm=None) -> tuple[str, dict | None]:
    """('waiting', None) while the post's finish.video job runs; ('chosen', the finished video asset) once it is
    recorded, or ('chosen', `video`) when nothing applies to its format, the job made nothing, or the finish was
    given up (finishing.max_failures failed jobs: body.finish_skipped, and the owner is told)."""
    from .. import storage
    fmt = post["format"]
    if body.get("finish_skipped") or not (finishes(policy, fmt) or captions_burned(policy, fmt)):
        return "chosen", video
    store = store or storage.from_url()
    existing = _existing_final(conn, store, post)
    if existing:
        return "chosen", existing
    # finishing.max_failures failed jobs in all: the first, then max_failures - 1 retries.
    retries = max(int(policy.get("finishing", "max_failures", default=3)) - 1, 0)
    step = Step(body.setdefault("finish_jobs", []), lambda j: _finish_state(conn, j, body),
                lambda attempt: _finish_job(conn, policy, post, body, video, store, source, attempt),
                ("finish_repairs", 0), ("finish_failures", retries), "finish")
    state, made = advance_step(body, step)
    if state == "waiting":
        return "waiting", None
    if state == "dropped":
        body.pop("drop_reason", None)
        body["finish_failures"] = int(body.get("finish_failures", 0)) + 1   # the job that failed last
        _give_up_finish(conn, post, body, body.get("finish_error") or "the finish job failed")
        return "chosen", video
    return "chosen", record_finish(conn, policy, {**post, "body": body}, body, video, made, store, llm=llm)


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
    body["finish_failures"] = int(body.get("finish_failures", 0)) + 1
    body["finish_error"] = f"{type(error).__name__}: {error}"[:500]
    if body["finish_failures"] < int(policy.get("finishing", "max_failures", default=3)):
        return False
    _give_up_finish(conn, post, body, body["finish_error"])
    return True


def _give_up_finish(conn, post: dict, body: dict, error: str) -> None:
    """The finish is given up: the post ships its unfinished video (body.finish_skipped) and the owner is told."""
    from .. import notify
    body["finish_skipped"] = error
    notify.owner_event(conn, "finish_failed", subject={"post_id": str(post["id"])},
                       payload={"format": post["format"], "error": error, "failures": body.get("finish_failures")},
                       creator_id=post["creator_id"], dedupe_key=f"finish_failed:{post['id']}")
