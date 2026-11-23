"""The production tick: start due posts, advance each producing post (under its own lease) through its scenes,
its video, the upscale and the finish, and count or drop a post whose advance keeps raising."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from .. import queue
from ..policy import Policy
from .common import _cfg, _set, log
from .finish import _advance_finish, _advance_upscale, _finish_failed, finish_mode, finish_video
from .followups import FOLLOW_UPS, _advance_follow_up
from .locks import post_lease
from .motion import _advance_clip_edit, clip_edit
from .scenes import _scene_steps, start
from .steps import advance_step
from .takes import _directed, _hybrid_cfg, hybrid_modes, script_takes, shot_takes, talk_segments
from .talk import _advance_long_talk


def _slide_formats(policy: Policy) -> list[str]:
    cfg = policy.get("slides", default={}) or {}
    return list(cfg.get("formats") or []) if cfg.get("enabled") else []


def advance_post(conn, policy: Policy, post: dict, store=None, *, llm=None) -> str:
    body = dict(post["body"])
    results = []
    for step in _scene_steps(conn, policy, post, body):
        results.append(advance_step(body, step))
        if results[-1][0] == "dropped":
            break   # no more repairs for a post that is being dropped
    states, chosen, video = {state for state, _ in results}, [r for _, r in results], None
    if states == {"chosen"}:
        body["images"] = [{"asset_id": str(a["id"]), "uri": a["uri"]} for a in chosen]
        if body.get("slides") and post["format"] in _slide_formats(policy):
            # Text slides in her carousel brand (manyworlds.slides), drawn here on the CPU around her photos.
            try:
                from .. import slides as slides_mod, storage
                made = slides_mod.render_post(conn, policy, store or storage.from_url(), post, body, chosen)
                body["photos"] = body["images"]
                body["images"] = [{"asset_id": str(a["id"]), "uri": a["uri"]} for a in made]
                chosen = made
            except Exception as e:  # noqa: BLE001 - her photos still make a carousel without the text slides
                body["slides_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        if post["format"] in FOLLOW_UPS:   # the edit or clip, once every photo is chosen
            if post["format"] == "talking_video" and "talk_segments" not in body:
                hybrid = bool(_hybrid_cfg(policy).get("enabled")) and len(chosen) >= 2
                directed = _directed(policy, body)
                takes = (shot_takes(policy, body["shots"], cutaways=hybrid) or script_takes(policy, body)
                         if directed else [])
                if takes:
                    # Directed: one take per shot that speaks, cut only where the shots change; cutaway shots are
                    # narrated over their own photo (hybrid), the rest lip-synced from theirs. Never the hybrid's
                    # short segments: a scene changes only where the shots do.
                    body["talk_shots"] = [i for _, _, i in takes]
                    if len(takes) > 1:
                        body["talk_segments"] = [text for text, _, _ in takes]
                        if any(mode == "narrated" for _, mode, _ in takes):
                            body["talk_modes"] = [mode for _, mode, _ in takes]
                    else:
                        body["talk_segments"], body["talk_script"] = [], takes[0][0]   # [] = one clip
                    post = {**post, "body": body}
                else:
                    # Hybrid (generation.talk_hybrid): with cutaway photos, short segments, only some on camera.
                    segments = talk_segments(policy, body.get("script", ""), hybrid=hybrid)
                    body["talk_segments"] = segments if len(segments) > 1 else []   # [] = one clip
                    if hybrid and len(segments) > 2:
                        body["talk_modes"] = hybrid_modes(segments,
                                                          float(_hybrid_cfg(policy).get("on_camera_share", 0.4)))
                        post = {**post, "body": body}
            if body.get("talk_segments"):   # a long script: one job per segment, then joined
                state, video = _advance_long_talk(conn, policy, post, body, chosen, store)
            elif clip_edit(policy, post, chosen) or body.get("clip_edit_jobs"):
                state, video = _advance_clip_edit(conn, policy, post, body, chosen, store)
            else:
                state, video = _advance_follow_up(conn, policy, post, body, chosen, store)
            target = (_cfg(policy).get("reroute_failed_video") or {}).get(post["format"])
            # Only when review kept rejecting the clips (not machine failures), and never for an experiment
            # idea, whose arm would then be measured on a photo.
            rejected = "passed review" in (body.get("drop_reason") or "")
            in_experiment = conn.execute("SELECT 1 FROM ideas WHERE post_id = %s AND experiment_id IS NOT NULL",
                                         (post["id"],)).fetchone()
            if state == "dropped" and target and rejected and not in_experiment:
                # The video kept failing; its start photo passed review. Ship that instead of nothing.
                body.update(rerouted_from=post["format"], reroute_reason=body.pop("drop_reason", ""))
                _set(conn, post["id"], "ready", body, qa_status="passed", asset_ids=[a["id"] for a in chosen],
                     format=target)
                return "ready"
            states = {state}
    if "dropped" in states:
        _set(conn, post["id"], "dropped", body, qa_status="failed")
        return "dropped"
    if "waiting" in states:
        _set(conn, post["id"], "producing", body)
        return "producing"
    asset_ids = [a["id"] for a in chosen]
    if video:
        final = video
        ustate, source = _advance_upscale(conn, policy, post, body, video)
        if ustate == "waiting":
            _set(conn, post["id"], "producing", body)
            return "producing"
        try:
            if finish_mode(policy) == "queue":   # a finish.video job on the finisher (production.finish: _advance_finish)
                fstate, done = _advance_finish(conn, policy, post, body, video, store, source=source, llm=llm)
                if fstate == "waiting":
                    _set(conn, post["id"], "producing", body)
                    return "producing"
                final = done or video
            else:
                final = finish_video(conn, policy, {**post, "body": body}, body, video, store, source=source, llm=llm)
        except Exception as e:  # noqa: BLE001 - ffmpeg or storage: tried again next tick, then shipped unfinished
            if not _finish_failed(conn, policy, post, body, e):
                _set(conn, post["id"], "producing", body)
                return "producing"
        if str(final["id"]) != str(video["id"]):
            meta = final.get("meta") or {}
            body["video_raw"] = {"asset_id": str(video["id"]), "uri": video["uri"]}
            body["finish"] = {k: meta.get(k) for k in ("finished", "look", "captions", "placement", "passes")}
            if meta.get("cover"):
                body["cover"] = meta["cover"]
            if meta.get("sound"):   # what was mixed in, with licences and credits (CC-BY needs its credit)
                body["sound"] = meta["sound"]
            if meta.get("sound_error"):   # shipped unmixed: kept where the feed QA and the owner can see it
                body["sound_error"] = meta["sound_error"]
        body["video"] = {"asset_id": str(final["id"]), "uri": final["uri"]}
        asset_ids = [final["id"], *asset_ids]
        if (body.get("final_review") or {}).get("passed") is False:   # review.final.enforce: the finished file failed
            body["drop_reason"] = "the finished video failed the final review"
            _set(conn, post["id"], "dropped", body, qa_status="failed")
            return "dropped"
    _set(conn, post["id"], "ready", body, qa_status="passed", asset_ids=asset_ids)
    return "ready"


def advance(conn, policy: Policy, *, now: datetime | None = None, store=None, llm=None) -> dict:
    now = now or datetime.now(timezone.utc)
    lead = timedelta(hours=float(_cfg(policy).get("lead_hours", 48)))
    # Jobs that used their last attempt and lost the lease are dead, so their scene gets repaired
    # instead of waiting forever.
    out_reaped = queue.reap(conn)
    out = {"started": 0, "producing": 0, "ready": 0, "dropped": 0}
    out["reaped"] = out_reaped
    # A planned post in a format production doesn't make, whose time has passed, won't happen.
    made = list(_cfg(policy).get("formats", ["photo", "carousel"]))
    out["expired"] = len(conn.execute(
        """UPDATE posts SET status = 'dropped', updated_at = now(),
                  body = body || jsonb_build_object('drop_reason', 'format ' || format || ' not produced yet')
           WHERE status = 'planned' AND scheduled_for < %s AND NOT (format = ANY(%s)) RETURNING id""",
        (now, made)).fetchall())
    for post in conn.execute(
            # Only creators with reference images: the reviewer judges identity against them.
            "SELECT p.* FROM posts p JOIN creators c ON c.id = p.creator_id WHERE p.status = 'planned' "
            "AND coalesce(p.scheduled_for, now()) <= %s AND p.format = ANY(%s) AND NOT p.body ? 'breadth_run' "
            "AND jsonb_array_length(coalesce(c.anchors->'reference_images', '[]'::jsonb)) > 0 "
            "ORDER BY p.scheduled_for", (now + lead, list(_cfg(policy).get("formats", ["photo", "carousel"])))).fetchall():
        try:
            # One producer per post (production.locks): another producer's post is skipped this tick, and the row
            # is read again under the lease, so a post started meanwhile is not started twice.
            with post_lease(conn, policy, post["id"]) as held:
                if not held:
                    out["locked"] = out.get("locked", 0) + 1
                    continue
                row = conn.execute("SELECT * FROM posts WHERE id = %s AND status = 'planned'", (post["id"],)).fetchone()
                if row is None:
                    continue
                start(conn, policy, row)
        except Exception:  # noqa: BLE001 - a post that can't start stays planned; the others still start
            log.exception("starting post %s failed", post["id"])
            out["start_errors"] = out.get("start_errors", 0) + 1
            continue
        out["started"] += 1
    # Breadth-test posts (manyworlds.breadth) are only ever advanced by their own driver, under the test overlay.
    for post in conn.execute("SELECT * FROM posts WHERE status = 'producing' AND NOT body ? 'breadth_run' "
                             "ORDER BY scheduled_for").fetchall():
        try:
            # advance_post writes the whole body back: under the post's lease, from the row as it is now (another
            # producer may have advanced it since the list above was read). A post another producer holds waits.
            with post_lease(conn, policy, post["id"]) as held:
                if not held:
                    out["locked"] = out.get("locked", 0) + 1
                    continue
                row = conn.execute("SELECT * FROM posts WHERE id = %s AND status = 'producing'",
                                   (post["id"],)).fetchone()
                if row is None:
                    continue
                post = row
                out[advance_post(conn, policy, post, store, llm=llm)] += 1
        except Exception as e:  # noqa: BLE001 - one broken post must never stop the others
            out["advance_errors"] = out.get("advance_errors", 0) + 1
            if _advance_failed(conn, policy, post, e):
                out["dropped"] += 1
    return out


def _advance_failed(conn, policy: Policy, post: dict, error: Exception) -> bool:
    """Count a tick in which advancing this post raised (body.advance_errors, the last one in body.advance_error).
    Once production.max_advance_errors is used up the post is dropped with the error as its reason and the owner
    hears of it; until then it is tried again next tick. True when it was dropped."""
    from .. import notify
    err = f"{type(error).__name__}: {error}"[:500]
    log.exception("advancing post %s failed", post["id"])
    try:
        n = int((post["body"] or {}).get("advance_errors", 0)) + 1
        dropped = n >= int(_cfg(policy).get("max_advance_errors", 5))
        patch = {"advance_errors": n, "advance_error": err}
        if dropped:
            patch["drop_reason"] = f"production failed {n} times: {err}"
        drop = ", status = 'dropped', qa_status = 'failed'" if dropped else ""
        row = conn.execute(f"UPDATE posts SET body = body || %s, updated_at = now(){drop} "
                           "WHERE id = %s AND status = 'producing' RETURNING id", (Jsonb(patch), post["id"])).fetchone()
        if dropped and row:
            notify.owner_event(conn, "post_production_failed", subject={"post_id": str(post["id"])},
                               payload={"format": post["format"], "error": err, "failures": n},
                               creator_id=post["creator_id"], dedupe_key=f"post_production_failed:{post['id']}")
        return bool(dropped and row)
    except Exception:  # noqa: BLE001 - counting must not stop the other posts either (the database may be down)
        log.exception("could not record the failure of post %s", post["id"])
        return False
