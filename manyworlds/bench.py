"""Day-1 benchmark: measure real seconds per output on each GPU pool and project
cost per 1,000 outputs. These measurements replace the research estimates in the
proposal, and feed the router's cost model."""

from __future__ import annotations

import statistics
import uuid
from typing import Any

from . import ledger, queue
from .creator import sets
from .policy import Policy

# Varied, everyday creator-feed prompts so the benchmark isn't tuned to one easy scene.
PHOTO_PROMPTS = [
    "candid phone photo of a young woman laughing at a cafe table, window light, slightly blurry background",
    "mirror selfie in a small apartment hallway, oversized hoodie, phone covering part of face, warm lamp light",
    "outfit photo on a city sidewalk, full body, overcast daylight, shot on iPhone, slightly tilted framing",
    "close-up selfie in a car, golden hour light through the window, natural skin texture",
    "photo of a man cooking pasta in a home kitchen, steam, messy counter, flash photography",
    "beach sunset photo from behind, wind in hair, imperfect horizon, phone camera grain",
    "gym mirror selfie, fluorescent lighting, water bottle in hand, casual pose",
    "dinner table photo with friends' hands in frame, candlelight, shallow depth of field",
]
# Reference-conditioned scene photos (sets.edit.kind): what the person does in the room; the pictures say who and where.
EDIT_PROMPTS = [
    "They sit at the table with a mug of coffee, candid phone photo, window light, slightly tilted framing",
    "They stand by the window looking out, natural daylight, shot on a phone, relaxed posture",
    "They tidy a shelf mid-motion, candid smartphone photo, soft lamp light",
    "Mirror selfie in the room, phone covering part of the face, warm evening light",
]
EDIT_ROLES = ("person", "set", "outfit")   # the input images, in this order (the handler takes at most this many)
SPEECH_LINES = [
    "Okay so I finally tried that new coffee place everyone keeps talking about, and honestly? Worth it.",
    "Quick update: the apartment is still a mess, but the plants are thriving, so that's a win.",
    "If you asked me five years ago where I'd be now, I would not have guessed any of this.",
]


def edit_inputs(extra: dict[str, Any] | None) -> list[dict]:
    """The edit benchmark's input images from --extra '{"images": [anchor, set plate, outfit reference]}' (1-3 uris,
    in that order), as [{kind, uri}]. Raises ValueError with how to pass them."""
    images = (extra or {}).get("images")
    usage = ("this benchmark needs its input images: --extra '{\"images\": [\"<creator anchor uri>\", "
             "\"<set plate uri>\", \"<outfit reference uri>\"]}' (1-3, in that order)")
    if not isinstance(images, list) or not images or len(images) > len(EDIT_ROLES):
        raise ValueError(usage)
    if not all(isinstance(u, str) and u.strip() for u in images):
        raise ValueError(f"every input image must be a uri; {usage}")
    return [{"kind": role, "uri": uri} for role, uri in zip(EDIT_ROLES, images)]


# Songs (music.song): original lyrics written for the benchmark (no real song, artist or "style of"), one
# instrumental, so the sung path (stems, lyric WER) and the bed path are both measured.
SONGS = [
    {"title": "Kettle at Midnight", "tags": "warm indie pop, soft female vocal, acoustic guitar, brushed drums, intimate",
     "bpm": 92, "key": "G major",
     "lyrics": "[Verse]\nThe kettle sings before I do\nSteam on the window, writing you\n"
               "[Chorus]\nStay up, stay up, the night is ours\nCounting the streetlights instead of stars\n"
               "[Verse]\nSocks on the tiles, a borrowed sweater\nEvery small hour feels a little better"},
    {"title": "Paper Planes Over Platform Nine", "tags": "upbeat synth pop, bright female vocal, driving bass, claps",
     "bpm": 118, "key": "A minor",
     "lyrics": "[Verse]\nTicket in my pocket and a song I can't stop humming\nTrains run late but I can hear the morning coming\n"
               "[Chorus]\nThrow it up, let it fly\nPaper planes across the sky\nEvery gate is open wide"},
    {"title": "Quiet Desk", "tags": "lo-fi instrumental, soft felt piano, warm pads, vinyl crackle, slow, no vocals",
     "bpm": 72, "key": "D minor", "instrumental": True},
]


def song_payload(policy: Policy, i: int, extra: dict[str, Any] | None = None) -> dict:
    """The i-th benchmark song (SONGS), at music.bench.seconds unless --extra gives duration_s, with the objective
    check (lyric WER of the vocal stem, length, loudness) on."""
    from . import music
    spec = {**SONGS[i % len(SONGS)], "duration_s": float(policy.get("music", "bench", "seconds", default=30))}
    if (extra or {}).get("duration_s") is not None:
        spec["duration_s"] = float(extra["duration_s"])
    return music.song_payload(policy, spec, seed=5000 + i, check=True, engine=(extra or {}).get("engine"),
                              variant=(extra or {}).get("variant"))


DANCE_KIND = "comfy.wan_animate2"
MUSIC_KIND = "music.song"
DANCE_PROMPT = "she dances in her kitchen, full body in frame, natural movement, handheld phone footage"


def dance_reference(policy: Policy, extra: dict[str, Any] | None) -> dict:
    """The dance benchmark's driving clip from --extra '{"reference": "<dance.references id>"}', checked as the worker
    will check it (manyworlds.dance.reference). Raises ValueError when dance is off or the clip is refused."""
    from . import dance
    if not dance.enabled(policy):
        raise ValueError("dance is off (policy dance.enabled): the worker would refuse every job")
    ref = (extra or {}).get("reference")
    if not ref:
        raise ValueError("this benchmark needs a driving clip: --extra '{\"image_uri\": ..., \"reference\": \"<dance.references id>\"}'")
    try:
        return dance.reference(str(ref), policy)
    except dance.RefusedReference as e:
        raise ValueError(str(e)) from None


def enqueue(conn, policy: Policy, kind: str, jobs: int, *, batch: str | None = None,
            extra: dict[str, Any] | None = None, pools: list[str] | None = None,
            profile: str | None = None) -> str:
    if kind.startswith(("talk.", "hosted.", "comfy.")) and not (extra or {}).get("image_uri"):
        # Otherwise a machine boots, every job fails at once, and it idles until stopped.
        raise ValueError("this benchmark needs the creator's start photo: --extra '{\"image_uri\": ...}'")
    if kind == DANCE_KIND:
        dance_reference(policy, extra)   # before anything is queued: dance off or a refused clip would fail every job
    edit = kind == sets.edit_kind(policy)
    if edit:
        inputs = edit_inputs(extra)   # before anything is queued: a bad list would fail every job on a large card
    batch = batch or f"bench-{kind}-{uuid.uuid4().hex[:6]}"
    gen = policy.get("generation", default={})
    for i in range(jobs):
        if edit:
            img = gen.get("image", {})
            payload = sets.edit_payload(policy, {"prompt": EDIT_PROMPTS[i % len(EDIT_PROMPTS)], "seed": 1000 + i,
                                                 "width": img.get("width", 896), "height": img.get("height", 1152)},
                                        inputs)
        elif kind.endswith("image") or kind.startswith("image."):
            img = gen.get("image", {})
            payload = {"prompt": PHOTO_PROMPTS[i % len(PHOTO_PROMPTS)], "n": policy.get(
                "job_kinds", kind, "outputs_per_job", default=4), "width": img.get("width", 896),
                "height": img.get("height", 1152), "seed": 1000 + i,
                "params": {"steps": img.get("steps", 9), "guidance": img.get("guidance", 0.0)}}
        elif kind == DANCE_KIND:
            payload = {"prompt": DANCE_PROMPT, "seed": 4000 + i, "keyframes": int(policy.get("dance", "identity", "keyframes", default=8)),
                       "params": dict(policy.get("dance", "generation", default={}) or {})}
        elif kind == MUSIC_KIND:
            payload = song_payload(policy, i, extra)
        elif kind == "comfy.wan_i2v":
            payload = {"prompt": "she turns toward the camera and smiles, handheld phone footage", "seed": 2000 + i,
                       "keyframes": 4, "params": {**gen.get("comfy_video", {}), **_profile(gen, "comfy_video_profiles", profile)}}
        elif kind == "hosted.video":
            payload = {"prompt": "she turns toward the camera and smiles, handheld phone footage", "seed": 2000 + i,
                       "keyframes": 4, "params": dict(gen.get("hosted_video", {}))}
        elif kind.startswith("talk."):
            # The start photo is the creator's: pass it with --extra '{"image_uri": "..."}'.
            talk = {**gen.get("talk", {}), **_profile(gen, "talk_profiles", profile)}
            payload = {"script": SPEECH_LINES[i % len(SPEECH_LINES)], "voice": talk.pop("voice", "af_heart"),
                       "prompt": "a young woman talks casually to her phone camera, natural head movement",
                       "seed": 3000 + i, "keyframes": 4, "params": talk}
        else:
            payload = {"text": SPEECH_LINES[i % len(SPEECH_LINES)], "voice": "af_heart"}
        # a song's --extra duration_s / variant / engine went into its payload above (the engine is a settings dict)
        payload.update({k: v for k, v in (extra or {}).items()
                        if not (kind == MUSIC_KIND and k in ("duration_s", "variant", "engine"))})
        # A creator_id in extra ties the job to that creator, so the review gates judge it against them.
        creator_id = payload.pop("creator_id", None)
        queue.enqueue(conn, kind, payload, batch=batch, pools=pools, creator_id=creator_id,
                      max_attempts=int(policy.get("queue", "max_attempts", default=3)))
    return batch


VIDEO_KIND_PREFIXES = ("video.", "comfy.", "talk.")
VIDEO_KINDS = ("hosted.video", "hosted.avatar")


def is_video_kind(kind: str) -> bool:
    return kind.startswith(VIDEO_KIND_PREFIXES) or kind in VIDEO_KINDS


def report(conn, policy: Policy, batch: str) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT j.kind, a.pool, a.gpu_type, a.gpu_seconds,
               (SELECT count(*) FROM assets s WHERE s.job_id = j.id AND s.kind <> 'crop') AS outputs,
               (SELECT count(*) FROM assets s WHERE s.job_id = j.id AND s.kind = 'video') AS videos,
               (SELECT count(*) FROM assets s WHERE s.job_id = j.id AND s.meta->>'role' = 'mix') AS songs
        FROM attempts a JOIN jobs j ON j.id = a.job_id
        WHERE j.batch = %s AND a.outcome = 'done'
        """,
        (batch,),
    ).fetchall()
    # A video job's outputs are its videos: keyframes and crops are only for review. A song job's is its mix (the
    # stems come with it).
    rows = [{**r, "outputs": r["videos"]} if is_video_kind(r["kind"])
            else {**r, "outputs": r["songs"]} if r["kind"] == MUSIC_KIND else r for r in rows]
    out: dict[str, Any] = {"batch": batch, "jobs": queue.counts(conn, batch=batch),
                           "spend": ledger.totals(conn, batch=batch), "measured": []}
    if any(r["kind"] == MUSIC_KIND for r in rows):
        out["music_checks"] = music_checks(conn, batch)
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["kind"], r["pool"], r["gpu_type"]), []).append(r)
    for (kind, pool, gpu), rs in sorted(groups.items()):
        per_output = [float(r["gpu_seconds"]) / max(r["outputs"], 1) for r in rs]
        med = statistics.median(per_output)
        p90 = sorted(per_output)[max(0, int(len(per_output) * 0.9) - 1)]
        # Project to every pool with the same GPU type (prices differ, speed shouldn't).
        projections = {
            name: round(med * float(p["usd_per_hour"]) / 3.6, 4)
            for name, p in policy.get("gpu_pools", default={}).items()
            if p.get("gpu") == gpu and p.get("usd_per_hour") is not None   # serverless pools bill per second
        }
        out["measured"].append({
            "kind": kind, "pool": pool, "gpu": gpu, "jobs": len(rs),
            "outputs": sum(r["outputs"] for r in rs),
            "sec_per_output_median": round(med, 3), "sec_per_output_p90": round(p90, 3),
            "usd_per_1k_outputs_by_pool": projections,
        })
    return out


def music_checks(conn, batch: str) -> dict[str, Any]:
    """The songs' objective check over a batch (the mix's meta.check, written on the worker): how many passed, the
    median lyric WER of the vocal stems, length error and loudness, and each song's numbers."""
    rows = conn.execute(
        """SELECT j.payload->>'title' AS title, s.meta->'check' AS song_check, (s.meta->>'seconds')::float AS seconds
           FROM assets s JOIN jobs j ON j.id = s.job_id
           WHERE j.batch = %s AND j.kind = %s AND s.meta->>'role' = 'mix' ORDER BY j.created_at, s.uri""",
        (batch, MUSIC_KIND)).fetchall()
    checks = [dict(r["song_check"] or {}) for r in rows]

    def med(key: str):
        vals = [float(c[key]) for c in checks if c.get(key) is not None]
        return round(statistics.median(vals), 3) if vals else None
    return {"songs": len(rows), "passed": sum(1 for c in checks if c.get("passed")),
            "wer_median": med("wer"), "line_coverage_median": med("line_coverage"),
            "duration_error_share_median": med("duration_error_share"),
            "lufs_median": med("lufs"),
            "each": [{"title": r["title"], "seconds": r["seconds"],
                      **{k: c.get(k) for k in ("wer", "line_coverage", "duration_error_share", "lufs", "failed",
                                               "passed")}}
                     for r, c in zip(rows, checks)]}


def _profile(gen: dict, section: str, profile: str | None) -> dict:
    """Settings under test (generation.<section>.<profile>), merged over the production ones; {} for none."""
    if not profile:
        return {}
    if profile not in (gen.get(section) or {}):
        raise ValueError(f"no generation.{section}.{profile}")
    return dict(gen[section][profile])
