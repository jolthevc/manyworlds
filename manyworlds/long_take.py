"""The long-take comparison (2026-11-23, docs/research/2026-11-23-avatar-and-music-critic.md): one ~90 s line in the
creator's designed voice and one still, rendered two ways so the owner can compare them side by side:

  - one continuous take: LongCat-Video-Avatar 1.6 in a single job (talk.longcat, worker/handlers/talk_longcat.py),
    autoregressive continuation re-anchored to the photo, no seams;
  - our segment chain: InfiniteTalk (talk.infinitetalk on ComfyUI) in segments of at most `max_segment_seconds`,
    each starting from the previous segment's last frame, joined on the control plane (production's `chain` join).

Both speak the SAME voice line (one voice.speak.<engine> job in the creator's engine, shared by every take of a run
with the same script): a chained segment gets its part of it as `audio_window`, cut between the line's sentences (its
meta.lines timings),
so the comparison isolates the video model. A creator without a designed voice is refused, with the reason.

A take is a breadth-test item (`take:` in a plan of policy/test_overlay.yaml; manyworlds.breadth drives it: its
batch, pools, priority, spend cap, deadline and the video gate's review of each segment). Nothing is repaired or
retried on a review verdict: the raw takes are what is compared. When a take's video is complete it is measured here
(the control plane, no GPU) and stored as one asset (kind video, meta.role long_take): per segment the worker's
lip-sync score and render seconds; on the finished take, at `breadth.takes.checkpoints_s` (0/30/60/90 s), the colour
drift (CIEDE2000 of the frame's mean Lab colour against the first frame) and the face identity (the video judge's
per-keyframe identity, 1-10, against the creator's references: one screen call per take).

The same item benchmarks any model as one take (plan `candidates`): `kind: comfy.graph` with `graph` (a named ComfyUI
workflow, policy comfy_graphs; worker/handlers/comfy_graph.py), `audio: "@shared:<key>"` (speak the voice line a
production comparison already made for this script: the voice.speak.* job of body.shared_key <key> whose lines say it,
so every model speaks the very same audio; nothing is voiced again), or `audio: none` (a graph that makes its own
sound: its length comes from the script), and `pools` (this take's own pools, over the plan's per family).
"""

from __future__ import annotations

import hashlib
import logging
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from . import queue
from .policy import Policy

log = logging.getLogger("manyworlds.long_take")

ROLE = "long_take"
VOICE_KINDS = "voice.speak.%"   # every designed-voice engine's kind (voice_lab.speak_kind)
GRAPH_KIND = "comfy.graph"
NO_AUDIO = "none"      # take.audio: the graph makes its own sound
SHARED = "@shared:"     # take.audio: the voice line a production comparison made (its body.shared_key)


# --------------------------------------------------------------------------------------------------------- the item

def cfg(policy: Policy) -> dict:
    return (policy.get("breadth", default={}) or {}).get("takes") or {}


def validate(item_id: str, take: Any) -> dict:
    """A plan item's `take`: {kind, script, image, prompt, profile?, max_segment_seconds?, keyframes?, graph?,
    audio?, pools?, params?}."""
    if not isinstance(take, dict):
        raise ValueError(f"item {item_id}: `take` must be a mapping")
    for key in ("kind", "script", "image", "prompt"):
        if not str(take.get(key) or "").strip():
            raise ValueError(f"item {item_id}: take.{key} is required")
    if take.get("max_segment_seconds") is not None and float(take["max_segment_seconds"]) <= 0:
        raise ValueError(f"item {item_id}: take.max_segment_seconds must be positive")
    if (take["kind"] == GRAPH_KIND) != bool(take.get("graph")):
        raise ValueError(f"item {item_id}: take.graph goes with kind {GRAPH_KIND}, and that kind needs one")
    audio = take.get("audio")
    if audio is not None and audio != NO_AUDIO and not (str(audio).startswith(SHARED) and str(audio)[len(SHARED):]):
        raise ValueError(f"item {item_id}: take.audio is {NO_AUDIO!r} or {SHARED}<shared_key>, not {audio!r}")
    if (audio is not None or take.get("graph")) and take.get("max_segment_seconds"):
        raise ValueError(f"item {item_id}: a graph or a reused line is one take (no max_segment_seconds)")
    return take


def shared_key(take: dict) -> str | None:
    audio = str(take.get("audio") or "")
    return audio[len(SHARED):] if audio.startswith(SHARED) else None


def voiced_here(take: dict) -> bool:
    """The take speaks a voice line of its own (one voice.speak.<engine> job), not a reused one or the graph's own
    sound."""
    return take.get("audio") is None


def voice_kind(policy: Policy, creator=None) -> str:
    """The kind that voices a take's line: the creator's engine's (voice_lab.speak_kind), else, for a dry estimate
    without her, the default engine's (policy voice.engine)."""
    from . import voices
    from .creator import voice_lab
    anchors = (creator or {}).get("anchors")
    engine = voices.engine(anchors) if voices.designed(anchors) else policy.get("voice", "engine")
    return voice_lab.speak_kind(policy, engine)


def graph_spec(policy: Policy, take: dict) -> dict | None:
    return policy.get("comfy_graphs", take["graph"], default=None) if take.get("graph") else None


def graph_unavailable(policy: Policy, graph: str, env=None) -> str | None:
    """Why the named graph can't run from here: not in comfy_graphs, or a gated repo's token missing in this
    environment (a worker gets this environment's). Shared with production's talk routes (production.talk)."""
    import os
    spec = graph_spec(policy, {"graph": graph})
    if not spec:
        return f"graph {graph} is not in comfy_graphs"
    env = os.environ if env is None else env
    missing = [k for k in spec.get("requires_env") or [] if not env.get(k)]
    if missing:
        return (f"graph {graph} downloads from a gated repo: {', '.join(missing)} is not set here, so "
                f"no worker would get it")
    return None


def graph_fields(policy: Policy, graph: str) -> dict:
    """What a comfy.graph job carries besides its inputs: the graph's name and its spec (a worker's baked policy may
    predate the graph). Shared with production's talk routes (production.talk)."""
    return {"graph": graph, "graph_spec": graph_spec(policy, {"graph": graph})}


def skip_reason(policy: Policy, take: dict, creator, env=None) -> str | None:
    """Why this take can't run: its kinds, graph or profile missing, a gated graph's token missing here (a worker
    gets this environment's), or a creator without a designed voice when the take voices its own line."""
    from . import voices
    kinds = policy.get("job_kinds", default={}) or {}
    for kind in (take["kind"], *([voice_kind(policy, creator)] if voiced_here(take) else [])):
        if kind not in kinds:
            return f"job kind {kind} is not in job_kinds"
    if take.get("graph"):
        reason = graph_unavailable(policy, take["graph"], env)
        if reason:
            return reason
    profile = take.get("profile")
    if profile and profile not in (policy.get("generation", "talk_profiles", default={}) or {}):
        return f"talk profile {profile} is not in generation.talk_profiles"
    if voiced_here(take) and not voices.designed((creator or {}).get("anchors")):
        return (f"creator {creator['handle']} has no designed voice (speaks with Kokoro): the long-take comparison "
                f"speaks one line in the designed voice (voice.speak.<engine>). Cast one first: "
                f"mw voice cast {creator['handle']}")
    return None


def talk_params(policy: Policy, take: dict) -> dict:
    """The job's settings: for a graph, its comfy_graphs params; otherwise generation.talk with the take's talk
    profile merged over it. The take's own `params` go on top."""
    if take.get("graph"):
        params = dict((graph_spec(policy, take) or {}).get("params") or {})
    else:
        params = dict(policy.get("generation", "talk", default={}) or {})
        params.pop("voice", None)
        params.pop("max_take_seconds", None)   # production's take limit, not a render setting
        if take.get("profile"):
            params.update(policy.get("generation", "talk_profiles", take["profile"], default={}) or {})
    params.update(take.get("params") or {})
    return params


def script_seconds(policy: Policy, take: dict) -> float:
    from .talk_segments import estimate_seconds
    return estimate_seconds(take["script"], float(policy.get("generation", "talk_long", "words_per_second",
                                                             default=2.5)))


def expected_jobs(policy: Policy, take: dict) -> list[dict]:
    """The take's jobs on a first try, for the dry run: its voice line (shared by the run's takes) when it voices one,
    and its renders. A graph's or a talk profile's render is priced at its own rate (`rate`: <kind>:<name>, breadth
    estimate) and carries what its machine downloads first (`download_gb`, once per machine and `weights`)."""
    seconds = script_seconds(policy, take)
    n = 1
    if take.get("max_segment_seconds"):
        n = max(1, math.ceil(seconds / float(take["max_segment_seconds"])))
    render: dict = {"kind": take["kind"], "n": 1, "output_seconds": round(seconds / n, 1)}
    if take.get("pools"):
        render["pools"] = list(take["pools"])
    spec = graph_spec(policy, take)
    name, source = (take.get("graph"), spec) if spec else \
        (take.get("profile"), policy.get("generation", "talk_profiles", take.get("profile") or "", default=None))
    if name:
        render["rate"] = f"{take['kind']}:{name}"
        if (source or {}).get("download_gb"):
            render.update(weights=(source or {}).get("weights_set") or name,
                          download_gb=float(source["download_gb"]))
    voice = [{"kind": voice_kind(policy), "n": 1, "output_seconds": None}] if voiced_here(take) else []
    return [*voice, *(dict(render) for _ in range(n))]


def shared_audio(conn, key: str, script: str):
    """The newest finished voice line of a production comparison (body.shared_key `key`: its voice.speak.* jobs are
    keyed shared:<key>:voice:...) whose lines say `script`, as its audio asset {id, uri, meta}; None when there is
    none (that comparison hasn't voiced this line yet)."""
    want = " ".join(str(script).lower().split())
    rows = conn.execute(
        """SELECT j.payload, a.id, a.uri, a.meta FROM jobs j
             JOIN LATERAL (SELECT id, uri, meta FROM assets WHERE job_id = j.id AND kind = 'audio'
                           ORDER BY uri LIMIT 1) a ON true
            WHERE j.kind LIKE %s AND j.status = 'done' AND j.idempotency_key LIKE %s
            ORDER BY j.updated_at DESC""",
        # the breadth driver scopes a run's shared key (shared:<key>:<run>:voice:...), so match either form
        (VOICE_KINDS, f"shared:{key}:%voice:%")).fetchall()
    for r in rows:
        said = " ".join(" ".join(str(ln.get("text") or "") for ln in (r["payload"] or {}).get("lines") or [])
                        .lower().split())
        if said == want:
            return {"id": r["id"], "uri": r["uri"], "meta": r["meta"] or {}}
    return None


def audio_status(conn, take: dict) -> str:
    """For the dry run: where the take's voice line comes from, and whether a reused one exists yet."""
    key = shared_key(take)
    if key:
        return f"reused from shared_key {key}" if shared_audio(conn, key, take["script"]) else \
            f"MISSING: no finished voice line of shared_key {key} says this script (run that comparison first)"
    return "the graph's own sound" if take.get("audio") == NO_AUDIO else "voiced by this run (voice.speak.<engine>)"


# ------------------------------------------------------------------------------------------------- the voice line

def digest(text: str) -> str:
    return hashlib.sha256(" ".join(str(text).split()).encode()).hexdigest()[:12]


def voice_batch(batch: str) -> str:
    return f"{batch}-take-voice"


def voice_job(conn, policy: Policy, *, run: str, batch: str, take: dict, creator, pools: list[str] | None = None,
              meta: dict | None = None):
    """The shared voice line: one voice.speak.<engine> job per run and script (an idempotency key), every sentence a
    line."""
    from .creator import voice_lab
    from .talk_segments import sentences
    d = digest(take["script"])
    payload = voice_lab.speak_payload(policy, (creator["anchors"] or {})["voice"],
                                      [{"text": s} for s in sentences(take["script"])], seed=int(d[:7], 16))
    payload["breadth"] = {**(meta or {}), "item": "take-voice", "exercises": ["designed_voice"]}
    return queue.enqueue(conn, voice_kind(policy, creator), payload, batch=voice_batch(batch), pools=pools,
                         creator_id=creator["id"],
                         idempotency_key=f"breadth:{run}:take-voice:{d}",
                         max_attempts=int(policy.get("queue", "max_attempts", default=3)))


def _job(conn, key: str):
    return conn.execute("SELECT id, status, error FROM jobs WHERE idempotency_key = %s", (key,)).fetchone()


def _output(conn, job_id, kind: str):
    return conn.execute("SELECT id, uri, meta FROM assets WHERE job_id = %s AND kind = %s ORDER BY uri LIMIT 1",
                        (job_id, kind)).fetchone()


FAILED = ("failed", "dead", "cancelled")


def windows(lines: list[dict], seconds: float, max_segment_seconds: float | None) -> list[dict]:
    """The voice line cut for a chain: whole sentences grouped greedily into parts of at most max_segment_seconds
    (a longer sentence is a part of its own), cut halfway through the pause between two sentences. One part, the
    whole line, without a limit. [{start_s, end_s, text}]"""
    if not max_segment_seconds:
        return [{"start_s": 0.0, "end_s": float(seconds), "text": " ".join(str(ln["text"]) for ln in lines)}]
    groups: list[list[dict]] = []
    for ln in lines:
        if groups and float(ln["end_s"]) - float(groups[-1][0]["start_s"]) <= float(max_segment_seconds):
            groups[-1].append(ln)
        else:
            groups.append([ln])
    out = []
    for i, g in enumerate(groups):
        start = 0.0 if i == 0 else out[-1]["end_s"]
        end = float(seconds) if i == len(groups) - 1 else \
            round((float(g[-1]["end_s"]) + float(groups[i + 1][0]["start_s"])) / 2, 3)
        out.append({"start_s": start, "end_s": end, "text": " ".join(str(ln["text"]) for ln in g)})
    return out


# ----------------------------------------------------------------------------------------------------- the renders

def segment_key(run: str, item_id: str, k: int) -> str:
    return f"breadth:{run}:{item_id}:seg{k}"


def take_asset(conn, run: str, item_id: str):
    return conn.execute("SELECT id, uri, meta FROM assets WHERE kind = 'video' AND meta->>'role' = %s "
                        "AND meta->'breadth'->>'run' = %s AND meta->'breadth'->>'item' = %s LIMIT 1",
                        (ROLE, run, item_id)).fetchone()


def _voice_line(conn, policy: Policy, *, run: str, batch: str, take: dict, creator, pins: dict, meta: dict):
    """(None, the take's audio {uri, meta}) once its line is there, else ({state...}, None): voiced here (one
    voice.speak.<engine> job), reused (take.audio @shared:<key>), or none for a graph that makes its own sound (its length
    then comes from the script)."""
    if take.get("audio") == NO_AUDIO:
        return None, {"uri": None, "meta": {"seconds": script_seconds(policy, take)}}
    key = shared_key(take)
    if key:
        audio = shared_audio(conn, key, take["script"])
        if audio is None:
            return {"state": "failed", "reason": f"no finished voice line of shared_key {key} says this script "
                                                 f"(run that comparison first)"}, None
        return None, audio
    voice = voice_job(conn, policy, run=run, batch=batch, take=take, creator=creator,
                      pools=pins.get(policy.family(voice_kind(policy, creator))), meta=meta)
    v = conn.execute("SELECT status, error FROM jobs WHERE id = %s", (voice,)).fetchone()
    if v["status"] in FAILED:
        return {"state": "failed", "reason": f"the voice line failed: {v['error']}"[:300]}, None
    if v["status"] != "done":
        return {"state": "waiting", "on": "voice"}, None
    audio = _output(conn, voice, "audio")
    if not audio:
        return {"state": "failed", "reason": "the voice line finished without audio"}, None
    return None, audio


def advance(conn, policy: Policy, *, run: str, batch: str, item_id: str, take: dict, creator, item_batch: str,
            anchors_uri: str, pins: dict, meta: dict, store=None, llm=None) -> dict:
    """One step of a take: its voice line, then its segments one after another, then the finished take measured
    and stored. Returns {state: waiting | done | failed, ...}. Safe to call again at any point (idempotent)."""
    done = take_asset(conn, run, item_id)
    if done:
        return {"state": "done", "asset": str(done["id"])}
    state, audio = _voice_line(conn, policy, run=run, batch=batch, take=take, creator=creator, pins=pins, meta=meta)
    if state:
        return state
    seconds = float((audio["meta"] or {}).get("seconds") or 0)
    lines = (audio["meta"] or {}).get("lines") or []
    params = talk_params(policy, take)
    limit = params.get("max_seconds")
    if limit is not None and not take.get("max_segment_seconds") and seconds > float(limit):
        return {"state": "failed", "reason": f"the voice line is {seconds:.1f} s; one take is at most {limit} s"}
    if take.get("max_segment_seconds") and not all("start_s" in ln and "end_s" in ln for ln in lines):
        return {"state": "failed", "reason": "the voice line has no sentence timings to cut the chain at"}
    parts = windows(lines, seconds, take.get("max_segment_seconds"))
    if not lines:   # a reused line without timings, or none: one part, the whole script
        parts = [{"start_s": 0.0, "end_s": seconds, "text": " ".join(str(take["script"]).split())}]
    videos = []
    for k, part in enumerate(parts):
        job = _job(conn, segment_key(run, item_id, k))
        if job is None:
            start = anchors_uri
            if k > 0:   # the chain: from the previous segment's last frame
                from .production.motion import _last_frame
                start = _last_frame(conn, videos[-1])
            job_id = segment_job(conn, policy, run=run, item_id=item_id, take=take, creator=creator, k=k,
                                 n=len(parts), part=part, audio_uri=audio["uri"], start_uri=start,
                                 item_batch=item_batch, meta=meta,
                                 pools=take.get("pools") or pins.get(policy.family(take["kind"])),
                                 params=params)
            job = conn.execute("SELECT id, status, error FROM jobs WHERE id = %s", (job_id,)).fetchone()
        if job["status"] in FAILED:
            return {"state": "failed", "reason": f"segment {k} failed: {job['error']}"[:300]}
        if job["status"] != "done":
            return {"state": "waiting", "on": f"segment {k} of {len(parts)}"}
        video = _output(conn, job["id"], "video")
        if not video:
            return {"state": "failed", "reason": f"segment {k} finished without a video"}
        videos.append({**video, "job_id": job["id"]})
    if store is None:
        return {"state": "waiting", "on": "a store to join and measure the take"}
    asset = finish(conn, policy, run=run, item_id=item_id, take=take, creator=creator, videos=videos, parts=parts,
                   audio=audio, store=store, llm=llm, meta=meta)
    return {"state": "done", "asset": str(asset["id"])}


def segment_job(conn, policy: Policy, *, run: str, item_id: str, take: dict, creator, k: int, n: int, part: dict,
                audio_uri: str | None, start_uri: str, item_batch: str, pools, meta: dict, params: dict):
    from .production.common import _with_crops
    text = part["text"]
    payload = {"image_uri": start_uri, "audio_uri": audio_uri, "script": text, "prompt": take["prompt"],
               "scene": f"{take.get('scene') or take['prompt']} Speaking to camera: {text}",
               "keyframes": int(take.get("keyframes") or policy.get("review", "video", "keyframes", default=4)),
               "seed": int(digest(f"{run}:{item_id}:{k}")[:7], 16), "params": params,
               "breadth": {**meta, "item": item_id, "take": {"segment": k, "of": n}}}
    if take.get("max_segment_seconds"):   # a chain: this segment's part of the shared line
        payload["audio_window"] = [part["start_s"], part["end_s"]]
        payload["segment"] = {"index": k, "of": n}
    if audio_uri is None:   # a graph that makes its own sound: as long as the script takes to say
        payload.pop("audio_uri")
        payload["params"] = {**params, "seconds": round(float(part["end_s"]) - float(part["start_s"]), 2)}
    if take.get("graph"):   # its spec travels with the job (a worker's baked policy may predate the graph)
        payload.update(graph_fields(policy, take["graph"]))
    _with_crops(payload, policy)
    return queue.enqueue(conn, take["kind"], payload, batch=item_batch, pools=pools, creator_id=creator["id"],
                         idempotency_key=segment_key(run, item_id, k),
                         max_attempts=int(policy.get("queue", "max_attempts", default=3)))


# ------------------------------------------------------------------------------------------------ the finished take

def finish(conn, policy: Policy, *, run: str, item_id: str, take: dict, creator, videos: list[dict], parts: list[dict],
           audio: dict, store, llm=None, meta: dict | None = None) -> dict:
    """The take as one video (a chain's segments joined end to end, as production's chain join does), measured, and
    stored as one asset (meta.role long_take)."""
    from . import media
    c = cfg(policy)
    timeout = float(policy.get("generation", "talk_long", "join_timeout_s", default=600))
    with tempfile.TemporaryDirectory(prefix="mw-take-") as d:
        d = Path(d)
        paths = [store.get_file(v["uri"], d / f"segment_{i:02d}.mp4") for i, v in enumerate(videos)]
        if len(paths) == 1:   # one take: the job's own video, recorded again as the take
            final, uri = paths[0], videos[0]["uri"]
            sha, size = hashlib.sha256(final.read_bytes()).hexdigest(), final.stat().st_size
        else:
            final = media.concat(paths, d / "take.mp4", timeout=timeout, policy=policy)
            stored = store.put_file(final, f"assets/{creator['id']}/breadth-{run}/{item_id}-take.mp4")
            uri, sha, size = stored.uri, stored.sha256, stored.bytes
        seconds = media.duration(final) or 0.0
        points = checkpoints(c.get("checkpoints_s") or [0], seconds)
        frames = [frame_at(final, t, d / f"at_{i}.png") for i, t in enumerate(points)]
        drift = colour_drift([f.read_bytes() for f in frames], int(c.get("colour_sample_px", 64)))
        identity = identity_at(conn, policy, llm, store, creator, frames, d, charge_to=videos[-1]["job_id"]) \
            if llm is not None and c.get("identity", True) else None
    segments = []
    for v, part in zip(videos, parts):
        att = conn.execute("SELECT coalesce(sum(gpu_seconds), 0) AS s, max(gpu_type) AS gpu FROM attempts "
                           "WHERE job_id = %s AND outcome = 'done'", (v["job_id"],)).fetchone()
        segments.append({"job_id": str(v["job_id"]), "uri": v["uri"], "window": [part["start_s"], part["end_s"]],
                         "sync": (v.get("meta") or {}).get("sync"), "render_seconds": float(att["s"] or 0),
                         "gpu": att["gpu"], "model": (v.get("meta") or {}).get("model"),
                         "quantization": (v.get("meta") or {}).get("quantization"),
                         "bench": (v.get("meta") or {}).get("bench")})
    corr = [s["sync"]["corr"] for s in segments if isinstance(s.get("sync"), dict) and s["sync"].get("corr") is not None]
    metrics = {"seconds": round(seconds, 2), "voice_seconds": (audio.get("meta") or {}).get("seconds"),
               "segments": len(segments), "checkpoints_s": points,
               "delta_e_vs_first": drift, "identity": identity,
               "lip_sync_corr": corr, "lip_sync_corr_min": min(corr) if corr else None,
               "render_seconds": round(sum(s["render_seconds"] for s in segments), 1)}
    metrics["render_seconds_per_output_second"] = round(metrics["render_seconds"] / seconds, 1) if seconds else None
    row = conn.execute(
        """INSERT INTO assets (creator_id, kind, uri, sha256, bytes, meta, recipe)
           VALUES (%s, 'video', %s, %s, %s, %s, %s) RETURNING id, uri""",
        (creator["id"], uri, sha, size,
         Jsonb({"role": ROLE, "breadth": {**(meta or {}), "item": item_id}, "take": metrics, "segments": segments,
                "audio_uri": audio["uri"]}),
         Jsonb({"kind": take["kind"], "profile": take.get("profile"),
                "join": "chain" if len(videos) > 1 else "single take", "segments": [v["uri"] for v in videos]})),
    ).fetchone()
    log.info("long take %s/%s: %s", run, item_id, metrics)
    return row


def checkpoints(points: list, seconds: float) -> list[float]:
    """The checkpoints inside the take (the last frame stands in for one at or past its end), in order, unique."""
    last = max(0.0, float(seconds) - 0.1)
    out: list[float] = []
    for p in points:
        t = round(min(float(p), last), 2)
        if t not in out:
            out.append(t)
    return out


def frame_at(video: Path, t: float, dest: Path, timeout: float = 120) -> Path:
    from .media import ffmpeg
    proc = subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(video),
                           "-frames:v", "1", str(dest)], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not dest.exists():
        raise RuntimeError(f"no frame at {t} s: {proc.stderr[-400:]}")
    return dest


_D65 = (0.95047, 1.0, 1.08883)   # the sRGB white point (CIE 1931 2 degrees), XYZ with Y = 1


def srgb_to_lab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """One 8-bit sRGB colour in CIELAB (D65), by the standard sRGB and CIE formulas."""
    def lin(c: float) -> float:
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(float(c)) for c in rgb)
    xyz = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b, 0.2126729 * r + 0.7151522 * g + 0.0721750 * b,
           0.0193339 * r + 0.1191920 * g + 0.9503041 * b)

    def f(t: float) -> float:
        return t ** (1 / 3) if t > (6 / 29) ** 3 else t / (3 * (6 / 29) ** 2) + 4 / 29
    fx, fy, fz = (f(v / w) for v, w in zip(xyz, _D65))
    return 116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)


def mean_lab(image_bytes: bytes, sample_px: int = 64) -> tuple[float, float, float]:
    """The frame's mean CIELAB colour: the frame shrunk to `sample_px` on its long side (box filter), each pixel
    in Lab, averaged."""
    import io

    from PIL import Image
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    scale = float(sample_px) / max(im.size)
    if scale < 1:
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.BOX)
    labs = [srgb_to_lab(p) for p in im.getdata()]
    n = len(labs)
    return tuple(sum(c[i] for c in labs) / n for i in range(3))   # type: ignore[return-value]


def ciede2000(lab1, lab2) -> float:
    """CIEDE2000 colour difference (Sharma, Wu, Dalal 2005), kL = kC = kH = 1."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if C1p else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if C2p else 0.0
    dLp, dCp = L2 - L1, C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    else:
        dhp = h2p - h1p - 360 if h2p > h1p else h2p - h1p + 360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))
    Lpm, Cpm = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hpm = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hpm = (h1p + h2p) / 2
    else:
        hpm = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    T = (1 - 0.17 * math.cos(math.radians(hpm - 30)) + 0.24 * math.cos(math.radians(2 * hpm))
         + 0.32 * math.cos(math.radians(3 * hpm + 6)) - 0.20 * math.cos(math.radians(4 * hpm - 63)))
    dtheta = 30 * math.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(Cpm ** 7 / (Cpm ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / math.sqrt(20 + (Lpm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cpm, 1 + 0.015 * Cpm * T
    Rt = -math.sin(math.radians(2 * dtheta)) * Rc
    return math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def colour_drift(frames: list[bytes], sample_px: int = 64) -> list[float]:
    """Each frame's colour difference from the first (CIEDE2000 of the mean Lab colours; 0 for the first)."""
    labs = [mean_lab(b, sample_px) for b in frames]
    return [round(ciede2000(labs[0], lab), 2) for lab in labs]


def identity_at(conn, policy: Policy, llm, store, creator, frames: list[Path], workdir: Path, *, charge_to=None):
    """The face identity (1-10) at each checkpoint frame: the video judge's screen model, per keyframe, against the
    creator's references (review.video.reference_images). None when the judge gave no per-frame scores."""
    from .review import common
    from .review import video as video_review
    c = policy.get("review", "video", default={}) or {}
    role, effort, _ = common.stages(c)[0]
    row = conn.execute("SELECT id, bible, anchors FROM creators WHERE id = %s", (creator["id"],)).fetchone()
    refs = common.reference_images(row, store, workdir, int(c.get("reference_images", 3)))
    with common.charged_to(llm, charge_to):
        r = video_review._judge(llm, policy, role, effort, row, refs, None,
                                [(f.read_bytes(), "image/png") for f in frames], None, per_frame=True)
    return {"per_checkpoint": r.identity_per_frame, "worst": r.identity, "model": llm.model_for(role)}


def report(conn, run: str, store=None, url_seconds: int = 3600) -> dict[str, dict]:
    """The run's finished takes by item: their numbers and a link to each video."""
    out = {}
    for r in conn.execute("SELECT id, uri, meta FROM assets WHERE kind = 'video' AND meta->>'role' = %s "
                          "AND meta->'breadth'->>'run' = %s ORDER BY created_at", (ROLE, run)).fetchall():
        m = r["meta"] or {}
        url = None
        if store is not None:
            try:
                url = store.public_url(r["uri"], seconds=url_seconds)
            except Exception:  # noqa: BLE001 - the numbers still show
                url = None
        out[(m.get("breadth") or {}).get("item")] = {"uri": r["uri"], "url": url, **(m.get("take") or {}),
                                                     "segments_detail": m.get("segments")}
    return out
