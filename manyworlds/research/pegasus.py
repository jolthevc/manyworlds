"""Native audiovisual reads of OUR OWN videos with Twelve Labs Pegasus (owner 2026-11-26: accelerate testing on original
content; no Gemini). Off as shipped; never for third-party media.

Guards, all checked before any upload (`refused`):
  - policy intelligence.pegasus.enabled, and training_opt_out_confirmed: the owner has turned off training on
    customer data in the Twelve Labs account settings (Terms of Use §4(a): training is ON by default; no API flag);
  - TWELVELABS_API_KEY set (an account the owner creates);
  - the asset is ours: a video asset of one of our posts (never a reference post, never research media);
  - the run's cap (max_usd_per_run) and the monthly research ceiling (research.budget).

Flow per video (API v1.3, docs.twelvelabs.io, read 2026-11-26): upload as an asset (POST /assets, multipart,
method=direct, <= 200 MB), wait until ready, one async analysis (POST /analyze/tasks, model pegasus1.6,
analysis_mode time_based_metadata, response_format segment_definitions: performance beats and the take's overall
read), wait until ready, then DELETE the asset (force) and confirm it is gone (404), whatever happened, so nothing of
ours stays with the processor longer than the read. The read goes into the spine: one `beat` unit per segment
(start_s/end_s, seq) under the asset's shot unit, its fields as attribute_values (method model, annotator
pegasus1.6). Cost: the video's minutes at usd_per_minute plus output tokens at usd_per_mtok_out, into the ledger
(agent research:pegasus; list prices from a third-party capture until a real usage report confirms them).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from .. import ledger
from ..policy import Policy

BASE = "https://api.twelvelabs.io/v1.3"
AGENT = "research:pegasus"

BEAT_FIELDS = [
    {"name": "kind", "type": "string", "description": "what changes at this moment",
     "enum": ["opening", "gesture", "expression_change", "head_move", "gaze_shift", "pause", "emphasis",
              "camera_move", "cut", "reaction", "settle", "ending"]},
    {"name": "expression", "type": "string", "description": "her facial expression, in a few plain words"},
    {"name": "gesture", "type": "string", "description": "hands and body, in a few plain words, or none"},
    {"name": "voice", "type": "string", "description": "how the line is delivered here (pace, emphasis, tone)"},
    {"name": "energy", "type": "integer", "description": "0 very calm to 4 very high"},
    {"name": "in_sync", "type": "boolean", "description": "the mouth matches the words heard here"},
]
OVERALL_FIELDS = [
    {"name": "hook", "type": "string", "description": "what in the first two seconds would stop a viewer, or none"},
    {"name": "naturalness", "type": "integer", "description": "1 obviously artificial to 10 indistinguishable from "
                                                              "real phone footage"},
    {"name": "charisma", "type": "integer", "description": "1 flat to 10 magnetic"},
    {"name": "lip_sync", "type": "integer", "description": "1 out of sync to 10 perfectly in sync"},
    {"name": "ending", "type": "string", "description": "how the take ends (lands, trails off, freezes, cuts)"},
    {"name": "defects", "type": "array", "description": "visible or audible defects", "items": {"type": "string"}},
]


def cfg(policy: Policy) -> dict:
    return policy.get("intelligence", "pegasus", default={}) or {}


def refused(conn, policy: Policy, asset_id) -> str | None:
    """Why this asset may not be read now, or None."""
    c = cfg(policy)
    if not c.get("enabled"):
        return "intelligence.pegasus.enabled is false"
    if not c.get("training_opt_out_confirmed"):
        return ("the owner has not confirmed the Twelve Labs training opt-out "
                "(intelligence.pegasus.training_opt_out_confirmed)")
    if not os.environ.get("TWELVELABS_API_KEY"):
        return "TWELVELABS_API_KEY is not set"
    row = conn.execute("SELECT a.kind, j.post_id FROM assets a JOIN jobs j ON j.id = a.job_id WHERE a.id = %s",
                       (asset_id,)).fetchone()
    if not row or row["kind"] != "video" or not row["post_id"]:
        return "only video assets of our own posts are read (never research or third-party media)"
    return None


# --- HTTP -----------------------------------------------------------------------------------------------------------
def _req(method: str, path: str, *, body: bytes | None = None, ctype: str | None = None, timeout: float = 120):
    headers = {"x-api-key": os.environ["TWELVELABS_API_KEY"]}
    if ctype:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(BASE + path, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw) if raw else {}
        except ValueError:
            return e.code, {"message": raw[:300].decode(errors="replace")}


def _multipart(fields: dict, file_field: str, path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{path.name}\"\r\n"
                 f"Content-Type: video/mp4\r\n\r\n".encode() + path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _wait(path: str, done: tuple[str, ...], *, every_s: float, timeout_s: float, sleep=time.sleep) -> dict:
    t0 = time.monotonic()
    while True:
        status, body = _req("GET", path)
        if status >= 400:
            raise RuntimeError(f"Twelve Labs {path}: {status} {str(body)[:300]}")
        if body.get("status") in done:
            return body
        if time.monotonic() - t0 > timeout_s:
            raise RuntimeError(f"Twelve Labs {path} not done after {timeout_s:.0f} s")
        sleep(every_s)


def analyse(video: Path, c: dict, *, sleep=time.sleep) -> dict:
    """Upload, read, delete. {segments, usage, duration_s, deleted}; the asset is deleted even when the read fails."""
    body, ctype = _multipart({"method": "direct"}, "file", video)
    status, up = _req("POST", "/assets", body=body, ctype=ctype, timeout=float(c.get("upload_timeout_s", 300)))
    if status >= 400 or not up.get("_id"):
        raise RuntimeError(f"Twelve Labs upload failed: {status} {str(up)[:300]}")
    asset = up["_id"]
    out: dict = {"asset": asset}
    try:
        ready = _wait(f"/assets/{asset}", ("ready", "failed"), every_s=float(c.get("poll_s", 5)),
                      timeout_s=float(c.get("timeout_s", 900)), sleep=sleep)
        if ready.get("status") != "ready":
            raise RuntimeError(f"Twelve Labs asset failed: {(ready.get('error') or {}).get('message')}")
        out["duration_s"] = ready.get("duration")
        task = {"model_name": str(c.get("model", "pegasus1.6")), "video": {"type": "asset_id", "asset_id": asset},
                "analysis_mode": "time_based_metadata", "temperature": float(c.get("temperature", 0.2)),
                "max_tokens": int(c.get("max_tokens", 4096)),
                "min_segment_duration": float(c.get("min_segment_s", 2)),
                "response_format": {"type": "segment_definitions", "segment_definitions": [
                    {"id": "beat", "description": "each moment where her performance, the camera or the edit "
                                                  "changes: gestures, expression changes, pauses, emphasis, cuts, the "
                                                  "opening and the ending", "fields": BEAT_FIELDS},
                    {"id": "overall", "description": "the whole take as one segment", "fields": OVERALL_FIELDS}]}}
        status, t = _req("POST", "/analyze/tasks", body=json.dumps(task).encode(), ctype="application/json")
        if status >= 400 or not t.get("task_id"):
            raise RuntimeError(f"Twelve Labs analysis failed to start: {status} {str(t)[:300]}")
        res = _wait(f"/analyze/tasks/{t['task_id']}", ("ready", "failed", "canceled"),
                    every_s=float(c.get("poll_s", 5)), timeout_s=float(c.get("timeout_s", 900)), sleep=sleep)
        if res.get("status") != "ready":
            raise RuntimeError(f"Twelve Labs analysis {res.get('status')}: {(res.get('error') or {}).get('message')}")
        r = res.get("result") or {}
        data = r.get("data")
        out["segments"] = json.loads(data) if isinstance(data, str) else (data or {})
        out["usage"] = r.get("usage") or {}
        out["warning"] = (res.get("error") or {}).get("message")
    finally:
        _req("DELETE", f"/assets/{asset}?force=true")
        gone, _ = _req("GET", f"/assets/{asset}")
        out["deleted"] = gone == 404
    return out


def cost(c: dict, duration_s: float | None, usage: dict) -> float:
    minutes = float(duration_s or 0) / 60.0
    return round(minutes * float(c.get("usd_per_minute", 0.029))
                 + float(usage.get("output_tokens") or 0) / 1e6 * float(c.get("usd_per_mtok_out", 7.5)), 6)


def store(conn, shot_unit: str, segments: dict, annotator: str) -> int:
    """The read in the spine: a beat unit per segment (time-ordered) under the shot, the overall read on the shot."""
    from . import spine
    n = 0
    for i, seg in enumerate(sorted(segments.get("beat") or [], key=lambda s: float(s.get("start_time") or 0))):
        unit = spine.upsert_unit(conn, source="own", level="beat", ref_table="pegasus", ref_id=shot_unit,
                                 parent_id=shot_unit, platform="own", seq=i, start_s=seg.get("start_time"),
                                 end_s=seg.get("end_time"))
        for k, v in (seg.get("metadata") or {}).items():
            num = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
            spine.put_value(conn, unit, f"av_{k}", value=None if num is not None else json.dumps(v)
                            if isinstance(v, (list, dict)) else str(v), num=num, method="model", annotator=annotator)
            n += 1
    for seg in (segments.get("overall") or [])[:1]:
        for k, v in (seg.get("metadata") or {}).items():
            num = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
            spine.put_value(conn, shot_unit, f"av_{k}", value=None if num is not None else json.dumps(v)
                            if isinstance(v, (list, dict)) else str(v), num=num, method="model", annotator=annotator)
            n += 1
    return n


def read_assets(conn, policy: Policy, store_, workdir: Path, asset_ids: list[str], *, usd: float | None = None,
                analyse_=analyse) -> dict:
    """Pegasus reads of these own video assets, within the run cap and the monthly ceiling. Each one refused, read or
    failed is reported; every billed read goes to the ledger, failed ones included."""
    from . import budget, spine
    c = cfg(policy)
    cap = min(float(usd if usd is not None else c.get("max_usd_per_run", 3.0)), budget.left(conn, policy))
    spent, out = 0.0, []
    for aid in asset_ids:
        why = refused(conn, policy, aid)
        if why:
            out.append({"asset": aid, "refused": why})
            continue
        est = float(c.get("est_usd_per_clip", 0.03))
        if spent + est > cap:
            out.append({"asset": aid, "skipped": "run cap or monthly research ceiling"})
            continue
        a = conn.execute("SELECT a.uri, j.post_id FROM assets a JOIN jobs j ON j.id = a.job_id WHERE a.id = %s",
                         (aid,)).fetchone()
        local = store_.get_file(a["uri"], Path(workdir) / f"{aid}.mp4")
        try:
            got = analyse_(Path(local), c)
        except Exception as e:  # noqa: BLE001 - one clip's failure is reported; the rest are read
            out.append({"asset": aid, "error": f"{type(e).__name__}: {str(e)[:300]}"})
            continue
        finally:
            Path(local).unlink(missing_ok=True)
        usd_ = cost(c, got.get("duration_s"), got.get("usage") or {})
        spent += usd_
        ledger.record(conn, category="hosted_api", provider="twelvelabs", quantity=float(got.get("duration_s") or 0),
                      unit="video_second", usd=usd_, agent=AGENT)
        post = spine.upsert_unit(conn, source="own", level="post", ref_table="posts", ref_id=a["post_id"], platform="own")
        shot = spine.upsert_unit(conn, source="own", level="shot", ref_table="assets", ref_id=aid, parent_id=post,
                                 platform="own", medium="video")
        n = store(conn, shot, got.get("segments") or {}, str(c.get("model", "pegasus1.6")))
        out.append({"asset": aid, "values": n, "usd": usd_, "deleted": got.get("deleted"), "warning": got.get("warning")})
    return {"cap_usd": round(cap, 2), "spent_usd": round(spent, 4), "reads": out}
