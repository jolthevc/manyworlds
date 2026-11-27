"""Live end-to-end checks on real data (owner 2026-11-27: demonstrated function over passing unit tests).

On 2026-11-26/27 four silent failures passed ~2,000 tests: the ComfyUI picker returned source clips as renders, the
lip-sync gate passed chance-level correlations, own-content reads silently read nothing, and one stray label killed
an annotation job. Each check here asks the database whether a stage really did its work, over a recent window:

    done_without_output     media jobs marked done with no output asset (they ran, but made nothing)
    output_is_input         an output asset byte-identical to an asset the job was given (nothing was rendered)
    render_without_provenance  ComfyUI videos without the render record comfy.verified writes (an old or bypassed
                            path made them)
    sync_pass_near_chance   lip-sync passes that barely beat the voice shifted away (null margin, report-only gate)
    empty_reads             annotations whose coded attributes are all unknown or missing (a read that saw nothing)
    spend_without_output    research spend in the window with nothing written by that agent (a stage that pays and
                            produces nothing)
    unattributed_gpu        finished attempts that used GPU seconds with no cost entry for their job

`run(conn, policy)` returns {check: {count, examples, severity}}; `alert(conn, policy, out)` tells the owner once a
day per failing check (notify.owner_event). `mw verify` prints it; the chief of staff runs it daily. Thresholds and
windows are policy `verify`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .policy import Policy


def cfg(policy: Policy) -> dict:
    return policy.get("verify", default={}) or {}


def _since(policy: Policy, now: datetime) -> datetime:
    return now - timedelta(hours=float(cfg(policy).get("window_hours", 48)))


def _rows(conn, sql: str, args) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def done_without_output(conn, policy: Policy, since) -> list[dict]:
    kinds = [f"{p}%" for p in cfg(policy).get("media_kinds") or []]
    return _rows(conn, """SELECT j.id::text, j.kind, j.batch FROM jobs j WHERE j.status = 'done' AND j.updated_at >= %s
                          AND j.kind LIKE ANY(%s) AND NOT EXISTS (SELECT 1 FROM assets a WHERE a.job_id = j.id)
                          ORDER BY j.updated_at DESC LIMIT 50""", (since, kinds))


def output_is_input(conn, policy: Policy, since) -> list[dict]:
    """An output whose sha256 equals that of an asset whose uri the job's payload names (any input field)."""
    return _rows(conn, """SELECT j.id::text, j.kind, j.batch, o.uri AS output FROM jobs j
                          JOIN assets o ON o.job_id = j.id AND o.kind IN ('video', 'image', 'audio')
                          JOIN assets i ON i.sha256 = o.sha256 AND i.id <> o.id AND i.job_id <> j.id
                          WHERE o.created_at >= %s AND position(i.uri IN j.payload::text) > 0
                          LIMIT 50""", (since,))


def render_without_provenance(conn, policy: Policy, since) -> list[dict]:
    kinds = [f"{p}%" for p in cfg(policy).get("comfy_kinds") or []]
    return _rows(conn, """SELECT j.id::text, j.kind, j.batch FROM jobs j JOIN assets a ON a.job_id = j.id AND a.kind = 'video'
                          WHERE a.created_at >= %s AND j.kind LIKE ANY(%s) AND NOT (a.meta ? 'render')
                          AND coalesce(j.payload->'params'->>'model', '') <> 'longcat_native'
                          LIMIT 50""", (since, kinds))


def sync_pass_near_chance(conn, policy: Policy, since) -> list[dict]:
    floor = float(cfg(policy).get("sync_min_margin_report", 0.1))
    return _rows(conn, """SELECT r.asset_id::text, (r.details->'null'->>'margin')::float AS margin, r.score
                          FROM reviews r WHERE r.gate = 'lip_sync' AND r.passed AND r.created_at >= %s
                          AND r.details ? 'null' AND (r.details->'null'->>'margin')::float < %s LIMIT 50""",
                 (since, floor))


def empty_reads(conn, policy: Policy, since) -> list[dict]:
    rows = _rows(conn, """SELECT id, post_id::text, own_asset_id::text, observed->'coded' AS coded FROM reference_annotations
                          WHERE created_at >= %s AND schema = 'iia_v2'""", (since,))
    out = []
    for r in rows:
        vals = [v for v in (r["coded"] or {}).values() if v not in (None, "", "unknown", [], ["unknown"])]
        if not vals:
            out.append({"annotation": r["id"], "post": r["post_id"] or r["own_asset_id"]})
    return out[:50]


def spend_without_output(conn, policy: Policy, since) -> list[dict]:
    """Research agents that spent in the window: each with what it should have written (policy verify.outputs:
    agent -> SQL counting its rows since %s). An agent that spent above min_usd and wrote nothing is reported."""
    out = []
    floor = float(cfg(policy).get("spend_min_usd", 0.02))
    for agent, sql in (cfg(policy).get("outputs") or {}).items():
        usd = float(conn.execute("SELECT coalesce(sum(usd), 0) u FROM cost_entries WHERE agent = %s AND created_at >= %s",
                                 (agent, since)).fetchone()["u"])
        if usd < floor:
            continue
        made = int(conn.execute(sql, (since,)).fetchone()["n"])
        if made == 0:
            out.append({"agent": agent, "usd": round(usd, 4), "rows": made})
    return out


def unattributed_gpu(conn, policy: Policy, since) -> list[dict]:
    return _rows(conn, """SELECT a.job_id::text, j.kind, a.gpu_seconds FROM attempts a JOIN jobs j ON j.id = a.job_id
                          WHERE a.finished_at >= %s AND coalesce(a.gpu_seconds, 0) > 0 AND j.kind NOT LIKE 'brain.%%'
                          AND j.kind NOT LIKE 'finish.%%'
                          AND NOT EXISTS (SELECT 1 FROM cost_entries c WHERE c.job_id = a.job_id) LIMIT 50""", (since,))


CHECKS = {"done_without_output": (done_without_output, "error"),
          "output_is_input": (output_is_input, "error"),
          "render_without_provenance": (render_without_provenance, "warning"),
          "sync_pass_near_chance": (sync_pass_near_chance, "warning"),
          "empty_reads": (empty_reads, "error"),
          "spend_without_output": (spend_without_output, "error"),
          "unattributed_gpu": (unattributed_gpu, "warning")}


def run(conn, policy: Policy, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    since = _since(policy, now)
    out = {}
    for name, (fn, severity) in CHECKS.items():
        try:
            with conn.transaction():
                rows = fn(conn, policy, since)
            out[name] = {"count": len(rows), "severity": severity, "examples": rows[:5]}
        except Exception as e:  # noqa: BLE001 - a check that can't run is itself reported
            out[name] = {"count": None, "severity": "error", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"since": since.isoformat(), "checks": out,
            "failing": sorted(k for k, v in out.items() if v.get("count") or v.get("error"))}


def alert(conn, policy: Policy, result: dict, *, now: datetime | None = None) -> int:
    """One owner event per failing check per day."""
    from . import notify
    day = (now or datetime.now(timezone.utc)).date().isoformat()
    n = 0
    for name in result.get("failing") or []:
        c = result["checks"][name]
        notify.owner_event(conn, "verification_failed", subject={"check": name},
                           payload={"count": c.get("count"), "severity": c.get("severity"),
                                    "examples": c.get("examples"), "error": c.get("error")},
                           dedupe_key=f"verification_failed:{name}:{day}")
        n += 1
    return n
