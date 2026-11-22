"""Storage retention: rejected media is deleted after a window, so storage doesn't grow with every candidate.

Each post generates several candidates per image and retries per clip; at 1,000 creators the rejected ones alone
would add ~1.8 TB a month (2026-11-22 estimate), kept forever. Once a day the always-on loop deletes the files of
assets older than `storage.retention_days` that a review rejected (and the keyframes and voice line of a rejected
clip), never one a post, a creator (reference pack, anchors, voice) or a publication still names. The row stays
with `deleted_at` set: its recipe and reviews are the provenance and the learning data, the pixels are not.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from .policy import Policy

URI = re.compile(r"(?:s3|file)://[^\"\s,\]\}]+")


def _referenced(conn) -> set[str]:
    """Every object URI a post, a creator or a publication names (their JSON documents, scanned as text)."""
    refs: set[str] = set()
    for sql in ("SELECT body::text AS t FROM posts", "SELECT anchors::text || bible::text AS t FROM creators"):
        for r in conn.execute(sql):
            refs.update(URI.findall(r["t"] or ""))
    return refs


def candidates(conn, policy: Policy, *, now: datetime) -> list[dict]:
    cfg = policy.get("storage", default={}) or {}
    cutoff = now - timedelta(days=float(cfg.get("retention_days", 30)))
    return [dict(r) for r in conn.execute(
        """WITH rejected AS (
               SELECT DISTINCT a.id, a.job_id, a.kind FROM assets a JOIN reviews r ON r.asset_id = a.id
               WHERE a.deleted_at IS NULL AND a.created_at < %(cutoff)s AND NOT r.passed
                 AND a.kind IN ('image', 'video'))
           SELECT a.id, a.uri, a.kind, a.bytes FROM assets a
           WHERE a.deleted_at IS NULL AND a.created_at < %(cutoff)s
             AND (a.id IN (SELECT id FROM rejected)
                  OR (a.kind IN ('keyframe', 'audio')
                      AND a.job_id IN (SELECT job_id FROM rejected WHERE kind = 'video')))
           ORDER BY a.created_at LIMIT %(limit)s""",
        {"cutoff": cutoff, "limit": int(cfg.get("max_deletes_per_run", 5000))}).fetchall()]


def due(conn, policy: Policy, *, now: datetime) -> bool:
    last = conn.execute("SELECT max(created_at) AS t FROM decisions WHERE kind = 'storage_retention'").fetchone()["t"]
    return last is None or now - last >= timedelta(hours=float((policy.get("storage", default={}) or {})
                                                                  .get("every_hours", 24)))


def run(conn, policy: Policy, store, *, now: datetime | None = None) -> dict:
    """Delete the files of rejected assets past the window. Returns what was removed."""
    now = now or datetime.now(timezone.utc)
    refs = _referenced(conn)
    deleted, freed, kept, failed = 0, 0, 0, 0
    for a in candidates(conn, policy, now=now):
        if a["uri"] in refs:
            kept += 1
            continue
        try:
            store.delete(a["uri"])
        except Exception:  # noqa: BLE001 - a storage error leaves the asset for the next run
            failed += 1
            continue
        conn.execute("UPDATE assets SET deleted_at = %s WHERE id = %s", (now, a["id"]))
        deleted += 1
        freed += int(a["bytes"] or 0)
    out = {"deleted": deleted, "freed_bytes": freed, "kept_referenced": kept, "failed": failed}
    conn.execute("INSERT INTO decisions (kind, inputs, output, policy_version, agent) VALUES "
                 "('storage_retention', %s, %s, %s, 'retention')",
                 (Jsonb({"retention_days": (policy.get("storage", default={}) or {}).get("retention_days", 30)}),
                  Jsonb(out), policy.version))
    return out
