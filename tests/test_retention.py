"""Storage retention: rejected media goes after the window; anything chosen, referenced or recent stays."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from psycopg.types.json import Jsonb

from manyworlds import retention
from manyworlds.deploy import autoscale

from brain_helpers import creator

NOW = datetime.now(timezone.utc)


def _asset(conn, store, tmp_path, cid, job, kind, *, days_old, review=None):
    p = tmp_path / f"{uuid.uuid4().hex}.bin"
    p.write_bytes(b"x" * 100)
    uri = store.put_file(p, f"a/{p.name}").uri
    aid = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, created_at) "
                       "VALUES (%s, %s, %s, %s, 'x', 100, %s) RETURNING id",
                       (job, cid, kind, uri, NOW - timedelta(days=days_old))).fetchone()["id"]
    if review is not None:
        conn.execute("INSERT INTO reviews (asset_id, gate, passed) VALUES (%s, 'identity', %s)", (aid, review))
    return aid, uri


def _exists(uri):
    from urllib.parse import urlparse
    from urllib.request import url2pathname
    return Path(url2pathname(urlparse(uri).path)).exists()


def test_rejected_media_past_the_window_is_deleted_and_everything_else_kept(conn, policy, store, tmp_path):
    days = policy.get("storage", "retention_days")
    cid = creator(conn)
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('image.zimage', 'done', %s, '{}') "
                       "RETURNING id", (cid,)).fetchone()["id"]
    clip_job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('comfy.wan_i2v', 'done', %s, "
                            "'{}') RETURNING id", (cid,)).fetchone()["id"]
    gone, gone_uri = _asset(conn, store, tmp_path, cid, job, "image", days_old=days + 1, review=False)
    chosen, chosen_uri = _asset(conn, store, tmp_path, cid, job, "image", days_old=days + 1, review=True)
    named, named_uri = _asset(conn, store, tmp_path, cid, job, "image", days_old=days + 1, review=False)
    recent, recent_uri = _asset(conn, store, tmp_path, cid, job, "image", days_old=1, review=False)
    clip, clip_uri = _asset(conn, store, tmp_path, cid, clip_job, "video", days_old=days + 1, review=False)
    key, key_uri = _asset(conn, store, tmp_path, cid, clip_job, "keyframe", days_old=days + 1)
    conn.execute("INSERT INTO posts (creator_id, format, body) VALUES (%s, 'photo', %s)",
                 (cid, Jsonb({"images": [{"uri": named_uri}]})))            # a post still names it: keep
    out = retention.run(conn, policy, store, now=NOW)
    assert out["deleted"] == 3 and out["kept_referenced"] == 1 and out["freed_bytes"] == 300
    for uri, kept in ((gone_uri, False), (clip_uri, False), (key_uri, False), (chosen_uri, True), (named_uri, True),
                      (recent_uri, True)):
        assert _exists(uri) == kept, uri
    deleted = {r["id"] for r in conn.execute("SELECT id FROM assets WHERE deleted_at IS NOT NULL")}
    assert deleted == {gone, clip, key}
    assert conn.execute("SELECT count(*) n FROM reviews WHERE asset_id = %s", (gone,)).fetchone()["n"] == 1  # provenance


def test_retention_runs_once_a_day_on_the_always_on_loop(conn, policy, store):
    assert autoscale.storage_retention(conn, policy, store=store, now=NOW) == []      # nothing to delete, recorded
    assert not retention.due(conn, policy, now=NOW + timedelta(hours=1))
    assert retention.due(conn, policy, now=NOW + timedelta(hours=policy.get("storage", "every_hours") + 1))
