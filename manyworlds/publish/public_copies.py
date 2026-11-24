"""The JPEG copies made for Instagram (`public/` in the bucket, publish.instagram.public_jpeg) are deleted once they
are no longer needed: `publishing.public_copy_keep_days` after their post was published on Instagram, or held or
dropped after trying (Instagram fetches a copy when its container is made; a container lasts a day). A copy's key
follows from the image it was made from (instagram.public_key), so nothing else records them. A purged post is marked
in its body (`public_copies_purged_at`, publishing's key, never production's) and not visited again. Run from the
autoscaler's housekeeping, which runs whatever the master switch says (it only deletes)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from ..policy import Policy
from .instagram import public_key


def _cfg(policy: Policy) -> dict:
    return policy.get("publishing", default={}) or {}


def candidates(conn, policy: Policy, *, now: datetime) -> list[dict]:
    """Posts whose copies are due: published on Instagram, or held or dropped after an Instagram attempt, before the
    cutoff; not purged yet."""
    cfg = _cfg(policy)
    cutoff = now - timedelta(days=float(cfg.get("public_copy_keep_days", 7)))
    return [dict(r) for r in conn.execute(
        """SELECT p.id, p.body FROM posts p
           WHERE NOT p.body ? 'public_copies_purged_at'
             AND (EXISTS (SELECT 1 FROM publications u WHERE u.post_id = p.id AND u.platform = 'instagram'
                          AND u.published_at < %(cutoff)s)
                  OR (p.status IN ('held', 'dropped') AND p.body ? 'ig_attempts' AND p.updated_at < %(cutoff)s))
           ORDER BY p.updated_at LIMIT %(limit)s""",
        {"cutoff": cutoff, "limit": int(cfg.get("public_copy_purge_batch", 200))}).fetchall()]


def keys(body: dict) -> list[str]:
    """The bucket keys of a post's copies: one per image, and the cover's."""
    uris = [im.get("uri") for im in body.get("images") or [] if isinstance(im, dict)]
    uris.append((body.get("cover") or {}).get("uri"))
    return sorted({public_key(u) for u in uris if u})


def purge(conn, policy: Policy, store, *, now: datetime | None = None) -> dict:
    """Delete due copies. `store` is the object store or a callable returning it (made only when something is due).
    A post whose copies could not all be deleted is left for the next run."""
    now = now or datetime.now(timezone.utc)
    due = candidates(conn, policy, now=now)
    deleted, posts, failed = 0, 0, 0
    if not due:
        return {"posts": 0, "deleted": 0, "failed": 0}
    store = store() if callable(store) else store
    for post in due:
        body = post["body"] or {}
        try:
            for key in keys(body):
                store.delete(store.uri_for(key))   # a missing object is fine (ObjectStore.delete)
                deleted += 1
        except Exception:  # noqa: BLE001 - a storage error leaves the post for the next run
            failed += 1
            continue
        conn.execute("UPDATE posts SET body = body || %s WHERE id = %s",
                     (Jsonb({"public_copies_purged_at": now.isoformat()}), post["id"]))
        posts += 1
    return {"posts": posts, "deleted": deleted, "failed": failed}
