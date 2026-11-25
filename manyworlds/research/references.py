"""Research references: named public creators and their posts, kept apart from production.

Owner 2026-11-25: we learn from and emulate successful creators freely, and research records may name them (handles,
profile and post links, annotations; tables reference_creators, reference_posts, reference_annotations, migration
041). Downloaded third-party media is a research reference only: it lives in the research store, under the key prefix
`research-refs/` (RESEARCH_STORAGE_URL, a private bucket the control plane and the finisher hold; GPU workers never
do), is deleted after `intelligence.retention.media_days`, and is never a generation input, LoRA or training data,
never published. A post's `licence.uses` records what more its licence permits; using it for more means copying it
into production storage as a licensed asset, a deliberate step outside this module, never a research URI in a job.

    is_research_uri(s)          a URI (or any string) that points into the research store
    check_job(kind, payload)    the guard queue.enqueue runs: only research kinds may carry research URIs
    research_store()            the research store, or None when RESEARCH_STORAGE_URL is not set
    media_key(post_id, name)    where a post's research copy goes
    forget(conn, store, who)    delete a creator and everything about them; never collect them again
    purge(conn, policy, store)  media past its date, and records past `records_days` that nothing cites
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from psycopg.types.json import Jsonb

from ..policy import Policy

PREFIX = "research-refs/"            # the key prefix of every research object (also checked by assets_not_research)
STORE_ENV = "RESEARCH_STORAGE_URL"
# The job kinds that may carry research URIs: the measurement on the finisher and the research agent. Everything
# else (image, video, talk, voice, dance, train, finish.video, publishing...) is refused at enqueue.
RESEARCH_KINDS = ("finish.measure_video", "brain.intelligence")
PLATFORMS = ("instagram", "tiktok", "youtube", "other")   # reference_creators.platform


class ResearchMediaRefused(ValueError):
    """A research reference was about to become a production input."""


def _research_bucket() -> str | None:
    url = os.environ.get(STORE_ENV) or ""
    if not url.startswith("s3://"):
        return None
    return url[len("s3://"):].split("/", 1)[0] or None


def is_research_uri(value: Any) -> bool:
    if not isinstance(value, str) or "://" not in value:
        return False
    if f"/{PREFIX}" in value:
        return True
    bucket = _research_bucket()
    return bool(bucket) and value.startswith(f"s3://{bucket}/")


def _strings(node: Any, path: str = ""):
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            yield from _strings(v, f"{path}[{i}]")


def research_uris(payload: Any) -> list[str]:
    """Where in a payload research URIs appear (dotted paths)."""
    return [p for p, s in _strings(payload) if is_research_uri(s)]


def check_job(kind: str, payload: Any) -> None:
    if kind in RESEARCH_KINDS:
        return
    found = research_uris(payload)
    if found:
        raise ResearchMediaRefused(f"{kind}: research media is a reference only, never a production input "
                                   f"(payload {', '.join(found[:5])})")


def research_store():
    from .. import storage
    url = os.environ.get(STORE_ENV)
    return storage.from_url(url) if url else None


def media_key(post_id, name: str) -> str:
    return f"{PREFIX}posts/{post_id}/{name}"


def _cfg(policy: Policy) -> dict:
    return policy.get("intelligence", "retention", default={}) or {}


def media_purge_after(policy: Policy, now: datetime | None = None) -> datetime:
    return (now or datetime.now(timezone.utc)) + timedelta(days=float(_cfg(policy).get("media_days", 180)))


# --- deletion on request ------------------------------------------------------------------------------------------
def _find(conn, who: str) -> list[dict]:
    """A creator by handle (with or without @), profile URL or one of their post URLs."""
    w = who.strip()
    rows = conn.execute(
        """SELECT DISTINCT c.* FROM reference_creators c LEFT JOIN reference_posts p ON p.creator_id = c.id
           WHERE lower(c.handle) = lower(%(h)s) OR c.profile_url = %(w)s OR p.url = %(w)s""",
        {"h": w.lstrip("@"), "w": w}).fetchall()
    return [dict(r) for r in rows]


def forget(conn, store, who: str) -> dict:
    """Delete a creator and everything we hold about them: their posts, annotations and research media, their links on
    pattern cards (the cards' evidence counts go down with them), the reference on auditions (marked, kept as a
    record of what was decided). Their handle goes on the blocklist as a salted hash, so nothing collects them again.
    Idempotent: forgetting someone already gone does nothing more."""
    from .. import notify
    from .apify import author_hash
    found = _find(conn, who)
    out = {"creators": 0, "posts": 0, "media_deleted": 0, "media_failed": 0, "cards": 0, "auditions": 0}
    for c in found:
        cid = str(c["id"])
        posts = conn.execute("SELECT id, media FROM reference_posts WHERE creator_id = %s", (c["id"],)).fetchall()
        for p in posts:
            for m in p["media"] or []:
                try:
                    if store is not None:
                        store.delete(m["uri"])
                    out["media_deleted"] += 1
                except Exception:  # noqa: BLE001 - counted; the row goes anyway, the sweep has no row to find it by
                    out["media_failed"] += 1
        post_ids = {str(p["id"]) for p in posts}
        with conn.transaction():
            _forget_rows(conn, c, cid, post_ids, out)
        out["posts"] += len(posts)
        out["creators"] += 1
    if not found and "://" not in who:          # someone we hold nothing on yet: never collect them either
        for platform in PLATFORMS:
            conn.execute("INSERT INTO research_forget (handle_hash, platform) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                         (author_hash(who.strip().lstrip("@")), platform))
        out["blocklisted"] = True
    if out["creators"]:
        notify.owner_event(conn, "research_forget", subject={"who": who}, payload=dict(out))
    return out


def _forget_rows(conn, c: dict, cid: str, post_ids: set, out: dict) -> None:
    from .apify import author_hash
    for card in conn.execute("SELECT id, sources, evidence FROM pattern_cards").fetchall():
        src = card["sources"] or []
        kept = [s for s in src if str(s.get("creator_id")) != cid and str(s.get("post_id")) not in post_ids]
        if len(kept) != len(src):
            ev = dict(card["evidence"] or {})
            ev["n_posts"] = len(kept)
            ev["n_creators"] = len({str(s.get("creator_id")) for s in kept if s.get("creator_id")})
            conn.execute("UPDATE pattern_cards SET sources = %s, evidence = %s, updated_at = now() WHERE id = %s",
                         (Jsonb(kept), Jsonb(ev), card["id"]))
            out["cards"] += 1
    for a in conn.execute("""SELECT id, "references", checks FROM auditions
                             WHERE "references" @> %s""", (Jsonb([cid]),)).fetchall():
        refs = [r for r in a["references"] if str(r) != cid]
        checks = {**(a["checks"] or {}), "reference_forgotten": True}
        conn.execute("""UPDATE auditions SET "references" = %s, checks = %s, updated_at = now() WHERE id = %s""",
                     (Jsonb(refs), Jsonb(checks), a["id"]))
        out["auditions"] += 1
    conn.execute("DELETE FROM reference_creators WHERE id = %s", (c["id"],))     # posts, annotations cascade
    conn.execute("INSERT INTO research_forget (handle_hash, platform) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                 (author_hash(c["handle"]), c["platform"]))


def forgotten(conn, platform: str, handle: str) -> bool:
    """True if this account asked to be forgotten: collection skips it."""
    from .apify import author_hash
    return conn.execute("SELECT 1 FROM research_forget WHERE platform = %s AND handle_hash = %s",
                        (platform, author_hash(handle))).fetchone() is not None


# --- retention ----------------------------------------------------------------------------------------------------
def purge(conn, policy: Policy, store, *, now: datetime | None = None) -> dict:
    """Research media past its date is deleted (the post row stays, without media); creators unseen for
    `records_days` whom no open hypothesis, audition or pattern card cites are deleted with their posts and
    annotations. Media of a creator that goes is deleted first."""
    now = now or datetime.now(timezone.utc)
    cfg = _cfg(policy)
    limit = int(cfg.get("max_deletes_per_run", 2000))
    out = {"media_deleted": 0, "media_failed": 0, "creators_deleted": 0}

    def drop_media(post) -> list:
        kept = []
        for m in post["media"] or []:
            try:
                if store is None:
                    raise RuntimeError("no research store")
                store.delete(m["uri"])
                out["media_deleted"] += 1
            except Exception:  # noqa: BLE001 - kept for the next run
                kept.append(m)
                out["media_failed"] += 1
        return kept

    for p in conn.execute("""SELECT id, media FROM reference_posts WHERE media_purge_after < %s
                             AND media <> '[]'::jsonb ORDER BY media_purge_after LIMIT %s""", (now, limit)).fetchall():
        conn.execute("UPDATE reference_posts SET media = %s WHERE id = %s", (Jsonb(drop_media(p)), p["id"]))

    cutoff = now - timedelta(days=float(cfg.get("records_days", 730)))
    stale = conn.execute(
        """SELECT c.id FROM reference_creators c
           WHERE c.last_seen < %(cutoff)s
             AND NOT EXISTS (SELECT 1 FROM auditions a JOIN theses t ON t.id = a.thesis_id
                             WHERE a."references" @> jsonb_build_array(c.id::text)
                               AND t.status NOT IN ('rejected', 'archived'))
             AND NOT EXISTS (SELECT 1 FROM pattern_cards k, jsonb_array_elements(k.sources) s
                             WHERE k.status IN ('draft', 'active') AND s->>'creator_id' = c.id::text)
           LIMIT %(limit)s""", {"cutoff": cutoff, "limit": limit}).fetchall()
    for c in stale:
        failed = False
        for p in conn.execute("SELECT id, media FROM reference_posts WHERE creator_id = %s", (c["id"],)).fetchall():
            left = drop_media(p)
            conn.execute("UPDATE reference_posts SET media = %s WHERE id = %s", (Jsonb(left), p["id"]))
            failed = failed or bool(left)
        if not failed:
            conn.execute("DELETE FROM reference_creators WHERE id = %s", (c["id"],))
            out["creators_deleted"] += 1
    return out
