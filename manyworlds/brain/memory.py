"""Creator memory: the facts that keep a creator's life coherent across months of posts.

Facts are versioned rows in `creator_memory`: places, routines, tastes, opinions and running jokes,
people, the signature item, and canon (small facts posts have established). A single-valued fact
(`key`, e.g. home) replaces the previous one by closing it (`valid_until`). The showrunner reads the
open facts; scheduled posts add their canon, and a dropped post takes its canon back (`retract_post`).
"""

from __future__ import annotations

import re
import uuid

from ..creator import profile

# Bible fields -> memory kind (the rendered bible carries these from the creator lab and profile).
_BIBLE_KINDS = {"home_and_places": "place", "interests": "taste", "opinions_and_running_jokes": "opinion",
                "routines": "routine", "people": "person"}


def add(conn, creator_id, kind: str, text: str, *, source_post=None, key: str | None = None) -> uuid.UUID:
    replaced = None
    if key:
        closed = conn.execute("UPDATE creator_memory SET valid_until = now() WHERE creator_id = %s AND key = %s "
                              "AND valid_until IS NULL RETURNING id", (creator_id, key)).fetchall()
        replaced = closed[0]["id"] if closed else None
    return conn.execute("INSERT INTO creator_memory (creator_id, kind, text, source_post, key, replaces) "
                        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                        (creator_id, kind, text.strip(), source_post, key, replaced)).fetchone()["id"]


def retract_post(conn, post_ids) -> int:
    """A dropped post never happened: close the canon it introduced (as `add` closes a replaced fact), and reopen
    the fact a keyed canon had replaced when nothing else holds that key now. Every path that drops a post calls
    this. Returns how many canon facts were closed."""
    ids = [p for p in (post_ids or []) if p is not None]
    if not ids:
        return 0
    closed = conn.execute(
        "UPDATE creator_memory SET valid_until = now() WHERE source_post = ANY(%s::uuid[]) AND kind = 'canon' "
        "AND valid_until IS NULL RETURNING creator_id, key, replaces", ([str(p) for p in ids],)).fetchall()
    for r in closed:
        if r["key"] and r["replaces"]:
            conn.execute(
                "UPDATE creator_memory SET valid_until = NULL WHERE id = %s AND valid_until IS NOT NULL "
                "AND NOT EXISTS (SELECT 1 FROM creator_memory o WHERE o.creator_id = %s AND o.key = %s "
                "AND o.valid_until IS NULL)", (r["replaces"], r["creator_id"], r["key"]))
    return len(closed)


def _items(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [p.strip() for p in re.split(r"[;\n]", value) if p.strip()]
    return []


def seed_from_profile(conn, creator_id) -> int:
    """Facts from the creator's bible and current profile; skips any text already open. Returns how many were added."""
    bible = conn.execute("SELECT bible FROM creators WHERE id = %s", (creator_id,)).fetchone()["bible"] or {}
    candidates = [(kind, t) for field, kind in _BIBLE_KINDS.items() for t in _items(bible.get(field))]
    current = profile.current(conn, creator_id)
    look = (current[1] if current else {}).get("look") or {}
    if look.get("signature_item"):
        candidates.append(("signature", look["signature_item"]))
    # Looks for particular moments: remembered so the showrunner can write them into a scene, never part
    # of the default appearance every photo prompt carries.
    candidates += [("look", t) for t in _items(look.get("situational_looks"))]
    have = {r["text"] for r in conn.execute("SELECT text FROM creator_memory WHERE creator_id = %s "
                                            "AND valid_until IS NULL", (creator_id,)).fetchall()}
    added = 0
    for kind, text in candidates:
        if text not in have:
            add(conn, creator_id, kind, text, key="signature" if kind == "signature" else None)
            have.add(text)
            added += 1
    return added


def facts(conn, creator_id, *, limit: int) -> list[dict]:
    """Every open core fact (places, routines, tastes, opinions, people, signature), then the newest
    `limit` canon facts: canon accumulates with every post and must never crowd out who they are."""
    core = conn.execute("SELECT kind, text FROM creator_memory WHERE creator_id = %s AND valid_until IS NULL "
                        "AND kind <> 'canon' ORDER BY created_at DESC, id DESC", (creator_id,)).fetchall()
    canon = conn.execute("SELECT kind, text FROM creator_memory WHERE creator_id = %s AND valid_until IS NULL "
                         "AND kind = 'canon' ORDER BY created_at DESC, id DESC LIMIT %s", (creator_id, limit)).fetchall()
    return core + canon
