"""One producer per post at a time. Advancing a post reads posts.body, changes it in memory (job lists, repair and
failure counters) and writes the whole body back, so two producers on one post (the Fly producer and a manual `mw`
run) would each write over the other's counters. Each post's start and advance therefore holds a lease row named
`post:<id>` in `leases` (migration 006), and a post whose lease another producer holds is skipped for this tick.

A lease row rather than pg_try_advisory_lock: session advisory locks don't survive a transaction-mode pooler (Neon's
pgbouncer may run the unlock on another backend, 006_leases.sql). The lease lapses after
production.post_lease_seconds, so a producer that died mid-advance never blocks its post for good."""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from typing import Iterator

from ..policy import Policy
from .common import _cfg

log = logging.getLogger("manyworlds.production")


def lease_name(post_id) -> str:
    return f"post:{post_id}"


def take_post_lease(conn, policy: Policy, post_id, holder: str) -> bool:
    """Take the post's lease for `holder` unless another holder's lease is still running. True when taken."""
    seconds = float(_cfg(policy).get("post_lease_seconds", 1800))
    got = conn.execute(
        """INSERT INTO leases (name, holder, until) VALUES (%(n)s, %(h)s, now() + make_interval(secs => %(s)s))
           ON CONFLICT (name) DO UPDATE SET holder = excluded.holder, until = excluded.until
           WHERE leases.until < now() RETURNING holder""",
        {"n": lease_name(post_id), "h": holder, "s": seconds}).fetchone()
    return got is not None


def release_post_lease(conn, post_id, holder: str) -> None:
    """Give the post's lease back (only this holder's: a lease that lapsed and was taken over stays with its taker)."""
    conn.execute("DELETE FROM leases WHERE name = %s AND holder = %s", (lease_name(post_id), holder))


@contextmanager
def post_lease(conn, policy: Policy, post_id) -> Iterator[bool]:
    """`with post_lease(conn, policy, id) as held:` held is False when another producer has the post (skip it this
    tick). The lease is released on the way out, whatever happened inside; a release that fails (the database gone)
    is logged, and the lease then lapses on its own."""
    holder = uuid.uuid4().hex
    if not take_post_lease(conn, policy, post_id, holder):
        yield False
        return
    try:
        yield True
    finally:
        try:
            release_post_lease(conn, post_id, holder)
        except Exception:  # noqa: BLE001 - never hide the advance's own outcome; the lease lapses by itself
            log.exception("could not release the lease of post %s", post_id)
