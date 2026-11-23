"""Event-driven wakeups for the control-plane loops (policy `wakeups`, migration 033).

A trigger on jobs NOTIFYs 'mw_jobs' with {id, kind, status, post_id} when a job is inserted or finishes (done,
failed, dead); one on reviews NOTIFYs 'mw_reviews'. A loop that used to sleep its interval instead calls
`Listener.wait(timeout)`: it returns as soon as a notification it cares about arrives, or after the timeout.

Debounce: after the first matching notification the wait lingers `debounce_seconds` collecting more, so a burst
(20 clips finishing in one go) is one tick, not twenty. Notifications that arrive while the loop is busy ticking queue
up on the connection and end the next wait at once: at most one extra tick, never a lost wakeup.

The listener has its own autocommit connection that runs nothing but LISTEN (any query on it would hand pending
notifications to psycopg before the generator sees them). Any error (the database restarts, Neon drops the
connection) closes it and the wait falls back to plain sleep for the rest of its timeout; the next wait reconnects,
at most once per `reconnect_seconds`. The listener never raises into the loop.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable

from . import db

log = logging.getLogger("manyworlds.wakeup")

JOBS = "mw_jobs"
REVIEWS = "mw_reviews"


def accepts(job_statuses, reviews: bool) -> Callable[[str, dict], bool]:
    """A filter: job notifications whose status is in `job_statuses` (an insert has status 'queued'), and review
    notifications when `reviews`."""
    statuses = set(job_statuses or ())

    def ok(channel: str, payload: dict) -> bool:
        if channel == JOBS:
            return payload.get("status") in statuses
        if channel == REVIEWS:
            return bool(reviews)
        return False
    return ok


class Listener:
    def __init__(self, channels=(JOBS, REVIEWS), *, accept: Callable[[str, dict], bool] | None = None,
                 debounce_seconds: float = 0.0, reconnect_seconds: float = 60.0, refresh_seconds: float = 0.0,
                 connect: Callable = db.connect, sleep: Callable[[float], None] = time.sleep):
        self.channels = tuple(channels)
        self.accept = accept or (lambda channel, payload: True)
        self.debounce_seconds = float(debounce_seconds)
        self.reconnect_seconds = float(reconnect_seconds)
        # A connection dropped without a FIN (a NAT, a proxy) raises nothing: the waits just time out. Renewing it
        # every refresh_seconds bounds how long wakeups can be silently lost (0: never).
        self.refresh_seconds = float(refresh_seconds)
        self._connect = connect
        self._sleep = sleep
        self.conn = None
        self._next_try = 0.0
        self._since = 0.0

    @classmethod
    def from_policy(cls, pol, role: str, **kw) -> "Listener | None":
        """The listener for `role` (producer, reviewer) under policy `wakeups`; None when wakeups are off."""
        if not pol.get("wakeups", "enabled", default=False):
            return None
        cfg = pol.get("wakeups", role, default=None) or {}
        connect_s = pol.get("queue", "connect_timeout_seconds", default=None)
        kw.setdefault("connect", lambda: db.connect(db.direct_url(), timeout=connect_s))   # LISTEN needs no pooler
        return cls(accept=accepts(cfg.get("jobs") or (), bool(cfg.get("reviews"))),
                   debounce_seconds=float(pol.get("wakeups", "debounce_seconds", default=0)),
                   reconnect_seconds=float(pol.get("wakeups", "reconnect_seconds", default=60)),
                   refresh_seconds=float(pol.get("wakeups", "refresh_seconds", default=0)), **kw)

    def ensure(self) -> bool:
        """Connect and LISTEN if not connected (at most once per reconnect_seconds after a failure); renew a
        connection older than refresh_seconds."""
        now = time.monotonic()
        if self.conn is not None and self.refresh_seconds > 0 and now - self._since >= self.refresh_seconds:
            self.close()
        if self.conn is not None:
            return True
        if now < self._next_try:
            return False
        conn = None
        try:
            conn = self._connect()
            conn.autocommit = True
            for ch in self.channels:
                conn.execute(f'LISTEN "{ch}"')
            self.conn, self._since = conn, now
            return True
        except Exception as e:  # noqa: BLE001 - no listener: the loop sleeps as before
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
            self._next_try = now + self.reconnect_seconds
            log.warning("wakeup listener could not connect (%s); plain sleep for now", type(e).__name__)
            return False

    def close(self) -> None:
        conn, self.conn = self.conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def _matching(self, timeout: float, stop_after: int | None) -> list[dict]:
        """Notifications received within `timeout` that pass the filter (stops after `stop_after` of any kind)."""
        out = []
        for n in self.conn.notifies(timeout=max(timeout, 0.0), stop_after=stop_after):
            try:
                payload = json.loads(n.payload) if n.payload else {}
            except ValueError:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            if self.accept(n.channel, payload):
                out.append({"channel": n.channel, **payload})
        return out

    def wait(self, timeout: float) -> list[dict]:
        """Block until a matching notification (then the debounce window) or `timeout` seconds. Returns the
        matching notifications' payloads (with 'channel'); empty after a plain timeout. Never raises."""
        deadline = time.monotonic() + max(float(timeout), 0.0)
        if not self.ensure():
            self._sleep(max(deadline - time.monotonic(), 0.0))
            return []
        try:
            got: list[dict] = []
            while not got:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                got = self._matching(remaining, stop_after=1)
            if self.debounce_seconds > 0:
                got += self._matching(self.debounce_seconds, stop_after=None)
            return got
        except Exception as e:  # noqa: BLE001 - a dropped listener: sleep the rest, reconnect next time
            log.warning("wakeup listener failed (%s); reconnecting next wait", type(e).__name__)
            self.close()
            self._sleep(max(deadline - time.monotonic(), 0.0))
            return []


def wait(listener: Listener | None, timeout: float, sleep: Callable[[float], None] = time.sleep) -> list[dict]:
    """`listener.wait(timeout)`, or a plain sleep without one (wakeups off)."""
    if listener is None:
        sleep(max(float(timeout), 0.0))
        return []
    return listener.wait(timeout)
