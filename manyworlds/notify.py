"""Tell the owner what needs them, on their phone.

Everything that needs the owner is an `events` row routed to 'owner' (a held post, a failed launch, a Brain step
that crashed, a GPU machine stopped at its spend cap...) or a pending `approvals` row. `deliver` pushes each one
once, from the always-on autoscaler loop, so alerts work while operations are paused.

Channel: ntfy (https://ntfy.sh, free, no account). The topic in NTFY_TOPIC works like a password: anyone who
knows it can read the alerts, so it lives in secrets and messages carry ids and reasons, never secrets.
Without a topic nothing is sent and nothing is marked, so alerts start flowing once the topic is set.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Callable

from psycopg.types.json import Jsonb

from .policy import Policy

log = logging.getLogger("manyworlds.notify")

Sender = Callable[[str, str, str], None]   # (title, body, priority)


def owner_event(conn, type: str, *, subject: dict | None = None, payload: dict | None = None,
                creator_id=None, dedupe_key: str | None = None) -> int | None:
    """Record something the owner should hear about. With a dedupe key it is recorded once; returns the
    event id, or None when it was already recorded."""
    row = conn.execute(
        "INSERT INTO events (type, subject, payload, routed_to, creator_id, dedupe_key) "
        "VALUES (%s, %s, %s, 'owner', %s, %s) ON CONFLICT DO NOTHING RETURNING id",
        (type, Jsonb(subject or {}), Jsonb(payload or {}), creator_id, dedupe_key)).fetchone()
    return row["id"] if row else None


def from_autoscaler(conn, policy: Policy, actions: list[dict], now: datetime) -> int:
    """Turn the autoscaler actions that need the owner into owner events (once a day per kind of trouble)."""
    cfg = policy.get("notify", default={})
    stop_reasons = set(cfg.get("machine_stop_reasons", []))
    day = f"{now:%Y-%m-%d}"
    n = 0
    for a in actions:
        act, reason = a.get("action"), a.get("reason")
        if act in ("stopped", "stop_failed") and (reason in stop_reasons or act == "stop_failed"):
            ev = ("machine_" + act, {"group": a.get("group")}, {"reason": reason}, f"{act}:{a.get('group')}")
        elif act == "not_launching":
            ev = ("not_launching", {"reason": reason}, {k: v for k, v in a.items() if k not in ("action", "reason")},
                  f"not_launching:{reason}:{day}")
        elif act == "provider_unfunded":
            ev = ("provider_unfunded", {"provider": a.get("provider")}, {}, f"unfunded:{a.get('provider')}:{day}")
        elif act == "tick_failed":
            ev = ("autoscaler_failed", {}, {"error": a.get("error")}, f"tick_failed:{day}")
        else:
            continue
        n += owner_event(conn, ev[0], subject=ev[1], payload=ev[2], dedupe_key=ev[3]) is not None
    return n


def ntfy_sender(policy: Policy) -> Sender | None:
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return None
    server = str(policy.get("notify", "ntfy_server", default="https://ntfy.sh")).rstrip("/")
    timeout = float(policy.get("notify", "timeout_seconds", default=10))

    def send(title: str, body: str, priority: str) -> None:
        req = urllib.request.Request(f"{server}/{topic}", data=body.encode(), method="POST",
                                     headers={"Title": title.encode("ascii", "replace").decode(),
                                              "Priority": priority, "Tags": "manyworlds"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
    return send


def _line(subject: dict, payload: dict, limit: int) -> str:
    from .deploy.providers import redact
    parts = {**(subject or {}), **(payload or {})}
    return redact(json.dumps(parts, default=str, separators=(", ", ": ")))[:limit]


def pending(conn, policy: Policy, now: datetime) -> list[dict]:
    """What the owner hasn't been told yet, oldest first: owner events, then pending approvals."""
    since = now - timedelta(hours=float(policy.get("notify", "max_age_hours", default=24)))
    quiet = list(policy.get("notify", "quiet", default=[]) or [])
    events = conn.execute(
        """SELECT 'event' AS source, id::text AS id, type, subject, payload, created_at FROM events
           WHERE routed_to = 'owner' AND notified_at IS NULL AND created_at >= %s AND NOT (type = ANY(%s))
           ORDER BY created_at""", (since, quiet)).fetchall()
    approvals = conn.execute(
        """SELECT 'approval' AS source, id::text AS id, 'approval_' || kind AS type, subject,
                  jsonb_build_object('reason', reason, 'requested_by', requested_by) AS payload, created_at
           FROM approvals WHERE status = 'pending' AND notified_at IS NULL AND created_at >= %s
           ORDER BY created_at""", (since,)).fetchall()
    return list(events) + list(approvals)


def deliver(conn, policy: Policy, *, send: Sender | None = None, now: datetime | None = None) -> list[dict]:
    """Push everything the owner hasn't been told (one message each, or one summary past `max_per_run`) and mark
    it told. Items older than `max_age_hours` are never pushed (no flood when the channel is first set)."""
    now = now or datetime.now(timezone.utc)
    send = send or ntfy_sender(policy)
    if send is None:
        return []
    cfg = policy.get("notify", default={})
    priorities = cfg.get("priority", {}) or {}
    limit, body_chars = int(cfg.get("max_per_run", 5)), int(cfg.get("body_chars", 400))
    items = pending(conn, policy, now)
    if not items:
        return []
    if len(items) > limit:
        kinds: dict[str, int] = {}
        for it in items:
            kinds[it["type"]] = kinds.get(it["type"], 0) + 1
        top = "high" if any(priorities.get(t) in ("high", "urgent") for t in kinds) else "default"
        send(f"Manyworlds: {len(items)} things need you",
             ", ".join(f"{t} x{n}" for t, n in sorted(kinds.items())) + " (mw status / mw brain approvals)", top)
        sent = items
    else:
        sent = []
        for it in items:
            try:
                send(f"Manyworlds: {it['type'].replace('_', ' ')}", _line(it["subject"], it["payload"], body_chars),
                     str(priorities.get(it["type"], "default")))
            except Exception as e:  # noqa: BLE001 - try the rest; unsent items go next round
                log.warning("notification failed (%s): %s", it["type"], type(e).__name__)
                continue
            sent.append(it)
    for source, table in (("event", "events"), ("approval", "approvals")):
        ids = [it["id"] for it in sent if it["source"] == source]
        if ids:
            cast = "bigint" if table == "events" else "uuid"
            conn.execute(f"UPDATE {table} SET notified_at = %s WHERE id = ANY(%s::{cast}[])", (now, ids))
    return [{"type": it["type"], "id": it["id"]} for it in sent]
