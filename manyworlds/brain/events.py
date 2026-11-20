"""Events: the Brain reacts to what happens, not only to its schedules (spec 4.3).

Detectors are code over durable state; each event is raised once (`dedupe_key`) and routed by policy
`brain.events.routes` (an agent name, or `owner`). Detectors read metrics through manyworlds.brain.evidence in
the creator's mode, so sandbox events are about SIMULATED outcomes and say so.

  breakout  a post published in the last `window_hours` whose latest reach is >= `breakout_ratio` x the median
            reach of the creator's previous `baseline_posts` posts at a similar age (within `age_tolerance`)
  drop      the creator's last `drop_posts` posts with outcomes all below `drop_ratio` x that baseline
  qa_surge  more than `qa_drop_share` of the creator's posts in the last day were dropped by production

Rate limits: at most `max_per_type_per_creator_per_day` events of a type per creator, and routing enqueues
at most one job per creator and route per tick (one analysis covers an event storm). Breakouts are also
shown to the next showrunner batch (`for_planning`) as a follow-up opportunity.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from .. import queue
from ..policy import Policy
from . import evidence


def _cfg(policy: Policy) -> dict:
    return policy.get("brain", "events", default={}) or {}


def _reach_at(rows: list[dict], age_s: float, tolerance: float) -> float | None:
    near = [r for r in rows if r.get("post_age_s") is not None and abs(r["post_age_s"] - age_s) <= age_s * tolerance]
    if not near:
        return None
    return float(min(near, key=lambda r: abs(r["post_age_s"] - age_s))["metrics"].get("reach") or 0)


def _posts(conn, creator_id, mode, limit: int) -> list[dict]:
    rows = conn.execute(
        "SELECT id, caption, format, published_at FROM posts WHERE creator_id = %s AND published_at IS NOT NULL "
        "AND status IN ('published', 'simulated') ORDER BY published_at DESC LIMIT %s", (creator_id, limit)).fetchall()
    out = []
    for p in rows:
        m = evidence.metrics(conn, p["id"], mode=mode)
        if m:
            out.append({**p, "metrics": m, "latest": max(m, key=lambda r: r.get("post_age_s") or 0)})
    return out


def detect(conn, policy: Policy, creator_id, *, now: datetime) -> list[dict]:
    cfg = _cfg(policy)
    mode = evidence.mode_for(conn, creator_id)
    window = timedelta(hours=float(cfg.get("window_hours", 72)))
    n_base, tol = int(cfg.get("baseline_posts", 10)), float(cfg.get("age_tolerance", 0.5))
    posts = _posts(conn, creator_id, mode, int(cfg.get("lookback_posts", 60)))
    found = []

    def baseline(i: int, age_s: float) -> float | None:
        prev = [_reach_at(p["metrics"], age_s, tol) for p in posts[i + 1:i + 1 + n_base]]
        prev = [x for x in prev if x is not None]
        return statistics.median(prev) if len(prev) >= max(3, n_base // 2) else None

    for i, p in enumerate(posts):
        if now - p["published_at"] > window:
            break
        age = float(p["latest"].get("post_age_s") or 0)
        reach, base = float(p["latest"]["metrics"].get("reach") or 0), baseline(i, age)
        if base and reach >= float(cfg.get("breakout_ratio", 3.0)) * base:
            found.append({"type": "breakout", "key": f"breakout:{p['id']}",
                          "subject": {"creator_id": str(creator_id), "post_id": str(p["id"])},
                          "payload": {"reach": reach, "baseline": base, "ratio": round(reach / base, 2), "age_s": age,
                                      "caption": p["caption"], "format": p["format"], "mode": mode}})
    k = int(cfg.get("drop_posts", 4))
    if len(posts) >= k + 3:
        recent = []
        for i, p in enumerate(posts[:k]):
            age = float(p["latest"].get("post_age_s") or 0)
            base = baseline(i, age)
            recent.append(base is not None and float(p["latest"]["metrics"].get("reach") or 0) < float(
                cfg.get("drop_ratio", 0.5)) * base)
        if all(recent):
            found.append({"type": "drop", "key": f"drop:{creator_id}:{posts[0]['id']}",
                          "subject": {"creator_id": str(creator_id)},
                          "payload": {"posts": [str(p["id"]) for p in posts[:k]], "mode": mode}})
    day = conn.execute("SELECT count(*) FILTER (WHERE status = 'dropped') AS dropped, count(*) AS n FROM posts "
                       "WHERE creator_id = %s AND updated_at > %s", (creator_id, now - timedelta(days=1))).fetchone()
    if day["n"] >= int(cfg.get("qa_min_posts", 4)) and day["dropped"] / day["n"] > float(cfg.get("qa_drop_share", 0.5)):
        found.append({"type": "qa_surge", "key": f"qa_surge:{creator_id}:{now.date()}",
                      "subject": {"creator_id": str(creator_id)},
                      "payload": {"dropped": day["dropped"], "of": day["n"]}})
    return found


def tick(conn, policy: Policy, *, now: datetime | None = None) -> list[dict]:
    """Detect for every active creator, record new events within the rate limits, and route them."""
    now = now or datetime.now(timezone.utc)
    cfg = _cfg(policy)
    per_day = int(cfg.get("max_per_type_per_creator_per_day", 3))
    routes = cfg.get("routes") or {}
    raised = []
    for c in conn.execute("SELECT id FROM creators WHERE status <> 'retired'").fetchall():
        routed_now: dict[str, object] = {}
        for ev in detect(conn, policy, c["id"], now=now):
            recent = conn.execute("SELECT count(*) n FROM events WHERE creator_id = %s AND type = %s AND created_at > %s",
                                  (c["id"], ev["type"], now - timedelta(days=1))).fetchone()["n"]
            if recent >= per_day:
                continue
            route = routes.get(ev["type"], "owner")
            row = conn.execute(
                "INSERT INTO events (type, subject, payload, routed_to, creator_id, dedupe_key, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (dedupe_key) WHERE dedupe_key IS NOT NULL DO NOTHING "
                "RETURNING id", (ev["type"], Jsonb(ev["subject"]), Jsonb(ev["payload"]), route, c["id"], ev["key"],
                                 now)).fetchone()
            if not row:
                continue   # raised before
            job = None
            if route != "owner":
                job = routed_now.get(route) or queue.enqueue(
                    conn, f"brain.{route}", {"creator_id": str(c["id"]), "reason": f"event {ev['type']}",
                                             "event_id": row["id"]},
                    creator_id=c["id"], idempotency_key=f"event:{route}:{c['id']}:{now:%Y-%m-%d}:{ev['type']}")
                routed_now[route] = job
                conn.execute("UPDATE events SET job_id = %s WHERE id = %s", (job, row["id"]))
            raised.append({"type": ev["type"], "creator_id": str(c["id"]), "event_id": row["id"], "routed_to": route})
    return raised


def for_planning(conn, creator_id, *, now: datetime | None = None, days: int = 7) -> list[dict]:
    """Recent breakouts for the showrunner: what worked unusually well, as a follow-up opportunity."""
    now = now or datetime.now(timezone.utc)
    rows = conn.execute("SELECT payload, created_at FROM events WHERE creator_id = %s AND type = 'breakout' "
                        "AND created_at > %s ORDER BY created_at DESC LIMIT 5",
                        (creator_id, now - timedelta(days=days))).fetchall()
    return [{"caption": r["payload"].get("caption"), "format": r["payload"].get("format"),
             "ratio": r["payload"].get("ratio"),
             "label": evidence.LABEL if r["payload"].get("mode") == "sandbox" else None} for r in rows]
