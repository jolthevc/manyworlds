"""The production capability card: what our production makes reliably and cheaply now, per post format (code, no model).

Casting designs creators around it (manyworlds.brain.casting: the feasibility gate), the talent manager plans careers
with it and the CEO reads it. Per format:

    reliability        accepted / judged posts (accepted: intelligence.feedback.accepted_statuses, dropped: its
                       dropped_statuses) over strategy.capability_card.window_days
    usd_per_accepted   every ledger entry booked to those posts or their jobs (attributed GPU time, LLM, data...)
                       over accepted posts
    status, limits     from policy
    source             `measured` once a format has min_tried judged posts in the window, else `policy` (its defaults)

`refresh` stores the card in the `capabilities` table (name `format:<fmt>`, interface `capability_card`: the table's
stats column was made for observed acceptance and cost), each chief-of-staff tick; `card` reads it back, falling back to
policy for a format with no row. The GPU families' throughput (deploy.report.throughput: $ per accepted output) rides
along as context.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from ..policy import Policy

INTERFACE = "capability_card"


def _cfg(policy: Policy) -> dict:
    return policy.get("strategy", "capability_card", default={}) or {}


def _statuses(policy: Policy) -> tuple[list[str], list[str]]:
    fb = policy.get("intelligence", "feedback", default={}) or {}
    return (list(fb.get("accepted_statuses") or ["ready", "approved", "scheduled", "published"]),
            list(fb.get("dropped_statuses") or ["dropped"]))


def measure(conn, policy: Policy, *, now: datetime | None = None) -> dict:
    """{format: {tried, accepted, dropped, usd, reliability, usd_per_accepted}} over the window, real and test posts
    alike (a breadth or casting render is as good a measure of production as a live one); simulated posts are not
    renders and are left out."""
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=float(_cfg(policy).get("window_days", 30)))
    ok, bad = _statuses(policy)
    rows = conn.execute(
        """SELECT p.format, p.status,
                  (SELECT coalesce(sum(c.usd), 0) FROM cost_entries c
                    WHERE c.post_id = p.id OR c.job_id IN (SELECT j.id FROM jobs j WHERE j.post_id = p.id)) AS usd
           FROM posts p WHERE p.updated_at > %s AND p.status = ANY(%s)""", (since, ok + bad)).fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        m = out.setdefault(r["format"], {"tried": 0, "accepted": 0, "dropped": 0, "usd": 0.0})
        m["tried"] += 1
        m["accepted" if r["status"] in ok else "dropped"] += 1
        m["usd"] += float(r["usd"] or 0)
    for m in out.values():
        m["usd"] = round(m["usd"], 4)
        m["reliability"] = round(m["accepted"] / m["tried"], 3) if m["tried"] else None
        m["usd_per_accepted"] = round(m["usd"] / m["accepted"], 4) if m["accepted"] else None
    return out


def compose(policy: Policy, measured: dict) -> dict:
    """The card: each policy format, measured where the evidence is enough, else its default (and the thin
    measurement beside it); a format production made that policy doesn't list is shown measured only."""
    cfg = _cfg(policy)
    min_tried = int(cfg.get("min_tried", 12))
    card = {}
    for fmt, d in (cfg.get("formats") or {}).items():
        m = measured.get(fmt) or {}
        enough = int(m.get("tried") or 0) >= min_tried
        card[fmt] = {"reliability": m["reliability"] if enough else d.get("reliability"),
                     "usd_per_accepted": (m.get("usd_per_accepted") if enough and m.get("usd_per_accepted") is not None
                                          else d.get("usd_per_accepted")),
                     "status": d.get("status"), "limits": d.get("limits"),
                     "source": "measured" if enough else "policy", "n": int(m.get("tried") or 0)}
        if m and not enough:
            card[fmt]["measured_so_far"] = {k: m.get(k) for k in ("tried", "accepted", "reliability")}
    for fmt, m in measured.items():
        if fmt not in card:
            card[fmt] = {"reliability": m.get("reliability"), "usd_per_accepted": m.get("usd_per_accepted"),
                         "status": None, "limits": None, "source": "measured", "n": int(m.get("tried") or 0)}
    return card


def refresh(conn, policy: Policy, *, now: datetime | None = None) -> dict:
    """Measure, compose and store the card (idempotent: a rerun writes the same rows). Returns the card."""
    card = compose(policy, measure(conn, policy, now=now))
    for fmt, entry in card.items():
        conn.execute(
            """INSERT INTO capabilities (name, interface, config, stats) VALUES (%s, %s, %s, %s)
               ON CONFLICT (name) DO UPDATE SET stats = excluded.stats, config = excluded.config""",
            (f"format:{fmt}", INTERFACE, Jsonb({"policy_version": policy.version}),
             Jsonb({**entry, "at": (now or datetime.now(timezone.utc)).isoformat()})))
    return card


def card(conn, policy: Policy) -> dict:
    """The stored card, with policy defaults for any format that has no row yet."""
    stored = {}
    if conn is not None:
        for r in conn.execute("SELECT name, stats FROM capabilities WHERE interface = %s", (INTERFACE,)).fetchall():
            stored[r["name"].split(":", 1)[1]] = r["stats"] or {}
    out = compose(policy, {})
    out.update(stored)
    return out


def reliability(card_: dict, fmt: str) -> float:
    return float((card_.get(fmt) or {}).get("reliability") or 0.0)


def feasibility(card_: dict, mix: dict | None) -> float | None:
    """0-10: a content mix's expected acceptance (each format's share times its reliability); None without a mix.
    A format the card doesn't know counts as unreliable (0)."""
    shares = {k: max(float(v or 0), 0.0) for k, v in (mix or {}).items()}
    total = sum(shares.values())
    if not total:
        return None
    return round(10 * sum(s / total * reliability(card_, f) for f, s in shares.items()), 2)


def brief(card_: dict) -> str:
    """One compact line per format for a prompt."""
    lines = []
    for fmt, c in sorted(card_.items(), key=lambda kv: -float(kv[1].get("reliability") or 0)):
        usd = c.get("usd_per_accepted")
        lines.append(f"- {fmt}: {c.get('status') or 'measured'}, acceptance {c.get('reliability')}, "
                     f"${usd if usd is not None else '?'} per accepted post ({c.get('source')}, n={c.get('n', 0)})"
                     + (f"; {c['limits']}" if c.get("limits") else ""))
    return "\n".join(lines)


def families(conn, policy: Policy, *, now: datetime | None = None) -> list[dict]:
    """The GPU families' throughput over the card's window (deploy.report.throughput): $ per accepted output."""
    from ..deploy import report
    hours = float(_cfg(policy).get("window_days", 30)) * 24
    return [{k: r.get(k) for k in ("family", "pool", "billed_usd", "accepted_outputs", "usd_per_accepted_output",
                                   "first_pass_rate")} for r in report.throughput(conn, hours=hours, now=now)]


def as_json(card_: dict) -> str:
    return json.dumps({f: {k: c.get(k) for k in ("reliability", "usd_per_accepted", "status", "source")}
                       for f, c in card_.items()}, separators=(",", ":"))
