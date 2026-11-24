"""Comment themes: what a creator's audience keeps saying, as a few themes the showrunner plans from.

A cheap periodic summary (role `comment_themes`, batchable: it runs as a brain job). `ensure` asks for one per creator
whose real posts gathered at least `brain.comment_themes.min_new` comments since the last summary, at most every
`every_hours`. The job reads the recent comments only through manyworlds.brain.evidence (handles and links scrubbed,
our own accounts' comments left out, commenters never shown) and writes a `comment_themes` decision: up to
`max_themes` themes (questions asked, requests, recurring reactions), each tied to a series when the comments were
on one. With no comments nothing is called, so a creator without an audience costs nothing.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, Field

from ... import queue
from ...worker.runner import NonRetryable
from .. import evidence
from ..agent import AgentContext, record_decision, register

SYSTEM = (
    "You read the comments an audience left on one creator's recent posts and name the few themes that recur: "
    "questions they ask, things they request (a part 2, a recipe, a tour, where something is from), and reactions "
    "that keep coming back. Each theme is one plain sentence in your own words, about what people want or feel; never "
    "quote a comment, never name or tag anyone, never invent a theme the comments don't show. Leave out spam, "
    "hostility and anything off-limits. When a theme belongs to one series (the comments were on its posts), name "
    "that series; otherwise leave it empty. Count roughly how many comments carry each theme.")


class Theme(BaseModel):
    kind: Literal["question", "request", "reaction"]
    text: str = Field(description="one plain sentence, in your words; no quotes, names or handles")
    series: str = Field(default="", description="the series these comments were on, or empty")
    count: int = Field(default=1, description="about how many comments carry it")


class Themes(BaseModel):
    themes: list[Theme]


def _cfg(policy) -> dict:
    return policy.get("brain", "comment_themes", default={}) or {}


def latest(conn, creator_id) -> dict | None:
    """The creator's latest themes decision: {id, created_at, output} or None."""
    return conn.execute("SELECT id, created_at, output FROM decisions WHERE kind = 'comment_themes' "
                        "AND creator_id = %s ORDER BY created_at DESC LIMIT 1", (creator_id,)).fetchone()


def for_planning(conn, policy, creator_id) -> list[dict]:
    """The latest themes, at most `shown`, largest first (empty without any)."""
    last = latest(conn, creator_id)
    themes = ((last["output"] or {}).get("themes") or []) if last else []
    return sorted(themes, key=lambda t: -int(t.get("count") or 0))[:int(_cfg(policy).get("shown", 6))]


def ensure(conn, policy, *, now: datetime | None = None) -> list:
    """A themes job for each active creator with enough new comments since its last summary."""
    now = now or datetime.now(timezone.utc)
    cfg = _cfg(policy)
    if not cfg.get("enabled"):
        return []
    out = []
    for cid in evidence.creators_with_new_comments(conn, min_new=int(cfg.get("min_new", 3)),
                                                   every=timedelta(hours=float(cfg.get("every_hours", 24))), now=now):
        out.append(queue.enqueue(conn, "brain.comment_themes", {"creator_id": str(cid)}, creator_id=cid,
                                 idempotency_key=f"comment_themes:{cid}:{now:%Y-%m-%d}"))
    return out


class CommentThemes:
    name = "comment_themes"

    def run(self, ctx: AgentContext) -> dict:
        p = ctx.job.payload or {}
        if not p.get("creator_id"):
            raise NonRetryable("comment_themes payload needs creator_id")
        done = ctx.conn.execute("SELECT output FROM decisions WHERE job_id = %s AND step = 'themes'",
                                (ctx.job.id,)).fetchone()
        if done:
            return done["output"]
        cid, cfg = p["creator_id"], _cfg(ctx.policy)
        since = ctx.now - timedelta(days=float(cfg.get("window_days", 14)))
        comments = evidence.audience_comments(ctx.conn, cid, since=since, limit=int(cfg.get("max_comments", 80)),
                                              chars=int(cfg.get("comment_chars", 240)))
        if not comments:
            return {"comments": 0, "themes": 0}   # nothing to read: no call, no cost
        role = cfg.get("role", "comment_themes")
        c = ctx.conn.execute("SELECT bible FROM creators WHERE id = %s", (cid,)).fetchone()
        content = {"creator": (c["bible"] or {}).get("display_name") if c else None,
                   "comments": [{k: v for k, v in m.items() if k != "at"} for m in comments]}
        out = ctx.llm.structured(role, system=SYSTEM, content=json.dumps(content, ensure_ascii=False), schema=Themes,
                                 effort=cfg.get("effort", "low"), creator_id=cid)
        themes = []
        for t in (out.themes or [])[:int(cfg.get("max_themes", 6))]:
            text = evidence.scrub(t.text)[:int(cfg.get("theme_chars", 200))]
            if text:
                themes.append({"kind": t.kind, "text": text, "count": max(int(t.count or 1), 1),
                               **({"series": t.series.strip()} if t.series.strip() else {})})
        output = {"themes": themes, "comments": len(comments), "since": since.isoformat()}
        record_decision(ctx, "comment_themes", step="themes", creator_id=cid, model=ctx.llm.model_for(role),
                        inputs={"comments": len(comments), "window_days": cfg.get("window_days", 14)},
                        output=output, rationale=f"{len(themes)} themes from {len(comments)} comments")
        return output


register(CommentThemes())
