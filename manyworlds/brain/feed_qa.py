"""Feed QA: the last check before ideas become posts, on the feed as a whole.

The evaluator scores each idea alone; a grid can still go wrong: the same outfit three posts in a row, a
caption that's nearly last week's, an idea that doesn't sound like the creator. Before a batch is
scheduled, each idea is checked against the creator's recent posts (policy `brain.qa.window`) and the
ideas already accepted from the batch:
  repetition   an attribute value (outfit, setting...) already used `max_repeat` times in that window
  originality  caption or scene text too similar (difflib ratio >= `similarity`) to a recent one
  voice        an idea-rubric score below its floor (`min_scores`, e.g. naturalness)
Rule-based, no LLM call. Experiment ideas are exempt: dropping some of an arm would bias the comparison.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from ..policy import Policy


def _cfg(policy: Policy) -> dict:
    return policy.get("brain", "qa", default={}) or {}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).strip()


def recent(conn, creator_id, window: int) -> list[dict]:
    rows = conn.execute(
        """SELECT p.caption, p.body->'scenes' AS scenes, i.attributes FROM posts p
           LEFT JOIN ideas i ON i.post_id = p.id
           WHERE p.creator_id = %s AND p.status <> 'dropped'
           ORDER BY coalesce(p.scheduled_for, p.created_at) DESC LIMIT %s""", (creator_id, window)).fetchall()
    return [{"caption": r["caption"] or "", "scenes": r["scenes"] or [], "attributes": r["attributes"] or {}}
            for r in rows]


def problem(policy: Policy, idea: dict, scores: dict, seen: list[dict]) -> str | None:
    """Why this idea shouldn't join the feed given what's already in it (`seen`), or None."""
    cfg = _cfg(policy)
    for dim, floor in (cfg.get("min_scores") or {}).items():
        s = (scores or {}).get(dim)
        if s is not None and float(s) < float(floor):
            return f"{dim} {s} below {floor}: doesn't sound like them"
    attrs = idea.get("attributes") or {}
    for key in cfg.get("repeat_attributes") or []:
        v = _norm(str(attrs.get(key) or ""))
        if not v:
            continue
        n = sum(_norm(str((s["attributes"] or {}).get(key) or "")) == v for s in seen)
        if n >= int(cfg.get("max_repeat", 2)):
            return f"repetitive feed: {key} '{v}' is already in {n} of the last {len(seen)} posts"
    limit, short = float(cfg.get("similarity", 0.85)), int(cfg.get("min_chars", 20))
    cap, scenes = _norm(idea.get("caption")), _norm(" ".join(idea.get("scenes") or []))
    cap, scenes = (cap if len(cap) >= short else ""), (scenes if len(scenes) >= short else "")
    for s in seen:
        if cap and SequenceMatcher(None, cap, _norm(s["caption"])).ratio() >= limit:
            return f"unoriginal: caption nearly repeats '{s['caption'][:60]}'"
        if scenes and SequenceMatcher(None, scenes, _norm(" ".join(s["scenes"]))).ratio() >= limit:
            return "unoriginal: scene nearly repeats a recent post"
    return None


def screen(conn, policy: Policy, creator_id, rows: list[dict]) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Split a batch's evaluated ideas (rows with id, body, attributes, scores, experiment_id) into kept and
    dropped-with-reason, in order, each checked against recent posts and the ideas kept before it."""
    seen = recent(conn, creator_id, int(_cfg(policy).get("window", 12)))
    kept, dropped = [], []
    for r in rows:
        idea = {"caption": r["body"].get("caption"), "scenes": r["body"].get("scenes"), "attributes": r["attributes"]}
        why = None if r.get("experiment_id") else problem(policy, idea, r.get("scores") or {}, seen)
        if why:
            dropped.append((r, why))
            continue
        kept.append(r)
        seen = [{"caption": idea["caption"] or "", "scenes": idea["scenes"] or [], "attributes": idea["attributes"] or {}},
                *seen]
    return kept, dropped
