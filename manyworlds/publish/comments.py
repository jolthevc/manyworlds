"""Comments on our own Instagram posts, answered in the creator's voice (policy `autonomy.comment_replies`).

collect: for each real publication younger than `comments.watch_days`, at most every `comments.poll_minutes`, the
top-level comments are read (GET /<media>/comments, newest 50) and stored once each. Commenters are kept only as
salted hashes and comment text is cleared after `comments.keep_days` (research privacy rules apply to our own
audience too). Comments from any of our own accounts are never answered: our creators never engage with each other.

respond: one LLM call per creator (role `comment_reply`) decides, for a batch of new comments, whether to reply and
what to say, in the creator's voice and within the caption voice rules. A reply is checked like a caption (banned
phrases, AI-shaped patterns), and never carries a mention, link or hashtag. Mode `draft` stores the replies for the
owner to read (`mw comments <handle>`); `autonomous` sends them (POST /<comment>/replies), at most
`comments.max_replies_per_account_day` per account. `reply_sent_at` is written before the reply is sent: a crash
after sending is reconciled by looking the reply up, never by replying again.

The creator is openly AI-generated: asked whether they are real or an AI, a reply says so plainly, in their voice.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel, Field

from ..policy import Policy
from . import instagram


class Reply(BaseModel):
    comment: str = Field(description="the comment's key, e.g. c3")
    action: Literal["reply", "skip"]
    text: str = Field(default="", description="the reply, when action is reply")
    reason: str = Field(default="", description="a few words: why skip, or what the reply does")


class Replies(BaseModel):
    items: list[Reply]


SYSTEM = (
    "You answer comments on your own Instagram posts as the creator described below: an openly AI-generated "
    "character with a life, taste and voice of their own. Reply like a real person who is good at this: short, "
    "specific to what the commenter said, often funny or warm, never generic, never a customer-service tone, never "
    "fishing for engagement. Skip anything spammy, hostile, sexual, about minors, political or on an off-limits "
    "topic, anything asking to meet, buy, DM or move off the platform, and anything only a real person could answer. "
    "Skip a comment that needs no answer (a lone emoji often does not). If someone asks whether you are real, human "
    "or an AI, say plainly that you are an AI-generated character, in your own voice; never claim or imply real-world "
    "experiences, a body, or a life offline. Never mention or tag anyone, never write links or hashtags, never "
    "promise anything. Keep each reply under {max_chars} characters.\n\n")


def _cfg(policy: Policy) -> dict:
    return policy.get("comments", default={}) or {}


def mode(policy: Policy) -> str:
    return str(policy.get("autonomy", "comment_replies", default="disabled"))


def _ours(conn) -> tuple[set[str], set[str]]:
    """Every account of ours (ids and handles): their comments are never answered."""
    rows = conn.execute("SELECT external_id, handle FROM accounts WHERE platform = 'instagram'").fetchall()
    return ({str(r["external_id"]) for r in rows if r["external_id"]},
            {r["handle"].lower() for r in rows if r["handle"]})


def collect(conn, policy: Policy, *, now: datetime | None = None) -> dict:
    if mode(policy) == "disabled":
        return {}
    from ..research.apify import author_hash
    now, cfg = now or datetime.now(timezone.utc), _cfg(policy)
    instagram._use(policy)
    out = {"read": 0, "new": 0, "errors": 0}
    conn.execute("UPDATE comments SET text = NULL, updated_at = now() WHERE text IS NOT NULL AND created_at < %s",
                 (now - timedelta(days=float(cfg.get("keep_days", 30))),))
    ids, handles = _ours(conn)
    pubs = conn.execute(
        """SELECT u.id, u.external_id, p.creator_id FROM publications u JOIN posts p ON p.id = u.post_id
           WHERE u.platform = 'instagram' AND NOT u.simulated AND u.external_id IS NOT NULL AND u.published_at > %s
             AND (u.comments_polled_at IS NULL OR u.comments_polled_at < %s)
           ORDER BY u.comments_polled_at NULLS FIRST LIMIT %s""",
        (now - timedelta(days=float(cfg.get("watch_days", 7))),
         now - timedelta(minutes=float(cfg.get("poll_minutes", 30))), int(cfg.get("polls_per_tick", 20)))).fetchall()
    max_age = timedelta(hours=float(cfg.get("max_age_hours", 48)))
    for u in pubs:
        try:
            creds = instagram.account(conn, u["creator_id"])
            data = instagram.graph("GET", f"/{u['external_id']}/comments",
                                   {"fields": "id,text,timestamp,username,from"}, creds).get("data") or []
        except (instagram.GraphError, instagram.AccountNotConnected):
            out["errors"] += 1
            continue
        conn.execute("UPDATE publications SET comments_polled_at = %s WHERE id = %s", (now, u["id"]))
        for c in data:
            out["read"] += 1
            who = (c.get("from") or {})
            name = (c.get("username") or who.get("username") or "").lower()
            at = datetime.fromisoformat(c["timestamp"].replace("+0000", "+00:00")) if c.get("timestamp") else now
            status, reason = "new", None
            if str(who.get("id") or "") in ids or name in handles:
                status, reason = "skipped", "one of our accounts"
            elif now - at > max_age:
                status, reason = "skipped", "too old to answer"
            row = conn.execute(
                """INSERT INTO comments (platform, external_id, publication_id, creator_id, author_hash, text,
                                         commented_at, status, reason)
                   VALUES ('instagram', %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (platform, external_id) DO NOTHING
                   RETURNING id""",
                (str(c["id"]), u["id"], u["creator_id"], author_hash(name or None), (c.get("text") or "")[:2000],
                 at, status, reason)).fetchone()
            out["new"] += int(row is not None and status == "new")
    return out


def problems(policy: Policy, text: str) -> list[str]:
    """Why a reply can't go out as written; empty if fine."""
    from ..brain.planner import caption_issues
    found = list(caption_issues(policy, text))
    if len(text) > int(_cfg(policy).get("max_chars", 220)):
        found.append("too long")
    if "@" in text or "#" in text or "http" in text.lower() or "www." in text.lower():
        found.append("mention, hashtag or link")
    return found


def _set(conn, cid, **cols) -> None:
    sets = ", ".join(f"{k} = %s" for k in cols)
    conn.execute(f"UPDATE comments SET {sets}, updated_at = now() WHERE id = %s", (*cols.values(), cid))


def _reconcile(conn, policy: Policy) -> int:
    """Replies sent but not recorded (a crash in between): found again, or marked failed; never sent twice."""
    n = 0
    for r in conn.execute("SELECT id, external_id, creator_id FROM comments WHERE status = 'replying'").fetchall():
        try:
            creds = instagram.account(conn, r["creator_id"])
            data = instagram.graph("GET", f"/{r['external_id']}/replies", {"fields": "id,from"}, creds).get("data") or []
        except (instagram.GraphError, instagram.AccountNotConnected):
            continue
        mine = [d for d in data if str((d.get("from") or {}).get("id") or "") == creds.ig_id]
        if mine:
            _set(conn, r["id"], status="replied", reply_external_id=str(mine[0]["id"]))
        else:
            _set(conn, r["id"], status="failed", reason="sent, but no reply found: not sent again")
        n += 1
    return n


def respond(conn, policy: Policy, llm, *, now: datetime | None = None) -> dict:
    m = mode(policy)
    if m not in ("draft", "autonomous"):
        return {}
    now, cfg = now or datetime.now(timezone.utc), _cfg(policy)
    instagram._use(policy)
    out = {"replied": 0, "drafted": 0, "skipped": 0, "failed": 0, "reconciled": _reconcile(conn, policy)}
    creators = conn.execute(
        """SELECT DISTINCT m.creator_id FROM comments m JOIN creators c ON c.id = m.creator_id
           WHERE m.status = 'new' AND c.status NOT IN ('paused', 'retired') LIMIT %s""",
        (int(cfg.get("creators_per_tick", 10)),)).fetchall()
    for row in creators:
        cid = row["creator_id"]
        done = conn.execute("SELECT count(*) AS n FROM comments WHERE creator_id = %s AND reply_sent_at > %s",
                            (cid, now - timedelta(hours=24))).fetchone()["n"]
        left = int(cfg.get("max_replies_per_account_day", 30)) - int(done)
        if left <= 0:
            continue
        batch = conn.execute(
            """SELECT m.id, m.external_id, m.text, p.caption FROM comments m JOIN publications u ON u.id = m.publication_id
               JOIN posts p ON p.id = u.post_id WHERE m.creator_id = %s AND m.status = 'new'
               ORDER BY m.commented_at LIMIT %s""", (cid, min(left, int(cfg.get("batch", 20))))).fetchall()
        c = conn.execute("SELECT bible FROM creators WHERE id = %s", (cid,)).fetchone()
        who = {k: (c["bible"] or {}).get(k) for k in ("display_name", "one_line", "personality", "caption_voice",
                                                      "opinions_and_running_jokes", "never")}
        keys = {f"c{i + 1}": r for i, r in enumerate(batch)}
        content = json.dumps({"you": who, "off_limits": policy.get("scope", "excluded_topics", default=[]),
                              "comments": [{"key": k, "on_post": (r["caption"] or "")[:300], "comment": r["text"]}
                                           for k, r in keys.items()]}, ensure_ascii=False)
        from ..brain.planner import voice_rules
        system = SYSTEM.format(max_chars=int(cfg.get("max_chars", 220))) + voice_rules(policy)
        try:
            with llm.tagged("comments", None):
                decided = llm.structured("comment_reply", system=system, content=content, schema=Replies,
                                         effort="low", creator_id=cid)
        except Exception as e:  # noqa: BLE001 - try this batch again next tick
            out["errors"] = out.get("errors", []) + [f"{type(e).__name__}: {str(e)[:200]}"]
            continue
        answered = set()
        try:
            creds = instagram.account(conn, cid) if m == "autonomous" else None
        except instagram.AccountNotConnected:
            continue
        for d in decided.items:
            r = keys.get(d.comment)
            if r is None or d.comment in answered:
                continue
            answered.add(d.comment)
            text = d.text.strip()
            bad = problems(policy, text) if d.action == "reply" else []
            if d.action == "skip" or not text or bad:
                _set(conn, r["id"], status="skipped", reason=(d.reason if d.action == "skip" else "; ".join(bad) or "empty")[:300])
                out["skipped"] += 1
            elif m == "draft":
                _set(conn, r["id"], status="drafted", reply_text=text, reason=d.reason[:300])
                out["drafted"] += 1
            else:
                _set(conn, r["id"], status="replying", reply_text=text, reply_sent_at=now)   # before the call
                try:
                    sent = instagram.graph("POST", f"/{r['external_id']}/replies", {"message": text}, creds)
                    _set(conn, r["id"], status="replied", reply_external_id=str(sent.get("id") or ""))
                    out["replied"] += 1
                except instagram.GraphError as e:
                    _set(conn, r["id"], status="failed", reason=instagram._clean(str(e), creds)[:300])
                    out["failed"] += 1
        for k, r in keys.items():
            if k not in answered:
                _set(conn, r["id"], status="skipped", reason="no decision")
                out["skipped"] += 1
    return out


def recent(conn, creator_id, limit: int = 30) -> list[dict]:
    """What the creator did with recent comments (for the owner: `mw comments <handle>`)."""
    rows = conn.execute("SELECT commented_at, status, text, reply_text, reason FROM comments WHERE creator_id = %s "
                        "ORDER BY commented_at DESC NULLS LAST LIMIT %s", (creator_id, limit)).fetchall()
    return [{**r, "commented_at": r["commented_at"].isoformat() if r["commented_at"] else None} for r in rows]
