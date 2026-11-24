"""Series continuity: each series' episodes and the threads they leave open.

A series (planner.series_catalogue: the profile's video series and the dossier's content system) runs as episodes.
An episode is recorded when its post is published or simulated-published (publish.scheduler), never when it is
scheduled, so a post dropped on the way never counts. Its number is its place among the series' published episodes,
its premise the one line the showrunner gave the idea. An episode can open a thread (a promised part 2, a running
joke, a question put to the audience) and close earlier ones by id; open threads stay until a published post closes
them. The showrunner sees, per active series, the last few episodes, the open threads and the next episode number
(`planning_view`), so it can write a part 2, a callback, or close a thread.

The episode fields live on the idea (ideas.body: episode_premise, opens_thread, closes_threads); the post is joined
to its idea at publication, so nothing about episodes is copied into posts.body.
"""

from __future__ import annotations

from datetime import datetime


def _idea_body(conn, post_id) -> dict:
    row = conn.execute("SELECT body FROM ideas WHERE post_id = %s ORDER BY updated_at DESC LIMIT 1",
                       (post_id,)).fetchone()
    return (row["body"] if row else None) or {}


def record_published(conn, post_id, *, published_at: datetime, simulated: bool) -> dict:
    """A post went out: if it belongs to a series it becomes that series' next episode; the thread it opens is
    opened and the threads it names are closed. Idempotent (a post is one episode, a thread opened once)."""
    post = conn.execute("SELECT creator_id, body FROM posts WHERE id = %s", (post_id,)).fetchone()
    if post is None:
        return {}
    body, idea = post["body"] or {}, _idea_body(conn, post_id)
    cid = post["creator_id"]
    name = str(body.get("series") or idea.get("series") or "").strip()
    out: dict = {}
    if name:
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"series:{cid}:{name}",))
        row = conn.execute(
            """INSERT INTO series_episodes (creator_id, series, number, post_id, premise, simulated, published_at)
               SELECT %s, %s, coalesce(max(number), 0) + 1, %s, %s, %s, %s FROM series_episodes
               WHERE creator_id = %s AND series = %s
               ON CONFLICT (post_id) DO NOTHING RETURNING number""",
            (cid, name, post_id, str(idea.get("episode_premise") or "").strip()[:300], simulated, published_at,
             cid, name)).fetchone()
        if row:
            out["episode"] = row["number"]
    opened = str(idea.get("opens_thread") or "").strip()[:300]
    if opened:
        conn.execute("INSERT INTO series_threads (creator_id, series, text, opened_post, opened_at) "
                     "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (opened_post, text) DO NOTHING",
                     (cid, name, opened, post_id, published_at))
        out["opened"] = opened
    closes = [str(t).strip().lower() for t in (idea.get("closes_threads") or []) if str(t).strip()]
    if closes:
        closed = conn.execute(
            "UPDATE series_threads SET closed_post = %s, closed_at = %s WHERE creator_id = %s AND closed_at IS NULL "
            "AND opened_post <> %s AND left(id::text, 8) = ANY(%s) RETURNING id",
            (post_id, published_at, cid, post_id, [c[:8] for c in closes])).fetchall()
        out["closed"] = len(closed)
    return out


def retract(conn, post_ids) -> int:
    """A post that is dropped after all is no episode: its episode and the threads it opened go, and the threads it
    closed open again. (Episodes enter only at publication, so this is a guard, not the usual path.)"""
    ids = [str(p) for p in (post_ids or []) if p is not None]
    if not ids:
        return 0
    conn.execute("UPDATE series_threads SET closed_post = NULL, closed_at = NULL WHERE closed_post = ANY(%s::uuid[])",
                 (ids,))
    conn.execute("DELETE FROM series_threads WHERE opened_post = ANY(%s::uuid[])", (ids,))
    return len(conn.execute("DELETE FROM series_episodes WHERE post_id = ANY(%s::uuid[]) RETURNING id",
                            (ids,)).fetchall())


def next_number(conn, creator_id, series: str) -> int:
    row = conn.execute("SELECT coalesce(max(number), 0) + 1 AS n FROM series_episodes WHERE creator_id = %s "
                       "AND series = %s", (creator_id, series)).fetchone()
    return int(row["n"])


def planning_view(conn, policy, creator_id, catalogue: dict) -> dict:
    """Per series (the catalogue's, then any other with episodes, most recently aired first, at most
    `brain.series.max_series`): the next episode number, the last `episodes_shown` episodes (number, date,
    premise) and up to `threads_shown` open threads ({id, text}); threads of no one series under "" ."""
    cfg = policy.get("brain", "series", default={}) or {}
    n_eps, n_threads = int(cfg.get("episodes_shown", 3)), int(cfg.get("threads_shown", 5))
    aired = [r["series"] for r in conn.execute(
        "SELECT series, max(published_at) AS last FROM series_episodes WHERE creator_id = %s GROUP BY series "
        "ORDER BY last DESC", (creator_id,)).fetchall()]
    names = list(dict.fromkeys(aired + [s for s in catalogue if s not in aired]))
    threads: dict[str, list] = {}
    for t in conn.execute("SELECT id, series, text FROM series_threads WHERE creator_id = %s AND closed_at IS NULL "
                          "ORDER BY opened_at DESC", (creator_id,)).fetchall():
        bucket = threads.setdefault(t["series"], [])
        if len(bucket) < n_threads:
            bucket.append({"id": str(t["id"])[:8], "text": t["text"]})
    out: dict = {}
    for name in names[:int(cfg.get("max_series", 6))]:
        eps = conn.execute("SELECT number, premise, published_at FROM series_episodes WHERE creator_id = %s "
                           "AND series = %s ORDER BY number DESC LIMIT %s", (creator_id, name, n_eps)).fetchall()
        entry: dict = {"next_episode": (eps[0]["number"] + 1) if eps else 1}
        if eps:
            entry["last_episodes"] = [{"number": e["number"], "aired": e["published_at"].date().isoformat(),
                                       **({"premise": e["premise"]} if e["premise"] else {})} for e in eps]
        if threads.get(name):
            entry["open_threads"] = threads[name]
        out[name] = entry
    if threads.get(""):
        out[""] = {"open_threads": threads[""]}
    return out
