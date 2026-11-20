"""Real metrics (Brain M4): Instagram insights for published posts, at the observation ages the analysis uses.

For each published post, at each age in policy `simulation.ages_s` that has passed with no snapshot at or
beyond it, one snapshot is taken at the post's real age (a late collection covers the ages it missed; it is
never relabelled as an earlier age). Meta's names are mapped to ours (`saved` -> `saves`), so real and
simulated snapshots have the same keys. Stories' insights expire after 24 h; their last ages are collected
before then. Real snapshots only: this module never writes simulated data.
"""

from __future__ import annotations

from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from ..policy import Policy
from . import instagram

NAMES = {"reach": "reach", "views": "views", "likes": "likes", "comments": "comments", "saved": "saves",
         "shares": "shares", "follows": "follows"}
# What Meta serves per media type (asking for one it doesn't fails the whole call).
BY_FORMAT = {"reels": ("reach", "views", "likes", "comments", "saved", "shares"),
             "story": ("reach", "views", "shares", "follows"),
             "feed": ("reach", "views", "likes", "comments", "saved", "shares", "follows")}
VIDEO = ("story_clip", "talking_video", "narrated_edit", "reel")
KEYS = ("reach", "views", "likes", "comments", "saves", "shares", "follows")


def collect(conn, policy: Policy, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    ages = sorted(int(a) for a in policy.get("simulation", "ages_s", default=[3600, 21600, 86400, 259200]))
    instagram._POLICY.clear()
    instagram._POLICY.update(policy.get("instagram", default={}) or {})
    out = {"snapshots": 0, "errors": 0}
    rows = conn.execute(
        """SELECT p.id, p.creator_id, p.format, u.external_id, u.published_at FROM posts p
           JOIN publications u ON u.post_id = p.id AND u.platform = 'instagram' AND NOT u.simulated
           WHERE u.published_at > %s - make_interval(secs => %s)""", (now, ages[-1] + 86400)).fetchall()
    pub = instagram.InstagramPublisher()
    for r in rows:
        age = (now - r["published_at"]).total_seconds()
        kind = "story" if r["format"] == "story" else ("reels" if r["format"] in VIDEO else "feed")
        if kind == "story" and age > 86400:
            continue   # Story insights expire after 24 h
        due = [a for a in ages if a <= age]
        if not due:
            continue
        last = conn.execute("SELECT max(post_age_s) a FROM metric_snapshots WHERE post_id = %s", (r["id"],)).fetchone()["a"]
        if last is not None and last >= due[-1]:
            continue
        try:
            _, token = pub.account(conn, r["creator_id"])
            data = instagram.graph("GET", f"/{r['external_id']}/insights",
                                   {"metric": ",".join(BY_FORMAT[kind])}, token).get("data") or []
        except (instagram.GraphError, instagram.AccountNotConnected) as e:
            if kind == "story" and ('"code":10' in str(e).replace(" ", "") or "(#10)" in str(e)):
                continue   # fewer than 5 viewers: Meta returns no Story data, not an error worth counting
            out["errors"] += 1
            continue
        values = {NAMES[d["name"]]: (d.get("values") or [{}])[0].get("value") for d in data if d.get("name") in NAMES}
        metrics = {k: int(values.get(k) or 0) for k in KEYS}
        conn.execute("INSERT INTO metric_snapshots (post_id, captured_at, simulated, metrics, post_age_s) "
                     "VALUES (%s, %s, false, %s, %s)", (r["id"], now, Jsonb(metrics), int(age)))
        out["snapshots"] += 1
    return out
