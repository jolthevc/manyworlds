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
         "shares": "shares", "follows": "follows", "profile_visits": "profile_visits"}
# What Meta serves per media type (asking for one it doesn't fails the whole call). Profile visits and follows come
# for feed posts and Stories only; Reels have neither (follows from Reels show only at account level: collect_account).
BY_FORMAT = {"reels": ("reach", "views", "likes", "comments", "saved", "shares"),
             "story": ("reach", "views", "shares", "follows", "profile_visits"),
             "feed": ("reach", "views", "likes", "comments", "saved", "shares", "follows", "profile_visits")}
VIDEO = ("story_clip", "talking_video", "narrated_edit", "reel")
KEYS = ("reach", "views", "likes", "comments", "saves", "shares", "follows", "profile_visits")
OPTIONAL = ("profile_visits",)   # kept only when Meta returned it (absent is not zero)


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
        metrics = {k: int(values.get(k) or 0) for k in KEYS if k not in OPTIONAL or k in values}
        conn.execute("INSERT INTO metric_snapshots (post_id, captured_at, simulated, metrics, post_age_s) "
                     "VALUES (%s, %s, false, %s, %s)", (r["id"], now, Jsonb(metrics), int(age)))
        out["snapshots"] += 1
    return out


def _total(data: list[dict], name: str):
    for d in data or []:
        if d.get("name") == name:
            tv = d.get("total_value") or {}
            if "value" in tv:
                return tv["value"]
            vals = d.get("values") or []
            return vals[-1].get("value") if vals else None
    return None


def _by_follow_type(data: list[dict], name: str) -> dict:
    """{FOLLOWER: n, NON_FOLLOWER: n} from a total_value metric broken down by follow_type."""
    out = {}
    for d in data or []:
        if d.get("name") != name:
            continue
        for b in (d.get("total_value") or {}).get("breakdowns") or []:
            for r in b.get("results") or []:
                key = ((r.get("dimension_values") or [None])[0] or "").upper()
                if key:
                    out[key] = r.get("value")
    return out


def collect_account(conn, policy: Policy, *, now: datetime | None = None) -> dict:
    """Account-level insights, once a day per connected real account (owner 2026-11-25: follower growth, follows per
    1k non-follower reach, profile visits): followers_count from the profile, yesterday's reach split by follower and
    non-follower, follows and unfollows, profile views. Each piece is its own Graph call, so one Meta refuses does not
    lose the others; nothing is written for a day already collected. Free (Graph API); real accounts only."""
    now = now or datetime.now(timezone.utc)
    if not (policy.get("instagram", "account_insights", "enabled", default=True)):
        return {"accounts": 0, "errors": 0}
    instagram._POLICY.clear()
    instagram._POLICY.update(policy.get("instagram", default={}) or {})
    day = now.date()
    out = {"accounts": 0, "errors": 0}
    rows = conn.execute(
        "SELECT a.creator_id FROM accounts a JOIN creators c ON c.id = a.creator_id WHERE a.platform = 'instagram' "
        "AND a.status = 'connected' AND c.publish_mode = 'instagram' AND NOT EXISTS (SELECT 1 FROM account_insights i "
        "WHERE i.creator_id = a.creator_id AND i.day = %s)", (day,)).fetchall()
    for r in rows:
        try:
            ig_id, token = instagram.account(conn, r["creator_id"])
        except instagram.AccountNotConnected:
            continue
        m: dict = {}
        calls = (("profile", lambda: instagram.graph("GET", f"/{ig_id}", {"fields": "followers_count"}, token)),
                 ("reach", lambda: instagram.graph("GET", f"/{ig_id}/insights", {
                     "metric": "reach", "period": "day", "metric_type": "total_value", "breakdown": "follow_type"},
                     token)),
                 ("follows", lambda: instagram.graph("GET", f"/{ig_id}/insights", {
                     "metric": "follows_and_unfollows", "period": "day", "metric_type": "total_value",
                     "breakdown": "follow_type"}, token)),
                 ("profile_views", lambda: instagram.graph("GET", f"/{ig_id}/insights", {
                     "metric": "profile_views", "period": "day", "metric_type": "total_value"}, token)))
        for name, call in calls:
            try:
                got = call()
            except instagram.GraphError:
                out["errors"] += 1
                continue
            if name == "profile" and got.get("followers_count") is not None:
                m["follower_count"] = int(got["followers_count"])
            elif name == "reach":
                split = _by_follow_type(got.get("data") or [], "reach")
                if split:
                    m["reach_follower"] = int(split.get("FOLLOWER") or 0)
                    m["reach_non_follower"] = int(split.get("NON_FOLLOWER") or 0)
                if (t := _total(got.get("data") or [], "reach")) is not None:
                    m["reach"] = int(t)
            elif name == "follows":
                split = _by_follow_type(got.get("data") or [], "follows_and_unfollows")
                if split:   # FOLLOWER: accounts that followed; NON_FOLLOWER: accounts that unfollowed
                    m["follows"] = int(split.get("FOLLOWER") or 0)
                    m["unfollows"] = int(split.get("NON_FOLLOWER") or 0)
            elif name == "profile_views" and (t := _total(got.get("data") or [], "profile_views")) is not None:
                m["profile_views"] = int(t)
        if m:
            conn.execute("INSERT INTO account_insights (creator_id, day, metrics, captured_at) VALUES (%s, %s, %s, %s) "
                         "ON CONFLICT (creator_id, day) DO NOTHING", (r["creator_id"], day, Jsonb(m), now))
            out["accounts"] += 1
    return out
