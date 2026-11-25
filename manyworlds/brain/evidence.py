"""The only sanctioned reader of findings and metrics for Brain agents.

Simulated results test the machinery; real Instagram results are the only evidence of demand.
The wall is structural: simulated findings and metrics live only in `simulated_findings` and
`simulated_metrics`, and the real tables reject simulated rows (CHECK constraints), so no query
on the real tables can see them. Only this module touches the simulated tables (a test enforces
it). Every read names a mode: `real` reads the real tables only; `sandbox` adds the simulated
ones and labels every simulated row "SIMULATED" (metrics in sandbox mode are the simulated ones). The
strategist and allocator read through `for_allocation`, which is always real. A test fails if
any other module under manyworlds/brain reads these tables directly.
"""

from __future__ import annotations

import re
import uuid
from typing import Literal

from psycopg.types.json import Jsonb

Mode = Literal["real", "sandbox"]
LABEL = "SIMULATED"
GRADES = ("proven", "tentative", "speculative")


def mode_for(conn, creator_id) -> Mode:
    row = conn.execute("SELECT publish_mode FROM creators WHERE id = %s", (creator_id,)).fetchone()
    return "real" if row and row["publish_mode"] == "instagram" else "sandbox"


def _label(row: dict) -> dict:
    return {**row, "label": LABEL} if row.get("evidence") == "simulated" else {**row, "label": None}


def findings(conn, *, mode: Mode, scope: str | None = None, creator_id=None, cluster_id=None, market_id=None,
             grades: tuple[str, ...] = GRADES, statuses: tuple[str, ...] = ("open",), domain: str | None = None,
             genre: str | None = None) -> list[dict]:
    """Findings in the mode; `domain` (content | music) and `genre` narrow them (None: any)."""
    if mode not in ("real", "sandbox"):
        raise ValueError(f"unknown evidence mode {mode!r}")
    filters = """ (%(scope)s::text IS NULL OR scope = %(scope)s)
             AND (%(creator)s::uuid IS NULL OR creator_id = %(creator)s)
             AND (%(cluster)s::uuid IS NULL OR cluster_id = %(cluster)s)
             AND (%(market)s::uuid IS NULL OR market_id = %(market)s)
             AND (%(domain)s::text IS NULL OR domain = %(domain)s)
             AND (%(genre)s::text IS NULL OR genre = %(genre)s)
             AND grade = ANY(%(grades)s)
             AND status = ANY(%(statuses)s)
"""
    params = {"scope": scope, "creator": creator_id, "cluster": cluster_id, "market": market_id,
              "grades": list(grades), "statuses": list(statuses), "domain": domain, "genre": genre}
    sql = f"SELECT * FROM findings WHERE {filters}"
    if mode == "sandbox":
        sql += f" UNION ALL SELECT * FROM simulated_findings WHERE {filters}"
    rows = conn.execute(f"SELECT * FROM ({sql}) f ORDER BY created_at", params).fetchall()
    return [_label(r) for r in rows]


def for_allocation(conn, **filters) -> list[dict]:
    """Findings the strategist may act on with money: real evidence only, always."""
    filters.pop("mode", None)
    return findings(conn, mode="real", **filters)


def metrics(conn, post_id, *, mode: Mode) -> list[dict]:
    if mode == "real":
        return conn.execute("SELECT * FROM metric_snapshots WHERE post_id = %s ORDER BY captured_at",
                            (post_id,)).fetchall()
    if mode == "sandbox":
        return [{**r, "label": LABEL} for r in conn.execute(
            "SELECT * FROM simulated_metrics WHERE post_id = %s ORDER BY captured_at", (post_id,)).fetchall()]
    raise ValueError(f"unknown evidence mode {mode!r}")


def post_outcomes(conn, creator_id, n: int, mode: Mode, *, policy=None) -> dict:
    """How the creator's last `n` published posts did: {post_id (str): {"reach": ..., "age_h": ..., ...}}, read at
    the analysis target age (policy brain.analysis.target_age_s; the first snapshot at or after it, as the analyst
    and the predictions ledger read it: a young post's reach would read as failure). Posts not yet that old are
    left out. `real` reads real publications and metric_snapshots only; `sandbox` reads the simulated ones, each
    outcome labelled SIMULATED. The numbers are the snapshot's own (reach, plus saves, shares, likes, comments and
    views where the platform gave them)."""
    if mode not in ("real", "sandbox"):
        raise ValueError(f"unknown evidence mode {mode!r}")
    target = int((policy.get("brain", "analysis", "target_age_s", default=259200) if policy else 259200))
    posts = conn.execute(
        "SELECT DISTINCT ON (p.id) p.id, u.published_at FROM posts p JOIN publications u ON u.post_id = p.id "
        "WHERE p.creator_id = %s AND u.simulated = %s ORDER BY p.id, u.published_at DESC",
        (creator_id, mode == "sandbox")).fetchall()
    posts = sorted(posts, key=lambda r: r["published_at"], reverse=True)[:int(n)]
    keys = ("reach", "saves", "shares", "likes", "comments", "views")
    out: dict = {}
    for p in posts:
        rows = [r for r in metrics(conn, p["id"], mode=mode) if (r.get("post_age_s") or 0) >= target]
        if not rows:
            continue
        first = min(rows, key=lambda r: r.get("post_age_s") or 0)
        m = first.get("metrics") or {}
        o = {k: m[k] for k in keys if m.get(k) is not None}
        if "reach" not in o:
            continue
        o["age_h"] = round(int(first.get("post_age_s") or 0) / 3600)
        if mode == "sandbox":
            o["label"] = LABEL
        out[str(p["id"])] = o
    return out


def record_finding(conn, *, scope: str, statement: str, evidence: str, grade: str, creator_id=None, cluster_id=None,
                   market_id=None, effect=None, interval: tuple | None = None, n: int | None = None,
                   sources=(), domain: str = "content", genre: str | None = None) -> uuid.UUID:
    """A finding about a sandbox creator, or drawn from one or from simulated metrics, can only be
    recorded as simulated: the label is checked here, not trusted, so nothing simulated can reach
    a real-mode read when the creator later goes live. `domain`: content (posts) or music (songs);
    `genre`: the genre a genre-scope finding is about."""
    if evidence != "simulated":
        if creator_id and mode_for(conn, creator_id) == "sandbox":
            raise ValueError(f"creator {creator_id} is in the sandbox: its findings must be evidence='simulated'")
        for s in sources:
            if isinstance(s, dict) and s.get("table") == "simulated_metrics":
                raise ValueError("a finding drawn from simulated_metrics must be evidence='simulated'")
            if isinstance(s, dict) and s.get("creator_id") and mode_for(conn, s["creator_id"]) == "sandbox":
                raise ValueError(f"source creator {s['creator_id']} is in the sandbox: evidence must be 'simulated'")
    low, high = interval if interval else (None, None)
    table = "simulated_findings" if evidence == "simulated" else "findings"
    return conn.execute(
        f"""INSERT INTO {table} (scope, creator_id, cluster_id, market_id, statement, evidence, grade, effect,
                                 interval_low, interval_high, n, sources, domain, genre)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
        (scope, creator_id, cluster_id, market_id, statement, evidence, grade, effect, low, high, n,
         Jsonb(list(sources)), domain, genre)).fetchone()["id"]


def record_simulated_metrics(conn, post_id, rows: list[dict], *, model: str, panel_version: str) -> None:
    """Outcomes from the scenario engine; captured_at is the publication time plus each age."""
    pub = conn.execute("SELECT coalesce(published_at, now()) AS t FROM posts WHERE id = %s", (post_id,)).fetchone()["t"]
    with conn.transaction():
        for r in rows:
            conn.execute("INSERT INTO simulated_metrics (post_id, captured_at, post_age_s, model, panel_version, metrics) "
                         "VALUES (%s, %s + make_interval(secs => %s), %s, %s, %s, %s)",
                         (post_id, pub, r["post_age_s"], r["post_age_s"], model, panel_version,
                          Jsonb(r["metrics"])))   # never the scenario's hidden truth (e.g. which posts broke out)


def supersede(conn, *, creator_id=None, domain: str = "content", scope: str = "creator",
              genre: str | None = None) -> int:
    """Close the open analysis findings (both tables) a new analysis replaces: one domain (content: the analyst's
    post findings; music: music_analysis), one scope, one creator (None: none) and, for music, one genre (None:
    none). Proven findings (experiments) and findings at other scopes or domains are left alone."""
    n = 0
    for table in ("findings", "simulated_findings"):
        n += conn.execute(f"UPDATE {table} SET status = 'superseded', updated_at = now() "
                          "WHERE creator_id IS NOT DISTINCT FROM %s AND status = 'open' AND scope = %s "
                          "AND domain = %s AND genre IS NOT DISTINCT FROM %s AND grade <> 'proven'",
                          (creator_id, scope, domain, genre)).rowcount
    return n


def song_posts(conn, *, mode: Mode, creator_ids=None, policy=None) -> list[dict]:
    """Every published post of the given creators (None: every creator with publications in this mode), in time
    order, with its song (posts.song_id, None for a post without one), format and its outcome at the analysis target
    age (brain.analysis.target_age_s: the first snapshot at or after it, within target_age_tolerance_s): reach,
    saves, shares, likes, comments, views where the platform gave them. Posts not yet that old are left out. The
    music analysis needs posts without songs too: they are each creator's baseline. `real` reads real publications
    and metric_snapshots only; `sandbox` the simulated ones, each row labelled SIMULATED."""
    if mode not in ("real", "sandbox"):
        raise ValueError(f"unknown evidence mode {mode!r}")
    target = int(policy.get("brain", "analysis", "target_age_s", default=259200) if policy else 259200)
    tolerance = int(policy.get("brain", "analysis", "target_age_tolerance_s", default=86400) if policy else 86400)
    ids = [str(c) for c in creator_ids] if creator_ids is not None else None
    posts = conn.execute(
        "SELECT DISTINCT ON (p.id) p.id, p.creator_id, p.song_id, p.format, u.published_at FROM posts p "
        "JOIN publications u ON u.post_id = p.id WHERE u.simulated = %s "
        "AND (%s::uuid[] IS NULL OR p.creator_id = ANY(%s::uuid[])) ORDER BY p.id, u.published_at DESC",
        (mode == "sandbox", ids, ids)).fetchall()
    keys = ("reach", "saves", "shares", "likes", "comments", "views")
    out = []
    for p in sorted(posts, key=lambda r: r["published_at"]):
        obs = [m for m in metrics(conn, p["id"], mode=mode)
               if m.get("post_age_s") is not None and target <= m["post_age_s"] <= target + tolerance]
        if not obs:
            continue
        m = min(obs, key=lambda r: r["post_age_s"]).get("metrics") or {}
        if m.get("reach") is None:
            continue
        row = {"post_id": str(p["id"]), "creator_id": str(p["creator_id"]),
               "song_id": str(p["song_id"]) if p["song_id"] else None, "format": p["format"],
               "published_at": p["published_at"], **{k: m[k] for k in keys if m.get(k) is not None}}
        if mode == "sandbox":
            row["label"] = LABEL
        out.append(row)
    return out


def niche_trends(conn, creator_id, policy) -> dict | None:
    """What is working in the creator's niche on public posts (the trend watcher's latest brief and its findings).
    Research evidence: correlation in other people's posts, never proof; the same in real and sandbox modes."""
    c = conn.execute("SELECT c.bible->>'niche' AS niche, m.name AS market FROM creators c "
                     "LEFT JOIN markets m ON m.id = c.market_id WHERE c.id = %s", (creator_id,)).fetchone()
    names = [n for n in ((c or {}).get("market"), (c or {}).get("niche"),
                         (policy.get("trends", "default_market") if policy else None)) if n]
    for name in names:
        row = conn.execute("SELECT day, brief FROM trend_briefs WHERE market = %s ORDER BY day DESC LIMIT 1",
                           (name,)).fetchone()
        if row:
            b = row["brief"]
            return {"niche": name, "as_of": str(row["day"]), "working": [p["statement"] for p in b.get("working", [])],
                    "rising": b.get("rising", []), "cooling": b.get("cooling", []),
                    "hooks_to_try": b.get("hooks_to_try", []), "avoid": b.get("avoid", [])}
    return None


_MENTION = re.compile(r"@[\w.]+")
_LINK = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)


def scrub(text: str) -> str:
    """Comment text with every @handle and link taken out: a handle is never passed on, quoted or stored."""
    return _LINK.sub("[link]", _MENTION.sub("@someone", text or "")).strip()


def audience_comments(conn, creator_id, *, since, limit: int, chars: int) -> list[dict]:
    """Recent comments on the creator's published posts, for planning (the comment themes): newest first, each
    {"comment", "on_post", "series", "at"}, with handles and links scrubbed. Audience evidence like metrics, so only
    this module reads it. Real only: comments exist only on real Instagram publications, never simulated ones.
    Comments from any of our own accounts are left out (our creators never engage with each other), as are those
    whose text has been cleared (comments.keep_days). Commenters never appear, not even as hashes."""
    from ..research.apify import author_hash
    ours = [h for h in (author_hash(r["handle"]) for r in conn.execute(
        "SELECT handle FROM accounts WHERE handle IS NOT NULL").fetchall()) if h]
    rows = conn.execute(
        """SELECT m.text, m.commented_at, p.caption, p.body->>'series' AS series
           FROM comments m JOIN publications u ON u.id = m.publication_id AND NOT u.simulated
           JOIN posts p ON p.id = u.post_id
           WHERE m.creator_id = %s AND m.text IS NOT NULL AND m.text <> '' AND m.created_at > %s
             AND coalesce(m.reason, '') <> 'one of our accounts' AND NOT (coalesce(m.author_hash, '') = ANY(%s))
           ORDER BY m.commented_at DESC NULLS LAST LIMIT %s""",
        (creator_id, since, ours, int(limit))).fetchall()
    return [{"comment": scrub(r["text"])[:chars], "on_post": scrub(r["caption"] or "")[:chars],
             **({"series": r["series"]} if r["series"] else {}),
             "at": r["commented_at"].isoformat() if r["commented_at"] else None} for r in rows]


def creators_with_new_comments(conn, *, min_new: int, every, now) -> list:
    """Active creators whose real posts gathered at least `min_new` comments (text kept, not from our accounts) since
    their last comment_themes decision, and whose last one is older than `every` (a timedelta)."""
    rows = conn.execute(
        """SELECT m.creator_id FROM comments m JOIN publications u ON u.id = m.publication_id AND NOT u.simulated
           JOIN creators c ON c.id = m.creator_id AND c.status = 'active'
           LEFT JOIN LATERAL (SELECT max(d.created_at) AS at FROM decisions d
                              WHERE d.kind = 'comment_themes' AND d.creator_id = m.creator_id) t ON true
           WHERE m.text IS NOT NULL AND m.text <> '' AND coalesce(m.reason, '') <> 'one of our accounts'
             AND m.created_at > coalesce(t.at, '-infinity'::timestamptz)
             AND coalesce(t.at, '-infinity'::timestamptz) < %s
           GROUP BY m.creator_id HAVING count(*) >= %s ORDER BY m.creator_id""",
        (now - every, int(min_new))).fetchall()
    return [r["creator_id"] for r in rows]


RATES = ("saves", "shares", "follows", "profile_visits")


def _target_snapshot(conn, post_id, mode: Mode, target: int) -> dict | None:
    rows = [r for r in metrics(conn, post_id, mode=mode) if (r.get("post_age_s") or 0) >= target]
    return (min(rows, key=lambda r: r.get("post_age_s") or 0).get("metrics") or {}) if rows else None


def attribution(conn, creator_ids, mode: Mode, *, policy=None, n: int = 60, dims=None) -> dict:
    """What works, by dimension (owner 2026-11-25: performance by aesthetic, outfit, environment, format, archetype,
    pattern card): the creators' last `n` published posts each, at the analysis target age, grouped per dimension value
    with n, median reach, and per-1k-reach saves, shares, follows and profile visits (where the platform gave them),
    and the median reach over the creators' own medians (`lift`). Dimensions: format, the idea's attributes (outfit,
    setting...), the shot types of its shot sheet, the pattern cards it applied, the creator's archetype. `real` reads
    real publications only; `sandbox` the simulated ones, labelled SIMULATED."""
    import statistics
    if mode not in ("real", "sandbox"):
        raise ValueError(f"unknown evidence mode {mode!r}")
    target = int((policy.get("brain", "analysis", "target_age_s", default=259200) if policy else 259200))
    dims = list(dims or ("format", "outfit", "setting", "shot_type", "pattern_card", "archetype"))
    groups: dict = {}
    for cid in [str(c) for c in creator_ids]:
        prof = conn.execute("SELECT body FROM creator_profiles WHERE creator_id = %s ORDER BY version DESC LIMIT 1",
                            (cid,)).fetchone()
        archetype = ((prof or {}).get("body") or {}).get("archetypes") or [None]
        rows = conn.execute(
            """SELECT DISTINCT ON (p.id) p.id, p.format, p.body, u.published_at, i.attributes FROM posts p
               JOIN publications u ON u.post_id = p.id LEFT JOIN ideas i ON i.post_id = p.id
               WHERE p.creator_id = %s AND u.simulated = %s ORDER BY p.id, u.published_at DESC""",
            (cid, mode == "sandbox")).fetchall()
        rows = sorted(rows, key=lambda r: r["published_at"], reverse=True)[:int(n)]
        posts = []
        for r in rows:
            m = _target_snapshot(conn, r["id"], mode, target)
            if not m or not m.get("reach"):
                continue
            body, attrs = r["body"] or {}, r["attributes"] or {}
            values = {"format": [r["format"]], "archetype": [archetype[0]] if archetype[0] else [],
                      "shot_type": sorted({s.get("shot_type") for s in body.get("shots") or []
                                           if isinstance(s, dict) and s.get("shot_type")}),
                      "pattern_card": sorted({c.get("key") for c in body.get("pattern_cards") or []
                                              if isinstance(c, dict) and c.get("applied") and c.get("key")})}
            for d in dims:
                if d not in values and attrs.get(d):
                    values[d] = [str(attrs[d])[:60]]
            posts.append((m, values))
        if not posts:
            continue
        own = statistics.median(float(m["reach"]) for m, _ in posts)
        for m, values in posts:
            for d in dims:
                for v in values.get(d) or []:
                    groups.setdefault(d, {}).setdefault(v, []).append((m, float(m["reach"]) / own if own else None))
    out: dict = {}
    for d, by in groups.items():
        out[d] = {}
        for v, items in by.items():
            reach = [float(m["reach"]) for m, _ in items]
            row = {"n": len(items), "median_reach": statistics.median(reach)}
            lifts = [x for _, x in items if x is not None]
            row["lift"] = round(statistics.median(lifts), 3) if lifts else None
            for k in RATES:
                vals = [1000.0 * float(m[k]) / float(m["reach"]) for m, _ in items if m.get(k) is not None]
                if vals:
                    row[f"{k}_per_1k"] = round(statistics.median(vals), 2)
            if mode == "sandbox":
                row["label"] = LABEL
            out[d][v] = row
    return out


def growth(conn, creator_id, mode: Mode, *, days: int = 28) -> dict | None:
    """The account's growth over `days` from account-level insights (publish/metrics.collect_account): followers
    at the start and end, net change, follows, reach from non-followers and profile views, and follows per 1k
    non-follower reach (how well being discovered turns into following). Real accounts only: there are no simulated
    account insights, so `sandbox` returns None."""
    if mode not in ("real", "sandbox"):
        raise ValueError(f"unknown evidence mode {mode!r}")
    if mode == "sandbox":
        return None
    rows = conn.execute("SELECT day, metrics FROM account_insights WHERE creator_id = %s "
                        "AND day > current_date - %s ORDER BY day", (creator_id, int(days))).fetchall()
    if not rows:
        return None
    ms = [r["metrics"] or {} for r in rows]
    counts = [m["follower_count"] for m in ms if m.get("follower_count") is not None]

    def total(k):
        vals = [float(m[k]) for m in ms if m.get(k) is not None]
        return sum(vals) if vals else None
    out = {"days": len(rows), "followers_start": counts[0] if counts else None,
           "followers_end": counts[-1] if counts else None,
           "net_followers": (counts[-1] - counts[0]) if len(counts) > 1 else None,
           "follows": total("follows"), "non_follower_reach": total("reach_non_follower"),
           "follower_reach": total("reach_follower"), "profile_views": total("profile_views")}
    if out["follows"] is not None and out["non_follower_reach"]:
        out["follows_per_1k_non_follower_reach"] = round(1000 * out["follows"] / out["non_follower_reach"], 2)
    return out
