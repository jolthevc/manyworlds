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
                   sources=(), domain: str = "content", genre: str | None = None, origin: str | None = None,
                   level: str | None = None, applies_to: dict | None = None, replications=(), contradictions=(),
                   last_tested=None, retest_after=None, parent_id=None) -> uuid.UUID:
    """A finding about a sandbox creator, or drawn from one or from simulated metrics, can only be
    recorded as simulated: the label is checked here, not trusted, so nothing simulated can reach
    a real-mode read when the creator later goes live. `domain`: content (posts) or music (songs);
    `genre`: the genre a genre-scope finding is about. The hypothesis record (manyworlds.brain.hypotheses; migration
    055): origin and level default from the scope, evidence and sources; last tested now unless given; retest_after
    NULL means policy brain.hypotheses.retest_days after last_tested."""
    from . import hypotheses
    d = hypotheses.defaults(scope, evidence, sources)
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
                                 interval_low, interval_high, n, sources, domain, genre, origin, level, applies_to,
                                 replications, contradictions, last_tested, retest_after, parent_id)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                   coalesce(%s, now()), %s, %s) RETURNING id""",
        (scope, creator_id, cluster_id, market_id, statement, evidence, grade, effect, low, high, n,
         Jsonb(list(sources)), domain, genre, origin or d["origin"], level or d["level"], Jsonb(applies_to or {}),
         Jsonb(list(replications)), Jsonb(list(contradictions)), last_tested, retest_after,
         parent_id)).fetchone()["id"]


def finding(conn, finding_id) -> dict | None:
    """One finding by id, from either table (labelled SIMULATED when simulated), or None."""
    for table in ("findings", "simulated_findings"):
        row = conn.execute(f"SELECT * FROM {table} WHERE id = %s", (finding_id,)).fetchone()
        if row:
            return _label(row)
    return None


def archetype_hypotheses(conn, *, mode: Mode, effect_key: str | None = None,
                         statuses: tuple[str, ...] = ("open",)) -> list[dict]:
    """Archetype-level hypotheses (scope archetype, manyworlds.brain.hypotheses) in the mode, optionally for one
    effect key (applies_to.effect_key)."""
    if mode not in ("real", "sandbox"):
        raise ValueError(f"unknown evidence mode {mode!r}")
    cond = "scope = 'archetype' AND status = ANY(%(st)s) AND (%(key)s::text IS NULL OR applies_to->>'effect_key' = %(key)s)"
    sql = f"SELECT * FROM findings WHERE {cond}"
    if mode == "sandbox":
        sql += f" UNION ALL SELECT * FROM simulated_findings WHERE {cond}"
    rows = conn.execute(f"SELECT * FROM ({sql}) f ORDER BY created_at", {"st": list(statuses), "key": effect_key}).fetchall()
    return [_label(r) for r in rows]


def record_test(conn, finding_id, *, replications: list, contradictions: list, grade: str, level: str,
                status: str = "open") -> None:
    """A hypothesis tested again (manyworlds.brain.hypotheses.on_experiment): its replications and contradictions,
    the grade, level and status they give it, last tested now and its retest date reset (NULL: from policy)."""
    for table in ("findings", "simulated_findings"):
        conn.execute(f"UPDATE {table} SET replications = %s, contradictions = %s, grade = %s, level = %s, status = %s, "
                     "last_tested = now(), retest_after = NULL, updated_at = now() WHERE id = %s",
                     (Jsonb(list(replications)), Jsonb(list(contradictions)), grade, level, status, finding_id))


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


def annotated_posts(conn, *, mode: Mode, schemas: tuple[str, ...] | list[str], creator_ids=None,
                    outcome: tuple[str, str] | None = None) -> list[dict]:
    """The research reference posts with their newest annotation (schemas: newest first, e.g. annotation.READ_SCHEMAS)
    for the recipe analysis (manyworlds.brain.recipes): {post_id, creator_id, platform, media_type, posted_at,
    fetched_at, schema, observed, over_own_median (the post's latest, else the one stored with its read), metrics
    (the public counts present), archetype (the creator's scores.archetype, else its cluster), followers}.

    Research evidence: other people's public posts, correlation never proof, and never simulated (there are no
    simulated reference posts), so `real` and `sandbox` read the same rows; the mode is still named and checked like
    every reader's. Post-level comments and commenters are not read.

    `outcome` (metric, horizon): the post's normalised outcome from the analytical spine (manyworlds.research.spine:
    log of the metric over the creator's median at that horizon) replaces over_own_median, as its ratio (exp); a
    post without that outcome gets None (excluded and counted by the analysis, never zero)."""
    if mode not in ("real", "sandbox"):
        raise ValueError(f"unknown evidence mode {mode!r}")
    ids = [str(c) for c in creator_ids] if creator_ids is not None else None
    rows = conn.execute(
        """SELECT DISTINCT ON (a.post_id) a.post_id, a.schema, a.observed, a.evidence, a.created_at AS read_at,
                  p.creator_id, p.platform, p.media_type, p.posted_at, p.fetched_at, p.metrics,
                  coalesce(c.scores->>'archetype', c.cluster) AS archetype, c.followers
           FROM reference_annotations a JOIN reference_posts p ON p.id = a.post_id
           JOIN reference_creators c ON c.id = p.creator_id
           WHERE a.schema = ANY(%s) AND (%s::uuid[] IS NULL OR p.creator_id = ANY(%s::uuid[]))
           ORDER BY a.post_id, array_position(%s::text[], a.schema)""",
        (list(schemas), ids, ids, list(schemas))).fetchall()
    keys = ("views", "likes", "comments", "shares", "saves")
    spine = {}
    if outcome is not None:
        import math
        spine = {r["ref_id"]: math.exp(float(r["normalized"])) for r in conn.execute(
            """SELECT u.ref_id, o.normalized FROM outcomes o JOIN content_units u ON u.id = o.unit_id
               WHERE u.ref_table = 'reference_posts' AND u.level = 'post' AND o.metric = %s AND o.horizon = %s
                 AND o.normalized IS NOT NULL""", (str(outcome[0]), str(outcome[1]))).fetchall()}
    out = []
    for r in rows:
        m, ev = r["metrics"] or {}, r["evidence"] or {}
        ratio = m.get("over_own_median") if m.get("over_own_median") is not None else ev.get("over_own_median")
        if outcome is not None:
            ratio = spine.get(str(r["post_id"]))
        out.append({"post_id": str(r["post_id"]), "creator_id": str(r["creator_id"]), "platform": r["platform"],
                    "media_type": r["media_type"], "posted_at": r["posted_at"], "fetched_at": r["fetched_at"],
                    "schema": r["schema"], "observed": r["observed"] or {}, "over_own_median": ratio,
                    "metrics": {k: m[k] for k in keys if m.get(k) is not None},
                    "archetype": r["archetype"], "followers": r["followers"]})
    return out


def own_annotations(conn, *, mode: Mode, schemas: tuple[str, ...] | list[str], creator_ids=None) -> list[dict]:
    """Our own accepted assets coded in the research vocabulary (reference_annotations source own; manyworlds.review
    .own_annotation), joined with what was directed and achieved and what it cost, so own and external posts compare
    on the same coded attributes: {asset_id, post_id, creator_id, format, status, schema, observed (the coded read),
    performance (the newest observation: requested, observed, scores, closeness), job_usd (the asset's job's ledger
    cost)}. Outcomes join by post_id through the mode's readers (real metrics never mix with simulated ones). A
    sandbox creator's posts are left out of `real`."""
    if mode not in ("real", "sandbox"):
        raise ValueError(f"unknown evidence mode {mode!r}")
    ids = [str(c) for c in creator_ids] if creator_ids is not None else None
    rows = conn.execute(
        """SELECT DISTINCT ON (a.own_asset_id) a.own_asset_id, a.own_post_id, a.schema, a.observed,
                  p.creator_id, p.format, p.status, s.job_id,
                  o.requested, o.observed AS perf_observed, o.scores, o.closeness,
                  (SELECT coalesce(sum(c.usd), 0) FROM cost_entries c WHERE c.job_id = s.job_id) AS job_usd
           FROM reference_annotations a JOIN posts p ON p.id = a.own_post_id
           JOIN creators cr ON cr.id = p.creator_id
           JOIN assets s ON s.id = a.own_asset_id
           LEFT JOIN LATERAL (SELECT requested, observed, scores, closeness FROM performance_observations po
                              WHERE po.asset_id = a.own_asset_id
                              ORDER BY (po.stage = 'confirm') DESC, po.created_at DESC LIMIT 1) o ON true
           WHERE a.source = 'own' AND a.schema = ANY(%s) AND (%s::uuid[] IS NULL OR p.creator_id = ANY(%s::uuid[]))
             AND (%s = 'sandbox' OR cr.publish_mode = 'instagram')
           ORDER BY a.own_asset_id, array_position(%s::text[], a.schema)""",
        (list(schemas), ids, ids, mode, list(schemas))).fetchall()
    return [{"asset_id": str(r["own_asset_id"]), "post_id": str(r["own_post_id"]), "creator_id": str(r["creator_id"]),
             "format": r["format"], "status": r["status"], "schema": r["schema"], "observed": r["observed"] or {},
             "performance": ({"requested": r["requested"] or {}, "observed": r["perf_observed"] or {},
                              "scores": r["scores"] or {},
                              "closeness": float(r["closeness"]) if r["closeness"] is not None else None}
                             if r["requested"] is not None else None),
             "job_usd": float(r["job_usd"] or 0)} for r in rows]


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
    setting, exploration...), the shot types of its shot sheet, the pattern cards it applied, the creator's archetype,
    the performance controls the shots asked for (`control`: "name=level", from performance_observations) and the
    directed shot-sheet fields named in policy brain.attribution.directed (shot_size, device, camera_move...). Each
    group also carries its posts' median all-in cost (`usd_per_post`: every ledger entry booked to the post or its
    jobs, with its jobs' share of their machines' whole bill, deploy.report.machine_cost_by_job) and `reach_per_usd`.
    `real` reads real publications only; `sandbox` the simulated ones, labelled SIMULATED."""
    import statistics
    if mode not in ("real", "sandbox"):
        raise ValueError(f"unknown evidence mode {mode!r}")
    target = int((policy.get("brain", "analysis", "target_age_s", default=259200) if policy else 259200))
    dims = list(dims or ("format", "outfit", "setting", "shot_type", "pattern_card", "archetype"))
    directed = dict((policy.get("brain", "attribution", "directed", default={}) if policy else {}) or {})
    groups: dict = {}
    seen: list[tuple[dict, dict, str]] = []
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
            shots = [s for s in body.get("shots") or [] if isinstance(s, dict)]
            values = {"format": [r["format"]], "archetype": [archetype[0]] if archetype[0] else [],
                      "shot_type": sorted({s.get("shot_type") for s in body.get("shots") or []
                                           if isinstance(s, dict) and s.get("shot_type")}),
                      "pattern_card": sorted({c.get("key") for c in body.get("pattern_cards") or []
                                              if isinstance(c, dict) and c.get("applied") and c.get("key")})}
            if "control" in dims:
                values["control"] = _controls(conn, r["id"])
            for d, path in directed.items():
                if d in dims:
                    values[d] = sorted({str(v)[:60] for v in (_dig(s, path) for s in shots) if v not in (None, "")})
            for d in dims:
                if d not in values and attrs.get(d):
                    values[d] = [str(attrs[d])[:60]]
            posts.append((m, values))
            seen.append((m, values, str(r["id"])))
        if not posts:
            continue
        own = statistics.median(float(m["reach"]) for m, _ in posts)
        for m, values in posts:
            m["_lift"] = float(m["reach"]) / own if own else None
    costs = post_costs(conn, [pid for _, _, pid in seen])
    for m, values, pid in seen:
        for d in dims:
            for v in values.get(d) or []:
                groups.setdefault(d, {}).setdefault(v, []).append((m, m["_lift"], costs.get(pid)))
    out: dict = {}
    for d, by in groups.items():
        out[d] = {}
        for v, items in by.items():
            reach = [float(m["reach"]) for m, _, _ in items]
            row = {"n": len(items), "median_reach": statistics.median(reach)}
            lifts = [x for _, x, _ in items if x is not None]
            row["lift"] = round(statistics.median(lifts), 3) if lifts else None
            usd = [c for _, _, c in items if c is not None]
            if usd:
                row["usd_per_post"] = round(statistics.median(usd), 4)
                if row["usd_per_post"] > 0:
                    row["reach_per_usd"] = round(row["median_reach"] / row["usd_per_post"], 1)
            for k in RATES:
                vals = [1000.0 * float(m[k]) / float(m["reach"]) for m, _, _ in items if m.get(k) is not None]
                if vals:
                    row[f"{k}_per_1k"] = round(statistics.median(vals), 2)
            if mode == "sandbox":
                row["label"] = LABEL
            out[d][v] = row
    return out


def _dig(node, path: str):
    for k in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(k)
    return node


def _controls(conn, post_id) -> list[str]:
    """The performance controls the post's assets were directed at ("name=level", performance_observations
    requested, the newest observation per asset)."""
    rows = conn.execute("SELECT DISTINCT ON (asset_id) requested FROM performance_observations WHERE post_id = %s "
                        "ORDER BY asset_id, (stage = 'confirm') DESC, created_at DESC", (post_id,)).fetchall()
    return sorted({f"{k}={v}" for r in rows for k, v in (r["requested"] or {}).items() if v is not None})


def post_costs(conn, post_ids: list[str]) -> dict[str, float]:
    """Each post's all-in cost: every non-GPU ledger entry booked to it or its jobs, plus its jobs' share of the whole
    bill of the machines they ran on (deploy.report.machine_cost_by_job; boot, downloads, idle and failures
    included), or, for jobs no machine bill covers, their attributed GPU entries."""
    if not post_ids:
        return {}
    from ..deploy import report
    rows = conn.execute(
        """SELECT p.id, p.created_at,
                  (SELECT coalesce(sum(c.usd), 0) FROM cost_entries c
                    WHERE (c.post_id = p.id OR c.job_id IN (SELECT j.id FROM jobs j WHERE j.post_id = p.id))
                      AND c.category NOT IN ('gpu', 'gpu_billed')) AS other,
                  (SELECT array_agg(j.id::text) FROM jobs j WHERE j.post_id = p.id) AS jobs
           FROM posts p WHERE p.id = ANY(%s::uuid[])""", (list(post_ids),)).fetchall()
    if not rows:
        return {}
    from datetime import timedelta
    machine = report.machine_cost_by_job(conn, since=min(r["created_at"] for r in rows) - timedelta(days=1))
    out = {}
    for r in rows:
        jobs = r["jobs"] or []
        billed = sum(float(machine[j]) for j in jobs if j in machine)
        loose = [j for j in jobs if j not in machine]
        gpu = float(conn.execute("SELECT coalesce(sum(usd), 0) u FROM cost_entries WHERE category = 'gpu' "
                                 "AND job_id = ANY(%s::uuid[])", (loose,)).fetchone()["u"]) if loose else 0.0
        out[str(r["id"])] = round(float(r["other"] or 0) + billed + gpu, 6)
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
