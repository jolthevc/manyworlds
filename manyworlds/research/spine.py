"""The analytical spine (migration 056; docs/2026-11-26-unified-intelligence.md §4): one shape for every content unit,
ours or external, any format, at any level, so research, production and outcomes share one foundation.

    sync(conn, policy)       mirrors what the research and production tables hold into the spine, idempotently:
                             reference creators and posts (external), their newest coded read and code measures
                             (attribute_values), their public counts at each observed age (outcomes, raw and
                             normalised against the creator's own median at the same horizon); our creators, posts and
                             accepted assets (own), their own coded reads, real metric snapshots and production cost.
    registry checks          check_names(): a platform, medium, format, vertical or level the policy registry
                             (intelligence.spine) doesn't list is stored as `other` and reported, never a crash and never
                             a migration.

Rights: a unit holds metadata, attribute values and counts, never media. External units are `amber` (third-party
content; their derived data follows the data policy's green rules: facts, metrics and our own observations). Real
and simulated never mix: own outcomes come only from metric_snapshots rows that are not simulated (simulated metrics
live in simulated_* tables this module never reads). Sandbox creators' units carry context.sandbox.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from psycopg.types.json import Jsonb

from ..policy import Policy
from . import annotation as ann


def cfg(policy: Policy) -> dict:
    return policy.get("intelligence", "spine", default={}) or {}


def attributes(policy: Policy) -> dict:
    """The attribute registry: name -> {tier, class, level, measured?}."""
    return policy.get("intelligence", "attributes", default={}) or {}


def _known(policy: Policy, kind: str, value, problems: list | None = None) -> str | None:
    if value in (None, ""):
        return None
    v = str(value).strip().lower()
    allowed = {str(x) for x in cfg(policy).get(kind) or []}
    if allowed and v not in allowed:
        if problems is not None:
            problems.append(f"{kind} {v!r} is not in intelligence.spine.{kind}: stored as other")
        return "other"
    return v


def horizon(policy: Policy, age_s: float | None) -> str | None:
    """The named horizon an observation's age (seconds since posting) falls in, or None (too fresh)."""
    if age_s is None:
        return None
    days = float(age_s) / 86400.0
    for name, (lo, hi) in (cfg(policy).get("horizons") or {}).items():
        if float(lo) <= days < float(hi):
            return str(name)
    return None


def external_format(platform: str | None, medium: str | None) -> str | None:
    """The format of a public post from its platform and medium (a video is a Reel on Instagram, a Short on YouTube
    when short; TikTok videos are shorts)."""
    if medium in ("photo", "carousel"):
        return medium
    if medium == "video":
        return {"instagram": "reel", "tiktok": "short", "youtube": "short"}.get(str(platform or ""), "other")
    return None


# --- writes ---------------------------------------------------------------------------------------------------------
def upsert_unit(conn, *, source: str, level: str, ref_table: str, ref_id, parent_id=None, platform=None,
                format=None, medium=None, vertical=None, seq=None, start_s=None, end_s=None, published_at=None,
                context: dict | None = None, rights: str = "green") -> str:
    row = conn.execute(
        """INSERT INTO content_units (source, level, parent_id, ref_table, ref_id, platform, format, medium, vertical,
                                      seq, start_s, end_s, published_at, context, rights)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (ref_table, ref_id, level, coalesce(seq, -1)) WHERE ref_table IS NOT NULL
           DO UPDATE SET parent_id = excluded.parent_id, platform = excluded.platform, format = excluded.format,
                         medium = excluded.medium, vertical = excluded.vertical, start_s = excluded.start_s,
                         end_s = excluded.end_s, published_at = excluded.published_at, context = excluded.context,
                         rights = excluded.rights, updated_at = now()
           RETURNING id""",
        (source, level, parent_id, ref_table, str(ref_id), platform, format, medium, vertical, seq, start_s, end_s,
         published_at, Jsonb(context or {}), rights)).fetchone()
    return str(row["id"])


def put_value(conn, unit_id, attribute: str, *, value=None, num=None, method: str, annotator: str,
              purpose: str = "production", confidence=None, vocab_version: str | None = None) -> None:
    conn.execute(
        """INSERT INTO attribute_values (unit_id, attribute, value, num, confidence, method, annotator, purpose,
                                         vocab_version)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (unit_id, attribute, method, annotator, purpose, coalesce(value, ''))
           DO UPDATE SET num = excluded.num, confidence = excluded.confidence, vocab_version = excluded.vocab_version""",
        (unit_id, attribute, None if value is None else str(value), num, confidence, method, annotator, purpose,
         vocab_version))


def coded_values(coded: dict | None, medium: str | None) -> list[tuple[str, str | None, float | None]]:
    """(attribute, value, number) rows of one coded read, for the fields its medium is asked: an enum gives one row
    (unknown kept as `unknown`), a label list one row per label (an empty list one row `(none)`: the read saw none
    of the labels), a number one numeric row."""
    out: list[tuple[str, str | None, float | None]] = []
    for f in ann.CODED_FIELDS:
        if f not in (coded or {}) or not ann.applies(f, medium):
            continue
        v = coded[f]
        if f in ann.NUMBER_FIELDS:
            try:
                out.append((f, None, float(v)))
            except (TypeError, ValueError):
                out.append((f, ann.UNKNOWN, None))
        elif f in ann.MULTI_FIELDS:
            labels = [str(x) for x in (v or []) if x not in (None, "")]
            if f in ann.ORDERED_FIELDS:
                labels = [f"{i}:{x}" for i, x in enumerate(labels)]
            out += [(f, x, None) for x in labels] or [(f, "(none)", None)]
        else:
            out.append((f, ann.UNKNOWN if v in (None, "") else str(v), None))
    return out


def put_read(conn, unit_id, observed: dict, *, medium: str | None, annotator: str, method: str = "model",
             purpose: str = "production") -> int:
    """A coded read's values on a unit (each with the field's stored vocabulary hash). Replaces the annotator's earlier
    values for the same purpose, so a re-read never leaves stale labels behind."""
    coded = (observed or {}).get("coded") or {}
    versions = (observed or {}).get("vocab_versions") or {}
    conn.execute("DELETE FROM attribute_values WHERE unit_id = %s AND annotator = %s AND purpose = %s AND method = %s",
                 (unit_id, annotator, purpose, method))
    rows = coded_values(coded, medium)
    for f, v, n in rows:
        put_value(conn, unit_id, f, value=v, num=n, method=method, annotator=annotator, purpose=purpose,
                  vocab_version=versions.get(f))
    return len(rows)


# Where each measured attribute sits in research.video_measure.measure()'s result (hard cuts, not punch-ins).
MEASURED = {"cuts_per_10s": ("hard", "cuts_per_10s"), "duration_s": ("duration_s",),
            "first_word_s": ("speech", "first_word_s"), "words_per_s": ("speech", "words_per_s"),
            "speech_share": ("speech_share",), "loudness_lufs": ("audio", "loudness_lufs"),
            "silence_share": ("audio", "silence_share")}


def _dig(d: dict, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def put_measured(conn, unit_id, measured: dict) -> int:
    """research.video_measure's numbers on a unit (method code), each read from its own place (MEASURED)."""
    n = 0
    for attr, path in MEASURED.items():
        v = _dig(measured or {}, *path)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
            put_value(conn, unit_id, attr, num=float(v), method="code", annotator="video_measure")
            n += 1
    return n


def put_outcome(conn, unit_id, metric: str, horizon_name: str, *, raw, age_s=None, observed_at=None,
                followers_at=None, source: str | None = None) -> None:
    conn.execute(
        """INSERT INTO outcomes (unit_id, metric, horizon, age_s, raw, followers_at, observed_at, source)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (unit_id, metric, horizon) DO UPDATE SET
               age_s = excluded.age_s, raw = excluded.raw, followers_at = excluded.followers_at,
               observed_at = excluded.observed_at, source = excluded.source
           WHERE outcomes.observed_at IS NULL OR excluded.observed_at >= outcomes.observed_at""",
        (unit_id, metric, horizon_name, age_s, raw, followers_at, observed_at, source))


def _ts(v) -> datetime | None:
    if v is None or isinstance(v, datetime):
        return v
    try:
        t = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def _with_derived(policy: Policy, counts: dict) -> dict:
    out = {k: v for k, v in counts.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    for name, parts in (cfg(policy).get("derived") or {}).items():
        have = [out[p] for p in parts if p in out]
        if have and len(have) == len(parts):
            out[name] = float(sum(have))
    return out


def put_counts(conn, policy: Policy, unit_id, posted_at, observations: list[tuple], metrics: list[str],
               *, followers=None, source: str) -> int:
    """Raw counts of a unit at each observed age, one row per metric and horizon (the newest observation in a
    horizon wins). observations: [(observed_at, {metric: count})]."""
    posted = _ts(posted_at)
    n = 0
    allowed = set(metrics) | set((cfg(policy).get("derived") or {}).keys())
    for at, counts in observations:
        at = _ts(at)
        if not posted or not at:
            continue
        age = (at - posted).total_seconds()
        h = horizon(policy, age)
        if h is None:
            continue
        for metric, raw in _with_derived(policy, counts or {}).items():
            if metric in allowed:
                put_outcome(conn, unit_id, metric, h, raw=float(raw), age_s=age, observed_at=at,
                            followers_at=followers, source=source)
                n += 1
    return n


def normalise(conn, policy: Policy, *, source: str | None = None) -> int:
    """Each outcome's normalised value: log(raw / the creator's median of the same metric at the same horizon over
    their OTHER posts published in the baseline_days before it), when at least baseline_min_n exist; the baseline,
    its n and window stored beside it. A raw or baseline of zero is floored at one count (log of counts + 1 would
    distort ratios at scale). Returns how many were set."""
    c = cfg(policy)
    days, min_n = int(c.get("baseline_days", 180)), int(c.get("baseline_min_n", 5))
    rows = conn.execute(
        """SELECT o.id, o.metric, o.horizon, o.raw, u.published_at, p.id AS creator_unit
           FROM outcomes o JOIN content_units u ON u.id = o.unit_id
           JOIN content_units p ON p.id = u.parent_id AND p.level = 'creator'
           WHERE u.level = 'post' AND (%s::text IS NULL OR u.source = %s) AND o.raw IS NOT NULL""",
        (source, source)).fetchall()
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(str(r["creator_unit"]), r["metric"], r["horizon"])].append(r)
    n = 0
    for (_, _, _), items in groups.items():
        for r in items:
            t = r["published_at"]
            others = [float(x["raw"]) for x in items if x["id"] != r["id"] and x["published_at"] and t
                      and t - timedelta(days=days) <= x["published_at"] <= t]
            if len(others) < min_n:
                conn.execute("UPDATE outcomes SET normalized = NULL, baseline = NULL, baseline_n = %s, "
                             "baseline_window = %s WHERE id = %s", (len(others), f"{days}d trailing", r["id"]))
                continue
            base = statistics.median(others)
            value = math.log(max(float(r["raw"]), 1.0) / max(base, 1.0))
            conn.execute("UPDATE outcomes SET normalized = %s, baseline = %s, baseline_n = %s, baseline_window = %s "
                         "WHERE id = %s", (round(value, 6), base, len(others), f"{days}d trailing", r["id"]))
            n += 1
    return n


# --- the sync -------------------------------------------------------------------------------------------------------
def _external(conn, policy: Policy, problems: list) -> dict:
    c = cfg(policy)
    creators = {}
    for r in conn.execute("SELECT id, platform, handle, role, cluster, scores, followers FROM reference_creators"
                          ).fetchall():
        scores = r["scores"] or {}
        creators[str(r["id"])] = upsert_unit(
            conn, source="external", level="creator", ref_table="reference_creators", ref_id=r["id"],
            platform=_known(policy, "platforms", r["platform"], problems),
            vertical=_known(policy, "verticals", scores.get("vertical"), problems),
            context={"archetype": scores.get("archetype") or r["cluster"], "role": r["role"],
                     "followers": r["followers"]}, rights="amber")
    posts = conn.execute(
        """SELECT p.id, p.creator_id, p.platform, p.media_type, p.posted_at, p.fetched_at, p.metrics, p.snapshots
           FROM reference_posts p ORDER BY p.fetched_at DESC LIMIT %s""",
        (int(c.get("max_posts_per_sync", 5000)),)).fetchall()
    reads = {str(r["post_id"]): r for r in conn.execute(
        """SELECT DISTINCT ON (post_id) post_id, schema, model, observed FROM reference_annotations
           WHERE source = 'external' AND schema = ANY(%s) ORDER BY post_id, array_position(%s::text[], schema),
           created_at DESC""", (list(ann.READ_SCHEMAS), list(ann.READ_SCHEMAS))).fetchall()}
    measures = {str(r["post_id"]): r["measured"] for r in conn.execute(
        """SELECT DISTINCT ON (post_id) post_id, measured FROM reference_annotations
           WHERE source = 'external' AND schema LIKE 'measure_%%' ORDER BY post_id, created_at DESC""").fetchall()}
    units = values = counts = 0
    for p in posts:
        parent = creators.get(str(p["creator_id"]))
        medium = _known(policy, "media", p["media_type"], problems)
        followers = None
        unit = upsert_unit(conn, source="external", level="post", ref_table="reference_posts", ref_id=p["id"],
                           parent_id=parent, platform=_known(policy, "platforms", p["platform"], problems),
                           format=_known(policy, "formats", external_format(p["platform"], p["media_type"]), problems),
                           medium=medium, published_at=p["posted_at"], rights="amber")
        units += 1
        read = reads.get(str(p["id"]))
        if read:
            values += put_read(conn, unit, read["observed"] or {}, medium=p["media_type"],
                               annotator=str(read["model"] or read["schema"]))
        if str(p["id"]) in measures:
            values += put_measured(conn, unit, measures[str(p["id"])] or {})
        obs = [(s.get("at"), s) for s in (p["snapshots"] or []) if isinstance(s, dict)]
        metrics = {k: v for k, v in (p["metrics"] or {}).items() if k in (c.get("external_metrics") or [])}
        if metrics:
            obs.append((p["fetched_at"], metrics))
        counts += put_counts(conn, policy, unit, p["posted_at"], obs, list(c.get("external_metrics") or []),
                             followers=followers, source=f"public:{p['platform']}")
    return {"creators": len(creators), "posts": units, "values": values, "outcomes": counts}


def _own(conn, policy: Policy, problems: list) -> dict:
    c = cfg(policy)
    creators = {}
    from ..creator import profile as profile_mod
    jobs_by_series: dict = {}
    for r in conn.execute("SELECT id, handle, publish_mode, vertical, anchors FROM creators").fetchall():
        got = profile_mod.current(conn, r["id"])
        prof = (got[1] if got else {}) or {}
        verticals = prof.get("verticals") or ([r.get("vertical")] if r.get("vertical") else [])
        creators[str(r["id"])] = upsert_unit(
            conn, source="own", level="creator", ref_table="creators", ref_id=r["id"], platform="own",
            vertical=_known(policy, "verticals", verticals[0] if verticals else None, problems),
            context={"handle": r["handle"], "sandbox": r["publish_mode"] != "instagram",
                     "propositions": prof.get("propositions") or [], "verticals": verticals,
                     "playbooks": prof.get("archetypes") or []})
        for s in ((prof.get("content_system") or {}).get("series") or []):
            jobs_by_series[(str(r["id"]), str(s.get("name") or ""))] = {
                "job": s.get("job"), "arc": s.get("arc"), "proposition": s.get("proposition")}
    posts = conn.execute(
        """SELECT p.id, p.creator_id, p.format, p.status, p.body->>'series' AS series,
                  coalesce((SELECT min(published_at) FROM publications pb WHERE pb.post_id = p.id AND NOT pb.simulated),
                           CASE WHEN p.status = 'published' THEN p.published_at END) AS published_at
           FROM posts p WHERE p.status NOT IN ('planned', 'dropped')""").fetchall()
    units = values = counts = 0
    for p in posts:
        parent = creators.get(str(p["creator_id"]))
        if p["series"]:
            parent = upsert_unit(conn, source="own", level="series", ref_table="series",
                                 ref_id=f"{p['creator_id']}:{p['series']}", parent_id=parent, platform="own",
                                 context={"series": p["series"]})
        role = jobs_by_series.get((str(p["creator_id"]), str(p["series"] or ""))) or {}
        unit = upsert_unit(conn, source="own", level="post", ref_table="posts", ref_id=p["id"], parent_id=parent,
                           platform="own", format=_known(policy, "formats", p["format"], problems),
                           published_at=p["published_at"],
                           context={"status": p["status"], **{k: v for k, v in role.items() if v}})
        units += 1
        usd = conn.execute("SELECT coalesce(sum(c.usd), 0) u FROM cost_entries c WHERE c.post_id = %s",
                           (p["id"],)).fetchone()["u"]
        if usd:
            put_outcome(conn, unit, "production_usd", "production", raw=float(usd), source="ledger")
            counts += 1
        snaps = conn.execute("SELECT captured_at, metrics FROM metric_snapshots WHERE post_id = %s AND NOT simulated "
                             "ORDER BY captured_at", (p["id"],)).fetchall()
        if snaps and p["published_at"]:
            counts += put_counts(conn, policy, unit, p["published_at"],
                                 [(s["captured_at"], s["metrics"]) for s in snaps],
                                 list(c.get("own_metrics") or []), source="instagram:insights")
    for a in conn.execute(
            """SELECT a.own_asset_id, a.own_post_id, a.model, a.schema, a.observed, s.kind, s.meta
               FROM reference_annotations a JOIN assets s ON s.id = a.own_asset_id
               WHERE a.source = 'own' AND a.schema = ANY(%s)""", (list(ann.READ_SCHEMAS),)).fetchall():
        post = conn.execute("SELECT id FROM content_units WHERE ref_table = 'posts' AND ref_id = %s AND level = 'post'",
                            (str(a["own_post_id"]),)).fetchone()
        medium = "video" if a["kind"] == "video" else "photo"
        unit = upsert_unit(conn, source="own", level="shot", ref_table="assets", ref_id=a["own_asset_id"],
                           parent_id=post["id"] if post else None, platform="own", medium=medium)
        values += put_read(conn, unit, a["observed"] or {}, medium=medium, annotator=str(a["model"] or a["schema"]))
    return {"creators": len(creators), "posts": units, "values": values, "outcomes": counts}


def sync(conn, policy: Policy) -> dict:
    """Mirror the research and production tables into the spine, then normalise the outcomes. Idempotent."""
    problems: list[str] = []
    ext = _external(conn, policy, problems)
    own = _own(conn, policy, problems)
    normalised = normalise(conn, policy)
    return {"external": ext, "own": own, "normalised": normalised, "problems": sorted(set(problems))[:20]}
