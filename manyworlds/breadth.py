"""The breadth test (2026-11-23): one owner-approved run that exercises every capability the director can compose
(talking, walking, posing, dancing, props, every camera device, outfit changes, transitions, labels, voiceover over
B-roll, sound), each at its best, with every gate's verdict, the file's numbers and the dollars recorded.

    mw test breadth --plan default --dry-run          # what it would make and roughly what it costs; creates nothing
    mw test breadth --plan default [--items a,b]      # make it (resumes the same run when run again)
    mw test breadth-report <run>                      # per item and per capability: gates, lip sync, WER, file, $

A plan (policy/test_overlay.yaml `breadth.plans.<name>`) is a list of items. Each names the capabilities it
`exercises` and is either a post (an existing format: photo, carousel, story, story_clip, narrated_edit,
talking_video, with a handwritten body: scenes, script, shot sheets) or a benchmark (`bench`: a job kind that only
runs as a benchmark, e.g. dance or reference-conditioned photos). Formats are not capabilities: a capability is
wired when `breadth.capabilities.<name>` says what it needs (policy switches, job kinds) and those are there. An item
whose format production doesn't make, or whose capability isn't wired yet, is skipped with the reason; nothing
crashes on it.

The driver runs under the test overlay (policy.with_overlay) and touches only its own posts (body.breadth_run): it
starts them (production.start) and advances them (production.advance_post), never production.advance, which starts
every due post. Their jobs move into the run's batch (`test-breadth-<run>`, a test batch, so it runs while operations
are paused), at the autoscaler's urgent priority (no batch wait) and pinned to the plan's pools. It loops until every
item is ready or dropped, or the deadline passes (queued jobs are then cancelled). Run ids are derived from the plan
and its items, so running the same plan again resumes the same run. Test posts are never scheduled
(scheduled_for NULL), so the publisher never sees them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from psycopg.types.json import Jsonb

from .creator import sets
from . import operations, policy as policy_mod, queue
from .policy import Policy

log = logging.getLogger("manyworlds.breadth")

POST_FORMATS = ("photo", "carousel", "story", "story_clip", "narrated_edit", "talking_video")
ANCHOR = "@anchor"   # in a bench item's extra: the creator's reference image (@anchor = the first, @anchor:N)


def test_policy(policy: Policy, path: str | None = None) -> Policy:
    """The policy the breadth test runs under: the test overlay (operations.test_overlay.path) over `policy`."""
    path = path or policy.get("operations", "test_overlay", "path", default="policy/test_overlay.yaml")
    return policy_mod.with_overlay(policy, path)


def _cfg(policy: Policy) -> dict:
    return policy.get("breadth", default={}) or {}


def batch_name(policy: Policy, run: str) -> str:
    return f"{_cfg(policy).get('batch_prefix', 'test-breadth-')}{run}"


def urgent_priority(policy: Policy) -> int:
    """The autoscaler's urgent priority (autoscale.batch.urgent_priority): such work starts a machine at once."""
    return int(policy.get("autoscale", "batch", "urgent_priority", default=1))


# ---------------------------------------------------------------------------------------------------------- plans

@dataclass
class Item:
    id: str
    exercises: list[str]
    format: str | None = None
    caption: str = ""
    body: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)   # talk_profile, voice_engine
    bench: dict | None = None                     # {kind, jobs, extra, profile}
    creator: str | None = None                    # else the plan's
    take: dict | None = None                      # a long take (manyworlds.long_take): {kind, script, image, ...}


def parse_item(raw: dict) -> Item:
    """One plan item, checked: an id, what it exercises, and a post format with scenes, a bench kind, or a long
    take (manyworlds.long_take)."""
    if not isinstance(raw, dict) or not raw.get("id"):
        raise ValueError(f"a plan item needs an id: {raw!r}"[:200])
    iid = str(raw["id"])
    exercises = raw.get("exercises") or []
    if not isinstance(exercises, list) or not exercises:
        raise ValueError(f"item {iid}: `exercises` must list the capabilities it tests")
    item = Item(id=iid, exercises=[str(x) for x in exercises], format=raw.get("format"),
                caption=str(raw.get("caption") or ""), body=dict(raw.get("body") or {}),
                options=dict(raw.get("options") or {}), bench=raw.get("bench"), creator=raw.get("creator"),
                take=raw.get("take"))
    if sum(bool(x) for x in (item.format, item.bench, item.take)) != 1:
        raise ValueError(f"item {iid}: give one of a post `format`, a `bench` or a `take`")
    if item.take is not None:
        from . import long_take
        long_take.validate(iid, item.take)
    if item.bench is not None and not (isinstance(item.bench, dict) and item.bench.get("kind")):
        raise ValueError(f"item {iid}: `bench` needs a job kind")
    if item.format:
        scenes = item.body.get("scenes")
        if not isinstance(scenes, list) or not scenes or not all(isinstance(s, str) and s.strip() for s in scenes):
            raise ValueError(f"item {iid}: body.scenes must be a list of scene descriptions")
        shots = item.body.get("shots")
        if shots is not None and (not isinstance(shots, list) or len(shots) != len(scenes)):
            raise ValueError(f"item {iid}: body.shots needs one shot sheet per scene")
    return item


def load_plan(policy: Policy, name: str, only: list[str] | None = None) -> tuple[dict, list[Item]]:
    """The plan `breadth.plans.<name>` and its items (only those named in `only`, in plan order)."""
    plans = _cfg(policy).get("plans") or {}
    if name not in plans:
        raise ValueError(f"no breadth plan {name!r}; known: {sorted(plans)}")
    plan = plans[name] or {}
    items = [parse_item(raw) for raw in plan.get("items") or []]
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"plan {name}: item ids must be unique")
    if only:
        unknown = sorted(set(only) - set(ids))
        if unknown:
            raise ValueError(f"plan {name} has no items {unknown}; it has {ids}")
        items = [i for i in items if i.id in only]
    return plan, items


def run_id(plan_name: str, items: list[Item], creator: str | None = None) -> str:
    """The run's id: the plan's name and a digest of exactly what it makes (items and creator), so the same plan
    run again resumes the same run, and a changed plan is a new run."""
    digest = hashlib.sha256(json.dumps({"items": [asdict(i) for i in items], "creator": creator},
                                       sort_keys=True, default=str).encode()).hexdigest()[:6]
    return f"{_slug(plan_name)}-{digest}"


def _slug(text: str) -> str:
    """Batch-safe: lowercase letters, digits and hyphens only (no LIKE wildcards in a batch prefix)."""
    return re.sub(r"[^a-z0-9-]+", "-", str(text).lower()).strip("-") or "run"


# ------------------------------------------------------------------------------------------------------ skipping

def _creator(conn, ref: str | None):
    if not ref:
        return None
    return conn.execute("SELECT id, handle, anchors FROM creators WHERE handle = %s OR id::text = %s",
                        (ref, ref)).fetchone()


def _truthy_path(policy: Policy, path: str) -> bool:
    return bool(policy.get(*str(path).split(".")))


def skip_reason(policy: Policy, item: Item, creator) -> str | None:
    """Why this item can't run under `policy` now, or None. Never raises on a plan's content."""
    if creator is None:
        return f"no creator {item.creator!r}" if item.creator else "no creator: give the plan or item a creator"
    anchors = creator["anchors"] or {}
    if not anchors.get("reference_images"):
        return f"creator {creator['handle']} has no reference images (the review gates judge identity against them)"
    caps = _cfg(policy).get("capabilities") or {}
    kinds = policy.get("job_kinds", default={}) or {}
    for cap in item.exercises:
        if cap not in caps:
            return f"capability {cap} is not registered in breadth.capabilities"
        need = caps[cap] or {}
        for path in need.get("requires") or []:
            if not _truthy_path(policy, path):
                return f"capability {cap} is not wired yet: policy {path} is off or missing"
        for kind in need.get("kinds") or []:
            if kind not in kinds:
                return f"capability {cap} is not wired yet: job kind {kind} is not in job_kinds"
    if item.format:
        if item.format not in POST_FORMATS:
            return f"format {item.format} is not a post format ({', '.join(POST_FORMATS)})"
        made = policy.get("production", "formats", default=[]) or []
        if item.format not in made:
            return f"format {item.format} is not in production.formats"
        if item.body.get("shots"):
            from .direction import ShotSheet
            for i, shot in enumerate(item.body["shots"]):
                try:
                    ShotSheet.model_validate(shot)
                except Exception as e:  # noqa: BLE001 - a bad shot sheet skips its item, not the run
                    return f"shot {i} is not a valid shot sheet: {str(e)[:200]}"
    elif item.take:
        from . import long_take
        reason = long_take.skip_reason(policy, item.take, creator)
        if reason:
            return reason
        try:
            _fill_anchors(item.take["image"], anchors)
        except ValueError as e:
            return f"take.image: {e}"
    else:
        if item.bench["kind"] not in kinds:
            return f"bench kind {item.bench['kind']} is not in job_kinds"
    profile = item.options.get("talk_profile")
    if profile and profile not in (policy.get("generation", "talk_profiles", default={}) or {}):
        return f"talk profile {profile} is not in generation.talk_profiles"
    want = item.options.get("voice_engine")
    if want:
        from . import voices
        have = voices.engine(anchors)
        if have != want:
            return f"creator {creator['handle']} speaks with {have}; this item needs {want} (mw voice design/pick)"
    return None


# ----------------------------------------------------------------------------------------------------- estimates

def _pools_by_family(policy: Policy, plan: dict) -> dict[str, list[str]]:
    return {str(k): list(v) for k, v in ((plan.get("pools") or _cfg(policy).get("pools_by_family") or {})).items()}


def _pool_price(policy: Policy, plan: dict, kind: str) -> tuple[str | None, float]:
    """(the pool a kind's job would run in for this test, its usd per hour): the plan's pin, else the kind's first."""
    pools = _pools_by_family(policy, plan).get(policy.family(kind)) or \
        list(policy.get("job_kinds", kind, "pools", default=[]) or [])
    if not pools:
        return None, 0.0
    return pools[0], float(policy.get("gpu_pools", pools[0], "usd_per_hour", default=0) or 0)


def expected_jobs(policy: Policy, item: Item, creator) -> list[dict]:
    """The jobs an item should make on its first try: [{kind, n, output_seconds}] (repairs come on top)."""
    if item.bench:
        return [{"kind": item.bench["kind"], "n": int(item.bench.get("jobs", 1)), "output_seconds": None}]
    if item.take:
        from . import long_take
        return long_take.expected_jobs(policy, item.take)
    from . import production, voices
    from .talk_segments import estimate_seconds
    prod = policy.get("production", default={}) or {}
    body, scenes = item.body, item.body["scenes"]
    out = [{"kind": prod.get("image_kind", "image.zimage"), "n": len(scenes), "output_seconds": None}]
    wps = float(policy.get("generation", "talk_long", "words_per_second", default=2.5))
    designed = voices.engine((creator or {}).get("anchors") if creator else None) == "voxcpm"
    if item.format == "story_clip":
        routing = (prod.get("routing") or {}).get("story_clip") or {}
        kind = (routing.get("kinds") or [prod.get("video_kind", "comfy.wan_i2v")])[0]
        out.append({"kind": kind, "n": 1, "output_seconds": None})
    elif item.format == "narrated_edit":
        out.append({"kind": "image.narrated", "n": 1, "output_seconds": None})
        if designed:
            out.append({"kind": "voice.speak", "n": 1, "output_seconds": None})
    elif item.format == "talking_video":
        shots = body.get("shots") or []
        hybrid = bool(policy.get("generation", "talk_hybrid", "enabled", default=False)) and len(scenes) >= 2
        if shots:
            takes = production.shot_takes(policy, shots, cutaways=hybrid)
        else:
            takes = [(t, "talk", 0) for t in production.talk_segments(policy, body.get("script", ""), hybrid=hybrid)]
        motion = (policy.get("generation", "talk_motion", default={}) or {}).get("kind") or prod.get("video_kind")
        for text, mode, i in takes:
            seconds = estimate_seconds(production.unmarked(text), wps)
            if mode == "narrated":
                out.append({"kind": "image.narrated", "n": 1, "output_seconds": seconds})
            else:
                out.append({"kind": prod.get("talk_kind", "talk.infinitetalk"), "n": 1, "output_seconds": seconds})
                shot = shots[i] if i < len(shots) else None
                if production.moves(policy, shot) and production.talk_v2v(policy):
                    out.append({"kind": motion, "n": len(production.motion_lengths(policy, seconds)),
                                "output_seconds": None})
            if designed:
                out.append({"kind": "voice.speak", "n": 1, "output_seconds": None})
    return out


def estimate(conn, policy: Policy, plan: dict, item: Item, creator) -> dict:
    """What an item should cost on its first try: GPU seconds per job (measured from finished attempts where they
    exist, else policy: autoscale.scale.default_job_seconds, or breadth.estimate.seconds_per_output_second for
    kinds priced by length), at its pool's price, plus the review calls. Labelled by source."""
    from .deploy import autoscale
    est = _cfg(policy).get("estimate") or {}
    per_second = est.get("seconds_per_output_second") or {}
    review_usd = float(est.get("review_usd_per_output", 0))
    candidates = int(policy.get("production", "candidates_per_scene", default=4))
    jobs, gpu_usd, reviews = [], 0.0, 0.0
    for j in expected_jobs(policy, item, creator):
        if j["output_seconds"] is not None and j["kind"] in per_second:
            seconds, source = float(per_second[j["kind"]]) * float(j["output_seconds"]), "policy (per output second)"
        else:
            seconds, source = autoscale.job_seconds(conn, policy, j["kind"])
        pool, usd_h = _pool_price(policy, plan, j["kind"])
        usd = j["n"] * seconds * usd_h / 3600
        gpu_usd += usd
        outputs = j["n"] * (candidates if j["kind"].startswith("image.") and j["kind"] != "image.narrated" else 1)
        reviews += outputs * review_usd
        jobs.append({**j, "seconds_each": round(seconds), "source": source, "pool": pool, "usd": round(usd, 4)})
    return {"jobs": jobs, "gpu_usd": round(gpu_usd, 4), "review_usd": round(reviews, 4),
            "usd": round(gpu_usd + reviews, 4)}


def machines_estimate(policy: Policy, plan: dict, kinds: set[str]) -> list[dict]:
    """The machines the run needs (one per machine family: merged families share one) and their boot cost
    (breadth.estimate.boot_seconds, per family or default), at each family's pinned pool price."""
    from .deploy import autoscale
    est = _cfg(policy).get("estimate") or {}
    boots = est.get("boot_seconds") or {}
    fams: dict[str, str] = {}
    for kind in sorted(kinds):
        if policy.get("gpu_pools", (policy.get("job_kinds", kind, "pools", default=[]) or ["local"])[0],
                      "provider") == "local":
            continue   # hosted APIs and the control plane rent nothing
        fams.setdefault(autoscale.machine_family(policy, kind), kind)
    out = []
    for fam, kind in sorted(fams.items()):
        pool, usd_h = _pool_price(policy, plan, kind)
        seconds = float(boots.get(fam, boots.get("default", 0)))
        out.append({"family": fam, "pool": pool, "boot_seconds": seconds, "boot_usd": round(seconds * usd_h / 3600, 4),
                    "kinds": sorted(k for k in kinds if autoscale.machine_family(policy, k) == fam)})
    return out


# -------------------------------------------------------------------------------------------------------- the run

def _run_posts(conn, run: str) -> list[dict]:
    return conn.execute("SELECT * FROM posts WHERE body->>'breadth_run' = %s ORDER BY created_at, id",
                        (run,)).fetchall()


def _post_for(conn, run: str, item_id: str):
    return conn.execute("SELECT * FROM posts WHERE body->>'breadth_run' = %s AND body->>'breadth_item' = %s",
                        (run, item_id)).fetchone()


def _bench_batch(batch: str, item: Item) -> str:
    return f"{batch}-{_slug(item.id)}"


def _fill_anchors(value: Any, anchors: dict) -> Any:
    """@anchor / @anchor:N in a bench item's extra -> the creator's reference image uri."""
    if isinstance(value, str) and value.startswith(ANCHOR):
        refs = anchors.get("reference_images") or []
        n = int(value.split(":", 1)[1]) if ":" in value else 0
        if n >= len(refs):
            raise ValueError(f"{value}: the creator has {len(refs)} reference images")
        return refs[n]["uri"]
    if isinstance(value, list):
        return [_fill_anchors(v, anchors) for v in value]
    if isinstance(value, dict):
        return {k: _fill_anchors(v, anchors) for k, v in value.items()}
    return value


def create_post(conn, policy: Policy, plan_name: str, run: str, item: Item, creator) -> dict:
    """The item's post, once: a 'planned' post carrying the run (body.breadth_*), never scheduled, then started."""
    existing = _post_for(conn, run, item.id)
    if existing is None:
        body = {**item.body, "hashtags": item.body.get("hashtags") or [],
                "platform_ai_label": True,   # disclosure: every post sets the platform's AI flag
                "breadth_run": run, "breadth_plan": plan_name, "breadth_item": item.id,
                "breadth_creator": creator["handle"],
                "breadth_exercises": item.exercises, "breadth_options": item.options}
        conn.execute("INSERT INTO posts (creator_id, platform, format, status, caption, body, scheduled_for) "
                     "VALUES (%s, %s, %s, 'planned', %s, %s, NULL)",
                     (creator["id"], item.body.get("platform", "instagram"), item.format, item.caption, Jsonb(body)))
        existing = _post_for(conn, run, item.id)
    if existing["status"] == "planned":
        from . import production
        production.start(conn, policy, existing)
        existing = _post_for(conn, run, item.id)
    return existing


def create_bench(conn, policy: Policy, plan: dict, plan_name: str, run: str, batch: str, item: Item, creator) -> str:
    """The item's benchmark jobs, once, in their own batch under the run's (bench.enqueue checks its inputs)."""
    from . import bench
    b = _bench_batch(batch, item)
    if conn.execute("SELECT 1 FROM jobs WHERE batch = %s LIMIT 1", (b,)).fetchone():
        return b
    extra = _fill_anchors(dict(item.bench.get("extra") or {}), creator["anchors"] or {})
    extra.setdefault("creator_id", str(creator["id"]))   # the review gates judge it against the creator
    extra["breadth"] = {"run": run, "plan": plan_name, "item": item.id, "exercises": item.exercises,
                      "creator": creator["handle"]}
    kind = item.bench["kind"]
    bench.enqueue(conn, policy, kind, int(item.bench.get("jobs", 1)), batch=b, extra=extra,
                  pools=_pools_by_family(policy, plan).get(policy.family(kind)), profile=item.bench.get("profile"))
    return b


def adopt_jobs(conn, policy: Policy, plan: dict, run: str, batch: str) -> int:
    """Move the run's posts' new jobs (production puts them in post-<id> batches, which paused machines never take)
    into the run's batch at urgent priority, pinned to the plan's pools for their family, with the item's talk
    profile merged into talk jobs' settings. Bench jobs get the priority. Only queued jobs; returns how many moved."""
    prio, pins = urgent_priority(policy), _pools_by_family(policy, plan)
    talk_kind = policy.get("production", "talk_kind", default="talk.infinitetalk")
    profiles = policy.get("generation", "talk_profiles", default={}) or {}
    rows = conn.execute(
        """SELECT j.id, j.kind, j.pools, j.payload, j.batch, p.body->'breadth_options' AS options
           FROM jobs j JOIN posts p ON p.id = j.post_id
           WHERE p.body->>'breadth_run' = %s AND j.status = 'queued' AND j.batch IS DISTINCT FROM %s""",
        (run, batch)).fetchall()
    for r in rows:
        payload = dict(r["payload"] or {})
        profile = (r["options"] or {}).get("talk_profile")
        if r["kind"] == talk_kind and profile and profile in profiles and payload.get("talk_profile") != profile:
            payload["params"] = {**(payload.get("params") or {}), **(profiles[profile] or {})}
            payload["talk_profile"] = profile
        conn.execute("UPDATE jobs SET batch = %s, priority = greatest(priority, %s), pools = coalesce(pools, %s), "
                     "payload = %s, updated_at = now() WHERE id = %s AND status = 'queued'",
                     (batch, prio, pins.get(policy.family(r["kind"])), Jsonb(payload), r["id"]))
    conn.execute("UPDATE jobs SET priority = %s, updated_at = now() WHERE batch LIKE %s AND status = 'queued' "
                 "AND priority < %s", (prio, f"{batch}-%", prio))
    return len(rows)


def plates_batch(batch: str) -> str:
    """Where the run's creators' set and outfit plates are made: a test batch (paused machines take it), outside the
    bench pattern {batch}-%."""
    return f"{batch}.plates"


def open_test_work(conn, policy: Policy, batch: str) -> dict:
    """Open jobs of other test batches (operations.test_batch_prefixes): they could share this run's merged machine
    or start a second one. {batch: open jobs}."""
    prefixes = policy.get("operations", "test_batch_prefixes", default=["bench-", "test-"]) or []
    rows = conn.execute(
        "SELECT batch, count(*) AS n FROM jobs WHERE status IN ('queued', 'running') AND batch LIKE ANY(%s) "
        "AND batch <> %s AND batch NOT LIKE %s AND batch <> %s GROUP BY batch ORDER BY batch",
        ([f"{p}%" for p in prefixes], batch, f"{batch}-%", plates_batch(batch))).fetchall()
    return {r["batch"]: r["n"] for r in rows}


def adopt_plates(conn, policy: Policy, batch: str, creator_ids: list) -> int:
    """Set and outfit plates are queued in post-sets-<creator> batches, which paused machines never take, and only
    the (paused) producer tick picks them. Move the run's creators' queued plate jobs into the plates batch at the
    run's priority, so items testing set/outfit consistency get their references."""
    if not creator_ids:
        return 0
    return conn.execute(
        "UPDATE jobs SET batch = %s, priority = greatest(priority, %s), updated_at = now() WHERE status = 'queued' "
        "AND batch = ANY(%s)", (plates_batch(batch), urgent_priority(policy),
                                [f"post-sets-{c}" for c in creator_ids])).rowcount


def run_spend(conn, batch: str) -> float:
    """What the run has cost so far: every ledger entry of its jobs except attributed GPU time, plus what the
    machines that ran its jobs billed (idle and boot included)."""
    pats = (batch, f"{batch}-%", plates_batch(batch))
    jobs = conn.execute("SELECT coalesce(sum(c.usd), 0) AS u FROM cost_entries c JOIN jobs j ON j.id = c.job_id "
                        "WHERE c.category <> 'gpu' AND (j.batch = %s OR j.batch LIKE %s OR j.batch = %s)",
                        pats).fetchone()["u"]
    machines = conn.execute(
        "SELECT coalesce(sum(g.billed_usd + coalesce(g.est_extra_usd, 0)), 0) AS u FROM gpu_groups g WHERE g.name IN "
        "(SELECT a.gpu_group FROM attempts a JOIN jobs j ON j.id = a.job_id "
        " WHERE j.batch = %s OR j.batch LIKE %s OR j.batch = %s)", pats).fetchone()["u"]
    return float(jobs) + float(machines)


def _take_lease(conn, run: str, seconds: float) -> str | None:
    """One driver per run at a time (a lease row that lapses, so a crashed driver can be restarted)."""
    import uuid
    holder = uuid.uuid4().hex
    got = conn.execute(
        """INSERT INTO leases (name, holder, until) VALUES (%(n)s, %(h)s, now() + make_interval(secs => %(s)s))
           ON CONFLICT (name) DO UPDATE SET holder = excluded.holder, until = excluded.until
           WHERE leases.until < now() RETURNING holder""", {"n": f"breadth:{run}", "h": holder, "s": seconds}).fetchone()
    return holder if got else None


class LostLease(RuntimeError):
    """Another driver took this run over (this one stalled past its lease): stop, don't advance twice."""


def _renew_lease(conn, run: str, holder: str, seconds: float) -> None:
    r = conn.execute("UPDATE leases SET until = now() + make_interval(secs => %s) WHERE name = %s AND holder = %s "
                     "RETURNING name", (seconds, f"breadth:{run}", holder)).fetchone()
    if r is None:
        raise LostLease(f"breadth run {run}: another driver holds the lease")


def _release_lease(conn, run: str, holder: str) -> None:
    conn.execute("DELETE FROM leases WHERE name = %s AND holder = %s", (f"breadth:{run}", holder))


def _review(conn, policy: Policy, llm, store, batches: list[str]) -> dict:
    """Review the run's outputs under the test overlay (the direction gate report-only), batch by batch."""
    from .review import gates
    from .review import video as video_review
    out: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        for b in batches:
            d = Path(tmp) / _slug(b)
            d.mkdir()
            r = {"images": gates.run_pending(conn, policy, llm, store, d, batch=b),
                 "clips": video_review.run_pending(conn, policy, llm, store, d, batch=b)}
            if any(v for x in r.values() for v in x.values()):
                out[b] = r
    return out


def run(conn, policy: Policy, plan_name: str, *, items: list[str] | None = None, run: str | None = None,
        creator: str | None = None, store=None, llm=None, dry_run: bool = False, once: bool = False,
        hours: float | None = None, sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> dict:
    """Make the plan (or, with dry_run, say what it would make and cost, creating nothing). `policy` is the shipped
    policy; the run applies the test overlay itself. Returns the run's summary."""
    pol = test_policy(policy)
    cfg = _cfg(pol)
    plan, chosen = load_plan(pol, plan_name, items)
    who = creator or plan.get("creator")
    rid = _slug(run) if run else run_id(plan_name, chosen, who)
    batch = batch_name(pol, rid)
    creators = {i.id: _creator(conn, i.creator or who) for i in chosen}
    skipped = {i.id: r for i in chosen if (r := skip_reason(pol, i, creators[i.id]))}
    todo = [i for i in chosen if i.id not in skipped]
    head = {"run": rid, "batch": batch, "plan": plan_name, "priority": urgent_priority(pol),
            "pools": _pools_by_family(pol, plan), "skipped": skipped}
    blockers = open_test_work(conn, pol, batch)
    if dry_run:
        return {**head, **_dry_run(conn, pol, plan, rid, todo, creators), "blockers": blockers}
    if blockers and cfg.get("refuse_beside_open_tests", True):
        raise SystemExit(f"other test work is open and could share or start machines: {blockers}. "
                         "Finish or cancel it first.")
    if operations.is_live(pol) and cfg.get("require_paused", True):
        raise SystemExit("operations are live: the producer would advance and publish these posts too. "
                         "The breadth test runs while operations.mode is paused.")
    policy_mod.register(conn, pol)   # cost entries name the overlay's version
    loop_s = float(cfg.get("loop_seconds", 60))
    holder = _take_lease(conn, rid, loop_s * 3)
    if holder is None:
        raise SystemExit(f"another driver is running {rid}; wait for it, or for its lease to lapse")
    try:
        return {**head, **_drive(conn, pol, plan, plan_name, rid, batch, todo, creators, store=store, llm=llm,
                                 once=once, hours=hours, sleep=sleep, now=now, holder=holder)}
    finally:
        _release_lease(conn, rid, holder)


def _dry_run(conn, pol: Policy, plan: dict, rid: str, todo: list[Item], creators: dict) -> dict:
    out, kinds, total = [], set(), 0.0
    for item in todo:
        e = estimate(conn, pol, plan, item, creators[item.id])
        kinds |= {j["kind"] for j in e["jobs"]}
        total += e["usd"]
        post = _post_for(conn, rid, item.id) if item.format else None
        out.append({"id": item.id, "exercises": item.exercises, "format": item.format,
                    "bench": (item.bench or {}).get("kind"), "options": item.options,
                    "would": f"resume (post is {post['status']})" if post else "create", "estimate": e})
    machines = machines_estimate(pol, plan, kinds)
    total += sum(m["boot_usd"] for m in machines)
    return {"dry_run": True, "items": out, "machines": machines, "estimated_usd": round(total, 2),
            "estimate_note": "first try only (repairs cost more); seconds are 'measured' from finished attempts where "
                             "they exist, else 'policy'; boots from breadth.estimate.boot_seconds",
            "daily_gpu_cap_usd": pol.get("gpu_runs", "daily_gpu_usd")}


def _drive(conn, pol: Policy, plan: dict, plan_name: str, rid: str, batch: str, todo: list[Item], creators: dict, *,
           store, llm, once: bool, hours: float | None, sleep, now, holder: str) -> dict:
    from . import production
    cfg = _cfg(pol)
    loop_s = float(cfg.get("loop_seconds", 60))
    errors: dict[str, str] = {}
    benches: dict[str, str] = {}
    takes = {item.id: item for item in todo if item.take}
    take_states: dict[str, dict] = {}
    if takes:   # their segment jobs and the shared voice line are reviewed and counted like a bench's
        from . import long_take
        benches.update({i: _bench_batch(batch, takes[i]) for i in takes})
        benches["take-voice"] = long_take.voice_batch(batch)
    for item in todo:   # created once; a restart finds them
        try:
            if item.take:
                continue   # a take's jobs are made step by step in the loop (_advance_take)
            if item.format:
                create_post(conn, pol, plan_name, rid, item, creators[item.id])
            else:
                benches[item.id] = create_bench(conn, pol, plan, plan_name, rid, batch, item, creators[item.id])
        except Exception as e:  # noqa: BLE001 - e.g. a bench whose inputs are refused: that item only
            log.exception("creating %s failed", item.id)
            errors[item.id] = f"{type(e).__name__}: {e}"[:300]
    adopt_jobs(conn, pol, plan, rid, batch)
    posts = _run_posts(conn, rid)
    # the run's start survives restarts: its first post or, for a bench-only run, its first job
    job_start = conn.execute("SELECT min(created_at) AS t FROM jobs WHERE batch = %s OR batch LIKE %s",
                             (batch, f"{batch}-%")).fetchone()["t"]
    first = min([p["created_at"] for p in posts] + ([job_start] if job_start else []), default=now())
    deadline = first + timedelta(hours=float(hours if hours is not None else cfg.get("deadline_hours", 12)))
    while True:
        queue.reap(conn)
        for post in conn.execute("SELECT * FROM posts WHERE body->>'breadth_run' = %s AND status = 'producing'",
                                 (rid,)).fetchall():
            _renew_lease(conn, rid, holder, loop_s * 3)   # a finish or review can take minutes: keep the run ours
            try:
                production.advance_post(conn, pol, post, store, llm=llm)
            except Exception as e:  # noqa: BLE001 - one post's bug must not stop the others
                log.exception("advancing %s failed", post["id"])
                errors[post["body"].get("breadth_item", str(post["id"]))] = f"{type(e).__name__}: {e}"[:300]
        for iid, item in takes.items():
            if (take_states.get(iid) or {}).get("state") in ("done", "failed"):
                continue
            _renew_lease(conn, rid, holder, loop_s * 3)   # joining and measuring a take can take minutes
            try:
                take_states[iid] = _advance_take(conn, pol, plan, plan_name, rid, batch, item, creators[iid],
                                                 store=store, llm=llm)
            except Exception as e:  # noqa: BLE001 - one take's bug must not stop the others
                log.exception("advancing take %s failed", iid)
                errors[iid] = f"{type(e).__name__}: {e}"[:300]
        adopt_jobs(conn, pol, plan, rid, batch)
        cids = sorted({str(c["id"]) for c in creators.values() if c})
        adopt_plates(conn, pol, batch, cids)
        if llm is not None and cfg.get("review_in_driver", True):
            try:
                _review(conn, pol, llm, store, [batch, plates_batch(batch), *benches.values()])
            except Exception as e:  # noqa: BLE001 - reviews are retried next loop
                errors["review"] = f"{type(e).__name__}: {e}"[:300]
        for cid in cids:   # choose reviewed plates (the paused producer tick would)
            try:
                sets.refresh(conn, pol, cid)
            except Exception as e:  # noqa: BLE001
                errors["plates"] = f"{type(e).__name__}: {e}"[:300]
        state = _state(conn, rid, list(benches.values()), takes=take_states, expected_takes=list(takes))
        spent = run_spend(conn, batch)
        log.info("breadth %s: posts %s, open bench jobs %s, spent $%.2f", rid, state["posts"],
                 state["bench_open_jobs"], spent)
        if state["done"] or once:
            break
        cap = plan.get("max_usd", cfg.get("max_usd"))   # a plan may set its own cap (longtake: owner-approved $3)
        if cap is not None and spent >= float(cap):
            state["spend_cap"] = {"spent_usd": round(spent, 2), "max_usd": float(cap)}
            state["cancelled_jobs"] = cancel_queued(conn, batch)
            break
        if now() >= deadline:
            state["deadline_passed"] = True
            if cfg.get("cancel_on_deadline", True):
                state["cancelled_jobs"] = cancel_queued(conn, batch)
            break
        _renew_lease(conn, rid, holder, loop_s * 3)
        sleep(loop_s)
    return {**state, "deadline": deadline.isoformat(), "spent_usd": round(run_spend(conn, batch), 2), "errors": errors}


def _state(conn, rid: str, bench_batches: list[str], *, takes: dict | None = None,
           expected_takes: list[str] | None = None) -> dict:
    posts = {p["body"]["breadth_item"]: p["status"] for p in _run_posts(conn, rid)}
    open_bench = {b: conn.execute("SELECT count(*) AS n FROM jobs WHERE batch = %s AND status IN ('queued', 'running')",
                                  (b,)).fetchone()["n"] for b in bench_batches}
    takes = takes or {}
    takes_done = all((takes.get(t) or {}).get("state") in ("done", "failed") for t in expected_takes or [])
    done = all(s in ("ready", "dropped") for s in posts.values()) and not any(open_bench.values()) and takes_done
    out = {"posts": posts, "bench_open_jobs": open_bench, "done": done}
    if expected_takes:
        out["takes"] = {t: takes.get(t) or {"state": "waiting"} for t in expected_takes}
    return out


def _advance_take(conn, pol: Policy, plan: dict, plan_name: str, rid: str, batch: str, item: Item, creator, *,
                  store=None, llm=None) -> dict:
    """One step of a long take (manyworlds.long_take) in the run's batches, pinned to the plan's pools."""
    from . import long_take
    meta = {"run": rid, "plan": plan_name, "item": item.id, "exercises": item.exercises,
            "creator": creator["handle"]}
    return long_take.advance(conn, pol, run=rid, batch=batch, item_id=item.id, take=item.take, creator=creator,
                             item_batch=_bench_batch(batch, item),
                             anchors_uri=_fill_anchors(item.take["image"], creator["anchors"] or {}),
                             pins=_pools_by_family(pol, plan), meta=meta, store=store, llm=llm)


def cancel_queued(conn, batch: str) -> int:
    """At the deadline: the run's queued jobs, and running ones whose lease has lapsed, are failed; running ones
    get no further attempt (max_attempts = attempts), so a retry can never boot a machine after the driver left."""
    n = len(conn.execute(
        "UPDATE jobs SET status = 'failed', error = 'breadth test deadline passed', updated_at = now() "
        "WHERE (status = 'queued' OR (status = 'running' AND lease_until < now())) "
        "AND (batch = %s OR batch LIKE %s OR batch = %s) RETURNING id",
        (batch, f"{batch}-%", plates_batch(batch))).fetchall())
    conn.execute("UPDATE jobs SET max_attempts = attempts, updated_at = now() WHERE status = 'running' "
                 "AND (batch = %s OR batch LIKE %s OR batch = %s)", (batch, f"{batch}-%", plates_batch(batch)))
    return n


# ------------------------------------------------------------------------------------------------------ the report

def _fps(path: Path) -> float | None:
    """The video stream's frame rate from ffmpeg's banner, when media.probe gives none."""
    from . import media
    try:
        r = subprocess.run([media.ffmpeg(), "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                           timeout=60)
    except (OSError, subprocess.SubprocessError, RuntimeError):
        return None
    m = re.search(r"Stream #.*Video:.*?(\d+(?:\.\d+)?) fps", r.stderr or "")
    return float(m.group(1)) if m else None


def file_info(store, uri: str) -> dict:
    """{seconds, width, height, audio, fps} of a stored video; {error} when it can't be read here."""
    from . import media
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = store.get_file(uri, Path(tmp) / ("final" + (Path(uri).suffix or ".mp4")))
            info = media.probe(path)
            return {**info, "fps": info.get("fps") or _fps(path)}
    except Exception as e:  # noqa: BLE001 - the report still shows everything else
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def _url(store, uri: str | None, seconds: int) -> str | None:
    if not uri or store is None:
        return None
    try:
        return store.public_url(uri, seconds=seconds)
    except Exception:  # noqa: BLE001
        return None


def _gates(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        g = out.setdefault(r["gate"], {"passed": 0, "failed": 0, "scores": []})
        g["passed" if r["passed"] else "failed"] += 1
        if r["score"] is not None:
            g["scores"].append(round(float(r["score"]), 3))
    return out


def _jobs_numbers(conn, where: str, arg) -> dict:
    """Render seconds, dollars by category and review rows for the jobs matching `where` (an SQL condition on j)."""
    seconds = conn.execute(f"SELECT coalesce(sum(a.gpu_seconds), 0) AS s FROM attempts a JOIN jobs j ON j.id = a.job_id "
                           f"WHERE {where}", (arg,)).fetchone()["s"]
    spend = conn.execute(f"SELECT c.category, round(sum(c.usd)::numeric, 4) AS usd FROM cost_entries c "
                         f"JOIN jobs j ON j.id = c.job_id WHERE {where} GROUP BY c.category", (arg,)).fetchall()
    reviews = conn.execute(f"SELECT j.kind, r.gate, r.passed, r.score FROM reviews r JOIN assets a ON a.id = r.asset_id "
                           f"JOIN jobs j ON j.id = a.job_id WHERE {where} ORDER BY r.created_at", (arg,)).fetchall()
    jobs = conn.execute(f"SELECT j.kind, j.status, count(*) AS n FROM jobs j WHERE {where} GROUP BY j.kind, j.status "
                        f"ORDER BY j.kind", (arg,)).fetchall()
    gates = _gates(reviews)
    return {"jobs": jobs, "render_seconds": round(float(seconds or 0)),
            "usd": round(sum(float(s["usd"] or 0) for s in spend), 4),
            "usd_by_category": {s["category"]: float(s["usd"] or 0) for s in spend},
            "gates": gates, "lip_sync_corr": gates.get("lip_sync", {}).get("scores", []),
            "wer": gates.get("intelligibility", {}).get("scores", [])}


def report(conn, policy: Policy, run: str, *, store=None, probe: bool = True) -> dict:
    """Everything the run made, per item and grouped by capability, with the machines it used and the bill."""
    pol = test_policy(policy)
    batch = batch_name(pol, run)
    url_s = int(_cfg(pol).get("report_url_seconds", 3600))
    items: list[dict] = []
    plan_name = who = None
    for p in _run_posts(conn, run):
        body = p["body"] or {}
        plan_name = plan_name or body.get("breadth_plan")
        who = who or body.get("breadth_creator")
        video = (body.get("video") or {}).get("uri")
        cover = body.get("cover")
        cover_uri = cover.get("uri") if isinstance(cover, dict) else cover
        items.append({"id": body.get("breadth_item"), "exercises": body.get("breadth_exercises") or [],
                      "format": p["format"], "options": body.get("breadth_options") or {}, "status": p["status"],
                      "drop_reason": body.get("drop_reason"), "rerouted_from": body.get("rerouted_from"),
                      **_jobs_numbers(conn, "j.post_id = %s", p["id"]),
                      "video": video, "video_url": _url(store, video, url_s),
                      "file": file_info(store, video) if (probe and video and store is not None) else None,
                      "cover": cover_uri, "cover_url": _url(store, cover_uri, url_s),
                      "finish": body.get("finish"), "finish_error": body.get("finish_error")})
    for b in conn.execute("SELECT DISTINCT batch, payload->'breadth' AS meta FROM jobs WHERE batch LIKE %s",
                          (f"{batch}-%",)).fetchall():
        meta = b["meta"] or {}
        plan_name = plan_name or meta.get("plan")
        who = who or meta.get("creator")
        numbers = _jobs_numbers(conn, "j.batch = %s", b["batch"])
        open_jobs = sum(j["n"] for j in numbers["jobs"] if j["status"] in ("queued", "running"))
        failed = sum(j["n"] for j in numbers["jobs"] if j["status"] in ("failed", "dead"))
        outputs = conn.execute("SELECT a.uri FROM assets a JOIN jobs j ON j.id = a.job_id WHERE j.batch = %s "
                               "AND a.kind IN ('video', 'image') ORDER BY a.uri", (b["batch"],)).fetchall()
        items.append({"id": meta.get("item") or b["batch"].removeprefix(f"{batch}-"),
                      "exercises": meta.get("exercises") or [], "bench_batch": b["batch"],
                      "status": "running" if open_jobs else ("failed" if failed and not outputs else "done"),
                      **numbers, "outputs": [{"uri": o["uri"], "url": _url(store, o["uri"], url_s)} for o in outputs]})
    from . import long_take
    takes = long_take.report(conn, run, store, url_s)
    for i in items:   # a long take: its finished video's numbers (colour drift, identity, sync per segment)
        if i["id"] in takes:
            i["take"] = takes[i["id"]]
        elif i.get("bench_batch") and str(i.get("bench_batch")).endswith(f"-{_slug(str(i['id']))}") and \
                conn.execute("SELECT 1 FROM jobs WHERE batch = %s AND payload->'breadth' ? 'take' LIMIT 1",
                             (i["bench_batch"],)).fetchone() and i["status"] == "done":
            i["status"] = "running"   # a chain between segments, or a take not yet joined and measured
    skipped = {}
    if plan_name:
        try:
            plan, planned = load_plan(pol, plan_name)
            seen = {i["id"] for i in items}
            for item in planned:
                if item.id not in seen:
                    skipped[item.id] = {"exercises": item.exercises, "reason": skip_reason(
                        pol, item, _creator(conn, item.creator or who or plan.get("creator"))) or "not in this run"}
        except ValueError:
            pass
    machines = conn.execute(
        """SELECT g.name, g.family, g.pool, round(g.billed_usd::numeric, 4) AS billed_usd, g.started_at, g.stopped_at,
                  g.stop_reason FROM gpu_groups g
           WHERE g.name IN (SELECT a.gpu_group FROM attempts a JOIN jobs j ON j.id = a.job_id
                            WHERE j.batch = %s OR j.batch LIKE %s)
           ORDER BY g.started_at""", (batch, f"{batch}-%")).fetchall()
    return {"run": run, "batch": batch, "verdicts": verdicts(pol, items), "items": items, "skipped": skipped,
            "takes": takes,
            "by_capability": by_capability(items, skipped), "machines": machines,
            "spent_usd": round(run_spend(conn, batch), 2),
            "usd": {"jobs": round(sum(i["usd"] for i in items), 4),
                    "machines_billed": round(sum(float(m["billed_usd"] or 0) for m in machines), 4)}}


def by_capability(items: list[dict], skipped: dict) -> dict:
    """Coverage: for each capability, the items that exercised it and how they ended."""
    out: dict = {}
    for i in items:
        for cap in i.get("exercises") or []:
            c = out.setdefault(cap, {"items": [], "statuses": {}})
            c["items"].append(i["id"])
            c["statuses"][i["status"]] = c["statuses"].get(i["status"], 0) + 1
    for iid, s in skipped.items():
        for cap in s.get("exercises") or []:
            c = out.setdefault(cap, {"items": [], "statuses": {}})
            c["items"].append(iid)
            c["statuses"]["skipped"] = c["statuses"].get("skipped", 0) + 1
            c.setdefault("skip_reasons", {})[iid] = s["reason"]
    return dict(sorted(out.items()))


def verdicts(policy: Policy, items: list[dict]) -> dict:
    """Go / no-go per format: a format whose items reached ready at breadth.go.min_pass_share or better (with at least
    min_items finished) is ready to switch on; one still producing is incomplete. Drop reasons and the cost per ready
    item (its own job costs) say why and at what price. Bench items are judged one by one by status."""
    go = _cfg(policy).get("go") or {}
    share_min, n_min = float(go.get("min_pass_share", 0.6)), int(go.get("min_items", 1))
    out: dict = {}
    for i in items:
        key = i.get("format") or f"bench:{i['id']}"
        v = out.setdefault(key, {"items": 0, "ready": 0, "dropped": 0, "open": 0, "usd": 0.0, "drop_reasons": []})
        v["items"] += 1
        v["usd"] += float(i.get("usd") or 0)
        st = i.get("status")
        if st in ("ready", "done"):
            v["ready"] += 1
        elif st in ("dropped", "failed"):
            v["dropped"] += 1
            if i.get("drop_reason"):
                v["drop_reasons"].append(f"{i['id']}: {i['drop_reason']}"[:200])
        else:
            v["open"] += 1
    for v in out.values():
        done = v["ready"] + v["dropped"]
        v["pass_share"] = round(v["ready"] / done, 2) if done else None
        v["usd_per_ready"] = round(v["usd"] / v["ready"], 3) if v["ready"] else None
        v["usd"] = round(v["usd"], 3)
        v["verdict"] = ("incomplete" if v["open"] else
                        "go" if done >= n_min and (v["pass_share"] or 0) >= share_min else "no-go")
    return dict(sorted(out.items()))
