"""The influencer intelligence agent (IIA): learns from named public creators what makes their photos and videos work,
and writes it into the creative pattern library the showrunner, director, talk production and casting retrieve from
(manyworlds.brain.patterns). Spec: docs/superpowers/specs/2026-11-25-influencer-intelligence.md.

A `brain.intelligence` job, commissioned by a typed request of kind `intelligence` (talent manager, creator lab, CEO;
manyworlds.brain.proposals, accepted at once within the requester's allowance and capped at
`intelligence.agent.max_usd_per_commission`) or by `mw research pilot`. Modes (payload `mode`):

    scout     a hypothesis (a thesis id or text) and any named reference handles: one plan call (iia_curate) names
              the hashtags when the payload gives none; code pulls the candidate pool (the trend watcher's hashtag
              actors), picks the sample by role quotas (named references, leaders, mid-sized, risers, adjacent, low
              comparators always) and pulls each creator's profile and recent posts (per-creator Apify actors,
              policy `intelligence.agent.actors`), with research copies of their media in the research store
    annotate  videos get `finish.measure_video` (cuts, shot lengths, pace, frames; the Fly finisher); every post gets
              a vision read (iia_annotate, Haiku, several posts per call, Message Batches) of its images or frames
              and caption into the three-layer schema (manyworlds.research.annotation); code fills evidence
    distill   code clusters the annotations (medium x format or situation) and compares high against low traction
              over each creator's own median; one iia_distill call per cluster writes a pattern card in our names
              (shot types, moves, talk-edit grammars, deliveries, motion references), validated before it is stored
              as a new card or a new version of one, status draft, with its sources (the posts) and evidence
    pilot     scout, annotate, distill in order

Every Apify item is charged to the ledger (category data, agent intelligence:<mode>, this job); every LLM call is
charged by the runtime to this job. Before each paid step the run's spend so far plus the step's estimate (policy
`intelligence.agent.estimate`) is checked against its cap (payload `usd`, else `max_usd_per_run`): a run stops at its
cap and reports what it got. Each mode is a remembered step: a rerun never pulls or asks again.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from psycopg.types.json import Jsonb

from ...llm import image_block
from ...research import annotation as ann, pulls, references, video_measure
from ...research.apify import run_actor
from ...worker.runner import NonRetryable
from .. import aesthetics, archetypes, patterns
from ..agent import AgentContext, register, remember

AGENT = "intelligence"
MODES = ("scout", "annotate", "distill", "pilot")
CLUSTER_FIELDS = (("visual", "situation"), ("visual", "nearest_shot_type"), ("visual", "device"), ("visual", "gaze"),
                  ("performance", "format"), ("performance", "movement"), ("craft", "hook"),
                  ("craft", "camera_movement"), ("craft", "structure"), ("personality", "relationship"))

SYSTEM_PLAN = """You plan research for Manyworlds, a studio of openly AI-generated creators. Given a creator
hypothesis, name the hashtags (without #) on Instagram and TikTok where public creators who fit it post, and the
dimensions worth studying (formats, looks, performance). Logged-out public data only; adults only."""

SYSTEM_READ = """You annotate public social media posts for a creative pattern library. Describe the content, never the
person's private life: nothing about health, relationships, family, religion, politics, home address, or other people
in frame. Unknown is a valid answer: never invent. Keep observed (what is in the frames or words), mechanism (a
plausible reason it works) and speculation apart. Notes are short and in your own words, never quotes."""

SYSTEM_DISTILL = """You turn research on public creators into one pattern card for an AI creator studio: a pattern in
how photos or videos are made (situation, shot types, pose, camera, expression, styling, light, social energy,
movement; for video the move, edit grammar, pace and deliveries) and why it works. Use only the vocabulary names given
(shot types, moves, grammars, deliveries, archetypes, devices, grades); leave a field empty when nothing fits. It
describes a pattern, never a specific post or person to copy."""


class ScoutPlan(BaseModel):
    hashtags_instagram: list[str] = Field(default_factory=list, description="up to 6, without #")
    hashtags_tiktok: list[str] = Field(default_factory=list, description="up to 6, without #")
    dimensions: list[str] = Field(default_factory=list, description="what to study, a few words each")
    why: str = ""


class PostRead(BaseModel):
    post: int = Field(description="the post's number in this batch")
    annotation: ann.Annotation = Field(default_factory=ann.Annotation)


class ReadBatch(BaseModel):
    reads: list[PostRead] = Field(default_factory=list)


class CardDraft(BaseModel):
    key: str = Field(default="", description="snake_case name of the pattern, 2-4 words")
    medium: Literal["photo", "carousel", "video", "sequence"] = "photo"
    situation: str = ""
    shot_types: list[str] = Field(default_factory=list, description="names from the shot vocabulary, in order")
    pose: str = ""
    camera: str = ""
    expression: str = ""
    styling: str = ""
    lighting: str = ""
    social_energy: str = ""
    movement: str = ""
    mechanism: str = Field(default="", description="why it works, one or two sentences")
    audience_role: list[Literal["attraction", "intimacy", "humour", "aspiration", "utility"]] = Field(
        default_factory=list)
    best_fit: list[str] = Field(default_factory=list, description="archetype names")
    grade: str = ""
    move: str = ""
    grammar: str = ""
    device: str = ""
    deliveries: list[str] = Field(default_factory=list)
    pace_wps: list[float] = Field(default_factory=list, description="[low, high] words a second, video only")
    eye_contact: float | None = None
    cutaway_share: float | None = None
    hook: str = ""
    constraints: str = ""
    failure_modes: list[str] = Field(default_factory=list)
    guardrails: str = ""
    signature_ok: bool = False


def _cfg(policy) -> dict:
    return policy.get("intelligence", "agent", default={}) or {}


class Spend:
    """The run's spend so far (every ledger entry of this job) against its cap."""

    def __init__(self, conn, job_id, cap: float):
        self.conn, self.job_id, self.cap = conn, job_id, float(cap)
        self.capped = False

    def spent(self) -> float:
        return float(self.conn.execute("SELECT coalesce(sum(usd), 0) AS u FROM cost_entries WHERE job_id = %s",
                                       (self.job_id,)).fetchone()["u"])

    def allow(self, estimate: float) -> bool:
        ok = self.spent() + float(estimate) <= self.cap
        self.capped = self.capped or not ok
        return ok


def estimate(policy, *, creators: int | None = None, posts: int | None = None, platforms=None,
             hashtags_per_platform: int | None = None, plan: bool = True) -> dict:
    """What a pilot run (scout, annotate, distill) would cost at most, from policy prices: the dry run's answer."""
    cfg = _cfg(policy)
    est = cfg.get("estimate") or {}
    price = cfg.get("usd_per_item") or {}
    tag_price = cfg.get("usd_per_hashtag_item") or {}
    platforms = list(platforms or cfg.get("platforms") or ["instagram"])
    n = int(creators or cfg.get("creators", 40))
    k = int(posts or cfg.get("posts_per_creator", 30))
    tags = int(hashtags_per_platform if hashtags_per_platform is not None else cfg.get("hashtags_per_platform", 4))
    per_tag = int(cfg.get("items_per_hashtag", 50))
    lines = {}
    if plan:
        lines["plan_call"] = float(est.get("plan_call", 0))
    lines["candidate_pool"] = sum(tags * per_tag * float(tag_price.get(p, 0)) for p in platforms)
    share = {p: n // len(platforms) + (1 if i < n % len(platforms) else 0) for i, p in enumerate(platforms)}
    video = float(price.get("tiktok_video_download", 0)) if cfg.get("download_videos", True) else 0.0
    lines["creator_pulls"] = (share.get("instagram", 0) * (float(price.get("instagram_profile", 0))
                                                          + k * float(price.get("instagram_posts", 0)))
                              + share.get("tiktok", 0) * k * (float(price.get("tiktok_profile", 0)) + video))
    total_posts = n * k
    per_call = max(1, int((cfg.get("annotate") or {}).get("posts_per_call", 4)))
    lines["measure_video"] = total_posts * float(est.get("measure_video", 0))
    lines["annotate"] = -(-total_posts // per_call) * float(est.get("annotate_call", 0))
    lines["distill"] = int((cfg.get("distill") or {}).get("max_cards", 12)) * float(est.get("distill_call", 0))
    lines = {kk: round(v, 2) for kk, v in lines.items()}
    return {"creators": n, "posts_per_creator": k, "platforms": platforms, "posts": total_posts, "lines": lines,
            "total_usd": round(sum(lines.values()), 2), "cap_usd": float(cfg.get("max_usd_per_run", 0))}


# --- the hypothesis -----------------------------------------------------------------------------------------------
def hypothesis(conn, p: dict) -> dict:
    """{text, archetype, thesis_id, handles} from a payload's thesis id or text and named handles."""
    out = {"text": str(p.get("hypothesis") or p.get("question") or ""), "thesis_id": p.get("thesis_id"),
           "archetype": str(p.get("archetype") or ""), "handles": list(p.get("handles") or [])}
    if p.get("thesis_id"):
        t = conn.execute("SELECT id, summary, body FROM theses WHERE id::text = %s", (str(p["thesis_id"]),)).fetchone()
        if not t:
            raise NonRetryable(f"no thesis {p['thesis_id']}")
        body = t["body"] or {}
        out.update(text=out["text"] or str(body.get("statement") or t["summary"] or ""),
                   archetype=out["archetype"] or str(body.get("archetype") or ""),
                   handles=out["handles"] or list(body.get("references") or []))
    if not out["text"]:
        raise NonRetryable("intelligence payload needs a hypothesis (thesis_id or text)")
    return out


def _named(handles, platforms) -> list[dict]:
    """Named references: "instagram:handle", "@handle" (each platform) or {platform, handle}."""
    out = []
    for h in handles or []:
        if isinstance(h, dict) and h.get("handle"):
            out.append({"platform": h.get("platform") or platforms[0], "handle": str(h["handle"]).lstrip("@"),
                        "engagement": 0, "posts_seen": 0, "named": True})
            continue
        text = str(h).strip()
        plat, _, handle = text.partition(":") if ":" in text and not text.startswith("http") else ("", "", text)
        for pl in ([plat] if plat else platforms):
            out.append({"platform": pl, "handle": handle.lstrip("@"), "engagement": 0, "posts_seen": 0,
                        "named": True})
    return out


# --- the agent ----------------------------------------------------------------------------------------------------
class Intelligence:
    name = AGENT

    def fetch(self, actor: str, payload: dict) -> list[dict]:       # replaced in tests
        return run_actor(actor, payload)

    def download(self, url: str, dest: Path, max_bytes: int):      # replaced in tests
        return pulls.default_download(url, dest, max_bytes)

    def store(self, policy):                                         # replaced in tests
        return references.research_store(policy)

    def run(self, ctx: AgentContext) -> dict:
        p = ctx.job.payload or {}
        mode = p.get("mode") or "pilot"
        if mode not in MODES:
            raise NonRetryable(f"intelligence mode must be one of {MODES}")
        cfg = _cfg(ctx.policy)
        spend = Spend(ctx.conn, ctx.job.id, float(p.get("usd") or cfg.get("max_usd_per_run", 0)))
        steps = ["scout", "annotate", "distill"] if mode == "pilot" else [mode]
        out: dict = {"mode": mode, "cap_usd": spend.cap}
        scope = list(p.get("creator_ids") or [])
        for step in steps:
            res = remember(ctx, step, lambda step=step: self._step(ctx, step, p, spend, scope))
            out[step] = res
            if step == "scout":
                scope = [c["creator_id"] for c in res.get("creators") or [] if c.get("creator_id")] or scope
            if res.get("capped"):
                break
        out["spent_usd"] = round(spend.spent(), 4)
        out["capped"] = any((out.get(s) or {}).get("capped") for s in steps)
        return out

    def _step(self, ctx, step: str, p: dict, spend: Spend, scope: list) -> dict:
        res = {"scout": self.scout, "annotate": self.annotate, "distill": self.distill}[step](ctx, p, spend, scope)
        res["capped"] = bool(res.get("capped") or spend.capped)
        return {"kind": f"intelligence_{step}", "inputs": {k: v for k, v in p.items() if k != "reason"},
                "output": res, "rationale": f"influencer intelligence: {step}",
                "model": ctx.llm.model_for(str((_cfg(ctx.policy).get("roles") or {}).get(
                    {"scout": "plan"}.get(step, step), "")))}

    # --- scout ----------------------------------------------------------------------------------------------------
    def scout(self, ctx: AgentContext, p: dict, spend: Spend, scope: list) -> dict:
        conn, policy = ctx.conn, ctx.policy
        cfg = _cfg(policy)
        h = hypothesis(conn, p)
        platforms = list(p.get("platforms") or cfg.get("platforms") or ["instagram"])
        hashtags = {pl: list((p.get("hashtags") or {}).get(pl) or []) for pl in platforms} \
            if isinstance(p.get("hashtags"), dict) else {pl: list(p.get("hashtags") or []) for pl in platforms}
        plan = {}
        if not any(hashtags.values()) and spend.allow(float((cfg.get("estimate") or {}).get("plan_call", 0))):
            role = (cfg.get("roles") or {}).get("plan", "iia_curate")
            got = ctx.llm.structured(role, system=SYSTEM_PLAN, content=(
                f"Creator hypothesis: {h['text']}\nArchetype: {h['archetype'] or 'unknown'}\n"
                f"Named reference creators (study them; find comparable ones): {h['handles']}\n"
                f"Platforms: {platforms}. Up to {int(cfg.get('hashtags_per_platform', 4))} hashtags each."),
                schema=ScoutPlan, effort=str((cfg.get("effort") or {}).get("plan", "medium")))
            plan = got.model_dump(mode="json") if hasattr(got, "model_dump") else {}
            k = int(cfg.get("hashtags_per_platform", 4))
            hashtags = {pl: [t.lstrip("#").strip() for t in (plan.get(f"hashtags_{pl}") or []) if t.strip()][:k]
                        for pl in platforms}
        per_tag = int(cfg.get("items_per_hashtag", 50))
        tag_price = cfg.get("usd_per_hashtag_item") or {}
        pool_cost = sum(len(t) * per_tag * float(tag_price.get(pl, 0)) for pl, t in hashtags.items())
        pool = []
        if any(hashtags.values()) and spend.allow(pool_cost):
            pool = pulls.candidates(conn, policy, self.fetch, hashtags, job_id=ctx.job.id)
        sample = pulls.select(policy, pool, named=_named(h["handles"], platforms), n=p.get("creators"))
        k = int(p.get("posts") or cfg.get("posts_per_creator", 30))
        price = cfg.get("usd_per_item") or {}
        store = self.store(policy)
        done, skipped = [], []
        with tempfile.TemporaryDirectory(prefix="mw-iia-") as tmp:
            for c in sample:
                cost = (float(price.get("instagram_profile", 0)) + k * float(price.get("instagram_posts", 0))
                        if c["platform"] == "instagram" else
                        k * (float(price.get("tiktok_profile", 0)) + float(price.get("tiktok_video_download", 0))))
                if not spend.allow(cost):
                    break
                try:
                    r = pulls.pull_creator(conn, policy, self.fetch, {**c, "why": c.get("role")}, store=store,
                                           download=self.download, workdir=Path(tmp), job_id=ctx.job.id, posts=k)
                except Exception as e:  # noqa: BLE001 - one creator's failed pull is recorded, the rest go on
                    skipped.append({"handle": c["handle"], "error": f"{type(e).__name__}: {str(e)[:200]}"})
                    continue
                (skipped if r.get("skipped") else done).append({**r, "role": c.get("role"),
                                                               "platform": c["platform"]})
        return {"hypothesis": h["text"][:300], "hashtags": hashtags, "plan": plan, "pool": len(pool),
                "creators": done, "skipped": skipped, "storage": "research" if references._research_bucket()
                else ("main_bucket:" + references.PREFIX if store is not None else "none")}

    # --- annotate -------------------------------------------------------------------------------------------------
    def annotate(self, ctx: AgentContext, p: dict, spend: Spend, scope: list) -> dict:
        conn, policy = ctx.conn, ctx.policy
        cfg = _cfg(policy)
        acfg = cfg.get("annotate") or {}
        est = cfg.get("estimate") or {}
        collected = video_measure.collect(conn, policy)
        posts = [dict(r) for r in conn.execute(
            """SELECT p.* FROM reference_posts p WHERE (%s::uuid[] IS NULL OR p.creator_id = ANY(%s::uuid[]))
               AND NOT EXISTS (SELECT 1 FROM reference_annotations a WHERE a.post_id = p.id AND a.schema = %s)
               ORDER BY p.creator_id, p.posted_at DESC NULLS LAST LIMIT %s""",
            (scope or None, scope or None, ann.ANNOTATION_SCHEMA, int(p.get("max_posts") or 5000))).fetchall()]
        queued = 0
        for post in posts:
            if post["media_type"] == "video" and any(m.get("kind") == "video" for m in post["media"] or []) \
                    and not self._measured(conn, post["id"]) and spend.allow(float(est.get("measure_video", 0))):
                video_measure.enqueue(conn, policy, post, batch=ctx.job.batch)
                queued += 1
        store = self.store(policy)
        role = (cfg.get("roles") or {}).get("annotate", "iia_annotate")
        per_call = max(1, int(acfg.get("posts_per_call", 4)))
        written = 0
        for i in range(0, len(posts), per_call):
            if not spend.allow(float(est.get("annotate_call", 0))):
                break
            chunk = posts[i:i + per_call]
            content = self._read_content(conn, policy, store, chunk, acfg)
            got = ctx.llm.structured(role, system=SYSTEM_READ, content=content, schema=ReadBatch,
                                     effort=str((cfg.get("effort") or {}).get("annotate", "low")))
            reads = {r.post: r.annotation for r in (getattr(got, "reads", None) or [])
                     if isinstance(getattr(r, "post", None), int)}
            for j, post in enumerate(chunk):
                a = reads.get(j + 1) or ann.Annotation()
                self._store_read(conn, ctx, post, a, ctx.llm.model_for(role))
                written += 1
        return {"posts": len(posts), "annotated": written, "measure_jobs": queued, "measures_collected": collected}

    def _measured(self, conn, post_id) -> dict | None:
        row = conn.execute("SELECT measured FROM reference_annotations WHERE post_id = %s AND schema LIKE 'measure_%%' "
                           "ORDER BY updated_at DESC LIMIT 1", (post_id,)).fetchone()
        return row["measured"] if row else None

    def _read_content(self, conn, policy, store, chunk: list[dict], acfg: dict) -> list[dict]:
        blocks: list[dict] = [{"type": "text", "text": (
            f"Annotate each of these {len(chunk)} public posts (numbered from 1) into the schema. Use the shot "
            f"vocabulary for nearest_shot_type ({', '.join(aesthetics.names(policy))}) or new:<words>; capabilities "
            f"from: {', '.join((policy.get('capabilities', default={}) or {}).keys())}.")}]
        cap = int(acfg.get("image_max_bytes", 400000))
        chars = int(acfg.get("caption_chars", 600))
        with tempfile.TemporaryDirectory(prefix="mw-iia-read-") as tmp:
            for j, post in enumerate(chunk, 1):
                m = self._measured(conn, post["id"]) or {}
                view = {k: m[k] for k in ("duration_s", "hard_cuts", "shots", "first_hard_cut_s", "speech")
                        if k in m}
                blocks.append({"type": "text", "text": (
                    f"Post {j}: {post['media_type']}; caption: {(post.get('caption') or '')[:chars]!r}; "
                    f"measured: {json.dumps(view, default=str)}")})
                media = post.get("media") or []
                pics = [x for x in media if x.get("kind") == "grid"] or [x for x in media if x.get("kind") == "image"]
                for x in pics[:int(_cfg(policy).get("images_per_post", 3))]:
                    if store is None or int(x.get("bytes") or 0) > cap:
                        continue
                    try:
                        local = store.get_file(x["uri"], Path(tmp) / f"{post['id']}-{Path(x['uri']).name}")
                        data = Path(local).read_bytes()
                    except Exception:  # noqa: BLE001 - a missing copy: the read goes on with what remains
                        continue
                    if len(data) <= cap:
                        blocks.append(image_block(data, "image/jpeg"))
        return blocks

    def _store_read(self, conn, ctx, post: dict, a: ann.Annotation, model: str) -> None:
        m = post.get("metrics") or {}
        ev = ann.Evidence(views=m.get("views"), likes=m.get("likes"), comments=m.get("comments"),
                                 shares=m.get("shares"), saves=m.get("saves"),
                                 over_own_median=m.get("over_own_median"),
                                 missing=[k for k in ("views", "saves", "shares") if m.get(k) is None])
        notes = lambda xs: [{"text": ann.clip_note(n.text), "audience_role": n.audience_role}  # noqa: E731
                            for n in xs]
        conn.execute(
            """INSERT INTO reference_annotations (post_id, schema, model, measured, observed, mechanism, speculation,
                                                  evidence, confidence)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (post_id, schema) DO NOTHING""",
            (post["id"], ann.ANNOTATION_SCHEMA, model, Jsonb(self._measured(conn, post["id"]) or {}),
             Jsonb(a.observed.model_dump(mode="json")), Jsonb(notes(a.mechanism)), Jsonb(notes(a.speculation)),
             Jsonb(ev.model_dump(mode="json")), float(a.confidence)))

    # --- distill --------------------------------------------------------------------------------------------------
    def distill(self, ctx: AgentContext, p: dict, spend: Spend, scope: list) -> dict:
        conn, policy = ctx.conn, ctx.policy
        cfg = _cfg(policy)
        dcfg = cfg.get("distill") or {}
        rows = [dict(r) for r in conn.execute(
            """SELECT a.post_id, a.observed, a.mechanism, a.evidence, a.measured, p.media_type, p.url, p.creator_id
               FROM reference_annotations a JOIN reference_posts p ON p.id = a.post_id
               WHERE a.schema = %s AND (%s::uuid[] IS NULL OR p.creator_id = ANY(%s::uuid[]))""",
            (ann.ANNOTATION_SCHEMA, scope or None, scope or None)).fetchall()]
        clusters = cluster(rows, dcfg)
        role = (cfg.get("roles") or {}).get("distill", "iia_distill")
        vocab = self._vocabulary(policy)
        made, skipped = [], []
        for c in clusters[:int(dcfg.get("max_cards", 12))]:
            done = conn.execute("SELECT key, version FROM pattern_cards WHERE evidence->>'job_id' = %s "
                                "AND evidence->>'cluster' = %s", (str(ctx.job.id), c["name"])).fetchone()
            if done:
                made.append({"key": done["key"], "version": done["version"], "cluster": c["name"], "again": True})
                continue
            if not spend.allow(float((cfg.get("estimate") or {}).get("distill_call", 0))):
                break
            examples = [{"medium": r["media_type"], "over_own_median": (r["evidence"] or {}).get("over_own_median"),
                         "observed": r["observed"], "mechanism": [n.get("text") for n in r["mechanism"] or []],
                         "measured": {k: (r["measured"] or {}).get(k) for k in ("duration_s", "shots", "speech")
                                      if (r["measured"] or {}).get(k) is not None}}
                        for r in c["high"][:int(dcfg.get("examples", 6))]]
            got = ctx.llm.structured(role, system=SYSTEM_DISTILL, content=(
                f"Cluster {c['name']}: {c['n']} posts from {c['n_creators']} creators; "
                f"{len(c['high'])} high and {len(c['low'])} low traction against each creator's own median.\n"
                f"What high-traction posts share more than low ones (attribute: high share, low share): "
                f"{json.dumps(c['lift'])}\nHigh-traction examples: {json.dumps(examples, default=str)}\n"
                f"Vocabulary (use only these names): {json.dumps(vocab)}\n"
                f"Existing cards (reuse a key to update one): {sorted(k['key'] for k in patterns.cards(conn, policy))}"),
                schema=CardDraft, effort=str((cfg.get("effort") or {}).get("distill", "medium")))
            card, problems = to_card(policy, got, vocab)
            if card is None:
                skipped.append({"cluster": c["name"], "problems": problems[:5]})
                continue
            made.append({**store_card(conn, card, c, ctx.job.id), "cluster": c["name"],
                         "dropped_names": problems[:5]})
        return {"annotations": len(rows), "clusters": len(clusters), "cards": made, "skipped": skipped}

    def _vocabulary(self, policy) -> dict:
        names = patterns._names(policy)
        return {"shot_types": sorted(names["shot_types"]), "moves": sorted(names["moves"]),
                "grammars": sorted(names["grammars"]), "deliveries": sorted(names["deliveries"]),
                "archetypes": sorted(archetypes.names(policy)), "devices": sorted(names["devices"]),
                "grades": sorted(names["grades"]), "motion_moves": sorted(names["motion_moves"]),
                "hooks": sorted(patterns._hook_kinds())}


def _get(observed: dict, section: str, field: str):
    v = ((observed or {}).get(section) or {}).get(field)
    return v if v not in (None, "", "unknown", []) else None


def cluster(rows: list[dict], dcfg: dict) -> list[dict]:
    """Annotated posts grouped by medium x (video format, else situation); for each group with enough posts and
    creators, the coded attributes that high-traction posts share more than low ones (over each creator's own
    median: >= high, <= low), sorted by size x spread."""
    hi, lo = float(dcfg.get("high", 1.5)), float(dcfg.get("low", 0.8))
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        o = r["observed"] or {}
        sub = _get(o, "performance", "format") if r["media_type"] == "video" else None
        groups[f"{r['media_type']}:{sub or _get(o, 'visual', 'situation') or 'other'}"].append(r)
    out = []
    for name, rs in groups.items():
        creators = {str(r["creator_id"]) for r in rs}
        if len(rs) < int(dcfg.get("min_posts", 6)) or len(creators) < int(dcfg.get("min_creators", 1)):
            continue
        ratio = lambda r: (r["evidence"] or {}).get("over_own_median")   # noqa: E731
        high = sorted([r for r in rs if ratio(r) is not None and ratio(r) >= hi], key=lambda r: -ratio(r))
        low = [r for r in rs if ratio(r) is not None and ratio(r) <= lo]
        if not high:
            continue

        def shares(group):
            cnt: Counter = Counter()
            for r in group:
                for section, field in CLUSTER_FIELDS:
                    v = _get(r["observed"], section, field)
                    if v is not None:
                        cnt[f"{field}={v}"] += 1
            return {k: v / max(len(group), 1) for k, v in cnt.items()}
        sh, sl = shares(high), shares(low)
        lift = sorted(((k, round(v, 2), round(sl.get(k, 0.0), 2)) for k, v in sh.items()),
                      key=lambda t: -(t[1] - t[2]))[:10]
        spread = (lift[0][1] - lift[0][2]) if lift else 0.0
        out.append({"name": name, "n": len(rs), "n_creators": len(creators), "high": high, "low": low,
                    "lift": [{"attribute": k, "high": a, "low": b} for k, a, b in lift], "_rank": len(rs) * spread})
    return sorted(out, key=lambda c: -c["_rank"])


_KEY = re.compile(r"[^a-z0-9_]+")


def to_card(policy, d: CardDraft, vocab: dict) -> tuple[dict | None, list[str]]:
    """A drafted card in the library's shape, with names the pipeline doesn't know dropped (reported), then validated
    (manyworlds.brain.patterns.validate). None when nothing renderable is left."""
    dropped = []
    key = _KEY.sub("_", str(getattr(d, "key", "") or "").lower()).strip("_")[:48]
    shot_types = [s for s in (getattr(d, "shot_types", None) or []) if s in vocab["shot_types"]]
    dropped += [f"shot type {s}" for s in (getattr(d, "shot_types", None) or []) if s not in vocab["shot_types"]]
    recipe: dict = {}
    for field, names in (("grade", "grades"), ("move", "moves"), ("grammar", "grammars"), ("device", "devices")):
        v = str(getattr(d, field, "") or "")
        if v and v in vocab[names]:
            recipe[field] = v
        elif v:
            dropped.append(f"{field} {v}")
    deliveries = [x for x in (getattr(d, "deliveries", None) or []) if x in vocab["deliveries"]]
    if deliveries:
        recipe["deliveries"] = deliveries
    pace = [float(x) for x in (getattr(d, "pace_wps", None) or [])][:2]
    if len(pace) == 2:
        recipe["pace_wps"] = sorted(pace)
    if getattr(d, "eye_contact", None) is not None:
        recipe["eye_contact"] = max(0.0, min(1.0, float(d.eye_contact)))
    edit = {}
    if getattr(d, "cutaway_share", None) is not None:
        edit["cutaway_share"] = max(0.0, min(1.0, float(d.cutaway_share)))
    if getattr(d, "hook", "") and d.hook in vocab["hooks"]:
        edit["hook"] = d.hook
    if edit:
        recipe["edit"] = edit
    card = {"medium": getattr(d, "medium", "photo"), "status": "draft", "signature_ok": bool(d.signature_ok),
            "situation": str(d.situation or ""),
            "dims": {"shot_types": shot_types, **{k: str(getattr(d, k, "") or "unknown")
                                                  for k in patterns.DIMENSIONS}},
            "mechanism": str(d.mechanism or ""), "audience_role": list(d.audience_role or []),
            "best_fit": [a for a in (d.best_fit or []) if a in vocab["archetypes"]], "recipe": recipe,
            "constraints": str(d.constraints or ""), "failure_modes": list(d.failure_modes or [])[:6],
            "guardrails": str(d.guardrails or "No near-duplicates of specific posts; never impersonate anyone.")}
    if not key:
        return None, dropped + ["no key"]
    problems = patterns.validate(policy, {"cards": {key: card}})["problems"]
    if problems:
        return None, dropped + problems
    card["key"] = key
    return card, dropped


def store_card(conn, card: dict, c: dict, job_id) -> dict:
    """A new card, or a new version of the card with that key (the old version marked superseded), status draft,
    with its sources (the cluster's high-traction posts) and evidence."""
    key = card["key"]
    src = c["high"] or []
    sources = [{"post_id": str(r["post_id"]), "creator_id": str(r["creator_id"]), "url": r.get("url")} for r in src]
    evidence = {"n_posts": len(sources), "n_creators": len({s["creator_id"] for s in sources}),
                "cluster": c["name"], "cluster_posts": c["n"], "high": len(c["high"]), "low": len(c["low"]),
                "lift": c["lift"][:5], "job_id": str(job_id)}
    body = {k: v for k, v in card.items() if k not in ("key", "medium", "status", "signature_ok", "sources")}
    last = conn.execute("SELECT id, version FROM pattern_cards WHERE key = %s ORDER BY version DESC LIMIT 1",
                        (key,)).fetchone()
    version = (last["version"] + 1) if last else 1
    row = conn.execute(
        """INSERT INTO pattern_cards (key, version, status, medium, signature_ok, body, sources, evidence)
           VALUES (%s, %s, 'draft', %s, %s, %s, %s, %s) RETURNING id""",
        (key, version, card["medium"], card["signature_ok"], Jsonb(body), Jsonb(sources), Jsonb(evidence))).fetchone()
    if last:
        conn.execute("UPDATE pattern_cards SET superseded_by = %s, updated_at = now() WHERE id = %s",
                     (row["id"], last["id"]))
    return {"key": key, "version": version, "id": str(row["id"]), "n_posts": len(sources)}


register(Intelligence())
