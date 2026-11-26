"""The creative evaluator: scores ideas (idea rubric, from the plan) and finished images
(execution rubric, image + caption) independently of whoever planned them.

It sees the creator's profile summary and the item, never the showrunner's reasoning. Each item is
scored `brain.evaluation.samples` times by the `evaluator` role; a deterministic share
(`check_rate`) is also scored by `evaluator_check`, to measure agreement. Each item's scoring is
one remembered step, so a rerun of the job returns the stored evaluation instead of paying again.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path

from psycopg.types.json import Jsonb

from ... import storage
import logging

from ...llm import Refused, image_block
from ...creator import profile
from ...review import common as review_common
from .. import evaluation, revise, rubrics
from ...worker.runner import NonRetryable
from ..agent import AgentContext, record_decision, register

# What the evaluator may know about the creator: who they are, who follows them and why, what they
# post about. From the current profile (versioned); the bible only for creators without one.
log = logging.getLogger("manyworlds.brain.evaluator")
_BIBLE_FIELDS = ("display_name", "one_line", "caption_voice", "voice_notes", "personality", "interests", "never")


def creator_summary(conn, creator_id) -> tuple[int | None, str]:
    c = conn.execute("SELECT handle, bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
    current = profile.current(conn, creator_id)
    if current:
        version, body = current
        content = body.get("content") or {}
        facts = {"persona": body.get("persona"), "audience": body.get("audience"),
                 "pillars": content.get("pillars"), "video_series": content.get("video_series")}
    else:
        version, bible = None, c["bible"] or {}
        facts = {k: bible[k] for k in _BIBLE_FIELDS if bible.get(k)}
    facts = {k: v for k, v in facts.items() if v}
    return version, evaluation.CREATOR_TEMPLATE.format(handle=c["handle"], facts=json.dumps(facts, ensure_ascii=False))


def near_floor(policy, rubric_name: str):
    """Extra samples for an idea whose score sits within `resample.margin` of a feed-QA floor (the scores that
    drop ideas), so one noisy sample never decides a drop; ideas far from every floor are scored once."""
    cfg = policy.get("brain", "evaluation", "resample", default={}) or {}
    floors = (policy.get("brain", "qa", "min_scores", default={}) or {}) if rubric_name == "idea" else {}
    if not cfg or not floors:
        return None
    margin, extra = float(cfg.get("margin", 1)), int(cfg.get("extra", 2))

    def more(medians: dict) -> int:
        near = any(medians.get(d) is not None and abs(float(medians[d]) - float(f)) <= margin
                   for d, f in floors.items())
        return extra if near else 0
    return more


def score_and_store(conn, policy, llm, rubric_name: str, item_kind: str, content: list, *, creator_id, fmt,
                    idea_id=None, asset_id=None, item_key, null_dimensions=(), on_stored=None) -> list[str]:
    """Score an item with the evaluator (and the strong model when sampled) and store the rows.
    `on_stored(ids)` runs in the same transaction as the inserts, so what remembers them (the job's
    decision) commits with them or not at all: a crash in between can never leave rows a rerun repeats."""
    cfg = policy.get("brain", "evaluation", default={}) or {}
    r = rubrics.load(rubric_name)
    rubrics.register(conn, r)
    roles = ["evaluator"] + (["evaluator_check"] if evaluation.checked(item_key, float(cfg.get("check_rate", 0)))
                             else [])
    scored = []
    for role in roles:
        samples = int(cfg.get("check_samples", 1) if role == "evaluator_check" else cfg.get("samples", 3))
        more = None if role == "evaluator_check" else near_floor(policy, rubric_name)
        try:
            scored.append((role, evaluation.score(llm, policy, r, content, role=role, samples=samples,
                                                  creator_id=creator_id, more=more)))
        except Refused:
            if role != "evaluator_check":
                raise
            # The strong-model check only measures agreement; a (mistaken) refusal there never blocks the idea.
            log.warning("evaluator_check refused item %s; volume evaluation kept", item_key)
    for _, s in scored:   # dimensions the evaluator had no basis for are null, never a guess
        for d in null_dimensions:
            s.medians[d], s.spreads[d] = None, None
    profile_version, _ = creator_summary(conn, creator_id)
    ids = []
    with conn.transaction():
        for role, s in scored:
            ids.append(str(conn.execute(
                """INSERT INTO evaluations (item_kind, idea_id, asset_id, creator_id, rubric, rubric_version, role,
                       evaluator_model, prompt_version, scores, spreads, rationale, samples, profile_version, format)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (item_kind, idea_id, asset_id, creator_id, r.rubric, r.version, role, s.model,
                 evaluation.prompt_version(policy), Jsonb(s.medians), Jsonb(s.spreads),
                 Jsonb({**s.rationale, "_notes": s.notes}), s.samples, profile_version, fmt)).fetchone()["id"]))
        if on_stored is not None:
            on_stored(ids)
    return ids


def _remembered(ctx: AgentContext, step: str, compute) -> dict:
    """`remember` for a step whose output is stored rows: `compute(record)` stores them and calls
    `record(decision)` inside the same transaction, so the rows and the decision commit together."""
    row = ctx.conn.execute("SELECT output FROM decisions WHERE job_id = %s AND step = %s",
                           (ctx.job.id, step)).fetchone()
    if row:
        return row["output"]

    def record(d: dict) -> None:
        record_decision(ctx, d["kind"], step=step, inputs=d.get("inputs", {}), output=d["output"],
                        rationale=d.get("rationale", ""), creator_id=d.get("creator_id"))
    compute(record)
    return ctx.conn.execute("SELECT output FROM decisions WHERE job_id = %s AND step = %s",
                            (ctx.job.id, step)).fetchone()["output"]


# What the execution rubric can judge: an image in a type the model reads. Anything else (a video, a
# keyframe in another format) is a permanent failure, never a guess at the media type.
_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def idea_content(conn, idea_id) -> tuple:
    """(creator_id, format, content) for scoring an idea: the creator summary and the plan, never
    the planner's reasoning."""
    idea = conn.execute("SELECT creator_id, concept, format, body FROM ideas WHERE id = %s", (idea_id,)).fetchone()
    if idea is None:   # retrying cannot make the row appear
        raise NonRetryable(f"evaluator: no idea {idea_id}")
    # Never what the planner thought (reasoning) or how it labelled the idea (exploratory, exploration), which
    # would bias the scores; scheduling bookkeeping (day, platform, canon_key) is irrelevant to them, and a
    # revision is scored on its plan, never on its history (manyworlds.brain.revise.BOOKKEEPING).
    plan = {k: v for k, v in (idea["body"] or {}).items()
            if k not in ("reasoning", "exploratory", "exploration", "exploration_off_target", "day", "platform",
                         "canon_key", *revise.BOOKKEEPING)}
    _, who = creator_summary(conn, idea["creator_id"])
    text = evaluation.IDEA_TEMPLATE.format(format=idea["format"], concept=idea["concept"],
                                           plan=json.dumps(plan, ensure_ascii=False))
    return idea["creator_id"], idea["format"], [who, text]


def evaluate_idea(ctx: AgentContext, idea_id) -> uuid.UUID:
    def compute(record) -> None:
        creator_id, fmt, content = idea_content(ctx.conn, idea_id)
        score_and_store(ctx.conn, ctx.policy, ctx.llm, "idea", "idea", content, creator_id=creator_id,
                        fmt=fmt, idea_id=idea_id, item_key=idea_id, on_stored=lambda ids: record(
                            {"kind": "evaluation", "inputs": {"idea_id": str(idea_id)},
                             "output": {"evaluation_ids": ids}, "creator_id": creator_id,
                             "rationale": f"scored idea on the idea rubric ({len(ids)} evaluator(s))"}))
    return uuid.UUID(_remembered(ctx, f"evaluate:idea:{idea_id}", compute)["evaluation_ids"][0])


def asset_row(conn, asset_id) -> dict:
    """The asset to score with its post's caption, format and scenes; a permanent failure when there is
    nothing the execution rubric can judge (no such asset, or not an image in a readable type)."""
    a = conn.execute(
        """SELECT a.uri, a.creator_id, a.kind, p.caption, p.format, p.body
           FROM assets a LEFT JOIN jobs j ON j.id = a.job_id LEFT JOIN posts p ON p.id = j.post_id
           WHERE a.id = %s""", (asset_id,)).fetchone()
    if a is None:
        raise NonRetryable(f"evaluator: no asset {asset_id}")
    if a["kind"] != "image":
        raise NonRetryable(f"evaluator: asset {asset_id} is a {a['kind']}, the execution rubric scores images")
    if Path(a["uri"]).suffix.lower() not in _MEDIA_TYPES:
        raise NonRetryable(f"evaluator: asset {asset_id} is not an image type the evaluator reads ({a['uri']})")
    return a


def evaluate_asset(ctx: AgentContext, asset_id, *, store=None) -> uuid.UUID:
    def compute(record) -> None:
        a = asset_row(ctx.conn, asset_id)
        st = store or storage.from_url()
        creator = ctx.conn.execute("SELECT id, anchors FROM creators WHERE id = %s", (a["creator_id"],)).fetchone()
        n_refs = int(ctx.policy.get("brain", "evaluation", "reference_images", default=2))
        suffix = Path(a["uri"]).suffix.lower()
        with tempfile.TemporaryDirectory() as tmp:
            data = st.get_file(a["uri"], Path(tmp) / f"asset{suffix}").read_bytes()
            try:
                refs = review_common.reference_images(creator, st, Path(tmp), n_refs)
            except ValueError:   # no reference photos: identity can't be judged
                refs = []
        _, who = creator_summary(ctx.conn, a["creator_id"])
        scenes = (a["body"] or {}).get("scenes") or []
        ref_text, scene_text, image_text, caption_text = evaluation.ASSET_TEMPLATES
        content = [who]
        if refs:
            content += [ref_text, *[image_block(b) for b in refs]]
        content += [scene_text.format(scenes=" | ".join(scenes) or "(none recorded)"), image_text,
                    image_block(data, _MEDIA_TYPES[suffix]), caption_text.format(caption=a["caption"] or "(none)")]
        score_and_store(ctx.conn, ctx.policy, ctx.llm, "execution", "asset", content,
                        creator_id=a["creator_id"], fmt=a["format"], asset_id=asset_id, item_key=asset_id,
                        null_dimensions=() if refs else ("identity",), on_stored=lambda ids: record(
                            {"kind": "evaluation", "inputs": {"asset_id": str(asset_id)},
                             "output": {"evaluation_ids": ids}, "creator_id": a["creator_id"],
                             "rationale": f"scored image on the execution rubric ({len(ids)} evaluator(s))"}))
    return uuid.UUID(_remembered(ctx, f"evaluate:asset:{asset_id}", compute)["evaluation_ids"][0])


def _uuid(value, what: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        raise NonRetryable(f"evaluator: {what} {value!r} is not a uuid") from None


class Evaluator:
    name = "evaluator"

    def run(self, ctx: AgentContext) -> dict:
        p = ctx.job.payload or {}
        if p.get("idea_id"):
            return {"evaluation_id": str(evaluate_idea(ctx, _uuid(p["idea_id"], "idea_id")))}
        if p.get("asset_id"):
            return {"evaluation_id": str(evaluate_asset(ctx, _uuid(p["asset_id"], "asset_id")))}
        raise NonRetryable("evaluator payload needs idea_id or asset_id")


register(Evaluator())
