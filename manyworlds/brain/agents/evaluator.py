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
from ...llm import image_block
from ...creator import profile
from ...review import common as review_common
from .. import evaluation, rubrics
from ..agent import AgentContext, register, remember

# What the evaluator may know about the creator: who they are, who follows them and why, what they
# post about. From the current profile (versioned); the bible only for creators without one.
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


def score_and_store(conn, policy, llm, rubric_name: str, item_kind: str, content: list, *, creator_id, fmt,
                    idea_id=None, asset_id=None, item_key, null_dimensions=()) -> list[str]:
    """Score an item with the evaluator (and the strong model when sampled) and store the rows."""
    cfg = policy.get("brain", "evaluation", default={}) or {}
    r = rubrics.load(rubric_name)
    rubrics.register(conn, r)
    roles = ["evaluator"] + (["evaluator_check"] if evaluation.checked(item_key, float(cfg.get("check_rate", 0)))
                             else [])
    scored = [(role, evaluation.score(llm, policy, r, content, role=role, samples=int(cfg.get("samples", 3)),
                                      creator_id=creator_id)) for role in roles]
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
    return ids


def idea_content(conn, idea_id) -> tuple:
    """(creator_id, format, content) for scoring an idea: the creator summary and the plan, never
    the planner's reasoning."""
    idea = conn.execute("SELECT creator_id, concept, format, body FROM ideas WHERE id = %s", (idea_id,)).fetchone()
    plan = {k: v for k, v in (idea["body"] or {}).items() if k != "reasoning"}
    _, who = creator_summary(conn, idea["creator_id"])
    text = evaluation.IDEA_TEMPLATE.format(format=idea["format"], concept=idea["concept"],
                                           plan=json.dumps(plan, ensure_ascii=False))
    return idea["creator_id"], idea["format"], [who, text]


def evaluate_idea(ctx: AgentContext, idea_id) -> uuid.UUID:
    def compute() -> dict:
        creator_id, fmt, content = idea_content(ctx.conn, idea_id)
        ids = score_and_store(ctx.conn, ctx.policy, ctx.llm, "idea", "idea", content, creator_id=creator_id,
                              fmt=fmt, idea_id=idea_id, item_key=idea_id)
        return {"kind": "evaluation", "inputs": {"idea_id": str(idea_id)}, "output": {"evaluation_ids": ids},
                "rationale": f"scored idea on the idea rubric ({len(ids)} evaluator(s))", "creator_id": creator_id}
    return uuid.UUID(remember(ctx, f"evaluate:idea:{idea_id}", compute)["evaluation_ids"][0])


def evaluate_asset(ctx: AgentContext, asset_id, *, store=None) -> uuid.UUID:
    def compute() -> dict:
        a = ctx.conn.execute(
            """SELECT a.uri, a.creator_id, a.kind, p.caption, p.format, p.body
               FROM assets a JOIN jobs j ON j.id = a.job_id LEFT JOIN posts p ON p.id = j.post_id
               WHERE a.id = %s""", (asset_id,)).fetchone()
        st = store or storage.from_url()
        creator = ctx.conn.execute("SELECT id, anchors FROM creators WHERE id = %s", (a["creator_id"],)).fetchone()
        n_refs = int(ctx.policy.get("brain", "evaluation", "reference_images", default=2))
        with tempfile.TemporaryDirectory() as tmp:
            data = st.get_file(a["uri"], Path(tmp) / f"asset{Path(a['uri']).suffix or '.png'}").read_bytes()
            try:
                refs = review_common.reference_images(creator, st, Path(tmp), n_refs)
            except ValueError:   # no reference photos: identity can't be judged
                refs = []
        media = "image/jpeg" if a["uri"].lower().endswith((".jpg", ".jpeg")) else "image/png"
        _, who = creator_summary(ctx.conn, a["creator_id"])
        scenes = (a["body"] or {}).get("scenes") or []
        ref_text, scene_text, image_text, caption_text = evaluation.ASSET_TEMPLATES
        content = [who]
        if refs:
            content += [ref_text, *[image_block(b) for b in refs]]
        content += [scene_text.format(scenes=" | ".join(scenes) or "(none recorded)"), image_text,
                    image_block(data, media), caption_text.format(caption=a["caption"] or "(none)")]
        ids = score_and_store(ctx.conn, ctx.policy, ctx.llm, "execution", "asset", content,
                              creator_id=a["creator_id"], fmt=a["format"], asset_id=asset_id, item_key=asset_id,
                              null_dimensions=() if refs else ("identity",))
        return {"kind": "evaluation", "inputs": {"asset_id": str(asset_id)}, "output": {"evaluation_ids": ids},
                "rationale": f"scored image on the execution rubric ({len(ids)} evaluator(s))",
                "creator_id": a["creator_id"]}
    return uuid.UUID(remember(ctx, f"evaluate:asset:{asset_id}", compute)["evaluation_ids"][0])


class Evaluator:
    name = "evaluator"

    def run(self, ctx: AgentContext) -> dict:
        p = ctx.job.payload or {}
        if p.get("idea_id"):
            return {"evaluation_id": str(evaluate_idea(ctx, p["idea_id"]))}
        if p.get("asset_id"):
            return {"evaluation_id": str(evaluate_asset(ctx, p["asset_id"]))}
        from ...worker.runner import NonRetryable
        raise NonRetryable("evaluator payload needs idea_id or asset_id")


register(Evaluator())
