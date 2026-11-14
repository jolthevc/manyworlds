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
from .. import evaluation, rubrics
from ..agent import AgentContext, register, remember

# Creator facts the evaluator may see (from the rendered bible).
_CREATOR_FIELDS = ("display_name", "one_line", "niche", "caption_voice", "voice_notes", "pillars", "audience")


def creator_summary(conn, creator_id) -> tuple[int | None, str]:
    c = conn.execute("SELECT handle, bible, profile_version FROM creators WHERE id = %s", (creator_id,)).fetchone()
    bible = c["bible"] or {}
    facts = {k: bible[k] for k in _CREATOR_FIELDS if bible.get(k)}
    return c["profile_version"], f"THE CREATOR (@{c['handle']}): {json.dumps(facts, ensure_ascii=False)}"


def score_and_store(conn, policy, llm, rubric_name: str, item_kind: str, content: list, *, creator_id, fmt,
                    idea_id=None, asset_id=None, item_key) -> list[str]:
    """Score an item with the evaluator (and the strong model when sampled) and store the rows."""
    cfg = policy.get("brain", "evaluation", default={}) or {}
    r = rubrics.load(rubric_name)
    rubrics.register(conn, r)
    roles = ["evaluator"] + (["evaluator_check"] if evaluation.checked(item_key, float(cfg.get("check_rate", 0)))
                             else [])
    scored = [evaluation.score(llm, policy, r, content, role=role, samples=int(cfg.get("samples", 3)),
                               creator_id=creator_id) for role in roles]
    profile_version, _ = creator_summary(conn, creator_id)
    ids = []
    with conn.transaction():
        for s in scored:
            ids.append(str(conn.execute(
                """INSERT INTO evaluations (item_kind, idea_id, asset_id, creator_id, rubric, rubric_version,
                       evaluator_model, prompt_version, scores, spreads, rationale, samples, profile_version, format)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (item_kind, idea_id, asset_id, creator_id, r.rubric, r.version, s.model,
                 cfg.get("prompt_version", "eval-v1"), Jsonb(s.medians), Jsonb(s.spreads),
                 Jsonb({**s.rationale, "_notes": s.notes}), s.samples, profile_version, fmt)).fetchone()["id"]))
    return ids


def idea_content(conn, idea_id) -> tuple:
    """(creator_id, format, content) for scoring an idea: the creator summary and the plan, never
    the planner's reasoning."""
    idea = conn.execute("SELECT creator_id, concept, format, body FROM ideas WHERE id = %s", (idea_id,)).fetchone()
    plan = {k: v for k, v in (idea["body"] or {}).items() if k != "reasoning"}
    _, who = creator_summary(conn, idea["creator_id"])
    text = f"THE IDEA ({idea['format']}): {idea['concept']}\nPLAN: {json.dumps(plan, ensure_ascii=False)}"
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
        with tempfile.TemporaryDirectory() as tmp:
            data = st.get_file(a["uri"], Path(tmp) / f"asset{Path(a['uri']).suffix or '.png'}").read_bytes()
        media = "image/jpeg" if a["uri"].lower().endswith((".jpg", ".jpeg")) else "image/png"
        _, who = creator_summary(ctx.conn, a["creator_id"])
        scenes = (a["body"] or {}).get("scenes") or []
        content = [who, f"PLANNED SCENE: {' | '.join(scenes) or '(none recorded)'}", "THE IMAGE:",
                   image_block(data, media), f"CAPTION: {a['caption'] or '(none)'}"]
        ids = score_and_store(ctx.conn, ctx.policy, ctx.llm, "execution", "asset", content,
                              creator_id=a["creator_id"], fmt=a["format"], asset_id=asset_id, item_key=asset_id)
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
