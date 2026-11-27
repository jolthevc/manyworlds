"""Our own accepted content, coded in the research engine's vocabulary (owner 2026-11-26).

The research engine codes other creators' posts with schema iia_v2 (manyworlds.research.annotation: 33 coded
attributes from policy intelligence.annotation.vocab). The same read, with the same models, prompt and vocabularies, is
run on our own accepted assets, and stored in the same table (reference_annotations, `source` own, with our post and
asset instead of a reference post), so research, direction (performance_observations), achieved characteristics, cost
and outcomes join later (manyworlds.brain.evidence.own_annotations).

Runs from the reviewer loop while policy performance.own_annotation.enabled (off as shipped): only accepted assets
(every review row passed, confirmed or a clear pass) of our own posts, never set plates; images as they are, a clip as
a few of its keyframes; batched (`assets_per_call` per call, at most `max_assets_per_run` per run), on the cheapest
vision role (`role`, the research read's), charged to the ledger under `agent`. Idempotent: an asset is read once per
schema (a unique index on own_asset_id, schema).
"""

from __future__ import annotations

import logging
from pathlib import Path

from psycopg.types.json import Jsonb

from ..llm import StructuredLLM, image_block
from ..policy import Policy
from ..research import annotation as ann
from ..storage import ObjectStore

log = logging.getLogger("manyworlds.own_annotation")

OWN_NOTE = ("These are the studio's own generated posts (openly AI-generated creators), coded in the same vocabulary "
            "as the research posts so the two can be compared. ")


def _cfg(policy: Policy) -> dict:
    return policy.get("performance", "own_annotation", default={}) or {}


def pending(conn, policy: Policy, *, limit: int | None = None) -> list[dict]:
    """Accepted image and clip assets of our own posts (not dropped) without an own annotation under the current
    schema, oldest first."""
    limit = int(limit or _cfg(policy).get("max_assets_per_run", 40))
    return [dict(r) for r in conn.execute(
        """SELECT a.id, a.uri, a.kind, a.job_id, j.post_id, p.format, p.body->>'caption' AS caption
           FROM assets a JOIN jobs j ON j.id = a.job_id JOIN posts p ON p.id = j.post_id
           WHERE a.kind IN ('image', 'video') AND a.deleted_at IS NULL AND p.status <> 'dropped'
             AND coalesce(j.payload->>'plate', '') NOT IN ('true')
             AND EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id = a.id)
             AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id = a.id AND NOT r.passed)
             AND EXISTS (SELECT 1 FROM reviews r WHERE r.asset_id = a.id
                         AND (r.details->>'stage' = 'confirm' OR r.details->>'clear_pass' = 'true'))
             AND NOT EXISTS (SELECT 1 FROM reference_annotations o WHERE o.source = 'own' AND o.own_asset_id = a.id
                             AND o.schema = %s)
           ORDER BY a.created_at, a.id LIMIT %s""", (ann.ANNOTATION_SCHEMA, limit)).fetchall()]


def _images(conn, store: ObjectStore, asset: dict, workdir: Path, cfg: dict) -> list[bytes]:
    """What the reader sees of an asset: the image, or a clip's first keyframes (keyframes_per_clip); each under
    image_max_bytes."""
    cap = int(cfg.get("image_max_bytes", 400000))
    uris = [asset["uri"]] if asset["kind"] == "image" else [r["uri"] for r in conn.execute(
        "SELECT uri FROM assets WHERE job_id = %s AND kind = 'keyframe' ORDER BY uri LIMIT %s",
        (asset["job_id"], int(cfg.get("keyframes_per_clip", 3)))).fetchall()]
    out = []
    for i, uri in enumerate(uris):
        try:
            data = store.get_file(uri, workdir / f"own-{asset['id']}-{i}{Path(uri).suffix or '.png'}").read_bytes()
        except Exception:  # noqa: BLE001 - a missing file: the read goes on with what remains
            continue
        if len(data) > cap:   # our photos are larger than the cap: downscale, never skip (2026-11-27: every own
            data = _shrink(data, cap, int(cfg.get("image_max_side", 1024)))   # read was silently empty)
        if data and len(data) <= cap:
            out.append(data)
    return out


def _shrink(data: bytes, cap: int, side: int) -> bytes | None:
    """The image as a JPEG of at most `side` px on its long edge and under `cap` bytes (quality stepped down), or None."""
    import io

    from PIL import Image
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            im.thumbnail((side, side))
            for q in (85, 75, 65, 55):
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=q)
                if buf.tell() <= cap:
                    return buf.getvalue()
    except Exception:  # noqa: BLE001 - an unreadable image is left out
        return None
    return None


def _media_type(data: bytes) -> str:
    return "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"


def run_pending(conn, policy: Policy, llm: StructuredLLM, store: ObjectStore, workdir: Path) -> dict:
    """Code our pending accepted assets (see the module docstring). {enabled, assets, annotated, calls}."""
    cfg = _cfg(policy)
    if not cfg.get("enabled"):
        return {"enabled": False}
    from ..brain.agents.intelligence import ReadBatch, read_system
    assets = pending(conn, policy)
    per_call = max(1, int(cfg.get("assets_per_call", 4)))
    role, chars = str(cfg.get("role", "iia_annotate")), int(cfg.get("caption_chars", 600))
    written = calls = 0
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(assets), per_call):
        chunk = assets[i:i + per_call]
        blocks: list[dict] = [{"type": "text", "text": (
            f"{OWN_NOTE}Annotate each of these {len(chunk)} posts (numbered from 1) into the schema, observed.coded "
            f"from the coded lists.")}]
        seen = []
        for j, a in enumerate(chunk, 1):
            pics = _images(conn, store, a, workdir, cfg)
            if not pics:
                continue
            medium = "video" if a["kind"] == "video" else "photo"
            blocks.append({"type": "text", "text": f"Post {j}: {medium}; caption: {(a.get('caption') or '')[:chars]!r}"})
            blocks += [image_block(b, _media_type(b)) for b in pics]
            seen.append((j, a))
        if not seen:
            continue
        with llm.tagged(str(cfg.get("agent", "own_annotation")), None):
            got = llm.structured(role, system=OWN_NOTE + read_system(policy), content=blocks, schema=ReadBatch,
                                 effort=str(cfg.get("effort", "low")))
        calls += 1
        reads = {r.post: r.annotation for r in (getattr(got, "reads", None) or [])
                 if isinstance(getattr(r, "post", None), int)}
        for j, a in seen:
            written += store_read(conn, policy, a, reads.get(j) or ann.Annotation(), llm.model_for(role))
    return {"enabled": True, "assets": len(assets), "annotated": written, "calls": calls}


def store_read(conn, policy: Policy, asset: dict, a: ann.Annotation, model: str) -> int:
    """One own read stored as the research engine stores its own (coded values against this policy's vocabularies,
    vocab_versions kept), with source own. Returns 1 when written, 0 when it already existed."""
    coded = getattr(a.observed, "coded", None)
    raw = coded.model_dump(mode="json") if hasattr(coded, "model_dump") else (coded or {})
    a.observed.coded, _ = ann.coerce_coded(raw, ann.vocab(policy))
    notes = lambda xs: [{"text": ann.clip_note(n.text), "audience_role": n.audience_role} for n in xs]  # noqa: E731
    cur = conn.execute(
        """INSERT INTO reference_annotations (source, own_post_id, own_asset_id, schema, model, observed, mechanism,
                                              speculation, confidence)
           VALUES ('own', %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (own_asset_id, schema) WHERE source = 'own' DO NOTHING""",
        (asset["post_id"], asset["id"], ann.ANNOTATION_SCHEMA, model,
         Jsonb({**a.observed.model_dump(mode="json"), "vocab_versions": ann.vocab_versions(policy)}),
         Jsonb(notes(a.mechanism)), Jsonb(notes(a.speculation)), float(a.confidence)))
    return cur.rowcount or 0
