"""Creator Lab: turn a chosen thesis into a persistent creator.

    write_bible -> create_creator -> enqueue_reference_pack -> select_anchors
                -> enqueue_lora_training -> enqueue_voice_sample

The bible is the creator's source of truth: fixed appearance descriptors (so every
generation describes the same person), personality, taste, voice rules, a canon log
for memory, and the AI disclosure line.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from .. import queue
from ..llm import StructuredLLM, image_block
from ..policy import Policy
from ..storage import ObjectStore

KOKORO_VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_nova", "af_river",
                 "am_adam", "am_michael", "am_eric", "am_liam", "am_puck",
                 "bf_emma", "bf_isabella", "bf_alice", "bm_george", "bm_lewis", "bm_daniel"]

SYSTEM = """You design original fictional social media creators for Manyworlds, a studio whose
creators are all openly labeled as AI. A creator must feel like a believable, specific person with
an ordinary, phone-native life: mundane moments, recurring tastes, opinions and jokes, without
forced storylines. Never base a creator on a real person, celebrity or existing influencer; never
make a minor; never sexualize. Appearance descriptors must be precise and stable so an image model
draws the same person every time."""


class Appearance(BaseModel):
    apparent_age: int = Field(description="apparent age in years; at least the policy minimum")
    gender_presentation: str
    ethnicity_and_skin: str
    face: str = Field(description="face shape, features, distinctive marks, in precise visual terms")
    eyes: str
    hair: str
    build: str
    signature_details: str = Field(description="recurring accessories, glasses, tattoos, nails, etc.")


class CreatorBible(BaseModel):
    handle: str = Field(description="lowercase handle, letters/digits/underscores/periods, no '@'")
    display_name: str
    archetype: str
    niche: str
    one_line: str = Field(description="who they are, in one line")
    appearance: Appearance
    style_and_wardrobe: str
    home_and_places: str = Field(description="their apartment, neighborhood, recurring spots")
    personality: str
    interests: list[str]
    opinions_and_running_jokes: list[str]
    caption_voice: str = Field(description="how captions read: length, punctuation, emoji habits, slang")
    posting_habits: str
    never: list[str] = Field(description="things this creator never does or posts")
    disclosure_line: str = Field(description="bio line disclosing the account is AI-generated")
    kokoro_voice: str = Field(description="one voice id from the provided list")
    voice_notes: str = Field(description="how they sound when talking to camera")
    reference_prompts: list[str] = Field(
        description="prompts for the reference pack: varied angles, lighting and everyday scenes of "
                    "the same person, phone-photo style; each restates the fixed appearance")


class AnchorPick(BaseModel):
    best_index: int = Field(description="0-based index of the best candidate, or -1 if none is usable")
    consistent_with_anchor: bool
    defects: list[str]
    reasoning: str


def _slug(handle: str) -> str:
    return re.sub(r"[^a-z0-9_.]", "", handle.lower().lstrip("@"))[:30] or "creator"


def appearance_prompt(bible: dict) -> str:
    a = bible["appearance"]
    return (f"{a['apparent_age']}-year-old {a['gender_presentation']}, {a['ethnicity_and_skin']}, "
            f"{a['face']}, {a['eyes']}, {a['hair']}, {a['build']}, {a['signature_details']}")


def photo_prompt(bible: dict, scene: str, *, trigger: str | None = None) -> str:
    """The prompt for any photo of a creator: LoRA trigger, the full fixed appearance, then the scene.
    Measured 2026-11-14 on Maren: a short description with her LoRA kept 0 of 12 images on
    identity; the full appearance with the same LoRA kept 12 of 12."""
    head = f"{trigger}, " if trigger else ""
    return f"{head}candid smartphone photo of a {appearance_prompt(bible)}. Scene: {scene}. Natural skin texture, realistic."


def write_bible(conn, policy: Policy, llm: StructuredLLM, thesis: dict) -> CreatorBible:
    scope = policy.get("scope", default={})
    n = int(policy.get("creator_lab", "reference_prompts", default=12))
    taken = [r["handle"] for r in conn.execute("SELECT handle FROM creators").fetchall()]
    prompt = (
        f"Opportunity to test:\n{json.dumps({k: v for k, v in thesis.items() if k != 'id'}, default=str)}\n\n"
        f"Design the creator. Minimum apparent age {scope.get('creator_min_age', 21)}. "
        f"Off-limits topics: {scope.get('excluded_topics')}. Disclosure must say '{scope.get('disclosure_line')}'.\n"
        f"Handles already taken: {taken}.\nKokoro voices to choose from: {KOKORO_VOICES}.\n"
        f"Write exactly {n} reference_prompts.")
    bible = llm.structured("creator", system=SYSTEM, content=prompt, schema=CreatorBible, effort="high")
    bible.handle = _slug(bible.handle)
    if bible.appearance.apparent_age < int(scope.get("creator_min_age", 21)):
        raise ValueError(f"bible apparent_age {bible.appearance.apparent_age} below policy minimum")
    if bible.kokoro_voice not in KOKORO_VOICES:
        bible.kokoro_voice = "af_heart" if bible.appearance.gender_presentation.lower().startswith(("f", "w")) else "am_michael"
    if (scope.get("disclosure_line") or "").lower() not in bible.disclosure_line.lower():
        bible.disclosure_line = f"{scope.get('disclosure_line')} · {bible.disclosure_line}"
    return bible


def create_creator(conn, bible: CreatorBible, thesis_id) -> dict:
    handle, i = bible.handle, 1
    while conn.execute("SELECT 1 FROM creators WHERE handle = %s", (handle,)).fetchone():
        i += 1
        handle = f"{bible.handle}{i}"
    body = bible.model_dump()
    body["handle"] = handle
    row = conn.execute(
        "INSERT INTO creators (handle, status, bible, anchors) VALUES (%s, 'draft', %s, %s) RETURNING id, handle",
        (handle, Jsonb(body), Jsonb({"voice": {"engine": "kokoro", "voice": bible.kokoro_voice}})),
    ).fetchone()
    conn.execute("UPDATE theses SET creator_id = %s, status = 'launched' WHERE id = %s", (row["id"], thesis_id))
    conn.execute("INSERT INTO decisions (kind, creator_id, inputs, output) VALUES ('creator_launch', %s, %s, %s)",
                 (row["id"], Jsonb({"thesis_id": str(thesis_id)}), Jsonb({"handle": handle})))
    return row


def enqueue_reference_pack(conn, policy: Policy, creator_id, *, kind: str = "image.zimage") -> str:
    c = conn.execute("SELECT bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
    look = appearance_prompt(c["bible"])
    n = int(policy.get("creator_lab", "candidates_per_reference", default=4))
    img = policy.get("generation", "image", default={})
    batch = f"refpack-{creator_id}"
    for i, scene in enumerate(c["bible"]["reference_prompts"]):
        prompt = f"{scene}. The person: {look}. Candid smartphone photo, natural skin texture, realistic."
        queue.enqueue(conn, kind, {"prompt": prompt, "n": n, "seed": 7000 + i * 10,
                                   "width": img.get("width", 896), "height": img.get("height", 1152),
                                   "params": {"steps": img.get("steps", 9), "guidance": img.get("guidance", 0.0)}},
                      creator_id=creator_id, batch=batch, idempotency_key=f"{creator_id}:ref:{i}")
    return batch


def select_anchors(conn, policy: Policy, llm: StructuredLLM, store: ObjectStore, creator_id, workdir) -> dict:
    """Pick one image per reference prompt that matches the first pick (the anchor face).
    Day 4 adds a face-embedding score; for now a vision judgement does the selection."""
    from pathlib import Path

    jobs = conn.execute(
        "SELECT id, idempotency_key FROM jobs WHERE creator_id = %s AND idempotency_key LIKE %s AND status = 'done' "
        "ORDER BY idempotency_key", (creator_id, f"{creator_id}:ref:%")).fetchall()
    jobs.sort(key=lambda j: int(j["idempotency_key"].rsplit(":", 1)[1]))
    chosen: list[dict] = []
    anchor_bytes = None
    for j in jobs:
        assets = conn.execute("SELECT id, uri FROM assets WHERE job_id = %s ORDER BY uri", (j["id"],)).fetchall()
        imgs = [store.get_file(a["uri"], Path(workdir) / f"{a['id']}.png").read_bytes() for a in assets]
        content: list = []
        if anchor_bytes:
            content += [{"type": "text", "text": "ANCHOR (the creator's established face):"}, image_block(anchor_bytes)]
        for i, b in enumerate(imgs):
            content += [{"type": "text", "text": f"Candidate {i}:"}, image_block(b)]
        content.append({"type": "text", "text": (
            "Pick the most usable candidate: natural phone-photo realism, no anatomical or text defects"
            + (", and clearly the same person as the ANCHOR." if anchor_bytes else ", with a clear, well-lit face.")
            + " Return -1 if none is usable.")})
        pick = llm.structured("judge", system="You are a strict photo editor for an AI creator studio.",
                              content=content, schema=AnchorPick, effort="medium", creator_id=creator_id)
        if 0 <= pick.best_index < len(assets) and (pick.consistent_with_anchor or anchor_bytes is None):
            a = assets[pick.best_index]
            chosen.append({"asset_id": str(a["id"]), "uri": a["uri"], "reasoning": pick.reasoning})
            anchor_bytes = anchor_bytes or imgs[pick.best_index]
    anchors = conn.execute("SELECT anchors FROM creators WHERE id = %s", (creator_id,)).fetchone()["anchors"]
    anchors["reference_images"] = chosen
    conn.execute("UPDATE creators SET anchors = %s WHERE id = %s", (Jsonb(anchors), creator_id))
    return {"selected": len(chosen), "of": len(jobs)}


def enqueue_lora_training(conn, policy: Policy, creator_id) -> object:
    c = conn.execute("SELECT handle, bible, anchors FROM creators WHERE id = %s", (creator_id,)).fetchone()
    refs = c["anchors"].get("reference_images", [])
    if len(refs) < 8:
        raise ValueError(f"need at least 8 approved reference images, have {len(refs)}")
    trigger = f"mw_{c['handle']}"
    caption = f"photo of {trigger}, {appearance_prompt(c['bible'])}"
    cfg = policy.get("creator_lab", "lora", default={})
    payload = {"creator_id": str(creator_id), "trigger": trigger,
               "images": [{"uri": r["uri"], "caption": caption} for r in refs], **cfg}
    job = queue.enqueue(conn, "train.lora", payload, creator_id=creator_id,
                        idempotency_key=f"{creator_id}:lora:v1", max_attempts=2)
    # Asking again after a failure retries it (with the current settings); a finished
    # or in-progress training is left alone.
    conn.execute("""UPDATE jobs SET status = 'queued', attempts = 0, worker_faults = 0, worker_id = NULL,
                        lease_until = NULL, error = NULL, payload = %s, updated_at = now()
                    WHERE id = %s AND status IN ('failed', 'dead')""", (Jsonb(payload), job))
    return job


def enqueue_voice_sample(conn, creator_id) -> object:
    c = conn.execute("SELECT bible, anchors FROM creators WHERE id = %s", (creator_id,)).fetchone()
    line = f"Hi, I'm {c['bible']['display_name']}. {c['bible']['one_line']}"
    return queue.enqueue(conn, "tts.kokoro", {"text": line, "voice": c["anchors"]["voice"]["voice"]},
                         creator_id=creator_id, idempotency_key=f"{creator_id}:voice:sample")
