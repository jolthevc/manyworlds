"""The creator profile: everything that makes a creator who they are, in versioned sections.

    identity          face, age, skin, eyes, build. Fixed: the LoRA and reference photos are
                      trained on it, so changing it means a new creator look (not allowed here)
    look              hair, wardrobe, accessories, makeup, and the signature item: one recurring
                      thing they always have, never pushed, that becomes recognisable (and later
                      a product)
    persona           traits, vibe, ethos, humour, caption voice, speaking voice, never-list
    audience          who it is for: age, gender, regions, income, life stage, what they get
    content           pillars (broad, lasting themes) and video series (recurring formats that
                      say something: they build the arc, trust and the path to products)
    arc               the long arc, the current chapter, open threads
    commercialization product lines, how the signature item becomes one, stages, product lines
                      other creators could also carry, and the rules that keep it non-obvious
    dials             posts per day per platform, images per video, tone mix, hashtag limits

Every change is a new version (author + reason), and posts record the version they were
made under, so the learning loop can compare versions. `render_bible` gives the flat view
the image prompts, reviewer and planner read, so a haircut flows into every new photo.
"""

from __future__ import annotations

import copy
import json

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from ..llm import StructuredLLM
from ..policy import Policy

IDENTITY_FIELDS = ("apparent_age", "gender_presentation", "ethnicity_and_skin", "face", "eyes", "build")


class Look(BaseModel):
    hair: str
    wardrobe: str
    accessories: str = Field(description="recurring accessories other than the signature item")
    makeup: str
    signature_item: str = Field(description="ONE distinctive recurring item they always have (e.g. a particular "
                                            "pair of glasses, a necklace, a mug). Precise visual description. "
                                            "Never named, praised or pushed in posts; people notice it over time")


class Persona(BaseModel):
    traits: list[str]
    vibe: str
    ethos: str = Field(description="what they believe and stand for, in one or two sentences")
    humour: str
    caption_voice: str
    speaking_voice: str
    never: list[str]


class Audience(BaseModel):
    age_range: str
    gender: str
    regions: str
    income: str
    ethnicity_mix: str
    life_stage: str
    what_they_get: str = Field(description="why they follow: the feeling or value they come back for")


class Pillar(BaseModel):
    name: str
    about: str
    why_it_lasts: str = Field(description="why this theme still works in a year, not just this season")


class VideoSeries(BaseModel):
    name: str
    premise: str
    format: str = Field(description="narrated_edit (photos + her voice + captions), story_clip (short motion "
                                    "from a photo), or talking_head")
    what_it_says: str = Field(description="what each episode adds: to her arc, to audience trust, and to the "
                                          "long-run product path (without selling)")
    pillar: str


class Arc(BaseModel):
    long_arc: str
    current_chapter: str
    open_threads: list[str]


class Commercialization(BaseModel):
    product_lines: list[str] = Field(description="own-brand product lines that fit the world (e.g. candles, home goods)")
    signature_hook: str = Field(description="how the signature item becomes a product people already recognise")
    stages: list[str] = Field(description="ordered path from no selling, to recognisable world, to products")
    shared_lines: list[str] = Field(description="product lines other creators in the same space could also carry")
    rules: list[str] = Field(description="how it stays non-obvious until it's time")


class Weight(BaseModel):
    name: str
    weight: float


class Cadence(BaseModel):
    platform: str
    posts_per_day: float


class ProfileDraft(BaseModel):
    look: Look
    persona: Persona
    audience: Audience
    pillars: list[Pillar] = Field(description="3-5 broad, lasting content pillars")
    video_series: list[VideoSeries] = Field(description="3-5 recurring video series")
    arc: Arc
    commercialization: Commercialization
    tone_mix: list[Weight] = Field(description="3-5 named tones or flavours of post with weights summing to 1")
    cadence: list[Cadence] = Field(description="posts per day per platform")
    images_per_video: float = Field(description="photo posts per video post")


def current(conn, creator_id) -> tuple[int, dict] | None:
    row = conn.execute("SELECT version, body FROM creator_profiles WHERE creator_id = %s ORDER BY version DESC LIMIT 1",
                       (creator_id,)).fetchone()
    return (row["version"], row["body"]) if row else None


def render_bible(profile: dict, base: dict | None = None) -> dict:
    """The flat bible the image prompts, reviewer and planner read, from the current profile."""
    b = copy.deepcopy(base or {})
    ident, look, persona = profile["identity"], profile["look"], profile["persona"]
    b["appearance"] = {**{k: ident.get(k) for k in IDENTITY_FIELDS}, "hair": look["hair"],
                       "signature_details": f"{look['signature_item']}; {look['accessories']}"}
    b.update(caption_voice=persona["caption_voice"], voice_notes=persona["speaking_voice"], never=persona["never"],
             personality=f"{persona['vibe']} {', '.join(persona['traits'])}. {persona['humour']}",
             style_and_wardrobe=f"{look['wardrobe']} Makeup: {look['makeup']}")
    for k in ("display_name", "handle", "disclosure_line", "kokoro_voice", "home_and_places", "interests",
              "opinions_and_running_jokes", "one_line", "archetype", "niche"):
        if k in profile:
            b[k] = profile[k]
    return b


def _merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def clamp_dials(policy: Policy, dials: dict) -> dict:
    lim = policy.get("creators", "dials_limits", default={})
    d = copy.deepcopy(dials)
    top = float(lim.get("posts_per_day_max", 6))
    platforms = list(policy.get("scope", "platforms", default=[]) or [])
    d["posts_per_day"] = {p: min(max(float(n), 0.0), top) for p, n in (d.get("posts_per_day") or {}).items()
                          if not platforms or p in platforms}
    d["images_per_video"] = min(max(float(d.get("images_per_video", 3)), float(lim.get("images_per_video_min", 1))),
                                float(lim.get("images_per_video_max", 6)))
    hmax = int(lim.get("hashtags_max", 3))
    d["hashtags_max"] = {p: min(max(int(n), 0), hmax) for p, n in (d.get("hashtags_max") or {}).items()}
    mix = {k: max(float(w), 0.0) for k, w in (d.get("tone_mix") or {}).items()}
    total = sum(mix.values())
    d["tone_mix"] = {k: round(w / total, 3) for k, w in mix.items()} if total else {}
    return d


def save(conn, policy: Policy, creator_id, body: dict, *, author: str, reason: str) -> int:
    body = copy.deepcopy(body)
    body["dials"] = clamp_dials(policy, body.get("dials") or {})
    cur = current(conn, creator_id)
    version = (cur[0] + 1) if cur else 1
    conn.execute("INSERT INTO creator_profiles (creator_id, version, body, author, reason, policy_version) "
                 "VALUES (%s, %s, %s, %s, %s, %s)", (creator_id, version, Jsonb(body), author, reason, policy.version))
    bible = conn.execute("SELECT bible FROM creators WHERE id = %s", (creator_id,)).fetchone()["bible"]
    conn.execute("UPDATE creators SET profile_version = %s, bible = %s WHERE id = %s",
                 (version, Jsonb(render_bible(body, bible)), creator_id))
    return version


def update(conn, policy: Policy, creator_id, patch: dict, *, author: str, reason: str) -> int:
    """Change any part of the profile as a new version. Identity is fixed: the creator's face
    is what their LoRA and reference photos are; hair, clothes and everything else can move."""
    cur = current(conn, creator_id)
    if not cur:
        raise LookupError("creator has no profile yet; build one first")
    if "identity" in patch:
        raise ValueError("identity (face, age, skin, eyes, build) is fixed; change look.* instead")
    if not reason.strip():
        raise ValueError("every profile change needs a reason")
    return save(conn, policy, creator_id, _merge(cur[1], patch), author=author, reason=reason)


def set_path(conn, policy: Policy, creator_id, path: str, value, *, author: str, reason: str) -> int:
    """update() for one dotted path, e.g. set_path(.., "look.hair", "a short bob", ...)."""
    patch: dict = {}
    node = patch
    keys = path.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    return update(conn, policy, creator_id, patch, author=author, reason=reason)


SYSTEM = """You are the creative director of Manyworlds, a studio of original fictional social media
creators, all openly labeled as AI. You turn a creator's bible into a full, specific profile. Creators
must be distinct and memorable, never generic: a recognisable world, a point of view, recurring
details people come to know. Monetization is long-game and quiet: a recognisable world and one
signature item first, products later, never obvious selling. Themes must be broad enough to last
for years. Every video series must say something that adds to the arc and to audience trust."""


def build(conn, policy: Policy, llm: StructuredLLM, creator_id, *, force: bool = False) -> int:
    """Version 1 of a creator's profile from their bible and thesis (one Brain call)."""
    if current(conn, creator_id) and not force:
        return current(conn, creator_id)[0]
    c = conn.execute("SELECT handle, bible FROM creators WHERE id = %s", (creator_id,)).fetchone()
    t = conn.execute("SELECT body FROM theses WHERE creator_id = %s ORDER BY created_at DESC LIMIT 1",
                     (creator_id,)).fetchone()
    bible = {k: v for k, v in c["bible"].items() if k != "reference_prompts"}
    defaults = policy.get("creators", "dials_default", default={})
    prompt = (f"Creator bible:\n{json.dumps(bible, default=str)}\n\nThe opportunity they test:\n"
              f"{json.dumps(t['body'] if t else {}, default=str)}\n\nPlatforms: {policy.get('scope', 'platforms')}. "
              f"Never touch: {policy.get('scope', 'excluded_topics')}.\n"
              f"Keep their face, age and existing signature details; pick the signature item from what they "
              f"already wear or carry if one fits. Starting dials for reference (adjust with reason): "
              f"{json.dumps(defaults)}. Early on, photos outnumber videos.")
    d = llm.structured(policy.get("creators", "profile_role", default="creator"), system=SYSTEM, content=prompt,
                       schema=ProfileDraft, effort="high", creator_id=creator_id)
    a = c["bible"]["appearance"]
    body = {
        "identity": {k: a.get(k) for k in IDENTITY_FIELDS},
        "look": d.look.model_dump(), "persona": d.persona.model_dump(), "audience": d.audience.model_dump(),
        "content": {"pillars": [p.model_dump() for p in d.pillars],
                    "video_series": [v.model_dump() for v in d.video_series]},
        "arc": d.arc.model_dump(), "commercialization": d.commercialization.model_dump(),
        "dials": {"posts_per_day": {x.platform: x.posts_per_day for x in d.cadence},
                  "images_per_video": d.images_per_video,
                  "tone_mix": {w.name: w.weight for w in d.tone_mix},
                  "hashtags_max": defaults.get("hashtags_max", {})},
        **{k: c["bible"][k] for k in ("display_name", "handle", "disclosure_line", "kokoro_voice", "home_and_places",
                                      "interests", "opinions_and_running_jokes", "one_line", "archetype", "niche")
           if k in c["bible"]},
    }
    return save(conn, policy, creator_id, body, author="brain", reason="initial profile from bible and thesis")
