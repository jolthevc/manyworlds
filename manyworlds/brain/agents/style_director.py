"""The style director: one judgement role that owns how each creator looks and feels.

The talent manager owns her career (who she is for, what she says, which series); the style director owns her
visual identity: what stays the same so she is recognisable (face, style, palette, grade, signature items,
motifs, the 1-2 home spaces where she talks), and what must change so she is worth watching (outfits, places,
trips, light, shot types). It writes three parts of her profile, each one small checked call:

    style    vibe, palette, grade, signature items, wardrobe rules, motifs, tastes, the shot families her feed is
             made of and how often her face shows (manyworlds.brain.aesthetics is the vocabulary)
    world    anchor sets (set_id, what she does there, the fixed room), her home base and hemisphere, how she
             travels, and her carousel brand (fonts, colours, layout) for the slide renderer (manyworlds.slides)
    season   refreshed every `season_days`: this month's capsule wardrobe (new outfits built from her style), where
             she is (home, or a trip with its days), seasonal notes for light, weather and things to do

The showrunner and director plan from these; the director names only anchor sets as recurring sets, and draws
outfits from the capsule (manyworlds.creator.sets keeps no canonical outfits for a creator with a style).

Modes: onboard (everything missing), season (the monthly refresh). Code checks what a model can't be trusted with:
grade and shot names from the library, fonts the renderer has, hex colours, a normalised face share.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from ... import queue
from ...creator import profile as profile_mod
from ...worker.runner import NonRetryable
from .. import aesthetics
from ..agent import AgentContext, register, remember

ROLE = "style_director"
PARTS = ("style", "world", "season")
FONTS = ("DMSerifDisplay", "Poppins", "Anton", "TikTokSans")   # manyworlds/fonts (OFL), what the renderer draws
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

SYSTEM = """You are the style director of Manyworlds, a studio of original fictional social media creators, all
openly labelled as AI. You own how each creator looks and feels: the vibe, aesthetics and visual world that make
her feed aspirational and exciting to follow, never boring and never generic.

What stays the same so she is recognisable: her face and appearance (fixed, never yours to change), her style
(silhouettes, fabrics, colours she wears), one grade, a small palette, 2-3 signature items, a few motifs, and 1-2
anchor sets: the home spaces where she talks to camera, gets ready or records.
What must change so she is worth watching: outfits (new ones built from her style; a re-wear is fine after a
week or so, styled differently), places (she goes out, sees friends, travels, has a life beyond her flat), light,
time of day, and shot types: most of an aspirational feed is not her face to camera but her world (her back on a
street at sunset, a ringed hand on a glass, a dinner table from above, a window, her shoes on cobblestones).
Be specific and concrete: name colours, fabrics, places and objects. Plain and unbranded: no real brands."""


class Colour(BaseModel):
    name: str
    hex: str = Field(description="#rrggbb")


class WardrobeStyle(BaseModel):
    silhouettes: list[str] = Field(description="the cuts and shapes she wears, e.g. 'wide-leg trousers', 'cropped knit'")
    fabrics: list[str]
    colours: list[str] = Field(description="the colours she wears (within the palette or near it)")
    footwear: list[str]
    never: list[str] = Field(description="what she never wears")
    rewear_after_days: int = Field(description="days before an outfit may come back, styled differently (7-21)")


class Tastes(BaseModel):
    foods: list[str]
    drinks: list[str]
    places: list[str] = Field(description="kinds of places she loves: 'corner bakeries', 'lidos', 'bookshop cafes'")


class FaceShare(BaseModel):
    face: float = Field(description="share of shots with her face clearly shown, 0-1")
    partial: float = Field(description="share with her face partly shown or turned away")
    none: float = Field(description="share with no face: hands, details, places, food, objects")


class StylePart(BaseModel):
    vibe: str = Field(description="the feeling of her feed in one sentence")
    palette: list[Colour] = Field(description="3-5 colours her feed lives in")
    grade: str = Field(description="one grade family from the library")
    night_grade: str = Field(description="the grade family for night and flash shots")
    signature_items: list[str] = Field(description="2-3 items people come to know her by: a ring, a coat, a mug")
    wardrobe: WardrobeStyle
    motifs: list[str] = Field(description="3-5 recurring visual motifs: 'fresh flowers on every table', 'film tickets'")
    tastes: Tastes
    shot_families: list[str] = Field(description="8-14 shot types from the library her feed is made of")
    face_share: FaceShare


class AnchorSet(BaseModel):
    set_id: str = Field(description="short snake_case name, e.g. 'living_room_corner'")
    purpose: str = Field(description="what she does there: talks to camera, gets ready, records the podcast")
    description: str = Field(description="the fixed room as an empty set: layout, materials, colours, furniture, "
                                         "light sources; objects and surfaces only")


class Travel(BaseModel):
    trips_per_quarter: int = Field(description="0-3")
    destinations: list[str] = Field(description="places she goes, plausible for her life and budget")


class CarouselBrand(BaseModel):
    headline_font: Literal["DMSerifDisplay", "Poppins", "Anton", "TikTokSans"]
    body_font: Literal["Poppins", "TikTokSans", "DMSerifDisplay"]
    background: str = Field(description="#rrggbb")
    text: str = Field(description="#rrggbb, readable on the background")
    accent: str = Field(description="#rrggbb")
    layout: Literal["editorial", "notebook", "bold"] = Field(description="editorial: serif, airy, photo-led; "
                                                                          "notebook: soft, handwritten feel; bold: big type")


class WorldPart(BaseModel):
    anchor_sets: list[AnchorSet] = Field(description="1-2 home spaces where she talks, gets ready or records")
    home_base: str = Field(description="her city and neighbourhood")
    hemisphere: Literal["north", "south"]
    travel: Travel
    carousel: CarouselBrand


class Trip(BaseModel):
    where: str = Field(description="'home' or the destination")
    first_day: int = Field(description="day of the season the trip starts (0 when at home)")
    days: int = Field(description="how many days it lasts (0 when at home)")


class SeasonPart(BaseModel):
    capsule: list[str] = Field(description="8-12 outfits for this month, each one line: garments, colours, fabrics")
    trips: list[Trip] = Field(description="where she is this month: home weeks and at most one trip")
    notes: list[str] = Field(description="3-6 seasonal notes: the light, the weather, what she does this month")


SCHEMAS = {"style": StylePart, "world": WorldPart, "season": SeasonPart}


def _cfg(policy) -> dict:
    return policy.get("brain", "style_director", default={}) or {}


def clean(policy, part: str, out: BaseModel) -> dict:
    """What a part saves, after code checks."""
    d = out.model_dump()
    if part == "style":
        known = set(aesthetics.grades(policy))
        fallback = next(iter(known), "raw_phone")
        d["grade"] = d["grade"] if d["grade"] in known else fallback
        d["night_grade"] = d["night_grade"] if d["night_grade"] in known else d["grade"]
        d["palette"] = [c for c in d["palette"] if HEX.match(str(c.get("hex") or ""))][:5]
        lib = aesthetics.library(policy)["shots"]
        fam = [s for s in dict.fromkeys(d["shot_families"]) if s in lib]
        faceless = [s for s in fam if str(lib[s].get("face")) != "yes"]
        need = int(_cfg(policy).get("min_faceless_families", 4))
        if len(faceless) < need:   # an aspirational feed is mostly her world, not her face
            fam += [s for s in lib if str(lib[s].get("face")) != "yes" and s not in fam][:need - len(faceless)]
        d["shot_families"] = fam
        share = {k: max(float(v or 0), 0.0) for k, v in d["face_share"].items()}
        total = sum(share.values()) or 1.0
        d["face_share"] = {k: round(v / total, 3) for k, v in share.items()}
        d["signature_items"] = d["signature_items"][:3]
        d["wardrobe"]["rewear_after_days"] = max(5, min(int(d["wardrobe"]["rewear_after_days"] or 7), 30))
        return {"style": d}
    if part == "world":
        seen, sets = set(), []
        for s in d["anchor_sets"][:2]:
            sid = re.sub(r"[^a-z0-9_]+", "_", str(s.get("set_id") or "").lower()).strip("_")
            if sid and sid not in seen:
                seen.add(sid)
                sets.append({**s, "set_id": sid})
        d["anchor_sets"] = sets
        c = d["carousel"]
        defaults = {"background": "#f4efe8", "text": "#2b2622", "accent": "#b5654a"}
        for k, v in defaults.items():
            c[k] = c[k] if HEX.match(str(c.get(k) or "")) else v
        d["travel"]["trips_per_quarter"] = max(0, min(int(d["travel"]["trips_per_quarter"] or 0), 3))
        return {"world": d}
    days = int(_cfg(policy).get("season_days", 30))
    trips = [t for t in d["trips"] if str(t.get("where", "")).strip().lower() != "home" and int(t.get("days") or 0) > 0]
    d["trips"] = [{**t, "first_day": max(0, min(int(t["first_day"]), days - 1)),
                   "days": max(1, min(int(t["days"]), days))} for t in trips[:1]]
    d["capsule"] = d["capsule"][:12]
    return {"season": {**d, "started": datetime.now(timezone.utc).date().isoformat(), "days": days}}


def _prompt(policy, prof: dict, bible: dict, part: str, now: datetime) -> str:
    import json
    lib = aesthetics.library(policy)
    have = {k: prof.get(k) for k in ("persona", "archetypes", "positioning", "audience_insight", "story", "style",
                                     "world") if prof.get(k)}
    ask = {
        "style": ("Write her style. Grades to pick from: " + json.dumps(lib["grades"]) + ". Shot types to pick her "
                  "families from (most should not show her face): " + json.dumps(aesthetics.catalogue(policy))),
        "world": ("Write her world: 1-2 anchor sets (only the home spaces where recurring talk, get-ready or "
                  "recording happens; every other place should vary), her home base, travel, and her carousel "
                  f"brand (fonts: {list(FONTS)}; colours from her palette)."),
        "season": (f"Write this month for her (it is {now:%B %Y}): a capsule of new outfits built from her style "
                   "(not the same outfits as last month), where she is (home, or one trip if her travel allows), "
                   "and seasonal notes for her hemisphere."),
    }[part]
    last = prof.get("season") or {}
    return (f"Creator bible: {json.dumps(bible, default=str)}\n\nProfile so far: {json.dumps(have, default=str)}\n\n"
            + (f"Last month's capsule (make this one different): {json.dumps(last.get('capsule'))}\n\n" if part == 'season' and last else "")
            + f"Rules: {json.dumps(lib['rules'])}\n\n{ask}")


def build_part(ctx: AgentContext, creator_id, part: str) -> dict:
    conn, policy = ctx.conn, ctx.policy
    prof = profile_mod.current(conn, creator_id)[1]
    bible = conn.execute("SELECT bible FROM creators WHERE id = %s", (creator_id,)).fetchone()["bible"] or {}
    now = datetime.now(timezone.utc)
    role = _cfg(policy).get("role", ROLE)

    def compute():
        out = ctx.llm.structured(role, system=profile_mod.persona_system(policy, SYSTEM),
                                 content=_prompt(policy, prof, bible, part, now), schema=SCHEMAS[part],
                                 effort=_cfg(policy).get("effort", "medium"), creator_id=creator_id)
        return {"kind": "style_part", "creator_id": creator_id, "inputs": {"part": part},
                "output": clean(policy, part, out), "rationale": f"style director: {part}",
                "model": ctx.llm.model_for(role)}
    step = f"{part}:{now:%Y-%m}" if part == "season" else part
    patch = remember(ctx, step, compute)
    profile_mod.update(conn, policy, creator_id, patch, author=ROLE, reason=f"style director: {part}")
    return patch


def season_due(policy, prof: dict, now: datetime) -> bool:
    s = prof.get("season") or {}
    try:
        started = datetime.fromisoformat(str(s.get("started"))).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return True
    return (now - started).days >= int(_cfg(policy).get("season_days", 30))


class StyleDirector:
    name = ROLE

    def run(self, ctx: AgentContext) -> dict:
        p = ctx.job.payload or {}
        creator_id = p.get("creator_id")
        cur = profile_mod.current(ctx.conn, creator_id) if creator_id else None
        if not cur:
            raise NonRetryable("the creator has no profile yet (the launch builds it first)")
        built = []
        for part in PARTS:
            prof = profile_mod.current(ctx.conn, creator_id)[1]
            if part != "season" and prof.get(part):
                continue
            if part == "season" and not season_due(ctx.policy, prof, datetime.now(timezone.utc)):
                continue
            build_part(ctx, creator_id, part)
            built.append(part)
        return {"built": built}


register(StyleDirector())


def ensure(conn, policy, *, now: datetime | None = None) -> list:
    """A style director job for each active creator missing a style or world, or whose season is due."""
    now = now or datetime.now(timezone.utc)
    if not _cfg(policy).get("enabled"):
        return []
    out = []
    rows = conn.execute("SELECT c.id FROM creators c WHERE c.status = 'active' AND c.anchors->'simulation' IS NULL "
                        "AND EXISTS (SELECT 1 FROM creator_profiles p WHERE p.creator_id = c.id)").fetchall()
    for r in rows:
        prof = profile_mod.current(conn, r["id"])[1]
        if prof.get("style") and prof.get("world") and not season_due(policy, prof, now):
            continue
        out.append(queue.enqueue(conn, "brain.style_director", {"creator_id": str(r["id"])}, creator_id=r["id"],
                                 idempotency_key=f"style:{r['id']}:{now:%Y-%m}"))
    return out


def planning_view(policy, prof: dict | None, *, today=None) -> dict:
    """What the showrunner and director plan from: the style, the world (anchor sets, home, travel), this
    season (capsule, where she is today) and the shot vocabulary of her families."""
    prof = prof or {}
    style, world, season = prof.get("style") or {}, prof.get("world") or {}, prof.get("season") or {}
    if not style:
        return {}
    where = "home"
    try:
        started = datetime.fromisoformat(str(season.get("started"))).date()
        day = ((today or datetime.now(timezone.utc).date()) - started).days
        for t in season.get("trips") or []:
            if int(t["first_day"]) <= day < int(t["first_day"]) + int(t["days"]):
                where = t["where"]
    except (TypeError, ValueError, KeyError):
        pass
    return {"vibe": style.get("vibe"), "palette": [c.get("name") for c in style.get("palette") or []],
            "grade": style.get("grade"), "signature_items": style.get("signature_items"),
            "wardrobe": style.get("wardrobe"), "motifs": style.get("motifs"), "tastes": style.get("tastes"),
            "face_share": style.get("face_share"),
            "anchor_sets": {s["set_id"]: f"{s.get('purpose')}: {s.get('description')}" for s in world.get("anchor_sets") or []},
            "home_base": world.get("home_base"), "where_she_is_now": where,
            "capsule": season.get("capsule"), "season_notes": season.get("notes"),
            "shot_types": aesthetics.catalogue(policy, style.get("shot_families")),
            **aesthetics.brief(policy)}
