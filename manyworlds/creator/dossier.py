"""The creator dossier: the sections of a profile that make a creator a brand, built section by section.

The profile (manyworlds.creator.profile) already holds look, persona, audience demographics, pillars, video
series, arc and commercialization. The dossier adds what a talent manager would insist on before any content:

    archetypes        primary (and secondary) playbook from the library (manyworlds.brain.archetypes)
    positioning       why anyone follows her, the gap she fills, her point of view, how she differs from her
                      siblings in the portfolio, the patterns of successful creators she resembles (never people)
    audience_insight  one specific person she talks to, their desires, frustrations, when they scroll, what they
                      save and share, their words, what turns them off
    story             backstory, values in action, quirks, routines, small goals, recurring fictional people
    content_system    the content mix and talk share for her archetypes, series built from capabilities, hooks,
                      what she never makes

and the Talent & Brand Bible (owner 2026-11-25; spec 2026-11-25-talent-agency): why she exists as talent, what she
represents, why people follow and how her value compounds. It complements the production bible (the creator lab's
appearance and the style director's style and world):

    brand_core        the stable core: why she exists, the desire she embodies, what she represents, values, ethos,
                      aspirations, personality, the desirable traits of her look, her relationship with her audience,
                      her fictional narrative; changes only with evidence (brain.talent_manager.brand)
    brand_world       style essence, environments, hobbies, routines, visual language, creative repertoire (the content
                      she does best, on formats production renders well: strategy.capability_card)
    brand_commerce    her natural commercial categories (the registry: manyworlds.brain.commerce), how products
                      would enter her world, and what she will and won't do, wear and promote

Each section is its own small structured call (cheap, retried alone, versioned as its own profile change), in
order, each seeing the ones before. Code checks what a model can't be trusted with: archetype names exist,
capabilities exist and render now, the mix sums to 1, repertoire formats are on the capability card, commercial
categories are in the registry, and positioning overlaps no sibling past policy `brain.talent_manager.max_overlap`
(word overlap, deterministic).
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field

from .. import direction
from ..brain import archetypes as arch
from ..llm import StructuredLLM
from ..policy import Policy
from . import profile as profile_mod

SECTIONS = ("archetypes", "positioning", "brand_core", "audience_insight", "story", "brand_world", "content_system",
            "brand_commerce")
BRAND_SECTIONS = ("brand_core", "brand_world", "brand_commerce")

SYSTEM = """You are the talent manager of a Manyworlds creator: an original fictional social media creator, openly
labeled as AI. Manyworlds is a talent agency and lifestyle media network that owns its talent: each creator is a
desirable, aspirational personality and a brand that compounds. You own their career the way the best human talent
managers do: identity first, then audience, then content, then commercial value. You are forensic and specific: every
line must be something a writer, a director and a stylist can act on, and that a real audience would recognise. Never
generic ("relatable", "authentic", "engaging" say nothing). Desirability is the goal: attractive, magnetic, stylish
and aspirational in the creator's own register, judged in context, never explicit. Never base anything on a real
person or existing creator; describe patterns, not people. Never touch excluded topics."""


class ArchetypePick(BaseModel):
    primary: str = Field(description="one archetype name from the library")
    secondary: str | None = Field(default=None, description="an optional second archetype name, or null")
    why: str


class Positioning(BaseModel):
    why_follow: str = Field(description="the one reason someone follows her, in a viewer's words")
    the_gap: str = Field(description="what is missing in this niche that she fills")
    point_of_view: str = Field(description="what she believes about her subject that others don't say")
    signature_angle: str = Field(description="the recognisable way she does things: framing, ritual, phrase, look")
    distinct_from_siblings: str = Field(description="how she differs from the other creators in the portfolio")
    comparable_patterns: list[str] = Field(description="2-4 patterns of successful creators she resembles "
                                                       "(kinds of accounts, never named people)")


class AudienceInsight(BaseModel):
    who_exactly: str = Field(description="one specific person she talks to: age, situation, a day in their life")
    desires: list[str]
    frustrations: list[str]
    when_they_scroll: str
    what_they_save: list[str]
    what_they_share: list[str]
    words_they_use: list[str] = Field(description="phrases this audience actually uses")
    turn_offs: list[str] = Field(description="what makes them scroll past or unfollow")


class Story(BaseModel):
    backstory: str = Field(description="a few sentences of ordinary history that explains her taste and habits")
    values_in_action: list[str] = Field(description="what she does (not says) that shows what she values")
    quirks: list[str]
    routines: list[str] = Field(description="recurring routines content can come back to")
    small_goals: list[str] = Field(description="small, ongoing goals that give posts a thread to follow")
    recurring_people: list[str] = Field(description="fictional people she mentions (never shown faces, never other "
                                                    "accounts); may be empty")


class Mix(BaseModel):
    photo: float
    carousel: float
    story: float
    video: float


class Series(BaseModel):
    name: str
    premise: str
    archetype: str = Field(description="which of her archetypes it serves")
    capabilities: list[str] = Field(description="capability ids (group.name) from the list given that carry it")
    per_week: float
    hook: str = Field(description="how an episode opens")
    why_it_works: str = Field(description="what it gives the audience and adds to her brand")


class ContentSystem(BaseModel):
    mix: Mix = Field(description="share of posts by shape; sums to 1")
    talk_share: float = Field(description="share of videos where she talks to camera, 0-1")
    series: list[Series] = Field(description="3-5 recurring series")
    hooks: list[str] = Field(description="her go-to openings")
    never_makes: list[str] = Field(description="content shapes she never makes, and why")


class Presentation(BaseModel):
    register: str = Field(description="one register name from the list given (e.g. refined, sporty, glamorous, "
                                      "sultry_editorial)")
    attractiveness: str = Field(description="how her attractiveness shows: beauty, ruggedness, elegance, "
                                            "playfulness, mystery, natural sex appeal...")
    wardrobe_range: str = Field(description="what she wears across her life, swimwear, fitness and evening "
                                            "included where they fit her")
    photographic_style: str = Field(description="how she is photographed: poses, gaze, framing, mood")
    never: list[str] = Field(description="what her register never includes")


class BrandCore(BaseModel):
    why_exist: str = Field(description="why this talent exists: the role she plays in people's feeds and lives")
    desire: str = Field(description="the desire or aspiration she embodies, in a viewer's words")
    represents: str = Field(description="what she stands for, in one line")
    why_people_follow: str
    value_compounds: str = Field(description="how her value compounds over years: what accumulates (taste authority, "
                                             "a world, trust, a recognisable look)")
    values: list[str]
    ethos: str = Field(description="her worldview, in a sentence or two")
    aspirations: list[str] = Field(description="what she herself is reaching for")
    personality: list[str] = Field(description="the essence of her personality, a few traits")
    appearance_traits: list[str] = Field(description="the stable, desirable traits of her look and presence "
                                                     "(bearing, signature features, what makes her attractive)")
    audience_relationship: str = Field(description="who she is to her audience: muse, friend, guide, older brother...")
    narrative: str = Field(description="her fictional narrative as a talent: where she comes from, where she is headed")
    presentation: Presentation = Field(description="how her attractiveness is expressed: the register every post "
                                                   "stays within (owner 2026-11-25: per creator, not a global default)")


class Repertoire(BaseModel):
    name: str
    format: str = Field(description="one post format from the capability card")
    why: str = Field(description="what it shows of her and why it works")


class PersonalStyle(BaseModel):
    """Personal presentation and finer details as part of identity (owner 2026-11-25): a stylist's repertoire."""
    finish: str = Field(description="her level of refinement, glamour, ruggedness or ease: intentional, never sloppy")
    grooming: str = Field(description="grooming and beauty conventions (skin, brows, beard, scent)")
    hair_and_makeup: str
    nails: str = Field(description="nail care and styles, where relevant to her")
    jewelry_and_accessories: list[str] = Field(description="watch, chain, rings, earrings, necklace, belts, bags")
    eyewear: str
    footwear: str
    silhouettes_materials_colours: str
    possessions: list[str] = Field(description="recurring possessions that say who she is (a leather bag, a film "
                                               "camera, a fountain pen)")
    seasonal_and_occasion: str = Field(description="how she dresses by season, destination, activity and occasion")
    signature_elements: list[str] = Field(description="2-4 elements that recur subtly so she is recognised")
    varies: str = Field(description="what changes with outfit, season, destination and activity, as a stylist "
                                    "would choose, never repetition")


class BrandWorld(BaseModel):
    style: str = Field(description="her style essence (the style director owns the details)")
    environments: list[str] = Field(description="the kinds of places her world is made of")
    hobbies: list[str]
    routines: list[str]
    visual_language: str = Field(description="light, palette mood, framing and composition that make a post hers")
    repertoire: list[Repertoire] = Field(description="4-8 kinds of content she does best, on formats production "
                                                     "renders reliably")
    personal_style: PersonalStyle
    world_details: list[str] = Field(description="objects and surroundings that communicate her taste: the coffee "
                                                 "cup, linens, sunglasses on the table, books, flowers, furniture, "
                                                 "restaurant light")


class SignatureAssociation(BaseModel):
    element: str = Field(description="a recurring style element, e.g. vintage baseball hats, a signet ring")
    category: str = Field(description="the commercial category key it belongs to")
    how_it_recurs: str = Field(description="when it appears and when it doesn't (situational: never in every image, "
                                           "never at the pool or a formal event if it wouldn't be worn there)")


class OwnedBrandOption(BaseModel):
    category: str
    why: str = Field(description="why her audience might want this from her")
    evidence_needed: str = Field(description="what would have to be true first: affiliate demand, engagement on the "
                                             "association, partner interest")


class BrandCommerce(BaseModel):
    categories: list[str] = Field(description="2-5 natural commercial category keys from the registry given")
    why_categories: str
    commercial_thesis: str = Field(default="", description="why this personality is a valuable commercial asset: "
                                   "the audience, its buying power and intent, the adjacencies, in 2-3 sentences")
    signature_associations: list[SignatureAssociation] = Field(default_factory=list, max_length=4)
    advertiser_fit: list[str] = Field(default_factory=list, description="kinds of advertisers who would pay to reach "
                                      "her audience")
    monetization_path: str = Field(default="", description="the likely order: sponsorship, affiliate, licensing, "
                                   "then an owned product if evidence supports it")
    owned_brand_options: list[OwnedBrandOption] = Field(default_factory=list, max_length=3)
    portfolio_leverage: str = Field(default="", description="which of these categories could also travel across "
                                    "other personalities in the portfolio")
    product_integration: str = Field(description="how products would enter her world organically, openly virtual and "
                                                 "disclosed, never claiming first-hand experience")
    will_do: list[str]
    wont_do: list[str]
    will_wear: list[str]
    wont_wear: list[str]
    will_promote: list[str]
    wont_promote: list[str]


SCHEMAS = {"archetypes": ArchetypePick, "positioning": Positioning, "audience_insight": AudienceInsight,
           "story": Story, "content_system": ContentSystem, "brand_core": BrandCore, "brand_world": BrandWorld,
           "brand_commerce": BrandCommerce}

ASK = {
    "archetypes": "Choose her archetypes from the library: the one her content is mostly built on, and a secondary "
                  "only if a real part of her content needs a different playbook.",
    "positioning": "Write her positioning: why anyone follows her, the gap she fills, her point of view and signature "
                   "angle, and how she is clearly different from the sibling creators listed.",
    "audience_insight": "Describe the audience she is for, forensically: the one person, what they want, what they "
                        "save and share, the words they use, what makes them scroll past.",
    "story": "Write her story: enough ordinary history, values, quirks, routines and small goals that a writer can "
             "plan a year of posts that feel like the same life.",
    "content_system": "Design her content system from her archetypes' playbooks: the mix, how often she talks to "
                      "camera, 3-5 series built only from the capabilities listed, her hooks, and what she never "
                      "makes. Series serve her positioning and audience, not trends. Allocate the mix by expected "
                      "return per dollar using the capability card's cost and reliability: photo-led at launch, with "
                      "motion or talking only where her archetype needs it or evidence shows it buys more growth or "
                      "commercial value per dollar.",
    "brand_core": "Write the core of her Talent & Brand Bible: why she exists as talent, the desire she embodies, what "
                  "she represents, why people follow, how her value compounds over years, her values, ethos, "
                  "aspirations, personality, the desirable traits of her look, her relationship with her audience and "
                  "her narrative, and her presentation: the register (from the list given) her attractiveness is "
                  "expressed in, her wardrobe range and photographic style, in her own character, neither sterile "
                  "nor beyond her brand. This is her stable core: write what should still be true in three years.",
    "brand_world": "Write her brand world: style essence, the environments, hobbies and routines her life is made of, "
                   "her visual language, her creative repertoire (the content she does best, each on a format "
                   "production renders reliably: the capability card, photography first), her personal style as a "
                   "personal stylist would (grooming, hair and makeup, nails, jewelry and accessories, eyewear, "
                   "footwear, silhouettes, materials and colours, possessions, seasonal and occasion dress, a few "
                   "signature elements that recur subtly, the rest varied with judgment; rugged stays rugged, "
                   "effortless stays effortless, always intentional), and the details of her world that show taste.",
    "brand_commerce": "Write her commercial profile as a talent agency executive would: her commercial thesis (why "
                      "she is a valuable commercial asset), her natural commercial categories (keys from the "
                      "registry) and why they fit, 1-4 signature associations (style elements that recur naturally "
                      "and situationally and could become product affinities, never in every image, never as an "
                      "advert), the advertisers who would pay to reach her audience, her likely monetisation path, "
                      "up to 3 owned-product options with the evidence each would need first, which categories could "
                      "travel across the portfolio, how products enter her world organically, and what she will and "
                      "won't do, wear and promote. She is openly virtual: never claims first-hand experience of a "
                      "product. Her influence creates the opportunities; she never reads as an advertisement.",
}


def strategy_text(policy: Policy) -> str:
    """The portfolio thesis and desirability (objective, standard, context, the only restrictions, anti-drift), for
    prompts that shape a brand or a post."""
    s = policy.get("strategy", default={}) or {}
    e = s.get("economics") or {}
    econ = (f" How we create value: {str(e.get('value_chain') or '').strip()} {str(e.get('principles') or '').strip()}"
            if e else "")
    return f"The agency's thesis: {str(s.get('thesis') or '').strip()}{econ} {desirability_text(policy)}"


def desirability_text(policy: Policy) -> str:
    d = policy.get("strategy", "desirability", default={}) or {}
    regs = policy.get("strategy", "presentation", "registers", default={}) or {}
    return (f"Desirability: {str(d.get('objective') or '').strip()} Standard: {str(d.get('standard') or '').strip()} "
            f"{str(d.get('context') or '').strip()} Restricted, and only these: {'; '.join(d.get('restrictions') or [])}. "
            f"{str(d.get('anti_drift') or '').strip()} Presentation registers (name: sensuality level): "
            f"{json.dumps(regs)}.")


def register_level(policy: Policy, register: str | None) -> int | None:
    regs = policy.get("strategy", "presentation", "registers", default={}) or {}
    return int(regs[register]) if register in regs else None


def styling_line(prof: dict | None) -> str:
    """Her personal style for the director's wardrobe, look and props fields ('' without one): signature elements
    recur subtly; everything else follows the outfit, season, place and activity."""
    world = (prof or {}).get("brand_world") or {}
    ps = world.get("personal_style") or {}
    if not ps:
        return ""
    keep = {k: ps[k] for k in ("finish", "grooming", "hair_and_makeup", "nails", "jewelry_and_accessories",
                               "eyewear", "footwear", "signature_elements") if ps.get(k)}
    return (f"Her personal style (fill wardrobe.accessories, look and the set's props from it; her signature elements "
            f"recur subtly, the rest changes with outfit, season, place and activity, as a stylist would choose; "
            f"intentional, never sloppy or generic): {json.dumps(keep)}. Details of her world that show her taste: "
            f"{json.dumps(world.get('world_details') or [])}. ")


def presentation_line(prof: dict | None) -> str:
    """Her presentation register as a bound for planning prompts ('' without one)."""
    pres = ((prof or {}).get("brand_core") or {}).get("presentation") or {}
    if not pres.get("register"):
        return ""
    return (f"Her presentation register is {pres['register']} ({pres.get('attractiveness', '')}; wears: "
            f"{pres.get('wardrobe_range', '')}; photographed: {pres.get('photographic_style', '')}; never: "
            f"{pres.get('never')}). Every post stays within it and lives up to it: reach and engagement never justify "
            f"going past it, and don't sanitise below it either. ")


def registry_keys(policy: Policy) -> list[str]:
    """The commercial category keys (the registry's seed; research refreshes rows, never adds keys)."""
    return [c["key"] for c in policy.get("strategy", "commercial", "seed", default=[]) or [] if c.get("key")]


def card_formats(policy: Policy) -> list[str]:
    return list((policy.get("strategy", "capability_card", "formats", default={}) or {}).keys())


def missing(prof: dict | None) -> list[str]:
    return [s for s in SECTIONS if not (prof or {}).get(s)]


def _words(text: str) -> set[str]:
    stop = {"the", "and", "a", "an", "of", "to", "in", "her", "she", "for", "with", "on", "is", "that", "it", "at",
            "as", "by", "or", "but", "from", "not", "who", "what", "their", "they", "be", "are", "this", "into"}
    return {w for w in re.findall(r"[a-z']+", (text or "").lower()) if len(w) > 2 and w not in stop}


def fingerprint(prof: dict) -> set[str]:
    """The words that make a creator distinct: positioning, pillar and series names, archetypes."""
    pos = prof.get("positioning") or {}
    content = prof.get("content") or {}
    system = prof.get("content_system") or {}
    parts = [pos.get(k, "") for k in ("why_follow", "the_gap", "point_of_view", "signature_angle")]
    parts += [p.get("name", "") for p in content.get("pillars") or []]
    parts += [s.get("name", "") + " " + s.get("premise", "") for s in system.get("series") or []]
    parts += [str(prof.get("niche") or ""), str(prof.get("one_line") or "")]
    return _words(" ".join(parts))


def overlap(a: dict, b: dict) -> float:
    fa, fb = fingerprint(a), fingerprint(b)
    return round(len(fa & fb) / len(fa | fb), 3) if fa and fb else 0.0


def siblings(conn, creator_id) -> list[dict]:
    """The other portfolio creators' current profiles (sandbox clones and retired creators excluded)."""
    rows = conn.execute(
        """SELECT c.id, c.handle, p.body FROM creators c
           JOIN LATERAL (SELECT body FROM creator_profiles WHERE creator_id = c.id ORDER BY version DESC LIMIT 1) p ON true
           WHERE c.id <> %s AND c.status <> 'retired' AND c.anchors->'simulation' IS NULL""",
        (creator_id,)).fetchall()
    return [{"creator_id": str(r["id"]), "handle": r["handle"], "profile": r["body"]} for r in rows]


def differentiation(conn, policy: Policy, creator_id, prof: dict) -> dict:
    """The closest sibling by fingerprint overlap and whether it is past the policy bar."""
    bar = float(policy.get("brain", "talent_manager", "max_overlap", default=0.35))
    scored = sorted(((overlap(prof, s["profile"]), s["handle"]) for s in siblings(conn, creator_id)), reverse=True)
    top, handle = scored[0] if scored else (0.0, None)
    return {"closest": handle, "overlap": top, "max_overlap": bar, "too_close": top > bar}


def sibling_summary(conn, creator_id) -> list[dict]:
    return [{"handle": s["handle"], "archetypes": s["profile"].get("archetypes"),
             "positioning": (s["profile"].get("positioning") or {}).get("why_follow"),
             "niche": s["profile"].get("niche"), "one_line": s["profile"].get("one_line")}
            for s in siblings(conn, creator_id)]


def clean(policy: Policy, section: str, out: BaseModel, prof: dict) -> dict:
    """What a section saves, after code checks: archetypes are library names; series use capabilities that exist
    and render now; the mix is normalised; talk share is clamped."""
    d = out.model_dump()
    if section == "archetypes":
        lib = set(arch.names(policy))
        picked = [a for a in (d.get("primary"), d.get("secondary")) if a and a in lib]
        if not picked:
            raise ValueError(f"no known archetype in {d.get('primary')!r}/{d.get('secondary')!r}")
        return {"archetypes": list(dict.fromkeys(picked)), "archetype_why": d.get("why", "")}
    if section == "content_system":
        ok = set(direction.capability_ids(policy, available_only=True))
        own = set((prof.get("archetypes") or []))
        for s in d["series"]:
            s["capabilities"] = [c for c in s.get("capabilities") or [] if c in ok]
            if own and s.get("archetype") not in own:
                s["archetype"] = (prof.get("archetypes") or [None])[0]
            s["per_week"] = max(0.0, min(float(s.get("per_week") or 0), 7.0))
        mix = {k: max(float(v or 0), 0.0) for k, v in d["mix"].items()}
        total = sum(mix.values())
        d["mix"] = {k: round(v / total, 3) for k, v in mix.items()} if total else mix
        d["talk_share"] = max(0.0, min(float(d.get("talk_share") or 0), 1.0))
        return {"content_system": d}
    if section == "brand_core":
        reg = (d.get("presentation") or {}).get("register")
        if register_level(policy, reg) is None:
            known = sorted((policy.get("strategy", "presentation", "registers", default={}) or {}))
            raise ValueError(f"presentation register {reg!r} is not one of {known}")
        return {"brand_core": d}
    if section == "brand_world":
        formats = set(card_formats(policy))
        d["repertoire"] = [r for r in d.get("repertoire") or [] if not formats or r.get("format") in formats]
        return {"brand_world": d}
    if section == "brand_commerce":
        known = set(registry_keys(policy))
        d["categories"] = [c for c in dict.fromkeys(d.get("categories") or []) if not known or c in known]
        if known and not d["categories"]:
            raise ValueError(f"no known commercial category in {out.model_dump().get('categories')!r}")
        return {"brand_commerce": d}
    return {section: d}


def section_prompt(conn, policy: Policy, creator_id, section: str, prof: dict, *, extra: str = "") -> str:
    who = {k: v for k, v in prof.items() if k not in ("dials", "identity", "commercialization")}
    parts = [f"The creator's profile so far:\n{json.dumps(who, default=str)}"]
    if section == "archetypes":
        parts.append(f"The archetype library (name: what it is): {json.dumps(arch.catalogue(policy))}")
    else:
        parts.append(f"Her archetypes' playbooks: {json.dumps(arch.playbooks(policy, prof), default=str)}")
    if section in ("positioning", "content_system"):
        parts.append(f"Sibling creators in the portfolio (she must be clearly different): "
                     f"{json.dumps(sibling_summary(conn, creator_id), default=str)}")
    if section == "content_system":
        from ..brain.director import catalogue
        usable, _ = catalogue(policy)
        parts.append(f"Capabilities the studio renders now (use only these ids, group.name): "
                     f"{json.dumps({f'{g}.{n}': e.get('description', '') for g, es in usable.items() for n, e in es.items()})}")
    if section in BRAND_SECTIONS:
        parts.append(strategy_text(policy))
    if section in ("brand_world", "content_system"):
        from ..brain import capability_card
        parts.append(f"What production makes reliably and cheaply now, per post format (the capability card; design "
                     f"around the reliable ones):\n{capability_card.brief(capability_card.card(conn, policy))}")
        parts.append(f"Content guidance: {str(policy.get('strategy', 'content', default='')).strip()}")
    if section == "brand_commerce":
        from ..brain import commerce
        cfg = policy.get("strategy", "commercial", default={}) or {}
        parts.append(f"Categories her own style and world touch (adjacency): "
                     f"{json.dumps(commerce.adjacent(policy, [prof.get('brand_world'), prof.get('style')]))}")
        parts.append(f"The commercial category registry (keys, bands, index; estimates until researched):\n"
                     f"{commerce.brief(commerce.categories(conn, policy))}")
        parts.append(f"Now: {str(cfg.get('content_rule') or '').strip()} Later: {str(cfg.get('end_state') or '').strip()}")
    parts.append(f"Never touch: {policy.get('scope', 'excluded_topics')}.")
    parts.append(ASK[section] + (f"\n{extra}" if extra else ""))
    return "\n\n".join(parts)


def build_section(conn, policy: Policy, llm: StructuredLLM, creator_id, section: str, prof: dict, *,
                  extra: str = "") -> dict:
    """One section, cleaned (not saved): {section: value} (archetypes also gives archetype_why)."""
    out = llm.structured(policy.get("brain", "talent_manager", "role", default="talent_manager"),
                         system=profile_mod.persona_system(policy, SYSTEM),
                         content=section_prompt(conn, policy, creator_id, section, prof, extra=extra),
                         schema=SCHEMAS[section],
                         effort=policy.get("brain", "talent_manager", "effort", default="medium"),
                         creator_id=creator_id)
    return clean(policy, section, out, prof)


def checked_change(policy: Policy, prof: dict, path: str, value) -> dict:
    """A profile change inside a dossier section, put through the same checks as building it (schema, library
    archetypes, capabilities that render now, a normalised mix): {path: value} to save. Raises ValueError when the
    value doesn't fit. A path outside the dossier is returned as is."""
    section, _, rest = path.partition(".")
    if section not in SECTIONS:
        return {path: value}
    if section == "archetypes":
        lib = set(arch.names(policy))
        names = value if isinstance(value, list) else [value]
        if rest or not names or any(n not in lib for n in names):
            raise ValueError(f"archetypes must be a list of library names ({sorted(lib)})")
        return {"archetypes": list(dict.fromkeys(names))}
    current = dict(prof.get(section) or {})
    if rest:
        if "." in rest:
            raise ValueError("change a dossier field, not part of one")
        current[rest] = value
    elif isinstance(value, dict):
        current = value
    else:
        raise ValueError(f"{section} is a section: its value must be an object")
    try:
        model = SCHEMAS[section].model_validate(current)
    except Exception as e:  # noqa: BLE001 - pydantic's error, said in one line
        raise ValueError(f"{section} does not fit its schema: {str(e).splitlines()[0]}") from e
    return clean(policy, section, model, prof)


def planning_view(prof: dict | None) -> dict:
    """What the showrunner and director need of the dossier, compact."""
    if not prof:
        return {}
    view = {k: prof[k] for k in ("archetypes", "positioning", "content_system") if prof.get(k)}
    ai = prof.get("audience_insight") or {}
    if ai:
        view["audience_insight"] = {k: ai[k] for k in ("who_exactly", "desires", "what_they_save", "what_they_share",
                                                       "words_they_use", "turn_offs") if k in ai}
    st = prof.get("story") or {}
    if st:
        view["story"] = {k: st[k] for k in ("routines", "small_goals", "quirks") if k in st}
    view.update(brand_view(prof))
    return view


def brand_view(prof: dict | None) -> dict:
    """The brand bible as planners need it, compact: the core's essentials, her world, her commercial lines."""
    prof = prof or {}
    out = {}
    core = prof.get("brand_core") or {}
    if core:
        out["brand_core"] = {k: core[k] for k in ("desire", "represents", "values", "ethos", "appearance_traits",
                                                  "audience_relationship", "presentation") if core.get(k)}
    world = prof.get("brand_world") or {}
    if world:
        out["brand_world"] = {k: world[k] for k in ("environments", "hobbies", "routines", "visual_language",
                                                    "repertoire", "personal_style", "world_details") if world.get(k)}
    com = prof.get("brand_commerce") or {}
    if com:
        out["brand_commerce"] = {k: com[k] for k in ("categories", "wont_do", "wont_wear", "will_promote",
                                                     "wont_promote") if com.get(k)}
    return out
