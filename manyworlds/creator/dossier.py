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

Each section is its own small structured call (cheap, retried alone, versioned as its own profile change), in
order, each seeing the ones before. Code checks what a model can't be trusted with: archetype names exist,
capabilities exist and render now, the mix sums to 1, and positioning overlaps no sibling past policy
`brain.talent_manager.max_overlap` (word overlap, deterministic).
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

SECTIONS = ("archetypes", "positioning", "audience_insight", "story", "content_system")

SYSTEM = """You are the talent manager of a Manyworlds creator: an original fictional social media creator, openly
labeled as AI. You own their career the way the best human talent managers do: identity first, then audience, then
content. You are forensic and specific: every line must be something a writer, a director and a stylist can act on,
and that a real audience would recognise. Never generic ("relatable", "authentic", "engaging" say nothing). Never
base anything on a real person or existing creator; describe patterns, not people. Never touch excluded topics."""


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


SCHEMAS = {"archetypes": ArchetypePick, "positioning": Positioning, "audience_insight": AudienceInsight,
           "story": Story, "content_system": ContentSystem}

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
                      "makes. Series serve her positioning and audience, not trends.",
}


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
    return view
