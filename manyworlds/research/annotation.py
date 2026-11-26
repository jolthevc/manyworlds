"""The annotation schema for research reference posts: photo, carousel and video in one record (models only).

Three layers, as the brief asks, and three kinds of statement kept apart in every record:
  observed     what is in the frames, slides or words (coded values, few words)
  mechanism    a plausible reason it works, marked as such (short notes)
  speculation  guesses worth testing (short notes)
plus `evidence` (numbers and economics) and the code's own measure (manyworlds.research.video_measure), which no
model writes. Unknown is a value: nothing is invented to fill a field.

The content is described, not the person's private life: there is no field for health, relationships, family,
religion, politics, home address or other people in frame, and the vision read is told not to describe them. Named
public creators are fine (reference_creators); their comments and commenters are not collected.

Schema iia_v2 (2026-11-25, docs/research/2026-11-25-annotation-v2.md) adds `observed.coded` (Coded): every field a
controlled vocabulary read from policy `intelligence.annotation.vocab`, so distill can count and test it (v1's
pose, wardrobe, lighting and expression were free text no statistic could use). The read gets the lists in its prompt;
`coerce_coded` (with the run's vocab) turns any value outside a field's list into "unknown" instead of failing the
read. A v1 record still loads: its `coded` is all unknown. The read is blind to performance: the model sees media and
caption only; code attaches evidence afterwards.

The batcher that fills these is `iia_annotate` (Haiku, Message Batches; manyworlds.brain.agents.intelligence).
`ANNOTATION_SCHEMA` names the stored version (reference_annotations.schema); `READ_SCHEMAS` every version a post may
already carry (distill reads them all, the newest per post; retention counts any of them as read).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel as _PydanticModel, Field, ValidationInfo, field_validator, model_validator


class BaseModel(_PydanticModel):
    """Annotation models read back from the strict structured-output shape (llm.strict_json_schema: every field
    present, "" for an unset enum or string, 0 or [] for unset numbers and lists): a "" falls back to the field's
    default ("unknown", None or empty), so a strict answer validates like a loose one."""

    @model_validator(mode="before")
    @classmethod
    def _blank_is_default(cls, data):
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v != ""}
        return data

ANNOTATION_SCHEMA = "iia_v2"
READ_SCHEMAS = ("iia_v2", "iia_v1")    # newest first
NOTE_WORDS = 20            # a mechanism or speculation note: at most this many words, never a quote

Unknown = Literal["unknown"]
Situation = Literal["dinner", "bar_or_club", "street", "mirror", "bathroom", "bedroom", "kitchen", "pool_or_beach",
                    "car", "desk_or_work", "cafe", "event", "travel", "gym_or_sport", "nature", "studio", "other",
                    "unknown"]
Device = Literal["self_arm", "mirror", "friend", "propped", "pro", "pov", "screen", "unknown"]
Gaze = Literal["lens", "off", "down", "mixed", "hidden", "unknown"]
Imperfection = Literal["flash", "motion_blur", "off_centre_crop", "grain", "mess_in_frame", "none", "unknown"]
VideoFormat = Literal["grwm", "storytime_selfie", "vlog", "day_in_life", "outfit_check", "lifestyle_montage",
                      "conversational_selfie", "tutorial", "reaction", "skit", "other", "unknown"]
HookKind = Literal["text", "line", "text_and_line", "visual", "action", "question", "none", "unknown"]
Transition = Literal["cut", "jump", "match", "whip", "punch_in", "none", "unknown"]
Movement = Literal["still", "multitasking", "walking", "dancing", "spinning", "unknown"]
Relationship = Literal["friend", "big_sibling", "expert", "entertainer", "muse", "unknown"]
AudienceRole = Literal["attraction", "intimacy", "humour", "aspiration", "utility"]


class Visual(BaseModel):
    situation: Situation = "unknown"
    nearest_shot_type: str = Field(default="unknown", description="the closest aesthetics shot type, or new:<words>")
    pose: str = Field(default="unknown", description="a few words")
    hands: str = Field(default="unknown", description="what the hands do, a few words")
    gaze: Gaze = "unknown"
    framing: str = Field(default="unknown", description="close-up, medium, full body, wide")
    device: Device = Field(default="unknown", description="who or what took it")
    imperfections: list[Imperfection] = Field(default_factory=list)
    lighting: str = Field(default="unknown", description="source, quality, time of day")
    wardrobe: str = Field(default="unknown", description="silhouette, palette, formality; never brands")
    expression: str = Field(default="unknown")
    face_share: float | None = Field(default=None, description="share of frames or slides showing her face, 0-1")


class VideoPerformance(BaseModel):
    format: VideoFormat = "unknown"
    eye_contact: float | None = Field(default=None, description="share of frames looking into the lens, 0-1")
    gestures_per_10s: float | None = None
    movement: Movement = "unknown"
    mannerisms: list[str] = Field(default_factory=list, description="hair touch, laugh, aside to camera, pause...")
    humour: str = Field(default="unknown")
    delivery: str = Field(default="unknown", description="emotional delivery, a few words")


class VideoCraft(BaseModel):
    angle: str = Field(default="unknown")
    camera_movement: Literal["static", "handheld", "follow", "push", "mixed", "unknown"] = "unknown"
    transitions: list[Transition] = Field(default_factory=list)
    hook: HookKind = Field(default="unknown", description="what carries the first 2 seconds")
    broll_share: float | None = Field(default=None, description="share of frames that are cutaways, 0-1")
    broll_kinds: list[str] = Field(default_factory=list, description="hands, product, place, screen")
    burned_captions: bool | None = None
    sound: Literal["voice", "voice_and_music", "music", "room", "unknown"] = "unknown"
    structure: Literal["setup_turn_payoff", "list", "routine", "reveal", "loop", "other", "unknown"] = "unknown"


class Personality(BaseModel):
    tone: str = Field(default="unknown", description="tone of voice, a few words")
    humour: str = Field(default="unknown")
    disclosure: Literal["none", "everyday", "vulnerable", "unknown"] = "unknown"
    ritual: str = Field(default="", description="a recurring ritual or series, described generically; empty if none")
    relationship: Relationship = "unknown"
    aspirational_vs_relatable: int | None = Field(default=None, description="1 relatable ... 5 aspirational")
    parasocial_hooks: list[Literal["question", "callback", "you_address", "inside_joke", "update"]] = Field(
        default_factory=list)


class Economics(BaseModel):
    post_type: Literal["photo", "carousel", "video", "unknown"] = "unknown"
    sponsored: Literal["yes", "no", "unknown"] = "unknown"
    product_categories: list[str] = Field(default_factory=list, description="apparel, beauty, accessories...")
    capabilities_needed: list[str] = Field(default_factory=list, description="names from policy capabilities")
    not_renderable_yet: list[str] = Field(default_factory=list, description="what our pipeline can't make now")


# --- the coded section (iia_v2) -----------------------------------------------------------------------------------
# Field names and kinds are the schema (code); the values each field may take are policy
# (intelligence.annotation.vocab), so the lists change without a deploy.
ENUM_FIELDS = ("styling_category", "fit", "palette", "hair", "makeup", "nails", "jewelry_level", "pose",
               "body_language", "expression", "composition", "subject_scale", "camera_angle", "lighting_kind",
               "candid_level", "travel_context", "product_integration", "opener_has_face", "face_in_first_second",
               "opening", "movement_complexity", "sound", "production_complexity", "renderable_now")
MULTI_FIELDS = ("garments", "accessories", "environment", "lifestyle_narrative", "emotional_tone",
                "product_categories", "slide_roles")
ORDERED_FIELDS = ("slide_roles",)          # one label per slide, in order: positions matter
NUMBER_FIELDS = ("aspiration", "setting_changes")
ASPIRATION_SCALE = (1, 5)                  # the field's own scale: 1 relatable ... 5 aspirational
CAROUSEL_FIELDS = ("slide_roles", "opener_has_face")
VIDEO_FIELDS = ("face_in_first_second", "opening", "setting_changes", "movement_complexity", "sound",
                "production_complexity", "renderable_now")
CODED_FIELDS = ENUM_FIELDS + MULTI_FIELDS + NUMBER_FIELDS
UNKNOWN = "unknown"


def applies(field: str, medium: str | None) -> bool:
    """Whether a coded field is asked of a medium: carousel fields only of carousels, video fields only of videos."""
    if field in CAROUSEL_FIELDS:
        return medium == "carousel"
    if field in VIDEO_FIELDS:
        return medium == "video"
    return True


def vocab(policy) -> dict[str, list[str]]:
    """The coded vocabularies from policy intelligence.annotation.vocab, every value as a normalised name (a YAML
    yes/no read as a boolean comes back as "yes"/"no")."""
    raw = (policy.get("intelligence", "annotation", "vocab", default={}) if policy is not None else {}) or {}
    return {str(k): [_norm(v) for v in (vals or [])] for k, vals in raw.items()}


def _norm(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    return "_".join(str(v).strip().lower().replace("-", " ").split())


def _coerce(data: dict, vocab_: dict) -> tuple[dict, list[str]]:
    """The coded fields of `data` checked against the vocabularies: a value outside its field's list (or a field with
    no list) becomes "unknown"; multi-label lists keep their in-list labels once each (slide_roles keeps every slide
    in order, an unlisted one as "unknown"). Returns (data, the "field=value" misses)."""
    out, misses = dict(data), []
    for f in ENUM_FIELDS:
        if f not in out or out[f] in (None, ""):
            continue
        v = _norm(out[f])
        if v != UNKNOWN and v not in (vocab_.get(f) or []):
            misses.append(f"{f}={v}"[:80])
            v = UNKNOWN
        out[f] = v
    for f in MULTI_FIELDS:
        if f not in out or out[f] in (None, ""):
            continue
        vals = out[f] if isinstance(out[f], list) else [out[f]]
        labels = []
        for x in vals:
            v = _norm(x)
            if v != UNKNOWN and v not in (vocab_.get(f) or []):
                misses.append(f"{f}={v}"[:80])
                v = UNKNOWN
            if f in ORDERED_FIELDS or v not in labels:
                labels.append(v)
        out[f] = labels if f in ORDERED_FIELDS else [v for v in labels if v != UNKNOWN] or (
            [UNKNOWN] if labels else [])
    return out, misses


class Coded(BaseModel):
    """Controlled-vocabulary codes for the whole post (a carousel's slides together, a video's frames together). Each
    single-value field is one name from its list in the prompt, else "unknown"; each list field names every label
    that applies (slide_roles: one per slide, in order). Never brands, never private life, never other people:
    `friends` in slide_roles only says a slide shows others, nothing about them."""
    # fashion
    styling_category: str = Field(default=UNKNOWN, description="one name from its list")
    garments: list[str] = Field(default_factory=list, description="every visible garment, names from its list")
    fit: str = UNKNOWN
    palette: str = Field(default=UNKNOWN, description="the dominant colour story")
    # grooming
    hair: str = UNKNOWN
    makeup: str = UNKNOWN
    nails: str = UNKNOWN
    accessories: list[str] = Field(default_factory=list)
    jewelry_level: str = UNKNOWN
    # body and face
    pose: str = UNKNOWN
    body_language: str = UNKNOWN
    expression: str = UNKNOWN
    # composition and camera
    composition: str = UNKNOWN
    subject_scale: str = UNKNOWN
    camera_angle: str = UNKNOWN
    lighting_kind: str = UNKNOWN
    candid_level: str = Field(default=UNKNOWN, description="candid, semi_candid, posed or editorial")
    # place and story
    environment: list[str] = Field(default_factory=list)
    travel_context: str = UNKNOWN
    lifestyle_narrative: list[str] = Field(default_factory=list)
    emotional_tone: list[str] = Field(default_factory=list)
    aspiration: int | None = Field(default=None, description="1 relatable ... 5 aspirational; null if unsure")
    # commerce
    product_integration: str = Field(default=UNKNOWN, description="none, incidental, featured or ad")
    product_categories: list[str] = Field(default_factory=list, description="categories only, never brands")
    # carousel only
    slide_roles: list[str] = Field(default_factory=list, description="carousel: one role per slide, in order")
    opener_has_face: str = Field(default=UNKNOWN, description="carousel: yes or no")
    # video only
    face_in_first_second: str = Field(default=UNKNOWN, description="video: yes or no")
    opening: str = Field(default=UNKNOWN, description="video: what the first second is (the hook kind)")
    setting_changes: int | None = Field(default=None, description="video: how many times the place changes")
    movement_complexity: str = UNKNOWN
    sound: str = UNKNOWN
    production_complexity: str = UNKNOWN
    renderable_now: str = Field(default=UNKNOWN, description="video: could the listed capabilities make it now: "
                                                             "yes, partly or no")

    @model_validator(mode="before")
    @classmethod
    def _against_vocab(cls, data, info: ValidationInfo):
        """With a vocabulary in the validation context ({"vocab": ...}), values outside it become "unknown"."""
        voc = (info.context or {}).get("vocab") if info is not None else None
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k in cls.model_fields}    # unasked keys are dropped, not fatal
            for f in MULTI_FIELDS:
                if isinstance(data.get(f), str) and data[f]:
                    data[f] = [data[f]]
            if voc is not None:
                data, _ = _coerce(data, voc)
        return data

    @field_validator("aspiration", mode="before")
    @classmethod
    def _scale(cls, v):
        try:
            n = int(round(float(v)))
        except (TypeError, ValueError):
            return None
        return n if ASPIRATION_SCALE[0] <= n <= ASPIRATION_SCALE[1] else None

    @field_validator("setting_changes", mode="before")
    @classmethod
    def _count(cls, v):
        try:
            n = int(round(float(v)))
        except (TypeError, ValueError):
            return None
        return n if n >= 0 else None


def coerce_coded(raw: dict | None, vocab_: dict) -> tuple[Coded, list[str]]:
    """A coded section validated against `vocab_` (out-of-list values -> "unknown"), and the values that were out of
    list ("field=value"), for the vision benchmark's in-vocabulary rate."""
    data = {k: v for k, v in (raw or {}).items() if k in Coded.model_fields and v != ""}
    for f in MULTI_FIELDS:
        if isinstance(data.get(f), str) and data[f]:
            data[f] = [data[f]]
    data, misses = _coerce(data, vocab_)
    return Coded.model_validate(data), misses


def load(observed: dict | None, vocab_: dict | None = None) -> "Observed":
    """A stored `observed` (any schema in READ_SCHEMAS) as the current model: a v1 record has no coded section and
    comes back with every code unknown. With `vocab_`, coded values outside it become "unknown"."""
    return Observed.model_validate(observed or {}, context={"vocab": vocab_} if vocab_ is not None else None)


class Observed(BaseModel):
    coded: Coded = Field(default_factory=Coded)       # iia_v2: controlled vocabularies (policy)
    visual: Visual = Field(default_factory=Visual)
    performance: VideoPerformance | None = None      # videos only
    craft: VideoCraft | None = None                  # videos only
    personality: Personality = Field(default_factory=Personality)
    economics: Economics = Field(default_factory=Economics)


class Note(BaseModel):
    text: str = Field(description=f"at most {NOTE_WORDS} words, in your own words, never a quote")
    audience_role: AudienceRole | None = None


class Evidence(BaseModel):
    """Numbers, filled by code from the post's public counts and its creator's baseline; never by a model."""
    age_days: float | None = None
    views: int | None = None
    likes: int | None = None
    comments: int | None = None
    shares: int | None = None
    saves: int | None = None
    over_own_median: float | None = Field(default=None, description="the post's engagement over its creator's median")
    over_band_median: float | None = Field(default=None, description="over the niche's follower-band median")
    missing: list[str] = Field(default_factory=list, description="what the platform doesn't show (reach, follows)")


class Annotation(BaseModel):
    """One post's record (reference_annotations, schema iia_v2; v1 records load too). The model fills observed,
    mechanism, speculation and confidence; code fills evidence and measured."""
    observed: Observed = Field(default_factory=Observed)
    mechanism: list[Note] = Field(default_factory=list, max_length=3)
    speculation: list[Note] = Field(default_factory=list, max_length=3)
    confidence: float = Field(default=0.5, ge=0, le=1, description="how sure the read is, 0-1")


def clip_note(text: str, words: int = NOTE_WORDS) -> str:
    """A note cut to `words` words: notes are summaries, never long quotes from a post."""
    return " ".join(str(text or "").split()[:words])


def vocab_prompt(policy, fields: tuple[str, ...] | None = None) -> str:
    """The coded vocabularies as prompt text, one field a line (`field (one|list|ordered list): a, b, c`), with the
    rules for filling them. Deterministic for a policy version, so it caches as part of a system prompt."""
    voc = vocab(policy)
    lines = []
    for f in fields or CODED_FIELDS:
        if f == "aspiration":
            lines.append(f"aspiration (number): {ASPIRATION_SCALE[0]} relatable ... {ASPIRATION_SCALE[1]} aspirational,"
                         " null if unsure")
        elif f == "setting_changes":
            lines.append("setting_changes (number, video only): how many times the place changes, null if unsure")
        else:
            kind = "ordered list, one per slide" if f in ORDERED_FIELDS else "list" if f in MULTI_FIELDS else "one"
            where = " (carousel only)" if f in CAROUSEL_FIELDS else " (video only)" if f in VIDEO_FIELDS else ""
            lines.append(f"{f} ({kind}){where}: {', '.join(voc.get(f) or [])}")
    return ("Coded section: answer every field only with names from its list below, exactly as written. When "
            "nothing fits or you cannot tell, use \"unknown\" (an empty list for list fields); a field marked carousel "
            "or video only is \"unknown\" for other posts. Describe what is visible, never brands, never private life, "
            "never other people in frame.\n" + "\n".join(lines))


def bands(policy) -> dict[str, dict[str, list]]:
    """Number bands for testing and comparing coded numbers (policy intelligence.annotation.bands)."""
    return (policy.get("intelligence", "annotation", "bands", default={}) if policy is not None else {}) or {}


def band(field: str, value, bands_: dict) -> str | None:
    """The band name a coded number falls in, None when it is unset or in no band."""
    if value is None:
        return None
    for name, (lo, hi) in (bands_.get(field) or {}).items():
        if float(lo) <= float(value) <= float(hi):
            return str(name)
    return None


def coded_attributes(coded: dict | None, medium: str | None, bands_: dict) -> set[str]:
    """A post's coded values as testable attributes: `field=value` for single values, one `field=label` per label for
    lists, `field=<band>` for numbers; unknowns, empty lists and fields the medium is not asked are left out."""
    out: set[str] = set()
    c = coded or {}
    for f in CODED_FIELDS:
        if not applies(f, medium):
            continue
        v = c.get(f)
        if f in NUMBER_FIELDS:
            b = band(f, v, bands_)
            if b:
                out.add(f"{f}={b}")
        elif f in MULTI_FIELDS:
            out |= {f"{f}={x}" for x in (v or []) if x not in (None, "", UNKNOWN)}
        elif v not in (None, "", UNKNOWN):
            out.add(f"{f}={v}")
    return out
