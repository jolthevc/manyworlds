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

The batcher that fills these with `iia_annotate` (Haiku, Message Batches) is a later step (spec section 4, task 7).
`ANNOTATION_SCHEMA` names the stored version (reference_annotations.schema).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel as _PydanticModel, Field, model_validator


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

ANNOTATION_SCHEMA = "iia_v1"
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


class Observed(BaseModel):
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
    """One post's record (reference_annotations, schema iia_v1). The model fills observed, mechanism, speculation
    and confidence; code fills evidence and measured."""
    observed: Observed = Field(default_factory=Observed)
    mechanism: list[Note] = Field(default_factory=list, max_length=3)
    speculation: list[Note] = Field(default_factory=list, max_length=3)
    confidence: float = Field(default=0.5, ge=0, le=1, description="how sure the read is, 0-1")


def clip_note(text: str, words: int = NOTE_WORDS) -> str:
    """A note cut to `words` words: notes are summaries, never long quotes from a post."""
    return " ".join(str(text or "").split()[:words])
