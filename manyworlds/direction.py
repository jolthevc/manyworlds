"""Direction: the shot sheet for one scene, and the compiler that turns it into each model's prompt.

A shot sheet is structured data on the post (`body.shots`, one per scene). Every field is optional: a missing
field falls back to the creator's or the format's default, and a post without shots keeps today's prompts
(creator.lab builds those itself). The compiler writes sections in a fixed order, so models stay swappable:

    camera/lens -> subject (the full appearance) -> wardrobe and look -> action/blocking -> set and lighting
    -> style -> sound cues (only for targets in `direction.sound_targets`) -> positive stand-ins

Negatives never go in the positive prompt: a negation in a prompt with no negative channel ("no mirrors") tends to
summon the thing. `compile_prompts` returns {prompt, negative}: a target whose model has a negative input
(`direction.negatives.channel`: the Wan graphs' negative text encode) gets them there, after the target's base list;
one without (Z-Image Turbo at guidance 0) gets only the positive stand-ins of `direction.positive_instead_of`
("unbranded clothes" for "logos"), and negatives without a stand-in are dropped. `compile_prompt` is the prompt alone.

In video with the creator in shot (talk, and i2v unless the shot is a cutaway), the action also gets natural eye
behaviour (blinks, glances tied to the beats whose eyeline leaves the lens) and an idle/micro-expression line from
her energy or body language, both worded in policy `direction.eyes` and `direction.idle`.

Each target has a token budget (policy `direction.prompt_budget`, counted by `estimate_tokens`): the text encoders
read 512 tokens and drop the rest, so a longer prompt is trimmed lowest priority first (`_fit`) while camera, action
and the positive stand-ins stay whole. A prompt within its budget is never touched.

Words, defaults and the allowed camera moves come from policy `direction`. Text never goes on a generated
surface: `edit.on_screen_text` is a list of overlay items (label, hook, timestamp, question; manyworlds.overlays draws
them in post; an older sheet's plain string reads as one label) and never reaches an image or video prompt.

The rest of the edit is read by the finish: `edit.caption_style` (manyworlds.captions.caption_style_for),
`edit.transition_in` mapped to the joiner's kinds by `transition_kind`, and `motion_reference`, a licensed
`dance.references` clip whose motion the shot performs (`motion_reference` keeps only listed ids).

What the pipeline can render is the capability catalogue (policy `capabilities`, `capabilities()`): devices, framing,
camera moves, actions, edits, text and sound, each with how it is rendered, the switches it needs and its limits.
The director composes shots from it (manyworlds.brain.director); formats are containers, not recipes.
"""

import math
import re
import types
import typing
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Target = Literal["photo", "i2v", "talk"]


def _union_args(annotation) -> tuple:
    origin = typing.get_origin(annotation)
    return typing.get_args(annotation) if origin in (typing.Union, types.UnionType) else (annotation,)


class _Sheet(BaseModel):
    """A part of a shot sheet. Read back from a strict schema (manyworlds.llm.strict_json_schema: every field
    required, no nulls, "" added to every enum for "unset"), an empty string means "not set": it becomes None
    where the field may be None, and is ignored (the default stays) where the field is a number or a bool."""

    @model_validator(mode="before")
    @classmethod
    def _unset_empty_strings(cls, data):
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for name, field in cls.model_fields.items():
            if name not in out or out[name] != "":
                continue
            args = _union_args(field.annotation)
            if type(None) in args:
                out[name] = None
            elif any(a in (int, float, bool) for a in args):
                del out[name]
        return out


class SetSpec(_Sheet):
    location: str = Field(default="", description="where the scene happens, concretely: 'her small galley kitchen', "
                                                  "'a bus stop on a wet side street'")
    set_id: str = Field(default="", description="a short name for a recurring set, reused across posts "
                                                "(e.g. 'home_kitchen'); empty for a one-off place")
    time_of_day: str = Field(default="", description="e.g. early morning, golden hour, late night")
    season: str = Field(default="", description="e.g. late autumn; empty if it does not show")
    weather: str = Field(default="", description="e.g. light drizzle, overcast, hot and bright; empty indoors")
    light_direction: str = Field(default="", description="where the light comes from: 'window on her left', "
                                                         "'overhead', 'behind her'")
    light_quality: str = Field(default="", description="soft, hard, diffused, dappled, mixed")
    color_temp: str = Field(default="", description="warm, neutral, cool, or a mix such as 'warm lamp, cool window'")
    practical_lights: str = Field(default="", description="lights visible in the frame: fairy lights, a desk lamp, "
                                                          "a fridge glow; empty if none")
    set_dressing: str = Field(default="", description="what the place looks like around her: clutter, furniture, "
                                                      "plants; plain, unbranded surfaces, in positive words")
    props: str = Field(default="", description="objects she uses or holds, if any: plain and unbranded, "
                                               "in positive words")
    background_action: str = Field(default="", description="what moves behind her: traffic, steam from a pot; "
                                                           "empty if still")
    background_people: int = Field(default=0, description="how many other people are visible in the background; "
                                                          "0 for nobody (the usual)")
    depth_of_field: str = Field(default="", description="shallow (soft background) or deep (everything sharp)")


class WardrobeSpec(_Sheet):
    outfit_id: str = Field(default="", description="a short name for a recurring outfit, reused across posts "
                                                   "(e.g. 'grey_hoodie'); empty for a one-off")
    outfit: str = Field(default="", description="what she wears, visually: colours, fabrics, fit; plain, unbranded")
    accessories: str = Field(default="", description="extra accessories for this scene, beyond her signature ones")
    state: str = Field(default="", description="the clothes' condition: crisp, creased, rain-damp, sleeves rolled up")


class LookSpec(_Sheet):
    hair_style: str = Field(default="", description="only if it differs from her usual hair: e.g. 'in a low bun'")
    hair_state: str = Field(default="", description="e.g. freshly washed, windswept, a bit messy")
    makeup: str = Field(default="", description="e.g. none, light everyday, a bold lip")
    skin_state: str = Field(default="", description="e.g. a little flushed, post-run sheen, tired under the eyes")


class FramingSpec(_Sheet):
    shot_size: str = Field(default="", description="close-up, medium close-up, medium shot, full body, wide")
    angle: str = Field(default="", description="eye level, slightly high, low angle, over the shoulder")
    lens_mm: float | None = Field(default=None, description="equivalent focal length in mm; a phone's main "
                                                            "camera is about 26, selfie camera about 24")
    aspect: str = Field(default="", description="frame shape, e.g. 9:16 or 4:5; empty for the format's default")
    camera_move: str = Field(default="", description="one of: static, handheld, push_in, pull_out, pan_left, "
                                                     "pan_right, tilt_up, tilt_down, follow, orbit_slow; "
                                                     "empty for none (photos ignore it)")
    handheld: Literal["none", "light", "held"] | None = Field(
        default=None, description="camera shake: none (propped or tripod), light (a little sway), held "
                                  "(clearly in someone's hand)")
    device: Literal["propped", "held", "selfie", "mirror", "pov"] | None = Field(
        default=None, description="how the phone films her: propped (set down on something, locked off), held "
                                  "(someone else holds it, also following her), selfie (she holds it at arm's "
                                  "length), mirror (she films her reflection, phone in hand), pov (what she sees, "
                                  "phone at her eye level, only her hands in frame)")
    headroom: str = Field(default="", description="space above her head: tight, normal, generous")


class Beat(_Sheet):
    at_s: float = Field(default=0.0, description="seconds from the start of the shot when this beat begins")
    action: str = Field(default="", description="what she does, in a few plain words: 'sips her coffee', "
                                                "'looks up and laughs'")
    eyeline: Literal["lens", "off", "prop", "screen"] | None = Field(
        default=None, description="where she looks: lens (into the camera), off (an off-camera person or thing), "
                                  "prop (the object she is using), screen (a phone or laptop screen)")
    hands: str = Field(default="", description="what her hands do; relaxed or out of frame unless the action "
                                               "needs them")


class Line(_Sheet):
    text: str = Field(default="", description="exactly what she says, as spoken; no stage directions")
    delivery: str = Field(default="", description="how she says it, never spoken aloud: 'dry, under her breath', "
                                                  "'excited, fast'")
    pause_after_s: float | None = Field(default=None, description="silence after the line in seconds; empty for "
                                                                  "the default pause")


class PerformanceSpec(_Sheet):
    emotion_arc: str = Field(default="", description="how her mood moves through the shot: 'bored to amused'")
    energy: Literal["low", "medium", "high"] | None = Field(default=None, description="overall energy: low, "
                                                                                       "medium or high")
    body_language: str = Field(default="", description="posture and gestures: 'slouched on the sofa', "
                                                       "'leaning in, animated hands'")
    lines: list[Line] = Field(default_factory=list, description="the spoken lines of this shot, in order; empty "
                                                                "for a silent shot")


class SoundSpec(_Sheet):
    mic: Literal["phone", "lav", "room"] | None = Field(default=None, description="how it sounds recorded: phone "
                                                                                  "(built-in mic), lav (clip-on), "
                                                                                  "room (distant, roomy)")
    room_tone: str = Field(default="", description="the background bed: 'quiet flat, fridge hum', 'busy street'")
    effects: list[str] = Field(default_factory=list, description="sounds that happen in the shot: 'kettle click'")
    music: str = Field(default="", description="music in the scene or under it; empty for none")


TEXT_KINDS = ("label", "hook", "timestamp", "question")
TEXT_POSITIONS = ("top_bubble", "upper_third", "upper_left", "centre", "lower")


class OnScreenText(_Sheet):
    """One piece of on-screen text drawn in post (manyworlds.overlays): a label, the hook, a timestamp or a
    question bubble. Times are seconds from the start of the shot."""
    kind: Literal["label", "hook", "timestamp", "question"] = Field(
        default="label", description="label (a short line: a tip, a numbered look), hook (big text in the first "
                                     "second), timestamp ('7:30 AM coffee'), question (a white bubble at the top, "
                                     "like a reply to a comment)")
    text: str = Field(default="", description="the words, short, in the creator's voice; never a real username")
    start_s: float = Field(default=0.0, description="when it appears, seconds from the start of the shot")
    end_s: float | None = Field(default=None, description="when it goes, seconds from the start of the shot; 0 for "
                                                          "the end of the shot")
    style: str = Field(default="", description="an on-screen text style name (listed in the brief), or empty for "
                                               "the kind's default")
    position: Literal["top_bubble", "upper_third", "upper_left", "centre", "lower", ""] = Field(
        default="", description="where it sits, or empty for the kind's default; never over her face")
    number: int | None = Field(default=None, description="for a numbered look or tip, its number; 0 for none")
    time: str = Field(default="", description="for a timestamp whose text does not start with the time, the time "
                                              "('7:30 AM'); else empty")

    @model_validator(mode="before")
    @classmethod
    def _unset_zeros(cls, data):
        """The strict schema's unset values: an empty kind is a label; a number of 0 is none; an end at 0 (or
        not after the start) is the end of the shot."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if out.get("kind") in ("", None):
            out["kind"] = "label"
        if out.get("number") in ("", 0):
            out["number"] = None
        try:
            start = float(out.get("start_s") or 0.0)
        except (TypeError, ValueError):
            start = 0.0
        end = out.get("end_s")
        if end in ("", None):
            out["end_s"] = None
        else:
            try:
                out["end_s"] = float(end) if float(end) > start else None
            except (TypeError, ValueError):
                out["end_s"] = None
        return out


class EditSpec(_Sheet):
    hook: str = Field(default="", description="what grabs attention in the first second")
    on_screen_text: list[OnScreenText] = Field(
        default_factory=list, description="text drawn in post, in time order; the only place text may appear, "
                                          "never on surfaces in the image; empty for none")
    caption_style: str = Field(default="", description="the spoken-word captions' style: a caption style name "
                                                       "(listed in the brief), 'none' for no word captions (labels "
                                                       "only), or empty for the creator's default")
    transition_in: str = Field(default="", description="how this shot starts after the previous one: cut, match "
                                                       "(cut inside a motion both shots share: a spin, a hair flip, "
                                                       "a hand over the lens), whip, blur or flash")
    cover: bool = Field(default=False, description="true if this shot's frame should be the post's cover")

    @field_validator("on_screen_text", mode="before")
    @classmethod
    def _plain_text_is_one_label(cls, value):
        """The older sheets' plain string: one label for the whole shot (blank: none)."""
        if value is None:
            return []
        if isinstance(value, str):
            return [{"kind": "label", "text": value.strip()}] if value.strip() else []
        return value


class ContinuitySpec(_Sheet):
    carry_over: str = Field(default="", description="what must match the previous shot: 'same mug, half full', "
                                                    "'hair still wet'")
    story_time: str = Field(default="", description="when this happens in the story: 'ten minutes later', "
                                                    "'the next morning'")


class ShotSheet(_Sheet):
    set: SetSpec = Field(default_factory=SetSpec, description="the place and its light")
    wardrobe: WardrobeSpec = Field(default_factory=WardrobeSpec, description="what she wears")
    look: LookSpec = Field(default_factory=LookSpec, description="hair, makeup and skin for this scene")
    framing: FramingSpec = Field(default_factory=FramingSpec, description="shot size, angle, lens, camera")
    blocking: list[Beat] = Field(default_factory=list, description="what she does, beat by beat, in time order")
    performance: PerformanceSpec = Field(default_factory=PerformanceSpec, description="mood, energy and her lines")
    sound: SoundSpec = Field(default_factory=SoundSpec, description="how the shot sounds")
    edit: EditSpec = Field(default_factory=EditSpec, description="hook, overlay text and the cut into this shot")
    continuity: ContinuitySpec = Field(default_factory=ContinuitySpec, description="what carries over between shots")
    cutaway: bool = Field(default=False, description="true for a shot that shows something else (b-roll) while "
                                                     "her voice continues over it")
    motion_reference: str = Field(default="", description="the id of a licensed motion reference clip (listed in "
                                                          "the brief) whose movement this shot performs: a dance, "
                                                          "a pose sequence, a twirl; empty for none")
    shot_type: str = Field(default="", description="the shot type from her shot vocabulary (the brief), e.g. "
                                                   "walk_away_sunset or ringed_hand_glass; empty for a plain shot of her")
    # Filled by code from shot_type and her style (manyworlds.brain.director), never asked of the model:
    subject: str = Field(default="", description="what of her is in frame (aesthetics subjects); '' or 'face' shows "
                                                 "her face")
    grade: str = Field(default="", description="her grade family's prompt words")


# What the director answers (manyworlds.brain.director). It is asked with a strict schema (`strict_required`,
# manyworlds.llm.strict_json_schema): ShotSheet's own schema, with its optional fields, compiles to a grammar too
# large for structured outputs (HTTP 400, measured 2026-11-22). The docstring is the schema's description.
class DirectedShots(BaseModel):
    """One shot sheet per scene of an accepted post idea, in the scenes' order."""
    strict_required: ClassVar[bool] = True
    # Each shot's `edit` (text on screen, cuts, caption style) is a second, small call (DirectedEdits): with it the
    # grammar is too large again (HTTP 400, measured 2026-11-23: any list nested in a shot, even of strings, failed).
    strict_omit: ClassVar[tuple] = (("shots", "edit"), ("shots", "subject"), ("shots", "grade"))
    shots: list[ShotSheet] = Field(description="one shot sheet per scene, in the same order as the scenes")
    move_shot: int = Field(default=-1, description="the index (from 0) of the shot that performs the move; "
                                                   "-1 when there is no move")


class DirectedEdits(BaseModel):
    """The edit of each shot, in the shots' order: the on-screen text, the cut into it, the caption style."""
    strict_required: ClassVar[bool] = True
    edits: list[EditSpec] = Field(description="one edit per shot, in the same order as the shots")


def _join(parts, sep: str = ", ") -> str:
    return sep.join(p.strip() for p in parts if p and str(p).strip())


def _coerce(shot: "ShotSheet | dict | None") -> ShotSheet:
    if shot is None:
        return ShotSheet()
    if isinstance(shot, ShotSheet):
        return shot
    return ShotSheet.model_validate(shot)


def faceless(shot: "ShotSheet | dict | None") -> bool:
    """A shot with no face of hers in it: her point of view (device pov: only her hands) or a cutaway (b-roll of
    what she talks about). Its prompt describes no face and no eyes, and no gate judges it on identity."""
    if not shot:
        return False
    try:
        s = _coerce(shot)
    except Exception:  # noqa: BLE001 - a malformed sheet is judged as before
        return False
    return s.framing.device == "pov" or bool(s.cutaway) or (s.subject or "") not in ("", "face")


def _hands_subject(bible: dict) -> str:
    """The subject of a faceless shot: her hands, with what makes them hers (skin, signature details)."""
    a = bible.get("appearance") or {}
    return _join(["seen from her own eyes, her hands and forearms in frame", str(a.get("ethnicity_and_skin") or ""),
                  str(a.get("signature_details") or "")])


def _faceless_subject(s: "ShotSheet", bible: dict, policy: Any) -> str:
    """A shot without her face: an aesthetic shot's subject (her back, hands, reflection, a small figure, or no one;
    manyworlds.brain.aesthetics), else her own point of view (her hands)."""
    if (s.subject or "") not in ("", "face"):
        from .brain import aesthetics
        a = bible.get("appearance") or {}
        hands = _join([str(a.get("ethnicity_and_skin") or ""), str(a.get("signature_details") or "")])
        return aesthetics.subject_text(policy if hasattr(policy, "get") and not isinstance(policy, dict) else None,
                                       s.subject, hands)
    return _hands_subject(bible)


def _lighting(s: SetSpec) -> str:
    light = _join([s.light_quality, s.color_temp])
    light = f"{light} light" if light else ""
    if s.light_direction:
        light = _join([light, f"from {s.light_direction}" if light else f"light from {s.light_direction}"], " ")
    return light


def summary(shot: "ShotSheet | dict | None") -> str:
    """The one-line `scenes` entry for a shot: location, framing, first action, wardrobe, light."""
    s = _coerce(shot)
    first = s.blocking[0].action if s.blocking else ""
    return _join([_join([s.set.location, s.set.time_of_day]), _join([s.framing.shot_size, s.framing.angle]),
                  first, f"wearing {s.wardrobe.outfit}" if s.wardrobe.outfit.strip() else "", _lighting(s.set)])


_PAREN = re.compile(r"\([^)]*\)")


def spoken_lines(shot: "ShotSheet | dict | None") -> list[dict]:
    """The shot's lines that say something, in order: {text (as `spoken` reads it), delivery, pause_after_s}.
    What the designed voice speaks (production, voice.speak), line by line with its delivery."""
    s = _coerce(shot)
    out = []
    for line in s.performance.lines:
        text = " ".join(_PAREN.sub(" ", line.text).split())
        if text:
            out.append({"text": text, "delivery": line.delivery.strip(), "pause_after_s": line.pause_after_s})
    return out


def spoken(shot: "ShotSheet | dict | None") -> str:
    """What is actually said in the shot: the lines' text in order. Deliveries are never part of it, and a
    parenthetical left in a line's text is a stage direction, so it is removed too."""
    return " ".join(line["text"] for line in spoken_lines(shot))


def _policy(policy: Any, *path: str, default: Any = None) -> Any:
    if hasattr(policy, "get") and not isinstance(policy, dict):
        return policy.get(*path, default=default)
    node: Any = policy or {}
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def camera_move(shot: "ShotSheet | dict | None", policy: Any) -> str:
    """The shot's camera move if policy `direction.camera_moves` allows it, else '' (unknown moves are dropped)."""
    move = re.sub(r"[\s\-]+", "_", _coerce(shot).framing.camera_move.strip().lower())
    allowed = _policy(policy, "direction", "camera_moves", default=[]) or []
    return move if move and move in allowed else ""


def overlay_text(shot: "ShotSheet | dict | None") -> str:
    """Everything the shot's on-screen text says, in order (what the voice rules read)."""
    return " ".join(t.text.strip() for t in _coerce(shot).edit.on_screen_text if t.text.strip())


# The joiner's transitions (manyworlds.media.TRANSITIONS); a test keeps the two the same.
TRANSITION_KINDS = ("cut", "match", "whip", "blur", "flash")


def transition_kind(text: str, policy: Any) -> str:
    """The joiner's transition for a shot's free-text `edit.transition_in`: a kind's own name, else the first kind
    in policy `direction.transitions.keywords` (in order) one of whose words starts a word in the text ('whip pan'
    -> whip, 'cut on the spin' -> match); anything else, or nothing, is a cut."""
    low = " ".join(str(text or "").lower().replace("_", " ").replace("-", " ").split())
    if not low:
        return "cut"
    if low in TRANSITION_KINDS:
        return low
    keywords = _policy(policy, "direction", "transitions", "keywords", default={}) or {}
    for kind, words in keywords.items():
        if kind in TRANSITION_KINDS and _mentions(low, words):
            return kind
    return "cut"


def motion_reference_ids(policy: Any) -> list[str]:
    """The motion reference clips a shot may name (policy `dance.references` ids), [] while `dance.enabled` is off."""
    if not _policy(policy, "dance", "enabled", default=False):
        return []
    return [str(r["id"]) for r in (_policy(policy, "dance", "references", default=[]) or [])
            if isinstance(r, dict) and r.get("id")]


def motion_reference(shot: "ShotSheet | dict | None", policy: Any) -> str:
    """The shot's motion reference if it is one of `motion_reference_ids`, else '' (unknown ids are dropped)."""
    ref = _coerce(shot).motion_reference.strip()
    return ref if ref and ref in motion_reference_ids(policy) else ""


# --- the capability catalogue (policy `capabilities`) -------------------------------------------------------------

def _switch_on(policy: Any, switch: str) -> bool:
    """One `requires` entry: 'a.b.c' is on when that policy value is truthy; 'a.b:x,y' when that list (or value)
    contains any of x, y."""
    path, _, wanted = str(switch).partition(":")
    value = _policy(policy, *[p for p in path.strip().split(".") if p], default=None)
    if not wanted:
        return bool(value)
    options = {w.strip() for w in wanted.split(",") if w.strip()}
    have = {str(v) for v in value} if isinstance(value, (list, tuple, set)) else {str(value)} if value else set()
    return bool(options & have)


def capability_available(entry: dict, policy: Any) -> bool:
    """Whether the pipeline can render a capability now: every switch in its `requires` is on."""
    return all(_switch_on(policy, s) for s in (entry or {}).get("requires") or [])


def capabilities(policy: Any) -> dict[str, dict[str, dict]]:
    """The catalogue (policy `capabilities`: group -> name -> {description, how, requires, limits, write}), each entry
    with `available` set from its `requires`."""
    out: dict[str, dict[str, dict]] = {}
    for group, entries in (_policy(policy, "capabilities", default={}) or {}).items():
        if not isinstance(entries, dict):
            continue
        out[group] = {name: {**e, "available": capability_available(e, policy)}
                      for name, e in entries.items() if isinstance(e, dict)}
    return out


def capability_ids(policy: Any, *, available_only: bool = False) -> list[str]:
    """Every catalogue entry as 'group.name' (only those the pipeline can render now, with `available_only`)."""
    return [f"{g}.{n}" for g, entries in capabilities(policy).items() for n, e in entries.items()
            if e["available"] or not available_only]


_EYELINE = {"lens": "looking into the lens", "off": "looking off-camera", "prop": "looking at the object in hand",
            "screen": "looking at a screen"}
_DEVICE = {"propped": "phone propped up", "held": "phone held by someone else",
           "selfie": "selfie, phone at arm's length",
           "mirror": "filmed in a mirror, she holds the phone at chest height",
           "pov": "point of view from her eyes, only her hands in frame"}
_MIC = {"phone": "phone microphone", "lav": "clip-on microphone", "room": "distant room sound"}


def _beat(b: Beat, with_time: bool) -> str:
    text = _join([b.action, _EYELINE.get(b.eyeline or "", ""), f"hands {b.hands}" if b.hands.strip() else ""])
    if with_time and text:
        return f"at {b.at_s:g} s: {text}"
    return text


def _sentence(label: str, body: str) -> str:
    return f"{label}: {body}." if body else ""


_GLANCING = ("off", "prop", "screen")


def _eyes(s: ShotSheet, target: Target, policy: Any) -> str:
    """Natural eye behaviour for a video with the creator in shot (not a photo, not a cutaway): the target's
    policy line (`direction.eyes.<target>`), then one glance per beat whose eyeline leaves the lens, at its time
    ("looks at the mug at 4s"). A prop glance names the shot's first prop when it has one."""
    if target == "photo" or faceless(s):
        return ""
    eyes = _policy(policy, "direction", "eyes", default={}) or {}
    template = eyes.get("glance", "") or ""
    names = eyes.get("glance_targets", {}) or {}
    prop = s.set.props.split(",")[0].strip()
    glances = []
    if template:
        for b in sorted(s.blocking, key=lambda b: b.at_s):
            if b.eyeline not in _GLANCING:
                continue
            what = prop if b.eyeline == "prop" and prop else names.get(b.eyeline, "")
            if what:
                glances.append(template.format(target=what, t=f"{b.at_s:g}"))
    return _join([eyes.get(target, "") or "", *glances])


def _idle(s: ShotSheet, target: Target, policy: Any) -> str:
    """The micro-expression and idle-movement line for a video with the creator in shot, from her energy; with
    no energy set, from the first `direction.idle.keywords` energy whose word appears in her body language; else
    the target's `idle.default_energy` (talk only, as shipped)."""
    if target == "photo" or faceless(s):
        return ""
    idle = _policy(policy, "direction", "idle", default={}) or {}
    energy = s.performance.energy or ""
    if not energy:
        body = s.performance.body_language.lower()
        for level, words in (idle.get("keywords") or {}).items():
            # a word start: 'slouch' matches 'slouched', 'fast' does not match 'breakfast'
            if body and any(re.search(rf"\b{re.escape(str(w).lower())}", body) for w in (words or [])):
                energy = level
                break
    if not energy:
        energy = (idle.get("default_energy") or {}).get(target, "") or ""
    if not energy:
        return ""
    return (idle.get("by_energy") or {}).get(energy, "") or ""


def _shot_size_key(size: str) -> str:
    return re.sub(r"[\s_\-]+", "", size.strip().lower())


def talk_profile_for(shot: "ShotSheet | dict | None", policy: Any) -> str | None:
    """The talk settings profile (policy generation.talk_profiles) a shot should render with, or None for the
    default `generation.talk`. While `direction.hd_close_ups` is on, a shot framed at one of
    `direction.hd_for_shot_sizes` (spelling-tolerant: 'close up', 'Close-Up', 'closeup') gets
    `direction.hd_profile` (720p), when that profile exists. Off as shipped until 720p is benchmarked."""
    if shot is None or not _policy(policy, "direction", "hd_close_ups", default=False):
        return None
    size = _shot_size_key(_coerce(shot).framing.shot_size)
    sizes = {_shot_size_key(str(x)) for x in (_policy(policy, "direction", "hd_for_shot_sizes", default=[]) or [])}
    if not size or size not in sizes:
        return None
    name = _policy(policy, "direction", "hd_profile", default="hd720") or "hd720"
    profiles = _policy(policy, "generation", "talk_profiles", default={}) or {}
    return name if name in profiles else None


def estimate_tokens(text: str) -> int:
    """A rough token count for the prompt encoders, without loading a tokenizer (no new dependency): the larger of
    words x TOKENS_PER_WORD and characters / CHARS_PER_TOKEN, rounded up. English prose runs about 1.3 tokens per
    word in umt5's SentencePiece vocabulary (Wan 2.1/2.2) and Qwen3's BPE (Z-Image Turbo); text dense with commas,
    numbers and short words ('26 mm lens', 'at 2.5 s:') runs nearer one token per 3.6 characters. Taking the
    larger leans high, which is the safe side for a budget."""
    return math.ceil(max(len(text.split()) * TOKENS_PER_WORD, len(text) / CHARS_PER_TOKEN))


TOKENS_PER_WORD = 1.35
CHARS_PER_TOKEN = 3.6

# Trimming a directed prompt over its budget (policy direction.prompt_budget), in this order, stopping as soon as
# it fits: shorten long free text to its first clause, then (video only) the short appearance, then drop optional
# fields, then cut the lowest-priority section that is left at a word boundary. Camera, action and the stand-ins are
# never trimmed. These are the compiler's priorities, not tunables: they follow what each field is for.
_SHORTEN = (("set", "set_dressing"), ("set", "props"), ("wardrobe", "outfit"))
_DROP = (("set", "background_action"), ("set", "practical_lights"), ("wardrobe", "accessories"),
         ("wardrobe", "state"), ("set", "season"))
_TRUNCATE = ("sound", "style", "set", "wardrobe", "subject")          # lowest priority first
_LABELS = {"camera": "Camera", "subject": "Subject", "wardrobe": "Wardrobe and look", "action": "Action",
           "set": "Set", "style": "Style", "sound": "Sound"}
_DANGLING = frozenset({"a", "an", "the", "and", "or", "with", "of", "to", "in", "on", "at", "by", "for", "from"})
_CLAUSE_END = re.compile(r"[.;:!?](?=\s|$)|\s[—–-]\s")


def _first_clause(text: str, max_words: int) -> str:
    """The first clause of a long free-text field: up to its first sentence end (. ; : ! ? or a spaced dash), then
    whole comma items while they fit in `max_words` (the first item always), then a word cut if that one item is
    still longer."""
    text = " ".join(text.split())
    m = _CLAUSE_END.search(text)
    first = (text[:m.start()] if m else text).strip().rstrip(",")
    if len(first.split()) <= max_words:
        return first
    kept: list[str] = []
    for item in (x.strip() for x in first.split(",")):
        if not item:
            continue
        if kept and len(", ".join([*kept, item]).split()) > max_words:
            break
        kept.append(item)
    words = ", ".join(kept).split()
    return " ".join(words[:max_words]).rstrip(",;:")


def _wardrobe(s: ShotSheet) -> str:
    w, lk = s.wardrobe, s.look
    return _join([f"wearing {w.outfit}" if w.outfit.strip() else "", w.accessories, w.state,
                  f"hair {lk.hair_style}" if lk.hair_style.strip() else "",
                  f"hair {lk.hair_state}" if lk.hair_state.strip() else "",
                  f"makeup {lk.makeup}" if lk.makeup.strip() else "", lk.skin_state])


def feasible_props(props: str, policy: Any = None) -> str:
    """The shot's props cut to policy `direction.feasibility.max_props` (comma-separated, first ones kept): an image
    model draws one or two named objects reliably; a fifth one is where scenes failed review (2026-11-22: 341 of
    596 real-post images missed their scene, mostly on props). Free, and checked before any GPU time."""
    cap = (_policy(policy, "direction", "feasibility", default={}) or {}).get("max_props") if policy is not None else None
    if not cap:
        return props
    parts = [p.strip() for p in props.split(",") if p.strip()]
    return ", ".join(parts[:int(cap)])


def _set(s: ShotSheet, video: bool, scene: str, policy: Any = None) -> str:
    st = s.set
    people = (f"{st.background_people} {'person' if st.background_people == 1 else 'people'} in the background"
              if st.background_people > 0 else "")
    set_ = _join([st.location, st.time_of_day, st.season, st.weather, _lighting(st), st.practical_lights,
                  st.set_dressing, feasible_props(st.props, policy), st.background_action if video else "", people,
                  f"{st.depth_of_field} depth of field" if st.depth_of_field.strip() else "",
                  f"same as before: {s.continuity.carry_over}" if s.continuity.carry_over.strip() else ""])
    if not st.location.strip() and scene.strip():
        set_ = _join([scene.strip(), set_])
    return set_


def _render(head: str, sections: dict[str, str], tail: str) -> str:
    parts = [_sentence(_LABELS[k], v) for k, v in sections.items()]
    return head + " ".join(x for x in [*parts, tail] if x)


def _fit(s: ShotSheet, sections: dict[str, str], *, head: str, tail: str, budget: int, video: bool,
         scene: str, short_subject: str, policy: Any, subject_floor: str = "") -> str:
    """Trim a directed prompt into `budget` estimated tokens (see _SHORTEN, _DROP, _TRUNCATE). Camera, action and the
    tail (the positive stand-ins) are kept whole; if they alone are over budget, the prompt is returned with
    everything else cut. `subject_floor` (a photo of her face: her appearance without its signature details) is
    where the subject may be cut to before the set and the wardrobe are: perf_compare 2026-11-25 lost both
    sections whole to a long signature item (a cardigan she wasn't wearing), and eight of nine photos left the set."""
    fits = lambda: estimate_tokens(_render(head, sections, tail)) <= budget  # noqa: E731
    if fits():
        return _render(head, sections, tail)
    t = s.model_copy(deep=True)

    def refresh():
        sections["wardrobe"], sections["set"] = _wardrobe(t), _set(t, video, scene)

    words = int(_policy(policy, "direction", "prompt_trim", "clause_words", default=20) or 20)
    for part, field in _SHORTEN:                                      # (1) long free text to its first clause
        spec = getattr(t, part)
        setattr(spec, field, _first_clause(getattr(spec, field), words))
        refresh()
        if fits():
            return _render(head, sections, tail)
    if video and short_subject:                                       # (2) the start photo carries identity
        sections["subject"] = short_subject
        if fits():
            return _render(head, sections, tail)
    for part, field in _DROP:                                         # (3) optional detail
        setattr(getattr(t, part), field, "")
        refresh()
        if fits():
            return _render(head, sections, tail)
    order = list(_TRUNCATE)
    if subject_floor and "set" in order:   # her signature details go before the set and the wardrobe
        order.insert(order.index("set"), "signature")
    for key in order:                                                 # (4) cut at a word boundary, lowest first
        floor = 0
        if key == "signature":
            key, floor = "subject", len(subject_floor.split())
        kept = sections.get(key, "").split()
        while len(kept) > floor and not fits():
            kept.pop()
            while kept and kept[-1].lower() in _DANGLING:              # no cut ending on 'with the'
                kept.pop()
            sections[key] = " ".join(kept).rstrip(",;:.")
        if fits():
            break
    return _render(head, {k: v for k, v in sections.items() if v}, tail)


def _shake(s: ShotSheet, policy: Any) -> str:
    """How the phone moves in a video (policy `direction.realism.shake`): the shot's `framing.handheld`, else what
    its `framing.device` implies (`realism.device_shake`: a propped phone is locked off, a selfie sways a little).
    A gimbal-smooth "phone" is one of the strongest AI tells, so a held phone always says it is held."""
    realism = _policy(policy, "direction", "realism", default={}) or {}
    level = s.framing.handheld or (realism.get("device_shake") or {}).get(s.framing.device or "", "") or ""
    return (realism.get("shake") or {}).get(level, "") or ""


def _style(s: ShotSheet, target: Target, policy: Any) -> str:
    """The target's phone-native realism words (`direction.realism.<target>`), the light fallback when the shot
    says nothing about its light (`realism.light_fallback`: motivated light, never unmotivated studio light), then
    the target's style words."""
    realism = _policy(policy, "direction", "realism", default={}) or {}
    style_words = _policy(policy, "direction", "style", default={}) or {}
    st = s.set
    unlit = not any(x.strip() for x in (st.light_direction, st.light_quality, st.color_temp, st.practical_lights))
    return _join([realism.get(target, "") or "", (realism.get("light_fallback", "") or "") if unlit else "",
                  style_words.get(target, "") or "", s.grade or ""])


def _mentions(text: str, words) -> bool:
    """Whether any of `words` starts a word in the text ('mirror' matches 'mirrors' and 'mirrored')."""
    low = text.lower()
    return any(re.search(rf"\b{re.escape(str(w).lower())}", low) for w in (words or []) if str(w).strip())


def negative_items(shot: "ShotSheet | dict | None", target: Target, policy: Any, *, positive: str = "") -> list[str]:
    """What the shot must not show, for one target (policy `direction.negatives`): `always`, then `solo` when nobody
    is asked for in the background, then `by_target.<target>`, each once. An item yields to the shot itself: one
    whose `unless_mentioned` words appear in the positive prompt (a mirror shot asks for the mirror) is left out."""
    s = _coerce(shot)
    neg = _policy(policy, "direction", "negatives", default={}) or {}
    items = list(neg.get("always") or [])
    if s.set.background_people <= 0:
        items += list(neg.get("solo") or [])
    items += list((neg.get("by_target") or {}).get(target) or [])
    unless = neg.get("unless_mentioned") or {}
    out, seen = [], set()
    for item in (str(x).strip() for x in items):
        if not item or item.lower() in seen or (positive and _mentions(positive, unless.get(item))):
            continue
        seen.add(item.lower())
        out.append(item)
    return out


def has_negative_channel(target: Target, policy: Any) -> bool:
    """Whether the target's model reads a negative prompt (policy `direction.negatives.channel`): the Wan graphs do
    (i2v, talk: a negative text encode); Z-Image Turbo at guidance 0 does not (photo)."""
    return bool((_policy(policy, "direction", "negatives", "channel", default={}) or {}).get(target))


def _negative_text(items: list[str], target: Target, policy: Any) -> str:
    """The negative prompt for a target with a channel: its `negatives.base` (Wan's standard list, minus what fights
    the phone-native look), then the shot's items."""
    base = (_policy(policy, "direction", "negatives", "base", default={}) or {}).get(target, "") or ""
    return _join([base, *items])


def _stand_ins(items: list[str], policy: Any) -> str:
    """For a target without a negative channel: the positive phrasing of each item (`direction.positive_instead_of`),
    once each, as one sentence; an item without one is dropped (a negation would summon it)."""
    instead = _policy(policy, "direction", "positive_instead_of", default={}) or {}
    phrases: list[str] = []
    for item in items:
        phrase = str(instead.get(item) or "").strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return _capitalised(_join(phrases))


def compile_prompts(shot: "ShotSheet | dict | None", bible: dict, target: Target, policy: Any, *,
                    trigger: str | None = None, scene: str = "", motion: str = "") -> dict[str, str]:
    """{prompt, negative} for one shot on one target model (photo, image-to-video, or lip-synced talk).

    `scene` and `motion` are the post's plain-text scene and movement, used only where the shot leaves the
    set or the blocking empty; `trigger` is the creator's LoRA trigger word (photos).

    The prompt never carries a negation. `negative` is for the job payload's `negative_prompt` (the Wan graphs'
    negative text encode); it is "" for a target without a negative channel, whose prompt gets the positive
    stand-ins instead (see the module docstring).

    A prompt over the target's policy `direction.prompt_budget` (estimated tokens, `estimate_tokens`) is trimmed
    (`_fit`): the text encoders read 512 tokens (Wan's umt5-xxl; Z-Image's Qwen3 at diffusers' default
    max_sequence_length) and silently drop the rest, which is where the set and style are. A prompt within budget
    is exactly as compiled."""
    from .creator.lab import appearance_prompt, short_appearance

    s = _coerce(shot)
    video = target != "photo"
    f = s.framing

    medium = "candid smartphone photo" if target == "photo" else "smartphone video"
    lens = f"{f.lens_mm:g} mm lens" if f.lens_mm else ""
    move = camera_move(s, policy).replace("_", " ") if video else ""
    camera = _join([medium, f.shot_size, f.angle, lens, f"{f.headroom} headroom" if f.headroom.strip() else "",
                    _DEVICE.get(f.device or "", ""), _shake(s, policy) if video else "",
                    f"camera {move}" if move else ""])

    wardrobe = _wardrobe(s)

    p = s.performance
    if target == "photo":
        beats = _beat(s.blocking[0], False) if s.blocking else ""       # a photo is one moment: the first beat
    else:
        beats = _join([_beat(b, True) for b in sorted(s.blocking, key=lambda b: b.at_s)], "; then ")
    if not beats and target == "i2v" and motion.strip():
        beats = motion.strip()
    if target == "talk":
        beats = _join(["talking to the camera naturally" if not beats else "talking", beats], "; ")
    perf = _join([f"{p.energy} energy" if p.energy else "", p.body_language,
                  f"mood {p.emotion_arc}" if video and p.emotion_arc.strip() else ""])
    # eyes and idle movement only in video with the creator in shot; both empty leave the action as it was
    action = _join([_join([beats, perf]), _eyes(s, target, policy), _idle(s, target, policy)], "; ")

    set_ = _set(s, video, scene, policy)
    style = _style(s, target, policy)

    # Sound is for the mix (the shot sheet keeps it): the video models cannot render it, so by default no target's
    # prompt budget is spent on it (policy direction.sound_targets).
    sound = ""
    if target in (_policy(policy, "direction", "sound_targets", default=[]) or []):
        snd = s.sound
        sound = _join([_MIC.get(snd.mic or "", ""), f"room tone {snd.room_tone}" if snd.room_tone.strip() else "",
                       *snd.effects, f"music {snd.music}" if snd.music.strip() else ""])

    head = f"{trigger}, " if trigger else ""
    no_face = faceless(s)
    sections = {"camera": camera, "subject": _faceless_subject(s, bible, policy) if no_face else appearance_prompt(bible),
                "wardrobe": wardrobe, "action": action,
                "set": set_, "style": style, "sound": sound}
    items = negative_items(s, target, policy, positive=_render(head, sections, ""))
    channel = has_negative_channel(target, policy)
    tail = "" if channel else _stand_ins(items, policy)
    negative = _negative_text(items, target, policy) if channel else ""
    budget = (_policy(policy, "direction", "prompt_budget", default={}) or {}).get(target)
    if not budget:
        return {"prompt": _render(head, sections, tail), "negative": negative}
    prompt = _fit(s, sections, head=head, tail=tail, budget=int(budget), video=video, scene=scene,
                  short_subject=short_appearance(bible) if video and not no_face else "", policy=policy,
                  subject_floor=_appearance_core(bible) if not video and not no_face else "")
    return {"prompt": prompt, "negative": negative}


def compile_prompt(shot: "ShotSheet | dict | None", bible: dict, target: Target, policy: Any, *,
                   trigger: str | None = None, scene: str = "", motion: str = "") -> str:
    """The positive prompt alone (compile_prompts' `prompt`), for callers with no negative channel to fill."""
    return compile_prompts(shot, bible, target, policy, trigger=trigger, scene=scene, motion=motion)["prompt"]


def _capitalised(text: str) -> str:
    return f"{text[0].upper()}{text[1:]}." if text else ""


def _appearance_core(bible: dict) -> str:
    """Her full appearance without its signature details (the part of a photo's subject a set outranks)."""
    from .creator.lab import appearance_prompt
    a = dict(bible.get("appearance") or {})
    if not str(a.get("signature_details") or "").strip():
        return ""
    return appearance_prompt({**bible, "appearance": {**a, "signature_details": ""}}).rstrip(", ")
