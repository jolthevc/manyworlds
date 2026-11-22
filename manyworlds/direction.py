"""Direction: the shot sheet for one scene, and the compiler that turns it into each model's prompt.

A shot sheet is structured data on the post (`body.shots`, one per scene). Every field is optional: a missing
field falls back to the creator's or the format's default, and a post without shots keeps today's prompts
(creator.lab builds those itself). The compiler writes sections in a fixed order, so models stay swappable:

    camera/lens -> subject (the full appearance) -> wardrobe and look -> action/blocking -> set and lighting
    -> style -> sound cues (talk only) -> negatives

Words, defaults and the allowed camera moves come from policy `direction`. Text never goes on a generated
surface: `edit.on_screen_text` is an overlay added in post and never reaches an image or video prompt.
"""

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

Target = Literal["photo", "i2v", "talk"]


class SetSpec(BaseModel):
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
                                                      "plants; no legible signs or labels")
    props: str = Field(default="", description="objects she uses or holds, if any; no readable text or logos")
    background_action: str = Field(default="", description="what moves behind her: traffic, steam from a pot; "
                                                           "empty if still")
    background_people: int = Field(default=0, description="how many other people are visible in the background; "
                                                          "0 for nobody (the usual)")
    depth_of_field: str = Field(default="", description="shallow (soft background) or deep (everything sharp)")


class WardrobeSpec(BaseModel):
    outfit_id: str = Field(default="", description="a short name for a recurring outfit, reused across posts "
                                                   "(e.g. 'grey_hoodie'); empty for a one-off")
    outfit: str = Field(default="", description="what she wears, visually: colours, fabrics, fit; no logos")
    accessories: str = Field(default="", description="extra accessories for this scene, beyond her signature ones")
    state: str = Field(default="", description="the clothes' condition: crisp, creased, rain-damp, sleeves rolled up")


class LookSpec(BaseModel):
    hair_style: str = Field(default="", description="only if it differs from her usual hair: e.g. 'in a low bun'")
    hair_state: str = Field(default="", description="e.g. freshly washed, windswept, a bit messy")
    makeup: str = Field(default="", description="e.g. none, light everyday, a bold lip")
    skin_state: str = Field(default="", description="e.g. a little flushed, post-run sheen, tired under the eyes")


class FramingSpec(BaseModel):
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
    device: Literal["propped", "held", "selfie"] | None = Field(
        default=None, description="how the phone films her: propped (set down on something), held (someone "
                                  "else holds it), selfie (she holds it at arm's length)")
    headroom: str = Field(default="", description="space above her head: tight, normal, generous")


class Beat(BaseModel):
    at_s: float = Field(default=0.0, description="seconds from the start of the shot when this beat begins")
    action: str = Field(default="", description="what she does, in a few plain words: 'sips her coffee', "
                                                "'looks up and laughs'")
    eyeline: Literal["lens", "off", "prop", "screen"] | None = Field(
        default=None, description="where she looks: lens (into the camera), off (an off-camera person or thing), "
                                  "prop (the object she is using), screen (a phone or laptop screen)")
    hands: str = Field(default="", description="what her hands do; relaxed or out of frame unless the action "
                                               "needs them")


class Line(BaseModel):
    text: str = Field(default="", description="exactly what she says, as spoken; no stage directions")
    delivery: str = Field(default="", description="how she says it, never spoken aloud: 'dry, under her breath', "
                                                  "'excited, fast'")
    pause_after_s: float | None = Field(default=None, description="silence after the line in seconds; empty for "
                                                                  "the default pause")


class PerformanceSpec(BaseModel):
    emotion_arc: str = Field(default="", description="how her mood moves through the shot: 'bored to amused'")
    energy: Literal["low", "medium", "high"] | None = Field(default=None, description="overall energy: low, "
                                                                                       "medium or high")
    body_language: str = Field(default="", description="posture and gestures: 'slouched on the sofa', "
                                                       "'leaning in, animated hands'")
    lines: list[Line] = Field(default_factory=list, description="the spoken lines of this shot, in order; empty "
                                                                "for a silent shot")


class SoundSpec(BaseModel):
    mic: Literal["phone", "lav", "room"] | None = Field(default=None, description="how it sounds recorded: phone "
                                                                                  "(built-in mic), lav (clip-on), "
                                                                                  "room (distant, roomy)")
    room_tone: str = Field(default="", description="the background bed: 'quiet flat, fridge hum', 'busy street'")
    effects: list[str] = Field(default_factory=list, description="sounds that happen in the shot: 'kettle click'")
    music: str = Field(default="", description="music in the scene or under it; empty for none")


class EditSpec(BaseModel):
    hook: str = Field(default="", description="what grabs attention in the first second")
    on_screen_text: str = Field(default="", description="text overlaid in post; the only place text may appear, "
                                                        "never on surfaces in the image")
    caption_style: str = Field(default="", description="how the overlay text looks: e.g. 'small lowercase, "
                                                       "bottom third'")
    transition_in: str = Field(default="", description="how this shot starts after the previous one: hard cut, "
                                                       "match cut, whip")
    cover: bool = Field(default=False, description="true if this shot's frame should be the post's cover")


class ContinuitySpec(BaseModel):
    carry_over: str = Field(default="", description="what must match the previous shot: 'same mug, half full', "
                                                    "'hair still wet'")
    story_time: str = Field(default="", description="when this happens in the story: 'ten minutes later', "
                                                    "'the next morning'")


class ShotSheet(BaseModel):
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


def _join(parts, sep: str = ", ") -> str:
    return sep.join(p.strip() for p in parts if p and str(p).strip())


def _coerce(shot: "ShotSheet | dict | None") -> ShotSheet:
    if shot is None:
        return ShotSheet()
    if isinstance(shot, ShotSheet):
        return shot
    return ShotSheet.model_validate(shot)


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


def spoken(shot: "ShotSheet | dict | None") -> str:
    """What is actually said in the shot: the lines' text in order. Deliveries are never part of it, and a
    parenthetical left in a line's text is a stage direction, so it is removed too."""
    s = _coerce(shot)
    texts = (" ".join(_PAREN.sub(" ", line.text).split()) for line in s.performance.lines)
    return " ".join(t for t in texts if t)


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


_EYELINE = {"lens": "looking into the lens", "off": "looking off-camera", "prop": "looking at the object in hand",
            "screen": "looking at a screen"}
_DEVICE = {"propped": "phone propped up", "held": "phone held by someone else",
           "selfie": "selfie, phone at arm's length"}
_HANDHELD = {"none": "steady, no camera shake", "light": "lightly handheld", "held": "handheld"}
_MIC = {"phone": "phone microphone", "lav": "clip-on microphone", "room": "distant room sound"}


def _beat(b: Beat, with_time: bool) -> str:
    text = _join([b.action, _EYELINE.get(b.eyeline or "", ""), f"hands {b.hands}" if b.hands.strip() else ""])
    if with_time and text:
        return f"at {b.at_s:g} s: {text}"
    return text


def _sentence(label: str, body: str) -> str:
    return f"{label}: {body}." if body else ""


def compile_prompt(shot: "ShotSheet | dict | None", bible: dict, target: Target, policy: Any, *,
                   trigger: str | None = None, scene: str = "", motion: str = "") -> str:
    """The prompt for one shot on one target model (photo, image-to-video, or lip-synced talk).

    `scene` and `motion` are the post's plain-text scene and movement, used only where the shot leaves the
    set or the blocking empty; `trigger` is the creator's LoRA trigger word (photos)."""
    from .creator.lab import appearance_prompt

    s = _coerce(shot)
    video = target != "photo"
    f = s.framing

    medium = "candid smartphone photo" if target == "photo" else "smartphone video"
    lens = f"{f.lens_mm:g} mm lens" if f.lens_mm else ""
    move = camera_move(s, policy).replace("_", " ") if video else ""
    camera = _join([medium, f.shot_size, f.angle, lens, f"{f.headroom} headroom" if f.headroom.strip() else "",
                    _DEVICE.get(f.device or "", ""), _HANDHELD.get(f.handheld or "", "") if video else "",
                    f"camera {move}" if move else ""])

    w, lk = s.wardrobe, s.look
    wardrobe = _join([f"wearing {w.outfit}" if w.outfit.strip() else "", w.accessories, w.state,
                      f"hair {lk.hair_style}" if lk.hair_style.strip() else "",
                      f"hair {lk.hair_state}" if lk.hair_state.strip() else "",
                      f"makeup {lk.makeup}" if lk.makeup.strip() else "", lk.skin_state])

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
    action = _join([beats, perf])

    st = s.set
    people = (f"{st.background_people} {'person' if st.background_people == 1 else 'people'} in the background"
              if st.background_people > 0 else "")
    set_ = _join([st.location, st.time_of_day, st.season, st.weather, _lighting(st), st.practical_lights,
                  st.set_dressing, st.props, st.background_action if video else "", people,
                  f"{st.depth_of_field} depth of field" if st.depth_of_field.strip() else "",
                  f"same as before: {s.continuity.carry_over}" if s.continuity.carry_over.strip() else ""])
    if not st.location.strip() and scene.strip():
        set_ = _join([scene.strip(), set_])

    realism = _policy(policy, "direction", "realism", default={}) or {}
    style_words = _policy(policy, "direction", "style", default={}) or {}
    style = _join([realism.get("photo", "") if target == "photo" else "",
                   realism.get("video", "") if video and f.handheld in ("light", "held") else "",
                   style_words.get(target, "")])

    sound = ""
    if target == "talk":
        snd = s.sound
        sound = _join([_MIC.get(snd.mic or "", ""), f"room tone {snd.room_tone}" if snd.room_tone.strip() else "",
                       *snd.effects, f"music {snd.music}" if snd.music.strip() else ""])

    neg = _policy(policy, "direction", "negatives", default={}) or {}
    negatives = list(neg.get("always", []))
    if st.background_people <= 0:
        negatives += list(neg.get("solo", []))

    head = f"{trigger}, " if trigger else ""
    sections = [_sentence("Camera", camera), _sentence("Subject", appearance_prompt(bible)),
                _sentence("Wardrobe and look", wardrobe), _sentence("Action", action),
                _sentence("Set", set_), _sentence("Style", style), _sentence("Sound", sound),
                _negatives(negatives)]
    return head + " ".join(x for x in sections if x)


def _negatives(items: list[str]) -> str:
    text = _join(items)
    return f"{text[0].upper()}{text[1:]}." if text else ""
