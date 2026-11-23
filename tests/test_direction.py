"""The shot sheet (manyworlds.direction) and its prompt compiler: posts without shots keep today's prompts,
directed prompts come in a fixed order, and whatever the model writes into a shot never crashes the compiler."""

from __future__ import annotations

import re

import pytest

from manyworlds import direction
from manyworlds.creator import lab
from manyworlds.direction import Beat, FramingSpec, Line, PerformanceSpec, SetSpec, ShotSheet, SoundSpec, WardrobeSpec
from manyworlds.worker import comfy

BIBLE = {"appearance": {"apparent_age": 29, "gender_presentation": "woman", "ethnicity_and_skin": "olive skin",
                        "face": "oval face", "eyes": "brown eyes", "hair": "black bob with a clip",
                        "build": "slim", "signature_details": "thin gold wire glasses"}}
LOOK = "29-year-old woman, olive skin, oval face, brown eyes, black bob with a clip, slim, thin gold wire glasses"


def bus_stop(**framing) -> ShotSheet:
    return ShotSheet(set=SetSpec(location="a bus stop on a wet side street", light_quality="soft", color_temp="cool"),
                     wardrobe=WardrobeSpec(outfit="yellow raincoat"),
                     framing=FramingSpec(**framing),
                     blocking=[Beat(at_s=0, action="checks the timetable", eyeline="off"),
                               Beat(at_s=3, action="waves at the bus", eyeline="lens")])


# --- Task 1: the shot sheet ---

def test_summary_mentions_location_outfit_and_first_beat():
    s = direction.summary(bus_stop(shot_size="medium shot"))
    assert "a bus stop on a wet side street" in s and "yellow raincoat" in s and "checks the timetable" in s
    assert "waves at the bus" not in s and "medium shot" in s and "soft, cool light" in s


def test_empty_shot_has_an_empty_summary():
    assert direction.summary(ShotSheet()) == ""
    assert direction.summary({}) == ""


def test_spoken_excludes_deliveries():
    shot = ShotSheet(performance=PerformanceSpec(lines=[
        Line(text="Bus is late again.", delivery="dry, under her breath"),
        Line(text="(sighs) Of course it is.", delivery="resigned", pause_after_s=0.5),
        Line(text="", delivery="laughs")]))
    said = direction.spoken(shot)
    assert said == "Bus is late again. Of course it is."
    assert "breath" not in said and "resigned" not in said and "sighs" not in said
    assert direction.spoken(ShotSheet()) == ""


def test_shot_sheet_round_trips_as_a_dict():
    shot = bus_stop(camera_move="push_in", handheld="light", device="selfie")
    shot.cutaway = True
    again = ShotSheet.model_validate(shot.model_dump())
    assert again == shot and again.cutaway and again.framing.device == "selfie"
    assert ShotSheet().cutaway is False


# --- Task 2: the prompt compiler ---

def test_post_without_shots_keeps_old_prompts(policy):
    assert lab.photo_prompt(BIBLE, "at a bus stop", trigger="mw_x") == (
        f"mw_x, candid smartphone photo of a {LOOK}. Scene: at a bus stop. Natural skin texture, realistic.")
    assert lab.photo_prompt(BIBLE, "at a bus stop", shot=None, policy=policy) == (
        f"candid smartphone photo of a {LOOK}. Scene: at a bus stop. Natural skin texture, realistic.")
    assert lab.clip_prompt(BIBLE, "at a bus stop", "she waves") == (
        f"Handheld smartphone video of a {LOOK}. Scene: at a bus stop. Movement: she waves. "
        "Small natural movements, steady face, realistic.")
    assert lab.clip_prompt(BIBLE, "at a bus stop", "") == (
        f"Handheld smartphone video of a {LOOK}. Scene: at a bus stop. "
        "Small natural movements, steady face, realistic.")
    assert lab.talk_prompt(BIBLE, "at a bus stop", shot=None) == (
        f"Smartphone video of a {LOOK}. Scene: at a bus stop. Talking to the camera naturally, with small head "
        "movements and expressions; steady face, realistic.")


def test_unknown_camera_move_is_dropped(policy):
    shot = bus_stop(camera_move="drone orbit")
    assert direction.camera_move(shot, policy) == ""
    p = direction.compile_prompt(shot, BIBLE, "i2v", policy)
    assert "drone" not in p and "orbit" not in p
    # a known move survives, whatever its spelling
    assert direction.camera_move(bus_stop(camera_move="Push In"), policy) == "push_in"
    assert "camera push in" in direction.compile_prompt(bus_stop(camera_move="push-in"), BIBLE, "i2v", policy)
    # empty and partial values from the model never crash the compiler
    for target in ("photo", "i2v", "talk"):
        p = direction.compile_prompt({"framing": {"camera_move": "", "lens_mm": None}, "blocking": [{}],
                                      "performance": {"lines": [{}]}}, BIBLE, target, policy)
        assert LOOK in p


def test_talk_prompt_orders_camera_subject_action_set(policy):
    shot = ShotSheet(set=SetSpec(location="her galley kitchen", light_quality="soft", light_direction="the window"),
                     framing=FramingSpec(shot_size="medium close-up", camera_move="push_in", device="propped",
                                         handheld="light"),
                     blocking=[Beat(at_s=0, action="leans on the counter", eyeline="lens"),
                               Beat(at_s=4, action="glances at the kettle", eyeline="off")],
                     sound=SoundSpec(mic="phone", room_tone="fridge hum", effects=["kettle click"]))
    p = lab.talk_prompt(BIBLE, "in her kitchen", shot=shot, policy=policy)
    assert p == direction.compile_prompt(shot, BIBLE, "talk", policy, scene="in her kitchen")
    order = [p.index(x) for x in ("Camera:", "Subject:", "Action:", "Set:", "Style:")]
    assert order == sorted(order)
    assert LOOK in p and "camera push in" in p and "phone propped up" in p
    assert "at 0 s: leans on the counter, looking into the lens; then at 4 s: glances at the kettle" in p
    assert policy.get("direction", "realism", "shake", "light") in p      # handheld shot: the phone's own shake
    assert policy.get("direction", "realism", "talk") in p
    # sound is for the mix: the video model cannot render it, so no prompt budget is spent on it
    assert "Sound:" not in p and "phone microphone" not in p and "kettle click" not in p and "fridge" not in p
    assert "in her kitchen" not in p                               # the shot names its own location
    assert "Sound:" not in direction.compile_prompt(shot, BIBLE, "i2v", policy)
    # a target listed in direction.sound_targets still gets it
    loud = {"direction": {**policy.get("direction"), "sound_targets": ["talk"]}}
    q = direction.compile_prompt(shot, BIBLE, "talk", loud)
    assert q.index("Style:") < q.index("Sound:") and "phone microphone" in q and "kettle click" in q


def test_photo_prompt_has_realism_and_positive_stand_ins(policy):
    shot = bus_stop(camera_move="push_in", handheld="held", lens_mm=26)
    p = lab.photo_prompt(BIBLE, "at a bus stop", trigger="mw_x", shot=shot, policy=policy)
    assert p.startswith("mw_x, Camera: candid smartphone photo")
    assert policy.get("direction", "realism", "photo") in p
    assert "visible pores" in p
    assert "26 mm lens" in p and LOOK in p and "wearing yellow raincoat" in p
    assert "push in" not in p and "handheld" not in p              # a photo has no camera move or shake
    assert "checks the timetable" in p and "waves at the bus" not in p   # only the first beat
    # Z-Image at guidance 0 has no negative channel: no negation at all, the stand-ins instead
    assert "no " not in p.lower() and "logos" not in p.lower()
    instead = policy.get("direction", "positive_instead_of")
    assert p.endswith(f"{instead['logos'][0].upper()}{instead['logos'][1:]}, {instead['extra people']}, "
                      f"{instead['waxy skin']}, {instead['studio lighting']}.")
    assert direction.compile_prompts(shot, BIBLE, "photo", policy)["negative"] == ""
    # people asked for in the background lift the solo item
    crowd = bus_stop()
    crowd.set.background_people = 2
    q = direction.compile_prompt(crowd, BIBLE, "photo", policy)
    assert "2 people in the background" in q and instead["extra people"] not in q


def test_a_shot_needs_the_policy():
    with pytest.raises(ValueError):
        lab.photo_prompt(BIBLE, "at a bus stop", shot=ShotSheet())


def test_clip_without_blocking_falls_back_to_the_motion(policy):
    p = lab.clip_prompt(BIBLE, "at a bus stop", "she waves", shot={"framing": {"shot_size": "wide"}}, policy=policy)
    assert "she waves" in p and "at a bus stop" in p and "wide" in p


# --- Wave 2: natural eyes and idle movement ---

def kitchen_talk(**performance) -> ShotSheet:
    return ShotSheet(set=SetSpec(location="her galley kitchen", props="a chipped mug, a spoon"),
                     blocking=[Beat(at_s=0, action="leans on the counter", eyeline="lens"),
                               Beat(at_s=2.5, action="stirs her tea", eyeline="prop"),
                               Beat(at_s=5, action="checks her phone", eyeline="screen"),
                               Beat(at_s=7, action="hears the door", eyeline="off")],
                     performance=PerformanceSpec(**performance))


def test_talk_prompt_has_natural_eyes_tied_to_the_beats(policy):
    p = direction.compile_prompt(kitchen_talk(), BIBLE, "talk", policy)
    assert policy.get("direction", "eyes", "talk") in p
    assert "looks at a chipped mug at 2.5s" in p                   # a prop glance names the shot's first prop
    assert "looks at the screen at 5s" in p and "looks at something off-camera at 7s" in p
    assert "at 0s" not in p                                        # a beat on the lens is no glance
    # eyes and idle are part of the action, before the set
    assert p.index("Action:") < p.index("blinks naturally") < p.index("Set:")
    # a prop glance without props on the set names the object in hand
    bare = kitchen_talk()
    bare.set.props = ""
    assert "looks at the object in hand at 2.5s" in direction.compile_prompt(bare, BIBLE, "talk", policy)


def test_idle_line_follows_energy_then_body_language(policy):
    idle = policy.get("direction", "idle", "by_energy")
    talk = lambda **perf: direction.compile_prompt(kitchen_talk(**perf), BIBLE, "talk", policy)  # noqa: E731
    assert idle["low"] in talk(energy="low") and idle["high"] not in talk(energy="low")
    assert idle["high"] in talk(energy="high")
    # no energy: the body language decides (a word start: 'slouch' matches 'slouched')
    assert idle["high"] in talk(body_language="leaning in, animated hands")
    assert idle["low"] in talk(body_language="slouched on the sofa")
    # nothing to go on: talk's default energy ('fast' is not found inside 'breakfast')
    assert idle["medium"] in talk(body_language="eating breakfast") and idle["high"] not in talk(
        body_language="eating breakfast")
    assert idle["medium"] in talk()


def test_eyes_only_in_video_with_the_creator_in_shot(policy):
    idle = policy.get("direction", "idle", "by_energy")
    clip = direction.compile_prompt(kitchen_talk(), BIBLE, "i2v", policy)
    assert policy.get("direction", "eyes", "i2v") in clip and "looks at the screen at 5s" in clip
    assert policy.get("direction", "eyes", "talk") not in clip     # 'returns to the lens' is for talking clips
    assert not any(line in clip for line in idle.values())         # i2v has no default energy
    assert idle["high"] in direction.compile_prompt(kitchen_talk(energy="high"), BIBLE, "i2v", policy)
    # a cutaway shows something else: no eyes, no idle
    away = kitchen_talk(energy="high")
    away.cutaway = True
    q = direction.compile_prompt(away, BIBLE, "i2v", policy)
    assert "blinks" not in q and "looks at" not in q and idle["high"] not in q
    # a photo is one moment: no blinks, glances or idle movement
    photo = direction.compile_prompt(kitchen_talk(energy="high"), BIBLE, "photo", policy)
    assert "blinks" not in photo and "looks at" not in photo and idle["high"] not in photo
    # undirected prompts are untouched
    assert "blinks" not in lab.talk_prompt(BIBLE, "at a bus stop") and "blinks" not in lab.clip_prompt(
        BIBLE, "at a bus stop", "she waves")


def test_without_eye_and_idle_policy_the_action_is_unchanged(policy):
    bare = {"direction": {k: v for k, v in policy.get("direction").items() if k not in ("eyes", "idle")}}
    shot = kitchen_talk(energy="high")
    p = direction.compile_prompt(shot, BIBLE, "talk", bare)
    assert "blinks" not in p and "looks at" not in p
    assert "Action: talking; at 0 s: leans on the counter, looking into the lens; " in p
    assert "at 7 s: hears the door, looking off-camera, high energy." in p


# --- Wave 2: 720p close-ups ---

def hd_policy(policy, on: bool = True) -> dict:
    return {"direction": {**policy.get("direction"), "hd_close_ups": on},
            "generation": {"talk_profiles": policy.get("generation", "talk_profiles")}}


def test_hd720_profile_keeps_the_talk_steps(policy):
    hd = policy.get("generation", "talk_profiles", "hd720")
    assert (hd["width"], hd["height"]) == (720, 1280)
    assert hd["steps"] == policy.get("generation", "talk", "steps") == 10
    assert policy.get("direction", "hd_close_ups") is False        # off until benchmarked


def test_talk_profile_for_close_ups(policy):
    shot = lambda size: ShotSheet(framing=FramingSpec(shot_size=size))  # noqa: E731
    # shipped: off
    assert direction.talk_profile_for(shot("close-up"), policy) is None
    on = hd_policy(policy)
    for size in ("close-up", "Close Up", "closeup", "extreme close-up", "Extreme_Close-Up"):
        assert direction.talk_profile_for(shot(size), on) == "hd720", size
    for size in ("medium close-up", "medium shot", "wide", ""):
        assert direction.talk_profile_for(shot(size), on) is None, size
    assert direction.talk_profile_for(None, on) is None
    assert direction.talk_profile_for({"framing": {"shot_size": "close-up"}}, on) == "hd720"
    # a profile name that is not in generation.talk_profiles is never returned
    missing = {**on, "generation": {"talk_profiles": {}}}
    assert direction.talk_profile_for(shot("close-up"), missing) is None


# --- Prompt budgets (21:20 gap): the text encoders read 512 tokens and drop the rest ---

# Sizes mirror the real director test: a ~700-character appearance, a ~600-character outfit, a ~700-character set.
LONG_BIBLE = {"appearance": {
    "apparent_age": 31, "gender_presentation": "woman",
    "ethnicity_and_skin": "fair Northern European skin with cool pink undertones, light freckles across the nose and "
                          "cheekbones, faint redness around the nostrils",
    "face": "soft heart-shaped face with high rounded cheekbones, a slightly upturned nose, a small scar through the "
            "left eyebrow, a full lower lip, a defined cupid's bow and a gentle jawline",
    "eyes": "wide-set grey-green eyes with heavy upper lids, thin laugh lines at the outer corners, straight "
            "light-brown brows",
    "hair": "shoulder-length wavy strawberry-blonde hair with a centre parting, usually tucked behind one ear, a few "
            "loose strands framing the face",
    "build": "tall and lean, narrow shoulders, long arms",
    "signature_details": "a small silver hoop in her left ear, a thin red string bracelet on her right wrist, chipped "
                         "sage-green nail polish"}}
LONG_OUTFIT = ("a slouchy oatmeal cable-knit wool jumper with dropped shoulders and slightly too-long sleeves pushed up "
               "to the forearms, worn over a faded cream cotton t-shirt whose hem peeks out at the bottom, high-waisted "
               "straight-leg indigo jeans with a soft worn fade at the knees and a rolled cuff, thick ribbed charcoal "
               "socks and battered tan suede ankle boots with a stacked heel; the jumper has a small darned patch near "
               "the left cuff and a few pills of wool along the sides, which gives the whole outfit a lived-in, "
               "comfortable, slightly rumpled weekend feel, with no visible logos or text anywhere")
LONG_DRESSING = ("white metro tiles, a pale oak worktop scattered with crumbs, a green enamel kettle on the hob, a chipped "
                 "blue ceramic fruit bowl holding two lemons and a browning banana, a row of mismatched mugs hanging "
                 "from brass hooks under a wall cabinet, a trailing pothos plant on the windowsill above the sink, a tea "
                 "towel with a faded check pattern draped over the oven door handle, a small wooden chopping board "
                 "leaning against the backsplash, a stack of unopened post weighed down by a jar of honey, a slightly "
                 "crooked framed botanical print on the far wall, a dish rack holding two plates and a frying pan, and "
                 "a single window over the sink looking out onto a grey courtyard")
BUDGETED = ("photo", "i2v", "talk")


def long_kitchen() -> ShotSheet:
    return ShotSheet(
        set=SetSpec(location="her narrow galley kitchen", time_of_day="early morning", season="late autumn",
                    light_direction="the window on her left", light_quality="soft",
                    color_temp="cool daylight with a warm lamp", practical_lights="a warm under-cabinet strip light",
                    set_dressing=LONG_DRESSING,
                    props="a chipped mug of builder's tea, a teaspoon, a half-eaten slice of toast with marmalade",
                    background_action="steam curling from the kettle spout", depth_of_field="shallow"),
        wardrobe=WardrobeSpec(outfit=LONG_OUTFIT, accessories="a black hair tie on her wrist",
                              state="sleeves pushed up, a little creased"),
        look={"hair_state": "slightly messy, just woken up", "makeup": "none",
              "skin_state": "a little puffy under the eyes"},
        framing=FramingSpec(shot_size="medium close-up", angle="eye level", lens_mm=26, camera_move="push_in",
                            handheld="light", device="propped", headroom="normal"),
        blocking=[Beat(at_s=0, action="leans on the worktop and sighs", eyeline="lens"),
                  Beat(at_s=3.5, action="lifts the mug and blows on the tea", eyeline="prop",
                       hands="both around the mug"),
                  Beat(at_s=7, action="glances at the window, then back", eyeline="off")],
        performance=PerformanceSpec(energy="low", body_language="slouched against the counter, shoulders loose",
                                    emotion_arc="groggy to wry"),
        sound=SoundSpec(mic="phone", room_tone="fridge hum and a dripping tap",
                        effects=["kettle click", "spoon on ceramic"]))


def unbudgeted(policy) -> dict:
    return {"direction": {k: v for k, v in policy.get("direction").items() if k != "prompt_budget"}}


def _trigger(target: str) -> str | None:
    return "mw_m" if target == "photo" else None


def test_budgets_leave_headroom_under_the_encoders_512_tokens(policy):
    budget = policy.get("direction", "prompt_budget")
    assert set(budget) == set(BUDGETED) and all(0 < budget[t] < 512 for t in BUDGETED)
    assert direction.estimate_tokens("") == 0
    assert direction.estimate_tokens("one two three") == 5           # 3 words x 1.35 = 4.05 -> 5
    assert direction.estimate_tokens("a,b,c,d,e,f,g,h") == 5         # one word, but 15 chars / 3.6 = 4.2 -> 5


def test_the_fixture_is_over_budget_untrimmed(policy):
    # the sizes of the real director test (2026-11-22 21:20)
    assert len(lab.appearance_prompt(LONG_BIBLE)) > 700 and len(LONG_OUTFIT) > 550 and len(LONG_DRESSING) > 650
    for target in BUDGETED:
        full = direction.compile_prompt(long_kitchen(), LONG_BIBLE, target, unbudgeted(policy),
                                        trigger=_trigger(target))
        assert len(full) > 1500
        assert direction.estimate_tokens(full) > policy.get("direction", "prompt_budget", target), target


def test_every_target_fits_its_budget_and_keeps_camera_action_and_stand_ins(policy):
    budget = policy.get("direction", "prompt_budget")
    instead = policy.get("direction", "positive_instead_of")
    for target in BUDGETED:
        out = direction.compile_prompts(long_kitchen(), LONG_BIBLE, target, policy, trigger=_trigger(target))
        p = out["prompt"]
        full = direction.compile_prompt(long_kitchen(), LONG_BIBLE, target, unbudgeted(policy),
                                        trigger=_trigger(target))
        assert direction.estimate_tokens(p) <= budget[target], target
        camera = full[:full.index(" Subject:")]
        action = full[full.index("Action:"):full.index(" Set:")]
        assert p.startswith(camera) and action in p, target
        if target == "photo":                                      # no negative channel: the stand-ins, kept whole
            tail = full[full.index(f"{instead['logos'][0].upper()}{instead['logos'][1:]}"):]
            assert p.endswith(tail) and out["negative"] == ""
        else:                                                      # the negative channel, outside the budget
            assert instead["logos"] not in p and "logos" in out["negative"]
        # what is left keeps the fixed order, and every cut is clean
        labels = [x for x in ("Camera:", "Subject:", "Wardrobe and look:", "Action:", "Set:", "Style:", "Sound:")
                  if x in p]
        assert [p.index(x) for x in labels] == sorted(p.index(x) for x in labels)
        assert "Subject:" in p and "  " not in p and ".." not in p and ", ." not in p
        # long free text was cut to its first clause first
        assert LONG_OUTFIT not in p and "darned patch" not in p and "courtyard" not in p


def test_video_uses_the_short_appearance_and_photos_keep_the_full_one(policy):
    short = lab.short_appearance(LONG_BIBLE)
    assert short == ("31-year-old woman, shoulder-length wavy strawberry-blonde hair with a centre parting, usually "
                     "tucked behind one ear, a few loose strands framing the face, a small silver hoop in her left ear")
    assert lab.short_appearance(BIBLE) == "29-year-old woman, black bob with a clip, thin gold wire glasses"
    for target in ("i2v", "talk"):
        p = direction.compile_prompt(long_kitchen(), LONG_BIBLE, target, policy)
        assert f"Subject: {short}." in p and lab.appearance_prompt(LONG_BIBLE) not in p, target
    # the photo is where identity is made (LoRA + the full appearance: 12 of 12 kept on Maren)
    photo = direction.compile_prompt(long_kitchen(), LONG_BIBLE, "photo", policy, trigger="mw_m")
    assert f"Subject: {lab.appearance_prompt(LONG_BIBLE)}." in photo and photo.startswith("mw_m, Camera:")


def test_short_shots_are_not_trimmed(policy):
    shots = [bus_stop(shot_size="medium shot", camera_move="push_in", handheld="light", lens_mm=26),
             kitchen_talk(energy="high"), kitchen_talk(), ShotSheet()]
    for shot in shots:
        for target in BUDGETED:
            assert direction.compile_prompt(shot, BIBLE, target, policy, scene="at a bus stop") == (
                direction.compile_prompt(shot, BIBLE, target, unbudgeted(policy), scene="at a bus stop"))
            assert LOOK in direction.compile_prompt(shot, BIBLE, target, policy)


def test_undirected_prompts_are_never_budgeted():
    look = lab.appearance_prompt(LONG_BIBLE)
    assert lab.photo_prompt(LONG_BIBLE, "in her kitchen", trigger="mw_m") == (
        f"mw_m, candid smartphone photo of a {look}. Scene: in her kitchen. Natural skin texture, realistic.")
    assert lab.clip_prompt(LONG_BIBLE, "in her kitchen", "she sips") == (
        f"Handheld smartphone video of a {look}. Scene: in her kitchen. Movement: she sips. "
        "Small natural movements, steady face, realistic.")
    assert lab.talk_prompt(LONG_BIBLE, "in her kitchen") == (
        f"Smartphone video of a {look}. Scene: in her kitchen. Talking to the camera naturally, with small head "
        "movements and expressions; steady face, realistic.")


def test_first_clause():
    assert direction._first_clause("a sofa. a lamp by the door", 20) == "a sofa"
    assert direction._first_clause("white tiles; a kettle", 20) == "white tiles"
    assert direction._first_clause("white tiles, a kettle", 20) == "white tiles, a kettle"      # short: as it was
    many = ", ".join(f"item number {i}" for i in range(10))                                    # 30 words
    assert direction._first_clause(many, 7) == "item number 0, item number 1"                 # whole items
    assert direction._first_clause("one two three four five six", 4) == "one two three four"  # one long item
    assert direction._first_clause("", 20) == ""


# --- Phone-native prompts (2026-11-23): negatives out of the positive prompt ---

NEGATION = re.compile(r"\b(no|not|never|without)\b", re.IGNORECASE)


def test_no_directed_prompt_carries_a_negation(policy):
    for shot in (bus_stop(handheld="light"), kitchen_talk(energy="low"), ShotSheet(),
                 bus_stop(device="propped")):
        for target in BUDGETED:
            p = direction.compile_prompt(shot, LONG_BIBLE, target, policy, trigger=_trigger(target))
            assert not NEGATION.search(p), (target, NEGATION.search(p).group(0), p)


def test_video_negatives_go_to_the_negative_channel_after_the_base(policy):
    neg = policy.get("direction", "negatives")
    for target in ("i2v", "talk"):
        out = direction.compile_prompts(kitchen_talk(), BIBLE, target, policy)
        assert out["negative"].startswith(neg["base"][target]), target
        for item in (*neg["always"], *neg["solo"], *neg["by_target"][target]):
            assert item in out["negative"], (target, item)
            assert not re.search(rf"\b{re.escape(item)}\b", out["prompt"]), (target, item)
        assert out["prompt"] == direction.compile_prompt(kitchen_talk(), BIBLE, target, policy)
        assert direction.has_negative_channel(target, policy)
    assert not direction.has_negative_channel("photo", policy)
    # the shipped base leaves out Wan's "bright tones, overexposed": they fight the phone-native look
    assert "overexposed" not in neg["base"]["i2v"] and "overexposed" in comfy.NEGATIVE


def test_a_negative_yields_to_what_the_shot_asks_for(policy):
    mirror = ShotSheet(set=SetSpec(location="her bedroom, in front of the full-length mirror"),
                       framing=FramingSpec(device="selfie"))
    neg = direction.compile_prompts(mirror, BIBLE, "talk", policy)["negative"]
    assert "mirror reflections" not in neg and "logos" in neg and "extra people" in neg
    crowd = kitchen_talk()
    crowd.set.background_people = 3
    assert "extra people" not in direction.compile_prompts(crowd, BIBLE, "i2v", policy)["negative"]
    # each item once, and an item named in no list never appears
    items = direction.negative_items(kitchen_talk(), "talk", policy)
    assert len(items) == len({i.lower() for i in items}) and "gimbal-smooth camera" not in items


def test_without_a_stand_in_a_negative_is_dropped_for_the_photo(policy):
    bare = {"direction": {**policy.get("direction"), "positive_instead_of": {}}}
    p = direction.compile_prompt(kitchen_talk(), BIBLE, "photo", bare)
    assert p[p.rindex("Style:"):] == f"Style: {direction._style(kitchen_talk(), 'photo', bare)}."   # nothing after it
    assert "logos" not in p and "mirror" not in p
    assert direction.compile_prompts(kitchen_talk(), BIBLE, "photo", bare)["negative"] == ""


def test_the_phones_shake_follows_handheld_then_device(policy):
    shake = policy.get("direction", "realism", "shake")
    clip = lambda **f: direction.compile_prompt(ShotSheet(framing=FramingSpec(**f)), BIBLE, "i2v", policy)  # noqa: E731
    assert shake["none"] in clip(device="propped")                 # locked off: no gimbal glide
    assert shake["light"] in clip(device="selfie")
    assert shake["held"] in clip(device="held")
    assert shake["light"] in clip(device="propped", handheld="light")   # the shot's own handheld wins
    assert not any(v in clip() for v in shake.values())             # nothing said, nothing added
    assert not any(v in direction.compile_prompt(ShotSheet(framing=FramingSpec(handheld="held")), BIBLE, "photo",
                                                 policy) for v in shake.values())


def test_realism_is_per_target_and_light_is_motivated(policy):
    realism = policy.get("direction", "realism")
    for target in BUDGETED:
        p = direction.compile_prompt(ShotSheet(), BIBLE, target, policy)
        assert realism[target] in p and realism["light_fallback"] in p, target
        assert "cinematic" not in p.lower() and "studio" not in p.lower()
    lit = ShotSheet(set=SetSpec(light_direction="the window on her left"))
    assert realism["light_fallback"] not in direction.compile_prompt(lit, BIBLE, "photo", policy)


def test_lab_prompts_give_the_payloads_prompt_and_negative(policy):
    undirected = lab.prompts("i2v", BIBLE, "at a bus stop", motion="she waves")
    assert undirected == {"prompt": lab.clip_prompt(BIBLE, "at a bus stop", "she waves"), "negative": ""}
    assert lab.prompts("talk", BIBLE, "at a bus stop")["prompt"] == lab.talk_prompt(BIBLE, "at a bus stop")
    assert lab.prompts("photo", BIBLE, "x", trigger="mw_x")["prompt"] == lab.photo_prompt(BIBLE, "x", trigger="mw_x")
    shot = kitchen_talk()
    assert lab.prompts("talk", BIBLE, "in her kitchen", shot=shot, policy=policy) == direction.compile_prompts(
        shot, BIBLE, "talk", policy, scene="in her kitchen")
    with pytest.raises(ValueError):
        lab.prompts("dance", BIBLE, "x")
    with pytest.raises(ValueError):
        lab.prompts("talk", BIBLE, "x", shot=shot)                  # a shot needs the policy
