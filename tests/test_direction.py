"""The shot sheet (manyworlds.direction) and its prompt compiler: posts without shots keep today's prompts,
directed prompts come in a fixed order, and whatever the model writes into a shot never crashes the compiler."""

from __future__ import annotations

import pytest

from manyworlds import direction
from manyworlds.creator import lab
from manyworlds.direction import Beat, FramingSpec, Line, PerformanceSpec, SetSpec, ShotSheet, SoundSpec, WardrobeSpec

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
    order = [p.index(x) for x in ("Camera:", "Subject:", "Action:", "Set:", "Style:", "Sound:", "No text")]
    assert order == sorted(order)
    assert LOOK in p and "camera push in" in p and "phone propped up" in p
    assert "at 0 s: leans on the counter, looking into the lens; then at 4 s: glances at the kettle" in p
    assert "slight handheld micro-jitter" in p                     # handheld shot: the video realism cue
    assert "phone microphone" in p and "kettle click" in p
    assert "in her kitchen" not in p                               # the shot names its own location
    # sound cues belong to talking clips only
    assert "Sound:" not in direction.compile_prompt(shot, BIBLE, "i2v", policy)


def test_photo_prompt_has_realism_and_negatives(policy):
    shot = bus_stop(camera_move="push_in", handheld="held", lens_mm=26)
    p = lab.photo_prompt(BIBLE, "at a bus stop", trigger="mw_x", shot=shot, policy=policy)
    assert p.startswith("mw_x, Camera: candid smartphone photo")
    assert policy.get("direction", "realism", "photo") in p
    assert "visible pores" in p
    assert "26 mm lens" in p and LOOK in p and "wearing yellow raincoat" in p
    assert "no text, no logos, no mirrors behind the subject, no extra people." in p.lower()
    assert "push in" not in p and "handheld" not in p              # a photo has no camera move or shake
    assert "checks the timetable" in p and "waves at the bus" not in p   # only the first beat
    # people asked for in the background lift the solo negative
    crowd = bus_stop()
    crowd.set.background_people = 2
    q = direction.compile_prompt(crowd, BIBLE, "photo", policy)
    assert "2 people in the background" in q and "no extra people" not in q.lower()


def test_a_shot_needs_the_policy():
    with pytest.raises(ValueError):
        lab.photo_prompt(BIBLE, "at a bus stop", shot=ShotSheet())


def test_clip_without_blocking_falls_back_to_the_motion(policy):
    p = lab.clip_prompt(BIBLE, "at a bus stop", "she waves", shot={"framing": {"shot_size": "wide"}}, policy=policy)
    assert "she waves" in p and "at a bus stop" in p and "wide" in p
