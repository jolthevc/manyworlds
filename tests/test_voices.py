"""Every creator speaks in a voice of their own: a fixed blend of stock voices, distinct from every sibling."""

from __future__ import annotations

import pytest

from manyworlds import voices
from manyworlds.creator.lab import KOKORO_VOICES


def test_a_blend_is_led_by_the_chosen_voice_stays_in_its_group_and_is_reproducible(policy):
    cfg = policy.get("creator_lab", "voice_blend")
    spec = voices.design("af_heart", KOKORO_VOICES, [], seed="maren", mix=cfg["mix"], lead=cfg["lead"])
    parts = voices.parse(spec)
    assert parts[0][0] == "af_heart" and parts[0][1] >= cfg["lead"] - 1e-6 and len(parts) == cfg["mix"]
    assert all(n.startswith("af_") for n, _ in parts) and abs(sum(w for _, w in parts) - 1) < 0.02
    assert voices.design("af_heart", KOKORO_VOICES, [], seed="maren", mix=cfg["mix"], lead=cfg["lead"]) == spec


def test_fifty_creators_with_the_same_pick_still_all_sound_different(policy):
    cfg = policy.get("creator_lab", "voice_blend")
    taken = []
    for i in range(50):
        taken.append(voices.design("af_heart", KOKORO_VOICES, taken, seed=f"c{i}", mix=cfg["mix"], lead=cfg["lead"],
                                   min_distance=cfg["min_distance"]))
    closest = min(voices.distance(a, b) for i, a in enumerate(taken) for b in taken[i + 1:])
    assert len(set(taken)) == 50 and closest > 0


def test_a_single_stock_voice_passes_through_and_a_blend_is_mixed_by_weight():
    assert voices.voice_arg(None, "af_heart") == "af_heart"

    class Pipe:
        def load_single_voice(self, name):
            return {"af_bella": 10.0, "af_sky": 20.0}[name]
    assert voices.voice_arg(Pipe(), "af_bella:0.75+af_sky:0.25") == 12.5


# --- the named deliveries: which one a line's delivery asks for (policy voice.deliveries) ---------------------------
BANK = {"excited": ["excited", "upbeat", "fast"], "deadpan": ["flat", "dry", "tired", "matter of fact"],
        "amused": ["amused", "laugh", "playful"], "confiding": ["quiet", "sincere"]}


@pytest.mark.parametrize("delivery, expected", [
    ("upbeat", "excited"),
    ("flat, tired, then a small laugh", "deadpan"),      # two deadpan words beat one amused
    ("playful, a bit dry", "amused"),                    # a tie: the first named wins
    ("Matter of fact", "deadpan"),                       # phrases, any case
    ("A quiet, sincere tone", "confiding"),
    ("laughing", None),                                  # whole words only: "laugh" is not "laughing"
    ("musing", None),                                    # nothing named: the base clip
    ("", None),
    (None, None),
])
def test_a_delivery_picks_the_bank_clip_it_names_most(delivery, expected):
    assert voices.nearest_delivery(delivery, BANK) == expected
