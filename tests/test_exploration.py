"""Thompson-sampled exploration: promising-but-uncertain directions keep getting tried."""

from __future__ import annotations

from collections import Counter

from manyworlds.brain import exploration


def _pick_counts(policy, effects, untried, n=300):
    c = Counter()
    for i in range(n):
        for d in exploration.directions(policy, effects=effects, untried=untried, seed_key=f"s{i}"):
            c[d["direction"]] += 1
    return c


def test_a_clear_winner_is_almost_always_picked_and_a_clear_loser_almost_never(policy):
    effects = [{"kind": "attribute", "key": "setting=market", "effect": 0.6, "se": 0.05},
               {"kind": "attribute", "key": "mood=gloomy", "effect": -0.6, "se": 0.05}]
    c = _pick_counts(policy, effects, ["format:carousel", "format:story", "staging:asmr", "staging:tour_guide"])
    assert c["attribute:setting=market"] > 290 and c["attribute:mood=gloomy"] < 10


def test_uncertain_directions_still_get_explored(policy):
    effects = [{"kind": "format", "key": "story", "effect": 0.05, "se": 0.4}]
    c = _pick_counts(policy, effects, ["staging:asmr", "staging:tour_guide", "staging:cooking_host", "format:carousel"])
    assert all(c[d] > 30 for d in ("format:story", "staging:asmr", "staging:tour_guide", "staging:cooking_host"))


def test_draws_are_repeatable_per_batch_and_described_in_plain_words(policy):
    a = exploration.directions(policy, effects=[], untried=["staging:asmr", "format:carousel"], seed_key="c:2026-11-20")
    assert a == exploration.directions(policy, effects=[], untried=["staging:asmr", "format:carousel"],
                                       seed_key="c:2026-11-20")
    assert {d["describe"] for d in a} <= {"the asmr staging", "the carousel format"}
