"""Creators must read as real people worth following, never "too AI" (owner 2026-11-21): the rules are in the
prompts so the model writes within them, and whatever still breaks them is dropped."""

from __future__ import annotations

import pytest

from manyworlds.brain import planner
from manyworlds.creator import lab


@pytest.mark.parametrize("text", [
    "It's not just coffee, it's a whole ritual",
    "rainy sunday reading. what about you?",
    "a gentle reminder that slow mornings matter",
    "feeling so grateful for this view",
    "hey guys, so today I want to talk about my week",
    "this tapestry of a city",
    "golden hour — finally",
])
def test_ai_shaped_lines_are_caught(policy, text):
    assert planner.caption_issues(policy, text)


@pytest.mark.parametrize("text", [
    "the 6 bus was late again so I bought the good croissant out of spite",
    "raindrop a minute on the window and I'm cancelling everything",   # word start, not a substring
    "Seoul in november. cold hands, hot tteokbokki",
    "ok this lamp was $12 and I would die for it",
])
def test_real_sounding_lines_pass(policy, text):
    assert planner.caption_issues(policy, text) == []


def test_the_rules_are_in_the_prompts_not_only_in_the_filter(policy):
    rules = planner.voice_rules(policy)
    assert "cringe" in rules and "embrace" in rules and "it's not just X, it's Y" in rules
    assert planner.system_prompt(policy).startswith(planner.SYSTEM) and rules in planner.system_prompt(policy)
    system = lab.persona_system(policy, lab.SYSTEM)
    assert policy.get("creator_lab", "persona_guide").split()[0] in system and "cringe" in system


def test_curly_apostrophes_are_caught_too(policy):
    assert planner.caption_issues(policy, "it\u2019s not just a lamp, it\u2019s a lifestyle")
