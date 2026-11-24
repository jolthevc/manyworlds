"""The WER normaliser: numbers, times, "No." and contractions in one spelling, and the short-line rule
(2026-11-23 breadth run `default-0229fe`: good voice lines dropped for being written differently)."""

from __future__ import annotations

import pytest

from manyworlds import speech


def test_the_breadth_run_lines_that_were_dropped_now_match():
    script = "six p.m. Coffee number two. I'm ready for anything tonight."
    assert speech.wer(script, "6 p.m. Coffee No. 2. I'm ready for anything tonight.") == 0.0
    assert speech.wer("Look two", "Look 2") == 0.0
    assert speech.wer("Look one", "Look ones.") == 0.0


@pytest.mark.parametrize("a, b", [
    ("six p.m.", "6pm"), ("six pm", "6 PM"), ("6 p.m.", "six p m"), ("at six o'clock", "at 6:00"),
    ("coffee number two", "coffee #2"), ("coffee number two", "coffee no. two"), ("coffee number 2", "coffee No 2"),
    ("twenty one", "21"), ("twenty-one", "21"), ("two hundred and five", "205"), ("one thousand", "1,000"),
    ("the third time", "the 3rd time"), ("my twenty first birthday", "my 21st birthday"), ("second", "2nd"),
    ("six thirty", "6:30"), ("I'm fine", "I am fine"), ("don't go", "do not go"), ("it's ok", "it is okay"),
    ("we can't", "we cannot"), ("they'll see", "they will see"), ("you're late", "you are late"),
    ("Maren's coffee", "marens coffee"), ("ten percent", "10%"), ("salt & pepper", "salt and pepper"),
])
def test_one_spelling_on_both_sides(a, b):
    assert speech.normalise(a) == speech.normalise(b), (speech.normalise(a), speech.normalise(b))
    assert speech.wer(a, b) == 0.0


def test_real_differences_still_count():
    assert speech.wer("one two three four", "one two four") == 0.25
    assert speech.wer("coffee number two", "coffee number three") > 0
    assert speech.wer("six pm", "seven pm") > 0
    assert speech.normalise("no way") == "no way"                      # "no" is only "number" before a number
    assert speech.normalise("No comment on the trade.") == "no comment on the trade"


class _Policy:
    def __init__(self, cfg):
        self.cfg = cfg

    def get(self, *path, default=None):
        return self.cfg if path == ("review", "video", "intelligibility") else default


def test_a_short_line_is_judged_by_the_lenient_threshold(policy):
    cfg = policy.get("review", "video", "intelligibility")
    assert cfg["short_line_words"] > 2 and cfg["short_line_max_wer"] >= 0.5 > cfg["max_wer"]
    r = speech.score(policy, "Look three", "Look free")             # one word of two misheard
    assert r["wer"] == 0.5 and r["passed"] and r["max_wer"] == cfg["short_line_max_wer"]
    assert not speech.score(policy, "Look three", "Book free")["passed"]   # both words wrong
    long = "I am ready for anything tonight honestly"
    assert not speech.score(policy, long, "I am ready for nothing tonight honest")["passed"]


def test_the_short_line_rule_is_off_without_its_keys():
    p = _Policy({"max_wer": 0.15})
    assert not speech.score(p, "Look three", "Look free")["passed"]
    assert speech.max_wer_for({"max_wer": 0.15, "short_line_words": 4, "short_line_max_wer": 0.5}, 4) == 0.15
    assert speech.max_wer_for({"max_wer": 0.15, "short_line_words": 4, "short_line_max_wer": 0.5}, 3) == 0.5
