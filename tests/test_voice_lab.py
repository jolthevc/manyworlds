"""A designed voice per creator (creator/voice_lab.py): Kokoro until she is cast, then her engine's speak payload
with natural speech from policy. VoxCPM2's design / pick / delivery-bank flow was retired 2026-11-26 (its tests went
with it; tests/test_voxcpm_retired.py checks nothing can choose it any more)."""

from __future__ import annotations

import uuid

from psycopg.types.json import Jsonb

from manyworlds import voices
from manyworlds.creator import lab, voice_lab

BIBLE = {"display_name": "Maren", "one_line": "slow mornings, fast opinions",
         "appearance": {"apparent_age": 27, "gender_presentation": "woman"},
         "voice_notes": "warm and slightly husky (laughs mid-sentence), unhurried"}
KOKORO = {"voice": {"engine": "kokoro", "voice": "af_bella:0.6+af_nova:0.4"}}


def creator(conn, anchors=None):
    return conn.execute("INSERT INTO creators (handle, bible, anchors) VALUES (%s, %s, %s) RETURNING id",
                        (f"c{uuid.uuid4().hex[:6]}", Jsonb(BIBLE), Jsonb(anchors or KOKORO))).fetchone()["id"]


def anchors(conn, cid):
    return conn.execute("SELECT anchors FROM creators WHERE id = %s", (cid,)).fetchone()["anchors"]


def test_the_description_comes_from_the_bible_without_parentheses():
    d = voice_lab.describe(BIBLE)
    assert "27-year-old woman" in d and "husky" in d
    assert "(" not in d and ")" not in d       # a leading parenthetical is a delivery


def test_kokoro_until_voice_cast(conn, policy):
    """Review Focus 4: a creator without a designed voice (Maren today) keeps speaking with Kokoro, also while a
    cast is pending."""
    cid = creator(conn)
    voice_lab.cast(conn, policy, cid)
    assert voices.engine(anchors(conn, cid)) == "kokoro"
    sample = conn.execute("SELECT kind, payload FROM jobs WHERE id = %s",
                          (lab.enqueue_voice_sample(conn, cid),)).fetchone()
    assert sample["kind"] == "image.voice" and sample["payload"]["voice"] == KOKORO["voice"]["voice"]
    assert voices.engine({"voice": {"engine": "qwen3"}}) == "kokoro"        # no reference clip yet: not usable
    assert voices.engine(None) == "kokoro"


def test_a_speak_payload_carries_the_designed_voice_and_the_takes(policy):
    voice = {"engine": "qwen3", "ref_uri": "s3://b/v.wav", "ref_text": "Okay so this happened."}
    p = voice_lab.speak_payload(policy, voice, [{"text": "Hi.", "delivery": "softly"}, {"text": "Bye."}], seed=3)
    assert (p["ref_uri"], p["ref_text"], p["takes"], p["seed"]) == (
        "s3://b/v.wav", voice["ref_text"], policy.get("voice", "engines", "qwen3", "takes"), 3)
    assert p["lines"] == [{"text": "Hi.", "delivery": "softly"}, {"text": "Bye."}]   # the delivery never in the text
    assert p["params"]["pause_s"] == policy.get("voice", "pause_s")
    assert set(p["params"]["engines"]) == {"qwen3"}


def test_an_inline_parenthetical_becomes_the_delivery(policy):
    voice = {"engine": "indextts", "ref_uri": "s3://b/v.wav", "ref_text": "Okay so this happened."}
    lines = [{"text": "(musing) Walking home.", "pause_after_s": 0.4}]
    [line] = voice_lab.speak_payload(policy, voice, lines)["lines"]
    assert line["text"] == "Walking home." and line["delivery"] == "musing" and line["pause_after_s"] == 0.4


def test_the_speak_payload_carries_natural_speech_as_policy_sets_it(policy):
    """Item 7 (plan 2026-11-22-video-quality-autonomous): clause splitting, the gap ranges and the line ceiling reach
    the worker from policy and parse there; breaths and fillers are experimental and ship off."""
    from manyworlds.worker.handlers import speech_audio

    voice = {"engine": "qwen3", "ref_uri": "s3://b/v.wav", "ref_text": "Hi."}
    params = voice_lab.speak_payload(policy, voice, [{"text": "Hi."}])["params"]
    for key in ("synthesis", "clause_gap_s", "sentence_gap_s", "trim", "fade_ms", "line_max_wer", "breaths",
                "disfluency"):
        assert params[key] == policy.get("voice", key)
    cfg = speech_audio.speech_config(params)
    assert cfg["clause_gap"][1] < cfg["sentence_gap"][0]                            # clause pauses shorter
    assert cfg["breaths"] is None and cfg["disfluency"] is None
    assert cfg["max_wer"] == policy.get("voice", "max_wer") and cfg["retry_factor"] >= 1
    assert policy.get("voice", "synthesis") == "whole_line" and cfg["split"] is False
    assert params["asr_model"] == policy.get("review", "video", "intelligibility", "model")   # the gate's ASR
