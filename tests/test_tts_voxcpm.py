"""Designed voices on the worker (worker/handlers/tts_voxcpm.py), with the model, transcriber and embedder replaced
by fakes: payload checks, takes scored by WER against the words only, lines joined with their pauses at 48 kHz."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from manyworlds import voices
from manyworlds.speech import wer
from manyworlds.storage import LocalStore
from manyworlds.worker.handlers import tts_voxcpm as tts
from manyworlds.worker.runner import JobContext, NonRetryable

SR = 48000


class Fake:
    """VoxCPM2's place: half a second of audio per call; hears back what it was asked to say (without the
    instruction) unless given transcripts to return in order."""
    sample_rate = SR

    def __init__(self, transcripts=None):
        self.calls, self.transcripts = [], list(transcripts or [])

    def generate(self, *, seed, text, **kw):
        self.calls.append({"seed": seed, "text": text, **kw})
        return np.full(SR // 2, 0.1, dtype=np.float32)

    def hear(self, wav, sr):
        return self.transcripts.pop(0) if self.transcripts else voices.strip_delivery(self.calls[-1]["text"])


@pytest.fixture()
def fake(monkeypatch):
    f = Fake()
    monkeypatch.setattr(tts, "_tts", lambda params: f)
    monkeypatch.setattr(tts, "_asr", lambda params: f.hear)
    monkeypatch.setattr(tts, "_embedder", lambda params: (lambda wav, sr: [1.0, 0.0, float(len(f.calls))]))
    monkeypatch.setattr(tts, "_pitch_variance", lambda wav, sr, params: 0.1 * len(f.calls))
    return f


@pytest.fixture()
def ctx(tmp_path):
    store = LocalStore(tmp_path / "objects")
    work = tmp_path / "work"
    work.mkdir()
    return JobContext(job=None, workdir=work, store=store)


def _ref(ctx, tmp_path) -> str:
    path = tts.write_wav(tmp_path / "ref.wav", np.zeros(SR, dtype=np.float32), SR)
    return ctx.store.put_file(path, "voices/ref.wav").uri


def _speak(ctx, tmp_path, lines, **kw):
    return tts.speak({"ref_uri": _ref(ctx, tmp_path), "ref_text": "Okay so this happened.", "lines": lines,
                      "seed": 7, **kw}, ctx)


@pytest.mark.parametrize("bad", [
    {"text": "hi there", "n": 2},                                   # no description
    {"description": "warm", "text": "   ", "n": 2},                 # nothing to say
    {"description": "warm", "text": "(softly)", "n": 2},            # only an instruction
    {"description": "warm", "text": "hi", "n": 0},
    {"description": "warm", "text": "hi", "n": "4"},
    {"description": "warm", "text": "hi", "n": True},
    {"description": "warm", "text": "hi", "n": 2, "seed": "x"},
    {"description": "warm", "text": "hi", "n": 2, "params": ["model"]},
])
def test_a_bad_design_payload_is_not_retried(bad):
    with pytest.raises(NonRetryable):
        tts.validate_design(bad)


@pytest.mark.parametrize("bad", [
    {"ref_text": "x", "lines": [{"text": "hi"}]},                            # no reference clip
    {"ref_uri": "s3://b/r.wav", "lines": [{"text": "hi"}]},                  # no reference transcript
    {"ref_uri": "s3://b/r.wav", "ref_text": "x", "lines": []},
    {"ref_uri": "s3://b/r.wav", "ref_text": "x", "lines": ["hi"]},
    {"ref_uri": "s3://b/r.wav", "ref_text": "x", "lines": [{"delivery": "softly"}]},
    {"ref_uri": "s3://b/r.wav", "ref_text": "x", "lines": [{"text": "hi", "pause_after_s": -1}]},
    {"ref_uri": "s3://b/r.wav", "ref_text": "x", "lines": [{"text": "hi", "delivery": 3}]},
    {"ref_uri": "s3://b/r.wav", "ref_text": "x", "lines": [{"text": "hi"}], "takes": 0},
])
def test_a_bad_speak_payload_is_not_retried(bad):
    with pytest.raises(NonRetryable):
        tts.validate_speak(bad)


def test_design_makes_n_scored_candidates_in_the_described_voice(fake, ctx):
    res = tts.design({"description": "warm, low, a little husky", "text": "Okay so this happened.", "n": 3,
                      "seed": 10}, ctx)
    assert [o.meta["seed"] for o in res.outputs] == [10, 11, 12]
    assert fake.calls[0]["text"] == "(warm, low, a little husky)Okay so this happened."
    assert all(o.kind == "audio" and o.meta["wer"] == 0.0 and o.meta["embedding"] for o in res.outputs)
    assert [o.meta["pitch_var"] for o in res.outputs] == pytest.approx([0.1, 0.2, 0.3])
    with wave.open(str(res.outputs[0].path)) as w:
        assert w.getframerate() == SR and w.getnframes() == SR // 2
    assert res.recipe["engine"] == "voxcpm" and res.meta["output_seconds"] == 1.5


def test_delivery_is_not_in_transcript_check(fake, ctx, tmp_path):
    """Review Focus 5: the delivery instruction goes to the model, never into the words the transcript is checked
    against (it is not spoken, so counting it would fail every directed line)."""
    res = _speak(ctx, tmp_path, [{"text": "I knew it.", "delivery": "whispering, amused"},
                                 {"text": "(sighs) Fine."}])                   # an inline instruction counts too
    assert [c["text"] for c in fake.calls] == ["(whispering, amused)I knew it.", "(sighs)Fine."]
    lines = res.outputs[0].meta["lines"]
    assert [line["text"] for line in lines] == ["I knew it.", "Fine."]
    assert [line["wer"] for line in lines] == [0.0, 0.0]
    assert [line["delivery"] for line in lines] == ["whispering, amused", "sighs"]
    assert wer("(whispering, amused) I knew it.", "I knew it") == 0.4   # what checking the raw text would score
    # a delivery is followed on the reference clip alone (VoxCPM README: controllable cloning)
    assert "prompt_text" not in fake.calls[0] and fake.calls[0]["reference_wav_path"].endswith("reference.wav")


def test_without_a_delivery_the_clone_gets_its_transcript_too(fake, ctx, tmp_path):
    _speak(ctx, tmp_path, [{"text": "Hi."}])
    call = fake.calls[0]
    assert call["text"] == "Hi." and call["prompt_text"] == "Okay so this happened."
    assert call["prompt_wav_path"] == call["reference_wav_path"]


def test_the_take_closest_to_the_line_is_kept(fake, ctx, tmp_path):
    fake.transcripts = ["I knew that.", "I knew it."]
    res = _speak(ctx, tmp_path, [{"text": "I knew it."}], takes=2)
    line = res.outputs[0].meta["lines"][0]
    assert (line["take"], line["wer"]) == (1, 0.0) and len(fake.calls) == 2
    assert fake.calls[0]["seed"] != fake.calls[1]["seed"]


def test_lines_are_joined_with_their_pauses_into_one_48khz_file(fake, ctx, tmp_path):
    res = _speak(ctx, tmp_path, [{"text": "One.", "pause_after_s": 0.5}, {"text": "Two."}, {"text": "Three."}],
                 params={"pause_s": 0.25})                    # line two takes the default pause; the last none
    meta = res.outputs[0].meta
    assert [(line["start_s"], line["end_s"]) for line in meta["lines"]] == [(0.0, 0.5), (1.0, 1.5), (1.75, 2.25)]
    assert meta["seconds"] == 2.25 and meta["sample_rate"] == SR and meta["wer"] == 0.0
    with wave.open(str(res.outputs[0].path)) as w:
        assert w.getframerate() == SR and w.getnframes() == int(2.25 * SR)


def test_instructions_are_split_from_the_words():
    assert voices.split_instruction("(softly) (amused) I knew it.") == ("softly, amused", "I knew it.")
    assert voices.split_instruction("No instruction (here).") == ("", "No instruction (here).")
    assert voices.with_instruction("slightly faster (cheerful)", "(old) Go.") == "(slightly faster, cheerful)Go."
    assert voices.with_instruction("", "Go.") == "Go."
