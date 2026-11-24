"""Designed voices on the worker (worker/handlers/tts_voxcpm.py), with the model, transcriber and embedder replaced
by fakes: payload checks, takes scored by WER against the words only, lines joined with their pauses at 48 kHz, and
natural speech (clauses, seeded gaps, trim and fades, per-clause retries, the experimental breaths and fillers)."""

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


def test_every_line_is_cloned_with_the_reference_and_its_transcript(fake, ctx, tmp_path):
    """Voice consistency (2026-11-23): a line with a delivery used to be cloned from the clip alone, so its timbre
    drifted from the lines without one. Now every line gets the clip and its transcript; the delivery stays the
    instruction in the text."""
    _speak(ctx, tmp_path, [{"text": "Hi.", "delivery": "dry"}, {"text": "Bye."}])
    for call in fake.calls:
        assert call["prompt_text"] == "Okay so this happened."
        assert call["prompt_wav_path"] == call["reference_wav_path"]
        assert call["prompt_wav_path"].endswith("reference.wav")
    assert [c["text"] for c in fake.calls] == ["(dry)Hi.", "Bye."]


def test_controllable_cloning_is_still_available(fake, ctx, tmp_path):
    res = _speak(ctx, tmp_path, [{"text": "Hi.", "delivery": "dry"}, {"text": "Bye."}],
                 params={"clone": "controllable"})
    assert "prompt_text" not in fake.calls[0] and fake.calls[0]["reference_wav_path"].endswith("reference.wav")
    assert fake.calls[1]["prompt_text"] == "Okay so this happened."     # without a delivery: the transcript too
    assert res.recipe["clone"] == "controllable"
    with pytest.raises(NonRetryable):
        tts.validate_speak({"ref_uri": "s3://b/r.wav", "ref_text": "x", "lines": [{"text": "hi"}],
                            "params": {"clone": "sometimes"}})


def test_the_take_closest_to_the_line_is_kept(fake, ctx, tmp_path):
    fake.transcripts = ["I knew that.", "I knew it."]
    res = _speak(ctx, tmp_path, [{"text": "I knew it."}], takes=2)
    meta = res.outputs[0].meta
    assert (meta["clauses"][0]["take"], meta["lines"][0]["wer"]) == (1, 0.0) and len(fake.calls) == 2
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


# --- natural speech (plan 2026-11-22-video-quality-autonomous item 7) --------------------------------------------
NATURAL = {"split_clauses": True, "clause_min_words": 1, "clause_gap_s": [0.18, 0.30], "sentence_gap_s": [0.45, 0.70],
           "pause_s": 0.25}
TOL = 2e-3   # timings are rounded to the millisecond and gaps to whole samples


@pytest.mark.parametrize("text, expected", [
    ("Okay, so I went — and it rained.", ["Okay,", "so I went —", "and it rained."]),
    ("It cost $1,000. Then 3.5 more.", ["It cost $1,000.", "Then 3.5 more."]),            # numbers stay whole
    ("Meet at 5:30, okay?", ["Meet at 5:30,", "okay?"]),
    ("Dr. Smith said hi; e.g. this, at 5 p.m. Sharp.", ["Dr. Smith said hi;", "e.g. this,", "at 5 p.m. Sharp."]),
    ("J. K. Rowling wrote it.", ["J. K. Rowling wrote it."]),                              # initials
    ("From 10–20 people, well-known ones.", ["From 10–20 people,", "well-known ones."]),   # a range, a hyphen
    ("I was going to—never mind.", ["I was going to —", "never mind."]),
    ("Wait... what? No!", ["Wait...", "what?", "No!"]),
    ('I said "stop," and left.', ['I said "stop,"', "and left."]),
    ("hello there", ["hello there"]),
    ("", []),
])
def test_lines_split_into_clauses_never_inside_numbers_or_abbreviations(text, expected):
    assert voices.clauses(text) == expected


def test_clause_boundaries_say_how_long_the_pause_is():
    assert voices.segments("One, two. Three") == [("One,", "clause"), ("two.", "sentence"), ("Three", "end")]
    assert voices.segments("Wait... What now?") == [("Wait...", "sentence"), ("What now?", "end")]


def test_a_too_short_clause_joins_its_neighbour():
    assert voices.clauses("Okay, so this morning did not go to plan.", min_words=2) == [
        "Okay, so this morning did not go to plan."]
    assert voices.clauses("So I went, home.", min_words=2) == ["So I went, home."]
    assert voices.clauses("Yes. I know.", min_words=2) == ["Yes.", "I know."]        # a sentence stays its own
    assert voices.clauses("I missed the early train, spilled half my coffee, and still made it.", min_words=2) == [
        "I missed the early train,", "spilled half my coffee,", "and still made it."]


def _lines(*items):
    return [{"text": t, "delivery": "", "pause_after_s": p} for t, p in items]


def test_gaps_fall_in_their_ranges_and_repeat_for_a_seed():
    cfg = tts.speech_config(NATURAL)
    lines = _lines(("One, two, three. Four, five.", None), ("Six.", None))
    a, b, c = tts.plan(lines, cfg, 5), tts.plan(lines, cfg, 5), tts.plan(lines, cfg, 6)
    gaps = [u["gap_after_s"] for u in a]
    assert [u["text"] for u in a] == ["One,", "two,", "three.", "Four,", "five.", "Six."]
    assert gaps == [u["gap_after_s"] for u in b] and gaps != [u["gap_after_s"] for u in c]
    for i in (0, 1, 3):
        assert 0.18 <= gaps[i] <= 0.30
    for i in (2, 4):                                    # a sentence end, and the end of a line with no pause given
        assert 0.45 <= gaps[i] <= 0.70
    assert gaps[5] == 0.0                               # nothing after the last piece
    assert len({gaps[0], gaps[1], gaps[3]}) == 3        # not metronomic
    lines[0]["pause_after_s"] = 1.25                    # a line's own pause wins
    assert tts.plan(lines, cfg, 5)[4]["gap_after_s"] == 1.25


def test_without_natural_keys_a_line_is_one_piece_joined_with_pause_s():
    cfg = tts.speech_config({"pause_s": 0.3})
    units = tts.plan(_lines(("One, two. Three.", None), ("Four.", None)), cfg, 1)
    assert [(u["text"], u["gap_after_s"]) for u in units] == [("One, two. Three.", 0.3), ("Four.", 0.0)]
    assert cfg["trim"] is None and cfg["fade_ms"] == 0 and cfg["breaths"] is None and cfg["disfluency"] is None


@pytest.mark.parametrize("bad", [
    {"clause_gap_s": [0.3, 0.1]}, {"clause_gap_s": "short"}, {"sentence_gap_s": [-1, 0.5]},
    {"trim": {"threshold_db": 10, "frame_ms": 10}}, {"trim": "yes"}, {"clause_min_words": 0},
    {"breaths": {"enabled": True, "every_n_sentences": 2, "gain_db": -40, "seconds": [0.1, 0.2]}},   # no band
    {"disfluency": {"enabled": True, "rate": 0.1, "fillers": []}},
    {"disfluency": {"enabled": True, "rate": 2, "fillers": ["um"]}},
])
def test_bad_natural_speech_params_are_not_retried(bad):
    with pytest.raises(NonRetryable):
        tts.speech_config(bad)


def test_clause_timings_are_monotonic_and_lines_cover_their_clauses(fake, ctx, tmp_path):
    res = _speak(ctx, tmp_path, [{"text": "Okay, so I went. And then it rained."},
                                 {"text": "Fine.", "pause_after_s": 0.4}, {"text": "Bye now."}],
                 params={**NATURAL, "trim": {"threshold_db": -40, "frame_ms": 10, "pad_ms": 20}, "fade_ms": 8})
    meta = res.outputs[0].meta
    assert [c["text"] for c in fake.calls] == ["Okay,", "so I went.", "And then it rained.", "Fine.", "Bye now."]
    clauses = meta["clauses"]
    assert [c["line"] for c in clauses] == [0, 0, 0, 1, 2]
    for prev, cur in zip(clauses, clauses[1:]):
        assert prev["start_s"] < prev["end_s"] <= cur["start_s"] < cur["end_s"]
    assert 0.18 - TOL <= clauses[1]["start_s"] - clauses[0]["end_s"] <= 0.30 + TOL     # a clause gap
    assert 0.45 - TOL <= clauses[2]["start_s"] - clauses[1]["end_s"] <= 0.70 + TOL     # a sentence gap
    assert clauses[4]["start_s"] - clauses[3]["end_s"] == pytest.approx(0.4, abs=TOL)  # the line's own pause
    for line in meta["lines"]:
        mine = [clauses[i] for i in line["clauses"]]
        assert (line["start_s"], line["end_s"]) == (mine[0]["start_s"], mine[-1]["end_s"])
    assert [line["clauses"] for line in meta["lines"]] == [[0, 1, 2], [3], [4]]
    assert meta["seconds"] == pytest.approx(clauses[-1]["end_s"], abs=0.01) and meta["fillers"] == []
    with wave.open(str(res.outputs[0].path)) as w:
        assert w.getnframes() / SR == pytest.approx(clauses[-1]["end_s"], abs=TOL)


def test_each_clause_is_scored_against_its_words_without_the_delivery(fake, ctx, tmp_path):
    res = _speak(ctx, tmp_path, [{"text": "I knew it, (laughing) really.", "delivery": "whispering"}], params=NATURAL)
    assert [c["text"] for c in fake.calls] == ["(whispering)I knew it,", "(whispering, laughing)really."]
    clauses = res.outputs[0].meta["clauses"]
    assert [c["text"] for c in clauses] == ["I knew it,", "really."]
    assert [c["wer"] for c in clauses] == [0.0, 0.0] and res.outputs[0].meta["lines"][0]["wer"] == 0.0


def test_a_clause_every_take_garbled_gets_one_more_round_of_seeds(fake, ctx, tmp_path):
    fake.transcripts = ["garbage words here", "I knew it", "really"]
    res = _speak(ctx, tmp_path, [{"text": "I knew it, really."}],
                 params={**NATURAL, "max_wer": 0.1, "clause_retry_wer_factor": 2.0})
    clauses = res.outputs[0].meta["clauses"]
    assert [c["seed"] for c in fake.calls] == [7, 9, 8]      # clause 0, its retry (after every first-round seed), 1
    assert (clauses[0]["take"], clauses[0]["retried"], clauses[0]["wer"]) == (1, True, 0.0)
    assert clauses[1]["retried"] is False


def test_a_line_still_garbled_after_assembly_fails_the_job_for_a_retry(fake, ctx, tmp_path):
    fake.transcripts = ["nothing like it", "at all"]
    with pytest.raises(RuntimeError, match="line 0"):          # retryable: not NonRetryable
        _speak(ctx, tmp_path, [{"text": "I knew it, really."}], params={**NATURAL, "line_max_wer": 0.25})


def test_a_later_attempt_draws_new_seeds(fake, ctx, tmp_path):
    ctx.job = type("Job", (), {"attempts": 2})()
    _speak(ctx, tmp_path, [{"text": "Hi."}], params=NATURAL)
    assert fake.calls[0]["seed"] == 7 + 2                   # past every seed the first attempt could use (2 x 1 x 1)


def test_without_disfluency_the_scripted_words_are_spoken_untouched(fake, ctx, tmp_path):
    lines = [{"text": "So, like, I went and it rained."}, {"text": "Um, okay."}]
    res = _speak(ctx, tmp_path, lines, params=NATURAL)
    spoken = " ".join(voices.strip_delivery(c["text"]) for c in fake.calls)
    assert spoken.split() == "So, like, I went and it rained. Um, okay.".split()
    assert res.outputs[0].meta["fillers"] == [] and all(c["filler"] is None for c in res.outputs[0].meta["clauses"])


def test_a_filler_is_spoken_but_kept_out_of_the_script(fake, ctx, tmp_path):
    dis = {"enabled": True, "rate": 1.0, "max_per_line": 1, "fillers": ["um"]}
    res = _speak(ctx, tmp_path, [{"text": "I went, and it rained."}], params={**NATURAL, "disfluency": dis})
    assert [c["text"] for c in fake.calls] == ["Um, I went,", "and it rained."]        # at most one per line
    meta = res.outputs[0].meta
    assert [c["text"] for c in meta["clauses"]] == ["I went,", "and it rained."]
    assert meta["clauses"][0]["filler"] == "um" and meta["fillers"] == [{"clause": 0, "line": 0, "word": "um"}]
    assert meta["lines"][0]["text"] == "I went, and it rained." and meta["lines"][0]["wer"] == 0.0
    # a clause that already opens with the filler word is left alone
    so = tts.speech_config({**NATURAL, "disfluency": {**dis, "fillers": ["so"]}})
    units = tts.plan(_lines(("So I went.", None)), so, 1)
    assert units[0]["spoken"] == "So I went." and units[0]["filler"] is None


def test_breaths_go_before_every_nth_sentence_only_when_enabled():
    b = {"enabled": True, "every_n_sentences": 1, "gain_db": -42, "seconds": [0.12, 0.22], "band_hz": [500, 4000],
         "lead_s": 0.05}
    lines = _lines(("One. Two, three.", None), ("Four.", None))

    def flags(breaths):
        return [u["breath"] for u in tts.plan(lines, tts.speech_config({**NATURAL, "breaths": breaths}), 1)]

    assert flags(b) == [False, True, False, True]           # each sentence start after the first; never mid-sentence
    assert flags({**b, "enabled": False}) == [False] * 4
    assert flags({**b, "every_n_sentences": 2}) == [False, False, False, True]   # the third sentence


def test_a_breath_is_quiet_band_limited_noise_reproducible_by_seed():
    x = tts.breath(SR, 0.15, gain_db=-42, band_hz=(500, 4000), seed=3)
    assert len(x) == round(0.15 * SR) and x.dtype == np.float32
    assert float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) == pytest.approx(10 ** (-42 / 20), rel=1e-3)
    assert abs(float(x[0])) < 1e-6 and abs(float(x[-1])) < 1e-6           # rises and falls
    power = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    assert power[(f >= 400) & (f <= 4500)].sum() / power.sum() > 0.95
    assert np.array_equal(x, tts.breath(SR, 0.15, gain_db=-42, band_hz=(500, 4000), seed=3))
    assert not np.array_equal(x, tts.breath(SR, 0.15, gain_db=-42, band_hz=(500, 4000), seed=4))


def test_assembly_places_gaps_exactly_and_a_breath_inside_its_gap():
    sr = 1000
    one = np.ones(100, dtype=np.float32)
    puff = np.full(50, 0.01, dtype=np.float32)
    wav, spans = tts.assemble([one, one, one], [0.2, 0.03, 0.0], sr, breaths=[None, puff, puff], lead_s=0.04)
    assert spans == [(0, 100), (300, 400), (430, 530)] and len(wav) == 530
    assert np.all(wav[100:210] == 0) and np.allclose(wav[210:260], 0.01) and np.all(wav[260:300] == 0)
    assert np.all(wav[400:430] == 0)                      # a breath longer than its gap is left out


def _tone(sr, seconds, hz=440.0, amp=0.3):
    t = np.arange(int(round(sr * seconds))) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_trim_cuts_leading_and_trailing_silence_by_energy():
    tone = _tone(SR, 0.3)
    hiss = np.random.default_rng(0).standard_normal(int(0.2 * SR)).astype(np.float32) * 1e-4   # -60 dB under it
    clip = np.concatenate([hiss, tone, hiss[: int(0.1 * SR)]])
    cut = tts.trim(clip, SR, threshold_db=-40, frame_ms=10)
    assert len(cut) == len(tone) and np.array_equal(cut, tone)
    padded = tts.trim(clip, SR, threshold_db=-40, frame_ms=10, pad_ms=5)
    assert len(padded) == len(tone) + 2 * int(0.005 * SR)
    silent = np.zeros(SR // 10, dtype=np.float32)
    assert len(tts.trim(silent, SR, threshold_db=-40, frame_ms=10)) == len(silent)   # nothing to keep: unchanged


def test_fade_ramps_both_edges_without_changing_length():
    sr = 1000
    x = tts.fade(np.ones(1000, dtype=np.float32), sr, 10)
    assert len(x) == 1000 and x[0] == 0.0 and x[-1] == 0.0
    assert np.all(np.diff(x[:10]) > 0) and np.all(np.diff(x[-10:]) < 0) and np.all(x[10:-10] == 1.0)
    assert len(tts.fade(np.ones(5, dtype=np.float32), sr, 50)) == 5     # a fade never overlaps itself


# --- whole line vs clauses (2026-11-23): whole lines by default, both on request for the comparison --------------
WHOLE = {**{k: v for k, v in NATURAL.items() if k != "split_clauses"}, "synthesis": "whole_line"}


def test_whole_line_synthesis_speaks_each_line_in_one_piece(fake, ctx, tmp_path):
    res = _speak(ctx, tmp_path, [{"text": "Okay, so I went. And then it rained."}, {"text": "Fine."}], params=WHOLE)
    assert [c["text"] for c in fake.calls] == ["Okay, so I went. And then it rained.", "Fine."]
    assert res.outputs[0].meta["synthesis"] == "whole_line" and len(res.outputs) == 1
    # the older key still works when the new one is absent; the new one wins
    assert tts.synthesis({"split_clauses": True}) == "clauses" and tts.synthesis({}) == "whole_line"
    assert tts.synthesis({"synthesis": "clauses", "split_clauses": False}) == "clauses"
    with pytest.raises(NonRetryable):
        tts.speech_config({"synthesis": "words"})


def test_compare_makes_both_syntheses_from_the_same_seeds(fake, ctx, tmp_path):
    res = _speak(ctx, tmp_path, [{"text": "Okay, so I went."}], params=WHOLE, compare=True)
    main, alt = res.outputs
    assert (main.meta["synthesis"], alt.meta["synthesis"]) == ("whole_line", "clauses")
    assert main.path.name == "speech.wav" and alt.path.name == "speech_clauses.wav"   # production reads the first
    assert alt.meta["compare_of"] == 0 and alt.meta["failed_lines"] == []
    assert [(c["output"], c["synthesis"]) for c in main.meta["compare"]] == [(0, "whole_line"), (1, "clauses")]
    assert [c["text"] for c in fake.calls] == ["Okay, so I went.", "Okay,", "so I went."]
    assert fake.calls[0]["seed"] == fake.calls[1]["seed"] == 7
    assert res.recipe["compare"] is True
    assert res.meta["output_seconds"] == pytest.approx(main.meta["seconds"] + alt.meta["seconds"], abs=0.01)
    with pytest.raises(NonRetryable):
        tts.validate_speak({"ref_uri": "s3://b/r.wav", "ref_text": "x", "lines": [{"text": "hi"}], "compare": "yes"})


def test_a_garbled_comparison_never_fails_the_job(fake, ctx, tmp_path):
    fake.transcripts = ["Okay so I went", "nothing", "alike"]
    res = _speak(ctx, tmp_path, [{"text": "Okay, so I went."}], params={**WHOLE, "line_max_wer": 0.25}, compare=True)
    assert res.outputs[0].meta["lines"][0]["wer"] == 0.0 and res.outputs[1].meta["failed_lines"] == [0]


# --- lost first words (breadth rerun 2026-11-24: "Two am. I take it back" heard as "and take it back") -------------
def test_trim_keeps_a_soft_first_word():
    """The trim is not what ate "Two am.": it cuts only frames more than threshold_db under the loudest, and a tired,
    quiet first word 20 dB under the rest of the line is far above that."""
    soft, loud = _tone(SR, 0.25, amp=0.03), _tone(SR, 0.6, amp=0.3)
    clip = np.concatenate([np.zeros(SR // 10, dtype=np.float32), soft, np.zeros(SR // 5, dtype=np.float32), loud])
    cut = tts.trim(clip, SR, threshold_db=-40, frame_ms=10, pad_ms=30)
    assert len(cut) >= len(soft) + SR // 5 + len(loud)


def test_a_prompt_tail_gives_the_line_a_boundary_to_start_from(fake, ctx, tmp_path):
    _speak(ctx, tmp_path, [{"text": "Two am. I take it back."}], params={"prompt_tail_silence_s": 0.3})
    call = fake.calls[0]
    assert call["prompt_wav_path"] != call["reference_wav_path"] and call["reference_wav_path"].endswith("reference.wav")
    padded, sr = tts.read_wav(tts.Path(call["prompt_wav_path"]))
    original, _ = tts.read_wav(tts.Path(call["reference_wav_path"]))
    assert sr == SR and len(padded) == len(original) + int(0.3 * SR) and not padded[len(original):].any()
    fake.calls.clear()
    _speak(ctx, tmp_path, [{"text": "Two am."}])                    # off by default: the reference itself
    assert fake.calls[0]["prompt_wav_path"] == fake.calls[0]["reference_wav_path"]
    with pytest.raises(NonRetryable):
        tts.speech_config({"prompt_tail_silence_s": -1})


# --- the delivery bank: a line continues from its delivery's clip, her base clip the timbre reference ------------
def test_a_bank_line_continues_from_its_clip_and_keeps_the_base_timbre(fake, ctx, tmp_path):
    clip = tts.write_wav(tmp_path / "deadpan.wav", np.zeros(SR // 2, dtype=np.float32), SR)
    bank_uri = ctx.store.put_file(clip, "voices/bank-deadpan.wav").uri
    res = _speak(ctx, tmp_path, [
        {"text": "Two am. I take it back.", "bank": "deadpan",
         "prompt": {"ref_uri": bank_uri, "ref_text": "Great. Love that for me."}},
        {"text": "Bye."}])
    bank_call, plain = fake.calls
    assert bank_call["prompt_text"] == "Great. Love that for me." and "prompt_" in bank_call["prompt_wav_path"]
    assert bank_call["reference_wav_path"].endswith("reference.wav")            # timbre: her base clip
    assert bank_call["text"] == "Two am. I take it back."                       # nothing to read aloud
    assert plain["prompt_text"] == "Okay so this happened." and plain["prompt_wav_path"].endswith("reference.wav")
    assert [line["bank"] for line in res.outputs[0].meta["lines"]] == ["deadpan", None]
    assert "embedding" not in res.outputs[0].meta


def test_a_bank_clip_job_reports_its_speaker_embedding(fake, ctx, tmp_path):
    res = _speak(ctx, tmp_path, [{"text": "No, wait, this worked!", "delivery": "excited"}], embed=True,
                 params={"clone": "controllable"})
    meta = res.outputs[0].meta
    assert meta["embedding"] and "pitch_var" in meta
    assert fake.calls[0]["text"] == "(excited)No, wait, this worked!" and "prompt_text" not in fake.calls[0]


@pytest.mark.parametrize("line", [
    {"text": "hi", "prompt": "s3://b/c.wav"},
    {"text": "hi", "prompt": {"ref_uri": "s3://b/c.wav"}},
    {"text": "hi", "prompt": {"ref_text": "x"}},
    {"text": "hi", "bank": 3},
])
def test_a_bad_bank_prompt_is_not_retried(line):
    with pytest.raises(NonRetryable):
        tts.validate_speak({"ref_uri": "s3://b/r.wav", "ref_text": "x", "lines": [line]})
    with pytest.raises(NonRetryable):
        tts.validate_speak({"ref_uri": "s3://b/r.wav", "ref_text": "x", "lines": [{"text": "hi"}], "embed": "yes"})
