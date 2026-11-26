"""Expressive engines and casting on the worker (worker/handlers/tts_engines.py), with the engines, transcriber,
embedder and judges replaced by fakes: an engine gets only the words (its emotion control carries the delivery), a
failing line is regenerated at the next emotion strength and the best take kept, the scores land in the meta, a cast
scores every candidate, and the engine server speaks its line protocol."""

from __future__ import annotations

import json
import sys
import textwrap

import numpy as np
import pytest

from manyworlds.storage import LocalStore
from manyworlds.worker.handlers import tts_engines as te
from manyworlds.worker.handlers import speech_audio as vox
from manyworlds.worker.handlers import voice_judge as vj
from manyworlds.worker.runner import JobContext, NonRetryable, handlers_for

SR = 22050


class FakeEngine:
    """An engine's place: a quarter second per word (4 words a second, on pace); `flat_until` calls come back flat
    (quiet, no pitch movement). Designed references are quieter still, so the fake transcriber knows them."""
    sample_rate = SR
    anchor = ""

    def __init__(self, flat_until=0):
        self.calls, self.flat_until = [], flat_until

    def speak(self, *, text, ref, ref_text, emotion, strength, seed, out):
        self.calls.append({"text": text, "emotion": emotion, "strength": strength, "seed": seed})
        n = int(SR * max(0.5, len(text.split()) / 4.0))
        return np.full(n, 0.1 if len(self.calls) <= self.flat_until else 0.2, dtype=np.float32)

    def design(self, *, description, text, seed, out):
        self.calls.append({"design": description, "seed": seed})
        return vox.write_wav(out, np.full(SR, 0.05, dtype=np.float32), SR)

    def hear(self, wav, sr):
        if abs(float(np.max(wav)) - 0.05) < 1e-3:
            return self.anchor
        return self.calls[-1].get("text", "")


@pytest.fixture()
def fake(monkeypatch, policy):
    f = FakeEngine()
    f.anchor = policy.get("voice", "anchor_text")
    monkeypatch.setattr(te, "_engine", lambda name, params: f)
    monkeypatch.setitem(te.DESIGNERS, "qwen3_design", lambda params: f)
    monkeypatch.setattr(vox, "_asr", lambda params: f.hear)
    monkeypatch.setattr(vox, "_embedder", lambda params: (lambda wav, sr: [1.0, 0.0, 0.0]))
    monkeypatch.setattr(vj, "_mos", lambda params: (lambda wav, sr: 4.1))
    # pitch movement follows loudness: quiet takes are flat, the 0.2 takes move
    monkeypatch.setattr(vj, "_f0", lambda wav, sr, params: np.array(
        [200.0] * 50 if float(np.max(wav)) < 0.15 else [180.0, 200.0, 240.0, 210.0] * 20))
    monkeypatch.setattr(vj, "_gender", lambda params: (lambda wav, sr: {"label_0": 0.92, "label_1": 0.08}))
    monkeypatch.setattr(vj, "_age", lambda params: (lambda wav, sr: {"20s": 0.7, "30s": 0.3}))
    return f


@pytest.fixture()
def ctx(tmp_path):
    store = LocalStore(tmp_path / "objects")
    (tmp_path / "work").mkdir()
    ref = vox.write_wav(tmp_path / "ref.wav", np.full(SR, 0.1, dtype=np.float32), SR)
    store.put_file(ref, "voices/ref.wav")
    return JobContext(job=None, workdir=tmp_path / "work", store=store)


def _params(policy, **check):
    from manyworlds.creator import voice_lab
    p = voice_lab.engine_params(policy, ["indextts"])
    p["line_check"] = {**p["line_check"], **check}
    return p


@pytest.fixture()
def ref_uri(ctx, tmp_path):
    return ctx.store.put_file(tmp_path / "ref.wav", "voices/ref.wav").uri


def test_the_new_kinds_have_handlers():
    assert set(handlers_for(["voice.speak.indextts", "voice.speak.qwen3", "voice.cast"])) == {
        "voice.speak.indextts", "voice.speak.qwen3", "voice.cast"}


def test_the_engine_hears_only_the_words_and_takes_the_emotion_as_its_own_control(fake, ctx, ref_uri, policy):
    emotion = {"name": "confiding", "vector": [0.1, 0, 0.1, 0, 0, 0.15, 0, 0.45], "alpha": 1.0}
    res = te.speak_indextts({"ref_uri": ref_uri, "ref_text": "Okay so this happened.", "seed": 5,
                             "lines": [{"text": "(quiet, sincere) I almost quit last spring.", "emotion": emotion}],
                             "params": _params(policy)}, ctx)
    spoken = [c for c in fake.calls if "text" in c]
    assert spoken[0]["text"] == "I almost quit last spring." and spoken[0]["emotion"] == emotion
    meta = res.outputs[0].meta
    assert meta["engine"] == "indextts" and meta["sample_rate"] == SR
    line = meta["lines"][0]
    assert line["delivery"] == "quiet, sincere" and line["emotion"] == "confiding"
    assert line["judge"]["passed"] and line["judge"]["repairs"] == 0 and meta["quality"]["passed"] == 1


def _robotic_when_quiet(monkeypatch):
    """Quiet (flat) takes also sound robotic: a MOS under voice.judge.min_mos, a material defect."""
    monkeypatch.setattr(vj, "_mos", lambda params: (lambda wav, sr: 2.0 if float(np.max(wav)) < 0.15 else 4.1))


def test_a_flat_line_keeps_its_take_and_is_recorded_never_regenerated(fake, ctx, ref_uri, policy):
    """Generous publishing (owner 2026-11-26): flat or off-pace is a preference, not a defect."""
    fake.flat_until = 1                      # the first take is flat (and otherwise fine)
    res = te.speak_indextts({"ref_uri": ref_uri, "ref_text": "Okay.", "seed": 5,
                             "lines": [{"text": "So I get to the market really early, right.",
                                        "emotion": {"name": "storytelling", "vector": [0.3] + [0] * 7}}],
                             "params": _params(policy)}, ctx)
    assert [c["strength"] for c in fake.calls] == [1.0]          # spoken once
    judge = res.outputs[0].meta["lines"][0]["judge"]
    assert not judge["passed"] and judge["repairs"] == 0 and not judge["defect"]
    assert any(r.startswith("flat") for r in judge["reasons"])   # recorded


def test_a_robotic_line_is_regenerated_at_the_next_strength_and_the_best_take_kept(fake, ctx, ref_uri, policy,
                                                                                    monkeypatch):
    _robotic_when_quiet(monkeypatch)
    fake.flat_until = 1                      # the first take is robotic, the repair is fine
    res = te.speak_indextts({"ref_uri": ref_uri, "ref_text": "Okay.", "seed": 5,
                             "lines": [{"text": "So I get to the market really early, right.",
                                        "emotion": {"name": "storytelling", "vector": [0.3] + [0] * 7}}],
                             "params": _params(policy)}, ctx)
    strengths = [c["strength"] for c in fake.calls]
    steps = policy.get("voice", "line_check", "strength_steps")
    assert strengths == [1.0, steps[0]]
    judge = res.outputs[0].meta["lines"][0]["judge"]
    assert judge["passed"] and judge["repairs"] == 1 and judge["f0_std_st"] > 1.2 and not judge["defect"]
    seeds = [c["seed"] for c in fake.calls]
    assert seeds[1] != seeds[0]


def test_a_line_that_never_passes_keeps_its_best_take_and_says_why(fake, ctx, ref_uri, policy, monkeypatch):
    _robotic_when_quiet(monkeypatch)
    fake.flat_until = 99
    res = te.speak_indextts({"ref_uri": ref_uri, "ref_text": "Okay.", "seed": 5,
                             "lines": [{"text": "So I get to the market really early, right."}],
                             "params": _params(policy)}, ctx)
    judge = res.outputs[0].meta["lines"][0]["judge"]
    assert len(fake.calls) == 1 + policy.get("voice", "line_check", "repairs")
    assert not judge["passed"] and judge["defect"] and any(r.startswith("robotic") for r in judge["reasons"])
    assert res.outputs[0].meta["quality"] == {"lines": 1, "passed": 0, "composite": judge["composite"],
                                              "repairs": judge["repairs"]}


def test_the_line_check_off_leaves_lines_unjudged(fake, ctx, ref_uri, policy):
    res = te.speak_indextts({"ref_uri": ref_uri, "ref_text": "Okay.", "seed": 5, "lines": [{"text": "Hi there."}],
                             "params": _params(policy, enabled=False)}, ctx)
    assert "judge" not in res.outputs[0].meta["lines"][0] and "quality" not in res.outputs[0].meta


def test_an_engine_without_its_settings_is_not_retried(ctx, ref_uri):
    with pytest.raises(NonRetryable):
        te.speak_qwen3({"ref_uri": ref_uri, "ref_text": "Okay.", "lines": [{"text": "Hi."}], "params": {}}, ctx)


@pytest.mark.parametrize("bad", [
    {"engine": "nope"}, {"designer": "nope"}, {"description": " "}, {"script": []}, {"script": ["x"]}, {"n": 0},
    {"script": [{"text": "hi", "emotion": [1, 2]}]},
])
def test_a_bad_cast_payload_is_not_retried(bad):
    good = {"engine": "indextts", "designer": "qwen3_design", "description": "a warm woman", "anchor_text": "Okay.",
            "n": 2, "script": [{"text": "hi"}], "params": {}}
    with pytest.raises(NonRetryable):
        te.validate_cast({**good, **bad})


def test_a_cast_designs_each_candidate_then_scores_every_line_in_the_engine(fake, ctx, policy):
    from manyworlds.creator import voice_lab
    payload = voice_lab.cast_payload(policy, description="A 27-year-old woman, warm", engine="indextts", seed=11, n=3)
    res = te.cast(payload, ctx)
    designs = [c for c in fake.calls if "design" in c]
    assert [c["seed"] for c in designs] == [11, 1011, 2011]
    refs = [o for o in res.outputs if o.kind == "audio"]
    assert len(refs) == 3 and len(res.outputs) == 6
    script = policy.get("voice", "casting", "script")
    for i, out in enumerate(refs):
        m = out.meta
        assert m["index"] == i and m["role"] == "cast_reference" and m["engine"] == "indextts"
        assert [x["name"] for x in m["lines"]] == [s["name"] for s in script]
        assert all({"wer", "mos", "f0_std_st", "wps", "ref_distance", "passed", "composite"} <= set(x)
                   for x in m["lines"])
        assert m["who"]["gender_p"]["female"] == 0.92 and m["who"]["age"] == 28.0   # mapped LABEL_0, band midpoints
        assert m["ref_wer"] == 0.0 and m["embedding"] == [1.0, 0.0, 0.0]
    excited = next(c for c in fake.calls if c.get("text", "").startswith("No, wait"))
    assert excited["emotion"]["name"] == "excited" and "(" not in excited["text"]


def test_the_engine_server_speaks_one_json_line_per_request(tmp_path):
    """The client half of worker/tts_server.py's protocol, against a stand-in server."""
    script = tmp_path / "server.py"
    script.write_text(textwrap.dedent("""
        import json, sys
        print(json.dumps({"ok": True, "ready": "x"}), flush=True)
        for raw in sys.stdin:
            r = json.loads(raw)
            if r["op"] == "quit":
                break
            ok = r["op"] == "speak"
            print(json.dumps({"ok": ok, "out": r.get("out"), "error": None if ok else "nope"}), flush=True)
    """))
    s = te._Server([sys.executable, str(script)], load_timeout_s=30)
    try:
        assert s.call({"op": "speak", "out": "a.wav"}, 30)["out"] == "a.wav"
        with pytest.raises(RuntimeError, match="nope"):
            s.call({"op": "design"}, 30)
    finally:
        s.close()
    assert s.proc.poll() is not None


def test_the_server_writes_int16_and_float_audio(tmp_path):
    from manyworlds.worker import tts_server
    seconds = tts_server._write(str(tmp_path / "a.wav"), np.zeros(SR, dtype=np.int16), SR)
    assert seconds == 1.0
    wav, sr = vox.read_wav(tmp_path / "a.wav")
    assert sr == SR and len(wav) == SR
    assert tts_server._write(str(tmp_path / "b.wav"), np.full(SR // 2, 0.5, dtype=np.float32), SR) == 0.5
    json.dumps(tts_server.INDEXTTS_CACHE)    # never the MaskGCT codec (CC-BY-NC): only these three sub-models
    assert not any("MaskGCT" in str(v) for v in tts_server.INDEXTTS_CACHE.values())


def test_a_cast_keeps_what_each_engine_actually_said_apart_from_the_references(fake, ctx, policy):
    """2026-11-26: every bake-off clip the owner heard was the Qwen3 VoiceDesign reference; the engines' own lines
    were scored and thrown away."""
    from manyworlds.creator import voice_lab
    payload = voice_lab.cast_payload(policy, description="A 27-year-old woman, warm", engine="indextts", seed=11, n=2)
    res = te.cast(payload, ctx)
    refs = [o for o in res.outputs if o.kind == "audio"]
    spoken = [o for o in res.outputs if o.kind == "audio_lines"]
    assert len(refs) == 2 and len(spoken) == 2 and res.outputs[:2] == refs      # references first: selection unchanged
    script = policy.get("voice", "casting", "script")
    for i, o in enumerate(spoken):
        assert o.meta["index"] == i and o.meta["role"] == "cast_lines" and o.meta["engine"] == "indextts"
        assert [x["name"] for x in o.meta["lines"]] == [s["name"] for s in script] and o.path.exists()
        assert o.meta["lines"][1]["start_s"] >= o.meta["lines"][0]["end_s"] + payload["params"]["listen_gap_s"] - 0.01
