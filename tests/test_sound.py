"""Sound design (manyworlds.sound): the mix graph (voice-led ducking, room tone, effects, impulse response, mic EQ),
the licence-checked library, and the generator handlers that stay off until the owner says so."""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from manyworlds import media, policy as policy_mod, sound
from manyworlds.worker import runner
from manyworlds.worker.handlers import sound_gen
from manyworlds.worker.runner import JobContext, NonRetryable


def _with(policy, **blocks):
    body = dict(policy.body)
    s = dict(body.get("sound") or {})
    for k, v in blocks.items():
        s[k] = {**(s.get(k) or {}), **v} if isinstance(v, dict) else v
    body["sound"] = s
    return policy_mod.from_dict(body)


def test_the_shipped_policy_has_sound_off_an_empty_library_and_the_sfx_licence_not_accepted(raw_policy):
    c = raw_policy.get("sound")
    assert c["enabled"] is False
    assert c["library"]["entries"] == [] and c["impulse_responses"] == {}
    assert c["sfx"]["enabled"] is False and c["sfx"]["licence_accepted"] is False
    assert c["music"]["enabled"] is False
    assert not sound.enabled(raw_policy)


def test_music_is_ducked_under_the_voice_with_room_tone_and_timed_effects(policy):
    p = sound.plan(has_voice=True, room_tone="room.wav", effects=[("click.wav", 1.5, None), ("door.wav", 3.25, -6)],
                   music="bed.wav", policy=policy)
    m = policy.get("sound", "mix")
    assert p.inputs == [["-stream_loop", "-1", "-i", "room.wav"], ["-i", "click.wav"], ["-i", "door.wav"],
                        ["-stream_loop", "-1", "-i", "bed.wav"]]
    g = p.graph
    assert g.startswith("[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[vo]")
    assert f"[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume={m['room_tone_gain_db']}dB[room]" in g
    assert f"volume={m['effect_gain_db']}dB,adelay=1500|1500[fx0]" in g
    assert "volume=-6dB,adelay=3250|3250[fx1]" in g
    assert f"[4:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume={m['music_gain_db']}dB[mus_in]" in g
    # the voice keys the compressor on the music, and the voice itself goes on to the mix
    assert "[vo]asplit=2[vo_mix][vo_key]" in g
    assert "[mus_in][vo_key]sidechaincompress=threshold=0.05:ratio=6:attack=80:release=400:makeup=1[mus]" in g
    assert g.endswith("[vo_mix][room][fx0][fx1][mus]amix=inputs=5:duration=first:dropout_transition=0:normalize=0[mix]")
    assert "loudnorm" not in g        # loudness belongs to the finish stage
    assert p.out == "mix"


def test_duck_attack_and_release_are_clamped_to_the_policy_limits(policy):
    fast = _with(policy, mix={**policy.get("sound", "mix"), "duck": {"threshold": 5, "ratio": 100, "attack_ms": 5,
                                                                      "release_ms": 5000}})
    f = sound.duck_filter(fast)
    assert "threshold=1:" in f and "ratio=20:" in f and "attack=50:" in f and "release=500:" in f


def test_an_impulse_response_adds_room_to_the_voice_at_a_low_wet_share_and_the_mic_eq_applies(policy):
    p = sound.plan(has_voice=True, ir="ir.wav", mic="phone", policy=policy)
    eq = policy.get("sound", "mic_eq", "phone")
    assert p.inputs == [["-i", "ir.wav"]]
    g = p.graph
    assert f"channel_layouts=stereo,highpass=f={eq['highpass']},lowpass=f={eq['lowpass']}[vo]" in g
    assert "[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=mono[ir]" in g
    assert "[vo]asplit=2[vo_dry][vo_wet_in];[vo_wet_in][ir]afir[vo_wet]" in g
    wet = policy.get("sound", "mix", "room_wet")
    assert f"[vo_dry][vo_wet]amix=inputs=2:weights='1 {wet}':duration=first:dropout_transition=0:normalize=0[vo_room]" in g
    assert g.endswith("[vo_room]anull[mix]")


def test_a_silent_video_gets_a_silent_stand_in_voice_and_its_music_is_not_ducked(policy):
    p = sound.plan(has_voice=False, duration_s=4.5, music="bed.wav", policy=policy)
    assert p.inputs[0] == ["-f", "lavfi", "-t", "4.5", "-i", "anullsrc=r=48000:cl=stereo"]
    assert "[1:a]aresample=48000" in p.graph and "sidechaincompress" not in p.graph
    assert p.graph.endswith("[vo][mus_in]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mix]")
    with pytest.raises(ValueError):
        sound.plan(has_voice=False, music="bed.wav", policy=policy)
    with pytest.raises(ValueError):
        sound.plan(has_voice=True, effects=[("x.wav", -1, None)], policy=policy)


def test_the_library_refuses_sounds_without_a_licence_or_source_or_with_an_unlisted_licence(policy):
    entries = [
        {"tag": "Kettle Click", "uri": "s3://sfx/kettle.wav", "licence": "CC0-1.0", "source": "own pack 2026-11"},
        {"tag": "fridge hum", "uri": "s3://sfx/fridge.wav", "source": "somewhere"},                     # no licence
        {"tag": "street", "uri": "s3://sfx/street.wav", "licence": "CC-BY-NC-4.0", "source": "x"},     # not allowed
        {"tag": "rain", "uri": "s3://sfx/rain.wav", "licence": "CC0-1.0"},                             # no source
    ]
    lib = sound.library(_with(policy, library={"allowed_licences": ["CC0-1.0", "CC-BY-4.0"], "entries": entries}))
    ref = lib.lookup("kettle  click")
    assert ref == sound.SoundRef(uri="s3://sfx/kettle.wav", licence="CC0-1.0", source="own pack 2026-11",
                                 tag="kettle click")
    assert [r["reason"] for r in lib.refused] == [
        "no licence", "licence 'CC-BY-NC-4.0' is not in sound.library.allowed_licences", "no source"]
    for tag in ("fridge hum", "street", "rain"):
        with pytest.raises(sound.UnlicensedSound):
            lib.lookup(tag)
    assert lib.lookup("thunder") is None
    assert lib.tags() == ["kettle click"]
    assert sound.library(policy).tags() == []       # the shipped catalogue is empty


def test_an_impulse_response_needs_a_path_a_licence_and_a_source(policy):
    ok = _with(policy, impulse_responses={"kitchen": {"path": "/irs/kitchen.wav", "licence": "CC-BY-4.0",
                                                       "source": "OpenAIR", "credit": "x"},
                                           "bad": {"path": "/irs/bad.wav", "source": "?"}})
    assert sound.impulse_response("kitchen", ok).uri == "/irs/kitchen.wav"
    assert sound.impulse_response("hall", ok) is None
    with pytest.raises(sound.UnlicensedSound, match="no licence"):
        sound.impulse_response("bad", ok)


def _tone(path: Path, src: str, seconds: float) -> Path:
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", f"{src}:d={seconds}", str(path)],
                   check=True)
    return path


def test_mix_runs_in_ffmpeg_and_keeps_the_video_and_its_length(tmp_path, policy):
    video = tmp_path / "talk.mp4"
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x288:rate=24:duration=3",
                    "-f", "lavfi", "-i", "sine=frequency=220:duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(video)], check=True)
    room = _tone(tmp_path / "room.wav", "anoisesrc=color=brown:amplitude=0.2", 1.0)
    click = _tone(tmp_path / "click.wav", "sine=frequency=2000", 0.1)
    bed = _tone(tmp_path / "bed.wav", "sine=frequency=110", 1.0)
    ir = _tone(tmp_path / "ir.wav", "anoisesrc=color=white:amplitude=0.5", 0.05)
    out = sound.mix(video, tmp_path / "mixed.mp4", room_tone=room, effects=[(click, 1.0, None)], music=bed, ir=ir,
                    mic="phone", policy=policy)
    info = media.probe(out)
    assert abs(media.duration(out) - 3.0) < 0.3
    assert info  # readable
    r = subprocess.run([media.ffmpeg(), "-hide_banner", "-i", str(out)], capture_output=True, text=True)
    assert "Video:" in r.stderr and "Audio:" in r.stderr


# --- generators: off until the owner says so ---------------------------------------------------------------------

def _ctx(tmp_path, kind):
    from manyworlds import queue
    job = queue.Job(id="j", kind=kind, payload={}, attempts=1, max_attempts=3, creator_id=None, batch=None)
    return JobContext(job=job, workdir=Path(tempfile.mkdtemp(dir=tmp_path)), store=None)


class FakeStableAudio:
    def __init__(self):
        self.calls = []
        self.model = type("M", (), {"sample_rate": 44100})()

    def generate(self, **kw):
        self.calls.append(kw)
        n = int(kw["duration"] * 44100)
        return np.zeros((kw["batch_size"], 2, n), dtype=np.float32) + 0.1


def test_sfx_refuses_unless_enabled_and_its_licence_is_accepted(tmp_path, policy, raw_policy):
    payload = {"prompt": "kettle click", "seconds": 1, "seed": 3}
    for p in (raw_policy, _with(policy, sfx={"enabled": True}), _with(policy, sfx={"licence_accepted": True})):
        with pytest.raises(NonRetryable):
            sound_gen.sfx(payload, _ctx(tmp_path, "sound.sfx"), policy=p, load_model=lambda m: FakeStableAudio())
    no_key = dict(policy.body)
    no_key["sound"] = {**policy.body["sound"], "sfx": {k: v for k, v in policy.body["sound"]["sfx"].items()
                                                       if k != "licence_accepted"} | {"enabled": True}}
    with pytest.raises(NonRetryable, match="licence"):      # removing the key never switches it on
        sound_gen.sfx(payload, _ctx(tmp_path, "sound.sfx"), policy=policy_mod.from_dict(no_key),
                      load_model=lambda m: FakeStableAudio())


def test_sfx_writes_wav_files_once_switched_on(tmp_path, policy):
    on = _with(policy, sfx={"enabled": True, "licence_accepted": True})
    model = FakeStableAudio()
    res = sound_gen.sfx({"prompt": "kettle click", "seconds": 0.5, "seed": 3, "n": 2}, _ctx(tmp_path, "sound.sfx"),
                        policy=on, load_model=lambda m: model)
    assert [o.kind for o in res.outputs] == ["audio", "audio"]
    with wave.open(str(res.outputs[0].path)) as w:
        assert w.getnchannels() == 2 and w.getframerate() == 44100 and w.getnframes() == 22050
    assert model.calls[0]["batch_size"] == 2 and model.calls[0]["seed"] == 3
    assert res.recipe["model"] == on.get("sound", "sfx", "repo")
    with pytest.raises(NonRetryable):
        sound_gen.sfx({"prompt": "x", "seconds": 999}, _ctx(tmp_path, "sound.sfx"), policy=on,
                      load_model=lambda m: model)


def test_music_refuses_while_off_and_returns_the_engines_files_when_on(tmp_path, policy, raw_policy):
    def engine(caption, seconds, seeds, save_dir):
        paths = []
        for s in seeds:
            p = Path(save_dir) / f"bed_{s}.wav"
            sound_gen.write_wav(p, np.zeros((2, 4800), dtype=np.float32), 48000)
            paths.append(p)
        return paths
    with pytest.raises(NonRetryable):
        sound_gen.music({"caption": "lofi bed", "seconds": 10}, _ctx(tmp_path, "sound.music"), policy=raw_policy,
                        engine=engine)
    on = _with(policy, music={"enabled": True})
    res = sound_gen.music({"caption": "lofi bed", "seconds": 10, "seed": 7, "n": 2}, _ctx(tmp_path, "sound.music"),
                          policy=on, engine=engine)
    assert [o.path.name for o in res.outputs] == ["bed_7.wav", "bed_8.wav"]
    assert res.recipe["licence"] == "MIT"


def test_the_sound_kinds_have_handlers(policy):
    found = runner.handlers_for(["sound.sfx", "sound.music"])
    assert found["sound.sfx"] is sound_gen.sfx and found["sound.music"] is sound_gen.music
