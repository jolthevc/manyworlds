"""Sound design (manyworlds.sound): the mix graph (voice-led ducking, room tone, effects, impulse response, mic EQ,
the voice polish chain), procedural room tone, the licence-checked library, the phone looks, and the generator
handlers that stay off until the owner says so."""

from __future__ import annotations

import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from manyworlds import finishing, media, policy as policy_mod, sound
from manyworlds.worker import runner
from manyworlds.worker.handlers import sound_gen
from manyworlds.worker.runner import JobContext, NonRetryable

FMT = "aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo"


def _with(policy, **blocks):
    body = dict(policy.body)
    s = dict(body.get("sound") or {})
    for k, v in blocks.items():
        s[k] = {**(s.get(k) or {}), **v} if isinstance(v, dict) else v
    body["sound"] = s
    return policy_mod.from_dict(body)


def test_the_shipped_policy_has_the_gpu_free_sound_on_and_the_generators_off(raw_policy):
    c = raw_policy.get("sound")
    assert c["enabled"] is True and sound.enabled(raw_policy)          # room tone and voice polish need no GPU
    assert c["room_tone"]["procedural"] is True and c["voice_polish"]["enabled"] is True
    assert c["library"]["entries"] == [] and c["impulse_responses"] == {}
    assert c["sfx"]["enabled"] is False and c["sfx"]["licence_accepted"] is False
    assert c["music"]["enabled"] is False                                # ACE-Step: off until benchmarked
    assert "story_clip" in c["silent_beds"]["formats"]


def test_music_and_room_tone_are_ducked_under_the_polished_voice_with_timed_effects(policy):
    p = sound.plan(has_voice=True, room_tone="room.wav", effects=[("click.wav", 1.5, None), ("door.wav", 3.25, -6)],
                   music="bed.wav", policy=policy)
    m = policy.get("sound", "mix")
    assert p.inputs == [["-stream_loop", "-1", "-i", "room.wav"], ["-i", "click.wav"], ["-i", "door.wav"],
                        ["-stream_loop", "-1", "-i", "bed.wav"]]
    g = p.graph
    polish = ",".join(sound.voice_polish(policy.get("sound", "voice_polish", "room", "default"), policy))
    assert g.startswith(f"[0:a]{FMT},{polish}[vo]")
    assert f"[1:a]{FMT},volume={m['room_tone_gain_db']}dB[room_in]" in g
    assert f"volume={m['effect_gain_db']}dB,adelay=1500|1500[fx0]" in g
    assert "volume=-6dB,adelay=3250|3250[fx1]" in g
    assert f"[4:a]{FMT},volume={m['music_gain_db']}dB[mus_in]" in g
    # the voice keys a compressor on each bed, and the voice itself goes on to the mix
    assert "[vo]asplit=3[vo_mix][vo_key_room][vo_key_mus]" in g
    assert "[room_in][vo_key_room]sidechaincompress=threshold=0.05:ratio=2:attack=80:release=450:makeup=1[room]" in g
    assert "[mus_in][vo_key_mus]sidechaincompress=threshold=0.05:ratio=6:attack=80:release=400:makeup=1[mus]" in g
    assert g.endswith("[vo_mix][room][fx0][fx1][mus]amix=inputs=5:duration=first:dropout_transition=0:normalize=0[mix]")
    assert "loudnorm" not in g        # loudness belongs to the finish stage
    assert p.out == "mix"


def test_with_polish_and_the_room_duck_off_the_graph_is_the_plain_mix(policy):
    off = _with(policy, voice_polish={"enabled": False}, room_tone={"duck": {"enabled": False}})
    g = sound.plan(has_voice=True, room_tone="room.wav", music="bed.wav", policy=off).graph
    assert g.startswith(f"[0:a]{FMT}[vo]")
    assert f"[1:a]{FMT},volume=" in g and "[vo]asplit=2[vo_mix][vo_key_mus]" in g
    assert g.endswith("[vo_mix][room_in][mus]amix=inputs=3:duration=first:dropout_transition=0:normalize=0[mix]")
    assert sound.plan(has_voice=True, polish=False, policy=policy).graph == f"[0:a]{FMT}[vo];[vo]anull[mix]"


def test_duck_attack_and_release_are_clamped_to_the_policy_limits(policy):
    fast = _with(policy, mix={**policy.get("sound", "mix"), "duck": {"threshold": 5, "ratio": 100, "attack_ms": 5,
                                                                      "release_ms": 5000}})
    f = sound.duck_filter(fast)
    assert "threshold=1:" in f and "ratio=20:" in f and "attack=50:" in f and "release=500:" in f
    room = sound.duck_filter(policy, {"ratio": 1, "attack_ms": 1000})            # a bed's own duck: same limits
    assert "ratio=2:" in room and "attack=100:" in room


def test_an_impulse_response_adds_room_to_the_voice_at_a_low_wet_share_and_the_mic_eq_applies(policy):
    p = sound.plan(has_voice=True, ir="ir.wav", mic="phone", policy=policy)
    eq = policy.get("sound", "mic_eq", "phone")
    assert p.inputs == [["-i", "ir.wav"]]
    g = p.graph
    no_room = ",".join(sound.voice_polish("small", policy, room=False))
    assert f"channel_layouts=stereo,highpass=f={eq['highpass']},lowpass=f={eq['lowpass']},{no_room}[vo]" in g
    assert "aecho" not in g                                                      # the IR is the room
    assert "[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=mono[ir]" in g
    assert "[vo]asplit=2[vo_dry][vo_wet_in];[vo_wet_in][ir]afir[vo_wet]" in g
    wet = policy.get("sound", "mix", "room_wet")
    assert f"[vo_dry][vo_wet]amix=inputs=2:weights='1 {wet}':duration=first:dropout_transition=0:normalize=0[vo_room]" in g
    assert g.endswith("[vo_room]anull[mix]")


def test_a_silent_video_gets_a_silent_stand_in_voice_and_its_music_is_not_ducked(policy):
    p = sound.plan(has_voice=False, duration_s=4.5, music="bed.wav", policy=policy)
    assert p.inputs[0] == ["-f", "lavfi", "-t", "4.5", "-i", "anullsrc=r=48000:cl=stereo"]
    assert "[1:a]aresample=48000" in p.graph and "sidechaincompress" not in p.graph
    assert "deesser" not in p.graph                                              # nothing to polish
    assert p.graph.endswith("[vo][mus_in]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mix]")
    with pytest.raises(ValueError):
        sound.plan(has_voice=False, music="bed.wav", policy=policy)
    with pytest.raises(ValueError):
        sound.plan(has_voice=True, effects=[("x.wav", -1, None)], policy=policy)


# --- the voice polish chain --------------------------------------------------------------------------------------

def test_the_voice_polish_chain_is_high_pass_de_ess_compress_and_a_small_room(policy):
    assert sound.voice_polish("small", policy) == [
        "highpass=f=80", "deesser=i=0.35:m=0.5:f=0.5",
        "acompressor=threshold=0.125:ratio=2.5:attack=15:release=200:makeup=1.5",
        "aecho=in_gain=1:out_gain=0.95:delays=7|13|19:decays=0.14|0.09|0.05"]
    assert sound.voice_polish("large", policy)[-1].startswith("aecho=in_gain=1:out_gain=0.95:delays=21|43|67:")
    assert not any(f.startswith("aecho") for f in sound.voice_polish("none", policy))   # outdoors: no room
    assert sound.voice_polish("small", _with(policy, voice_polish={"enabled": False})) == []


def test_each_polish_element_has_its_own_off_switch_and_values_are_clamped(policy):
    vp = policy.get("sound", "voice_polish")
    for element, prefix in (("highpass", "highpass"), ("deess", "deesser"), ("compress", "acompressor"),
                            ("room", "aecho")):
        off = _with(policy, voice_polish={element: {**vp[element], "enabled": False}})
        chain = sound.voice_polish("small", off)
        assert not any(f.startswith(prefix) for f in chain) and len(chain) == 3
    wild = _with(policy, voice_polish={"highpass": {"hz": 900}, "compress": {**vp["compress"], "ratio": 30},
                                       "room": {**vp["room"], "sizes": {"small": {"delays_ms": [500],
                                                                                  "decays": [0.9]}}}})
    chain = sound.voice_polish("small", wild)
    lim = vp["limits"]
    assert chain[0] == f"highpass=f={lim['highpass_hz'][1]}" and f":ratio={lim['ratio'][1]}:" in chain[2]
    assert chain[3] == f"aecho=in_gain=1:out_gain=1:delays={lim['delay_ms'][1]}:decays={lim['decay'][1]}"


def test_the_room_size_follows_the_set(policy):
    assert sound.room_size("home kitchen, her small galley kitchen", policy) == "small"
    assert sound.room_size("a busy cafe on the corner", policy) == "medium"
    assert sound.room_size("the gym", policy) == "large"
    assert sound.room_size("a park bench", policy) == "none"
    assert sound.room_size("", policy) == policy.get("sound", "voice_polish", "room", "default")


# --- procedural room tone ----------------------------------------------------------------------------------------

def test_room_tone_presets_are_picked_by_keyword_from_the_cue_then_the_set(policy):
    def preset(cue, set_text=""):
        return sound.room_tone(cue, set_text=set_text, seed_key="post-1", policy=policy).preset
    assert preset("quiet flat, fridge hum") == "kitchen"
    assert preset("rain on the window") == "rain_glass"
    assert preset("busy street outside the open window") == "street_window"
    assert preset("", "her bedroom, late night") == "bedroom"
    assert preset("", "a wet side street, light drizzle") == "rain_glass"          # rain wins: first rule
    assert preset("kitchen", "a bedroom") == "kitchen"                             # the cue wins over the set
    assert preset("", "") == policy.get("sound", "room_tone", "default")
    assert preset("cardigan") == policy.get("sound", "room_tone", "default")      # whole words only ("car")
    assert sound.room_tone("kitchen", seed_key=1, policy=_with(policy, room_tone={"procedural": False})) is None


def test_every_shipped_preset_builds_a_valid_looking_graph(policy):
    rc = policy.get("sound", "room_tone")
    for name, spec in rc["presets"].items():
        rt = sound.room_tone("", seed_key=name, policy=_with(policy, room_tone={"default": name, "keywords": []}))
        assert rt.preset == name and rt.colour in sound.NOISE_COLOURS
        g = sound.room_tone_graph(rt, sample_rate=48000, duration_s=3)
        assert g.startswith(f"anoisesrc=color={rt.colour}:seed={rt.seed}:sample_rate=48000:amplitude=1:duration=3")
        assert g.endswith(f"{FMT},volume={sound._num(rt.gain_db)}dB[room]")
        assert f"tremolo=f={sound._num(rt.rate_hz)}:d={sound._num(rt.depth)}" in g
        lo, hi = rc["gain_db_limits"]
        assert lo <= rt.gain_db <= hi
        if spec.get("hum"):
            assert f"sine=frequency={spec['hum']['hz']}:sample_rate=48000:duration=3" in g
            assert "[room_n][room_h]amix=inputs=2:duration=first" in g
        else:
            assert "sine=" not in g and ";" not in g
        for b in spec.get("bands") or []:
            assert f"equalizer=f={sound._num(b['f'])}:t=q:w={sound._num(b['q'])}:g={sound._num(b['gain_db'])}" in g


def test_room_tone_is_deterministic_per_post_and_varies_between_posts(policy):
    a = sound.room_tone("fridge hum", seed_key="post-42", policy=policy)
    assert a == sound.room_tone("fridge hum", seed_key="post-42", policy=policy)
    assert sound.room_tone_graph(a, sample_rate=48000) == sound.room_tone_graph(
        sound.room_tone("fridge hum", seed_key="post-42", policy=policy), sample_rate=48000)
    b = sound.room_tone("fridge hum", seed_key="post-43", policy=policy)
    assert a.seed != b.seed and (a.rate_hz, a.depth) != (b.rate_hz, b.depth)
    assert sound.seed_for("post-42") == sound.seed_for("post-42") and 0 <= sound.seed_for("x") < 2147483647
    mod = policy.get("sound", "room_tone", "modulation")
    for rt in (a, b):
        assert mod["rate_hz"][0] <= rt.rate_hz <= mod["rate_hz"][1] and mod["depth"][0] <= rt.depth <= mod["depth"][1]
    assert "duration=" not in sound.room_tone_graph(a, sample_rate=48000)       # ends with the voice instead


def test_a_bad_colour_is_refused(policy):
    bad = _with(policy, room_tone={"presets": {"room_quiet": {"colour": "plaid", "gain_db": -40}}})
    with pytest.raises(ValueError, match="colour"):
        sound.room_tone("", seed_key=1, policy=bad)


def test_a_procedural_bed_needs_no_input_and_ducks_under_the_voice(policy):
    rt = sound.room_tone("fridge hum", seed_key="p", policy=policy)
    p = sound.plan(has_voice=True, room_tone_synth=rt, policy=policy)
    assert p.inputs == []                                                         # ffmpeg's own sources
    assert "anoisesrc=color=pink" in p.graph and "[room_in]" in p.graph
    assert "[vo]asplit=2[vo_mix][vo_key_room]" in p.graph
    assert p.graph.endswith("[vo_mix][room]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mix]")
    # a file for the cue wins over the synthesised bed
    assert "anoisesrc" not in sound.plan(has_voice=True, room_tone="room.wav", room_tone_synth=rt, policy=policy).graph


def test_a_silent_clip_gets_its_bed_for_exactly_its_length(policy):
    rt = sound.room_tone("", set_text="her bedroom", seed_key="p", policy=policy)
    p = sound.plan(has_voice=False, duration_s=5, room_tone_synth=rt, policy=policy)
    assert p.inputs == [["-f", "lavfi", "-t", "5", "-i", "anullsrc=r=48000:cl=stereo"]]
    assert "color=brown" in p.graph and "duration=5" in p.graph and "sidechaincompress" not in p.graph
    assert p.graph.endswith("[vo][room_in]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[mix]")


def test_design_reads_the_shot_sheets(policy):
    shots = [{"set": {"set_id": "home_kitchen", "location": "her small galley kitchen"}, "sound": {}},
             {"set": {"location": "the street"}, "sound": {"mic": "phone", "room_tone": "rain on glass"}}]
    d = sound.design(shots, seed_key="post-7", policy=policy)
    assert d.room_tone.preset == "rain_glass" and d.mic == "phone"                # the first shot that names one
    assert d.room_size == "none"                                                  # that shot's set: outdoors
    d = sound.design(shots[:1], seed_key="post-7", policy=policy)
    assert d.room_tone.preset == "kitchen" and d.room_size == "small" and d.mic is None
    bare = sound.design(None, seed_key="post-7", policy=policy)                   # no shot sheets: still a bed
    assert bare.room_tone.preset == policy.get("sound", "room_tone", "default")
    assert sound.design(shots, seed_key=1, policy=_with(policy, enabled=False)) is None


def test_silent_formats_get_a_licensed_music_bed_only_when_the_library_has_one(policy):
    bed = {"tag": "soft piano bed", "uri": "s3://b/music/piano.wav", "licence": "own-generated",
           "source": "ACE-Step 1.5 seed 3", "kind": "music", "formats": ["story_clip"]}
    p = _with(policy, library={"entries": [bed, {**bed, "tag": "nc bed", "licence": "CC-BY-NC-4.0"}]})
    lib = sound.library(p)
    assert sound.bed_music(lib, "story_clip", p, seed_key="post-1").tag == "soft piano bed"
    assert sound.bed_music(lib, "talking_video", p) is None                       # has a voice
    assert sound.bed_music(lib, "story_clip", _with(p, silent_beds={"music": False})) is None
    assert sound.bed_music(sound.library(policy), "story_clip", policy) is None   # the shipped library is empty
    assert len(lib.refused) == 1 and sound.silent_format("story_clip", policy)


# --- looks (finishing.looks, applied by manyworlds.finishing) -----------------------------------------------------

PHONE_LOOKS = ("phone_native", "phone_golden_hour", "phone_night_indoor", "phone_overcast")


def test_the_default_look_is_phone_native_with_no_vignette(raw_policy):
    f = raw_policy.get("finishing")
    assert f["default_look"] == "phone_native" and f["vignette"]["enabled"] is False
    assert f["grain"]["strength"] <= 2                                            # phones denoise
    assert finishing.look_for(raw_policy) == "phone_native"


def test_the_phone_looks_build_valid_filters_inside_the_limits(raw_policy):
    f = raw_policy.get("finishing")
    lim = f["limits"]
    for name in PHONE_LOOKS:
        spec = f["looks"][name]
        for k in ("contrast", "brightness", "saturation", "gamma"):
            if k in spec:
                assert lim[k][0] <= spec[k] <= lim[k][1], (name, k)               # never silently clamped
        for v in (spec.get("balance") or {}).values():
            assert abs(v) <= lim["balance"]
        points = [tuple(map(float, pt.split("/"))) for pt in spec["curves"]["master"].split()]
        xs = [x for x, _ in points]
        assert xs == sorted(set(xs)) and all(0 <= y <= 1 for _, y in points)
        assert points[0][1] > 0 and points[-1][1] < 1                             # lifted shadows, kept highlights
        vf = finishing.filters(name, policy=raw_policy)
        parts = vf.split(",")
        assert parts[0].startswith("eq=") and parts[-1] == "format=yuv420p"
        assert "curves=master='" in vf and "vignette" not in vf
        assert vf != finishing.filters("neutral", policy=raw_policy)
    assert "colorbalance=" not in finishing.filters("phone_native", policy=raw_policy)   # neutral white balance
    assert "rh=0.04" in finishing.filters("phone_golden_hour", policy=raw_policy)


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


def test_a_generated_bed_carries_a_catalogue_entry_the_library_accepts(tmp_path, policy):
    captions = []

    def engine(caption, seconds, seeds, save_dir):
        captions.append(caption)
        p = Path(save_dir) / "bed.wav"
        sound_gen.write_wav(p, np.zeros((2, 4800), dtype=np.float32), 48000)
        return [p]
    on = _with(policy, music={"enabled": True})
    res = sound_gen.music({"format": "story_clip", "seconds": 10, "seed": 11}, _ctx(tmp_path, "sound.music"),
                          policy=on, engine=engine)
    assert captions == [on.get("sound", "music", "bed_captions", "story_clip")]   # the format's quiet bed
    entry = res.outputs[0].meta["library_entry"]
    assert entry["kind"] == "music" and entry["formats"] == ["story_clip"] and "seed 11" in entry["source"]
    assert entry["licence"] == on.get("sound", "music", "library_licence")
    lib = sound.library(_with(on, library={"entries": [{**entry, "uri": "s3://b/music/bed.wav"}]}))
    assert lib.refused == [] and sound.bed_music(lib, "story_clip", on) is not None
    with pytest.raises(NonRetryable, match="caption"):
        sound_gen.music({"format": "talking_video", "seconds": 10}, _ctx(tmp_path, "sound.music"), policy=on,
                        engine=engine)


def test_procedural_room_tone_runs_in_ffmpeg_under_a_voice_and_under_a_silent_clip(tmp_path, policy):
    talk = tmp_path / "talk.mp4"
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x288:rate=24:duration=3",
                    "-f", "lavfi", "-i", "sine=frequency=220:duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(talk)], check=True)
    clip = tmp_path / "clip.mp4"
    subprocess.run([media.ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x288:rate=16:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
    for video, cue, seconds in ((talk, "quiet flat, fridge hum", 3.0), (clip, "rain on glass", 2.0)):
        d = sound.design([{"set": {"location": "her small kitchen"}, "sound": {"room_tone": cue}}],
                         seed_key=video.name, policy=policy)
        out = sound.mix(video, tmp_path / f"mixed_{video.name}", room_tone_synth=d.room_tone, room_size=d.room_size,
                        policy=policy)
        assert abs(media.duration(out) - seconds) < 0.3
        r = subprocess.run([media.ffmpeg(), "-hide_banner", "-i", str(out)], capture_output=True, text=True)
        assert "Video:" in r.stderr and "Audio:" in r.stderr                    # the silent clip now has sound


def test_the_sound_kinds_have_handlers(policy):
    found = runner.handlers_for(["sound.sfx", "sound.music"])
    assert found["sound.sfx"] is sound_gen.sfx and found["sound.music"] is sound_gen.music


# --- production: the shot sheets' cues mixed before the finish (production.sound_cues, finish_video) ---------------
KETTLE = {"tag": "kettle click", "uri": "s3://b/sfx/kettle.wav", "licence": "CC0-1.0", "source": "hand-picked",
          "credit": ""}
HUM = {"tag": "fridge hum", "uri": "s3://b/sfx/hum.wav", "licence": "own-recording", "source": "us"}
SHOT_SOUND = {"mic": "phone", "room_tone": "quiet flat, fridge hum", "effects": ["kettle click", "door slam"],
              "music": ""}


def _finish_policy(policy, entries, *, enabled=True):
    p = _with(policy, enabled=enabled, library={"entries": entries})
    return policy_mod.from_dict({**p.body, "finishing": {**p.body["finishing"], "enabled": True},
                                 "burned_captions": {**p.body["burned_captions"], "enabled": False}})


def _made_video(conn, store, tmp_path):
    from psycopg.types.json import Jsonb
    from test_production import creator
    cid = creator(conn)
    src = tmp_path / "raw.mp4"
    src.write_bytes(b"mp4")
    uri = store.put_file(src, "v/raw.mp4").uri
    job = conn.execute("INSERT INTO jobs (kind, status, creator_id, payload) VALUES ('image.narrated', 'done', %s, %s) "
                       "RETURNING id", (cid, Jsonb({}))).fetchone()["id"]
    video = conn.execute("INSERT INTO assets (job_id, creator_id, kind, uri, sha256, bytes, meta) "
                         "VALUES (%s, %s, 'video', %s, 'x', 1, %s) RETURNING id, uri",
                         (job, cid, uri, Jsonb({"seconds": 3.0}))).fetchone()
    pid = conn.execute("INSERT INTO posts (creator_id, format, status, body) VALUES (%s, 'narrated_edit', 'producing', "
                       "'{}') RETURNING id", (cid,)).fetchone()["id"]
    post = {"id": pid, "creator_id": cid, "format": "narrated_edit"}
    body = {"scenes": ["the kitchen"], "shots": [{"sound": SHOT_SOUND}], "script": "Kettle's on."}
    return post, body, video


@pytest.fixture()
def fake_finish(monkeypatch):
    from manyworlds import finishing
    calls = []

    def fake(video, out, *, look, grain=None, hook_ass=None, policy, timeout=None):
        calls.append(("finish", Path(video).name))
        Path(out).write_bytes(b"finished")
        return out

    def fake_mix(video, out, **kw):
        calls.append(("mix", kw))
        Path(out).write_bytes(b"mixed")
        return Path(out)
    monkeypatch.setattr(finishing, "finish", fake)
    monkeypatch.setattr(sound, "mix", fake_mix)
    monkeypatch.setattr(media, "probe", lambda p, **k: {"seconds": 3.0, "width": 1080, "height": 1920, "audio": True})
    return calls


class _Sounds:
    """The object store, plus stand-in files for library sounds."""
    def __init__(self, store):
        self.store = store

    def __getattr__(self, name):
        return getattr(self.store, name)

    def get_file(self, uri, local):
        if uri.startswith("s3://b/sfx/"):
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(b"RIFF")
            return local
        return self.store.get_file(uri, local)


def test_cues_that_resolve_in_the_library_are_mixed_before_the_finish(conn, policy, store, tmp_path, fake_finish):
    from manyworlds import production
    p = _finish_policy(policy, [KETTLE, HUM])
    post, body, video = _made_video(conn, store, tmp_path)
    cues = production.sound_cues(conn, p, body, video)
    assert cues["room_tone"].tag == "fridge hum"                               # matched by one part of the cue
    assert [(r.tag, at) for r, at in cues["effects"]] == [("kettle click", p.get("sound", "mix", "effect_at_s"))]
    assert cues["music"] is None and cues["mic"] == "phone"                    # "door slam" isn't in the library
    final = production.finish_video(conn, p, post, body, video, _Sounds(store))
    assert [name for name, _ in fake_finish] == ["mix", "finish"]
    (_, mix), (_, finished) = fake_finish
    assert mix["room_tone"].name.startswith("room") and len(mix["effects"]) == 1 and mix["music"] is None
    assert mix["mic"] == "phone" and finished == "mixed.mp4"                  # the finish works on the mixed video
    meta = final["meta"]
    assert meta["passes"] == 2 and {s["tag"] for s in meta["sound"]} == {"fridge hum", "kettle click"}
    assert all(s["licence"] for s in meta["sound"])


def test_an_empty_library_or_sound_off_mixes_nothing(conn, policy, store, tmp_path, fake_finish):
    from manyworlds import production
    for p in (_finish_policy(policy, []), _finish_policy(policy, [KETTLE, HUM], enabled=False),
              _finish_policy(policy, [{**KETTLE, "licence": "CC-BY-NC-4.0"}])):   # refused: never mixed
        post, body, video = _made_video(conn, store, tmp_path)
        assert production.sound_cues(conn, p, body, video) is None
        final = production.finish_video(conn, p, post, body, video, store)
        assert "sound" not in final["meta"] and final["meta"]["passes"] == 1
    assert [name for name, _ in fake_finish] == ["finish"] * 3
