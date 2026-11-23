"""Sound design for a video (the shot sheet's `sound` module: mic, room tone, effects, music), on the control plane.

Plan 2026-11-22-video-quality-autonomous, wave 3 item 9; research docs/research/2026-11-22-video-quality-wave2-3.md
section 2, and docs/research/2026-11-23-readiness-and-craft.md: a silent story clip and a studio-dry voice are the
biggest "AI" tells after resolution, while the top outfit clips run on their own quiet original sound. So the parts
that need no GPU are on (policy `sound.enabled: true`): procedural room tone and the voice polish chain. The
generators stay off. production.finish_video mixes a post's sound before finishing (production.sound_cues).

  mix        one ffmpeg pass over a video's own audio (the voice):
               voice      resampled to stereo at `sound.sample_rate`, a mic-character EQ per shot-sheet mic
                          (`sound.mic_eq`: highpass/lowpass), the polish chain (`sound.voice_polish`: high-pass,
                          de-ess, light compression, and a touch of the room: a short `aecho` sized to the set, so
                          TTS sounds recorded in the room, not in a studio), and optionally the room from an
                          impulse response (`afir`, which then replaces the aecho) at a low wet share
               room tone  a library file looped, or a procedural bed (`room_tone`: ffmpeg's own noise sources,
                          band-limited, slowly modulated, seeded per post), at a low level and gently ducked under
                          the voice (`sound.room_tone.duck`)
               effects    each placed at its time (`adelay`) at its gain
               music      a looped bed, ducked under the voice (`sidechaincompress` keyed by the voice; attack and
                          release clamped to `sound.mix.limits`: 50-100 ms and 300-500 ms)
             The video stream is copied. Loudness is NOT set here: the finish stage (manyworlds.finishing) runs
             two-pass loudnorm on whatever this writes, so the beds are summed without normalising (`amix
             normalize=0`) and the voice keeps its level.
  room tone  `room_tone(cue, set_text=, seed_key=, policy=)`: a preset (kitchen hum, quiet bedroom, street outside
             the window, rain on glass, ...) picked by keyword from the shot sheet's `sound.room_tone` text, else
             the set's words, else `sound.room_tone.default`. Synthesised, so there is no licence question: we make
             it. Deterministic per post (`seed_for`), so a rerun of the mix is the same file.
  design     `design(shots, seed_key=, policy=)`: what a post's shot sheets ask of the mix (procedural room tone,
             the polish room size, the mic), for production to pass to `mix`.
  library    `library(policy).lookup(tag) -> SoundRef{uri, licence, source}` from the policy catalogue
             (`sound.library.entries`, empty by default). An entry without a licence string, without a source,
             or with a licence that is not in `sound.library.allowed_licences` is refused, never used.
             `bed_music(lib, format, policy)` is the quiet music bed for a silent format, when the catalogue has one.
  impulse responses  `impulse_response(key, policy)`: policy `sound.impulse_responses` (none shipped); each needs a
             path and a licence (OpenAIR files are licensed per file: check and credit each one).

Generators (Stable Audio 3 Small-SFX for effects, ACE-Step 1.5 for music beds) are worker handlers in
manyworlds/worker/handlers/sound_gen.py, off. `plan` and the builders are pure (tested in CI); `mix` runs ffmpeg
(manyworlds.media.ffmpeg, the imageio-ffmpeg static build).
"""

from __future__ import annotations

import hashlib
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .policy import Policy

NOISE_COLOURS = ("white", "pink", "brown", "blue", "violet", "velvet")   # ffmpeg anoisesrc's colours


class UnlicensedSound(ValueError):
    """A sound (or impulse response) without a usable licence was asked for: it is never mixed in."""


def cfg(policy: Policy | None) -> dict:
    return (policy.get("sound", default={}) or {}) if policy is not None else {}


def enabled(policy: Policy | None) -> bool:
    return bool(cfg(policy).get("enabled"))


def _num(v: float) -> str:
    """A filter number without float noise: 1.0 -> '1', 0.05 -> '0.05'."""
    s = f"{float(v):.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _clamp(value, lo_hi, default: float) -> float:
    v = float(default if value is None else value)
    if lo_hi:
        lo, hi = float(lo_hi[0]), float(lo_hi[1])
        v = min(hi, max(lo, v))
    return v


def _on(block: dict | None) -> bool:
    """A policy sub-block is on unless it says `enabled: false` (absent blocks are off)."""
    return bool(block) and block.get("enabled", True) is not False


# --- the library -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SoundRef:
    uri: str
    licence: str
    source: str
    tag: str = ""
    credit: str = ""
    kind: str = ""               # room_tone | music | effect (optional in the catalogue)


def _key(tag: str | None) -> str:
    return re.sub(r"\s+", " ", str(tag or "").strip().lower())


def _refusal(entry: dict, allowed: set[str] | None, *, needs: str = "uri") -> str | None:
    if not str(entry.get(needs) or "").strip():
        return f"no {needs}"
    licence = str(entry.get("licence") or "").strip()
    if not licence:
        return "no licence"
    if allowed is not None and licence.lower() not in allowed:
        return f"licence {licence!r} is not in sound.library.allowed_licences"
    if not str(entry.get("source") or "").strip():
        return "no source"
    return None


class SoundLibrary:
    """A catalogue of licence-clean sounds by tag. Entries that fail the licence check are kept in `refused` (with
    the reason) so `mw` can show them, and are never returned. An entry may also name its `kind` and the `formats`
    it suits (a quiet bed for story clips), which `bed_music` reads."""

    def __init__(self, entries: Iterable[dict] | None, *, allowed_licences: Iterable[str] | None = None):
        self.allowed = {str(x).strip().lower() for x in allowed_licences} if allowed_licences is not None else None
        self.refs: dict[str, list[SoundRef]] = {}
        self.formats: dict[str, list[SoundRef]] = {}
        self.refused: list[dict] = []
        for raw in entries or []:
            entry = dict(raw or {})
            tag = _key(entry.get("tag"))
            reason = "no tag" if not tag else _refusal(entry, self.allowed)
            if reason:
                self.refused.append({**entry, "reason": reason})
                continue
            ref = SoundRef(uri=str(entry["uri"]).strip(), licence=str(entry["licence"]).strip(),
                           source=str(entry["source"]).strip(), tag=tag, credit=str(entry.get("credit") or "").strip(),
                           kind=str(entry.get("kind") or "").strip())
            self.refs.setdefault(tag, []).append(ref)
            if ref.kind == "music":
                for fmt in entry.get("formats") or []:
                    self.formats.setdefault(str(fmt), []).append(ref)

    def tags(self) -> list[str]:
        return sorted(self.refs)

    def lookup(self, tag: str, *, index: int = 0) -> SoundRef | None:
        """The sound for a tag (the `index`-th when there are several, wrapping), or None when the catalogue has none.
        A tag whose only entries were refused raises UnlicensedSound: the caller asked for something we hold but
        may not use."""
        key = _key(tag)
        found = self.refs.get(key)
        if found:
            return found[int(index) % len(found)]
        refused = [r for r in self.refused if _key(r.get("tag")) == key]
        if refused:
            raise UnlicensedSound(f"every sound tagged {tag!r} was refused: {refused[0]['reason']}")
        return None

    def music_for(self, fmt: str, *, index: int = 0) -> SoundRef | None:
        found = self.formats.get(str(fmt))
        return found[int(index) % len(found)] if found else None


def library(policy: Policy | None) -> SoundLibrary:
    lib = cfg(policy).get("library") or {}
    return SoundLibrary(lib.get("entries") or [], allowed_licences=lib.get("allowed_licences"))


def impulse_response(key: str, policy: Policy | None) -> SoundRef | None:
    """The impulse response for a set or mic key (`sound.impulse_responses`), or None. One without a path, licence or
    source raises UnlicensedSound."""
    entry = (cfg(policy).get("impulse_responses") or {}).get(key)
    if not entry:
        return None
    lib = cfg(policy).get("library") or {}
    allowed = lib.get("allowed_licences")
    reason = _refusal(dict(entry), {str(x).strip().lower() for x in allowed} if allowed is not None else None,
                      needs="path")
    if reason:
        raise UnlicensedSound(f"impulse response {key!r} refused: {reason}")
    return SoundRef(uri=str(entry["path"]).strip(), licence=str(entry["licence"]).strip(),
                    source=str(entry["source"]).strip(), tag=key, credit=str(entry.get("credit") or "").strip())


def silent_format(fmt: str | None, policy: Policy | None) -> bool:
    """A format whose video has no voice of its own (`sound.silent_beds.formats`): it gets a bed, never silence."""
    return str(fmt or "") in ((cfg(policy).get("silent_beds") or {}).get("formats") or [])


def bed_music(lib: SoundLibrary, fmt: str | None, policy: Policy | None, *, seed_key=None) -> SoundRef | None:
    """The quiet music bed for a silent format, or None: off (`sound.silent_beds.music`), not a silent format, or
    the catalogue has no music entry listing this format (the shipped one has none: room tone carries it). Picked
    per post by seed, so a rerun picks the same bed."""
    sb = cfg(policy).get("silent_beds") or {}
    if not sb.get("music") or not silent_format(fmt, policy):
        return None
    return lib.music_for(str(fmt), index=seed_for(seed_key) if seed_key is not None else 0)


# --- keywords ----------------------------------------------------------------------------------------------------

def seed_for(key) -> int:
    """A stable 31-bit seed for a post (or any key): the same post always gets the same room tone."""
    return int.from_bytes(hashlib.sha256(str(key).encode("utf-8")).digest()[:4], "big") % 2147483647


def _by_keyword(texts: Iterable[str], rules: Iterable[dict] | None, field: str) -> str | None:
    """The first rule (in policy order) any of whose `words` appears as a whole word or phrase in the texts, tried
    text by text (the first text is the most specific)."""
    rules = list(rules or [])
    for text in texts:
        t = _key(text)
        if not t:
            continue
        for rule in rules:
            for w in rule.get("words") or []:
                if re.search(r"(?<![a-z])" + re.escape(_key(w)) + r"(?![a-z])", t):
                    return str(rule.get(field) or "") or None
    return None


def set_text(shot: dict | None) -> str:
    """The words of a shot sheet's set that say what the place sounds like (set_id, location, weather, dressing)."""
    s = (shot or {}).get("set") or {}
    parts = [str(s.get(k) or "").replace("_", " ") for k in ("set_id", "location", "weather", "set_dressing")]
    return ", ".join(p for p in parts if p.strip())


# --- procedural room tone ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RoomTone:
    """A synthesised bed: band-limited noise of a colour, an optional mains/appliance hum, a slow seeded swell."""
    preset: str
    seed: int
    colour: str
    gain_db: float
    rate_hz: float               # tremolo rate: the bed breathes, so it never reads as a static loop
    depth: float
    highpass: float | None = None
    lowpass: float | None = None
    bands: tuple = ()            # (f_hz, width_q, gain_db) peaking EQs
    hum_hz: float | None = None
    hum_db: float | None = None  # the hum against the noise, before the bed's gain


def room_tone(cue: str | None, *, set_text: str = "", seed_key, policy: Policy | None) -> RoomTone | None:
    """The procedural room tone for a shot: the preset its `sound.room_tone` text names by keyword
    (`sound.room_tone.keywords`), else the set's words, else `sound.room_tone.default`. None when procedural room
    tone is off or the preset is not defined."""
    rc = cfg(policy).get("room_tone") or {}
    if not rc.get("procedural"):
        return None
    name = _by_keyword([cue or "", set_text or ""], rc.get("keywords"), "preset") or rc.get("default")
    spec = (rc.get("presets") or {}).get(name or "")
    if not spec:
        return None
    colour = str(spec.get("colour") or "pink")
    if colour not in NOISE_COLOURS:
        raise ValueError(f"room tone preset {name!r}: colour {colour!r} is not one of {NOISE_COLOURS}")
    seed = seed_for(seed_key)
    rng = random.Random(seed)
    mod = rc.get("modulation") or {}
    rate = rng.uniform(*[float(x) for x in (mod.get("rate_hz") or (0.1, 0.1))])
    depth = rng.uniform(*[float(x) for x in (mod.get("depth") or (0.0, 0.0))])
    jitter = float(mod.get("gain_jitter_db", 0))
    gain = _clamp(float(spec.get("gain_db", -48)) + rng.uniform(-jitter, jitter), rc.get("gain_db_limits"), -48)
    hum = spec.get("hum") or {}
    return RoomTone(preset=str(name), seed=seed, colour=colour, gain_db=round(gain, 2),
                    rate_hz=round(max(0.1, rate), 3), depth=round(min(1.0, max(0.0, depth)), 3),
                    highpass=float(spec["highpass"]) if spec.get("highpass") else None,
                    lowpass=float(spec["lowpass"]) if spec.get("lowpass") else None,
                    bands=tuple((float(b["f"]), float(b.get("q", 1.0)), float(b["gain_db"]))
                                for b in spec.get("bands") or []),
                    hum_hz=float(hum["hz"]) if hum.get("hz") else None,
                    hum_db=float(hum.get("gain_db", -12)) if hum.get("hz") else None)


def room_tone_graph(rt: RoomTone, *, sample_rate: int, duration_s: float | None = None, out: str = "room",
                    amplitude: float = 1.0) -> str:
    """The filter_complex chains (source filters, no inputs) for a procedural bed, ending in `[out]`: stereo at
    `sample_rate`, at the bed's level. With `duration_s` the sources end there; without, the mix's
    `duration=first` ends them with the voice."""
    sr = int(sample_rate)
    d = f":duration={_num(duration_s)}" if duration_s else ""
    noise = [f"anoisesrc=color={rt.colour}:seed={rt.seed}:sample_rate={sr}:amplitude={_num(amplitude)}{d}"]
    if rt.highpass:
        noise.append(f"highpass=f={_num(rt.highpass)}")
    if rt.lowpass:
        noise.append(f"lowpass=f={_num(rt.lowpass)}")
    for f, q, g in rt.bands:
        noise.append(f"equalizer=f={_num(f)}:t=q:w={_num(q)}:g={_num(g)}")
    if rt.depth > 0:
        noise.append(f"tremolo=f={_num(rt.rate_hz)}:d={_num(rt.depth)}")
    tail = (f"aresample={sr},aformat=sample_fmts=fltp:channel_layouts=stereo,volume={_num(rt.gain_db)}dB[{out}]")
    if rt.hum_hz is None:
        return ",".join(noise) + "," + tail
    chains = [",".join(noise) + f"[{out}_n]",
              f"sine=frequency={_num(rt.hum_hz)}:sample_rate={sr}{d},volume={_num(rt.hum_db or 0)}dB[{out}_h]",
              f"[{out}_n][{out}_h]amix=inputs=2:duration=first:dropout_transition=0:normalize=0," + tail]
    return ";".join(chains)


# --- the voice polish chain ---------------------------------------------------------------------------------------

def room_size(set_words: str | None, policy: Policy | None) -> str | None:
    """The polish room size for a set (`sound.voice_polish.room.by_set` keywords), else the room's default."""
    room = (cfg(policy).get("voice_polish") or {}).get("room") or {}
    return _by_keyword([set_words or ""], room.get("by_set"), "size") or room.get("default")


def voice_polish(size: str | None, policy: Policy | None, *, room: bool = True) -> list[str]:
    """The voice polish filters (`sound.voice_polish`), each element with its own off-switch: a gentle high-pass,
    a de-esser, light compression, and (with `room`) a short aecho for the room `size` (`room.sizes`; a size of
    `none`, or one not listed, adds no room). Values are clamped to the policy's limits."""
    vp = cfg(policy).get("voice_polish") or {}
    if not vp.get("enabled"):
        return []
    lim = vp.get("limits") or {}
    out = []
    hp = vp.get("highpass") or {}
    if _on(hp):
        out.append(f"highpass=f={int(_clamp(hp.get('hz'), lim.get('highpass_hz'), 80))}")
    de = vp.get("deess") or {}
    if _on(de):
        out.append(f"deesser=i={_num(_clamp(de.get('intensity'), (0, 1), 0.4))}"
                   f":m={_num(_clamp(de.get('max_reduction'), (0, 1), 0.5))}"
                   f":f={_num(_clamp(de.get('frequency'), (0, 1), 0.5))}")
    co = vp.get("compress") or {}
    if _on(co):
        out.append(f"acompressor=threshold={_num(_clamp(co.get('threshold'), (0.001, 1), 0.125))}"
                   f":ratio={_num(_clamp(co.get('ratio'), lim.get('ratio'), 2.5))}"
                   f":attack={_num(_clamp(co.get('attack_ms'), (0.01, 2000), 15))}"
                   f":release={_num(_clamp(co.get('release_ms'), (0.01, 9000), 200))}"
                   f":makeup={_num(_clamp(co.get('makeup'), (1, 64), 1))}")
    rm = vp.get("room") or {}
    spec = (rm.get("sizes") or {}).get(size or "") if room and _on(rm) else None
    if spec:
        delays = [_clamp(x, lim.get("delay_ms"), 10) for x in spec.get("delays_ms") or []]
        decays = [_clamp(x, lim.get("decay"), 0.1) for x in spec.get("decays") or []]
        n = min(len(delays), len(decays))
        if n:
            out.append(f"aecho=in_gain={_num(_clamp(spec.get('in_gain'), (0, 1), 1))}"
                       f":out_gain={_num(_clamp(spec.get('out_gain'), (0, 1), 1))}"
                       f":delays={'|'.join(_num(x) for x in delays[:n])}"
                       f":decays={'|'.join(_num(x) for x in decays[:n])}")
    return out


# --- what a post's shot sheets ask for ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SoundDesign:
    room_tone: RoomTone | None   # procedural bed (used when no library file resolves)
    room_size: str | None        # the voice polish's room
    mic: str | None


def design(shots: Iterable[dict] | None, *, seed_key, policy: Policy | None) -> SoundDesign | None:
    """The procedural parts of a post's sound from its shot sheets: room tone from the first shot that names one
    (else the first shot's set), the polish room from that set, the mic from the first shot with one. None when
    sound is off. Works with no shot sheets at all (the defaults: a silent clip still gets a bed)."""
    if not enabled(policy):
        return None
    shots = [s or {} for s in (shots or [])]
    named = next((s for s in shots if str((s.get("sound") or {}).get("room_tone") or "").strip()), None)
    shot = named or (shots[0] if shots else {})
    words = set_text(shot)
    mic = next((str(s["sound"]["mic"]) for s in shots if (s.get("sound") or {}).get("mic")), None)
    return SoundDesign(room_tone=room_tone((shot.get("sound") or {}).get("room_tone"), set_text=words,
                                           seed_key=seed_key, policy=policy),
                       room_size=room_size(words, policy), mic=mic)


# --- the mix -----------------------------------------------------------------------------------------------------

@dataclass
class MixPlan:
    inputs: list[list[str]]      # ffmpeg input arguments after the video (the video is input 0)
    graph: str                   # the -filter_complex graph; its output label is `out`
    out: str = "mix"


def mic_eq(mic: str | None, policy: Policy | None) -> list[str]:
    """The mic character for a shot sheet's `sound.mic` (phone, lav, room): a highpass and a lowpass from
    `sound.mic_eq`. No mic, or one not listed, is no filter."""
    spec = (cfg(policy).get("mic_eq") or {}).get(mic or "") or {}
    out = []
    if spec.get("highpass"):
        out.append(f"highpass=f={int(spec['highpass'])}")
    if spec.get("lowpass"):
        out.append(f"lowpass=f={int(spec['lowpass'])}")
    return out


def duck_filter(policy: Policy | None, spec: dict | None = None) -> str:
    """sidechaincompress for a bed under the voice (`spec`, else the music's `sound.mix.duck`), every value clamped
    to `sound.mix.limits`."""
    m = cfg(policy).get("mix") or {}
    d, lim = (m.get("duck") or {}) if spec is None else spec, m.get("limits") or {}
    threshold = _clamp(d.get("threshold"), lim.get("threshold") or (0.000976563, 1.0), 0.05)
    ratio = _clamp(d.get("ratio"), lim.get("ratio") or (1, 20), 6)
    attack = _clamp(d.get("attack_ms"), lim.get("attack_ms") or (50, 100), 80)
    release = _clamp(d.get("release_ms"), lim.get("release_ms") or (300, 500), 400)
    makeup = _clamp(d.get("makeup"), (1, 64), 1)
    return (f"sidechaincompress=threshold={_num(threshold)}:ratio={_num(ratio)}:attack={_num(attack)}"
            f":release={_num(release)}:makeup={_num(makeup)}")


def plan(*, has_voice: bool, duration_s: float | None = None, room_tone: Path | str | None = None,
         effects: Iterable[tuple[Path | str, float, float | None]] = (), music: Path | str | None = None,
         ir: Path | str | None = None, mic: str | None = None, room_tone_synth: RoomTone | None = None,
         room_size: str | None = None, polish: bool | None = None, policy: Policy | None) -> MixPlan:
    """The inputs and filter graph for `mix`. `effects` are (path, at_s, gain_db or None for the policy gain).
    A video without audio gets a silent stand-in voice of `duration_s`, so every bed ends with the video.
    `room_tone_synth` is a procedural bed, used when no `room_tone` file is given. `polish` (None = policy) runs
    the voice polish chain with the room `room_size` (None = the policy default); an impulse response replaces
    its aecho room."""
    c = cfg(policy)
    m = c.get("mix") or {}
    sr = int(c.get("sample_rate", 48000))
    fmt = f"aresample={sr},aformat=sample_fmts=fltp:channel_layouts=stereo"
    inputs: list[list[str]] = []
    chains: list[str] = []

    def add(args: list[str]) -> int:
        inputs.append(args)
        return len(inputs)          # input 0 is the video

    if has_voice:
        voice_src = "0:a"
    else:
        if not duration_s or float(duration_s) <= 0:
            raise ValueError("a video without audio needs its duration to mix beds under it")
        voice_src = f"{add(['-f', 'lavfi', '-t', _num(duration_s), '-i', f'anullsrc=r={sr}:cl=stereo'])}:a"
    voice_chain = [fmt]
    if has_voice:
        voice_chain += mic_eq(mic, policy)
        if polish is not False:
            size = room_size if room_size is not None else ((c.get("voice_polish") or {}).get("room") or {}).get(
                "default")
            voice_chain += voice_polish(size, policy, room=ir is None)
    chains.append(f"[{voice_src}]{','.join(voice_chain)}[vo]")
    voice = "vo"

    if ir is not None and has_voice:
        n = add(["-i", str(ir)])
        wet = _clamp(m.get("room_wet"), (0.0, 1.0), 0.12)
        chains.append(f"[{n}:a]aresample={sr},aformat=sample_fmts=fltp:channel_layouts=mono[ir]")
        chains.append("[vo]asplit=2[vo_dry][vo_wet_in]")
        chains.append("[vo_wet_in][ir]afir[vo_wet]")
        chains.append(f"[vo_dry][vo_wet]amix=inputs=2:weights='1 {_num(wet)}':duration=first"
                      f":dropout_transition=0:normalize=0[vo_room]")
        voice = "vo_room"

    beds: list[str] = []
    ducks: list[tuple[str, str, str]] = []      # (bed in, bed out, sidechaincompress) keyed by the voice
    room_duck = (c.get("room_tone") or {}).get("duck") or {}
    if room_tone is not None or room_tone_synth is not None:
        if room_tone is not None:
            n = add(["-stream_loop", "-1", "-i", str(room_tone)])
            chains.append(f"[{n}:a]{fmt},volume={_num(float(m.get('room_tone_gain_db', -30)))}dB[room_in]")
        else:
            rc = c.get("room_tone") or {}
            chains.append(room_tone_graph(room_tone_synth, sample_rate=sr, duration_s=duration_s, out="room_in",
                                          amplitude=float(rc.get("amplitude", 1.0))))
        if has_voice and _on(room_duck):
            ducks.append(("room_in", "room", duck_filter(policy, room_duck)))
            beds.append("room")
        else:
            beds.append("room_in")
    for k, (path, at_s, gain_db) in enumerate(effects or ()):
        if float(at_s) < 0:
            raise ValueError(f"effect {k} starts before the video ({at_s} s)")
        n = add(["-i", str(path)])
        gain = float(m.get("effect_gain_db", -12) if gain_db is None else gain_db)
        ms = int(round(float(at_s) * 1000))
        chains.append(f"[{n}:a]{fmt},volume={_num(gain)}dB,adelay={ms}|{ms}[fx{k}]")
        beds.append(f"fx{k}")
    if music is not None:
        n = add(["-stream_loop", "-1", "-i", str(music)])
        chains.append(f"[{n}:a]{fmt},volume={_num(float(m.get('music_gain_db', -14)))}dB[mus_in]")
        if has_voice:   # the voice keys the compressor: the music dips while she speaks
            ducks.append(("mus_in", "mus", duck_filter(policy)))
            beds.append("mus")
        else:
            beds.append("mus_in")

    if ducks:
        keys = [f"vo_key_{bed_out}" for _, bed_out, _ in ducks]
        chains.append(f"[{voice}]asplit={len(keys) + 1}[vo_mix]" + "".join(f"[{k}]" for k in keys))
        for (bed_in, bed_out, f), k in zip(ducks, keys):
            chains.append(f"[{bed_in}][{k}]{f}[{bed_out}]")
        voice = "vo_mix"

    labels = [voice] + beds
    if len(labels) == 1:
        chains.append(f"[{voice}]anull[mix]")
    else:
        chains.append("".join(f"[{x}]" for x in labels)
                      + f"amix=inputs={len(labels)}:duration=first:dropout_transition=0:normalize=0[mix]")
    return MixPlan(inputs=inputs, graph=";".join(chains))


def _has_audio(video: Path, timeout: float) -> bool:
    from .media import ffmpeg
    r = subprocess.run([ffmpeg(), "-hide_banner", "-i", str(video)], capture_output=True, text=True, timeout=timeout)
    return bool(re.search(r"Stream #.*Audio:", r.stderr or ""))


def mix(video: Path | str, out: Path | str, *, room_tone: Path | str | None = None,
        effects: Iterable[tuple[Path | str, float, float | None]] = (), music: Path | str | None = None,
        ir: Path | str | None = None, mic: str | None = None, room_tone_synth: RoomTone | None = None,
        room_size: str | None = None, polish: bool | None = None, policy: Policy,
        timeout: float | None = None) -> Path:
    """The video with its sound designed (see the module docstring) into `out`: video copied, audio AAC. Inputs are
    local files the caller fetched (library URIs go through the object store first)."""
    from .media import duration, ffmpeg
    m = cfg(policy).get("mix") or {}
    timeout = float(timeout or m.get("timeout_seconds", 600))
    video, out = Path(video), Path(out)
    has_voice = _has_audio(video, timeout)
    length = duration(video) if (not has_voice or room_tone_synth is not None) else None
    p = plan(has_voice=has_voice, duration_s=length, room_tone=room_tone, effects=list(effects or ()), music=music,
             ir=ir, mic=mic, room_tone_synth=room_tone_synth, room_size=room_size, polish=polish, policy=policy)
    cmd = [ffmpeg(), "-y", "-v", "error", "-i", str(video)]
    for args in p.inputs:
        cmd += args
    cmd += ["-filter_complex", p.graph, "-map", "0:v:0?", "-map", f"[{p.out}]", "-c:v", "copy",
            "-c:a", "aac", "-b:a", str(m.get("audio_bitrate", "320k")), "-ac", "2", "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg could not mix {video.name}: {(r.stderr or '')[-500:]}")
    return out
