"""Sound design for a video (the shot sheet's `sound` module: mic, room tone, effects, music), on the control plane.

Plan 2026-11-22-video-quality-autonomous, wave 3 item 9; research docs/research/2026-11-22-video-quality-wave2-3.md
section 2. Everything here is switched off (policy `sound.enabled: false`) and nothing calls it yet.

  mix        one ffmpeg pass over a video's own audio (the voice):
               voice      resampled to stereo at `sound.sample_rate`, a mic-character EQ per shot-sheet mic
                          (`sound.mic_eq`: highpass/lowpass), and optionally the room: the voice convolved with an
                          impulse response (`afir`) mixed in at a low wet share (`sound.mix.room_wet`)
               room tone  a looped bed at a low level (`sound.mix.room_tone_gain_db`)
               effects    each placed at its time (`adelay`) at its gain
               music      a looped bed, ducked under the voice (`sidechaincompress` keyed by the voice; attack and
                          release clamped to `sound.mix.limits`: 50-100 ms and 300-500 ms)
             The video stream is copied. Loudness is NOT set here: the finish stage (manyworlds.finishing) runs
             two-pass loudnorm on whatever this writes, so the beds are summed without normalising (`amix
             normalize=0`) and the voice keeps its level.
  library    `library(policy).lookup(tag) -> SoundRef{uri, licence, source}` from the policy catalogue
             (`sound.library.entries`, empty by default). An entry without a licence string, without a source,
             or with a licence that is not in `sound.library.allowed_licences` is refused, never used.
  impulse responses  `impulse_response(key, policy)`: policy `sound.impulse_responses` (none shipped); each needs a
             path and a licence (OpenAIR files are licensed per file: check and credit each one).

Generators (Stable Audio 3 Small-SFX for effects and room tone, ACE-Step 1.5 for music beds) are worker handlers in
manyworlds/worker/handlers/sound_gen.py, also off. `plan` is pure (tested in CI); `mix` runs ffmpeg
(manyworlds.media.ffmpeg, the imageio-ffmpeg static build).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .policy import Policy


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


# --- the library -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SoundRef:
    uri: str
    licence: str
    source: str
    tag: str = ""
    credit: str = ""


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
    the reason) so `mw` can show them, and are never returned."""

    def __init__(self, entries: Iterable[dict] | None, *, allowed_licences: Iterable[str] | None = None):
        self.allowed = {str(x).strip().lower() for x in allowed_licences} if allowed_licences is not None else None
        self.refs: dict[str, list[SoundRef]] = {}
        self.refused: list[dict] = []
        for raw in entries or []:
            entry = dict(raw or {})
            tag = _key(entry.get("tag"))
            reason = "no tag" if not tag else _refusal(entry, self.allowed)
            if reason:
                self.refused.append({**entry, "reason": reason})
                continue
            self.refs.setdefault(tag, []).append(SoundRef(
                uri=str(entry["uri"]).strip(), licence=str(entry["licence"]).strip(),
                source=str(entry["source"]).strip(), tag=tag, credit=str(entry.get("credit") or "").strip()))

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


def duck_filter(policy: Policy | None) -> str:
    """sidechaincompress for the music under the voice, every value clamped to `sound.mix.limits`."""
    m = cfg(policy).get("mix") or {}
    d, lim = m.get("duck") or {}, m.get("limits") or {}
    threshold = _clamp(d.get("threshold"), lim.get("threshold") or (0.000976563, 1.0), 0.05)
    ratio = _clamp(d.get("ratio"), lim.get("ratio") or (1, 20), 6)
    attack = _clamp(d.get("attack_ms"), lim.get("attack_ms") or (50, 100), 80)
    release = _clamp(d.get("release_ms"), lim.get("release_ms") or (300, 500), 400)
    makeup = _clamp(d.get("makeup"), (1, 64), 1)
    return (f"sidechaincompress=threshold={_num(threshold)}:ratio={_num(ratio)}:attack={_num(attack)}"
            f":release={_num(release)}:makeup={_num(makeup)}")


def plan(*, has_voice: bool, duration_s: float | None = None, room_tone: Path | str | None = None,
         effects: Iterable[tuple[Path | str, float, float | None]] = (), music: Path | str | None = None,
         ir: Path | str | None = None, mic: str | None = None, policy: Policy | None) -> MixPlan:
    """The inputs and filter graph for `mix`. `effects` are (path, at_s, gain_db or None for the policy gain).
    A video without audio gets a silent stand-in voice of `duration_s`, so every bed ends with the video."""
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
    voice_chain = [fmt] + (mic_eq(mic, policy) if has_voice else [])
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
    if room_tone is not None:
        n = add(["-stream_loop", "-1", "-i", str(room_tone)])
        chains.append(f"[{n}:a]{fmt},volume={_num(float(m.get('room_tone_gain_db', -30)))}dB[room]")
        beds.append("room")
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
            chains.append(f"[{voice}]asplit=2[vo_mix][vo_key]")
            chains.append(f"[mus_in][vo_key]{duck_filter(policy)}[mus]")
            voice = "vo_mix"
            beds.append("mus")
        else:
            beds.append("mus_in")

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
        ir: Path | str | None = None, mic: str | None = None, policy: Policy, timeout: float | None = None) -> Path:
    """The video with its sound designed (see the module docstring) into `out`: video copied, audio AAC. Inputs are
    local files the caller fetched (library URIs go through the object store first)."""
    from .media import duration, ffmpeg
    m = cfg(policy).get("mix") or {}
    timeout = float(timeout or m.get("timeout_seconds", 600))
    video, out = Path(video), Path(out)
    has_voice = _has_audio(video, timeout)
    p = plan(has_voice=has_voice, duration_s=None if has_voice else duration(video), room_tone=room_tone,
             effects=list(effects or ()), music=music, ir=ir, mic=mic, policy=policy)
    cmd = [ffmpeg(), "-y", "-v", "error", "-i", str(video)]
    for args in p.inputs:
        cmd += args
    cmd += ["-filter_complex", p.graph, "-map", "0:v:0?", "-map", f"[{p.out}]", "-c:v", "copy",
            "-c:a", "aac", "-b:a", str(m.get("audio_bitrate", "320k")), "-ac", "2", "-movflags", "+faststart", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg could not mix {video.name}: {(r.stderr or '')[-500:]}")
    return out
