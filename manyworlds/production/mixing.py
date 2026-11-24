"""The sound under a finished video (policy `sound`): library cues from the shot sheets over the procedural design."""

from __future__ import annotations

from typing import Callable

from ..policy import Policy
from .common import _pieces, _shot
from .takes import _take_index


def _piece_loader(conn, video: dict) -> Callable[[], tuple[dict, list[dict]]]:
    """(the made video's asset meta, its pieces), read when first needed: the finish's cue and span helpers also run
    on the finisher (finish.video) from the pieces the producer read for it."""
    def load():
        row, pieces = _pieces(conn, video)
        return row["meta"] or {}, pieces
    return load


# Sound (policy `sound`, on as shipped for the parts that need no GPU): before finishing, every finished video is mixed
# once (sound.mix) while sound.enabled. The procedural design (sound.design: a room tone synthesised from the shot's
# sound.room_tone text or its set, the voice polish room, the mic) is always there, even without shot sheets, so no
# video ships studio-dry or dead silent; a silent format (sound.silent_beds) also gets its music bed
# (sound.bed_music). On top, the shot sheets' cues (sound.room_tone, .effects, .music: free text) are looked up as
# tags in the licence-checked library (manyworlds.sound.library; a cue may also match by one of its comma-separated
# parts): a library room tone replaces the synthesised one; each effect plays sound.mix.effect_at_s into its own
# piece (shot sheets carry no times). The finish stage's loudnorm then sets the level of the whole.
def _sound_ref(lib, cue: str):
    from .. import sound
    cue = str(cue or "").strip()
    for tag in dict.fromkeys([cue, *(p.strip() for p in cue.split(","))]):
        if not tag:
            continue
        try:
            ref = lib.lookup(tag)
        except sound.UnlicensedSound:
            ref = None   # held but not usable: never mixed in
        if ref is not None:
            return ref
    return None


def sound_cues(conn, policy: Policy, body: dict, video: dict) -> dict | None:
    """What to mix under a post's video: {room_tone, music, effects: [(SoundRef, at_s)], ir, mic}, or None when
    sound is off, the post has no shot sheets, or none of their cues resolve in the library."""
    return _sound_cues_from(policy, body, _piece_loader(conn, video))


def _sound_cues_from(policy: Policy, body: dict, load: Callable[[], tuple[dict, list[dict]]]) -> dict | None:
    """sound_cues from `load()`: (the video's asset meta, its pieces), called only when the cues need them."""
    from .. import sound
    if not sound.enabled(policy) or not body.get("shots"):
        return None
    lib = sound.library(policy)
    if not lib.tags():
        return None
    meta, pieces = load()
    measured = (meta or {}).get("segment_seconds") or []
    segmented = len(body.get("talk_segments") or []) == len(pieces) > 1
    at = float(((sound.cfg(policy).get("mix") or {}).get("effect_at_s", 0.5)))
    out: dict = {"room_tone": None, "music": None, "effects": [], "ir": None, "mic": None}
    t = 0.0
    for i, piece in enumerate(pieces):
        idx = _take_index(body, i if segmented else None) if body.get("scenes") else 0
        shot = _shot(body, idx) or {}
        snd = shot.get("sound") or {}
        if out["mic"] is None and snd.get("mic"):
            out["mic"] = str(snd["mic"])
        if out["ir"] is None:
            for key in (((shot.get("set") or {}).get("set_id") or ""), str(snd.get("mic") or "")):
                try:
                    out["ir"] = sound.impulse_response(key, policy) if key else None
                except sound.UnlicensedSound:
                    out["ir"] = None
                if out["ir"] is not None:
                    break
        for field in ("room_tone", "music"):
            if out[field] is None and snd.get(field):
                out[field] = _sound_ref(lib, snd[field])
        for cue in snd.get("effects") or []:
            ref = _sound_ref(lib, cue)
            if ref is not None:
                out["effects"].append((ref, round(t + at, 3)))
        if i < len(pieces) - 1:
            length = measured[i] if i < len(measured) and measured[i] is not None else (piece["meta"] or {}).get("seconds")
            if length is None:
                break   # later pieces can't be placed: their effects are left out
            t += float(length)
    if not (out["room_tone"] or out["music"] or out["effects"]):
        return None
    return out


def sound_plan(conn, policy: Policy, post: dict, body: dict, video: dict) -> dict | None:
    """Everything mixed under a post's video: the library cues (sound_cues) over the procedural design (sound.design):
    {room_tone, room_tone_synth (only when no library room tone resolves), room_size, music (a library cue, else the
    silent format's bed), effects, ir, mic}. None only when sound is off."""
    return _sound_plan_from(policy, post, body, _piece_loader(conn, video))


def _sound_plan_from(policy: Policy, post: dict, body: dict,
                     load: Callable[[], tuple[dict, list[dict]]]) -> dict | None:
    """sound_plan from `load()` (see _sound_cues_from): the same plan with no database, on the finisher. A sung post
    (music_video) is the one exception: its track is the song's full mix (body.song_media.mix) in place of the
    lip-synced vocal stem, nothing under it, whether or not sound is on (a vocal stem alone never ships)."""
    from .. import sound
    song = _song_ref(post, body, policy)
    if song is not None:
        return {"song": song, "room_tone": None, "music": None, "effects": [], "ir": None, "mic": None,
                "room_tone_synth": None, "room_size": None}
    design = sound.design(body.get("shots"), seed_key=str(post["id"]), policy=policy)
    if design is None:
        return None
    cues = _sound_cues_from(policy, body, load) or {"room_tone": None, "music": None, "effects": [], "ir": None,
                                                    "mic": None}
    if cues["music"] is None:
        cues["music"] = sound.bed_music(sound.library(policy), post["format"], policy, seed_key=str(post["id"]))
    return {**cues, "room_tone_synth": design.room_tone if cues["room_tone"] is None else None,
            "room_size": design.room_size, "mic": cues["mic"] or design.mic}


def _song_ref(post: dict, body: dict, policy: Policy):
    """The song under a sung post (production.common.SUNG), as a library-style SoundRef, or None."""
    from .. import sound
    from .common import SUNG
    mix = (body.get("song_media") or {}).get("mix") if post.get("format") in SUNG else None
    if not mix or not mix.get("uri"):
        return None
    spec = body.get("song") or {}
    return sound.SoundRef(uri=str(mix["uri"]), licence=str(policy.get("music", "library_licence", default="own-generated")),
                          source=f"music.song (asset {mix.get('asset_id')}): {spec.get('title') or 'untitled'}",
                          tag=str(spec.get("title") or "song"), kind="music")


def _mix_sound(policy: Policy, store, cues: dict, raw, workdir) -> tuple:
    """The raw video with its sound mixed (sound.mix), and what was used (tags, licences, credits) for the post."""
    from pathlib import Path

    from .. import sound
    d = Path(workdir)

    def local(ref, name):
        if "://" not in ref.uri:
            return Path(ref.uri)
        return store.get_file(ref.uri, d / f"{name}{Path(ref.uri).suffix or '.wav'}")
    used = [r for r in (cues.get("song"), cues["room_tone"], cues["music"], *(e for e, _ in cues["effects"]),
                        cues["ir"]) if r]
    mixed = sound.mix(raw, d / "mixed.mp4", song=local(cues["song"], "song") if cues.get("song") else None,
                      room_tone=local(cues["room_tone"], "room") if cues["room_tone"] else None,
                      effects=[(local(r, f"fx{k}"), at_s, None) for k, (r, at_s) in enumerate(cues["effects"])],
                      music=local(cues["music"], "music") if cues["music"] else None,
                      ir=local(cues["ir"], "ir") if cues["ir"] else None, mic=cues["mic"],
                      room_tone_synth=cues.get("room_tone_synth"), room_size=cues.get("room_size"), policy=policy)
    return mixed, [{"tag": r.tag, "uri": r.uri, "licence": r.licence, "source": r.source, "credit": r.credit}
                   for r in used]
