"""Takes: how a talking video's script is cut (one take per speaking shot for a directed post, else the hybrid's
short segments or a long script's segments) and which shot, photo and words each take has."""

from __future__ import annotations

from ..policy import Policy


def _talk_long_cfg(policy: Policy) -> dict:
    return policy.get("generation", "talk_long", default={}) or {}


def talk_segments(policy: Policy, script: str, *, hybrid: bool = False) -> list[str]:
    from ..talk_segments import split_script
    cfg = _talk_long_cfg(policy)
    seconds = _hybrid_cfg(policy).get("segment_seconds", 10) if hybrid else cfg.get("max_segment_seconds", 25)
    return split_script(script, max_seconds=float(seconds), words_per_second=float(cfg.get("words_per_second", 2.5)))


def _hybrid_cfg(policy: Policy) -> dict:
    return policy.get("generation", "talk_hybrid", default={}) or {}


def hybrid_modes(segments: list[str], on_camera_share: float) -> list[str]:
    """Which segments are lip-synced on camera ('talk') and which are cutaways over the post's other photos
    ('narrated'), in the same voice. The hook and the sign-off are always on camera; middle segments join them,
    shortest first, until on-camera speech is at least `on_camera_share` of the words. Only the on-camera part
    needs the expensive lip-sync render."""
    n = len(segments)
    if n <= 2:
        return ["talk"] * n
    modes = ["talk"] + ["narrated"] * (n - 2) + ["talk"]
    words = [len(s.split()) for s in segments]
    total = sum(words) or 1
    for i in sorted(range(1, n - 1), key=lambda i: words[i]):
        if sum(w for w, m in zip(words, modes) if m == "talk") / total >= on_camera_share:
            break
        modes[i] = "talk"
    return modes


def shot_takes(policy: Policy, shots: list, *, cutaways: bool = True,
               narrate=None) -> list[tuple[str, str, int]]:
    """A directed talking video's takes, one per shot that speaks: (text, mode, shot index), in order. The text is
    what the shot's lines say (`direction.spoken`, deliveries never included); mode is 'talk' (lip-synced on camera)
    or 'narrated' for a `cutaway` shot (its photo under the same voice). Without `cutaways` (hybrid off, or one
    photo), a cutaway's lines join the on-camera take before it (or after it, for a leading cutaway); when only
    cutaways speak, their lines become one on-camera take on the first shot that isn't a cutaway. A take longer than generation.talk.max_take_seconds is split at sentence ends, as a safety
    net: the scene changes only where the shots do. [] when no shot speaks."""
    from ..direction import ShotSheet, spoken
    from ..talk_segments import split_script
    sheets = [ShotSheet.model_validate(s) if isinstance(s, dict) else s for s in shots]
    # `narrate` (the edit grammar, manyworlds.cuts.narrates): a speaking shot that doesn't show her face to camera
    # is a cutaway under the voice too, so only the face-to-camera lines pay for lip sync.
    cut_of = [bool(s.cutaway) or bool(narrate and narrate(s)) for s in sheets]
    takes = [(spoken(s), cut_of[i], i) for i, s in enumerate(sheets)]
    takes = [t for t in takes if t[0]]
    if not takes:
        return []
    if all(cut for _, cut, _ in takes):
        # Only cutaways speak: a talking video needs her face, so the lines go to the first on-camera shot.
        anchor = next((i for i in range(len(sheets)) if not cut_of[i]), None)
        takes = ([(" ".join(text for text, _, _ in takes), False, anchor)] if anchor is not None
                 else [(text, False, i) for text, _, i in takes])
    elif not cutaways:
        merged: list[tuple[str, bool, int]] = []
        lead = ""   # cutaway lines before the first on-camera take
        for text, cut, i in takes:
            if cut:
                if merged:
                    merged[-1] = (f"{merged[-1][0]} {text}", False, merged[-1][2])
                else:
                    lead = f"{lead} {text}".strip()
                continue
            merged.append((f"{lead} {text}".strip(), False, i))
            lead = ""
        takes = merged
    max_take = float(policy.get("generation", "talk", "max_take_seconds", default=60))
    wps = float(_talk_long_cfg(policy).get("words_per_second", 2.5))
    return [(part, "narrated" if cut else "talk", i) for text, cut, i in takes
            for part in split_script(text, max_seconds=max_take, words_per_second=wps)]


def _directed(policy: Policy, body: dict) -> bool:
    """Whether a talking video's takes follow its shots: always for a directed post (body.shots), which is cut only
    where its shots change, never split into the hybrid's short segments."""
    return bool(body.get("shots"))


def script_takes(policy: Policy, body: dict) -> list[tuple[str, str, int]]:
    """A directed post whose shots speak no lines: the post's script as on-camera takes on its first on-camera shot,
    split only past generation.talk.max_take_seconds (as shot_takes splits a long take). [] when there's no script."""
    from ..direction import ShotSheet
    from ..talk_segments import split_script
    script = str(body.get("script") or "").strip()
    if not script:
        return []
    sheets = [ShotSheet.model_validate(s) if isinstance(s, dict) else s for s in body.get("shots") or []]
    anchor = next((i for i, s in enumerate(sheets) if not s.cutaway), 0)
    max_take = float(policy.get("generation", "talk", "max_take_seconds", default=60))
    wps = float(_talk_long_cfg(policy).get("words_per_second", 2.5))
    return [(part, "talk", anchor) for part in split_script(script, max_seconds=max_take, words_per_second=wps)]


def _take_shot(body: dict, segment: int | None) -> int:
    """The shot (and scene) a talk segment belongs to: from body.talk_shots for a directed post, else scene 0."""
    shots = body.get("talk_shots") or []
    k = segment or 0
    return int(shots[k]) if k < len(shots) else 0


def _photo_per_shot(body: dict, chosen: list[dict]) -> bool:
    """Whether every shot has its own chosen photo (chosen photos follow the scenes in order)."""
    n_shots = len(body.get("shots") or [])
    return bool(n_shots) and len(chosen) >= n_shots


def _take_photo(body: dict, chosen: list[dict], i: int) -> dict:
    """The chosen photo for shot i when the post has one per shot, else the first."""
    return chosen[i] if _photo_per_shot(body, chosen) and i < len(chosen) else chosen[0]


def _take_index(body: dict, segment: int | None) -> int:
    """The scene (and shot) a talk take is made from: its shot, or scene 0 when that is out of range."""
    i = _take_shot(body, segment)
    return i if i < len(body["scenes"]) else 0


def _take_script(body: dict, segment: int | None) -> str:
    return (body["talk_segments"][segment] if segment is not None
            else body.get("talk_script") or body.get("script", ""))
