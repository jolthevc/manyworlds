"""Finer checks on photos and clips, beside the main gates (policy `review.detail`).

  hands        the judge rates visible hands 1-10 (fingers, joints, grip), null when none are in frame
  teeth        the judge rates visible teeth 1-10 (count, shape, no smear), null when none show
  consistency  a shot naming a recurring set (set.set_id) or outfit (wardrobe.outfit_id), with reference images
               in the job payload's `references` [{kind: set|outfit, uri}]: the judge compares, 1-10
  lip_closure  talking clips: the lips close on each b, p and m (worker/lipsync.py's mouth series against the
               script's estimated word times); the share that closed
  background   keyframes: how much the edges outside the subject box change from one keyframe to the next
               (a warping or flickering background); 0 = identical, 1 = unrelated

Each check has `enabled` (measure it and write its own review row) and `enforce` (only then can it reject). A row
that isn't enforced is written passed, with `would_pass` in its details, because production accepts an output
only when every row passed. An output without the signal (no hands in frame, no references, no mouth series)
gets no row: it behaves exactly as before. The reviewer's image has no MediaPipe, so landmark crops of hands and
the mouth come from the clip worker (`crops`, worker/crops.py) and are shown beside the keyframes when it made
them; otherwise hands and teeth are scored on the full frames.
"""

from __future__ import annotations

from ..policy import Policy

JUDGED = ("hands", "teeth", "consistency")


def cfg(policy: Policy | None) -> dict:
    return (policy.get("review", "detail", default={}) or {}) if policy is not None else {}


def check(policy: Policy | None, name: str) -> dict:
    return cfg(policy).get(name) or {}


def enabled(policy: Policy | None, name: str) -> bool:
    return bool(check(policy, name).get("enabled"))


def verdict(policy: Policy | None, name: str, value, *, lower_is_better: bool = False) -> tuple[bool, dict] | None:
    """(the row's passed value, details) for one check's score, or None when the check is off or has no score.
    The bar is `min_score` (or `min_share`), or `max_flicker` for a lower-is-better measure."""
    c = check(policy, name)
    if not c.get("enabled") or value is None:
        return None
    if lower_is_better:
        bar = float(c.get("max_flicker", 1.0))
        ok = float(value) <= bar
    else:
        bar = float(c.get("min_score", c.get("min_share", 0)))
        ok = float(value) >= bar
    enforce = bool(c.get("enforce"))
    return (ok or not enforce), {"would_pass": ok, "enforced": enforce, "bar": bar}


def judged_verdicts(policy: Policy | None, review, *, consistency: bool) -> dict[str, tuple[bool, dict]]:
    """The judge-scored checks present on one review (gates.CandidateReview or video.ClipReview)."""
    out = {}
    for name in JUDGED:
        if name == "consistency" and not consistency:
            continue
        v = verdict(policy, name, getattr(review, name, None))
        if v is not None:
            out[name] = v
    return out


def clamp_scores(review, policy: Policy | None, *, consistency: bool) -> None:
    """Scores the judge returned for checks that are off (or no references) are dropped; the rest held to 1-10."""
    for name in JUDGED:
        v = getattr(review, name, None)
        keep = enabled(policy, name) and (consistency or name != "consistency")
        setattr(review, name, max(1, min(10, int(v))) if (keep and v is not None) else None)


def unasked(review, shot: dict | None) -> set[str] | None:
    """Fields left out of a review row's details: `direction` for an undirected output, and finer-check scores
    the judge didn't give, so rows of outputs without those signals look as they did before."""
    out = ({"direction"} if not shot else set()) | {n for n in JUDGED if getattr(review, n, None) is None}
    from .common import CONTEXT_FIELDS   # brand alignment, distribution and detail defects only when given
    out |= {n for n in CONTEXT_FIELDS if hasattr(review, n) and getattr(review, n) in (None, [])}
    from .common import PERFORMANCE_FIELDS   # the performance scores only when given (manyworlds.performance)
    out |= {n for n in PERFORMANCE_FIELDS if hasattr(review, n) and getattr(review, n) in (None, [], "")}
    return out or None


def judge_prompt(policy: Policy | None, *, consistency: list[str] | None = None, still: bool,
                 crops: set[str] | None = None) -> str | None:
    """The judge's instruction for the enabled judged checks, or None when there are none. `crops`: the close-up
    parts shown (crop_blocks), which hands and teeth are then judged from as well."""
    what = "the photo" if still else "the keyframes (score the worst one)"
    crops = crops or set()
    parts = []
    if enabled(policy, "hands"):
        also = " and the HAND CLOSE-UPs (look closely at those)" if "hands" in crops else ""
        parts.append(f"`hands` 1-10 for any visible hands in {what}{also}: 10 = natural fingers, joints and grip; "
                     "deduct for extra, missing, fused or bent fingers and impossible grips; null when no hand is "
                     "visible")
    if enabled(policy, "teeth"):
        also = " and the MOUTH CLOSE-UPs (look closely at those)" if "mouth" in crops else ""
        parts.append(f"`teeth` 1-10 for any visible teeth in {what}{also}: 10 = a natural single row of plausible "
                     "teeth; deduct for too many, merged, smeared or shifting teeth; null when no teeth show")
    if consistency and enabled(policy, "consistency"):
        parts.append("`consistency` 1-10 against the SET / OUTFIT REFERENCE images (" + ", ".join(consistency)
                     + "): 10 = the same place (layout, furniture, colours, light fittings) and the same clothes "
                     "(garments, colours, cut, accessories); 5 = recognisably similar, details changed; 1 = different")
    if not parts:
        return None
    return "DETAIL CHECKS (name each problem in `defects`): " + "; ".join(parts) + "."


def references_for(payload: dict | None, shot: dict | None, policy: Policy | None) -> list[dict]:
    """The set and outfit reference images this output should match: only kinds the shot names an id for
    (set.set_id for `set`, wardrobe.outfit_id for `outfit`), up to `consistency.max_references`."""
    if not enabled(policy, "consistency") or not isinstance(shot, dict) or not isinstance(payload, dict):
        return []
    ids = {"set": ((shot.get("set") or {}).get("set_id") or "").strip(),
           "outfit": ((shot.get("wardrobe") or {}).get("outfit_id") or "").strip()}
    out = []
    for r in payload.get("references") or []:
        if not isinstance(r, dict) or not r.get("uri"):
            continue
        kind = r.get("kind")
        if kind in ids and ids[kind]:
            out.append({"kind": kind, "id": ids[kind], "uri": r["uri"]})
    return out[:int(check(policy, "consistency").get("max_references", 2))]


def reference_blocks(refs: list[dict], store, workdir, image_block) -> tuple[list, list[str]]:
    """Judge content for the references (label then image), and their labels; a reference that can't be fetched
    is left out."""
    from pathlib import Path
    content, labels = [], []
    for i, r in enumerate(refs):
        try:
            data = store.get_file(r["uri"], Path(workdir) / f"detail_ref_{i}{Path(r['uri']).suffix or '.png'}").read_bytes()
        except Exception:  # noqa: BLE001 - no reference, no consistency score
            continue
        label = f"{r['kind'].upper()} REFERENCE ({r['id']})"
        content += [{"type": "text", "text": f"{label}:"}, image_block(data, _media(r["uri"]))]
        labels.append(label)
    return content, labels


CROP_LABELS = {"hands": "HAND", "mouth": "MOUTH"}


def crop_blocks(rows: list[dict], store, workdir, image_block, policy: Policy | None) -> tuple[list, set[str]]:
    """Judge content for a clip's close-up crops (assets kind crop, meta {of, part}; worker/crops.py): a label
    naming the part and its keyframe, then the image, at most crops.max_per_clip, only for parts whose check
    (hands, or teeth for the mouth) is on. Returns (content, the parts shown). Nothing when crops are off or the
    worker made none; a crop that can't be fetched is left out."""
    from pathlib import Path
    c = check(policy, "crops")
    if not c.get("enabled") or not rows:
        return [], set()
    wanted = {"hands"} if enabled(policy, "hands") and c.get("hands", True) else set()
    if enabled(policy, "teeth") and c.get("mouth", True):
        wanted.add("mouth")
    content, shown = [], set()
    for i, r in enumerate(rows):
        if len(content) // 2 >= int(c.get("max_per_clip", 8)):
            break
        meta = r.get("meta") or {}
        part = meta.get("part")
        if part not in wanted:
            continue
        try:
            data = store.get_file(r["uri"], Path(workdir) / f"crop_{i}{Path(r['uri']).suffix or '.png'}").read_bytes()
        except Exception:  # noqa: BLE001 - the full keyframes still show it
            continue
        of = meta.get("of")
        where = f" from KEYFRAME {int(of) + 1}" if isinstance(of, (int, float)) else ""
        content += [{"type": "text", "text": f"{CROP_LABELS[part]} CLOSE-UP{where}:"}, image_block(data, _media(r["uri"]))]
        shown.add(part)
    return content, shown


def _media(uri: str) -> str:
    return "image/jpeg" if uri.lower().endswith((".jpg", ".jpeg")) else "image/png"


# --- lip closure on bilabials ----------------------------------------------------------------------------------

def lip_closure(sync: dict | None, policy: Policy | None, *, script: str = "", lines: list[dict] | None = None,
                seconds: float | None = None) -> dict | None:
    """The lip-closure measure for a talking clip, or None when off or there is nothing to measure. The worker's
    own result (`sync.bilabial`, from timed words) is used when present; otherwise it is computed here from the
    mouth series (`sync.openness`, `sync.fps`) and the voice's line timings, or the script spread over the clip."""
    if not enabled(policy, "lip_closure") or not isinstance(sync, dict) or "error" in sync:
        return None
    c = check(policy, "lip_closure")
    if isinstance(sync.get("bilabial"), dict):
        return {**sync["bilabial"], "timing": "words"}
    openness, fps = sync.get("openness"), sync.get("fps")
    if not openness or not fps or not ((script or "").strip() or lines):
        return None
    from ..worker import lipsync
    fps = float(fps)
    words = lipsync.estimate_words(script, float(seconds or len(openness) / fps), lines)
    out = lipsync.lip_closure(openness, fps, lipsync.bilabial_onsets(words),
                              tolerance_frames=int(c.get("estimated_tolerance_frames", c.get("tolerance_frames", 2))),
                              closed_max=float(c.get("closed_max_openness", 0.05)),
                              relative_to_p90=float(c.get("relative_to_p90", 0)),
                              offset_frames=int(sync.get("offset_frames") or 0),
                              min_bilabials=int(c.get("min_bilabials", 1)))
    out["timing"] = "lines" if lines else "script"
    return out


# --- background warp -------------------------------------------------------------------------------------------

def _edges(data: bytes, width: int):
    import io

    import numpy as np
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("L")
        if im.width > width:
            im = im.resize((width, max(1, round(im.height * width / im.width))))
        g = np.asarray(im, dtype=np.float64) / 255.0
    gx, gy = np.zeros_like(g), np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    return np.hypot(gx, gy)


def background_flicker(frames: list[bytes], policy: Policy | None, *, box: tuple | None = None) -> dict | None:
    """How much the background's edges change between consecutive keyframes, outside the subject box (policy
    `background.exclude_box`, or a face box when one is known): per pair, sum |a - b| / sum (a + b) over the
    background. None when off, fewer than two readable frames, or the frames differ in size."""
    if not enabled(policy, "background") or len(frames) < 2:
        return None
    import numpy as np
    c = check(policy, "background")
    width = int(c.get("width", 160))
    try:
        maps = [_edges(b, width) for b in frames]
    except Exception:  # noqa: BLE001 - not images (or unreadable): no measure
        return None
    if len({m.shape for m in maps}) != 1:
        return None
    h, w = maps[0].shape
    x0, y0, x1, y1 = box or c.get("exclude_box") or (0.0, 0.0, 0.0, 0.0)
    mask = np.ones((h, w), dtype=bool)
    mask[int(y0 * h):int(round(y1 * h)), int(x0 * w):int(round(x1 * w))] = False
    if mask.sum() < 16:
        return None
    pairs = []
    for a, b in zip(maps, maps[1:]):
        a, b = a[mask], b[mask]
        total = float((a + b).sum())
        pairs.append(round(float(np.abs(a - b).sum()) / total, 3) if total > 1e-9 else 0.0)
    return {"flicker": max(pairs), "mean": round(sum(pairs) / len(pairs), 3), "pairs": pairs}
