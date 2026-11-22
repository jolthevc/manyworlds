"""What image and video review share: the stages, the references, how rows land, the loop."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Callable

from ..storage import ObjectStore


def stages(cfg: dict, role: str | None = None) -> list[tuple[str, str, str]]:
    """(role, effort, name) per stage: a cheap screen on everything, then a stronger confirm on
    what the screen passed. Passing `role` runs that one model alone (calibration, dry runs)."""
    out = [(role or cfg.get("role", "judge"), cfg.get("effort", "low"), "screen")]
    if cfg.get("confirm_role") and not role:
        out.append((cfg["confirm_role"], cfg.get("confirm_effort", "low"), "confirm"))
    return out


def direction_cfg(policy) -> dict:
    return (policy.get("review", "direction", default={}) or {}) if policy is not None else {}


def direction_passes(score: int | None, policy) -> bool:
    """The `direction` gate: a score below review.direction.min_score rejects. A judge that leaves it out passes,
    as for `scene` (the output is still judged on identity, realism and safety)."""
    return score is None or int(score) >= int(direction_cfg(policy).get("min_score", 6))


def direction_brief(shot: dict, policy, *, still: bool) -> str:
    """What the judge scores `direction` against: the shot's one-line summary, its beats (a photo is one moment:
    the first beat only) and, for video, the camera move and the performance; then the rubric's anchors
    (policy/rubrics/direction-v<n>.yaml), so scores mean the same thing on every judge."""
    from .. import direction
    from ..brain import rubrics
    s = direction.ShotSheet.model_validate(shot)
    beats = s.blocking[:1] if still else sorted(s.blocking, key=lambda b: b.at_s)
    looks = {"lens": "into the lens", "off": "off-camera", "prop": "at the object in hand", "screen": "at a screen"}

    def beat(b) -> str:
        text = ", ".join(x for x in (b.action.strip(), f"looking {looks[b.eyeline]}" if b.eyeline else "",
                                     f"hands {b.hands.strip()}" if b.hands.strip() else "") if x)
        return text if still or not text else f"at {b.at_s:g} s: {text}"

    lines = [f"Shot: {direction.summary(s) or 'no summary'}."]
    beat_text = "; then ".join(t for t in (beat(b) for b in beats) if t)
    if beat_text:
        lines.append(f"Beats: {beat_text}.")
    if not still:
        move = direction.camera_move(s, policy)
        perf = ", ".join(x for x in (f"camera {move.replace('_', ' ')}" if move else "",
                                     f"{s.performance.energy} energy" if s.performance.energy else "",
                                     f"mood {s.performance.emotion_arc}" if s.performance.emotion_arc.strip() else "",
                                     s.performance.body_language.strip()) if x)
        if perf:
            lines.append(f"Camera and performance: {perf}.")
    rubric = rubrics.load(str(direction_cfg(policy).get("rubric", "direction")))
    return " ".join(lines) + "\n" + rubric.as_text()


def reference_images(creator: dict, store: ObjectStore, workdir: Path, n: int) -> list[bytes]:
    refs = (creator["anchors"] or {}).get("reference_images", [])[:n]
    if not refs:
        raise ValueError(f"creator {creator['id']} has no reference images to review against")
    return [store.get_file(r["uri"], Path(workdir) / f"ref_{i}.png").read_bytes() for i, r in enumerate(refs)]


@contextlib.contextmanager
def charged_to(llm, job_id):
    """Charge the judge's calls to the job under review, so review counts in that output's cost
    (ledger.cost_per_output). No agent: review is production cost, not the Brain's."""
    tag = getattr(llm, "tagged", None)
    with (tag(None, job_id) if tag else contextlib.nullcontext()):
        yield


def write_rows(conn, rows: list[tuple]) -> None:
    """All of a job's review rows at once, after every stage: anyone reading reviews (production
    choosing what to use) must never see a screen without its confirm."""
    if not rows:
        return
    with conn.transaction():
        for row in rows:
            conn.execute("INSERT INTO reviews (asset_id, gate, passed, score, details) VALUES (%s, %s, %s, %s, %s)", row)


def run_jobs(job_ids: list, review: Callable[[object, Path], list[dict]], workdir: Path, prefix: str = "") -> dict:
    """Review each job in its own folder; one bad job never stops the others."""
    accepted = rejected = 0
    errors = []
    for job_id in job_ids:
        jd = Path(workdir) / f"{prefix}{job_id}"
        jd.mkdir(parents=True, exist_ok=True)
        try:
            for r in review(job_id, jd):
                accepted += r["accepted"]
                rejected += not r["accepted"]
        except Exception as e:  # noqa: BLE001
            errors.append({"job": str(job_id), "error": str(e)[:300]})
    return {"accepted": accepted, "rejected": rejected, "errors": errors, "jobs": len(job_ids)}
