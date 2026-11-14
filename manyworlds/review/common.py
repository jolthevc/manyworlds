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
    return {"accepted": accepted, "rejected": rejected, "errors": errors}
