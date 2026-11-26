"""What each job did with its weights: reused from the volume or cache, or downloaded (owner 2026-11-25: prove reuse
across machines, not only that a volume mounts). comfy.ensure_weights notes each file; the runner drains the notes
into the job's result (`weights`), so `mw weights-reuse` can read them per machine."""

from __future__ import annotations

_notes: list[dict] = []


def note(folder: str, name: str, action: str, gb: float | None = None, seconds: float | None = None) -> None:
    _notes.append({"file": f"{folder}/{name}", "action": action,
                   **({"gb": round(gb, 2)} if gb is not None else {}),
                   **({"seconds": round(seconds, 1)} if seconds is not None else {})})


def drain() -> dict:
    """{reused, downloaded, downloaded_gb, files: [...]} since the last drain; {} when nothing was checked."""
    global _notes
    notes, _notes = _notes, []
    if not notes:
        return {}
    down = [n for n in notes if n["action"] == "downloaded"]
    return {"reused": sum(n["action"] == "reused" for n in notes), "downloaded": len(down),
            "downloaded_gb": round(sum(n.get("gb", 0) for n in down), 2), "files": notes[:40]}
