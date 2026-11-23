"""The aesthetics library (policy/aesthetics/<file>, named by policy `brain.aesthetics.file`).

What makes a feed aspirational rather than a row of straight-on portraits: a vocabulary of shot types, most of them
without her face (her back at sunset, a ringed hand on a glass, a dinner table from above), the rules that keep a
style consistent while outfits and places change, the photo-dump order, grade families and carousel templates.

The style director (brain/agents/style_director.py) picks a creator's grade and shot families from it; the
showrunner and director plan with it; the prompt compiler writes a faceless shot's subject from it
(direction.faceless: a shot whose face is not `yes` shows no face and is never judged on identity).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import yaml

from ..policy import Policy


def _find_root() -> Path:
    for candidate in (Path(__file__).resolve().parents[2] / "policy" / "aesthetics",
                      Path.cwd() / "policy" / "aesthetics", Path("/app/policy/aesthetics")):
        if candidate.exists():
            return candidate
    return Path.cwd() / "policy" / "aesthetics"


ROOT = _find_root()
DEFAULT_FILE = "aesthetics-v1.yaml"
FACE_SUBJECTS = ("", "face")   # subjects that show her face: judged on identity, drawn with her appearance


@lru_cache(maxsize=8)
def _load(path: str, digest: str) -> dict:   # digest keys the cache on content, so an edited file is re-read
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def library(policy: Policy | None = None) -> dict:
    name = (policy.get("brain", "aesthetics", "file", default=DEFAULT_FILE) if policy else DEFAULT_FILE) or DEFAULT_FILE
    path = ROOT / name
    if not path.exists():
        return {"version": None, "shots": {}, "subjects": {}, "grades": {}, "rules": {}, "templates": {}}
    lib = _load(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    shots = {k: {**v, "face": _face(v.get("face"))} for k, v in (lib.get("shots") or {}).items()}
    return {"version": lib.get("version") or name, "shots": shots, "subjects": lib.get("subjects") or {},
            "grades": lib.get("grades") or {}, "rules": lib.get("rules") or {}, "templates": lib.get("templates") or {},
            "photo_dump": lib.get("photo_dump") or []}


def _face(v) -> str:
    """YAML 1.1 reads a bare yes/no as a boolean; the library means the words."""
    if v is True:
        return "yes"
    if v is False:
        return "no"
    return str(v or "no")


def shot(policy: Policy | None, name: str) -> dict | None:
    return library(policy)["shots"].get((name or "").strip())


def subject_of(policy: Policy | None, name: str) -> str:
    """What of her a shot type shows: face (or "" for an unknown or empty type), hands, back, figure, ..."""
    s = shot(policy, name)
    return str(s.get("subject") or "face") if s else ""


def grade_words(policy: Policy | None, grade: str) -> str:
    return str(library(policy)["grades"].get(grade or "", "") or "")


def grades(policy: Policy | None = None) -> list[str]:
    return list(library(policy)["grades"])


def names(policy: Policy | None = None) -> list[str]:
    return list(library(policy)["shots"])


def subject_text(policy: Policy | None, subject: str, hands: str) -> str:
    """The prompt's subject for a shot without her face ({hands}: her skin and signature details)."""
    template = str(library(policy)["subjects"].get(subject, "") or "")
    return template.replace("{hands}", hands).strip(" ,") if template else ""


def catalogue(policy: Policy | None, families: list[str] | None = None) -> dict:
    """The shot types for a planning prompt: {name: "face | framing | use"}, limited to a creator's families
    (plus the portraits, which every creator has) when given."""
    shots = library(policy)["shots"]
    keep = set(families or []) | {k for k, v in shots.items() if v.get("category") == "portrait"} if families else set(shots)
    return {k: f"face {v.get('face')}; {v.get('framing')}; {v.get('use')}" for k, v in shots.items() if k in keep}


def brief(policy: Policy | None = None) -> dict:
    """Rules, the photo-dump order and the carousel templates, for planning prompts."""
    lib = library(policy)
    return {"rules": lib["rules"], "photo_dump": lib["photo_dump"], "carousel_templates": lib["templates"]}
