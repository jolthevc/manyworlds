"""Creative rubrics as versioned files (policy/rubrics/<name>-v<n>.yaml).

Each dimension has a definition, what it is not, and anchors at 1/3/5/7/9. A file's content hash
is registered in `rubric_versions`; editing a rubric without bumping its version is refused, so
scores recorded under one version always mean the same thing. Changing a rubric means a new file
with a higher version, and the reference set is re-scored (manyworlds.brain.evaluation.drift).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path

import yaml
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field


def _find_root() -> Path:
    """policy/rubrics next to the source tree (dev, tests), or where images copy policy/ (/app)."""
    for candidate in (Path(__file__).resolve().parents[2] / "policy" / "rubrics",
                      Path.cwd() / "policy" / "rubrics", Path("/app/policy/rubrics")):
        if candidate.exists():
            return candidate
    return Path.cwd() / "policy" / "rubrics"


ROOT = _find_root()
log = logging.getLogger("manyworlds.brain.rubrics")


class RubricChanged(RuntimeError):
    pass


class Dimension(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    key: str
    name: str
    definition: str
    not_: str = Field(default="", alias="not")
    anchors: dict[int, str]


class Rubric(BaseModel):
    rubric: str
    version: int
    applies_to: str
    guidance: str = ""
    dimensions: list[Dimension]
    file_hash: str = ""

    def as_text(self) -> str:
        """The rubric as the evaluator reads it."""
        lines = [f"RUBRIC {self.rubric} v{self.version}. {self.applies_to}", self.guidance, ""]
        for d in self.dimensions:
            lines.append(f"[{d.key}] {d.name}: {d.definition}" + (f" Not: {d.not_}" if d.not_ else ""))
            lines += [f"  {k}: {v}" for k, v in sorted(d.anchors.items())]
        return "\n".join(lines)


def policy_root() -> Path:
    """Where rubrics load from: the `rubrics/` beside the policy file MW_POLICY names, when it holds any
    rubric, so a policy kept outside the repo carries its own; otherwise ROOT."""
    path = os.environ.get("MW_POLICY")
    if path:
        beside = Path(path).resolve().parent / "rubrics"
        if beside.is_dir() and any(beside.glob("*-v*.yaml")):
            return beside
    return ROOT


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else policy_root()


_VERSION = re.compile(r"-v(\d+)\.yaml$")


def load(name: str, *, root: Path | None = None) -> Rubric:
    """The highest version of rubric `name`. A file whose name carries no numeric version (a stray draft,
    `idea-vdraft.yaml`) is skipped with a warning; one whose `version:` disagrees with its name is refused."""
    base = _root(root)
    files = []
    for p in base.glob(f"{name}-v*.yaml"):
        m = _VERSION.search(p.name)
        if m is None or p.name != f"{name}-v{m.group(1)}.yaml":
            log.warning("skipping %s: not a versioned rubric file (%s-v<n>.yaml)", p, name)
            continue
        files.append((int(m.group(1)), p))
    if not files:
        raise FileNotFoundError(f"no rubric {name!r} in {base}")
    version, path = max(files)
    raw = path.read_bytes()
    r = Rubric(**yaml.safe_load(raw), file_hash=hashlib.sha256(raw).hexdigest())
    if r.rubric != name:
        raise ValueError(f"{path.name} declares rubric {r.rubric!r}")
    if r.version != version:
        raise ValueError(f"{path.name} declares version {r.version}; the file name says v{version}")
    return r


def register(conn, r: Rubric) -> int:
    row = conn.execute("SELECT file_hash FROM rubric_versions WHERE rubric = %s AND version = %s",
                       (r.rubric, r.version)).fetchone()
    if row:
        if row["file_hash"] != r.file_hash:
            raise RubricChanged(f"rubric {r.rubric} v{r.version} was edited after scores were recorded under it; "
                                f"save the change as v{r.version + 1}")
        return r.version
    conn.execute("INSERT INTO rubric_versions (rubric, version, file_hash, body) VALUES (%s, %s, %s, %s) "
                 "ON CONFLICT DO NOTHING",
                 (r.rubric, r.version, r.file_hash, Jsonb(r.model_dump(by_alias=True, exclude={"file_hash"}))))
    return r.version
