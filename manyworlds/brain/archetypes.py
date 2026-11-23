"""The archetype library (policy/archetypes/<file>, named by policy `brain.archetypes.file`).

An archetype is a playbook for a kind of creator: what the audience comes for, how this kind of creator
communicates, the mix of content shapes, the capabilities that carry it, series patterns, hooks, edit grammar,
success signals and pitfalls. The talent manager builds a creator's dossier from it (manyworlds.creator.dossier),
the showrunner plans from it, the director reads `communicates`. One library, read by all of them.

A creator names a primary archetype (and optionally a secondary) in `profile.archetypes`; a creator without that
is matched by words in its archetype, niche and one-liner, as the director always did.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import yaml

from ..policy import Policy


def _find_root() -> Path:
    for candidate in (Path(__file__).resolve().parents[2] / "policy" / "archetypes",
                      Path.cwd() / "policy" / "archetypes", Path("/app/policy/archetypes")):
        if candidate.exists():
            return candidate
    return Path.cwd() / "policy" / "archetypes"


ROOT = _find_root()
DEFAULT_FILE = "archetypes-v1.yaml"
# The fields a planning prompt gets (the rest, e.g. words, is for matching).
PLAYBOOK_FIELDS = ("summary", "audience_promise", "communicates", "mix", "talk_share", "lean_on", "avoid",
                   "series_patterns", "hooks", "edit_grammar", "success_signals", "pitfalls")


@lru_cache(maxsize=8)
def _load(path: str, digest: str) -> dict:   # digest keys the cache on content, so an edited file is re-read
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def library(policy: Policy | None = None) -> dict:
    """{version, default_communicates, archetypes: {name: playbook}}; empty when the file is missing."""
    name = (policy.get("brain", "archetypes", "file", default=DEFAULT_FILE) if policy else DEFAULT_FILE) or DEFAULT_FILE
    path = ROOT / name
    if not path.exists():
        return {"version": None, "archetypes": {}}
    raw = path.read_bytes()
    lib = _load(str(path), hashlib.sha256(raw).hexdigest())
    return {"version": lib.get("version") or name, "default_communicates": lib.get("default_communicates", ""),
            "archetypes": lib.get("archetypes") or {}}


def names(policy: Policy | None = None) -> list[str]:
    return list(library(policy)["archetypes"])


def get(policy: Policy | None, name: str) -> dict | None:
    return library(policy)["archetypes"].get(name)


def _words_in(text: str, words) -> bool:
    low = (text or "").lower()
    return any(re.search(rf"\b{re.escape(str(w).lower())}", low) for w in words or [] if str(w).strip())


def for_creator(policy: Policy | None, who: dict | None) -> list[str]:
    """The creator's archetypes, primary first: `archetypes` named in the profile (known names only), else every
    archetype whose words start a word in its archetype, niche or one-liner (library order)."""
    who = who or {}
    lib = library(policy)["archetypes"]
    named = [a for a in (who.get("archetypes") or []) if a in lib]
    if named:
        return named
    text = " ".join(str(who.get(k) or "") for k in ("archetype", "niche", "one_line"))
    return [n for n, a in lib.items() if _words_in(text, a.get("words"))]


def communicates(policy: Policy | None, who: dict | None) -> list[str]:
    """How this creator communicates, one line per archetype; none matching gives the library default."""
    lib = library(policy)
    found = [str(lib["archetypes"][n].get("communicates") or "") for n in for_creator(policy, who)]
    found = [f for f in found if f]
    return found or ([lib["default_communicates"]] if lib.get("default_communicates") else [])


def playbook(policy: Policy | None, name: str) -> dict:
    """The planning view of one archetype (no matching words)."""
    a = get(policy, name) or {}
    return {"archetype": name, **{k: a[k] for k in PLAYBOOK_FIELDS if k in a}}


def playbooks(policy: Policy | None, who: dict | None) -> list[dict]:
    return [playbook(policy, n) for n in for_creator(policy, who)]


def catalogue(policy: Policy | None) -> dict:
    """{name: summary}: what the talent manager chooses from when it sets a creator's archetypes."""
    return {n: a.get("summary", "") for n, a in library(policy)["archetypes"].items()}


def unknown_capabilities(policy: Policy) -> list[str]:
    """`lean_on`/`avoid`/series `uses` entries that name no capability in policy `capabilities` (a typo or a
    renamed capability: the playbook would steer toward something the director can't find)."""
    from ..direction import capabilities
    known = {f"{g}.{n}" for g, entries in capabilities(policy).items() for n in entries}
    bad = set()
    for name, a in library(policy)["archetypes"].items():
        refs = list(a.get("lean_on") or []) + list(a.get("avoid") or [])
        refs += [u for s in a.get("series_patterns") or [] for u in s.get("uses") or []]
        bad |= {f"{name}:{r}" for r in refs if r not in known}
    return sorted(bad)
