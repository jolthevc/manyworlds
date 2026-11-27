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


def _read(name: str) -> dict | None:
    path = ROOT / name
    if not path.exists():
        return None
    return _load(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


def library(policy: Policy | None = None) -> dict:
    """{version, default_communicates, archetypes: {name: playbook}}; empty when the file is missing. The files in
    `brain.archetypes.extra` (e.g. musician-v1.yaml) add their archetypes after the main file's; a name the main
    file already has keeps the main file's playbook."""
    name = (policy.get("brain", "archetypes", "file", default=DEFAULT_FILE) if policy else DEFAULT_FILE) or DEFAULT_FILE
    lib = _read(name)
    if lib is None:
        return {"version": None, "archetypes": {}}
    archetypes = dict(lib.get("archetypes") or {})
    versions = [str(lib.get("version") or name)]
    for extra in (policy.get("brain", "archetypes", "extra", default=[]) if policy else []) or []:
        more = _read(str(extra))
        if more is None:
            continue
        versions.append(str(more.get("version") or extra))
        for k, v in (more.get("archetypes") or {}).items():
            archetypes.setdefault(k, v)
    return {"version": "+".join(versions), "default_communicates": lib.get("default_communicates", ""),
            "archetypes": archetypes}


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


# --- audience propositions (owner 2026-11-26) -----------------------------------------------------------------------
# WHY someone follows (policy/archetypes/propositions-v1.yaml), kept apart from the vertical playbooks above (WHAT she
# posts about and how it is shaped). A creator is a weighted mix of propositions; a series names the proposition it
# serves and the job it does (attention, follow, return, affinity).
PROPOSITIONS_FILE = "propositions-v1.yaml"
PROPOSITION_FIELDS = ("follow_reason", "attention", "follow", "return", "affinity", "outcome_focus", "pitfalls")


def propositions(policy: Policy | None = None) -> dict:
    """{version, jobs, propositions: {name: {...}}}; empty when the file is missing."""
    name = (policy.get("brain", "archetypes", "propositions_file", default=PROPOSITIONS_FILE) if policy
            else PROPOSITIONS_FILE) or PROPOSITIONS_FILE
    lib = _read(str(name)) or {}
    return {"version": lib.get("version"), "jobs": list(lib.get("jobs") or ["attention", "follow", "return", "affinity"]),
            "propositions": dict(lib.get("propositions") or {})}


def proposition_names(policy: Policy | None = None) -> list[str]:
    return list(propositions(policy)["propositions"])


def jobs(policy: Policy | None = None) -> list[str]:
    return propositions(policy)["jobs"]


def proposition_catalogue(policy: Policy | None = None) -> dict:
    """{name: follow_reason}: what casting and the talent manager choose a creator's mix from."""
    return {n: p.get("follow_reason", "") for n, p in propositions(policy)["propositions"].items()}


def proposition_playbooks(policy: Policy | None, mix: list[dict] | None) -> list[dict]:
    """The planning view of a creator's propositions ([{name, share}], largest first), unknown names dropped."""
    lib = propositions(policy)["propositions"]
    out = []
    for m in sorted(mix or [], key=lambda x: -float(x.get("share") or 0)):
        p = lib.get(str(m.get("name")))
        if p:
            out.append({"proposition": m["name"], "share": m.get("share"),
                        **{k: p[k] for k in PROPOSITION_FIELDS if k in p}})
    return out


def clean_mix(policy: Policy | None, mix) -> list[dict]:
    """A proposition mix with unknown names kept only as `other:<words>` candidates (research harvests them), shares
    normalised to sum to 1. An empty or unusable mix is []."""
    known = set(proposition_names(policy))
    out = []
    for m in mix or []:
        m = m if isinstance(m, dict) else {"name": getattr(m, "name", ""), "share": getattr(m, "share", 0)}
        name = str(m.get("name") or "").strip().lower().replace(" ", "_")
        try:
            share = max(0.0, float(m.get("share") or 0))
        except (TypeError, ValueError):
            share = 0.0
        if not name or share <= 0:
            continue
        out.append({"name": name if name in known or name.startswith("other:") else f"other:{name}", "share": share})
    total = sum(m["share"] for m in out)
    return [{**m, "share": round(m["share"] / total, 3)} for m in out] if total else []
