"""Every literal policy path the code reads exists in policy/default.yaml.

Policy.get falls back to its `default=` when a key is missing, so a misspelt path in code (or a key never added to the
YAML) silently runs on the code's number. This scans manyworlds/ with the ast module for
`policy.get("a", "b", ...)`, `pol.get(...)`, `self.policy.get(...)` and `ctx.policy.get(...)` and checks the leading
string-constant segments of each call against the shipped file. A call whose later segments are not constants
(`policy.get("job_kinds", kind, "family")`) is checked up to the first non-constant segment.

Reads of a dict taken out of the policy (`cfg = policy.get("review", "image", default={})`, then `cfg.get("x", 10)`)
are not followed: the schema (manyworlds.policy_schema) covers typos inside the hot sections.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from manyworlds import policy as policy_mod

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "manyworlds"
POLICY_NAMES = {"policy", "pol"}          # policy.get(...), pol.get(...)
POLICY_OWNERS = {"self", "ctx"}           # self.policy.get(...), ctx.policy.get(...)

# Paths only the breadth test's overlay (policy/test_overlay.yaml) sets; absent from the shipped file on purpose.
# Each must exist in the overlay (checked below), so they can't hide a typo.
OVERLAY_ONLY: dict[tuple[str, ...], str] = {
    ("autoscale", "merge_families"): "absent = each worker family on its own machines; the breadth overlay merges some",
    ("breadth",): "the breadth driver's own section lives in the test overlay only",
}

# Paths that are intentionally optional in the shipped file (the code's default is the intended behaviour).
OPTIONAL: dict[tuple[str, ...], str] = {}


def _is_policy(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id in POLICY_NAMES
    return (isinstance(node, ast.Attribute) and node.attr == "policy"
            and isinstance(node.value, ast.Name) and node.value.id in POLICY_OWNERS)


def policy_reads() -> list[tuple[str, int, tuple[str, ...]]]:
    """(file, line, literal path) for every policy read with at least one leading string-constant segment."""
    reads = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and _is_policy(node.func.value)):
                continue
            segments = []
            for arg in node.args:
                if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                    break
                segments.append(arg.value)
            if segments:
                reads.append((path.relative_to(ROOT).as_posix(), node.lineno, tuple(segments)))
    return reads


def _exists(body, segments: tuple[str, ...]) -> bool:
    node = body
    for key in segments:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def _shipped() -> dict:
    return yaml.safe_load((ROOT / "policy" / "default.yaml").read_text(encoding="utf-8"))


def _overlay() -> dict:
    return yaml.safe_load((ROOT / "policy" / "test_overlay.yaml").read_text(encoding="utf-8")) or {}


def test_scanner_finds_policy_reads():
    reads = policy_reads()
    assert len(reads) > 200, f"only {len(reads)} policy reads found: the scanner no longer matches the code"
    assert any(r[2][:2] == ("queue", "lease_seconds") for r in reads)


def test_every_literal_policy_path_exists_in_the_shipped_file():
    body = _shipped()
    allowed = set(OVERLAY_ONLY) | set(OPTIONAL)
    missing = [f"{file}:{line}  {'.'.join(segs)}" for file, line, segs in policy_reads()
               if not _exists(body, segs) and segs not in allowed]
    assert not missing, ("policy paths read in code but missing from policy/default.yaml (fix the code's spelling, "
                         "or add the key to the YAML with the code's default):\n  " + "\n  ".join(missing))


def test_overlay_only_paths_exist_in_the_overlay():
    merged = policy_mod.deep_merge(_shipped(), _overlay())
    for segs, reason in OVERLAY_ONLY.items():
        assert _exists(merged, segs), f"{'.'.join(segs)} is allowlisted as overlay-only ({reason}) but not set there"
        assert not _exists(_shipped(), segs), f"{'.'.join(segs)} is in default.yaml now: drop it from OVERLAY_ONLY"


def test_allowlists_have_no_stale_entries():
    read = {segs for _, _, segs in policy_reads()}
    stale = [".".join(s) for s in (set(OVERLAY_ONLY) | set(OPTIONAL)) if s not in read]
    assert not stale, f"allowlisted paths no code reads any more: {stale}"
