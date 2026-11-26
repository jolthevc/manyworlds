"""Settings the Brain changes at runtime, bounded by its authority (policy `brain.authority`).

The policy file is the human-edited base. Overrides are rows in `policy_overrides`; the latest
version of each path wins, and the effective policy's version is `<file hash>+o<n>` with n the
latest override version, so every decision records exactly what it ran under.

`propose` applies a change when its path matches an `autonomous` entry (`*` = one path
segment) and the value is inside that entry's inclusive `min`/`max` (numbers) or `allowed`
list. Everything else becomes a pending approval that the owner decides; approving applies the
change in the same transaction, so an approval is never recorded without its change or vice versa.

Only sections listed in `brain.authority.observed_sections` can be overridden at all: those are
read through the effective policy. The producer, reviewer and autoscaler still read the policy
file, so a change to `production.*`, `limits.*` and the like is refused outright (not_observed)
rather than recorded and silently ignored.

`revert` (`mw policy revert <override-id>`) rolls the latest override of a path back: the value before it is
proposed again as a new override, under the same authority rules, with the reason and author recorded.
"""

from __future__ import annotations

import copy
import dataclasses
import fnmatch
import uuid

from psycopg.types.json import Jsonb

from .. import policy_schema
from ..policy import Policy

NEVER = ("limits.", "budget.master")


def _set_path(body: dict, path: str, value) -> None:
    node = body
    keys = path.split(".")
    for k in keys[:-1]:
        nxt = node.get(k)
        if not isinstance(nxt, dict):
            nxt = node[k] = {}
        node = nxt
    node[keys[-1]] = value


def latest_version(conn) -> int:
    return int(conn.execute("SELECT coalesce(max(version), 0) AS v FROM policy_overrides").fetchone()["v"])


def effective_policy(conn, policy: Policy) -> Policy:
    """The file policy with the latest override of each path, versioned by the latest override. Settings and version
    come from one statement (one snapshot): read in two, an override applied in between would give a decision a
    version that doesn't match the settings it ran under. The newest override is the latest row of its own path, so
    the max over the latest rows is the max over all of them."""
    body = copy.deepcopy(policy.body)
    rows = conn.execute("SELECT path, value, max(version) OVER () AS latest FROM "
                        "(SELECT DISTINCT ON (path) path, value, version FROM policy_overrides "
                        "ORDER BY path, version DESC) d").fetchall()
    for row in rows:
        _set_path(body, row["path"], row["value"])
    latest = int(rows[0]["latest"]) if rows else 0
    return dataclasses.replace(policy, body=body, version=f"{policy.version}+o{latest}")


def _matches(pattern: str, path: str) -> bool:
    return len(pattern.split(".")) == len(path.split(".")) and all(
        fnmatch.fnmatchcase(seg, pat) for seg, pat in zip(path.split("."), pattern.split(".")))


def within_authority(policy: Policy, path: str, value) -> bool:
    if path.startswith(NEVER):
        return False
    for entry in policy.get("brain", "authority", "autonomous", default=[]) or []:
        if not _matches(entry["path"], path):
            continue
        if "allowed" in entry:
            return value in entry["allowed"]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            lo, hi = entry.get("min"), entry.get("max")
            return (lo is None or value >= lo) and (hi is None or value <= hi)
        return False
    return False


def _apply(conn, path: str, value, *, author: str, reason: str, decision_id=None) -> int:
    version = latest_version(conn) + 1
    conn.execute("INSERT INTO policy_overrides (path, value, version, author, reason, decision_id) "
                 "VALUES (%s, %s, %s, %s, %s, %s)", (path, Jsonb(value), version, author, reason, decision_id))
    return version


def schema_problems(policy: Policy, path: str, value) -> list[str]:
    """What setting `path` to `value` would break in a section the policy schema checks (manyworlds.policy_schema):
    problems the policy had already are not counted. Empty for paths outside those sections."""
    if not policy_schema.covers(path):
        return []
    body = copy.deepcopy(policy.body)
    _set_path(body, path, value)
    before = set(policy_schema.errors(policy.body))
    return [p for p in policy_schema.errors(body) if p not in before]


def propose(conn, policy: Policy, path: str, value, *, author: str, reason: str, decision_id=None) -> dict:
    observed = policy.get("brain", "authority", "observed_sections", default=[]) or []
    paths = policy.get("brain", "authority", "observed_paths", default=[]) or []
    if path.split(".")[0] not in observed and not any(_matches(p, path) for p in paths):
        return {"applied": False, "not_observed": True,
                "reason": f"{path.split('.')[0]}.* is not read by any service through the effective policy yet"}
    problems = schema_problems(policy, path, value)
    if problems:   # a typo'd key or a wrong type in a schema-checked section is refused, never stored
        return {"applied": False, "invalid": True, "reason": "; ".join(problems)}
    if within_authority(policy, path, value):
        return {"applied": True, "version": _apply(conn, path, value, author=author, reason=reason,
                                                   decision_id=decision_id)}
    approval: uuid.UUID = conn.execute(
        "INSERT INTO approvals (kind, subject, reason, requested_by) VALUES ('policy_override', %s, %s, %s) "
        "RETURNING id", (Jsonb({"path": path, "value": value, "decision_id": str(decision_id) if decision_id else None}),
                         reason, author)).fetchone()["id"]
    return {"applied": False, "approval_id": approval}


def decide(conn, approval_id, *, approved: bool, by: str) -> None:
    with conn.transaction():
        _decide(conn, approval_id, approved=approved, by=by)


def _decide(conn, approval_id, *, approved: bool, by: str) -> None:
    row = conn.execute("UPDATE approvals SET status = %s, decided_by = %s, decided_at = now() "
                       "WHERE id = %s AND status = 'pending' RETURNING kind, subject, reason, requested_by",
                       ("approved" if approved else "rejected", by, approval_id)).fetchone()
    if row is None:
        raise ValueError(f"no pending approval {approval_id}")
    if approved and row["kind"] == "stage_move":   # e.g. a retirement the owner confirmed
        from . import portfolio
        s = row["subject"]
        portfolio.apply_stage_move(conn, s["creator_id"], from_stage=s.get("from"), to=s["to"],
                                   reason=f"{s.get('reason', '')} (approved by {by})", downward={"paused", "retired"})
    if approved and row["kind"] == "policy_override":
        s = row["subject"]
        _apply(conn, s["path"], s["value"], author=row["requested_by"],
               reason=f"{row['reason']} (approved by {by})", decision_id=s.get("decision_id"))


def _get_path(body: dict, path: str):
    node = body
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            raise KeyError(path)
        node = node[k]
    return node


def revert(conn, policy: Policy, override_id, *, author: str, reason: str) -> dict:
    """Roll one override back (`mw policy revert <override-id>`): the value its path had before it (the previous
    override of that path, else the policy file's) is proposed again as a new override, with the reason and author
    recorded, so history only grows and the revert can itself be reverted. It goes through `propose`, so the same
    authority rules apply: outside brain.authority it waits for the owner as an approval. Only the latest override
    of its path can be reverted (a later one would be undone silently); a path that already holds the restored value
    changes nothing."""
    row = conn.execute("SELECT id, path, value, version FROM policy_overrides WHERE id = %s",
                       (int(override_id),)).fetchone()
    if row is None:
        raise ValueError(f"no override {override_id}")
    prev = conn.execute("SELECT value, version FROM policy_overrides WHERE path = %s AND version < %s "
                        "ORDER BY version DESC LIMIT 1", (row["path"], row["version"])).fetchone()
    if prev is not None:
        value, source = prev["value"], f"override v{prev['version']}"
    else:
        try:
            value, source = _get_path(policy.body, row["path"]), "the policy file"
        except KeyError:
            raise ValueError(f"{row['path']} has no value before override {override_id} to restore") from None
    latest = conn.execute("SELECT id, value, version FROM policy_overrides WHERE path = %s ORDER BY version DESC "
                          "LIMIT 1", (row["path"],)).fetchone()
    if latest["version"] != row["version"]:   # rolling back past a later change would undo it silently
        raise ValueError(f"override {latest['id']} (v{latest['version']}) changed {row['path']} after it: revert that "
                         f"one first")
    current = latest["value"]
    out = {"reverts": int(row["id"]), "reverted_version": int(row["version"]), "path": row["path"], "value": value,
           "restored_from": source}
    if current == value:
        return {**out, "applied": False, "unchanged": True}
    why = f"revert of override {row['id']} (v{row['version']}) to {source}: {reason}"
    return {**out, **propose(conn, policy, row["path"], value, author=author, reason=why)}
