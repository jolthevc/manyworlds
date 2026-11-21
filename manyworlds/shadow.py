"""How a candidate model compares with a role's model on the same calls (rows of `llm_shadow`).

Per candidate: how often it produced a valid answer, what it cost against the role's model, and how close its
answers were, field by field: numbers by mean absolute difference (and the share within `tolerance`), booleans
and short labels by agreement. Free text is not compared. The owner switches a role (policy `llm.roles`) when
the numbers say the candidate does the job; the evaluator's agreement with owner labels (M2 calibration) is the
stricter test before a scoring role moves.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Any

LABEL_MAX_CHARS = 40    # strings longer than this are free text, not labels (not compared)


def _leaves(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out.update(_leaves(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(obj, list):
        out = {}
        for i, v in enumerate(obj):
            out.update(_leaves(v, f"{prefix}[{i}]"))
        return out
    return {prefix: obj}


def compare(a: dict, b: dict, tolerance: float = 1.0) -> dict[str, dict]:
    """Field-by-field comparison of two answers to the same call."""
    la, lb = _leaves(a), _leaves(b)
    out: dict[str, dict] = {}
    for path, va in la.items():
        if path not in lb:
            out[path] = {"kind": "missing"}
            continue
        vb = lb[path]
        if isinstance(va, bool) or isinstance(vb, bool):
            out[path] = {"kind": "label", "agree": va == vb}
        elif isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = abs(float(va) - float(vb))
            out[path] = {"kind": "number", "diff": d, "within": d <= tolerance}
        elif isinstance(va, str) and isinstance(vb, str) and max(len(va), len(vb)) <= LABEL_MAX_CHARS:
            out[path] = {"kind": "label", "agree": va.strip().lower() == vb.strip().lower()}
    return out


def _field_key(path: str) -> str:
    """`scores[3].value` and `scores[0].value` are the same field."""
    import re
    return re.sub(r"\[\d+\]", "[]", path)


def report(conn, role: str, *, days: float = 30, tolerance: float = 1.0, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    rows = conn.execute("SELECT * FROM llm_shadow WHERE role = %s AND created_at >= %s ORDER BY created_at",
                        (role, now - timedelta(days=days))).fetchall()
    by_candidate: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_candidate.setdefault((r["primary_model"], r["candidate_model"]), []).append(r)
    out = []
    for (primary, candidate), rs in by_candidate.items():
        ok = [r for r in rs if r["candidate_output"] is not None]
        fields: dict[str, dict] = {}
        for r in ok:
            for path, c in compare(r["primary_output"], r["candidate_output"], tolerance).items():
                f = fields.setdefault(_field_key(path), {"kind": c["kind"], "n": 0, "diffs": [], "agree": [],
                                                         "within": [], "missing": 0})
                f["n"] += 1
                if c["kind"] == "number":
                    f["diffs"].append(c["diff"])
                    f["within"].append(c["within"])
                elif c["kind"] == "label":
                    f["agree"].append(c["agree"])
                else:
                    f["missing"] += 1
        summary = {}
        for path, f in sorted(fields.items()):
            s: dict[str, Any] = {"n": f["n"]}
            if f["diffs"]:
                s.update(mean_abs_diff=round(mean(f["diffs"]), 3), within_tolerance=round(mean(f["within"]), 3))
            if f["agree"]:
                s["agreement"] = round(mean(f["agree"]), 3)
            if f["missing"]:
                s["missing"] = f["missing"]
            summary[path] = s
        p_usd = sum(float(r["primary_usd"]) for r in rs)
        c_usd = sum(float(r["candidate_usd"]) for r in rs)
        out.append({"role": role, "primary": primary, "candidate": candidate, "calls": len(rs),
                    "valid_share": round(len(ok) / len(rs), 3),
                    "errors": sorted({(r["candidate_error"] or "")[:120] for r in rs if r["candidate_error"]})[:5],
                    "primary_usd_per_call": round(p_usd / len(rs), 6),
                    "candidate_usd_per_call": round(c_usd / len(rs), 6),
                    "cost_ratio": round(c_usd / p_usd, 3) if p_usd else None,
                    "fields": summary})
    return out
