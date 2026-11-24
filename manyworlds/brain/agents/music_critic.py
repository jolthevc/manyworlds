"""The music critic (spec 2026-11-23-music-department.md, section 4): describe, measure, calibrate.

A `brain.music_critic` job ({song_id}) reviews one measured song before any video is made. Its inputs:
  - the song's measured anatomy (manyworlds.song_measure via manyworlds.songs), its brief and lyrics;
  - what good looks like for the artist or genre and where the song sits (music_analysis.bands, advisory);
  - the nearest past songs with their outcomes (music_analysis.nearest) and the open music findings;
  - Audiobox Aesthetics scores and a listener's factual answers, when their adapters have them
    (manyworlds.music_listening; stubs as shipped: the critic then says it heard nothing).
Claude (llm role music.critic.role) reasons over these and returns an Assessment: per-dimension findings, each
citing the evidence ids it rests on; the top fixes (music.critic.max_fixes), each addressed to the role that can make
it (music.critic.fix_roles); a verdict (ship / revise / rethink); and a checkable prediction about the song.

Never a raw score trusted alone: code drops a finding or fix that cites nothing in the evidence list, or only
aesthetics scores (uncalibrated 1-10s), and a fix addressed to a role that does not exist. The decision is recorded
(kind music_critique), the assessment kept (song_assessments), the fixes routed (song_feedback, read by the roles
through manyworlds.songs.feedback_for; with music.critic.commission, also a brain.<role> job for a role whose agent
exists) and the prediction filed (brain.predictions, metrics song_median_reach | song_reach_lift). A rerun of the
same job asks Claude nothing again (agent.remember).
"""

from __future__ import annotations

import json
from typing import Literal

from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from ... import music_listening, songs
from ...worker.runner import NonRetryable
from .. import agent as agent_mod
from .. import music_analysis
from ..agent import AgentContext, commission, register, remember

NAME = "music_critic"

SYSTEM = """You are the music critic of an AI music studio whose artists are openly AI-generated. You review one
song before any video is made, so the team can fix it cheaply. You are given MEASUREMENTS of the song (loudness,
structure, chorus repeats and timing, vocal entry and clarity, spectral balance, lyric repetition and rhyme...),
BANDS (what good looks like for this artist or genre; advisory, never a rule), the NEAREST past songs with how they
did, the open FINDINGS, and, when available, what a LISTENER model heard and AESTHETICS scores (Audiobox, 1-10,
uncalibrated: never a reason on their own).

Describe what the evidence shows, dimension by dimension (hook, structure, lyrics, vocal, production, mix, fit with
the artist). Every finding cites the evidence ids it rests on, exactly as listed under EVIDENCE IDS; a claim you can't
tie to evidence is not a finding. Then give at most {max_fixes} fixes, the most valuable first, each addressed to the
one role that can make it ({roles}), saying the problem, the specific change and the evidence. Verdict: ship (good
enough to release as it is), revise (specific fixes first), rethink (the concept or brief itself is wrong). Then
predict how the song will do, as a checkable claim: metric song_median_reach (median reach of the posts using the
song, a number of accounts) or song_reach_lift (log of that median over the artist's median before, 0 = as usual),
above or below a threshold, with your probability. With little history, predict near the artist's usual and say so.
Never name a real artist or song; never suggest imitating one."""


class DimensionFinding(BaseModel):
    dimension: Literal["hook", "structure", "lyrics", "vocal", "production", "mix", "fit"]
    finding: str = Field(description="what the evidence shows, one or two sentences")
    evidence: list[str] = Field(default_factory=list, description="evidence ids from the list, exactly as given")
    severity: Literal["strength", "minor", "major"] = "minor"


class Fix(BaseModel):
    role: str = Field(description="the one role that can make this change")
    problem: str
    change: str = Field(description="the specific change, concrete enough to act on")
    evidence: list[str] = Field(default_factory=list, description="evidence ids from the list, exactly as given")


class SongPrediction(BaseModel):
    metric: Literal["song_median_reach", "song_reach_lift"]
    comparator: Literal["above", "below"]
    threshold: float
    probability: float = Field(description="0 to 1: how likely the claim holds")
    claim: str = Field(description="the claim in words")


class Assessment(BaseModel):
    dimensions: list[DimensionFinding] = Field(default_factory=list)
    fixes: list[Fix] = Field(default_factory=list)
    verdict: Literal["ship", "revise", "rethink"] = "revise"
    prediction: SongPrediction | None = None
    summary: str = ""


def _cfg(policy) -> dict:
    return policy.get("music", "critic", default={}) or {}


def evidence_ids(context: dict, aesthetics: dict | None, heard: dict | None) -> dict[str, str]:
    """{evidence id: what kind it is} for everything the critic is shown: measured features, lyric features, bands,
    nearest songs, findings, the listener's fields, aesthetics axes."""
    ids: dict[str, str] = {}
    for name in context.get("features") or {}:
        ids[name] = "measured"
    for p in context.get("position") or []:
        ids[f"band.{p['feature']}"] = "band"
    for n in context.get("nearest") or []:
        ids[f"nearest.{n['song_id'][:8]}"] = "nearest"
    for i, _ in enumerate(context.get("findings") or []):
        ids[f"finding.{i + 1}"] = "finding"
    if context["song"].get("lyrics"):
        ids["lyrics"] = "lyrics"
    if context["song"].get("brief"):
        ids["brief"] = "brief"
    for k, v in (heard or {}).items():
        if v not in (None, "", []):
            ids[f"listener.{k}"] = "listener"
    for part, axes in (aesthetics or {}).items():
        for axis in axes or {}:
            ids[f"aesthetics.{part}.{axis}"] = "aesthetics"
    return ids


def _grounded(cited: list[str], ids: dict[str, str]) -> list[str] | None:
    """The cited ids that exist; None when none does, or when only aesthetics scores are cited."""
    known = [c for c in cited if c in ids]
    if not known or all(ids[c] == "aesthetics" for c in known):
        return None
    return known


def check(a: dict, ids: dict[str, str], policy) -> tuple[dict, list[dict]]:
    """The assessment with what it may not keep taken out, and what was taken out (with why)."""
    roles = set(_cfg(policy).get("fix_roles") or [])
    metrics = set((_cfg(policy).get("prediction") or {}).get("metrics") or [])
    dropped: list[dict] = []
    dims = []
    for d in a.get("dimensions") or []:
        ev = _grounded(d.get("evidence") or [], ids)
        if ev is None:
            dropped.append({"dimension": d.get("dimension"), "why": "cites no evidence (or only aesthetics scores)"})
            continue
        dims.append({**d, "evidence": ev})
    fixes = []
    for f in a.get("fixes") or []:
        ev = _grounded(f.get("evidence") or [], ids)
        if f.get("role") not in roles:
            dropped.append({"fix": f.get("problem"), "why": f"no role {f.get('role')!r} (music.critic.fix_roles)"})
        elif ev is None:
            dropped.append({"fix": f.get("problem"), "why": "cites no evidence (or only aesthetics scores)"})
        else:
            fixes.append({**f, "evidence": ev})
    fixes = fixes[:int(_cfg(policy).get("max_fixes", 3))]
    pred = a.get("prediction")
    if pred and (pred.get("metric") not in metrics or not 0 <= float(pred.get("probability", -1)) <= 1):
        dropped.append({"prediction": pred.get("claim"), "why": "unknown metric or a probability outside 0-1"})
        pred = None
    return {"dimensions": dims, "fixes": fixes, "verdict": a.get("verdict") or "revise", "prediction": pred,
            "summary": a.get("summary") or ""}, dropped


def _prompt(context: dict, aesthetics: dict | None, heard: dict | None, ids: dict[str, str], policy) -> str:
    song = context["song"]
    lines = [f"SONG: {song['title']!r} (version {song['version']}, genre {song['genre'] or 'unset'})",
             f"BRIEF: {json.dumps(song['brief'], default=str)}",
             f"LYRICS:\n{song['lyrics'] or '(instrumental)'}",
             f"MEASUREMENTS: {json.dumps(context['features'], default=str)}",
             f"BANDS ({context['bands']['source'] or 'none'}; advisory): {json.dumps(context['position'], default=str)}",
             "NEAREST SONGS: " + json.dumps([{"id": f"nearest.{n['song_id'][:8]}", "title": n["title"],
                                              "distance": n["distance"], "alike_on": n["alike_on"],
                                              "outcome": n["outcome"]} for n in context["nearest"]], default=str),
             "FINDINGS: " + json.dumps([{"id": f"finding.{i + 1}", **f} for i, f in enumerate(context["findings"])],
                                       default=str),
             "LISTENER: " + (json.dumps(heard, default=str) if heard else "nothing heard (no listener yet): say what "
                             "only listening could tell"),
             "AESTHETICS: " + (json.dumps(aesthetics) if aesthetics else "none"),
             f"DATA: {'SIMULATED (sandbox test, not real demand)' if context['mode'] == 'sandbox' else 'real'}",
             f"EVIDENCE IDS: {sorted(ids)}"]
    return "\n".join(lines)


def _aesthetic_features(scores: dict | None) -> dict:
    return {f"aesthetics_{axis.lower()}_{part}": v for part, axes in (scores or {}).items()
            for axis, v in (axes or {}).items() if isinstance(v, (int, float))}


def critique(ctx: AgentContext, song_id, *, aesthetics=None, listener=None) -> dict:
    """Review one song: see the module docstring. Adapters default to the policy's (music.critic.*.adapter)."""
    conn, policy = ctx.conn, ctx.policy
    c = _cfg(policy)
    song = songs.get(conn, song_id)
    if song is None:
        raise NonRetryable(f"no song {song_id}")
    context = music_analysis.song_context(conn, policy, song_id, nearest_k=int(c.get("nearest", 3)))
    if "duration_s" not in context["features"]:
        raise NonRetryable(f"song {song_id} is not measured yet (manyworlds.songs.measure_song first)")
    aesthetics = aesthetics or music_listening.aesthetics(policy, conn=conn)
    listener = listener or music_listening.listener(policy, conn=conn, llm=ctx.llm)
    scores = aesthetics.scores(song)
    if scores:
        songs.add_features(conn, song_id, _aesthetic_features(scores))
    report = listener.describe(song)
    heard = report.model_dump() if report is not None else None
    ids = evidence_ids(context, scores, heard)
    role = str(c.get("role") or NAME)

    def compute():
        system = SYSTEM.replace("{max_fixes}", str(c.get("max_fixes", 3))).replace(
            "{roles}", ", ".join(c.get("fix_roles") or []))
        out = ctx.llm.structured(role, system=system, content=_prompt(context, scores, heard, ids, policy),
                                 schema=Assessment, effort=c.get("effort", "medium"), creator_id=song["creator_id"])
        kept, dropped = check(out.model_dump(), ids, policy)
        return {"kind": "music_critique", "creator_id": song["creator_id"],
                "inputs": {"song_id": str(song_id), "mode": context["mode"], "bands": context["bands"]["source"],
                           "nearest": [n["song_id"] for n in context["nearest"]], "listener": heard is not None,
                           "aesthetics": scores},
                "output": {**kept, "dropped": dropped}, "rationale": kept["summary"], "model": ctx.llm.model_for(role)}
    assessment = remember(ctx, f"assess:{song_id}", compute)
    decision = conn.execute("SELECT id FROM decisions WHERE job_id = %s AND step = %s",
                            (ctx.job.id, f"assess:{song_id}")).fetchone()["id"]
    row = conn.execute("INSERT INTO song_assessments (song_id, decision_id, verdict, assessment) VALUES (%s, %s, %s, %s) "
                       "ON CONFLICT (decision_id) DO NOTHING RETURNING id",
                       (song_id, decision, assessment["verdict"], Jsonb(assessment))).fetchone()
    routed = []
    if row:   # first time through: route the fixes and file the prediction
        for f in assessment["fixes"]:
            conn.execute("INSERT INTO song_feedback (song_id, assessment_id, role, problem, change, evidence) "
                         "VALUES (%s, %s, %s, %s, %s, %s)",
                         (song_id, row["id"], f["role"], f["problem"], f["change"], Jsonb(f["evidence"])))
            routed.append(f["role"])
        if c.get("commission"):
            for r in sorted(set(routed)):
                if r in agent_mod.REGISTRY:
                    commission(ctx, r, {"song_id": str(song_id), "assessment_id": str(row["id"])},
                               reason="the music critic's fixes", key=f"fix:{song_id}:{r}")
    filed = None
    pred = assessment.get("prediction")
    if pred and not conn.execute("SELECT 1 FROM predictions WHERE decision_id = %s", (decision,)).fetchone():
        filed = music_analysis.file_prediction(conn, policy, role=role, song_id=song_id, claim=pred["claim"],
                                               metric=pred["metric"], comparator=pred["comparator"],
                                               threshold=float(pred["threshold"]),
                                               probability=float(pred["probability"]), decision_id=decision)
    return {"song_id": str(song_id), "verdict": assessment["verdict"], "fixes": len(assessment["fixes"]),
            "routed": routed, "dropped": len(assessment.get("dropped") or []),
            "prediction": str(filed) if filed else None, "decision_id": str(decision)}


class MusicCritic:
    name = NAME

    def run(self, ctx: AgentContext) -> dict:
        if not _cfg(ctx.policy).get("enabled", False):
            return {"skipped": "music.critic.enabled is off"}
        p = ctx.job.payload or {}
        if not p.get("song_id"):
            raise NonRetryable("music_critic payload needs song_id")
        return critique(ctx, p["song_id"])


register(MusicCritic())
