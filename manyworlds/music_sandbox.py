"""`mw music sandbox <handle> --songs N [--usd X]`: the whole music department, end to end, on a sandbox artist, with
the GPU stubbed (no music.song job runs: manyworlds.music_pipeline.StubTakes).

Refuses any creator that is not in the sandbox (publish_mode other than simulated: evidence mode real). Runs, in
one process, under one LLM spend cap for the whole run (music_roles.spend_cap; --usd, default music.sandbox.max_usd):

  1. the music talent manager: the dossier if the artist has none (onboard), then a review asking for N song briefs;
  2. music experiments: due ones read, new ones opened from tentative music findings (brain.experiments), and this
     release batch's arms assigned to the songs;
  3. each brief through the pipeline: the writing roles, the critic's draft review and the revision loop, the
     stand-in takes, melody's pick, the critic, the mix plan;
  4. the curator's release from the finished songs; A&R's pitches; head of music's slate (sandbox artists shown).

Every call is a recorded decision under one bookkeeping job (brain.music_sandbox). The run stops cleanly when the cap
is reached (budget.OverBudget) and reports what it did. Sandbox outcomes are SIMULATED: never evidence of demand.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from . import queue
from .policy import Policy


class NotSandbox(ValueError):
    """The creator is real: the sandbox runner never touches a real creator."""


def _ctx(conn, policy: Policy, llm, creator_id, run: str):
    from .brain.agent import AgentContext
    from .brain import overrides
    job = queue.enqueue(conn, "brain.music_sandbox", {"run": run}, creator_id=creator_id,
                        idempotency_key=f"music-sandbox:{run}")
    conn.execute("UPDATE jobs SET status = 'running', worker_id = 'music-sandbox', attempts = attempts + 1 "
                 "WHERE id = %s", (job,))
    row = conn.execute("SELECT id, kind, payload, attempts, max_attempts, creator_id, batch FROM jobs WHERE id = %s",
                       (job,)).fetchone()
    return AgentContext(conn=conn, policy=overrides.effective_policy(conn, policy), llm=llm, job=queue.Job(**row),
                        now=datetime.now(timezone.utc))


def run(conn, policy: Policy, llm, handle: str, *, songs_n: int | None = None, usd: float | None = None,
        run_id: str | None = None) -> dict:
    from . import music_pipeline
    from .brain import budget, evidence, experiments
    from .brain.agents import a_and_r, curator, head_of_music, music_talent_manager
    from .brain.music_roles import spend_cap
    c = policy.get("music", "sandbox", default={}) or {}
    row = conn.execute("SELECT id FROM creators WHERE handle = %s", (handle,)).fetchone()
    if row is None:
        raise NotSandbox(f"no creator {handle!r}")
    cid = row["id"]
    if evidence.mode_for(conn, cid) != "sandbox":
        raise NotSandbox(f"{handle} is a real creator: the music sandbox runs only on sandbox artists")
    n = int(songs_n if songs_n is not None else c.get("default_songs", 3))
    cap = float(usd if usd is not None else c.get("max_usd", 5.0))
    if cap > float(c.get("max_usd", 5.0)):
        raise ValueError(f"--usd {cap} is above music.sandbox.max_usd {c.get('max_usd')}")
    run_id = run_id or uuid.uuid4().hex[:8]
    conn.execute("UPDATE creators SET vertical = 'music' WHERE id = %s", (cid,))
    ctx = _ctx(conn, policy, llm, cid, run_id)
    report: dict = {"handle": handle, "run": run_id, "songs_asked": n, "usd_cap": cap, "songs": []}
    try:
        with spend_cap(conn, cap):
            report["dossier"] = music_talent_manager.onboard(ctx, cid)   # reused when one exists or was recorded
            if report["dossier"].get("refused"):
                report["stopped"] = f"the dossier was refused: {report['dossier']['refused']}"
                return _finish(conn, ctx, report)
            career = music_talent_manager.review(ctx, cid, songs_wanted=n)
            report["career"] = {k: career[k] for k in ("state", "predictions", "requests", "dropped_briefs")}
            briefs = career["briefs"] or []
            while briefs and len(briefs) < n:           # fewer briefs than songs asked: the last one again
                briefs.append(dict(briefs[-1]))
            report["experiments"] = experiments.music_review(conn, ctx.policy, cid)
            arms = experiments.music_assign(conn, ctx.policy, cid, batch_key=run_id, n_songs=len(briefs))
            for i, b in enumerate(briefs[:n]):
                out = music_pipeline.run_song(ctx, cid, b, key=f"{run_id}-{i}", takes=music_pipeline.StubTakes(),
                                              experiment=arms[i])
                report["songs"].append({"state": out["state"], "song_id": out.get("song_id"),
                                        "verdict": out.get("verdict"), "why": out.get("why"),
                                        "revisions": out["draft"].get("revisions"),
                                        "title": (out["draft"].get("words") or {}).get("title")})
            report["release"] = curator.curate(ctx, cid, step=f"release:{run_id}")
            report["a_and_r"] = a_and_r.pitch(ctx, step=f"pitch:{run_id}")
            report["slate"] = head_of_music.plan(ctx, include_sandbox=True, step=f"slate:{run_id}")
    except budget.OverBudget as e:
        report["stopped"] = str(e)
    return _finish(conn, ctx, report)


def _finish(conn, ctx, report: dict) -> dict:
    report["usd_spent"] = round(spent_in_cap_after(conn, ctx), 4)
    report["decisions"] = conn.execute("SELECT count(*) n FROM decisions WHERE job_id = %s",
                                       (ctx.job.id,)).fetchone()["n"]
    report["resume"] = f"mw music sandbox {report['handle']} --songs {report['songs_asked']} --run {report['run']}"
    queue.complete(conn, ctx.job.id, "music-sandbox", {k: v for k, v in report.items() if k != "slate"})
    return report


def spent_in_cap_after(conn, ctx) -> float:
    """What the run's LLM calls cost (the ledger's llm entries charged to this run's job, or tagged to its agents)."""
    row = conn.execute("SELECT coalesce(sum(usd), 0) u FROM cost_entries WHERE category = 'llm' AND created_at >= "
                       "(SELECT created_at FROM jobs WHERE id = %s)", (ctx.job.id,)).fetchone()
    return float(row["u"])


def create_artist(conn, policy: Policy, handle: str, *, genre: str, display_name: str | None = None) -> str:
    """A new sandbox music artist (publish_mode simulated, a sandbox scenario so its spend is tagged sandbox:, never a
    portfolio creator): a minimal original bible and its version-1 profile (no model call: the talent manager's
    dossier is stored in the profile, brain.agents.music_talent_manager.store_dossier)."""
    from psycopg.types.json import Jsonb
    if conn.execute("SELECT 1 FROM creators WHERE handle = %s", (handle,)).fetchone():
        raise ValueError(f"a creator {handle!r} exists")
    bible = {"display_name": display_name or handle, "age": 24, "archetype": "singer_songwriter",
             "persona": f"an original {genre} singer-songwriter (openly AI-generated, not based on anyone real)",
             "music": {"genre": genre}, "niche": f"{genre} music"}
    cid = str(conn.execute(
        "INSERT INTO creators (handle, status, bible, anchors, publish_mode, vertical) VALUES (%s, 'draft', %s, %s, "
        "'simulated', 'music') RETURNING id",
        (handle, Jsonb(bible), Jsonb({"simulation": {"scenario": "music_sandbox", "seed": 0}}))).fetchone()["id"])
    from .creator import profile as profile_mod
    from .brain.agents.music_talent_manager import initial_profile
    profile_mod.save(conn, policy, cid, initial_profile({"handle": handle, **bible}), author="music_sandbox",
                     reason="sandbox artist: initial profile")
    return cid
