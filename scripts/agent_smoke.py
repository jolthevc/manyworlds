"""Paid smoke test of the Brain's newest LLM paths, no GPU (owner-approved overnight test spend, under $5):
the talent manager onboarding a sandbox clone (dossier sections + first brief), and one director call that proves
the shot-sheet schema still compiles. Run on the control plane: python scripts/agent_smoke.py <sandbox handle>.
Prints what was built and what it cost; queues nothing for a GPU (plates the director registers are cancelled)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from manyworlds import db, policy as policy_mod, queue
from manyworlds.brain import director
from manyworlds.brain.agent import AgentContext
from manyworlds.brain.agents import talent_manager
from manyworlds.creator import profile
from manyworlds.llm import LLM


def main(handle: str) -> None:
    pol = policy_mod.load()
    conn = db.connect()
    c = conn.execute("SELECT id, anchors FROM creators WHERE handle = %s", (handle,)).fetchone()
    if not c or not (c["anchors"] or {}).get("simulation"):
        raise SystemExit(f"{handle} is not a sandbox clone; refusing to touch a real creator")
    cid = c["id"]
    t0 = datetime.now(timezone.utc)
    llm = LLM(conn, pol)
    out: dict = {"creator": handle}
    if profile.current(conn, cid):
        job = queue.enqueue(conn, "brain.talent_manager", {"creator_id": str(cid), "mode": "onboard", "test": True},
                            creator_id=cid, batch="test-agent-smoke")
        conn.execute("UPDATE jobs SET status = 'running', worker_id = 'smoke', attempts = 1 WHERE id = %s", (job,))
        row = conn.execute("SELECT id, kind, payload, attempts, max_attempts, creator_id, batch FROM jobs WHERE id = %s",
                           (job,)).fetchone()
        ctx = AgentContext(conn=conn, policy=pol, llm=llm, job=queue.Job(**row), now=t0)
        try:
            with llm.tagged("talent_manager", job):
                out["talent_manager"] = talent_manager.TalentManager().run(ctx)
            conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job,))
        except Exception as e:  # noqa: BLE001 - report it
            out["talent_manager_error"] = f"{type(e).__name__}: {e}"[:800]
            conn.execute("UPDATE jobs SET status = 'failed', error = %s WHERE id = %s", (str(e)[:500], job))
        prof = profile.current(conn, cid)[1]
        out["dossier"] = {k: prof.get(k) for k in ("archetypes", "positioning", "content_system")}
        b = talent_manager.latest_brief(conn, cid)
        out["brief"] = b["output"] if b else None
    body = {"scenes": ["behind an empty bar at closing time, talking to her propped phone about the night"],
            "script": "Closing playlist, rate it. Slow songs only, lights up, everybody out.", "format": "talking_video"}
    try:
        with llm.tagged("director", None):
            shots = director.direct(conn, pol, llm, body, cid, fmt="talking_video")
        out["director"] = {"shots": len(shots), "first": shots[0] if shots else None}
    except Exception as e:  # noqa: BLE001
        out["director_error"] = f"{type(e).__name__}: {e}"[:800]
    cancelled = conn.execute("UPDATE jobs SET status = 'failed', error = 'agent smoke: not for rendering' "
                             "WHERE creator_id = %s AND status = 'queued' AND created_at >= %s AND kind LIKE 'image.%%' "
                             "RETURNING id", (cid, t0)).fetchall()
    out["plates_cancelled"] = len(cancelled)
    out["usd"] = float(conn.execute("SELECT coalesce(sum(usd), 0) u FROM cost_entries WHERE created_at >= %s "
                                    "AND category = 'llm'", (t0,)).fetchone()["u"])
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
