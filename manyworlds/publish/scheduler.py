"""Publishes what is due: `ready` posts whose time has come, through the creator's publisher.

Sandbox creators with a test scenario (`anchors.simulation`) use the simulated publisher; their posts
become `simulated`. Sandbox creators without one are waiting for their real account: their posts stay
`ready` until it is connected. Instagram creators wait until the Instagram adapter exists (M4). Policy
`autonomy.publish: paused` is the kill switch: nothing is published. `simulation.fast_forward`
(sandbox only) also publishes `planned` posts without making them, to run the learning loop fast.
After each simulated publication the scenario engine produces its outcomes, when the creator has
a scenario (`anchors.simulation.scenario`).
"""

from __future__ import annotations

from datetime import datetime

from .simulated import SimulatedPublisher

PUBLISHERS = {"simulated": SimulatedPublisher()}


def publish_due(conn, policy, *, now: datetime, creator_id=None) -> dict:
    if policy.get("autonomy", "publish", default="human_approval") == "paused":
        return {"paused": True}
    fast = bool(policy.get("simulation", "fast_forward", default=False))
    out = {"simulated": 0, "published": 0, "waiting_for_adapter": 0, "waiting_for_account": 0}
    rows = conn.execute(
        """SELECT p.*, c.publish_mode, c.anchors FROM posts p JOIN creators c ON c.id = p.creator_id
           WHERE p.scheduled_for <= %s AND (p.status = 'ready' OR (%s AND p.status = 'planned'
                                             AND c.publish_mode = 'simulated'))
             AND (%s::uuid IS NULL OR p.creator_id = %s)
           ORDER BY p.scheduled_for""", (now, fast, creator_id, creator_id)).fetchall()
    for post in rows:
        scenario = ((post["anchors"] or {}).get("simulation") or {}).get("scenario")
        if post["publish_mode"] == "simulated" and not scenario:
            # A sandbox creator without a test scenario is waiting for their real account (e.g. Maren before
            # Instagram is connected): their posts stay ready for it, never consumed by simulation.
            out["waiting_for_account"] += 1
            continue
        platform = "simulated" if post["publish_mode"] == "simulated" else post["publish_mode"]
        publisher = PUBLISHERS.get(platform)
        if publisher is None:
            out["waiting_for_adapter"] += 1
            continue
        result = publisher.publish(conn, policy, post)
        with conn.transaction():
            row = conn.execute(
                """INSERT INTO publications (post_id, platform, external_id, permalink, ai_label, simulated, published_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (post_id, platform) DO NOTHING RETURNING id""",
                (post["id"], publisher.platform, result["external_id"], result["permalink"], result["ai_label"],
                 result["simulated"], post["scheduled_for"])).fetchone()
            if row is None:
                continue
            conn.execute("UPDATE posts SET status = %s, published_at = %s, external_id = %s, updated_at = now() "
                         "WHERE id = %s", ("simulated" if result["simulated"] else "published", post["scheduled_for"],
                                           result["external_id"], post["id"]))
        out["simulated" if result["simulated"] else "published"] += 1
        if result["simulated"] and scenario:
            from ..brain import simulation
            simulation.simulate_post(conn, policy, post["id"], scenario=scenario,
                                     seed=int(((post["anchors"] or {}).get("simulation") or {}).get("seed", 0)))
    return out
