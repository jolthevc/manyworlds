"""Publishes what is due: `ready` posts whose time has come, through the creator's publisher.

Sandbox creators (`publish_mode = simulated`) use the simulated publisher; their posts become
`simulated`. Instagram creators wait until the Instagram adapter exists (M4). Policy
`autonomy.publish: paused` is the kill switch: nothing is published. `simulation.fast_forward`
(sandbox only) also publishes `planned` posts without making them, to run the learning loop fast.
After each simulated publication the scenario engine produces its outcomes, when the creator has
a scenario (`anchors.simulation.scenario`).
"""

from __future__ import annotations

from datetime import datetime

from .simulated import SimulatedPublisher

PUBLISHERS = {"simulated": SimulatedPublisher()}


def publish_due(conn, policy, *, now: datetime) -> dict:
    if policy.get("autonomy", "publish", default="human_approval") == "paused":
        return {"paused": True}
    fast = bool(policy.get("simulation", "fast_forward", default=False))
    out = {"simulated": 0, "published": 0, "waiting_for_adapter": 0}
    rows = conn.execute(
        """SELECT p.*, c.publish_mode, c.anchors FROM posts p JOIN creators c ON c.id = p.creator_id
           WHERE p.scheduled_for <= %s AND (p.status = 'ready' OR (%s AND p.status = 'planned'
                                             AND c.publish_mode = 'simulated'))
           ORDER BY p.scheduled_for""", (now, fast)).fetchall()
    for post in rows:
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
        scenario = ((post["anchors"] or {}).get("simulation") or {}).get("scenario")
        if result["simulated"] and scenario:
            from ..brain import simulation
            simulation.simulate_post(conn, policy, post["id"], scenario=scenario,
                                     seed=int(((post["anchors"] or {}).get("simulation") or {}).get("seed", 0)))
    return out
