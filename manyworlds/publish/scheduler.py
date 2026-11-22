"""Publishes what is due: `ready` posts whose time has come, through the creator's publisher.

Sandbox creators with a test scenario (`anchors.simulation`) use the simulated publisher; their posts
become `simulated`. Sandbox creators without one are waiting for their real account: their posts stay
`ready` until it is connected. Instagram creators use the Instagram adapter (M4) once registered.

Gates, all from policy:
- `autonomy.publish: paused` is the kill switch: nothing is published.
- Real platforms publish only when `autonomy.publish` is `autonomous`; `human_approval` and `sampled`
  hold real posts (`awaiting_approval`) until an approval flow exists. Simulated posts aren't gated.
- A real post more than `publishing.max_late_hours` past its time is left `ready` (`stale`) rather than
  sent with others in a burst (e.g. a backlog when an account is first connected).
- Paused or retired creators publish nothing.
`simulation.fast_forward` (sandbox only) also publishes `planned` posts without making them. After each
simulated publication the scenario engine produces its outcomes.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from . import instagram
from .simulated import SimulatedPublisher

PUBLISHERS = {"simulated": SimulatedPublisher(), "instagram": instagram.InstagramPublisher()}


def publish_due(conn, policy, *, now: datetime, creator_id=None) -> dict:
    mode = policy.get("autonomy", "publish", default="human_approval")
    if mode == "paused":
        return {"paused": True}
    fast = bool(policy.get("simulation", "fast_forward", default=False))
    max_late = timedelta(hours=float(policy.get("publishing", "max_late_hours", default=6)))
    out = {"simulated": 0, "published": 0, "waiting_for_adapter": 0, "waiting_for_account": 0,
           "awaiting_approval": 0, "stale": 0, "pending": 0, "failed": 0, "quota": 0}
    rows = conn.execute(
        """SELECT p.*, c.publish_mode, c.anchors FROM posts p JOIN creators c ON c.id = p.creator_id
           WHERE p.scheduled_for <= %s AND c.status NOT IN ('paused', 'retired')
             AND (p.status = 'ready' OR (%s AND p.status = 'planned' AND c.publish_mode = 'simulated'))
             AND (%s::uuid IS NULL OR p.creator_id = %s)
           ORDER BY p.scheduled_for""", (now, fast, creator_id, creator_id)).fetchall()
    for post in rows:
        sim = (post["anchors"] or {}).get("simulation") or {}
        if post["publish_mode"] == "simulated":
            if not sim.get("scenario"):
                out["waiting_for_account"] += 1   # sandbox only until the real account is connected
                continue
            publisher = PUBLISHERS["simulated"]
        else:
            publisher = PUBLISHERS.get(post["publish_mode"])
            if publisher is None:
                out["waiting_for_adapter"] += 1
                continue
            in_flight = bool((post.get("body") or {}).get("ig"))   # a container exists: only reconcile it
            if mode != "autonomous" and not in_flight:
                out["awaiting_approval"] += 1
                continue
            if now - post["scheduled_for"] > max_late and not in_flight:
                out["stale"] += 1
                continue
        try:
            result = publisher.publish(conn, policy, post)
        except instagram.Pending:
            out["pending"] += 1            # container processing at Instagram: published on a later tick
            continue
        except instagram.Ramped:
            out["ramp"] = out.get("ramp", 0) + 1   # a new account: today's ramp allowance is used
            continue
        except instagram.QuotaReached:
            out["quota"] += 1
            continue
        except instagram.AccountNotConnected:
            out["waiting_for_account"] += 1
            continue
        except instagram.GraphError:
            out["failed"] += 1             # recorded on the post (body.publish_errors); retried next tick
            continue
        except instagram.Held:
            out["held"] = out.get("held", 0) + 1   # the owner decides (post status held, owner event)
            continue
        except Exception as e:  # noqa: BLE001 - one bad post must not stop the others or production
            out["errors"] = out.get("errors", []) + [f"{post['id']}: {type(e).__name__}: {str(e)[:200]}"]
            continue
        published_at = post["scheduled_for"] if result["simulated"] else now
        with conn.transaction():
            row = conn.execute(
                """INSERT INTO publications (post_id, platform, account_id, external_id, permalink, ai_label, simulated,
                                            published_at)
                   VALUES (%s, %s, (SELECT id FROM accounts WHERE creator_id = %s AND platform = %s), %s, %s, %s, %s, %s)
                   ON CONFLICT (post_id, platform) DO NOTHING RETURNING id""",
                (post["id"], publisher.platform, post["creator_id"], publisher.platform, result["external_id"],
                 result["permalink"], result["ai_label"], result["simulated"], published_at)).fetchone()
            if row is None:
                continue
            conn.execute("UPDATE posts SET status = %s, published_at = %s, external_id = %s, updated_at = now() "
                         "WHERE id = %s", ("simulated" if result["simulated"] else "published", published_at,
                                           result["external_id"], post["id"]))
        out["simulated" if result["simulated"] else "published"] += 1
        if result["simulated"]:
            from ..brain import simulation
            simulation.simulate_post(conn, policy, post["id"], scenario=sim["scenario"], seed=int(sim.get("seed", 0)))
    return out
