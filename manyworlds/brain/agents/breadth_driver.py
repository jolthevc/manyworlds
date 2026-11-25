"""The breadth test's driver as a durable brain job (2026-11-25, `mw test breadth --queue`).

The driver used to be a shell loop (`mw test breadth`, started over ssh into /tmp on the autoscaler machine): every
Fly deploy restarted the machine and killed it, and its run lease then blocked a restart until it lapsed. Queued, each
job run is one driver step (manyworlds.breadth.run with once=True: advance the run's posts and takes, adopt and
review its jobs, check its spend cap and deadline), then the job defers itself for breadth.loop_seconds (no attempt
charged). A deploy that kills a step hands the job back (brain.runtime) and the next brain worker resumes the run where
the database says it is: posts, jobs and reviews are all rows, and every step is idempotent.

It runs while operations are paused (its payload is a test). Its review calls go through a client on the test
overlay, as the CLI driver's did; the run's own breadth.max_usd caps the run's spend.
"""

from __future__ import annotations

from ..agent import AgentContext, Defer, register
from ...worker.runner import NonRetryable


def _store():
    from ... import storage
    return storage.from_url()


def _review_llm(ctx: AgentContext, tpol):
    """The driver's review client: a Claude client on the test overlay (as `mw test breadth` makes), or the context's
    client where it is not Claude (tests)."""
    from ...llm import Claude
    return Claude(ctx.conn, tpol) if isinstance(ctx.llm, Claude) else ctx.llm


class BreadthDriver:
    name = "breadth"

    def run(self, ctx: AgentContext) -> dict:
        from ... import breadth
        p = dict(ctx.job.payload or {})
        if not p.get("plan"):
            raise NonRetryable("a breadth driver job needs payload.plan")
        tpol = breadth.test_policy(ctx.policy)
        cfg = breadth._cfg(tpol)
        loop_s = float(cfg.get("loop_seconds", 60))
        llm = _review_llm(ctx, tpol) if cfg.get("review_in_driver", True) else None
        try:
            out = breadth.run(ctx.conn, ctx.policy, p["plan"], items=p.get("items"), run=p.get("run"),
                              creator=p.get("creator"), store=_store(), llm=llm, once=True, hours=p.get("hours"))
        except breadth.DriverBusy as e:   # a step killed mid-way still holds the run's lease: wait for it to lapse
            raise Defer(loop_s, {"waiting": str(e)}) from None
        except (SystemExit, ValueError) as e:   # refused (live operations, other open test work) or a bad plan
            raise NonRetryable(str(e) or type(e).__name__) from None
        summary = {k: out.get(k) for k in ("run", "batch", "posts", "bench_open_jobs", "takes", "spent_usd",
                                           "deadline", "errors", "spend_cap", "deadline_passed", "decision")
                   if out.get(k) is not None}
        if not out.get("finished"):
            raise Defer(loop_s, summary)
        return {**summary, "finished": True}


register(BreadthDriver())
