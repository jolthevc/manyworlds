"""`mw` command line: migrate, run workers, enqueue work, benchmark and report."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from . import bench, db, ledger, policy as policy_mod, queue, storage


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mw")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate", help="apply schema migrations and register the policy")

    w = sub.add_parser("worker", help="run a GPU worker (kinds from MW_WORKER_KINDS or --kinds)")
    w.add_argument("--kinds")
    w.add_argument("--pool")
    w.add_argument("--max-jobs", type=int)
    w.add_argument("--idle-exit", type=float, help="exit after this many idle seconds")

    e = sub.add_parser("enqueue", help="enqueue one or more jobs")
    e.add_argument("kind")
    e.add_argument("payload", help="JSON payload")
    e.add_argument("--n", type=int, default=1)
    e.add_argument("--batch")

    b = sub.add_parser("bench", help="enqueue a benchmark batch")
    b.add_argument("kind")
    b.add_argument("--jobs", type=int, default=20)
    b.add_argument("--extra", default="{}", help="JSON merged into each payload")
    b.add_argument("--pools", help="comma-separated GPU pools to pin the batch to, e.g. vast_4090")
    b.add_argument("--profile", help="talk settings under test (policy generation.talk_profiles), e.g. fast")

    r = sub.add_parser("report", help="benchmark / cost report for a batch")
    r.add_argument("batch")

    sub.add_parser("discover", help="Brain: plan research, pull it via Apify, propose creator theses")
    sub.add_parser("choose", help="Brain: choose the next thesis to launch")
    ac = sub.add_parser("account", help="connect a creator's Instagram account (the token stays in .env)")
    ac.add_argument("action", choices=["connect"])
    ac.add_argument("creator")
    ac.add_argument("--token-env", required=True, help="name of the .env variable holding the long-lived token")
    sub.add_parser("collect-metrics", help="pull Instagram insights for published posts (real metrics)")
    rc = sub.add_parser("reconcile", help="pull providers' itemised charges into the ledger (true machine cost)")
    rc.add_argument("--days", type=int, default=None, help="window to pull (default policy autoscale.reconcile_days)")
    lc = sub.add_parser("launch-creator", help="Creator Lab: bible + creator + reference pack + voice sample")
    lc.add_argument("--thesis", help="thesis id (default: the chosen one)")
    an = sub.add_parser("anchors", help="pick the reference images for a creator")
    an.add_argument("creator")
    tl = sub.add_parser("train-lora", help="queue identity LoRA training for a creator")
    tl.add_argument("creator")
    cr = sub.add_parser("creator", help="show a creator's bible, anchors and jobs")
    cr.add_argument("creator")
    sub.add_parser("status", help="job counts, spend and where GPU money went (72 h)")
    mc = sub.add_parser("machines", help="each GPU machine: billed vs finished work")
    mc.add_argument("--hours", type=float, default=72)
    wc = sub.add_parser("worker-credentials",
                        help="create or rotate the GPU machines' scoped database login; prints "
                             "WORKER_DATABASE_URL=... for piping into `fly secrets import` (never into a terminal)")
    wc.add_argument("--force", action="store_true", help="rotate even while GPU machines are up (they lose access)")
    lm = sub.add_parser("llm", help="LLM models: compare candidates with a role's model, choose the best per job")
    lm.add_argument("action", choices=["shadow-report", "recommend", "judge"],
                    help="shadow-report: the comparisons; recommend: which candidate qualifies (--propose files it as "
                         "an approval); judge: score a generative role's answers blind (spends up to --usd)")
    lm.add_argument("role", help="a role, or `all` for recommend")
    lm.add_argument("--propose", action="store_true")
    lm.add_argument("--usd", type=float, default=1.0, help="judge: most to spend")
    lm.add_argument("--days", type=float, default=30)
    lm.add_argument("--tolerance", type=float, default=1.0, help="numeric difference counted as agreeing")
    sub.add_parser("reap", help="mark jobs dead whose final lease expired")
    sub.add_parser("init-storage", help="create the object storage bucket if missing")
    rv = sub.add_parser("review", help="run review gates on new creator images (--loop to keep going)")
    rv.add_argument("--batch")
    rv.add_argument("--loop", action="store_true")
    rv.add_argument("--role", help="LLM role to judge with, for comparing models (with --dry-run)")
    rv.add_argument("--dry-run", action="store_true", help="print verdicts, write no reviews")
    pl = sub.add_parser("plan", help="Brain: plan a creator's posts for the coming period")
    pl.add_argument("creator")
    pl.add_argument("--start", help="first day, YYYY-MM-DD (default tomorrow, UTC)")
    pl.add_argument("--force", action="store_true", help="plan again even if the period has a plan")
    pr = sub.add_parser("produce", help="turn due planned posts into reviewed images (--loop to keep going)")
    pr.add_argument("--loop", action="store_true")
    pf = sub.add_parser("profile", help="show, build or change a creator's profile")
    pf.add_argument("creator")
    pf.add_argument("--build", action="store_true", help="write version 1 from the bible (Brain)")
    pf.add_argument("--set", action="append", default=[], metavar="PATH=JSON",
                    help="e.g. look.hair='a short bob' or dials.posts_per_day.tiktok=4 (repeatable)")
    pf.add_argument("--reason", default="")
    po = sub.add_parser("posts", help="list a creator's posts")
    po.add_argument("creator")
    sc = sub.add_parser("autoscale", help="start/stop GPU machines to follow the queue (loops; --once for one tick)")
    sc.add_argument("--once", action="store_true")
    sc.add_argument("--no-launch", action="store_true")
    br = sub.add_parser("brain", help="Brain agents: worker, run, trace, approvals")
    bsub = br.add_subparsers(dest="brain_cmd", required=True)
    bw = bsub.add_parser("worker", help="run brain.* agent jobs on this host")
    bw.add_argument("--loop", action="store_true")
    brn = bsub.add_parser("run", help="enqueue one agent job and run one tick")
    brn.add_argument("agent")
    brn.add_argument("--payload", default="{}")
    bt = bsub.add_parser("trace", help="a job, its decisions, commissioned work and LLM spend by agent")
    bt.add_argument("job")
    ba = bsub.add_parser("approvals", help="list pending approvals, or decide one")
    ba.add_argument("--decide")
    bdec = ba.add_mutually_exclusive_group()
    bdec.add_argument("--approve", action="store_true")
    bdec.add_argument("--reject", action="store_true")
    bdr = bsub.add_parser("rubric-drift", help="score the reference set under the current rubric and model; drift vs before")
    bdr.add_argument("rubric")
    bcal = bsub.add_parser("calibrate", help="score a creator's latest posts as ideas; noise per dimension")
    bcal.add_argument("creator")
    bcal.add_argument("--posts", type=int, default=12)
    bsim = bsub.add_parser("simulate", help="run the closed learning loop on a sandbox creator (fast-forward, SIMULATED)")
    bsim.add_argument("creator")
    bsim.add_argument("--batches", type=int, default=3)
    bsim.add_argument("--clone-from", help="create the sandbox creator as a copy of this creator first")
    bsim.add_argument("--trial-usd", type=float, help="also send every call to every llm.shadow candidate, up to "
                      "this much candidate spend (compare models on the same inputs: mw llm shadow-report)")
    bsim.add_argument("--scenario", default="personality_wins")
    bsim.add_argument("--seed", type=int, default=0)
    blearn = bsub.add_parser("learning", help="what the Brain learned for a creator and how its plans used it")
    blearn.add_argument("creator")

    args = p.parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    pol = policy_mod.load()

    if args.cmd == "worker":
        from .worker import runner
        kinds = [k.strip() for k in (args.kinds or os.environ["MW_WORKER_KINDS"]).split(",") if k.strip()]
        worker = runner.start(kinds, pool=args.pool or os.environ.get("MW_POOL", "local"), policy=pol,
                              store=storage.from_url())
        n = worker.run(max_jobs=args.max_jobs, idle_exit_seconds=args.idle_exit)
        print(f"worker {worker.worker_id} handled {n} jobs")
        return 0

    if args.cmd == "autoscale":
        from .deploy import autoscale
        return autoscale.main((["--once"] if args.once else []) + (["--no-launch"] if args.no_launch else []))

    if args.cmd == "init-storage":
        store = storage.from_url()
        if isinstance(store, storage.S3Store):
            store.ensure_bucket()
        print("storage ready")
        return 0

    conn = db.connect()
    if args.cmd == "migrate":
        applied = db.migrate(conn)
        policy_mod.register(conn, pol)
        print(f"applied: {applied or 'nothing new'}; policy {pol.version}")
    elif args.cmd == "enqueue":
        payload = json.loads(args.payload)
        ids = [queue.enqueue(conn, args.kind, payload, batch=args.batch) for _ in range(args.n)]
        _print({"enqueued": ids})
    elif args.cmd == "account":
        from .publish import instagram
        _print(instagram.connect(conn, pol, _creator_id(conn, args.creator), token_env=args.token_env))
    elif args.cmd == "collect-metrics":
        from .publish import metrics
        _print(metrics.collect(conn, pol))
    elif args.cmd == "reconcile":
        from .deploy import reconcile
        from .deploy.providers import all_providers
        _print(reconcile.run(conn, pol, all_providers(pol), days=args.days))
    elif args.cmd == "bench":
        batch = bench.enqueue(conn, pol, args.kind, args.jobs, extra=json.loads(args.extra),
                              pools=args.pools.split(",") if args.pools else None, profile=args.profile)
        print(batch)
    elif args.cmd == "report":
        _print(bench.report(conn, pol, args.batch))
    elif args.cmd == "status":
        from .deploy import report
        _print({"jobs": queue.counts(conn), "spend": ledger.totals(conn),
                "cost_per_output": ledger.cost_per_output(conn), "brain_costs": ledger.brain_costs(conn),
                "machines_72h": report.summary(report.machines(conn))})
    elif args.cmd == "machines":
        from .deploy import report
        rows = report.machines(conn, hours=args.hours)
        _print({"summary": report.summary(rows), "machines": rows})
    elif args.cmd == "posts":
        cid = _creator_id(conn, args.creator)
        _print(conn.execute("SELECT id, status, format, platform, scheduled_for, caption, qa_status, "
                            "body->'scenes' AS scenes, body->'images' AS images, body->>'drop_reason' AS drop_reason "
                            "FROM posts WHERE creator_id = %s ORDER BY scheduled_for", (cid,)).fetchall())
    elif args.cmd == "reap":
        print(f"reaped {queue.reap(conn)} jobs")
    elif args.cmd == "llm":
        from . import selection, shadow
        if args.action == "shadow-report":
            _print(shadow.report(conn, args.role, days=args.days, tolerance=args.tolerance))
        elif args.action == "judge":
            from .llm import LLM
            _print(selection.judge(conn, pol, LLM(conn, pol), args.role, usd=args.usd))
        else:
            roles = list(pol.get("llm", "selection", "roles", default={}) or {}) if args.role == "all" else [args.role]
            recs = [selection.recommend(conn, pol, r) for r in roles]
            _print([selection.propose(conn, pol, r["role"], r) for r in recs] if args.propose else recs)
    elif args.cmd == "worker-credentials":
        from .deploy import credentials
        up = conn.execute("SELECT count(*) AS n FROM gpu_groups WHERE stopped_at IS NULL").fetchone()["n"]
        if up and not args.force:
            raise SystemExit(f"{up} GPU machine(s) are up and would lose database access; rerun with --force")
        print(f"WORKER_DATABASE_URL={credentials.rotate_login(conn, os.environ['DATABASE_URL'])}")
    elif args.cmd == "brain":
        return _brain_ops(args, conn, pol)
    elif args.cmd in ("discover", "choose", "launch-creator", "anchors", "train-lora", "creator", "review", "plan", "profile", "produce"):
        return _brain(args, conn, pol)
    return 0


def _brain_ops(args, conn, pol) -> int:
    import socket
    import time

    from .brain import ops, runtime
    from .llm import Claude

    if args.brain_cmd == "trace":
        _print(ops.trace(conn, args.job))
        return 0
    if args.brain_cmd == "approvals":
        if args.decide:
            if not (args.approve or args.reject):
                raise SystemExit("--decide needs --approve or --reject")
            ops.decide(conn, args.decide, approved=args.approve, by="owner")
        _print(ops.approvals(conn))
        return 0
    if args.brain_cmd == "learning":
        from .brain import sandbox
        _print(sandbox.learning_trace(conn, _creator_id(conn, args.creator)))
        return 0
    llm = Claude(conn, pol)
    if args.brain_cmd == "simulate":
        llm.trial_usd = args.trial_usd
        from .brain import sandbox
        if args.clone_from and not conn.execute("SELECT 1 FROM creators WHERE handle = %s", (args.creator,)).fetchone():
            sandbox.clone(conn, args.clone_from, args.creator, scenario=args.scenario, seed=args.seed)
        cid = _creator_id(conn, args.creator)
        runs = sandbox.run_batches(conn, pol, llm, cid, batches=args.batches,
                                   worker_id=f"sandbox-{socket.gethostname()}-{os.getpid()}")
        _print({"note": "SIMULATED: images are not generated in fast-forward; outcomes come from the scenario engine",
                "runs": runs, "llm_usd": llm.spent})
        return 0
    if args.brain_cmd in ("rubric-drift", "calibrate"):
        from .brain import evaluation, rubrics
        with llm.tagged("evaluator", None):   # Brain spend, outside cost per output
            if args.brain_cmd == "calibrate":
                _print({**evaluation.calibrate(conn, pol, llm, _creator_id(conn, args.creator), posts=args.posts),
                        "llm_usd": llm.spent})
                return 0
            r = rubrics.load(args.rubric)
            rubrics.register(conn, r)
            after = evaluation.score_reference(conn, pol, llm, args.rubric)   # stores the run and its drift
            ref = conn.execute("SELECT reference FROM rubric_versions WHERE rubric = %s AND version = %s",
                               (r.rubric, r.version)).fetchone()["reference"]
            _print({"rubric": r.rubric, "version": r.version, "prompt_version": evaluation.prompt_version(pol),
                    "drift": (ref.get("drift") or [None])[-1], "scores": after, "llm_usd": llm.spent})
            return 0
    worker_id = f"brain-{socket.gethostname()}-{os.getpid()}"
    if args.brain_cmd == "run":
        job = queue.enqueue(conn, f"brain.{args.agent}", json.loads(args.payload))
        _print({"job": job, "results": runtime.tick(conn, pol, llm, worker_id=worker_id)})
        return 0
    if not args.loop:   # worker, one tick
        _print(runtime.tick(conn, pol, llm, worker_id=worker_id))
        return 0
    runtime.serve(lambda: db.connect(), policy_mod.load, lambda c, p: Claude(c, p), worker_id=worker_id,
                  on_output=_print)
    return 0

def _review(args, conn, pol, llm) -> int:
    import tempfile
    import time
    from pathlib import Path

    from .review import gates
    from .review import video as video_review

    store = storage.from_url()
    while True:
        with tempfile.TemporaryDirectory() as tmp:
            if args.dry_run:
                out = []
                for job in gates.pending_jobs(conn, pol, batch=args.batch):
                    d = Path(tmp) / str(job)
                    d.mkdir()
                    out += gates.review_job(conn, pol, llm, store, job, d, role=args.role, write=False)
                for job in video_review.pending_jobs(conn, pol, batch=args.batch):
                    d = Path(tmp) / f"video-{job}"
                    d.mkdir()
                    out += video_review.review_job(conn, pol, llm, store, job, d, write=False)
                _print({"reviews": out, "accepted": sum(r["accepted"] for r in out), "of": len(out),
                        "model": llm.model_for(args.role or pol.get("review", "image", "role")),
                        "llm_usd": llm.spent})
            else:
                result = {"images": gates.run_pending(conn, pol, llm, store, Path(tmp), batch=args.batch),
                          "clips": video_review.run_pending(conn, pol, llm, store, Path(tmp), batch=args.batch)}
                if any(v for r in result.values() for v in r.values()) or not args.loop:
                    _print({**result, "llm_usd": llm.spent})
        if not args.loop:
            return 0
        time.sleep(float(pol.get("review", "image", "loop_seconds", default=120)))
        pol = policy_mod.load()


def brain_steps(conn, pol) -> dict:
    """The Brain's part of each producer tick. Each step is isolated: one that fails is reported (an owner event,
    once a day per step) and the others, and production and publishing after them, still run."""
    from datetime import datetime, timezone

    from . import notify
    from .brain import events, ideas, launches, portfolio, showrunner_loop
    from .deploy.providers import redact

    now = datetime.now(timezone.utc)
    out: dict = {"planned": [], "errors": []}

    def scheduled():
        r = ideas.schedule_ready(conn, pol, now=now)
        return [r] if r["batches"] else []

    def raised():
        r = events.tick(conn, pol)   # breakouts, drops, QA surges: raised once, routed, rate-limited
        return [{"events": r}] if r else []

    def portfolio_cycle():
        if not pol.get("brain", "strategist", "enabled", default=False):
            return []
        portfolio.ensure_daily(conn, pol)          # the daily portfolio cycle (one strategist job a day)
        r = launches.advance(conn, pol)            # approved launches walk the creator lab's steps
        return [{"launches": r}] if any(r.values()) else []

    for name, step in (("schedule", scheduled), ("batches", lambda: showrunner_loop.ensure_batches(conn, pol)),
                       ("events", raised), ("portfolio", portfolio_cycle)):
        try:
            out["planned"] += step()
        except Exception as e:  # noqa: BLE001 - a Brain bug must not stop production or publishing
            logging.getLogger("manyworlds.produce").exception("brain step %s failed", name)
            err = redact(f"{type(e).__name__}: {e}")[:500]
            out["errors"].append({"step": name, "error": err})
            try:
                notify.owner_event(conn, "brain_step_failed", subject={"step": name}, payload={"error": err},
                                   dedupe_key=f"brain_step_failed:{name}:{now.date()}")
            except Exception:  # noqa: BLE001 - reporting must not fail the tick either
                pass
    return out


def produce_tick(conn, pol, llm=None) -> dict:
    """One producer tick. With `operations.mode: paused` (the default) it does nothing: no planning, no
    production, no publishing (manyworlds.operations)."""
    from datetime import datetime as _dt, timezone as _tz

    from . import operations, production
    from .brain import planner
    from .worker import models

    if not operations.is_live(pol):
        return {"operations": "paused"}
    if pol.get("brain", "showrunner", "enabled", default=False):
        # The showrunner plans (brain worker); here: ask for batches, schedule evaluated ones, events, launches.
        steps = brain_steps(conn, pol)
        planned = steps["planned"] + ([{"brain_errors": steps["errors"]}] if steps["errors"] else [])
    else:
        planned = planner.ensure_plans(conn, pol, llm)
    result = {"mirrors": models.ensure_mirrored(conn, pol, storage.from_url()),
              "planned": planned, **production.advance(conn, pol)}
    from .publish import scheduler
    try:
        published = scheduler.publish_due(conn, pol, now=_dt.now(_tz.utc))
    except Exception as e:  # noqa: BLE001 - publishing must never take production down
        published = {"error": f"{type(e).__name__}: {e}"[:300]}
    if any(published.get(k) for k in ("simulated", "published", "paused", "held", "errors", "failed", "error")):
        result["published"] = published
    try:   # real metrics for published posts, at the observation ages (nothing to do until an account exists)
        from .publish import metrics
        collected = metrics.collect(conn, pol)
        if collected["snapshots"] or collected["errors"]:
            result["metrics"] = collected
    except Exception as e:  # noqa: BLE001 - metrics must never stop production or publishing
        result["metrics"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    return result


def _produce(args, conn, pol, llm) -> int:
    import time

    last = None
    while True:
        result = produce_tick(conn, pol, llm)
        paused = result == {"operations": "paused"}
        if not args.loop or (paused and last != "paused") or (not paused and any(result.values())):
            _print(result)
        last = "paused" if paused else "live"
        if not args.loop:
            return 0
        time.sleep(float(pol.get("production", "loop_seconds", default=120)))
        pol = policy_mod.load()


def _creator_id(conn, ref: str):
    row = conn.execute("SELECT id FROM creators WHERE id::text = %s OR handle = %s", (ref, ref)).fetchone()
    if not row:
        raise SystemExit(f"no creator {ref!r}")
    return row["id"]


def _brain(args, conn, pol) -> int:
    import tempfile

    from .brain import discover as disc
    from .creator import lab
    from .llm import Claude
    from .research.apify import ApifySource

    if args.cmd == "creator":
        cid = _creator_id(conn, args.creator)
        c = conn.execute("SELECT handle, status, bible, anchors FROM creators WHERE id = %s", (cid,)).fetchone()
        _print({**c, "jobs": conn.execute("SELECT kind, status, count(*) n FROM jobs WHERE creator_id = %s "
                                           "GROUP BY kind, status", (cid,)).fetchall()})
        return 0
    llm = Claude(conn, pol)
    if args.cmd == "review":
        return _review(args, conn, pol, llm)
    if args.cmd == "produce":
        return _produce(args, conn, pol, llm)
    if args.cmd == "profile":
        from .creator import profile as prof
        cid = _creator_id(conn, args.creator)
        if args.build:
            prof.build(conn, pol, llm, cid)
        for item in args.set:
            path, _, raw = item.partition("=")
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            prof.set_path(conn, pol, cid, path, value, author="human", reason=args.reason)
        cur = prof.current(conn, cid)
        _print({"version": cur[0], "profile": cur[1], "llm_usd": llm.spent} if cur else {"profile": None})
        return 0
    if args.cmd == "discover":
        result = disc.discover(conn, pol, llm, ApifySource(conn, pol))
        _print({**result, "llm_usd": llm.spent})
    elif args.cmd == "choose":
        _print({**disc.choose_thesis(conn, pol, llm), "llm_usd": llm.spent})
    elif args.cmd == "launch-creator":
        if args.thesis:
            t = conn.execute("SELECT id, body FROM theses WHERE id::text = %s", (args.thesis,)).fetchone()
        else:
            t = conn.execute("SELECT id, body FROM theses WHERE status = 'chosen' ORDER BY created_at DESC LIMIT 1").fetchone()
        if not t:
            raise SystemExit("no chosen thesis; run `mw choose` first")
        bible = lab.write_bible(conn, pol, llm, {"id": t["id"], **t["body"]})
        row = lab.create_creator(conn, bible, t["id"], policy_version=pol.version)
        batch = lab.enqueue_reference_pack(conn, pol, row["id"])
        voice_job = lab.enqueue_voice_sample(conn, row["id"])
        _print({"creator": row, "reference_batch": batch, "voice_job": voice_job,
                "bible": bible.model_dump(), "llm_usd": llm.spent})
    elif args.cmd == "anchors":
        cid = _creator_id(conn, args.creator)
        with tempfile.TemporaryDirectory() as tmp:
            _print({**lab.select_anchors(conn, pol, llm, storage.from_url(), cid, tmp), "llm_usd": llm.spent})
    elif args.cmd == "plan":
        from datetime import date

        from .brain import planner
        start = date.fromisoformat(args.start) if args.start else None
        _print({**planner.plan_period(conn, pol, llm, _creator_id(conn, args.creator), start=start, force=args.force),
                "llm_usd": llm.spent})
    elif args.cmd == "train-lora":
        _print({"job": lab.enqueue_lora_training(conn, pol, _creator_id(conn, args.creator))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
