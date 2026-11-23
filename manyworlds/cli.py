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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mw")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate", help="apply schema migrations and register the policy")

    sub.add_parser("serverless-worker", help="run inside a RunPod Serverless endpoint: one queued job per request")
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
    ac = sub.add_parser("account", help="a creator's Instagram account: its setup sheet, or connect it once made")
    ac.add_argument("action", choices=["sheet", "connect"])
    ac.add_argument("creator")
    ac.add_argument("--username", help="the account's Instagram username (Unified Login, the system user's token)")
    ac.add_argument("--token-env", help="Instagram Login instead: the env variable holding this account's token")
    cm = sub.add_parser("comments", help="what a creator did with recent comments on its posts")
    cm.add_argument("creator")
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
    tr = sub.add_parser("trends", help="trend watcher: what works on TikTok and Instagram per niche (policy trends)")
    tr.add_argument("action", choices=["tick", "loop", "brief", "spend"])
    tr.add_argument("market", nargs="?", help="for brief: the niche name (policy trends.markets)")
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

    te = sub.add_parser("test", help="owner-approved tests: the breadth test (policy/test_overlay.yaml)")
    tsub = te.add_subparsers(dest="test_cmd", required=True)
    tb = tsub.add_parser("breadth", help="make a breadth plan's items (resumes the same run); --dry-run creates nothing")
    tb.add_argument("--plan", default="default", help="breadth.plans.<name> in the test overlay")
    tb.add_argument("--items", help="comma-separated item ids (default: every item)")
    tb.add_argument("--dry-run", action="store_true", help="print what it would make and an estimate; create nothing")
    tb.add_argument("--run", help="the run id to use or resume (default: derived from the plan and items)")
    tb.add_argument("--creator", help="creator handle (default: the plan's)")
    tb.add_argument("--hours", type=float, help="deadline from the run's first post (default breadth.deadline_hours)")
    tb.add_argument("--once", action="store_true", help="one pass, then exit (run again to continue)")
    tr2 = tsub.add_parser("breadth-report", help="a breadth run: per item and capability, gates, lip sync, WER, file, $")
    tr2.add_argument("run")
    tr2.add_argument("--no-probe", action="store_true", help="skip downloading the final videos to read fps and size")

    vo = sub.add_parser("voice", help="a creator's designed voice: design candidates, list them to listen, pick one")
    vsub = vo.add_subparsers(dest="voice_cmd", required=True)
    vd = vsub.add_parser("design", help="queue voice candidates from the bible (a test batch: runs while paused)")
    vd.add_argument("creator")
    vd.add_argument("--pools", help="comma-separated GPU pools to pin the job to, e.g. runpod_secure_l40s_vol")
    vc = vsub.add_parser("candidates", help="the latest design job's candidates: scores and links to listen")
    vc.add_argument("creator")
    vp = vsub.add_parser("pick", help="choose the voice from the latest finished design (automatic, or --candidate N)")
    vp.add_argument("creator")
    vp.add_argument("--candidate", type=int, help="this candidate index instead of the automatic choice")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    pol = policy_mod.load()

    if args.cmd == "serverless-worker":
        from .worker import serverless_entry
        serverless_entry.main()
        return 0

    if args.cmd == "worker":
        from .worker import runner
        kinds = ([k.strip() for k in args.kinds.split(",") if k.strip()] if args.kinds
                 else runner.kinds_from_env(pol))
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
        cid = _creator_id(conn, args.creator)
        if args.action == "sheet":
            _print(instagram.setup_sheet(conn, pol, cid))
        else:
            try:
                _print(instagram.connect(conn, pol, cid, username=args.username, token_env=args.token_env))
            except instagram.AccountNotConnected as e:
                raise SystemExit(str(e)) from None
    elif args.cmd == "comments":
        from .publish import comments
        _print(comments.recent(conn, _creator_id(conn, args.creator)))
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
    elif args.cmd == "test":
        return _test(args, conn, pol)
    elif args.cmd == "voice":
        return _voice(args, conn, pol)
    elif args.cmd in ("discover", "choose", "launch-creator", "anchors", "train-lora", "creator", "review", "plan", "profile", "produce", "trends"):
        return _brain(args, conn, pol)
    return 0


def _test(args, conn, pol) -> int:
    """`mw test breadth` / `mw test breadth-report` (manyworlds.breadth)."""
    from . import breadth
    store = None if getattr(args, "dry_run", False) else storage.from_url()
    if args.test_cmd == "breadth-report":
        _print(breadth.report(conn, pol, args.run, store=store, probe=not args.no_probe))
        return 0
    llm = None
    tpol = breadth.test_policy(pol)
    if not args.dry_run and breadth._cfg(tpol).get("review_in_driver", True):
        from .llm import Claude
        llm = Claude(conn, tpol)
    try:
        out = breadth.run(conn, pol, args.plan, items=[i.strip() for i in args.items.split(",") if i.strip()]
                          if args.items else None, run=args.run, creator=args.creator, store=store, llm=llm,
                          dry_run=args.dry_run, once=args.once, hours=args.hours)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    _print({**out, **({"llm_usd": llm.spent} if llm is not None else {})})
    return 0


def voice_candidates(conn, pol, creator_id, store=None) -> dict:
    """The latest voice design job of a creator and its candidates (creator.voice_lab.candidates: WER, pitch variance,
    distance from the nearest other creator's voice, whether each qualifies for the automatic pick, best first), which
    one is the current voice, and a presigned link to listen to each."""
    from .creator import voice_lab
    job = conn.execute("SELECT id, status, batch FROM jobs WHERE creator_id = %s AND kind = %s "
                       "ORDER BY created_at DESC LIMIT 1", (creator_id, voice_lab._kind(pol, "design"))).fetchone()
    if not job:
        return {"job": None, "candidates": [], "note": "no design job yet: mw voice design <creator>"}
    current = (conn.execute("SELECT anchors FROM creators WHERE id = %s", (creator_id,)).fetchone()["anchors"]
               or {}).get("voice") or {}
    out = [{**c, "current": current.get("ref_uri") == c["uri"],
            "url": store.public_url(c["uri"]) if store is not None else c["uri"]}
           for c in voice_lab.candidates(conn, pol, job["id"])]
    return {"job": str(job["id"]), "status": job["status"], "batch": job["batch"], "candidates": out,
            **({} if job["status"] == "done" else {"note": f"the design job is {job['status']}: candidates once done"})}


def _voice(args, conn, pol) -> int:
    """`mw voice design|candidates|pick <creator>` (manyworlds.creator.voice_lab)."""
    from .creator import voice_lab
    cid = _creator_id(conn, args.creator)
    if args.voice_cmd == "design":
        handle = conn.execute("SELECT handle FROM creators WHERE id = %s", (cid,)).fetchone()["handle"]
        # a test batch, so it runs while operations are paused; urgent, so no machine waits to batch it
        batch = f"test-voice-{handle}"
        pools = [p.strip() for p in args.pools.split(",") if p.strip()] if args.pools else None
        try:
            job = voice_lab.design(conn, pol, cid, batch=batch, pools=pools)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        prio = int(pol.get("autoscale", "batch", "urgent_priority", default=1))
        conn.execute("UPDATE jobs SET priority = greatest(priority, %s) WHERE id = %s AND status = 'queued'",
                     (prio, job))
        row = conn.execute("SELECT status, batch, pools FROM jobs WHERE id = %s", (job,)).fetchone()
        _print({"job": str(job), **row, "next": f"mw voice candidates {handle}, then mw voice pick {handle}"})
        return 0
    if args.voice_cmd == "candidates":
        _print(voice_candidates(conn, pol, cid, storage.from_url()))
        return 0
    job = conn.execute("SELECT id FROM jobs WHERE creator_id = %s AND kind = %s AND status = 'done' "
                       "ORDER BY created_at DESC LIMIT 1", (cid, voice_lab._kind(pol, "design"))).fetchone()
    if not job:
        raise SystemExit("no finished voice design job: mw voice design <creator>, then wait for it")
    try:   # --candidate: the owner's choice after listening (never one too close to another creator's voice)
        _print(voice_lab.pick(conn, pol, cid, job["id"], candidate=args.candidate))
    except ValueError as e:
        raise SystemExit(str(e)) from None
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
        # A full batch means more is waiting: go again at once (at scale a fixed sleep caps throughput).
        full = not args.dry_run and (
            result["images"].get("jobs", 0) >= int(pol.get("review", "image", "max_jobs_per_run", default=10)) or
            result["clips"].get("jobs", 0) >= int(pol.get("review", "video", "max_jobs_per_run", default=10)))
        if not full:
            time.sleep(float(pol.get("review", "image", "loop_seconds", default=120)))
        pol = policy_mod.load()


def brain_steps(conn, pol, llm=None) -> dict:
    """The Brain's part of each producer tick. Each step is isolated: one that fails is reported (an owner event,
    once a day per step) and the others, and production and publishing after them, still run. With an `llm`, newly scheduled posts are
    directed (shot sheets, manyworlds.brain.director)."""
    from datetime import datetime, timezone

    from . import notify
    from .brain import events, ideas, launches, portfolio, showrunner_loop
    from .deploy.providers import redact

    now = datetime.now(timezone.utc)
    out: dict = {"planned": [], "errors": []}

    def scheduled():
        r = ideas.schedule_ready(conn, pol, now=now, llm=llm)
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
        steps = brain_steps(conn, pol, llm)
        planned = steps["planned"] + ([{"brain_errors": steps["errors"]}] if steps["errors"] else [])
    else:
        planned = planner.ensure_plans(conn, pol, llm)
    from .creator import sets
    try:   # plates reviewed since the last tick become references before this tick starts any scene
        plates = sets.refresh_all(conn, pol)
    except Exception as e:  # noqa: BLE001 - a scene without its plate is made from the canonical text
        plates = {"error": f"{type(e).__name__}: {e}"[:300]}
    result = {"mirrors": models.ensure_mirrored(conn, pol, storage.from_url()),
              "planned": planned, **({"plates": plates} if plates else {}), **production.advance(conn, pol, llm=llm)}
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
    try:   # comments on our posts, answered in the creator's voice (autonomy.comment_replies; off by default)
        from .publish import comments
        seen, answered = comments.collect(conn, pol), comments.respond(conn, pol, llm)
        if seen.get("new") or seen.get("errors") or any(v for k, v in answered.items() if k != "skipped"):
            result["comments"] = {**seen, **answered}
    except Exception as e:  # noqa: BLE001 - comments must never stop production or publishing
        result["comments"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    return result


def _trends(args, conn, pol, llm) -> int:
    """The trend watcher (manyworlds.research.trends): one tick, or a loop every trends.loop_minutes."""
    import time

    from .research import trends
    from .research.scrapecreators import ScrapeCreators
    src = ScrapeCreators()
    while True:
        try:
            result = trends.tick(conn, pol, llm, src)
        except Exception as e:  # noqa: BLE001 - the loop keeps going; the error is printed
            result = {"error": f"{type(e).__name__}: {e}"[:300]}
        _print(result)
        if args.action != "loop":
            return 0
        time.sleep(60 * float(pol.get("trends", "loop_minutes", default=30)))
        pol = policy_mod.load()

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
    if args.cmd == "trends" and args.action in ("brief", "spend"):
        from datetime import datetime as _d, timezone as _z
        from .research import trends
        _print(trends.brief(conn, args.market) if args.action == "brief" else
               {"spent_today": trends.spent_today(conn, _d.now(_z.utc)), "max": pol.get("trends", "daily_usd_max")})
        return 0
    llm = Claude(conn, pol)
    if args.cmd == "review":
        return _review(args, conn, pol, llm)
    if args.cmd == "produce":
        return _produce(args, conn, pol, llm)
    if args.cmd == "trends":
        return _trends(args, conn, pol, llm)
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
        row = lab.create_creator(conn, bible, t["id"], policy_version=pol.version,
                                 voice_blend=pol.get("creator_lab", "voice_blend"))
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
