"""`mw` command line: migrate, run workers, enqueue work, benchmark and report."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import psycopg

from . import bench, db, ledger, logs, policy as policy_mod, queue, storage, wakeup


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _music(conn, pol, args) -> dict:
    """`mw music measure [--batch B]`: measure unmeasured songs now (a batch's are registered first);
    `mw music songs --creator H`: an artist's songs with their features; `mw music analyse [--creator H | --genre G]
    [--record]`: what goes with outcomes (brain.music_analysis), in the artist's mode (else real)."""
    from . import songs
    from .brain import music_analysis
    cid = None
    if args.creator:
        row = conn.execute("SELECT id FROM creators WHERE handle = %s", (args.creator,)).fetchone()
        if row is None:
            raise SystemExit(f"no creator {args.creator!r}")
        cid = row["id"]
    if args.action == "sandbox":
        from . import music_sandbox
        from .llm import Claude
        if not args.handle:
            raise SystemExit("mw music sandbox <handle> --songs N")
        if args.create:
            music_sandbox.create_artist(conn, pol, args.handle, genre=args.genre or "indie")
        try:
            return music_sandbox.run(conn, pol, Claude(conn, pol), args.handle, songs_n=args.songs, usd=args.usd,
                                     run_id=args.run)
        except music_sandbox.NotSandbox as e:
            raise SystemExit(str(e)) from None
    if args.action == "measure":
        if args.batch:
            songs.register_batch(conn, args.batch)
        inline = bool(getattr(args, "inline", False))
        out = songs.measure_pending(conn, pol, storage.from_url() if inline else None, limit=1000, inline=inline)
        if not inline:   # queued on the finisher: the autoscaler starts it; the producer's tick stores the results
            out["note"] = ("finish.measure_song jobs queued for the finisher; run `mw music measure` again (or wait "
                           "for the producer's tick) to collect them, `mw music songs` to see the features")
        return out
    if args.action == "songs":
        rows = conn.execute("SELECT id, title, version, genre, created_at, measure_error FROM songs "
                            "WHERE %s::uuid IS NULL OR creator_id = %s ORDER BY created_at", (cid, cid)).fetchall()
        feats = songs.latest_features(conn, [r["id"] for r in rows])
        return {"songs": [{**r, "features": feats.get(str(r["id"]))} for r in rows]}
    scope = "creator" if cid else "genre" if args.genre else "portfolio"
    result = music_analysis.analyse(conn, pol, scope=scope, creator_id=cid, genre=args.genre)
    if args.record:
        result["finding_ids"] = music_analysis.record(conn, pol, result)
    return result


def _open_run(group: str | None, connect=None) -> bool:
    """Whether `group` is a registered machine run that is not stopped (gpu_groups): the finisher works only then."""
    if not group:
        return False
    conn = (connect or db.connect)()
    try:
        row = conn.execute("SELECT stopped_at FROM gpu_groups WHERE name = %s", (group,)).fetchone()
    finally:
        conn.close()
    return row is not None and row["stopped_at"] is None


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
    vf = sub.add_parser("verify", help="live end-to-end checks on real data (manyworlds.verify): did each stage really do its work")
    vf.add_argument("--alert", action="store_true", help="also tell the owner about failing checks")
    ib = sub.add_parser("inbox", help="what needs the owner: alerts not yet marked done, pending approvals, requests")
    ib.add_argument("--done", nargs="*", type=int, metavar="ID", help="mark these alert ids done (no ids: all of them)")
    ib.add_argument("--type", help="with --done: only alerts of this type")
    ib.add_argument("--limit", type=int, default=50)
    wr = sub.add_parser("weights-reuse", help="per machine on a weights volume: what it held at boot, what its jobs reused or downloaded")
    wr.add_argument("--hours", type=float, default=72)
    ec = sub.add_parser("economics", help="the bill split: production, development (test/bench), research, agency, idle machines")
    ec.add_argument("--days", type=float, default=30)
    bg = sub.add_parser("budget", help="today's caps, spend and grants; `grant --usd X --reason R`: a one-day owner exception")
    bgs = bg.add_subparsers(dest="budget_cmd")
    bgg = bgs.add_parser("grant", help="raise today's caps (UTC day) by --usd; the policy is unchanged")
    bgg.add_argument("--usd", type=float, required=True)
    bgg.add_argument("--reason", required=True)
    mc = sub.add_parser("machines", help="each GPU machine: billed vs finished work; per family and pool: boot/busy/idle $, "
                                         "first-pass rate, $ per accepted output and video second, accepted per GPU-hour")
    mc.add_argument("--hours", type=float, default=72)
    wc = sub.add_parser("worker-credentials",
                        help="create or rotate the GPU machines' scoped database login; prints "
                             "WORKER_DATABASE_URL=... for piping into `fly secrets import` (never into a terminal)")
    wc.add_argument("--force", action="store_true", help="rotate even while GPU machines are up (they lose access)")
    lm = sub.add_parser("llm", help="LLM models: compare candidates with a role's model, choose the best per job")
    lm.add_argument("action", choices=["shadow-report", "recommend", "judge", "batches"],
                    help="shadow-report: the comparisons; recommend: which candidate qualifies (--propose files it as "
                         "an approval); judge: score a generative role's answers blind (spends up to --usd); "
                         "batches: the brain's call journal and Message Batches (what batching saved)")
    lm.add_argument("role", nargs="?", default="all", help="a role, or `all` for recommend")
    lm.add_argument("--propose", action="store_true")
    lm.add_argument("--usd", type=float, default=1.0, help="judge: most to spend")
    lm.add_argument("--days", type=float, default=30)
    lm.add_argument("--tolerance", type=float, default=1.0, help="numeric difference counted as agreeing")
    mu = sub.add_parser("music", help="songs: measure them, list an artist's songs, analyse what works")
    mu.add_argument("action", choices=["measure", "songs", "analyse", "sandbox"])
    mu.add_argument("handle", nargs="?", help="sandbox: the sandbox artist's handle (a real creator is refused)")
    mu.add_argument("--songs", type=int, help="sandbox: songs to make (default music.sandbox.default_songs)")
    mu.add_argument("--usd", type=float, help="sandbox: the run's LLM cap (at most music.sandbox.max_usd)")
    mu.add_argument("--create", action="store_true", help="sandbox: create the sandbox artist first (--genre)")
    mu.add_argument("--run", help="sandbox: resume this run id (its recorded calls are reused, not paid again)")
    mu.add_argument("--inline", action="store_true", help="measure: measure here instead of queueing finish.measure_song")
    mu.add_argument("--batch", help="measure: a benchmark batch's songs (registered first)")
    mu.add_argument("--creator", help="songs / analyse: the artist's handle")
    mu.add_argument("--genre", help="analyse: one genre")
    mu.add_argument("--record", action="store_true", help="analyse: record the findings (domain music)")
    sub.add_parser("reap", help="mark jobs dead whose final lease expired")
    rs = sub.add_parser("research", help="research references (named public creators): forget one on request, purge, "
                                         "patterns, pilot (the influencer intelligence agent: scout, annotate, distill)")
    rs.add_argument("action", choices=["forget", "purge", "patterns", "feedback", "pilot", "monitor", "data-audit", "graph-check",
                                       "vision-bench", "spine", "gold", "agreement", "ask", "answer", "gaps",
                                       "spot-check", "budget", "propositions"])
    rs.add_argument("--apply", action="store_true", help="data-audit: clean what the data policy no longer allows "
                                                         "(contact details in bios, caption features backfilled)")
    rs.add_argument("who", nargs="?", help="forget: a handle (@ optional), profile URL or one of their post URLs")
    rs.add_argument("--seed", action="store_true", help="patterns: also add the seed file's cards to pattern_cards")
    rs.add_argument("--hypothesis", help="pilot: a thesis id or the hypothesis in words")
    rs.add_argument("--usd", type=float, help="pilot: the run's spend cap (default intelligence.agent.max_usd_per_run)")
    rs.add_argument("--handles", nargs="*", default=[], help="pilot: named reference creators (instagram:<handle>, "
                                                             "tiktok:<handle> or @handle on every platform)")
    rs.add_argument("--hashtags", nargs="*", default=[], help="pilot: hashtags for the candidate pool (else one plan call)")
    rs.add_argument("--platforms", nargs="*", default=[], help="pilot: instagram, tiktok (default intelligence.agent.platforms)")
    rs.add_argument("--creators", type=int, help="pilot: creators in the sample")
    rs.add_argument("--posts", type=int, help="pilot: posts per creator")
    rs.add_argument("--reel-views", action="store_true", help="pilot: Instagram posts from the post scraper (reel "
                                                              "views; a billed item a post) instead of the profile's")
    rs.add_argument("--resume-annotate", metavar="BATCH", help="pilot: annotate and distill the posts this pilot batch "
                                                                 "already collected; pulls nothing")
    rs.add_argument("--dry-run", action="store_true", help="pilot, monitor: print the estimate, queue nothing")
    rs.add_argument("--estimate", type=int, help="monitor: $/month for a universe of N creators at the policy tier mix")
    rs.add_argument("--assets", type=int, help="vision-bench: accepted image assets of the sandbox creator "
                                               "(default intelligence.vision_bench.assets)")
    rs.add_argument("--models", help="vision-bench: candidates, comma separated (default intelligence.vision_bench.models)")
    rs.add_argument("--creator", help="vision-bench: one sandbox creator's handle (default every sandbox creator)")
    rs.add_argument("--question", help="ask: the research question in words (data, web or both: --kind)")
    rs.add_argument("--kind", default="data", choices=["data", "web", "both"], help="ask: how to answer it")
    rs.add_argument("--labels", nargs="*", default=[], help="spot-check: attribute=value pairs for the unit in `who`")
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
    brq = bsub.add_parser("requests", help="requests between the agency's roles; the owner's, or decide one")
    brq.add_argument("--role", default="owner", help="whose queue to list (owner, strategist, analyst...)")
    brq.add_argument("--decide")
    brq_d = brq.add_mutually_exclusive_group()
    brq_d.add_argument("--accept", action="store_true")
    brq_d.add_argument("--reject", action="store_true")
    brq.add_argument("--reason", default="")
    bsc = bsub.add_parser("scorecard", help="how good each role's predictions were, per month (real unless --sandbox)")
    bsc.add_argument("--sandbox", action="store_true")
    bsc.add_argument("--role")
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
    bpm = bsub.add_parser("portfolio-map", help="archetypes x audiences x commercial categories: coverage, overlaps, "
                          "gaps (code, no model)")
    bpm.add_argument("--text", action="store_true", help="the CEO's view, as lines")
    bcc = bsub.add_parser("capability-card", help="what production makes reliably and cheaply, per post format")
    bcc.add_argument("--refresh", action="store_true", help="measure it again from posts and the ledger, and store it")
    bco = bsub.add_parser("commerce", help="the commercial category registry (estimates until researched)")
    bco.add_argument("--seed", action="store_true", help="insert or update the policy seed (research rows kept)")
    bca = bsub.add_parser("casting", help="cast a thesis as a sandbox test: concepts, judge, auditions (a mini-feed "
                          "each, test batches), the winner; launches nothing. --dry-run: the estimate only")
    bca.add_argument("thesis")
    bca.add_argument("--dry-run", action="store_true")
    bca.add_argument("--once", action="store_true", help="one step, then stop (run again to resume)")
    bca.add_argument("--hours", type=float, default=None, help="stop driving after this long (default: the "
                     "audition deadline)")

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
    tb.add_argument("--queue", action="store_true",
                    help="queue the driver as a durable brain job (survives deploys) instead of looping in this process")
    tr2 = tsub.add_parser("breadth-report", help="a breadth run: per item and capability, gates, lip sync, WER, file, $")
    tr2.add_argument("run")
    tr2.add_argument("--no-probe", action="store_true", help="skip downloading the final videos to read fps and size")

    vo = sub.add_parser("voice", help="a creator's designed voice: cast it automatically, its status and quality")
    vsub = vo.add_subparsers(dest="voice_cmd", required=True)
    vcast = vsub.add_parser("cast", help="cast her voice automatically: candidates from the bible, judged by the "
                                         "worker, the best stored once the job finishes (a test batch: runs while "
                                         "paused)")
    vcast.add_argument("creator")
    vcast.add_argument("--engine", help="qwen3 | indextts (default: the bake-off's winner, else policy)")
    vcast.add_argument("--pools", help="comma-separated GPU pools to pin the job to")
    vcast.add_argument("--attempt", type=int, default=0, help="a new attempt draws new candidates")
    vcs = vsub.add_parser("cast-status", help="her latest cast: each candidate's scores and which one was stored")
    vcs.add_argument("creator")
    vq = vsub.add_parser("quality", help="voice lines' quality per accepted minute, per engine (the line check)")
    vq.add_argument("creator")
    vq.add_argument("--days", type=int, default=30)
    vbo = vsub.add_parser("bakeoff-decide", help="score a finished voice bake-off run and record the default engine")
    vbo.add_argument("run")

    pf2 = sub.add_parser("performance", help="performance direction: requested vs observed levels, judge agreement, "
                                             "calibration state (manyworlds.performance)")
    pfsub = pf2.add_subparsers(dest="performance_cmd", required=True)
    pfr = pfsub.add_parser("report", help="requested vs observed per model, target, control and level; why "
                                          "calibration does or doesn't adjust")
    pfr.add_argument("--creator", help="one creator's observations only (handle or id)")

    pc = sub.add_parser("policy", help="runtime policy overrides (manyworlds.brain.overrides): list them, or roll one "
                                       "back")
    pcsub = pc.add_subparsers(dest="policy_cmd", required=True)
    pco = pcsub.add_parser("overrides", help="the latest overrides, newest first (id, path, value, version, author, "
                                             "reason)")
    pco.add_argument("--path", help="one path only")
    pco.add_argument("--limit", type=int, default=50)
    pcr = pcsub.add_parser("revert", help="restore the value a path had before an override, as a new override "
                                          "(within brain.authority; beyond it, an approval for the owner)")
    pcr.add_argument("override_id", type=int, help="the override's id (mw policy overrides)")
    pcr.add_argument("--reason", required=True, help="why it is rolled back (recorded on the new override)")
    pcr.add_argument("--author", default="owner", help="who rolls it back (recorded; default owner)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pol = policy_mod.load()
    except Exception:
        logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
        raise
    # JSON lines on Fly (policy logging.json or MW_LOG_JSON=1), plain text otherwise; a no-op when logging is
    # already set up (tests).
    role = "-".join(str(x) for x in (args.cmd, getattr(args, "brain_cmd", None)) if x)
    logs.configure(logs.json_wanted(pol), role=role)

    if args.cmd == "serverless-worker":
        from .worker import serverless_entry
        serverless_entry.main()
        return 0

    if args.cmd == "worker":
        from .worker import runner
        kinds = ([k.strip() for k in args.kinds.split(",") if k.strip()] if args.kinds
                 else runner.kinds_from_env(pol))
        pool = args.pool or os.environ.get("MW_POOL", "local")
        idle_exit, self_destroy = args.idle_exit, None
        info = pol.get("gpu_pools", pool, default={}) or {}
        if info.get("provider") == "fly":
            # A machine of this Fly app the autoscaler started (the finisher): its run's name is in the machine's
            # metadata, and it stops its own machine at its deadline. It exits once its queue has been empty for the
            # pool's idle_exit_seconds; the machine then stops and stays stopped (fly.toml restarts it only on failure).
            from .deploy.providers import FlyMachines, ProviderError
            fly = FlyMachines()
            group = os.environ.get("MW_GROUP")
            if not group:
                try:
                    group = fly.own_group()
                except ProviderError as e:
                    logging.getLogger("manyworlds.worker").warning("could not read this machine's run: %s", e)
            if not _open_run(group):
                # Started by something other than the autoscaler (a deploy), or for a run already closed: a clean
                # exit, so the machine stops and is not restarted (the enforcer stops it too).
                # Stay up a moment first: `fly deploy` counts a machine that exits at once as a failed start and
                # stops the rollout of every other process group (2026-11-23). It stops and stays stopped after.
                grace = float(info.get("deploy_grace_seconds") or 0)
                print(f"no open run for this machine ({group or 'none'}): not working pool {pool}; "
                      f"stopping in {grace:.0f} s")
                time.sleep(grace)
                return 0
            os.environ["MW_GROUP"] = group
            self_destroy = fly.stop_self
            if idle_exit is None and info.get("idle_exit_seconds") is not None:
                idle_exit = float(info["idle_exit_seconds"])
        worker = runner.start(kinds, pool=pool, policy=pol, store=storage.from_url(), self_destroy=self_destroy)
        n = worker.run(max_jobs=args.max_jobs, idle_exit_seconds=idle_exit)
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
        _print({**metrics.collect(conn, pol), "account_insights": metrics.collect_account(conn, pol)})
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
        from . import notify
        _print({"inbox": notify.counts(conn), "jobs": queue.counts(conn), "spend": ledger.totals(conn),
                "cost_per_output": ledger.cost_per_output(conn), "brain_costs": ledger.brain_costs(conn),
                "machines_72h": report.summary(report.machines(conn)), "throughput_72h": report.throughput(conn)})
    elif args.cmd == "inbox":
        from . import notify
        if args.done is not None:
            n = notify.mark_done(conn, args.done or None, type=args.type)
            _print({"marked_done": n, **notify.counts(conn)})
        else:
            _print(notify.inbox(conn, limit=args.limit))
    elif args.cmd == "weights-reuse":
        from .deploy import report as machine_report
        _print(machine_report.weights_reuse(conn, hours=args.hours))
    elif args.cmd == "economics":
        from . import economics
        _print(economics.report(conn, pol, days=args.days))
    elif args.cmd == "budget":
        from . import budget
        if args.budget_cmd == "grant":
            _print(budget.grant(conn, args.usd, args.reason))
        _print(budget.status(conn, pol))
    elif args.cmd == "machines":
        from .deploy import report
        rows = report.machines(conn, hours=args.hours)
        _print({"summary": report.summary(rows), "throughput": report.throughput(conn, hours=args.hours),
                "machines": rows})
    elif args.cmd == "posts":
        cid = _creator_id(conn, args.creator)
        _print(conn.execute("SELECT id, status, format, platform, scheduled_for, caption, qa_status, "
                            "body->'scenes' AS scenes, body->'images' AS images, body->>'drop_reason' AS drop_reason "
                            "FROM posts WHERE creator_id = %s ORDER BY scheduled_for", (cid,)).fetchall())
    elif args.cmd == "music":
        _print(_music(conn, pol, args))
    elif args.cmd == "reap":
        print(f"reaped {queue.reap(conn)} jobs")
    elif args.cmd == "llm":
        from . import selection, shadow
        if args.action == "batches":
            from . import llm_batch
            _print(llm_batch.status(conn, pol))
        elif args.action == "shadow-report":
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
    elif args.cmd == "performance":
        return _performance(args, conn, pol)
    elif args.cmd == "policy":
        return _policy_overrides(args, conn, pol)
    elif args.cmd == "verify":
        from . import verify
        got = verify.run(conn, pol)
        if args.alert:
            got["alerted"] = verify.alert(conn, pol, got)
        _print(got)
        return 1 if got["failing"] else 0
    elif args.cmd == "research":
        return _research(args, conn, pol)
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
    if getattr(args, "queue", False) and not args.dry_run:   # the brain worker drives it, step by step
        try:
            _print(breadth.enqueue_driver(conn, pol, args.plan, items=[i.strip() for i in args.items.split(",")
                                                                     if i.strip()] if args.items else None,
                                          run=args.run, creator=args.creator, hours=args.hours))
        except ValueError as e:
            raise SystemExit(str(e)) from None
        return 0
    llm = None
    tpol = breadth.test_policy(pol)
    if not args.dry_run and breadth._cfg(tpol).get("review_in_driver", True):
        from .llm import Claude
        llm = Claude(conn, tpol)
    # A long unattended run outlives database connections (2026-11-23: Neon dropped the driver's after ~90 min):
    # reconnect and resume the same run, after its lease has lapsed, up to breadth.max_reconnects times.
    cfg = breadth._cfg(tpol)
    reconnects = 0
    while True:
        try:
            out = breadth.run(conn, pol, args.plan, items=[i.strip() for i in args.items.split(",") if i.strip()]
                              if args.items else None, run=args.run, creator=args.creator, store=store, llm=llm,
                              dry_run=args.dry_run, once=args.once, hours=args.hours)
            break
        except ValueError as e:
            raise SystemExit(str(e)) from None
        except (psycopg.OperationalError, psycopg.InterfaceError):
            reconnects += 1
            if args.dry_run or reconnects > int(cfg.get("max_reconnects", 20)):
                raise
            logging.getLogger("manyworlds.breadth").exception(
                "breadth driver lost its database connection; reconnecting (%d)", reconnects)
            time.sleep(float(cfg.get("loop_seconds", 60)) * 3)   # the dead driver's lease lapses first
            conn = db.connect()
            if llm is not None:
                from .llm import Claude
                spent, llm = llm.spent, Claude(conn, tpol)
                llm.spent = spent
    _print({**out, **({"llm_usd": llm.spent} if llm is not None else {})})
    return 0


def _policy_overrides(args, conn, pol) -> int:
    """`mw policy overrides [--path P]`, `mw policy revert <override-id> --reason R [--author A]`
    (manyworlds.brain.overrides.revert)."""
    from .brain import overrides
    if args.policy_cmd == "overrides":
        _print(conn.execute("SELECT id, path, value, version, author, reason, decision_id, created_at "
                            "FROM policy_overrides WHERE %s::text IS NULL OR path = %s ORDER BY version DESC LIMIT %s",
                            (args.path, args.path, args.limit)).fetchall())
        return 0
    if args.policy_cmd == "revert":
        try:
            _print(overrides.revert(conn, pol, args.override_id, author=args.author, reason=args.reason))
        except ValueError as e:
            raise SystemExit(str(e)) from None
        return 0
    raise SystemExit(f"unknown policy command {args.policy_cmd!r}")


def _performance(args, conn, pol) -> int:
    """`mw performance report [--creator H]` (manyworlds.performance.report)."""
    from . import performance
    if args.performance_cmd == "report":
        cid = _creator_id(conn, args.creator) if args.creator else None
        _print(performance.report(conn, pol, creator_id=cid))
        return 0
    raise SystemExit(f"unknown performance command {args.performance_cmd!r}")


def _voice(args, conn, pol) -> int:
    """`mw voice cast|cast-status|quality <creator>`, `mw voice bakeoff-decide <run>`
    (manyworlds.creator.voice_lab, manyworlds.voice_bakeoff)."""
    from .creator import voice_lab
    if args.voice_cmd == "bakeoff-decide":
        from . import voice_bakeoff
        _print(voice_bakeoff.decide(conn, pol, args.run))
        return 0
    cid = _creator_id(conn, args.creator)
    if args.voice_cmd == "cast":
        handle = conn.execute("SELECT handle FROM creators WHERE id = %s", (cid,)).fetchone()["handle"]
        pools = [p.strip() for p in args.pools.split(",") if p.strip()] if args.pools else None
        try:
            job = voice_lab.cast(conn, pol, cid, engine=args.engine, attempt=args.attempt,
                                 batch=f"test-voice-{handle}", pools=pools)
        except ValueError as e:
            raise SystemExit(str(e)) from None
        prio = int(pol.get("autoscale", "batch", "urgent_priority", default=1))
        conn.execute("UPDATE jobs SET priority = greatest(priority, %s) WHERE id = %s AND status = 'queued'",
                     (prio, job))
        row = conn.execute("SELECT status, batch, pools, payload->>'engine' AS engine FROM jobs WHERE id = %s",
                           (job,)).fetchone()
        _print({"job": str(job), **row, "next": "stored automatically once it finishes (the producer settles "
                                                 f"casts); mw voice cast-status {handle}"})
        return 0
    if args.voice_cmd == "cast-status":
        _print(voice_lab.cast_status(conn, pol, cid))
        return 0
    if args.voice_cmd == "quality":
        _print(voice_lab.quality_stats(conn, pol, cid, days=args.days))
        return 0
    raise SystemExit(f"unknown voice command {args.voice_cmd!r}")


def _agency(args, conn, pol) -> int:
    """`mw brain portfolio-map|capability-card|commerce|casting` (spec 2026-11-25-talent-agency)."""
    from .brain import capability_card, casting, commerce, portfolio_map
    if args.brain_cmd == "portfolio-map":
        m = portfolio_map.build(conn, pol)
        print(portfolio_map.text(m)) if args.text else _print(m)
        return 0
    if args.brain_cmd == "capability-card":
        _print(capability_card.refresh(conn, pol) if args.refresh else capability_card.card(conn, pol))
        return 0
    if args.brain_cmd == "commerce":
        if args.seed:
            commerce.seed(conn, pol)
        _print([{k: r.get(k) for k in ("key", "name", *commerce.BANDS, "index", "source", "evidence")}
                for r in commerce.categories(conn, pol)])
        return 0
    thesis = casting.thesis(conn, args.thesis)
    if args.dry_run:
        _print({"thesis": thesis["id"], "dry_run": True, **casting.estimate(pol)})
        return 0
    return _casting_drive(conn, pol, thesis["id"], once=args.once, hours=args.hours)


def _casting_drive(conn, pol, thesis_id, *, once: bool, hours: float | None) -> int:
    """The sandbox casting: the model steps inline, the renders in test batches (they run while operations are
    paused), the image review in the driver; loops until a winner is picked or the time is up. Launches nothing."""
    import tempfile
    import time
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    from .brain import casting
    from .llm import Claude
    from .review import gates
    llm, store = Claude(conn, pol), storage.from_url()
    run = f"t{str(thesis_id)[:8]}"
    casting.cast(conn, pol, llm, thesis_id, run=run)
    acfg = casting._acfg(pol)
    until = datetime.now(timezone.utc) + timedelta(hours=float(hours or acfg.get("deadline_hours", 24)) + 1)
    loop_s = float(pol.get("brain", "runtime", "loop_seconds", default=30))
    while True:
        with tempfile.TemporaryDirectory() as tmp:
            moved = casting.advance(conn, pol, run=run, llm=llm, store=store, workdir=tmp)
            for r in conn.execute("SELECT run, arm FROM auditions WHERE run = %s AND stage = 'feed'", (run,)).fetchall():
                d = Path(tmp) / r["arm"]
                d.mkdir(exist_ok=True)
                gates.run_pending(conn, pol, llm, store, d, batch=casting.batch(pol, r["run"], r["arm"]))
        picked = casting.pick(conn, pol, run)
        if once or picked is not None or datetime.now(timezone.utc) > until:
            break
        logging.getLogger("manyworlds.casting").info("casting %s: %s", run, moved)
        time.sleep(loop_s)
    _print({**casting.summary(conn, run), "picked": picked, "llm_usd": llm.spent,
            "note": "a sandbox test: nothing is launched (brain.strategist.launches_per_week decides launches)"})
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
    if args.brain_cmd == "requests":
        from .brain import proposals
        if args.decide:
            if not (args.accept or args.reject):
                raise SystemExit("--decide needs --accept or --reject")
            _print(proposals.decide(conn, pol, args.decide, decided_by="owner", accept=args.accept,
                                    reason=args.reason or "owner"))
        _print(proposals.open_for(conn, args.role))
        return 0
    if args.brain_cmd == "scorecard":
        from .brain import predictions
        _print(predictions.scorecard(conn, mode="sandbox" if args.sandbox else "real", role=args.role))
        return 0
    if args.brain_cmd in ("portfolio-map", "capability-card", "commerce", "casting"):
        return _agency(args, conn, pol)
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
        from .brain import budget, evaluation, rubrics
        # The evaluator's daily budget bounds these runs as it bounds its jobs: refused before the first call,
        # stopped before any call once spent.
        try:
            budget.check(conn, pol, "evaluator")
        except budget.OverBudget as e:
            raise SystemExit(f"refused: {e}") from None
        with llm.tagged("evaluator", None, before_call=lambda: budget.check(conn, pol, "evaluator")):
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
    def make_llm(c, p):
        llm = Claude(c, p)
        llm.batching = True   # the brain worker journals its jobs' calls and batches llm.batch.roles (llm_batch)
        return llm
    runtime.serve(lambda: db.connect(), policy_mod.load, make_llm, worker_id=worker_id, on_output=_print)
    return 0

def _review(args, conn, pol, llm) -> int:
    import tempfile
    import time
    from pathlib import Path

    from .review import gates
    from .review import video as video_review

    store = storage.from_url()
    # Wakes when a job finishes (policy wakeups.reviewer) instead of only on its interval.
    listener = wakeup.Listener.from_policy(pol, "reviewer") if args.loop and not args.dry_run else None
    while True:
        with tempfile.TemporaryDirectory() as tmp:
            if args.dry_run:
                out = []
                for job in gates.pending_jobs(conn, pol, batch=args.batch):
                    d = Path(tmp) / str(job)
                    d.mkdir()
                    with logs.context(job_id=job):
                        out += gates.review_job(conn, pol, llm, store, job, d, role=args.role, write=False)
                for job in video_review.pending_jobs(conn, pol, batch=args.batch):
                    d = Path(tmp) / f"video-{job}"
                    d.mkdir()
                    with logs.context(job_id=job):
                        out += video_review.review_job(conn, pol, llm, store, job, d, write=False)
                _print({"reviews": out, "accepted": sum(r["accepted"] for r in out), "of": len(out),
                        "model": llm.model_for(args.role or pol.get("review", "image", "role")),
                        "llm_usd": llm.spent})
            else:
                result = {"images": gates.run_pending(conn, pol, llm, store, Path(tmp), batch=args.batch),
                          "clips": video_review.run_pending(conn, pol, llm, store, Path(tmp), batch=args.batch)}
                if not args.batch:   # our accepted assets in the research vocabulary (performance.own_annotation)
                    from .review import own_annotation
                    try:
                        own = own_annotation.run_pending(conn, pol, llm, store, Path(tmp) / "own")
                    except Exception as e:  # noqa: BLE001 - learning must never stop review
                        logging.getLogger("manyworlds.review").warning("own annotation failed: %s", e)
                        own = {"error": str(e)[:300]}
                    if own.get("enabled") is not False:
                        result["own_annotation"] = own
                if any(v for r in result.values() for v in r.values()) or not args.loop:
                    _print({**result, "llm_usd": llm.spent})
        if not args.loop:
            return 0
        # A full batch means more is waiting: go again at once (at scale a fixed sleep caps throughput).
        full = not args.dry_run and (
            result["images"].get("jobs", 0) >= int(pol.get("review", "image", "max_jobs_per_run", default=10)) or
            result["clips"].get("jobs", 0) >= int(pol.get("review", "video", "max_jobs_per_run", default=10)))
        if not full:
            # While a post is producing, images and clips land at any moment: look again soon (busy_loop_seconds).
            # A finished job (NOTIFY mw_jobs) ends the wait early either way.
            idle_s = float(pol.get("review", "image", "loop_seconds", default=120))
            busy = not args.dry_run and (any(r.get("jobs") for r in result.values()) or work_in_flight(conn))
            wakeup.wait(listener, float(pol.get("review", "image", "busy_loop_seconds", default=idle_s))
                        if busy else idle_s, sleep=time.sleep)
        pol = policy_mod.load()


def brain_steps(conn, pol, llm=None) -> dict:
    """The Brain's part of each producer tick. Each step is isolated: one that fails is reported (an owner event,
    once a day per step) and the others, and production and publishing after them, still run. With an `llm`, newly scheduled posts are
    directed (shot sheets, manyworlds.brain.director)."""
    from datetime import datetime, timezone

    from . import notify
    from .brain import discover, events, ideas, launches, portfolio, showrunner_loop
    from .brain import chief_of_staff
    from .research import desk, monitor
    from .brain.agents import comment_themes, market_lead, style_director, talent_manager
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
        # Each step has its own switch: turning the CEO off (no launches) must not stop careers, predictions or
        # approved launches.
        if pol.get("brain", "strategist", "enabled", default=False):
            portfolio.ensure_daily(conn, pol)      # the daily portfolio cycle (one strategist job a day)
        talent_manager.ensure_reviews(conn, pol, now=now)   # each creator's career review, on its stage's cadence
        style_director.ensure(conn, pol, now=now)           # each creator's style and world, and the monthly season
        market_lead.ensure_reviews(conn, pol, now=now)      # each market that warrants a lead
        comment_themes.ensure(conn, pol, now=now)           # what each audience keeps saying, for the showrunner
        discover.ensure_due(conn, pol, now=now)             # research discovery on its cadence (live only)
        monitor.ensure_due(conn, pol, now=now)              # the reference universe's daily checks (live, enabled)
        desk.ensure_due(conn, pol, now=now)                 # the research desk's watched topics, once a day
        chief_of_staff.tick(conn, pol, now=now)             # predictions resolved, stalled requests and agents reported
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
        # A finished cast (a sandbox recast, a test) is still judged and stored while paused: no model call, and a
        # creator must not keep her old voice because operations are paused (2026-11-25: Maren's recast sat unjudged).
        from .creator import voice_lab
        try:
            casts = voice_lab.settle_casts(conn, pol)
        except Exception as e:  # noqa: BLE001 - reported, retried next tick
            casts = {"error": f"{type(e).__name__}: {e}"[:300]}
        # Research the owner lets run while paused (operations.while_paused: research_monitor): the daily monitor and
        # desk runs, each queued once a day (both check the leave themselves).
        research = {}
        from .research import desk, monitor
        from .brain import research_questions
        for name, due in (("monitor", monitor.ensure_due), ("desk", desk.ensure_due),
                          ("questions", research_questions.ensure_answering)):
            try:
                job = due(conn, pol)
                if job:
                    research[name] = str(job)
            except Exception as e:  # noqa: BLE001 - reported, retried next tick
                research[name] = f"error: {type(e).__name__}: {e}"[:300]
        # The research loop's code steps (spine sync, knowledge gaps) and the live end-to-end checks, each on its own
        # cadence (owner 2026-11-27: intelligence keeps working and failures stay visible while paused).
        from .brain import chief_of_staff
        allowed = pol.get("operations", "while_paused", default=[]) or []
        for name, fn in (("research_loop", chief_of_staff.research_tick), ("verify", chief_of_staff.verify_tick)):
            if name in allowed:
                try:
                    got = fn(conn, pol, _dt.now(_tz.utc))
                    if got and not got.get("skipped"):
                        research[name] = got
                except Exception as e:  # noqa: BLE001 - reported, retried next tick
                    research[name] = f"error: {type(e).__name__}: {e}"[:300]
        return {"operations": "paused", **({"casts": casts} if casts and any(casts.values()) else {}),
                **({"research": research} if research else {})}
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
        accounts = metrics.collect_account(conn, pol)   # follower growth, non-follower reach, profile views (daily)
        if accounts["accounts"] or accounts["errors"]:
            result["account_insights"] = accounts
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


def work_in_flight(conn) -> bool:
    """Whether a post is being produced: the producer and reviewer loops then poll every
    production.busy_loop_seconds / review.image.busy_loop_seconds instead of their idle interval."""
    try:
        return bool(conn.execute("SELECT 1 FROM posts WHERE status = 'producing' LIMIT 1").fetchone())
    except Exception:  # noqa: BLE001 - a dead connection: the idle interval
        return False


def _produce(args, conn, pol, llm) -> int:
    """The producer: a full tick (produce_tick) every production.loop_seconds; while a post is producing, light
    ticks in between (production.advance only) every production.busy_loop_seconds, so a post moves on soon after
    its jobs and reviews finish without polling the Brain, publishing and metrics more often."""
    import time

    from . import operations, production

    last, last_full = None, None
    # Wakes when a job finishes or reviews land (policy wakeups.producer); the sleep below stays the timeout.
    listener = wakeup.Listener.from_policy(pol, "producer") if args.loop else None
    while True:
        idle_s = float(pol.get("production", "loop_seconds", default=120))
        full = last_full is None or time.monotonic() - last_full >= idle_s
        try:
            if full:
                last_full = time.monotonic()   # a failing full tick waits its interval too
                result = produce_tick(conn, pol, llm)
            else:
                result = (production.advance(conn, pol, llm=llm) if operations.is_live(pol)
                          else {"operations": "paused"})
        except Exception as e:  # noqa: BLE001 - the loop keeps going; the error is printed
            logging.getLogger("manyworlds.produce").exception("producer tick failed")
            result = {"error": f"{type(e).__name__}: {e}"[:300]}
        paused = result == {"operations": "paused"}
        changed = any(v for k, v in result.items() if full or k != "producing")
        if not args.loop or (paused and last != "paused") or (not paused and changed):
            _print(result)
        last = "paused" if paused else "live"
        if not args.loop:
            return 0
        busy = not paused and work_in_flight(conn)
        timeout = float(pol.get("production", "busy_loop_seconds", default=idle_s)) if busy else idle_s
        # A wakeup runs a light tick (production.advance) unless the full one is due. Paused, nothing listens: an
        # unread LISTEN connection would hold back the server's notification queue (it reconnects on unpause).
        if paused and listener is not None:
            listener.close()
        wakeup.wait(None if paused else listener, timeout, sleep=time.sleep)
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


def _research(args, conn, pol) -> int:
    """mw research forget|purge|patterns (manyworlds.research.references, manyworlds.brain.patterns)."""
    from .brain import patterns
    from .research import external, references
    if args.action in ("pilot", "monitor", "graph-check", "propositions", "gold") and not external.allowed(pol):
        print(f"mw research {args.action}: {external.reason(pol)}")   # owner 2026-11-27
        return 2
    if args.action == "graph-check":   # one live Business Discovery call: which fields Meta returns to us
        from .research.providers import GraphProvider
        if not args.who:
            print("usage: mw research graph-check <business or creator handle>")
            return 2
        _print(GraphProvider(pol, conn=conn).check(args.who.lstrip("@")))
        return 0
    if args.action == "forget":
        if not args.who:
            print("usage: mw research forget <handle | profile url | post url>")
            return 2
        _print(references.forget(conn, references.research_store(pol), args.who))
    elif args.action == "purge":
        _print(references.purge(conn, pol, references.research_store(pol)))
    elif args.action == "data-audit":
        from .research import data_policy
        _print(data_policy.audit(conn, pol, apply=args.apply))
    elif args.action == "feedback":
        _print(patterns.feedback(conn, pol))
    elif args.action == "pilot":
        return _research_pilot(args, conn, pol)
    elif args.action == "monitor":
        return _research_monitor(args, conn, pol)
    elif args.action == "vision-bench":
        return _research_vision_bench(args, conn, pol)
    elif args.action in ("spine", "gold", "agreement", "ask", "answer", "gaps", "spot-check", "budget", "propositions"):
        return _research_loop(args, conn, pol)
    else:
        out = patterns.validate(pol)
        if args.seed and not out["problems"]:
            out["seeded"] = patterns.seed(conn, pol)
        _print({"cards": sorted(patterns.library(pol)["cards"]), **out})
        return 1 if out["problems"] else 0
    return 0


def _research_loop(args, conn, pol) -> int:
    """The research cycle by hand (docs/2026-11-26-unified-intelligence.md):
    spine                         mirror research and production tables into the analytical spine (code, free)
    gold [--assets N] [--usd X]   a gold run on our own accepted images, then the agreement report (--dry-run: count)
    agreement                     the agreement report over every gold read held (free)
    ask --question "..." [--kind data|web|both]    file a question (answered by `answer`)
    answer                        answer the queued questions (recipes, exploration, the desk); within the budgets
    gaps                          file knowledge gaps as pre-registered questions (code, free)
    spot-check <unit id> --labels attr=value ...   the owner's labels for a unit, joined to every agreement report
    budget                        the monthly research ceiling: spent, left, today's allowance"""
    from .research import budget, gold, spine
    if args.action == "spine":
        _print(spine.sync(conn, pol))
    elif args.action == "budget":
        _print(budget.status(conn, pol))
    elif args.action == "propositions":   # code every reference creator's propositions and verticals (text only)
        from .research import propositions
        ids = [str(r["id"]) for r in conn.execute("SELECT id FROM reference_creators ORDER BY id").fetchall()]
        if args.dry_run:
            _print({"dry_run": True, "creators": len(ids), "cap_usd": args.usd or propositions.cfg(pol).get(
                "max_usd_per_run"), "candidates": propositions.candidates(conn)})
            return 0
        from .llm import Claude
        _print({**propositions.read(conn, pol, Claude(conn, pol), ids, usd=args.usd),
                "candidates": propositions.candidates(conn)})
    elif args.action == "agreement":
        _print(gold.report(conn, pol, requester="cli"))
    elif args.action == "gold":   # third-party research posts (owner 2026-11-27: not our own posts yet)
        n = int(args.assets or gold.cfg(pol).get("external_items", 60))
        if args.dry_run:
            items = gold.external_items(conn, pol, n)
            _print({"dry_run": True, "posts": len(items), "creators": len({p["creator_id"] for p in items}),
                    "cap_usd": args.usd or gold.cfg(pol).get("max_usd_per_run"),
                    "monthly_left_usd": budget.left(conn, pol)})
            return 0
        from .llm import Claude
        _print(gold.run_external(conn, pol, Claude(conn, pol), n=n, usd=args.usd))
    elif args.action == "ask":
        if not args.question:
            print('usage: mw research ask --question "..." [--kind data|web|both]')
            return 2
        from .brain import research_questions
        _print({"question_id": research_questions.ask(conn, pol, "owner", args.question, kind=args.kind)})
    elif args.action == "answer":
        from .brain import research_questions
        from .llm import Claude
        _print(research_questions.run_due(conn, pol, Claude(conn, pol)))
    elif args.action == "gaps":
        from .brain import research_questions
        _print(research_questions.file_gaps(conn, pol))
    elif args.action == "spot-check":
        if not args.who or not args.labels:
            print("usage: mw research spot-check <unit id> --labels attribute=value ...")
            return 2
        labels = dict(x.split("=", 1) for x in args.labels if "=" in x)
        _print({"stored": gold.spot_check(conn, pol, args.who, labels)})
    return 0


def _research_vision_bench(args, conn, pol) -> int:
    """mw research vision-bench --assets N --models haiku,sonnet,opus --usd X [--creator H] [--dry-run]: the coded
    section read by each candidate on our own accepted images, scored against the reference model
    (manyworlds.research.vision_bench). --dry-run prints the estimate and calls nothing."""
    from .research import vision_bench
    cfg = pol.get("intelligence", "vision_bench", default={}) or {}
    models = [m.strip() for m in (args.models or "").split(",") if m.strip()] or list(cfg.get("models") or [])
    items = vision_bench.accepted_image_assets(conn, limit=int(args.assets or cfg.get("assets", 0)),
                                               creator=args.creator)
    est = vision_bench.estimate(pol, items=len(items), models=models)
    if args.dry_run or args.usd is None:
        _print({"dry_run": True, "assets_found": len(items), "estimate": est,
                **({} if args.dry_run else {"note": "give --usd <cap> to run"})})
        return 0
    from .llm import Claude
    out = vision_bench.run(conn, pol, Claude(conn, pol), items=items, models=models, usd_cap=args.usd,
                           keep_reads=False)
    _print({"estimate": est, **out})
    return 0


def _research_pilot(args, conn, pol) -> int:
    """mw research pilot --hypothesis <thesis id | text> --usd <cap> [--dry-run]: the influencer intelligence pilot's
    stage 1 (scout, annotate, distill) as one brain.intelligence job in a test-iia-<run> batch (a test batch: it runs
    while operations are paused, on the brain worker), capped at --usd. --dry-run prints the estimate only."""
    import re
    import uuid as uuid_mod

    from . import queue
    from .brain.agents import intelligence
    cfg = pol.get("intelligence", "agent", default={}) or {}
    cap = float(args.usd if args.usd is not None else cfg.get("max_usd_per_run", 0))
    if args.resume_annotate:
        return _research_resume(args, conn, pol, cap)
    if not args.hypothesis:
        print("usage: mw research pilot --hypothesis <thesis id | text> --usd <cap> [--dry-run]\n"
              "       mw research pilot --resume-annotate <batch> [--usd <cap>] [--dry-run]")
        return 2
    est = intelligence.estimate(pol, creators=args.creators, posts=args.posts, platforms=args.platforms or None,
                                hashtags_per_platform=len(args.hashtags) if args.hashtags else None,
                                plan=not args.hashtags)
    est["cap_usd"] = cap
    est["fits_cap"] = est["total_usd"] <= cap
    is_id = re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", args.hypothesis.lower())
    payload = {"mode": "pilot", "usd": cap, **({"thesis_id": args.hypothesis} if is_id
                                               else {"hypothesis": args.hypothesis})}
    for k in ("handles", "hashtags", "platforms"):
        if getattr(args, k):
            payload[k] = getattr(args, k)
    for k in ("creators", "posts"):
        if getattr(args, k):
            payload[k] = getattr(args, k)
    if args.reel_views:
        payload["reel_views"] = True
    if args.dry_run:
        _print({"dry_run": True, "estimate": est, "payload": payload,
                "storage": references_storage(pol)})
        return 0
    run = uuid_mod.uuid4().hex[:8]
    # test work: while operations are paused, workers claim only jobs marked test (queue.claim only_test)
    job = queue.enqueue(conn, "brain.intelligence", {**payload, "test": True}, batch=f"test-iia-{run}")
    _print({"queued": str(job), "batch": f"test-iia-{run}", "estimate": est, "storage": references_storage(pol)})
    return 0


def _research_resume(args, conn, pol, cap: float) -> int:
    """mw research pilot --resume-annotate <batch> [--usd <cap>] [--dry-run]: annotate and distill the posts a pilot
    batch already collected (its scout's creators, else those whose posts it fetched), without pulling anything, as
    one brain.intelligence job (mode resume) in a new test-iia-<run> batch. --dry-run prints the scope and the
    estimate only."""
    import uuid as uuid_mod

    from . import queue
    from .brain.agents import intelligence
    scope = intelligence.batch_scope(conn, args.resume_annotate)
    if not scope["jobs"]:
        print(f"no brain.intelligence job in batch {args.resume_annotate}")
        return 2
    if not scope["creator_ids"]:
        print(f"batch {args.resume_annotate} collected no creators: nothing to annotate")
        return 2
    est = intelligence.estimate_resume(conn, pol, scope["creator_ids"])
    est["cap_usd"] = cap
    est["fits_cap"] = est["total_usd"] <= cap
    payload = {"mode": "resume", "usd": cap, "creator_ids": scope["creator_ids"], "resumes": args.resume_annotate}
    if args.dry_run:
        _print({"dry_run": True, "resumes": args.resume_annotate, "jobs": scope["jobs"],
                "creators": len(scope["creator_ids"]), "estimate": est})
        return 0
    run = uuid_mod.uuid4().hex[:8]
    job = queue.enqueue(conn, "brain.intelligence", {**payload, "test": True}, batch=f"test-iia-{run}")
    _print({"queued": str(job), "batch": f"test-iia-{run}", "resumes": args.resume_annotate,
            "creators": len(scope["creator_ids"]), "estimate": est})
    return 0


def _research_monitor(args, conn, pol) -> int:
    """mw research monitor [--dry-run] [--estimate N] [--usd X]: the reference universe's due checks
    (manyworlds.research.monitor) as one brain.intelligence job (mode monitor) in a test-iia-mon-<run> batch, capped at
    --usd (default intelligence.monitor.daily_usd). --dry-run: what is due and what it would cost; --estimate N:
    $/month for N creators at the policy's tier mix."""
    import uuid as uuid_mod

    from . import queue
    from .research import monitor
    if args.estimate:
        _print(monitor.monthly_estimate(pol, int(args.estimate)))
        return 0
    if args.dry_run:
        _print({"dry_run": True, **monitor.due_estimate(conn, pol)})
        return 0
    cap = float(args.usd if args.usd is not None else monitor.cfg(pol).get("daily_usd", 0))
    run = uuid_mod.uuid4().hex[:8]
    job = queue.enqueue(conn, "brain.intelligence", {"mode": "monitor", "usd": cap, "test": True},
                        batch=f"test-iia-mon-{run}")
    _print({"queued": str(job), "batch": f"test-iia-mon-{run}", "cap_usd": cap, **monitor.due_estimate(conn, pol)})
    return 0


def references_storage(pol) -> str:
    """Where this run's research media would go (references.research_store)."""
    import os

    from .research import references
    if os.environ.get(references.STORE_ENV):
        return "research bucket (RESEARCH_STORAGE_URL)"
    fallback = pol.get("intelligence", "storage", "fallback", default="main_bucket")
    return f"main bucket under {references.PREFIX}" if fallback == "main_bucket" else "none (no media downloaded)"


if __name__ == "__main__":
    sys.exit(main())
