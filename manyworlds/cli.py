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

    r = sub.add_parser("report", help="benchmark / cost report for a batch")
    r.add_argument("batch")

    sub.add_parser("discover", help="Brain: plan research, pull it via Apify, propose creator theses")
    sub.add_parser("choose", help="Brain: choose the next thesis to launch")
    lc = sub.add_parser("launch-creator", help="Creator Lab: bible + creator + reference pack + voice sample")
    lc.add_argument("--thesis", help="thesis id (default: the chosen one)")
    an = sub.add_parser("anchors", help="pick the reference images for a creator")
    an.add_argument("creator")
    tl = sub.add_parser("train-lora", help="queue identity LoRA training for a creator")
    tl.add_argument("creator")
    cr = sub.add_parser("creator", help="show a creator's bible, anchors and jobs")
    cr.add_argument("creator")
    sub.add_parser("status", help="job counts and spend")
    sub.add_parser("reap", help="mark jobs dead whose final lease expired")
    sub.add_parser("init-storage", help="create the object storage bucket if missing")
    rv = sub.add_parser("review", help="run review gates on new creator images (--loop to keep going)")
    rv.add_argument("--batch")
    rv.add_argument("--loop", action="store_true")
    rv.add_argument("--role", help="LLM role to judge with, for comparing models (with --dry-run)")
    rv.add_argument("--dry-run", action="store_true", help="print verdicts, write no reviews")
    sc = sub.add_parser("autoscale", help="start/stop GPU machines to follow the queue (loops; --once for one tick)")
    sc.add_argument("--once", action="store_true")
    sc.add_argument("--no-launch", action="store_true")

    args = p.parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    pol = policy_mod.load()

    if args.cmd == "worker":
        from .worker.runner import Worker, handlers_for

        kinds = (args.kinds or os.environ["MW_WORKER_KINDS"]).split(",")
        worker = Worker(handlers=handlers_for([k.strip() for k in kinds if k.strip()]),
                        store=storage.from_url(), policy=pol, pool=args.pool or os.environ.get("MW_POOL", "local"))
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
    elif args.cmd == "bench":
        batch = bench.enqueue(conn, pol, args.kind, args.jobs, extra=json.loads(args.extra),
                              pools=args.pools.split(",") if args.pools else None)
        print(batch)
    elif args.cmd == "report":
        _print(bench.report(conn, pol, args.batch))
    elif args.cmd == "status":
        _print({"jobs": queue.counts(conn), "spend": ledger.totals(conn),
                "cost_per_output": ledger.cost_per_output(conn)})
    elif args.cmd == "reap":
        print(f"reaped {queue.reap(conn)} jobs")
    elif args.cmd in ("discover", "choose", "launch-creator", "anchors", "train-lora", "creator", "review"):
        return _brain(args, conn, pol)
    return 0


def _review(args, conn, pol, llm) -> int:
    import tempfile
    import time
    from pathlib import Path

    from .review import gates

    store = storage.from_url()
    while True:
        with tempfile.TemporaryDirectory() as tmp:
            if args.dry_run:
                out = []
                for job in gates.pending_jobs(conn, pol, batch=args.batch):
                    d = Path(tmp) / str(job)
                    d.mkdir()
                    out += gates.review_job(conn, pol, llm, store, job, d, role=args.role, write=False)
                _print({"reviews": out, "accepted": sum(r["accepted"] for r in out), "of": len(out),
                        "model": llm.model_for(args.role or pol.get("review", "image", "role")),
                        "llm_usd": llm.spent})
            else:
                result = gates.run_pending(conn, pol, llm, store, Path(tmp), batch=args.batch)
                if result["accepted"] or result["rejected"] or result["errors"] or not args.loop:
                    _print({**result, "llm_usd": llm.spent})
        if not args.loop:
            return 0
        time.sleep(float(pol.get("review", "image", "loop_seconds", default=120)))
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
        row = lab.create_creator(conn, bible, t["id"])
        batch = lab.enqueue_reference_pack(conn, pol, row["id"])
        voice_job = lab.enqueue_voice_sample(conn, row["id"])
        _print({"creator": row, "reference_batch": batch, "voice_job": voice_job,
                "bible": bible.model_dump(), "llm_usd": llm.spent})
    elif args.cmd == "anchors":
        cid = _creator_id(conn, args.creator)
        with tempfile.TemporaryDirectory() as tmp:
            _print({**lab.select_anchors(conn, pol, llm, storage.from_url(), cid, tmp), "llm_usd": llm.spent})
    elif args.cmd == "train-lora":
        _print({"job": lab.enqueue_lora_training(conn, pol, _creator_id(conn, args.creator))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
