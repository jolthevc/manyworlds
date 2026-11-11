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

    r = sub.add_parser("report", help="benchmark / cost report for a batch")
    r.add_argument("batch")

    sub.add_parser("status", help="job counts and spend")
    sub.add_parser("reap", help="mark jobs dead whose final lease expired")
    sub.add_parser("init-storage", help="create the object storage bucket if missing")

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
        batch = bench.enqueue(conn, pol, args.kind, args.jobs, extra=json.loads(args.extra))
        print(batch)
    elif args.cmd == "report":
        _print(bench.report(conn, pol, args.batch))
    elif args.cmd == "status":
        _print({"jobs": queue.counts(conn), "spend": ledger.totals(conn),
                "cost_per_output": ledger.cost_per_output(conn)})
    elif args.cmd == "reap":
        print(f"reaped {queue.reap(conn)} jobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
