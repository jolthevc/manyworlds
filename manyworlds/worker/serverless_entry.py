"""The worker inside a RunPod Serverless endpoint (`mw serverless-worker`; manyworlds.deploy.serverless).

Each request says which job kinds and pool it serves; the handler claims and runs up to `max_jobs` jobs of those
kinds from the queue, exactly as a pod's worker does (same handlers, outputs, idempotency and ledger), then returns.
The worker stays loaded between requests (models stay warm in the process), and RunPod bills only while a
request runs. The endpoint's environment carries the scoped worker credentials, never the control plane's.
"""

from __future__ import annotations

import os


def handle(event: dict) -> dict:
    from .. import policy as policy_mod, storage
    from . import runner
    inp = (event or {}).get("input") or {}
    kinds, pool = list(inp.get("kinds") or []), str(inp.get("pool") or "")
    if not kinds or not pool:
        return {"error": "input needs kinds and pool"}
    # Attempts and metering name the request; a serverless request is its own registered unit of work.
    os.environ["MW_GROUP"] = f"srv-{inp.get('request', 'x')}"
    os.environ.pop("MW_PROVIDER", None)      # nothing to self-destroy: the endpoint scales itself
    pol = policy_mod.load()
    worker = runner.start(kinds, pool=pool, policy=pol, store=storage.from_url())
    n = worker.run(max_jobs=int(inp.get("max_jobs", 1)),
                   idle_exit_seconds=float(pol.get("gpu_runs", "serverless_idle_exit_seconds", default=20)))
    return {"jobs": n, "request": inp.get("request")}


def main() -> None:
    import runpod
    runpod.serverless.start({"handler": handle})
