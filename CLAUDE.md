# Manyworlds

Autonomous engine that researches, creates, reviews and publishes a portfolio of **openly
AI-generated** social creators at high volume and low cost. The Brain (Claude) decides what to
make; GPU workers on cheap interruptible cards make it; every dollar lands in a ledger.

## Layout

- `manyworlds/queue.py` — Postgres job queue for interruptible GPUs (leases, heartbeats, reclaim, reaping)
- `manyworlds/worker/` — generic worker loop + handlers per job kind (`image.zimage`, `image.narrated`, `image.voice`, `video.wan_i2v`, `train.lora`, `fake.*`); `smoke.py` checks a built image before it is pushed
- `manyworlds/brain/` — orchestration decisions (research plan, theses, choices, content plans); each one recorded in `decisions`
- `manyworlds/production.py` — planned post → scene images → (narrated edit | story clip) → ready; each piece is a retried `Step`
- `manyworlds/review/` — review gates: `gates.py` (images), `video.py` (clips from keyframes), `common.py` (shared stages and writes)
- `manyworlds/creator/lab.py` — creator bible, reference pack, anchors, LoRA, voice; `profile.py` versioned profiles and dials
- `manyworlds/research/` — Apify data source, signal aggregation
- `manyworlds/llm.py` — the only place Claude is called; structured outputs, cost recorded per call
- `manyworlds/ledger.py` — cost entries (GPU seconds, tokens, data items)
- `manyworlds/deploy/autoscale.py` — starts GPU machines when jobs wait, stops them when the queue drains; runs always-on
- `manyworlds/deploy/providers.py` — SaladCloud, Vast.ai, RunPod adapters (launch, list, destroy, self-destroy)
- `manyworlds/deploy/enforce.py` — deadline / spend cap / idle / never-started / stuck / daily-budget checks (autoscaler and GitHub backstop)
- `manyworlds/deploy/scoring.py` — ranks pools by measured cost per accepted output; `report.py` — billed vs finished work per machine
- `scripts/build-worker.sh` — build, smoke-test and push a worker image
- `manyworlds/migrations/` — forward-only SQL; add a new numbered file, never edit an applied one
- `policy/default.yaml` — every tunable number (budgets, GPU pools/prices, models, scope, limits)

## Rules that matter

- **No magic numbers in code.** Budgets, prices, thresholds, models, candidate counts go in the policy file. Cost entries and decisions record the policy version they ran under.
- **Everything that costs money writes to the ledger** — including failed attempts. Cost per *accepted* output is the headline metric. `gpu` entries attribute job time to outputs; `gpu_billed` is what the provider charges (idle included). The bill is every category except `gpu`.
- **No GPU machine without a deadline and spend cap, and nothing manual.** The autoscaler starts machines when jobs are queued (pin a benchmark to one pool with `mw bench ... --pools`; small backlogs of production work wait to share a machine, policy `autoscale.batch`) and registers deadline and cap before creating anything. Three independent layers stop machines: the autoscaler, the GitHub backstop (`.github/workflows/gpu-enforcer.yml`, self-chaining because GitHub's cron proved unreliable), and workers on Vast/RunPod destroying their own machine when idle or past deadline. Never rely on one process to stop paid resources. SaladCloud replaces exited containers, so only deleting the group stops billing.
- **Choose providers by measured cost per accepted output**, not hourly price: SaladCloud's consumer Windows nodes have less usable VRAM and RAM than advertised.
- **Handlers must be idempotent.** Interruptible nodes vanish mid-job; the queue reruns the job elsewhere.
- **Smoke-test worker images before pushing** (`scripts/build-worker.sh`). A missing package found on a rented machine costs a boot and a model download; the test finds it for free.
- **Every external capability sits behind an adapter/handler**, so models and vendors can be swapped by config.
- **Disclosure is non-negotiable.** Every creator carries the policy `disclosure_line`; posts set platform AI flags; never imply real first-hand experience in sponsored content.
- **Creator guardrails:** adults only (policy `creator_min_age`), original people (never based on a real person), nothing in `scope.excluded_topics`.
- **Research privacy:** logged-out public data only; author handles stored only as salted hashes; raw items purged after 30 days (`research.apify.purge_expired`); store derived features, not dossiers.
- **No cross-account engagement** between our creators, ever (platforms ban coordinated networks).
- **Secrets live only in `.env`** (gitignored). Never print, log or commit them. Pass them to containers with `--env-file`.
- Brain/creator/judge roles use the models set in `policy.llm.roles`; change models there, not in code.

## Run it

```bash
docker compose up -d db minio                         # local Postgres + S3 stand-in
docker compose run --rm --entrypoint pytest mw -q     # tests (stub Claude/Apify; no spend)
docker compose run --rm mw migrate
```

Against real services (Neon, R2, Claude, Apify, SaladCloud), run the `mw` image with `--env-file .env`:
`mw migrate`, `mw discover`, `mw choose`, `mw launch-creator`, `mw anchors <handle>`,
`mw train-lora <handle>`, `mw bench <kind> --jobs N`, `mw report <batch>`, `mw status`, `mw machines`, `mw posts <handle>`.
Live services: `docker compose --profile live up -d --build` (autoscaler, reviewer, producer; rebuild after code changes, compose doesn't mount code).
GPU machines: `mw autoscale` (loop) or `mw autoscale --once`; always-on: `docker compose --profile live up -d autoscaler`.

## Status (2026-11-14)

- Done: queue, workers, ledger, storage, policy, benchmark (day 1); research → theses → creator bible → reference pack → LoRA → voice (day 2); planner + production for photo, carousel, narrated edit and story clip (day 3); review gates for images and clips (day 4).
- One creator live (Maren). Posts stop at `ready`; a human approves before anything is published.
- GPU: worker image v18 (Z-Image Turbo, Kokoro, Wan 2.2 TI2V-5B). RunPod 3090 is the working pool while Vast has no credit; a 720p 5 s clip takes ~14 min there.
- Next: operator console for approvals (day 5), publishing, learning loop (day 6), then a second creator.
