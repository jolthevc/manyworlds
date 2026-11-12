# Manyworlds

Autonomous engine that researches, creates, reviews and publishes a portfolio of **openly
AI-generated** social creators at high volume and low cost. The Brain (Claude) decides what to
make; GPU workers on cheap interruptible cards make it; every dollar lands in a ledger.

## Layout

- `manyworlds/queue.py` — Postgres job queue for interruptible GPUs (leases, heartbeats, reclaim, reaping)
- `manyworlds/worker/` — generic worker loop + handlers per job kind (`image.zimage`, `video.wan_i2v`, `tts.kokoro`, `train.lora`, `fake.*`)
- `manyworlds/brain/` — orchestration decisions (research plan, theses, choices); each one recorded in `decisions`
- `manyworlds/creator/lab.py` — creator bible, reference pack, anchors, LoRA, voice
- `manyworlds/research/` — Apify data source, signal aggregation
- `manyworlds/llm.py` — the only place Claude is called; structured outputs, cost recorded per call
- `manyworlds/ledger.py` — cost entries (GPU seconds, tokens, data items)
- `manyworlds/deploy/autoscale.py` — starts GPU machines when jobs wait, stops them when the queue drains; runs always-on
- `manyworlds/deploy/providers.py` — SaladCloud, Vast.ai, RunPod adapters (launch, list, destroy, self-destroy)
- `manyworlds/deploy/enforce.py` — deadline / spend cap / idle / daily-budget checks (autoscaler and GitHub backstop)
- `manyworlds/deploy/salad.py` — manual SaladCloud launcher for one-offs
- `manyworlds/migrations/` — forward-only SQL; add a new numbered file, never edit an applied one
- `policy/default.yaml` — every tunable number (budgets, GPU pools/prices, models, scope, limits)

## Rules that matter

- **No magic numbers in code.** Budgets, prices, thresholds, models, candidate counts go in the policy file. Cost entries and decisions record the policy version they ran under.
- **Everything that costs money writes to the ledger** — including failed attempts. Cost per *accepted* output is the headline metric. `gpu` entries attribute job time to outputs; `gpu_billed` is what the provider charges (idle included). The bill is every category except `gpu`.
- **No GPU machine without a deadline and spend cap, and nothing manual.** The autoscaler starts machines when jobs are queued (pin a benchmark to one pool with `mw bench ... --pools`) and registers deadline and cap before creating anything. Three independent layers stop machines: the autoscaler, the GitHub backstop (`.github/workflows/gpu-enforcer.yml`, self-chaining because GitHub's cron proved unreliable), and workers on Vast/RunPod destroying their own machine when idle or past deadline. Never rely on one process to stop paid resources. SaladCloud replaces exited containers, so only deleting the group stops billing.
- **Choose providers by measured cost per accepted output**, not hourly price: SaladCloud's consumer Windows nodes have less usable VRAM and RAM than advertised.
- **Handlers must be idempotent.** Interruptible nodes vanish mid-job; the queue reruns the job elsewhere.
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
`mw train-lora <handle>`, `mw bench <kind> --jobs N`, `mw report <batch>`, `mw status`.
GPU machines: `mw autoscale` (loop) or `mw autoscale --once`; always-on: `docker compose --profile live up -d autoscaler`.
One-off SaladCloud groups: `python -m manyworlds.deploy.salad up|status|down <family> --gpu "RTX 4090"`.

## Status

- Day 1 (done): queue, workers, ledger, storage, policy, benchmark, SaladCloud launcher.
- Day 2 (in progress): research → theses → creator bible → reference pack → anchors → LoRA → voice.
- GPU handlers (`image_zimage`, `video_wan`, `tts_kokoro`, `train_lora`) are written but only partly verified on real GPUs; the LightX2V video command and ai-toolkit Z-Image config need checking against their images.
- Next: content planner + production (day 3), review gates (day 4), operator console (day 5), learning loop (day 6).
