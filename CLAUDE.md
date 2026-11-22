# Manyworlds

Autonomous engine that researches, creates, reviews and publishes a portfolio of **openly
AI-generated** social creators at high volume and low cost. The Brain (Claude) decides what to
make; GPU workers on cheap interruptible cards make it; every dollar lands in a ledger.

## Layout

- `manyworlds/queue.py` — Postgres job queue for interruptible GPUs (leases, heartbeats, reclaim, reaping)
- `manyworlds/worker/` — generic worker loop + handlers per job kind (`image.zimage`, `image.narrated`, `image.voice`, `image.qwen_edit` (reference-conditioned scene photos; its own worker family `edit` on the image worker image, off), `video.wan_i2v`, `talk.infinitetalk` (lip-sync, own image; InfiniteTalk or LongCat-Video-Avatar by `params.model`; on the ComfyUI engine it also lip-syncs a motion clip, V2V), `voice.design`/`voice.speak`, `train.lora`, `fake.*`); a kind's worker family is its prefix unless `job_kinds.<kind>.family` says otherwise; `smoke.py` checks a built image before it is pushed
- `manyworlds/brain/` — orchestration decisions (research plan, theses, choices, content plans); each one recorded in `decisions`
- `manyworlds/brain/agent.py` — agent contract (`AgentContext`, `record_decision`, `commission`); agents in `brain/agents/` run as `brain.<name>` jobs via `brain/runtime.py` (`mw brain worker`)
- `manyworlds/brain/director.py` — director step: turns scheduled posts' scenes into shot sheets (`body.shots`, strict schema)
- `manyworlds/brain/evidence.py` — the only reader of findings and metrics; `real` mode never sees simulated data
- `manyworlds/brain/overrides.py` — settings the Brain changes at runtime within `brain.authority`; beyond it → `approvals`
- `manyworlds/brain/budget.py` — per-agent daily LLM budgets from the ledger's `agent` tag
- `manyworlds/production.py` — planned post → scene images → (narrated edit | story clip | talking video) → finish → ready; each piece is a retried `Step`; long talking videos run one job per segment (`talk_segments.py`) and are joined on the control plane (`media.py`); a creator with a designed voice has every spoken piece voiced first (`voice.speak`, then `audio_uri`); walk-and-talk (`generation.talk_motion`): a take whose shot moves is a silent motion clip first (chained from each last frame for longer lines, joined without the repeated frame), then lip-synced onto it (V2V); the finish stage (`finish_video`) upscales, grades and burns captions into every video post
- `manyworlds/finishing.py` — the finish pass (upscale to `finishing.output`, grade, grain, hook text, two-pass loudness) and the cover frame (the Reel's cover); `captions.py` + `fonts/` (OFL) — word-timed burned-in captions in named styles inside the platforms' safe zones (`burned_captions`)
- `manyworlds/direction.py` — shot sheet (set, wardrobe, framing, blocking, performance, sound, edit) + prompt compiler per model
- `manyworlds/operations.py` — the master switch `operations.mode` (shipped `paused`: only test jobs and `bench-`/`test-` batches run)
- `manyworlds/review/` — review gates: `gates.py` (images; set plates have no identity gate), `video.py` (story and talking clips from keyframes), `detail.py` (finer checks: hands, teeth, b/p/m lip closure, background flicker, set/outfit consistency against references; report-only until enforced), `common.py` (shared stages and writes)
- `manyworlds/creator/lab.py` — creator bible, reference pack, anchors, LoRA, voice; `profile.py` versioned profiles and dials; `sets.py` set and wardrobe bibles (canonical text plus a reviewed plate per set or outfit, refreshed each producer tick; scenes carry them as references; reference-conditioned scenes via `image.qwen_edit` while `sets.reference_conditioning`, off)
- `manyworlds/creator/voice_lab.py` + `worker/handlers/tts_voxcpm.py` — designed voices (VoxCPM2): candidates from the bible, picked automatically (WER, distance from other creators), lines spoken in the cloned voice with their delivery; `voices.engine` falls back to Kokoro
- `manyworlds/research/` — Apify data source, signal aggregation
- `manyworlds/llm.py` — the only place LLMs are called: per role Claude or an open-weight model on an OpenAI-compatible host (`llm.hosts`, `llm.open_models`, `{model, fallback}` roles); structured outputs, cost recorded per call; shadow tests of candidates into `llm_shadow`, compared by `shadow.py` (`mw llm shadow-report <role>`)
- `manyworlds/notify.py` — owner alerts: owner events and pending approvals pushed once via ntfy (`NTFY_TOPIC`) from the autoscaler loop; `owner_event()` is how code tells the owner something
- `manyworlds/ledger.py` — cost entries (GPU seconds, tokens, data items)
- `manyworlds/deploy/autoscale.py` — starts GPU machines when jobs wait, stops them when the queue drains; runs always-on
- `manyworlds/deploy/providers.py` — SaladCloud, Vast.ai, RunPod adapters (launch, list, destroy, self-destroy)
- `manyworlds/deploy/credentials.py` — what a rented machine may hold: a scoped `mw_worker` database login (`WORKER_DATABASE_URL`, `mw worker-credentials`), never the control plane's own
- `manyworlds/deploy/enforce.py` — deadline / spend cap / idle / never-started / stuck / daily-budget checks (autoscaler and GitHub backstop)
- `manyworlds/deploy/scoring.py` — ranks pools by measured cost per accepted output; `report.py` — billed vs finished work per machine; `reconcile.py` — providers' itemised charges (Vast bandwidth/disk) into the ledger
- `manyworlds/brain/portfolio.py`, `agents/strategist.py` — daily portfolio cycle; `launches.py`, `agents/creator_lab.py` — creator launches; `events.py` — breakout/drop/QA-surge events
- `manyworlds/publish/instagram.py`, `metrics.py` — Instagram Graph API publishing (AI label always) and real insights; Unified Login (one system-user token, `META_SYSTEM_USER_TOKEN`), setup sheet, verified `mw account connect --username`, posting ramp for new accounts (`docs/instagram-setup.md`)
- `manyworlds/publish/comments.py` — comments on our posts answered in the creator's voice (`autonomy.comment_replies`: draft by default; `mw comments <handle>`)
- `fly.toml` — the control plane on Fly.io (app `manyworlds-mw`: autoscaler, brain, producer, reviewer); `.github/workflows/deploy.yml` deploys it after tests pass on `main`
- `.github/workflows/build-worker.yml` — build a worker image on Depot, smoke-test, push (tag `worker-<family>-vN`); `scripts/build-worker.sh` is the local-Docker equivalent
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
- **Simulated results never count as evidence of demand.** Agents read findings and metrics only through `brain/evidence.py`; the strategist and allocator only through `evidence.for_allocation` (real only). A test fails if anything else under `manyworlds/brain/` reads those tables.
- **Disclosure is non-negotiable.** Every creator carries the policy `disclosure_line`; posts set platform AI flags; never imply real first-hand experience in sponsored content.
- **Creator guardrails:** adults only (policy `creator_min_age`), original people (never based on a real person), nothing in `scope.excluded_topics`.
- **Research privacy:** logged-out public data only; author handles stored only as salted hashes; raw items purged after 30 days (`research.apify.purge_expired`); store derived features, not dossiers.
- **No cross-account engagement** between our creators, ever (platforms ban coordinated networks).
- **Secrets live only in `.env`** (gitignored), Fly secrets and GitHub Actions secrets. Never print, log or commit them. Pass them to containers with `--env-file`; to Fly by piping (`... | fly secrets import`), never through a terminal.
- **Rented GPU machines are untrusted hosts.** They get only scoped credentials (`deploy/credentials.py`); the autoscaler refuses to launch without them (policy `security`).
- **Nothing runs on its own until the owner says so.** `operations.mode` stays `paused` and `brain.strategist.launches_per_week` stays 0 unless the owner changes them; agents and benchmarks run only as tests.
- **Tell the owner through `notify.owner_event`**, not ad-hoc inserts into `events`.
- Brain/creator/judge roles use the models set in `policy.llm.roles`; change models there, not in code.

## Run it

```bash
docker compose up -d db minio                         # local Postgres + S3 stand-in
docker compose run --rm --entrypoint pytest mw -q     # tests (stub Claude/Apify; no spend)
docker compose run --rm mw migrate
```

Tests also run on GitHub Actions on every push (no local Python or Docker needed); read failures from the
check-run annotations.

Against real services (Neon, R2, Claude, Apify, GPU providers), on the Fly control plane:
`fly ssh console -a manyworlds-mw -C "mw status"` (flyctl at `~/.fly/bin/flyctl.exe` on the owner's laptop).
Commands: `mw migrate`, `mw discover`, `mw choose`, `mw launch-creator`, `mw anchors <handle>`,
`mw train-lora <handle>`, `mw bench <kind> --jobs N`, `mw report <batch>`, `mw status`, `mw machines`, `mw posts <handle>`,
`mw brain approvals`, `mw llm shadow-report <role>`, `mw worker-credentials` (pipe its output to `fly secrets import`).
Live services run on Fly only (`fly deploy --remote-only --ha=false`, or the deploy workflow); the autoscaler
applies migrations at start. Never start a second autoscaler anywhere else.

## Status (2026-11-21)

- Done: queue, workers, ledger, storage, policy, benchmark (day 1); research → theses → creator bible → reference pack → LoRA → voice (day 2); planner + production for photo, carousel, narrated edit and story clip (day 3); review gates for images and clips (day 4).
- One creator live (Maren), sandbox until her Instagram account is connected (her ready posts wait for it). Publishing is autonomous with a kill switch (`autonomy.publish: paused`).
- GPU: worker image v18 (Z-Image Turbo, Kokoro, Wan 2.2 TI2V-5B). RunPod 3090 is the working pool while Vast has no credit; a 720p 5 s clip takes ~14 min there.
- Brain M1 (foundations) merged 2026-11-14: agent runtime, decisions, per-agent budgets, real/simulated wall, overrides + approvals. Plan: `docs/superpowers/plans/2026-11-14-brain-roadmap.md`. Known debt: `docs/tech-debt.md`.
- Brain M2 (creative evaluation) merged 2026-11-14: rubrics in `policy/rubrics/`, evaluator agent, `mw brain calibrate`, `mw brain rubric-drift`.
- Brain M3 (memory + showrunner) merged 2026-11-14: the producer asks the showrunner for batches (`brain.showrunner.enabled`); ideas are scored before they become posts; Stories (9:16).
- Brain M5 (simulated closed loop) merged 2026-11-14: publisher interface + kill switch, scenario engine (`policy/scenarios/`), analyst (`brain/analysis.py` + agent), `mw brain simulate` / `mw brain learning`. Sandbox runs use their own budget and never touch real creators.
- Lip-synced talking video (M10, moved up 2026-11-19) merged 2026-11-20: talk worker image (InfiniteTalk on Wan2.1-I2V-14B, ~94 GB of weights downloaded on the GPU machine), `talking_video` format, stagings and flavors libraries. `talking_video` is not in `production.formats` until the benchmark (`bench-talk.infinitetalk-bba634`) passes the video gate.
- 2026-11-20 overnight: RunPod adapter on REST v2 (v1 retired 2026-11-15; a CUDA floor, secure-cloud pools); machines carry their attempt limit and pool info; Brain M6 merged (experiments sized by power, Thompson exploration, rubric validity); Brain M7 merged (targeted repair, video-to-photo reroute, feed QA); brain worker reconnects and refreshes its Claude client.
- 2026-11-20 day: true machine cost (Vast bills bandwidth per GB; offers ranked by it, caps count it, itemised
  charges reconciled hourly; `mw reconcile --days N`); Brain M8 merged (events, daily strategist, portfolio
  numbers from real evidence, autonomous launches via `launches` + `brain.creator_lab`, launches wait for owner
  approval while `brain.strategist.launch_requires_approval`); Brain M4 built (Instagram publisher with
  `is_ai_generated` on every post, real metrics collector, `mw account connect <handle> --token-env NAME`),
  idle until an account is connected. First strategist run proposed 2 launches (pending in `mw brain approvals`).
- 2026-11-21: master switch (`operations.mode: paused`), launches off, both pending launches rejected by the owner.
  Cloud-native: control plane on Fly (`manyworlds-mw`, iad), tests and image builds (Depot) in GitHub Actions; the
  laptop runs nothing. Hardening: scoped worker database login, owner alerts (ntfy), open-weight LLM routing with
  fallback and shadow tests, long talking videos in segments (cut or chain) joined by the producer.
- 2026-11-21 evening: model review (`docs/research/2026-11-21-model-landscape.md`) and owner model decisions (roadmap);
  open-weight candidates live on Together and shadow-tested per role (no role switches without evidence); a family whose
  last 3 machines finished nothing stops renting (`autoscale.no_output_guard`); Salad 3090 out of the image pools.
- Next: owner sets `NTFY_TOPIC`, `FLY_API_TOKEN` (GitHub), `WORKER_DATABASE_URL` (via `mw worker-credentials`), an
  inference account for open models; benchmark talking video (8-step distill LoRA, >48 GB-RAM host) only on the
  owner's go; connect Maren's account (M4 live); M9 portfolio allocation; Brain chat (needs the web layer).
