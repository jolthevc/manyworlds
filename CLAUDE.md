# Manyworlds

Autonomous engine that researches, creates, reviews and publishes a portfolio of **openly
AI-generated** social creators at high volume and low cost. The Brain (Claude) decides what to
make; GPU workers on cheap interruptible cards make it; every dollar lands in a ledger.

## Layout

- `manyworlds/queue.py` — Postgres job queue for interruptible GPUs (leases, heartbeats, reclaim, reaping)
- `manyworlds/worker/` — generic worker loop + handlers per job kind (`image.zimage`, `image.narrated`, `image.voice`, `image.qwen_edit` (reference-conditioned scene photos; its own worker family `edit` on the image worker image, off), `comfy.wan_i2v` (story and motion clips: ComfyUI, Wan 2.2 I2V A14B + lightx2v; `hosted.video` is the fallback; the diffusers `video.wan_i2v` path was retired 2026-11-22), `comfy.wan_animate2` (motion transfer: dance, poses), `talk.infinitetalk` (lip-sync, own image; InfiniteTalk or LongCat-Video-Avatar by `params.model`; on the ComfyUI engine it also lip-syncs a motion clip, V2V), `voice.design`/`voice.speak`, `train.lora`, `fake.*`); a kind's worker family is its prefix unless `job_kinds.<kind>.family` says otherwise; a code fault (TypeError, missing package...) fails a job at once, a machine fault releases it; a card below `job_kinds.<kind>.min_vram_gb` never serves the kind; a worker prefers the loaded model's kinds and frees the last engine when it switches family; `keyframes.py` is shared by every clip handler; `smoke.py` checks a built image before it is pushed
- `manyworlds/brain/` — orchestration decisions (research plan, theses, choices, content plans); each one recorded in `decisions`
- `manyworlds/brain/agent.py` — agent contract (`AgentContext`, `record_decision`, `commission`); agents in `brain/agents/` run as `brain.<name>` jobs via `brain/runtime.py` (`mw brain worker`)
- `manyworlds/brain/archetypes.py` + `policy/archetypes/` — the archetype library (playbooks: audience promise, how they communicate, content mix, talk share, capabilities, series patterns, hooks, edit grammar, pitfalls); the director, showrunner and talent manager read it
- The agency (`docs/superpowers/plans/2026-11-23-agency.md`): four judgement roles (CEO = `agents/strategist.py`, market leads = `agents/market_lead.py` (activated per market), talent managers, the creative team: showrunner + director) on a bench of specialists, most of them code. Connective tissue: `brain/proposals.py` (typed requests between roles, one owner each, rights and money checked in code: `brain.rights`; `mw brain requests`), production tiers (`brain.tiers`, `creator_states.tier`, set by the CEO), `brain/predictions.py` (checkable claims scored against real outcomes; `mw brain scorecard`: proof that decisions improve), `brain/chief_of_staff.py` (code: resolves predictions, expires undecided requests, reports failing or idle-spending agents)
- `manyworlds/creator/dossier.py` + `brain/agents/talent_manager.py` — each creator's coordinator: a talent manager per creator builds the dossier (archetypes, positioning, audience insight, story, content system; one checked call per section, kept apart from siblings) at launch, reviews the career on a cadence by stage (drop events route to it as a diagnosis), states its conviction with its case, files requests and predictions, writes the brief the showrunner plans from, edits the profile only within `brain.talent_manager.editable`, asks the owner for the rest, and reports to the strategist (the CEO) through the portfolio snapshot
- `manyworlds/brain/agents/style_director.py` + `brain/aesthetics.py` + `policy/aesthetics/` — the style director owns each creator's look: `style` (vibe, palette, grade, signature items, wardrobe rules, motifs, tastes, shot families, face share), `world` (1-2 anchor sets, home base, travel, carousel brand) and a monthly `season` (capsule wardrobe, trips); a launch step, refreshed monthly. Consistency is face, appearance and style; outfits and places vary (only anchor sets recur). The aesthetics library is the shot vocabulary (44 types, most without her face: her back at sunset, a ringed hand, a table from above), grades, variety rules, the photo-dump order and carousel templates
- `manyworlds/slides.py` — carousel text slides (cover, list, step, quote, comparison, recap, ask) drawn in the creator's carousel brand on the producer (Pillow, OFL fonts), placed around her photos; ideas carry `slides` (capability `edits.text_slides`)
- `manyworlds/brain/series.py` — series episode state: episodes (numbered at publication, never at scheduling), open threads and callbacks; the showrunner sees each series' last episodes and open threads. `agents/comment_themes.py` turns a creator's recent comments into a few themes (Haiku, batched, nothing called without comments; comments read only through `evidence.py`, handles scrubbed)
- `manyworlds/brain/revise.py` + `agents/idea_reviser.py` — an idea just under a feed-QA floor (`brain.qa.min_scores`, within `brain.revise.margin`) gets one rewrite from the evaluator's reasons and is re-scored, instead of being dropped
- `manyworlds/production/pull_forward.py` — paid GPU time goes to real posts or the machine stops: a warm image machine about to run dry starts the next planned posts early (`production.pull_forward`), never alternates; with no demand an idle machine stops after `gpu_runs.idle_stop_minutes_no_demand`
- `manyworlds/brain/director.py` — director step: turns scheduled posts' scenes into shot sheets (`body.shots`, strict schema) in two calls: the shots, then their edit (on-screen text, cuts, caption style); one call is over the structured-output grammar limit; a shot names its `shot_type` from her vocabulary and code fills its subject and grade (a shot without her face is drawn and judged without it)
- `manyworlds/brain/evidence.py` — the only reader of findings and metrics; `real` mode never sees simulated data
- `manyworlds/brain/overrides.py` — settings the Brain changes at runtime within `brain.authority`; beyond it → `approvals`
- `manyworlds/brain/budget.py` — per-agent daily LLM budgets from the ledger's `agent` tag
- `manyworlds/production/` — one module per stage (`scenes`, `takes`, `voice`, `motion`, `talk`, `followups`, `captioning`, `mixing`, `finish`, `core`; `__init__` re-exports every name; patch a name in the module that looks it up), its posts.body state keys declared in `state.py` (a test fails on an undeclared key), one producer per post (`locks.py`, a lease row; a held post is skipped that tick): planned post → scene images → (narrated edit | story clip | talking video) → finish → ready; each piece is a retried `Step`; long talking videos run one job per segment (`talk_segments.py`) and are joined on the control plane (`media.py`); a creator with a designed voice has every spoken piece voiced first (`voice.speak`, then `audio_uri`); walk-and-talk (`generation.talk_motion`): a take whose shot moves is a silent motion clip first (chained from each last frame for longer lines, joined without the repeated frame), then lip-synced onto it (V2V); the finish stage (`finish_video`) upscales, grades and burns captions into every video post, as a `finish.video` job on the finisher while `production.finish_mode: queue` (`inline`: in the producer)
- `manyworlds/finishing.py` — the finish pass (upscale to `finishing.output`, grade, grain, hook text, two-pass loudness) and the cover frame (the Reel's cover); `captions.py` + `fonts/` (OFL) — word-timed burned-in captions in named styles inside the platforms' safe zones (`burned_captions`); `overlays.py` — on-screen labels, hook text, timestamps and question bubbles (`on_screen_text`, from the shot's `edit.on_screen_text`), drawn in the same ASS pass as the captions
- `manyworlds/direction.py` — shot sheet (set, wardrobe, framing, blocking, performance, sound, edit) + prompt compiler per model
- `manyworlds/operations.py` — the master switch `operations.mode` (shipped `paused`: only test jobs and `bench-`/`test-` batches run)
- `manyworlds/breadth.py` — the breadth test (`mw test breadth --plan default [--dry-run]`, `mw test breadth-report <run>`): handwritten posts organised by capability, run only for its own posts in `test-breadth-<run>` batches under `policy/test_overlay.yaml` (formats on, reroute off, direction gate report-only, one merged GPU machine); never publishes. Runs only with the owner's go
- `manyworlds/review/` — review gates (test batches judged under the test overlay). After media exists only defects in it reject (face identity, realism, safety, audio, sync); scene fit and direction are scored, never rejecting; a cheap screen judges everything, Sonnet confirms only close calls. `gates.py` (images; set plates and POV/cutaway shots have no identity gate), `video.py` (story and talking clips from keyframes; motion-transfer clips also get the `dance_identity` gate), `final.py` (the finished video: text over the face, seams, end sync, "real phone footage"; report-only until `review.final.enforce`), `detail.py` (finer checks: hands, teeth, b/p/m lip closure, background flicker, set/outfit consistency against references; report-only until enforced), `common.py` (shared stages and writes)
- `manyworlds/creator/lab.py` — creator bible, reference pack, anchors, LoRA, voice; `profile.py` versioned profiles and dials; `sets.py` set and wardrobe bibles (canonical text plus a reviewed plate per set or outfit, refreshed each producer tick; scenes carry them as references; reference-conditioned scenes via `image.qwen_edit` while `sets.reference_conditioning`, off)
- `manyworlds/creator/voice_lab.py` + `worker/handlers/tts_voxcpm.py` — designed voices (VoxCPM2): candidates from the bible, picked automatically (WER, distance from other creators), lines spoken in the cloned voice with their delivery; `voices.engine` falls back to Kokoro; `mw voice design|candidates|pick <handle> [--candidate N]`
- `manyworlds/research/` — Apify data source, signal aggregation; discovery runs as a `brain.scouts` job every `research.discover.every_days` while operations are live (`brain/discover.ensure_due`), or by hand (`mw discover`)
- `manyworlds/llm.py` — the only place LLMs are called: per role Claude or an open-weight model on an OpenAI-compatible host (`llm.hosts`, `llm.open_models`, `{model, fallback}` roles); structured outputs, cost recorded per call; shadow tests of candidates into `llm_shadow`, compared by `shadow.py` (`mw llm shadow-report <role>`); `selection.py` proposes a role's cheapest candidate that holds up (agreement, rubric quality, or blind preference judged by another model family for open-ended roles: `mw llm judge <role> --usd N`), the owner approves
- `manyworlds/llm_batch.py` — on the brain worker every brain job's calls are journaled (a rerun replays them free) and the roles in `llm.batch.roles` go through Anthropic's Message Batches at half price: the job steps aside (`queue.defer`) until its answer lands; direct call after `max_wait_minutes` or a batch error; `mw llm batches`
- `manyworlds/notify.py` — owner alerts: `owner_event()` is how code tells the owner something; the owner reads them with `mw inbox` (alerts, pending approvals, requests to the owner; `--done` clears; `mw status` counts them). No phone channel (owner, 2026-11-23); ntfy push only if `NTFY_TOPIC` is ever set
- `manyworlds/ledger.py` — cost entries (GPU seconds, tokens, data items)
- `manyworlds/deploy/autoscale.py` — starts GPU machines when jobs wait, stops them when the queue drains; runs always-on
- `manyworlds/deploy/providers.py` — SaladCloud, Vast.ai, RunPod adapters (launch, list, destroy, self-destroy)
- `manyworlds/deploy/credentials.py` — what a rented machine may hold: a scoped `mw_worker` database login (`WORKER_DATABASE_URL`, `mw worker-credentials`), never the control plane's own
- `manyworlds/deploy/enforce.py` — deadline / spend cap / idle / never-started / stuck / daily-budget checks (autoscaler and GitHub backstop)
- `manyworlds/deploy/scoring.py` — ranks pools by measured cost per accepted output; `report.py` — billed vs finished work per machine; `reconcile.py` — providers' itemised charges (Vast bandwidth/disk) into the ledger
- `manyworlds/brain/portfolio.py`, `agents/strategist.py` — daily portfolio cycle; `launches.py`, `agents/creator_lab.py` — creator launches; `events.py` — breakout/drop/QA-surge events
- `manyworlds/publish/instagram.py`, `metrics.py` — Instagram Graph API publishing (AI label always) and real insights; Unified Login (one system-user token, `META_SYSTEM_USER_TOKEN`), setup sheet, verified `mw account connect --username`, posting ramp for new accounts (`docs/instagram-setup.md`)
- `manyworlds/publish/comments.py` — comments on our posts answered in the creator's voice (`autonomy.comment_replies`: draft by default; `mw comments <handle>`); `public_copies.py` deletes the JPEG copies made for Instagram after `publishing.public_copy_keep_days`
- `fly.toml` — the control plane on Fly.io (`.github/workflows/deploy.yml` deploys it with the GitHub secret `FLY_API_TOKEN`, a deploy token set 2026-11-23) (app `manyworlds-mw`: autoscaler, brain, producer, reviewer; `finisher`, a performance-2x machine the autoscaler starts while finish jobs wait (`ensure_finisher`, pool `fly_finisher`) and that stops itself when idle); `.github/workflows/deploy.yml` deploys it after tests pass on `main`
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
- **No GPU test, benchmark or end-to-end run without the owner's yes for that run.** Build and test with stubs freely.
- **Check the brief before the GPU; after it, only media defects kill.** Ideas, scripts and shot sheets are checked and fixed cheaply before any image or video is made (evaluator, director rules, `direction.feasibility`); once media exists, a post is dropped only for bad media, never for the idea or wording. Judging must cost far less than making.
- **Capabilities, not formats.** The system is a toolbox (policy `capabilities`) the Brain composes per archetype; formats and recipes are examples, never templates.

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

## Status (2026-11-23)

- Built and running on Fly (paused): research → theses → creator lab (bible, references, LoRA, voice, profile, dossier) →
  showrunner ideas (scored) → director shot sheets → production (photo, carousel, Story, narrated edit, story clip,
  lip-synced talking video incl. long takes in segments, walk-and-talk V2V, motion transfer) → review → finish
  (1080x1920, 30 fps, phone-native looks, room tone, word captions, labels/hooks/timestamps) → publisher (Instagram,
  idle until an account is connected). Brain M1-M8 merged; the agency layer (archetypes, dossier, talent manager)
  merged 2026-11-23. Plans: `docs/superpowers/plans/2026-11-14-brain-roadmap.md`; debt: `docs/tech-debt.md`;
  the latest build notes: `docs/research/2026-11-23-readiness-and-craft.md`, `docs/2026-11-23-overnight-report.md`.
- One creator (Maren, sandbox). Workers: image v27 (Z-Image Turbo), talkcomfy v13 (ComfyUI: Wan 2.2 I2V A14B,
  InfiniteTalk, LongCat, Wan-Animate-2; VoxCPM2 voices). Lip sync on ComfyUI is proven (13 clips, 3-7 min each on an
  L40S with the weights volume); ComfyUI story clips have not finished a job on a GPU yet.
- Measured waste (ledger to 2026-11-22): 24% of billed GPU time ran a job; 74% of real posts were dropped at image
  review, mostly for brief mismatch, not bad media; judging cost more than the useful GPU work. Fixed in code
  2026-11-23 (code faults, memory floors, model affinity, face-only identity, report-only direction, cheaper
  confirms, prompt caching); unproven on a GPU.
- 2026-11-23 (gap sweep): dropped ideas cancel their queued evaluator jobs (`queue.cancel`); a dropped post retracts its canon (`memory.retract_post`); a video rerouted to a photo gets its caption rewritten (role `caption_rewrite`); faceless shots never judged on identity (`review.common.judges_identity`); a missing test overlay refuses that work (`OverlayMissing`); narrated-edit labels follow the real photo spans (needs image worker rebuild); launches set the creator's market; image worker ML deps pinned (`docker/constraints-image.txt`); stale ready posts from the old pipeline dropped live. Routing trial ($1.84, sandbox, blind Opus judge): director kept on Sonnet (Qwen4-Max and GLM-5.3 lost 4/4); style director: Qwen4-Max won 3/3 at ~53% of the cost, GLM-5.3 lost 3/3 (renamed keys); talent manager: Qwen4-Max won 2/2 but cost ~96% and broke JSON once, GLM-5.3 lost 2/2. Too few to switch; shadows keep collecting.
- 2026-11-24 (later): an architecture review and its fixes, deployed: production is a package with declared state
  and a per-post lease; the policy is checked (every path the code reads exists; a schema for hot sections); JSON
  logs with post/job ids (`logs.py`); producer and reviewer wake on job/review notifications (`wakeup.py`,
  migration 033, Neon's direct endpoint); richer director context with a cached static brief; reliability fixes
  (poison posts, failing reviews, lost leases, retry backoff). Finishing can run as `finish.video` on an on-demand
  Fly finisher (`finish_mode: queue` since 2026-11-23, started with an app-scoped `FLY_API_TOKEN`; no real video
  finished on it yet).
- 2026-11-24: a three-way review (pipeline seams, agency bugs, creative gaps) and its fixes: clear passes are
  chosen, real metrics reach analysis, accepted requests have effects, model memory is freed on a family switch.
  Built the style layer (style director, aesthetics library, aspirational faceless shots, branded carousel
  slides), clip edits (1-4 silent clips joined) and a moving first frame. Not yet run against a real model or a GPU.
- 2026-11-23 night: the agency layer (requests, rights, tiers, predictions, market lead, chief of staff) built and
  deployed; talent manager and two-call director verified live on a sandbox clone ($0.53 of Claude in total).
- Waiting on the owner: the breadth test (`mw test breadth --plan default`, ~$5-9, needs RunPod or Vast credit and
  the owner's go); connecting an Instagram account; `NTFY_TOPIC`, `FLY_API_TOKEN` in GitHub (deploys); `MW_HASH_SALT` and a
  bucket-scoped R2 token for workers (tech debt).
