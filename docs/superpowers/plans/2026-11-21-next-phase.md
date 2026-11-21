# Next phase (owner direction, 2026-11-21)

Standing rule: nothing runs on its own until the owner says so. `operations.mode: paused` (shipped default) and
`brain.strategist.launches_per_week: 0`; agents and workflows run only as tests.

## 1. Cloud-native platform (the laptop is out of the loop)

Done: the test suite runs on GitHub Actions on every push (`.github/workflows/tests.yml`).

| Piece | Plan | Owner action |
|---|---|---|
| Control plane (autoscaler, producer, reviewer, brain worker) | Fly.io: one app from the `mw` image, 4 process groups, auto-restart; ~$10-20/month | create Fly account + spend alert; `fly secrets import` from .env |
| Deploys | GitHub Action: tests pass on `main` → `flyctl deploy` | add `FLY_API_TOKEN` as a GitHub secret |
| Worker images | built in CI, smoke-tested, pushed to GHCR (free for public images; no Docker Hub pull limits). The CUDA-devel talk image (kernel compile) on Depot ($20/month) or a larger GitHub runner (~$1 per build) | approve GHCR; optional Depot account |
| Weights | never on the laptop; R2 mirror (no egress fees) with parallel ranged downloads; avoid per-GB-bandwidth hosts (already enforced) | none |
| Web (later) | dashboard + Brain chat as an HTTP process on the same Fly app | none yet |

Cut-over: deploy to Fly with operations paused, check `mw status`, then stop the laptop's services (never two
autoscalers at once). Research: hosting comparison with prices and sources (2026-11-21 session).

## 2. Open-weight models where they do the job

Finding (research 2026-11-21): at today's volume Claude costs about $140/month across the five roles, and Haiku is
already cheaper than most hosted open models for image review. Open models' best case saves ~$50-60/month now;
the case grows with volume. Candidates: the creative evaluator (DeepSeek V4 Flash, ~$1 vs ~$20), the showrunner
(GLM-5.2, ~$11 vs ~$40), and vision second opinions (Gemma 4 31B / Qwen3.8-27B, Apache 2.0). Strategist and creator
lab stay on Claude until tested. Model names, licences and prices come from that research and need re-checking
on the provider's live page before use.

Build (no spend): an OpenAI-compatible provider adapter in `llm.py` (provider, model, host, quantisation and prices
per role in policy); a shadow-run harness that scores a candidate against Claude and owner labels on our own
calibration sets (agreement, schema failures, score variance, zero misses on safety flags); switch one role at a
time behind policy thresholds, Claude as fallback. Owner action: one inference account (e.g. Together or DeepInfra,
US-hosted, zero data retention) and its key in .env.

## 3. Long-form lip-sync (quality first, time is not a constraint)

InfiniteTalk's streaming mode extends a clip chunk by chunk (81-frame windows, 9 motion frames carried over), so
length is bounded by settings (`--max_frame_num`, default 1000 frames = 40 s) and render time, not the model.
Build: scripts of any length split at sentence boundaries into segments of `talk.max_segment_seconds`; each segment
rendered as its own job (idempotent, retryable, parallel across machines) with the last frame of one segment as the
start of the next; segments joined with the full audio track; the video gate judges keyframes across the whole
clip. Policy decides max length and segment size. Benchmark first with the 8-step distill LoRA on a host with
>48 GB RAM (L40S/A100-class secure cloud), measured cost per accepted minute.

## 4. Brain chat (roadmap)

A chat with the Brain in the web app: the owner discusses strategy in natural language; the Brain answers from
its decisions, findings, ledger and portfolio (read tools), and proposes changes as policy overrides, approvals or
commissions (write tools within `brain.authority`; anything beyond becomes an approval the owner confirms in the
same chat). Every change it makes is a recorded decision citing the conversation. Depends on 1 (web host).
