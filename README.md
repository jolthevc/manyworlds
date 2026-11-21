# Manyworlds

Autonomous research and production engine for openly AI-generated creators.
Design and rationale: the "Manyworlds — Architecture & Build Proposal" doc. Working rules for
anyone (or any agent) changing the code: `CLAUDE.md`.

## What runs where

Nothing depends on a laptop.

- **Control plane** on Fly.io (`fly.toml`, app `manyworlds-mw`): the autoscaler (starts and stops GPU
  machines, pushes owner alerts), the Brain worker, the producer (plans posts, turns due ones into jobs,
  picks reviewed outputs, joins long talking videos) and the reviewer (judges every image and clip).
  Deployed by `.github/workflows/deploy.yml` after the tests pass on `main`. What actually runs is gated
  by policy `operations.mode` (shipped `paused`: only tests and benchmarks).
- **GPU workers** on rented machines (Vast.ai, RunPod, SaladCloud), started by the autoscaler with a
  deadline, a spend cap and a scoped database login (`mw worker-credentials`). `mw-worker-image` serves
  photos, voice lines, narrated edits and story clips; `mw-worker-talk` lip-synced talking video;
  `mw-worker-train` creator LoRAs.
- **Shared state**: Postgres (Neon) for the queue, ledger and posts; R2 for files and model weights.
- **CI**: tests on every push (`tests.yml`); worker images built on Depot (`build-worker.yml`); the GPU
  enforcer backstop (`gpu-enforcer.yml`).

## Pipeline

| Stage | Code | State |
| --- | --- | --- |
| Research (public, logged-out signals) → theses | `research/`, `brain/discover.py` | Working |
| Creator lab: bible, reference photos, LoRA, voice | `creator/lab.py`, `creator/profile.py` | Working (one creator, Maren) |
| Brain: showrunner, evaluator, analyst, strategist, launches | `brain/` | Working; launches need owner approval |
| Production: photo, carousel, narrated edit, story clip, talking video (any length, in segments) | `production.py`, `talk_segments.py` | Working; talking video awaits its benchmark |
| Review gates: images (identity, realism, safety, scene), clips from keyframes | `review/` | Working |
| Publishing (Instagram, AI label always) and real metrics | `publish/` | Built; no account connected yet |
| Owner alerts (ntfy) | `notify.py` | Built; needs `NTFY_TOPIC` |
| Open-weight LLMs per role, shadow tests | `llm.py`, `shadow.py` | Built; no open model configured yet |

## Run locally (no GPU needed)

```bash
docker compose up -d db minio
docker compose run --rm mw migrate
docker compose run --rm mw init-storage
docker compose run --rm --entrypoint pytest mw -q
docker compose run --rm mw bench fake.image --jobs 20      # prints a batch id
docker compose up -d fake-worker                           # 2 CPU workers drain the queue
docker compose run --rm mw report <batch id>
```

## Against real services

On the control plane: `fly ssh console -a manyworlds-mw -C "mw status"`. Useful commands:

- `mw status`: job counts, spend, cost per output, and where GPU money went over 72 h
- `mw machines --hours 72`: every GPU machine, billed against the work it finished
- `mw brain approvals [--decide ID --approve|--reject]`, `mw llm shadow-report <role>`
- `mw posts <creator>`, `mw creator <creator>`, `mw bench <kind> --jobs N [--pools p]`, `mw report <batch>`

## Worker images

Push a tag `worker-<family>-<version>` (e.g. `worker-image-v21`): `build-worker.yml` builds it on Depot,
runs `python -m manyworlds.worker.smoke <family>` inside the build (imports every handler's libraries and
runs the CPU parts: ffmpeg, clip export, keyframes) and pushes only if that passes. Then set
`worker_images.<family>.tag` in `policy/default.yaml`; the autoscaler won't rent a machine for an image
that isn't published. Images are public: the Dockerfiles copy only code and policy, and `.dockerignore`
drops `.env`. (`scripts/build-worker.sh` does the same with a local Docker.)
