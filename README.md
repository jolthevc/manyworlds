# Manyworlds

Autonomous research and production engine for openly AI-generated creators.
Design and rationale: the "Manyworlds — Architecture & Build Proposal" doc. Working rules for
anyone (or any agent) changing the code: `CLAUDE.md`.

## What runs where

- **Control services** (`docker compose --profile live up -d`): the autoscaler (starts and stops
  GPU machines), the producer (plans posts, turns due ones into jobs, picks reviewed outputs) and
  the reviewer (judges every image and clip). They run on one always-on host.
- **GPU workers** on rented machines (Vast.ai, RunPod, SaladCloud), started by the autoscaler
  with a deadline and a spend cap. One image, `mw-worker-image`, serves photos, voice lines,
  narrated edits and story clips; `mw-worker-train` trains creator LoRAs.
- **Shared state**: Postgres (Neon) for the queue, ledger and posts; R2 for files and model weights.

## Pipeline

| Stage | Code | State |
| --- | --- | --- |
| Research (public, logged-out signals) → theses | `research/`, `brain/discover.py` | Working |
| Creator lab: bible, reference photos, LoRA, voice | `creator/lab.py`, `creator/profile.py` | Working (one creator, Maren) |
| Planner: a week of posts in the creator's voice | `brain/planner.py` | Working |
| Production: photo, carousel, narrated edit, story clip | `production.py` | Working; story clips new (2026-11-14) |
| Review gates: images (identity, realism, safety, scene), clips from keyframes | `review/` | Working |
| Approval console, publishing, learning loop | — | Not built |

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

Keys go in `.env` (copy `.env.example`); never commit it. Run the `mw` image with `--env-file .env`:

- `mw status`: job counts, spend, cost per output, and where GPU money went over 72 h
- `mw machines --hours 72`: every GPU machine, billed against the work it finished
- `mw posts <creator>`, `mw creator <creator>`, `mw bench <kind> --jobs N [--pools p]`, `mw report <batch>`

## Worker images

```bash
MW_REGISTRY=docker.io/<you> scripts/build-worker.sh image v18
```

builds the image, runs `python -m manyworlds.worker.smoke image` inside it (imports every
handler's libraries and runs the CPU parts: ffmpeg, clip export, keyframes) and pushes only if
that passes. Then set `worker_images.<family>.tag` in `policy/default.yaml`; the autoscaler
won't rent a machine for an image that isn't published. Images are public: the Dockerfiles copy
only code and policy, and `.dockerignore` drops `.env`.
