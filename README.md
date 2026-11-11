# Manyworlds

Autonomous research and production engine for openly AI-generated creators.
Design and rationale: the "Manyworlds — Architecture & Build Proposal" doc.

## Day 1 status: the factory floor

| Piece | File | State |
| --- | --- | --- |
| Schema (creators, jobs, attempts, assets, reviews, posts, metrics, cost ledger, playbook) | `manyworlds/migrations/001_init.sql` | Done, tested |
| Versioned policy (budgets, GPU pools and prices, candidate counts, limits) | `policy/default.yaml`, `manyworlds/policy.py` | Done, tested |
| Job queue for interruptible GPUs (leases, heartbeats, reclaim after preemption, retries, dead jobs) | `manyworlds/queue.py` | Done, tested |
| Worker loop (claim → run → upload → assets with provenance → GPU cost → complete) | `manyworlds/worker/runner.py` | Done, tested with fake handlers |
| Cost ledger (every GPU-second, failed attempts included) | `manyworlds/ledger.py` | Done, tested |
| Object storage (R2 / MinIO / local) | `manyworlds/storage.py` | Done |
| Adapter interfaces and router | `manyworlds/adapters/` | Done, tested |
| Photo worker: Z-Image Turbo + creator LoRA | `manyworlds/worker/handlers/image_zimage.py` | Written, **needs a GPU run** |
| Video worker: LightWan2.2 (NVFP4, 4-step) or diffusers Wan 2.2 | `manyworlds/worker/handlers/video_wan.py` | Written, **needs a GPU run**; LightX2V command must be confirmed against its image |
| Voice worker: Kokoro | `manyworlds/worker/handlers/tts_kokoro.py` | Written, **needs a GPU run** |
| Benchmark: measured seconds per output → $ per 1k on every pool | `manyworlds/bench.py`, `mw bench` / `mw report` | Done, tested with fake handlers |

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

## Run the real GPU benchmark

1. **Database reachable from the internet.** Remote workers need it: create a hosted
   Postgres (Neon or Supabase free tier is enough) and set `DATABASE_URL`; run `mw migrate`.
2. **Storage.** Create a Cloudflare R2 bucket and API token; set `STORAGE_URL=s3://<bucket>`,
   `S3_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com`, and the key pair.
3. **Build and push worker images** to a registry the GPU clouds can pull from:

   ```bash
   docker build -f docker/worker.Dockerfile --build-arg EXTRA=image --build-arg PREFETCH="Tongyi-MAI/Z-Image-Turbo" -t <registry>/mw-worker-image .
   docker build -f docker/worker.Dockerfile --build-arg EXTRA=tts --build-arg PREFETCH="hexgrad/Kokoro-82M" -t <registry>/mw-worker-tts .
   docker build -f docker/worker-video-lightx2v.Dockerfile -t <registry>/mw-worker-video .
   ```

4. **Start workers** with the same env plus `MW_POOL` (e.g. `salad_batch_4090`) and
   `MW_WORKER_KINDS` (`image.zimage`, `video.wan_i2v` or `tts.kokoro`):
   one SaladCloud container group per image, or a Vast.ai / RunPod instance running the image.
5. **Benchmark:** `mw bench image.zimage --jobs 50`, `mw bench tts.kokoro --jobs 30`,
   and for video a start frame first: `mw bench video.wan_i2v --jobs 20 --extra '{"image_uri": "s3://<bucket>/<still>.png"}'`.
   Then `mw report <batch>` gives median and p90 seconds per output and projected $ per 1,000
   outputs for every pool with that GPU.

## What day 1 still needs from you

- A **SaladCloud** (or Vast.ai) account and API key, plus RunPod as backup, with ~$100 credit.
- A **hosted Postgres** URL and a **Cloudflare R2** bucket (both have free tiers).
- A **container registry** login (Docker Hub or GitHub Container Registry).
- A **Hugging Face token** if a model is gated.

Keys go in `.env` (copy `.env.example`); never commit it.
