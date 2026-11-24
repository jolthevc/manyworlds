# GPU throughput: finished, accepted content per dollar (2026-11-24)

North star: $ per *accepted* output; speed second. Prices below are from provider pages read 2026-11-24 (undated pages
noted). **E** marks an estimate. Nothing was rented or run for this note.

## Recommendations, ranked

1. **Stop paying for boot: move the talkcomfy family (lip sync, story clips, voice) to RunPod Serverless flex workers.**
   RunPod does not bill image pulls or cached-model downloads, which "normally happen while the worker is scaled
   down" ([pricing docs](https://docs.runpod.io/serverless/pricing)). Billing covers only model load to VRAM plus
   generation. Today 10-19% of talk billing is boot and 8-11% is idle tail, and first-job weight downloads are
   counted as "busy" on top of that. Bake the ComfyUI weights into the image, or put them in a private HF repo as a
   cached model. `manyworlds.deploy.serverless` already exists and is inactive. Run it as one A/B test against the
   `runpod_secure_l40s_vol` pool.
2. **Batch by the boot, not by the job.** A pod should start only when the queued work is worth at least ~3x its
   boot cost (**E**: about 10-20 GPU-minutes of work on a 4-5 min boot). The image family is the worst case: 23% boot,
   23% tail, 49% busy. Tighten `autoscale.batch` per family and let `production.pull_forward` fill the machine
   before it stops.
3. **Send story clips (Wan 2.2 I2V, 480p, 5 s) to a hosted API by default while volume is low. Keep lip sync and
   images self-hosted.** Hosted Wan 2.2 fast costs $0.05 a clip, about our own cost once overheads are included.
   Hosted InfiniteTalk costs 2-4x our cost, and hosted Z-Image 15-30x (table below). Route on measured
   $/accepted, which `scoring.py` already ranks, and keep at least one hosted lane per kind.
4. **End Vast bandwidth bleed.** Vast hosts set their own per-byte rates for upload and download, shown only on the
   offer card ([Vast pricing docs](https://docs.vast.ai/documentation/instances/pricing)). Bandwidth was $29 of our
   $40 Vast bill. Rank Vast offers by price per hour plus `download_gb` × the offer's download rate, with a hard
   cap, and keep Vast for the image family only (about 30 GB). Families of 45 GB or more should never run there.
5. **Make every GPU-second count inside the job.** Generate at 480p and upscale on the CPU finisher (about 2.25x
   fewer pixels than 720p). Use 4-step distilled Wan everywhere (lightx2v/Lightning). Keep the fp8 weights. Add
   SageAttention if the WanVideoWrapper build supports it. Skip torch.compile and TensorRT until machines live
   longer than ~30 min.
6. **Measure what "busy" hides.** Split billed time into pull, weight download, VRAM load, generate and tail, per
   machine. Alert on network-volume free space, since a full volume killed three machines today.
7. **At 10x volume, buy always-on capacity per family** once sustained work passes ~12-14 GPU-hours/day: Serverless
   active workers, or a dedicated pod.

## 1. Cold start: options and numbers

| Option | What is billed at boot | 30-50 GB weights, boot time | Notes |
|---|---|---|---|
| Pod + pull from HF/R2 (today) | pull + download + load | measured 2-4 min median to first job; boot grace 20-30 min | Vast also bills the bytes |
| Pod + RunPod network volume | load from volume | **E** 2-4 min at the stated 200-400 MB/s sustained ([network volumes](https://docs.runpod.io/storage/network-volumes)) | $0.07/GB-month (150 GB ≈ $10.50). Ties us to one data centre: US-NC-1 had LOW L40S stock on 2026-11-22. A full volume kills jobs |
| Pod, weights baked in image | image pull | same bytes as a download, unless the host already caches the layer | Only helps where layers are cached |
| **RunPod Serverless + cached model / baked image** | **load to VRAM only** | cached models: "a few seconds, even for large models" ([model caching](https://docs.runpod.io/serverless/endpoints/model-caching), undated) | One HF model per endpoint (private repos allowed). Talkcomfy's multi-file set fits better baked into the image. FlashBoot is listed but not specified |
| SaladCloud | node pull | "several minutes ... up to tens of minutes" ([Salad migration guide](https://docs.salad.com/container-engine/how-to-guides/migration/migrate-from-runpod)) | Image is cached once per region, then each node pulls. Fit only for long batches |
| Modal | container start + load | GPU memory snapshots cut a small model from ~20 s to ~2 s, but "will generally not improve" weight-load-bound cold starts ([memory snapshots](https://modal.com/docs/guide/memory-snapshots)) | L40S $0.000542/s ≈ $1.95/h ([pricing](https://modal.com/pricing)) |
| Hosted per-output APIs | nothing | none | Provider churn: fal's Veo endpoint ended 2026-11-17 (see `2026-11-21-model-landscape.md`) |

Lambda and Hyperbolic were not researched in depth. Both rent hourly on-demand instances with no scale-to-zero
primitive, so they only fit the always-on stage.

**The weights are not the main cost; booting for them over and over is.** At $1.09/h, 4 minutes of boot costs $0.07,
about the same as one lip-sync clip. On a $0.34/h 4090 it costs $0.023, the same as ~70 Z-Image renders.

## 2. Keep-warm vs scale-to-zero

Prices: RunPod pod L40S secure $1.09/h (our policy), RunPod Serverless 48 GB flex $0.00053/s ≈ $1.91/h, active
$0.00037/s ≈ $1.33/h ([RunPod pricing](https://www.runpod.io/pricing), snapshot via search; RunPod's own blog gives
older, different rates, so confirm in the console).

**E** workload: one finished talking/story video ≈ 3-8 GPU-minutes of L40S time, from the measured 3-7 min lip sync
plus 1-3 min of clips and voice.

| Videos/day | GPU-h/day (**E**) | Serverless flex/day | Always-on L40S pod/day | Better choice |
|---|---|---|---|---|
| 10-50 | 0.5-6.7 | $1-13 | $26 | flex |
| 150 | 7.5-20 | $14-38 | $26 | either; measure |
| 500 | 25-67 | — | 2-3 pods ≈ $52-78 (≈ $0.10-0.16/video) | dedicated pods or active workers |

Break-even: a pod's real utilisation has to beat 1.09 / 1.91 ≈ **57%** to undercut flex. Image pods run at 49%
busy, and less once downloads are subtracted, so today flex wins. An always-on card wins once a family keeps it about
60% busy, around 12-14 GPU-hours/day.

On spot and interruptible capacity: Salad's $0.09-0.16/h cards look cheap, but their cold starts are minutes to tens
of minutes and their VRAM is smaller than advertised. They suit long image batches only, with idempotent handlers,
and those are already in place.

## 3. Levers inside a job

- **Distillation dominates.** Wan 2.2 A14B on an H200 took 642.8 s for a 720p 5 s clip at baseline and 79.0 s with
  the 4-step Lightning LoRA ([Jarvislabs](https://jarvislabs.ai/blog/wan-2-2-lightning-different-video)). lightx2v
  lists 10.7 s (480p) and 26.7 s (720p) for a 4-step I2V on a single Blackwell GPU
  ([LightWan2.2-A14B](https://huggingface.co/lightx2v/LightWan2.2-A14B)). We already use lightx2v, so confirm it is
  on every clip path.
- **Resolution.** 720p costs ~2.5x the 480p time in lightx2v's table. Generate at 480p, then let `finishing.py`
  upscale on the CPU finisher (about $0.09/h, **E** from Fly). Judge the upscaled 480p output on acceptance before
  locking this in.
- **Batch per model load.** Model affinity plus "free the last engine on a family switch" is right. The next step is
  to queue jobs of one family back-to-back for one machine. The 47-63 GB LongCat load justifies batches of 5 or more
  takes.
- **Precision.** fp8-scaled weights are already used. Wan in fp8 fits a 24 GB card only with block swap, which is
  slower, so 48 GB cards stay the default for A14B and InfiniteTalk.
- **Compile, TensorRT, snapshots.** torch.compile adds minutes at warm-up, worthwhile only with a cache persisted
  across boots or on long-lived workers. Modal notes compile can break snapshots. TensorRT for video DiTs is not
  worth the engineering at this volume.
- **Card choice (E).** An H100 at $3.00/h is ~2-3x an L40S on DiTs at 2.75x the price, so $/output is about equal.
  Pick it only for speed or 80 GB needs (LongCat bf16). A 4090 or 5090 wins on $/output for images.

## 4. Hosted per-output APIs vs our cost

| Output | Hosted price | Our cost (**E**) |
|---|---|---|
| Wan 2.2 I2V 480p 5 s | Replicate fast $0.05 ([blog](https://replicate.com/blog/wan-22)); fal turbo $0.05 (480p), $0.10 (720p); fal standard $0.04/video-s at 480p ([fal](https://fal.ai/models/fal-ai/wan/v2.2-a14b/image-to-video/turbo)) | 1.5-3 min L40S ≈ $0.03-0.055 at 100% busy; $0.06-0.11 at today's utilisation |
| InfiniteTalk lip sync, 10 s | WaveSpeed $0.03/s (480p), $0.06/s (720p), fast $0.015/s, 5 s minimum ([WaveSpeed](https://wavespeed.ai/models/wavespeed-ai/infinitetalk-fast)) → $0.15-0.30 | measured 3-7 min ≈ $0.055-0.13 at 100%; ~$0.08-0.19 at 70% |
| Z-Image Turbo ~2 MP | fal $0.005/MP ≈ $0.01 ([fal](https://fal.ai/models/fal-ai/z-image/turbo)); RunPod public endpoint $0.005 | ~2.3 s/1024² on a 4090 → $0.0002-0.0004; ~$0.001 with overheads |

Hosted APIs carry risks a price table hides: our creator LoRAs (fal's LoRA endpoints cost more), refusals of
photoreal faces, and endpoint churn. Route by measured $/accepted.

## 5. Recommended architecture

**Now (10-50 videos/day, hundreds of images):**
- **talkcomfy** (lip sync, voice; story clips when self-hosted): RunPod Serverless flex, 48 GB tier, max workers
  2-3, weights baked into the image. Deadline and spend cap are enforced by the autoscaler: an endpoint is a paid
  resource like any machine.
- **Story clips**: hosted Wan 2.2 480p as the default lane, with self-hosted as the challenger in `scoring.py`.
- **Images**: pods on RunPod community or Vast 4090 offers with a bandwidth filter. Boot only for ≥ N queued images
  (**E**: 100 or more, or a 30-min deadline).
- **E $/finished video**: $0.20-0.50 GPU+API, made up of 3-4 clips at $0.05, a lip-sync take at ~$0.10, and voice
  and images in cents. At the ~26% post acceptance measured before the 2026-11-23 review fixes that is ~$0.80-1.90 per *accepted* video, so acceptance is
  the bigger lever.

**At 10x (500+ videos/day):**
- One or two RunPod Serverless active workers, or reserved pods, per hot family, sized to the measured daily
  GPU-hours. Flex workers absorb peaks.
- Bring story clips back in-house once a warm worker runs them at under $0.03.
- Ask sales for an active-worker discount. Price an 80 GB card for LongCat-type long takes.
- **E**: $0.12-0.25 per finished video in GPU time.

## What to measure, per family and pool

- $/accepted output: already the headline metric.
- Billed seconds split into pull, download, load, generate and tail. Generation share is the target: above 70% now,
  above 85% at 10x.
- Time to first job, bytes downloaded per boot and Vast bandwidth $.
- Jobs per boot.
- Network-volume free space.
- Hosted refusal rate and acceptance rate against self-hosted for the same briefs.
