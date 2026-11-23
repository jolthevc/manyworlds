# Image-to-video / text-to-video models for short social clips (November 2026)

Research date: 2026-11-21. About 17 tool calls. Many pricing and "best model" pages are SEO or reseller pages. Each one is labelled below. Leaderboard numbers come from Artificial Analysis (AA) pages fetched today. Those pages show an "Updated" badge but no date.

## 1. Open-weight leaders in late 2026 and their commercial licenses

### Takeaway
The open-weight frontier has moved past Wan 2.2. The leading open-weight models are now **MiniMax H3** (33B, native audio, top open-weights model on AA) and **LTX-2.5** (22B, native audio, multi-shot, 8-step distilled). Neither is OSI open source.
- MiniMax H3 **excludes the US, UK, EU and South Korea** unless you apply for a license.
- LTX-2.5 is free only under $10M annual revenue.

Alibaba's Wan 3.0 is API-only. **Wan 2.2 (Apache 2.0) is still the newest Wan with weights, and the cleanest license.**

### Cited findings
**AA open-weights image-to-video board (AA-Video-I2V v2.1, fetched 2026-11-21):**

| Rank | Model | Elo | Released | Price/min (creator API) |
|---|---|---|---|---|
| 1 | MiniMax H3 (768p) | 1137 | Jul 2026 | $4.80 |
| 2–3 | Hyperflow H3 | 1121 | Sep 2026 | $9.00 |
| 2–3 | Pika H3 Flex | 1117 | Oct 2026 | $2.10 |
| 4–5 | LTX-2.5 Fast | 914 | Aug 2026 | $7.80 |
| 4–5 | LTX-2.5 Pro | 913 | Aug 2026 | — |
| 6–7 | LTX-2.3 Pro / Fast | 842 / 835 | Mar 2026 | — |

  Hyperflow H3 and Pika H3 Flex appear to be derivatives of H3 (inferred from the name; not verified).
  Source: [AA open-weights I2V](https://artificialanalysis.ai/video/leaderboard/image-to-video/open-weights)
- Wan 2.2 and HunyuanVideo 1.5 do not appear on this current AA open-weights board. Either they are too old or they are not benchmarked under v2.1. — [AA open-weights I2V](https://artificialanalysis.ai/video/leaderboard/image-to-video/open-weights)

**MiniMax H3**
- Specs from the model card:
  - 33B dense single-stream transformer. About 13B of that sits in AdaLN branches that are not needed for inference.
  - 4–15 s clips at 24 fps. Default short side is 768 px, with up to 2K through the H3-Regenerate-2K module.
  - Aspect ratios include 9:16 and 3:4.
  - I2V through H3-Base-FL2VA (first and/or last frame) and Ref2VA (reference inputs).
  - Native 32 kHz stereo audio.
  - The checkpoints are CFG-distilled.
  - The SGLang example uses 4 GPUs (Ulysses degree 4). No VRAM or timing figures are published.
  - Source: [HF MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- License: **MiniMax H3 Community License**. Users in the **USA, EU, UK and South Korea must apply for a license** through a form. — [HF card](https://huggingface.co/MiniMaxAI/MiniMax-H3); [License Q&A](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md); [NYU RITS (secondary)](https://rits.shanghai.nyu.edu/ai/minimax-ships-h3-weights-with-the-us-and-eu-excluded/)
- Weights were reportedly published on Hugging Face on 2026-08-03. — secondary: [Atlas Cloud](https://www.atlascloud.ai/blog/guides/minimax-h3-open-source-weights)
- Pre-release press said commercial use is allowed under $20M revenue. **Unverified against the final LICENSE text.** — [SCMP](https://www.scmp.com/tech/article/3362540/video-ai-minimax-challenges-bytedance-low-price-open-weights-new-h3-model); [eWeek](https://www.eweek.com/news/minimax-h3-2k-ai-video-open-weights/)
- Other reported license conditions (secondary sources):
  - The model name must be displayed in products.
  - Outputs may not be distilled into other models.
  - The hosted version includes stages not in the downloadable H3-Base.
  - Sources: [deeplearning.ai The Batch](https://www.deeplearning.ai/the-batch/minimaxs-state-of-the-art-video-model-is-only-minimally-open); [Atlas Cloud](https://www.atlascloud.ai/blog/guides/minimax-h3-open-source-weights)

**LTX-2.5 (Lightricks), released 2026-08-11**
- Architecture: about 22B transformer (from file names).
  - Two transformers ship: a **distilled one with a fixed 8-step schedule and CFG=1**, and a trainable "dev" one.
  - A distilled LoRA and a Diffusers pack are also published.
- Generation:
  - Native audio (audio VAE plus vocoder).
  - **Native multi-shot generation** that keeps character, scene and voice consistent across cuts.
  - Width and height must be multiples of 32. Frame count must satisfy frames % 8 == 1; for example 121 frames is about 5 s.
  - 2x upscaler to 4K.
- Quantisation: fp8-cast, an NVFP4 distilled DiT (Blackwell only), and Comfy int8.
- Source: [HF Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5)
- License: **LTX-2.x Community License**.
  - Free for commercial use under **$10M annual recurring revenue**, measured across affiliates.
  - Fine-tunes and LoRAs inherit the license.
  - You must disclose machine-generated content and may not remove watermark or provenance features; doing so revokes the license.
  - It may not be used to train competing models.
  - Source: [VentureBeat 2026-08-11](https://venturebeat.com/technology/ltx-2-5-can-generate-a-10-second-ai-video-from-an-image-in-just-6-8-seconds-on-nvidia-superchips-and-its-open-weights); [HF card](https://huggingface.co/Lightricks/LTX-2.5)
- Earlier versions: LTX-2 was fully open-sourced in January 2026. LTX-2.3 (Mar 2026) added native audio. — [Wikipedia LTX-2](https://en.wikipedia.org/wiki/LTX-2) (secondary)

**Wan family (Alibaba)**
- **Wan 3.0 is closed, API-only.** It went into public beta around 2026-08-06, and the weights were kept closed.
  - The last open Wan flagship is Wan 2.2 (2025-07-28, Apache 2.0).
  - Alibaba open-sourced **Wan-Animate-2** (Apache 2.0) on 2026-08-07.
  - Sources: [eWeek](https://eweek.com/news/alibaba-wan3-ai-video-documents-apac-china/); [Atlas Cloud (secondary)](https://www.atlascloud.ai/blog/tips/is-wan-3.0-open-source)
  - Conflicting claim: [WaveSpeed](https://wavespeed.ai/blog/commercial-compliance/is-wan-30-open-source/) mentions an Apache-2.0 GitHub repo for Wan 3.0. That is unresolved; treat Wan 3.0 as closed.
- Wan 2.6 and 2.7 appear only as hosted APIs. One SEO page calls Wan 2.7 "Apache 2.0 open". A [RAVEN-Eval paper](https://arxiv.org/pdf/2608.09111) lists Wan 2.7 as closed-source. [Fuser](https://fuser.studio/articles/best-open-source-video-models) says Wan 2.2 is the newest Wan with published weights. **Treat Wan 2.5–3.0 as closed.**

**HunyuanVideo 1.5 (Tencent)**
- License: tencent-hunyuan-community. A third-party guide says territory limits apply.
  - Earlier Tencent Hunyuan licenses excluded the EU, UK and South Korea, but that comes from my prior knowledge, not a fetched source. Verify against the LICENSE file.
- Distilled variants:
  - A 480p I2V step-distilled model with 8–12 steps: "a single RTX 4090 can generate videos within 75 seconds".
  - A separate 720p step-distilled super-resolution model, for which no timing is published.
- lightx2v's distilled checkpoints claim the Apache 2.0 license, which conflicts with Tencent's license.
- Sources: [HF tencent/HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5); [lightx2v Hy1.5 distill](https://huggingface.co/lightx2v/Hy1.5-Distill-Models)

**SkyReels**
- V4 scores well on AA (Elo 1125, $21/min API). Its open-weight status is unclear: as of June 2026 reports said it was not released.
- V3 weights were released 2026-01-29. V1–V3 use the Skywork Community License, which allows commercial use under its terms.
- Sources: [AA I2V](https://artificialanalysis.ai/video/leaderboard/image-to-video); [SkyReels-V4 arXiv](https://arxiv.org/abs/2602.21818); [Skywork HF](https://huggingface.co/Skywork)

**Not researched:** Mochi, CogVideoX successors, Kandinsky 5 and MAGI. They are absent from the current AA open board and from the 2026 comparison pages I read; see Gaps.

### Inferences
- **MiniMax H3 may not be usable for a US- or EU-based operator** without a separately negotiated license. The answer depends on where the owner and company are domiciled; that needs checking. Its hosted API is available globally.
- **LTX-2.5 fits us** as a pre-revenue company under $10M. Its disclosure clause matches our policy anyway. Its watermark and provenance clause would need checking against our pipeline (re-encoding or stripping C2PA could breach it).
- **Wan 2.2 A14B / TI2V-5B are the only frontier-ish weights with a no-strings commercial license (Apache 2.0).** Their quality is now well below the 2026 frontier: they are not on the AA v2.1 board at all.

### Gaps
- Exact text of the MiniMax H3 LICENSE (revenue threshold, whether "excluded territory" means the user's domicile or the place of deployment).
- HunyuanVideo 1.5 territory clause, verified from the LICENSE file.
- Mochi, CogVideoX successors, Kandinsky 5, MAGI-1 successors and HappyHorse open-weight status: not researched (call budget).
- Wan-Animate-2 capabilities: not checked. It may be relevant to "walk-and-talk".

## 2. Acceleration: distillation, quantisation, realistic seconds per clip, VRAM

### Takeaway
Step distillation (4–8 steps) is now standard and gives speedups of 8–100x. Realistic single-GPU figures for a 5 s 720p clip:
- Wan 2.2 A14B 4-step: **~80 s on an H200** (community measurement).
- LTX-2.5 distilled: real-time or faster on datacenter GPUs, and runs on **16 GB** cards.

Our 14 min per clip on a 3090 with an undistilled TI2V-5B is far off the pace.

### Cited findings
**Wan 2.2**
- Wan 2.2 A14B with the Lightning LoRA (ComfyUI), community test on **one H200**, per 5 s clip:

  | Resolution | Base | 4-step Lightning |
  |---|---|---|
  | 1280×720 | 642.8 s | 79.0 s |
  | 640×640 | 151.8 s | 22.4 s |

  Source: [Jarvislabs blog (community)](https://jarvislabs.ai/blog/wan-2-2-lightning-different-video)
- LightWan2.2-A14B, 720p I2V, 4 steps: **26.7 s vs 2,685 s** for the 40-step baseline (100.5x). This is a vendor figure on **Blackwell** GPUs, not a 4090 or H100. — [HF lightx2v/LightWan2.2-A14B](https://huggingface.co/lightx2v/LightWan2.2-A14B)
- lightx2v's Wan2.2 distill models have separate configs for 80 GB GPUs (A100/H100) and 24 GB GPUs (RTX 4090). No timings are published. — [HF lightx2v/Wan2.2-Distill-Models](https://huggingface.co/lightx2v/Wan2.2-Distill-Models); [LightX2V GitHub](https://github.com/ModelTC/LightX2V)
- LightX2V's own per-step figures:
  - H100, 8 GPUs: 0.75 s/it with CFG, 0.39 s/it without.
  - 4090D, single GPU: 20.26 s/it.
  - These are older-model figures and not Wan 2.2 totals. — [LightX2V README](https://raw.githubusercontent.com/ModelTC/LightX2V/main/README.md)

**LTX-2.5**
- Self-hosted: 10 s 720p I2V in **6.8 s on two NVIDIA GB200** (vendor claim, steady state).
- Through the LTX API: 23.7 s at 1080p.
- Runs on "any GPU with at least 16GB of VRAM"; the distilled model runs on RTX cards.
- Source: [VentureBeat 2026-08-11](https://venturebeat.com/technology/ltx-2-5-can-generate-a-10-second-ai-video-from-an-image-in-just-6-8-seconds-on-nvidia-superchips-and-its-open-weights)
- End-to-end API latency measured by LTX (vendor, same task):

  | API | Time |
  |---|---|
  | Gemini Omni Flash | 52 s |
  | Grok 1.5 | 63 s |
  | Veo 3.1 (8 s clip) | 70 s |
  | MiniMax H3 | 180 s |
  | Seedance 2.5 | 317 s |
  | Kling 3.0 Pro | 398 s |

  Source: [VentureBeat](https://venturebeat.com/technology/ltx-2-5-can-generate-a-10-second-ai-video-from-an-image-in-just-6-8-seconds-on-nvidia-superchips-and-its-open-weights)
- Quantised community builds exist: LTX-2.5 distilled GGUF and int8 22B. — [HF Abiray GGUF](https://huggingface.co/Abiray/LTX-2.5-Distilled-GGUF); [HF deAPI int8](https://huggingface.co/deAPI-ai/ltx2-5-22b-dist-int8)

**HunyuanVideo 1.5**
- 480p step-distilled I2V: under 75 s on one RTX 4090, with end-to-end time cut by 75%. — [HF tencent/HunyuanVideo-1.5](https://huggingface.co/tencent/HunyuanVideo-1.5)
- lightx2v distill claims about 25x over 50-step inference. — [HF lightx2v Hy1.5](https://huggingface.co/lightx2v/Hy1.5-Distill-Models)

**MiniMax H3**
- No published VRAM or timing. The reference config is 4 GPUs with sequence parallel. Community 4-step Turbo LoRAs exist but are undocumented. — [HF card](https://huggingface.co/MiniMaxAI/MiniMax-H3)

### Inferences
- **Wan 2.2 A14B + Lightning, 720p 5 s:**
  - H100: probably ~100–130 s. This is extrapolated from the H200 79 s with a ~1.3–1.6x slower factor; unverified.
  - 4090 (24 GB, FP8 plus offload): probably several minutes. Unverified; no measured source was found.
- **Same family, our current model:** a 4-step Lightning/lightx2v LoRA for TI2V-5B, if one exists, would likely cut the 14 min on the 3090 to roughly 2–3 min. Not verified; we need our own benchmark.
- **MiniMax H3 at 33B with a 4-GPU reference** is a poor fit for cheap interruptible single consumer cards. Expect a single 80 GB card (H100) at minimum.
- **LTX-2.5 distilled at 16 GB+** is the only frontier-class open model plausibly fast on 4090/L40S-class interruptible cards. That suits our Salad/Vast pools, which have less VRAM than advertised.
- Prior knowledge, not sourced here: TeaCache and FBCache add roughly 1.5–2x on undistilled models but little on 4-step distilled ones. Sequence parallel (Ulysses/xDiT) needs multiple GPUs, which conflicts with our single-card interruptible model.

### Gaps
- No measured seconds-per-clip on a 4090 or L40S for Wan 2.2 4-step 720p, LTX-2.5 distilled 720p, or MiniMax H3. All need our own benchmark (`mw bench`).
- RAM needs (host RAM for offload) for each model were not found.
- Whether a 4-step distilled LoRA exists for Wan 2.2 TI2V-5B specifically was not verified.

## 3. Hosted APIs: per-second prices and native audio

### Takeaway
At 720p, a 5 s clip from a hosted API now costs **$0.10 to $2.40**:

| Tier | 5 s clip | Examples |
|---|---|---|
| Cheapest | $0.10–$0.35 | Wan 2.2 Turbo on fal, MiniMax H3 Max Turbo, PixVerse C1, Kling 2.6 Pro, Veo 3.1 Lite |
| Mid | $0.45–$1.00 | LTX-2.5, Wan 3.0, Gemini Omni Flash, Kling 3/4, Runway Gen-4.5, Luma Ray 3.2 |
| Frontier | $1.50–$2.40 | Seedance 2.0/2.5, Veo 3.1 with audio |

**The Sora 2 API was shut down on 2026-09-24.** Native audio is now common: Veo 3.1, Seedance 2.x, Kling 3, PixVerse C1, LTX-2.x, MiniMax H3 and Gemini Omni Flash (AA with-audio board).

### Cited findings
**fal.ai prices, per [TeamDay fal pricing roundup dated 2026-11-02](https://www.teamday.ai/blog/fal-ai-pricing-2026)** (secondary; it reproduces fal's pages):

| Model | 720p per second | 5 s clip | Audio and notes |
|---|---|---|---|
| Kling 2.6 Pro | $0.07 | $0.35 | Audio $0.14/s |
| Kling 3 Pro | $0.112 | $0.56 | Audio $0.168/s; voice control $0.196/s |
| Kling 4 | $0.157 (1080p $0.235) | $0.79 | — |
| Veo 3.1 | $0.40 with audio, $0.20 without | $2.00 with audio | Endpoint **ending 2026-11-17** |
| Seedance 2.0 Mini | ≈$0.155 | — | Audio optional |
| Seedance 2.0 Fast | ≈$0.242 | — | Audio included |
| Seedance 2.0 | ≈$0.303 (1080p ≈$0.68) | ≈$1.52 | Audio included |
| Seedance 2.5 (US) | ≈$0.473 | ≈$2.37 | Audio included |
| MiniMax H3 Max Turbo | $0.04 | $0.20 | 768p |
| MiniMax H3 Max | $0.08 | $0.40 | 768p; promo prices ended 2026-11-15 |
| Wan 2.6 | $0.10 | $0.50 | — |
| Wan 3.0 | $0.10 (1080p $0.20) | $0.50 | — |
| Wan 3.0 Prime | $0.14 | $0.70 | — |
| LTX-2.5 Pro | $0.12 | $0.60 | — |
| Luma Ray 3.2 | $0.20 | $1.00 | — |
| PixVerse C1 | $0.05 ($0.065 with audio) | $0.25 | — |
| Gemini Omni Flash 1.1 | $0.10 | $0.50 | — |

**LTX's own API**
- Fast: $0.09/s, with 720p and audio, 10 s clips, up to 20 s, 4K at $0.30/s.
- Pro: $0.12/s, up to 1080p.
- Comparison rates in the same article: Veo 3.1 Lite $0.05/s, Veo 3.1 Fast and Gemini Omni Flash $0.10/s, FLUX 3 Video $0.17/s.
- Source: [VentureBeat 2026-08-11](https://venturebeat.com/technology/ltx-2-5-can-generate-a-10-second-ai-video-from-an-image-in-just-6-8-seconds-on-nvidia-superchips-and-its-open-weights)

**Wan 2.2 A14B on fal**
- Text-to-video Turbo: **$0.10 per video at 720p**, $0.075 at 580p, $0.05 at 480p. — [fal Wan 2.2 T2V turbo](https://fal.ai/models/fal-ai/wan/v2.2-a14b/text-to-video/turbo)
- Image-to-video Turbo price was not confirmed from fal. Secondary sources say $0.05–$0.10 per video. — [aiidelist](https://aiidelist.com/blog/cheapest-wan-2-2-api-pricing-2026); [fal I2V turbo page](https://fal.ai/models/fal-ai/wan/v2.2-a14b/image-to-video/turbo)

**Other providers**
- **Sora 2 and Sora 2 Pro API and the Videos API were shut down on 2026-09-24.** The last price was $0.10/s at 720p; Pro was $0.30–0.70/s. — [OpenAI docs](https://developers.openai.com/api/docs/models/sora-2); [benchlm (secondary)](https://benchlm.ai/media-pricing/sora)
- **Runway Gen-4.5:** 12 credits/s at $0.01 per API credit, so **$0.12/s and $0.60 per 5 s clip**. Krea lists $0.126/s. — [Runway API pricing docs](https://docs.dev.runwayml.com/guides/pricing/); [Krea docs](https://www.krea.ai/docs/api-reference/video/runway-gen-45.md)
- **Creator API price per minute on the AA I2V board** (with $/5 s computed):

  | Model | $/min | $/5 s |
  |---|---|---|
  | Seedance 2.5 | $34.12 | ≈$2.84 |
  | Seedance 2.0 | $22.45 | ≈$1.87 |
  | SkyReels V4 | $21.00 | — |
  | Grok Imagine Video 1.5 | $15.00 | — |
  | FLUX 3 | $14.40 | — |
  | Wan 3.0 | $12.00 | — |
  | MiniMax H3 Max | $9.60 | — |
  | Hyperflow H3 | $9.00 | — |
  | Gemini Omni Flash 1.2 | $6.26 | ≈$0.52 |
  | P-Video-2-Pro | $5.70 | — |
  | MiniMax H3 768p | $4.80 | ≈$0.40 |
  | Boreal-H3 | $2.42 | ≈$0.20 |
  | HeyGen Video 1 | $2.40 | — |
  | Agnes-Video-2.5 | $2.40 | — |
  | Pika H3 Flex | $2.10 | ≈$0.175 |

  Prices are at benchmark settings, 1080p where offered. — [AA I2V](https://artificialanalysis.ai/video/leaderboard/image-to-video)
- Hailuo: not seen under that name in late-2026 sources. MiniMax's current line is H3.
- Not found: Pika's own API pricing (only Pika H3 Flex at $2.10/min on AA) and Together AI video pricing.

### Inferences
- Many fal prices are promotional and churn monthly (several promos ended mid-November). Endpoints get deprecated (Veo 3.1 on fal on 11-17; Sora shut down). **An adapter-per-vendor design and re-checking prices monthly are mandatory.**
- Cheap tiers ($0.04–0.07/s) sit on newer, stronger models than Wan 2.2: MiniMax H3 Max Turbo, PixVerse C1, Kling 2.6 Pro. They undercut realistic self-hosting once you count boot and model download time.

### Gaps
- Kling's and ByteDance's direct (non-fal) API prices; Alibaba Model Studio Wan prices; Replicate and Together current pages. Not fetched; the call limit was reached.
- Whether MiniMax H3 Max Turbo on fal includes audio, and its I2V identity fidelity.
- fal's own pages for each model were not opened individually. All fal figures come from one 2026-11-02 secondary roundup and should be re-verified.

## 4. Quality leaderboards (late 2026)

### Takeaway
On AA's image-to-video (with audio) board the leaders are **Seedance 2.5 (1162)**, **Boreal-H3 (1154)**, **Utopai X (1153)** and **Seedance 2.0 (1142)**. **MiniMax H3 (1137) is 5th overall and the top open-weights model.** On the silent board, Utopai X leads (1150), followed by Seedance 2.5 (1126). The confidence interval is about ±10–17 Elo, so places 2–11 are close to a statistical tie.

### Cited findings
- **AA-Video-I2V v2.1, with audio, default view** (fetched 2026-11-21):

  | Rank | Model | Elo |
  |---|---|---|
  | 1 | Seedance 2.5 | 1162 |
  | 2 | Boreal-H3 (Creatify) | 1154 |
  | 3 | Utopai X (no API) | 1153 |
  | 4 | Seedance 2.0 | 1142 |
  | 5 | MiniMax H3 768p | 1137 |
  | 6 | HeyGen Video 1 | 1127 |
  | 7 | SkyReels V4 | 1125 |
  | 8 | Hyperflow H3 | 1121 |
  | 9 | MiniMax H3 Max | 1120 |
  | 10 | P-Video-2-Pro | 1119 |
  | 11 | Pika H3 Flex | 1117 |
  | 12 | Wan 3.0 | 1110 |
  | 13 | Agnes-Video-2.5 | 1103 |
  | 14 | HiDream-O1 | 1103 |
  | 15 | Gemini Omni Flash 1.2 (Nov 2026) | 1086 |
  | 16 | FLUX 3 | 1072 |
  | 17 | FLUX 3 Max | 1067 |
  | 18 | Grok Imagine Video 1.5 | 1047 |
  | 19 | Gemini Omni Flash 1.1 | 1046 |
  | 20 | HappyHorse-1.1 | 1037 |

  Kling 3/4, Veo 3.1, Runway Gen-4.5 and Luma Ray 3.2 are **not in the top 20** of this view. LTX-2.5 scores 913–914. — [AA I2V](https://artificialanalysis.ai/video/leaderboard/image-to-video); [AA open-weights](https://artificialanalysis.ai/video/leaderboard/image-to-video/open-weights)
- **AA silent I2V top 5:** Utopai X 1150, Seedance 2.5 1126, MiniMax H3 Max 1124, Bach 1.0 Pro 1121, Hyperflow H3 1121. — [AA I2V](https://artificialanalysis.ai/video/leaderboard/image-to-video)
- **AA method:** Elo from blind pairwise human preference votes on the same input image. Separate audio and silent boards. Only 5% of prompts are public, and the subset rotates monthly. — [AA methodology](https://artificialanalysis.ai/methodology/video)
- **Older snapshots conflict with today's board.** In Aug 2026, Gemini Omni Flash led both T2V arenas. An earlier I2V snapshot had Seedance 2.0 at 1192 and H3 at 1185. Treat these as dated. — [VentureBeat](https://venturebeat.com/technology/ltx-2-5-can-generate-a-10-second-ai-video-from-an-image-in-just-6-8-seconds-on-nvidia-superchips-and-its-open-weights); [AlphaSignal](https://alphasignal.ai/news/artificial-analysis-ranks-seedance-2-5-and-minimax-h3-in-video-ai-benchmark)
- **LTX's vendor preference test** (preliminary, vendor-run) gave LTX-2.5 a 67% win rate. The same test gave Seedance 2.5 65%, Gemini Omni Flash 55%, H3 50%, Seedance 2.0 44%, Wan 2.6 42% and FLUX 3 28%. **This contradicts AA, where LTX-2.5 scores about 914 against about 1137 for H3.** Prefer AA. — [VentureBeat](https://venturebeat.com/technology/ltx-2-5-can-generate-a-10-second-ai-video-from-an-image-in-just-6-8-seconds-on-nvidia-superchips-and-its-open-weights)
- **Other arena:** arena.ai also runs an I2V arena; it was not fetched. — [arena.ai I2V](https://arena.ai/en/leaderboard/image-to-video)
- **VBench:** no late-2026 VBench results were found.

### Inferences
- **Price versus quality outliers on AA:**
  - Boreal-H3 (1154, $2.42/min ≈ $0.20/5 s) and Pika H3 Flex (1117, $2.10/min) sit near the top at very low prices. Both appear to be H3-derived.
  - MiniMax H3 768p (1137, $4.80/min ≈ $0.40/5 s) is the clear value pick among first-party models.
  - Seedance 2.5 costs about 7x more for +25 Elo.
- AA's Elo measures general human preference, not **identity preservation against the source photo**. Our gate measures identity, so we need our own identity-gated acceptance rate per model.

### Gaps
- Identity- or character-consistency benchmarks for I2V, beyond general preference.
- The AA text-to-video board and the full silent I2V board were not fetched.

## 5. Cost per 5 s clip: self-hosted vs API at hundreds of clips per month, and a recommendation

### Takeaway
At hundreds of clips a month, **the GPU-time saving from self-hosting is tens of dollars a month at most**. Boot time, model downloads (tens of GB) and idle billing on interruptible machines can erase it, and the weights we can license freely (Wan 2.2) are a generation behind. The cheapest frontier-adjacent APIs cost **$0.10–$0.40 per 5 s 720p clip**.

Recommendation:
- Add hosted-API adapters: MiniMax H3 (Turbo/Max), Kling 2.6/3 Pro, Wan 2.2 Turbo or Wan 3.0 on fal.
- Keep self-hosting only for a distilled Wan 2.2 or LTX-2.5 path.
- Choose by measured cost per **accepted** clip.

### Cited findings
- Today: 14 min per 720p 5 s clip on a RunPod 3090 at ~$0.22/h is about $0.05 of GPU time per clip, excluding boot, download and idle. The inputs come from the project CLAUDE.md; the result is my arithmetic.
- **Hosted, per 5 s 720p clip:**

  | Option | Cost per clip | Source |
  |---|---|---|
  | Wan 2.2 A14B Turbo, fal | $0.10 (stated for T2V) | [fal](https://fal.ai/models/fal-ai/wan/v2.2-a14b/text-to-video/turbo) |
  | MiniMax H3 Max Turbo | $0.20 | [TeamDay](https://www.teamday.ai/blog/fal-ai-pricing-2026) |
  | PixVerse C1 | $0.25 | [TeamDay](https://www.teamday.ai/blog/fal-ai-pricing-2026) |
  | Veo 3.1 Lite | $0.25 | [VentureBeat](https://venturebeat.com/technology/ltx-2-5-can-generate-a-10-second-ai-video-from-an-image-in-just-6-8-seconds-on-nvidia-superchips-and-its-open-weights) |
  | Kling 2.6 Pro | $0.35 | [TeamDay](https://www.teamday.ai/blog/fal-ai-pricing-2026) |
  | MiniMax H3 Max | $0.40 | [TeamDay](https://www.teamday.ai/blog/fal-ai-pricing-2026) |
  | LTX-2.5 Fast | $0.45 | [VentureBeat](https://venturebeat.com/technology/ltx-2-5-can-generate-a-10-second-ai-video-from-an-image-in-just-6-8-seconds-on-nvidia-superchips-and-its-open-weights) |
  | Wan 3.0 / Gemini Omni Flash | $0.50 | [TeamDay](https://www.teamday.ai/blog/fal-ai-pricing-2026) |
  | Kling 3 Pro | $0.56 | [TeamDay](https://www.teamday.ai/blog/fal-ai-pricing-2026) |
  | Runway Gen-4.5 | $0.60 | [Runway](https://docs.dev.runwayml.com/guides/pricing/) |
  | Seedance 2.0 | ≈$1.52 | [TeamDay](https://www.teamday.ai/blog/fal-ai-pricing-2026) |
  | Veo 3.1 with audio | $2.00 | [TeamDay](https://www.teamday.ai/blog/fal-ai-pricing-2026) |
  | Seedance 2.5 | ≈$2.37 | [TeamDay](https://www.teamday.ai/blog/fal-ai-pricing-2026) |

### Inferences
**Self-hosted estimates** (unverified; inference only, rental prices assumed rather than sourced here):
- Wan 2.2 A14B + 4-step Lightning on an H100: about 100–130 s per 720p clip. At an assumed ~$2–3/h for H100 rental, that is **~$0.06–0.11 per clip** of pure compute.
- On a 4090 at an assumed ~$0.35–0.6/h, with an unmeasured several minutes per clip: roughly **$0.03–0.06**.
- Both are at or near fal's $0.10 Wan 2.2 Turbo price before overheads. The talk image already showed a ~94 GB weights download per machine boot, and A14B weights are tens of GB.

**Monthly cost at 500 accepted clips/month and a 50% gate acceptance rate (1,000 generations):**

| Option | Monthly cost |
|---|---|
| Self-hosted Wan 2.2 | ~$50–100 |
| Wan 2.2 Turbo on fal | ~$100 |
| MiniMax H3 Max Turbo | ~$200 |
| Kling 3 Pro | ~$560 |
| Seedance 2.5 | ~$2,400 |

- The 50% acceptance rate is an assumption, not a measurement.
- **The acceptance rate moves cost per accepted clip more than the sticker price does.** A $0.40 model accepted 80% of the time ($0.50 per accepted clip) beats a $0.10 model accepted 15% of the time ($0.67).

**Recommendation for the report writer** (my synthesis; verify against the owner's priorities):
1. **Short term (story clip, one simple motion):**
   - Benchmark hosted MiniMax H3 Max Turbo (~$0.20), Kling 2.6 Pro (~$0.35), Wan 2.2 A14B Turbo on fal (~$0.10) and PixVerse C1 (~$0.25) against our identity and video gate.
   - Pick by cost per accepted clip.
   - This fits the "every capability behind an adapter" rule and avoids the 14 min per clip on a 3090.
2. **Self-hosted path:**
   - Switch to Wan 2.2 A14B (or TI2V-5B if a distill LoRA exists) with lightx2v/Lightning 4-step distillation and FP8, on a 48 GB+ card (A40/L40S) or an H100.
   - Alternatively, LTX-2.5 distilled (8 steps, 16 GB+, native audio and multi-shot, free under $10M revenue, but disclosure and provenance clauses apply).
   - Expect about 1–3 min per clip. Measure it with `mw bench`.
3. **Richer videos (walk-and-talk, multi-shot, native audio):**
   - Candidates: LTX-2.5 (native multi-shot), MiniMax H3 (best open quality, 4–15 s, stereo audio, but license excludes US/EU/UK/KR), Seedance 2.x and Kling 3 (top-tier hosted, $0.56–$2.40 per 5 s).
   - Reserve the premium tier for hero posts only.
4. **Avoid:**
   - Sora 2 (shut down).
   - Veo 3.1 on fal (endpoint ended 2026-11-17).
   - Building self-hosted MiniMax H3 before legal review, because of the territory exclusion.
   - HunyuanVideo 1.5 before checking its territory clause.

### Gaps
- No measured identity-preservation or acceptance rates per model. This must come from our own benchmark.
- Current RunPod, Vast and Salad hourly prices for the H100, L40S and 4090 were not fetched in this pass. The self-host cost figures above use assumed rates.
- Content-policy behaviour of the hosted APIs for realistic adult people (refusals of photoreal humans, I2V face-upload restrictions on Kling, Veo and Seedance) was not researched. It could affect acceptance rates strongly.
