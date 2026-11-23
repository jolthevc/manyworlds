# Image models for photorealistic, identity-consistent people (landscape as of 2026-11-21)

Method note: about 17 searches and fetches on 2026-11-21. The Artificial Analysis (AA) leaderboards were fetched directly, so the Elo and price figures below are that day's snapshot. AA shows no last-updated date, and Elos carry confidence intervals (rank 1 is shown as "1-3"), so a gap of 10 to 20 Elo is noise. AA Elo is an overall text-to-image preference score across all prompt types, not a people/photorealism score; I found no people-specific leaderboard. Several model names below (FLUX 3 Image Klein, H3-Image, Qwen-Image-2.1, MAI-Image-2.7, GPT Image 2.6, Seedream 5.5) are newer than my training data. I confirmed each against at least AA plus one vendor or press source, but most license and spec details come from secondary write-ups unless a primary link is given.

## 1. Which open-weight models lead, and their commercial licenses

### Takeaway
Open-weight quality has moved well past Z-Image Turbo (AA open-weights Elo 936, rank 19 of 36). Most models ranked above it are either non-commercial or too large for a 24 GB card. Two are viable for us on cheap consumer GPUs:
- **FLUX 3 Image Klein** (Elo 1040, 5.6B, LoRA support from day one), whose license is free for commercial use only below $5M annual revenue.
- **HiDream-O1-Image** (Elo 979, 8B, MIT license).

### Cited Findings

**AA open-weights text-to-image leaderboard (v2.1, fetched 2026-11-21).** Prices are per 1,000 images at 1024² on the creator's API. [AA open-weights](https://artificialanalysis.ai/image/leaderboard/text-to-image/open-weights)

| # | Model | Elo | $/1k | Released |
|---|---|---|---|---|
| 1 | MiniMax H3-Image | 1066 | $21.79 | Nov 2026 |
| 2 | Qwen-Image-2.1 | 1047 | no API | Sep 2026 |
| 3 | FLUX 3 Image Klein | 1040 | $9.00 | Oct 2026 |
| 4 | Bria Fibo 2.0 | 1036 | $32 | — |
| 8 | FLUX.2 [dev] (anchor) | 1000 | $12 | — |
| 9 | FLUX.2 [dev] Turbo (fal) | 996 | $8 | — |
| 12 | HunyuanImage 3.0 Instruct | 986 | — | — |
| 13 | FLUX.2 dev Flash (fal) | 985 | $5 | — |
| 16 | HiDream-O1-Image | 979 | $40 | May 2026 |
| 19 | **Z-Image Turbo** | **936** | **$5** | Dec 2025 |
| 20 | HunyuanImage 3.0 | 933 | — | — |
| 21 | FLUX.2 klein 9B | 932 | — | — |
| 22 | ERNIE Image Turbo | 916 | — | — |
| 23 | Qwen Image (orig.) | 909 | — | — |
| 29 | FLUX.2 klein 4B | 840 | — | — |
| 33 | HiDream-O1-Image-Dev | 808 | $5 | — |

Not ranked in this snapshot: NVIDIA Cosmos3-Super-Text2Image (995) and its 4-step variant (973), Ideogram 4.0 (1027), FLUX.1 Krea dev (868), FLUX.1 dev (813), schnell (742).

**Overall AA board (open and closed together).** H3-Image, the open-weights leader, ranks only 27th overall at Elo 1066, against 1263 for the overall #1 (GPT Image 2.6 Sunburst). Closed models therefore lead open weights by about 200 Elo. [AA overall](https://artificialanalysis.ai/image/leaderboard/text-to-image)

**Model by model**

- **FLUX 3 Image Klein**
  - Released 2026-10-15. 5.6B parameters.
  - Two checkpoints: a base model for fine-tuning (50 steps) and a step-distilled model (16 steps).
  - About 4 s per 1MP image on an RTX 5090 and under 9 s on an RTX 4070 (distilled, INT8, ComfyUI).
  - VRAM: 11.2 GB in BF16, 5.6 GB in FP8.
  - Up to 10 reference images.
  - LoRA training supported via Diffusers, Ostris AI Toolkit and NeMo.
  - API from $0.009 per image.
  - Source: [BFL blog](https://bfl.ai/blog/flux-3-image-klein)
- **FLUX 3 Image Klein license.** The "FLUX Kommunity License" allows free commercial use for individuals and companies under $5M annual revenue. Larger organizations need self-hosted licensing or the API. The blog says nothing about rights to outputs. [BFL blog](https://bfl.ai/blog/flux-3-image-klein); [VentureBeat](https://venturebeat.com/technology/black-forest-labs-opens-flux-3-image-klein-weights-with-free-commercial-use-limited-to-smaller-businesses)
- **Klein benchmark caveats.** BFL's "leads open weights" claims are internal. VentureBeat notes that the hosted version adds prompt-processing steps, so local output may differ from the API. [VentureBeat via search summary](https://venturebeat.com/technology/black-forest-labs-opens-flux-3-image-klein-weights-with-free-commercial-use-limited-to-smaller-businesses)
- **FLUX.2 klein.** The 4B models (distilled and base) are Apache 2.0, commercial OK. The 9B models are under the FLUX Non-Commercial License (renamed from "FLUX [dev] Non-Commercial License"), and commercial use needs a separate BFL licence. [BFL klein blog](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence?lid=ECSSqRZRPUuf3RTN); [HF FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B); [VentureBeat](https://venturebeat.com/technology/black-forest-labs-launches-open-source-flux-2-klein-to-generate-ai-images-in)
- **FLUX.2 [dev].** Listed under the FLUX Non-Commercial license, so commercial use of the weights needs a paid BFL licence. [kenerateai license table (secondary)](https://kenerateai.com/open-source-ai-image-generator). fal's distilled Turbo and Flash variants are paid hosted endpoints, where commercial use is covered by the endpoint. **Unverified:** fal's exact terms.
- **Qwen-Image-2.1**
  - Released 2026-09-20. About 7B, unified generation and editing.
  - Up to 10 reference images per edit; Qwen claims better facial identity preservation across edits.
  - Day-0 Diffusers and ComfyUI support.
  - **License: Qwen Research License, research and evaluation only; commercial use needs a separate paid licence.** [search summary of Qwen-Image-2.1 coverage](https://rits.shanghai.nyu.edu/ai/qwen-image-2-1-native-transparency-and-10-image-editing/); [README](https://raw.githubusercontent.com/QwenLM/Qwen-Image-2.1/main/README.md); [Alibaba Cloud blog](https://alibabacloud.com/blog/qwen-image-2-1-compact-efficient-and-unified-image-creation_603586)
  - The older Qwen-Image (Aug 2025) and Qwen-Image-2512 were Apache 2.0. Not verified against the license files this session.
  - Qwen-Image-3.0 and 3.0-Pro are closed, API only (overall Elo 1075 and 1094; $30 and $40 per 1k). [AA overall](https://artificialanalysis.ai/image/leaderboard/text-to-image)
- **MiniMax H3-Image**
  - 33.1B-parameter DiT. BF16 checkpoint about 68.65 GB, plus a 66.7 GB text encoder.
  - Distilled to 6 steps without CFG. Generation and editing with up to 16 reference images.
  - Released about 2026-11-10 ([aiweekly](https://aiweekly.co/alerts/minimax-open-sources-h3-image-331b-parameter-diffusion-model)).
  - License: "MiniMax-H3-Image Community License"; some uses need written authorization requested through a form (Section IV). [HF model card](https://huggingface.co/MiniMaxAI/minimax-h3-image)
  - Secondary sources report that the H3-family licence excludes the US, EU, UK and South Korea, and that MiniMax's license FAQ contradicts the operative text. **Not verified for H3-Image specifically.** [NYU RITS](https://rits.shanghai.nyu.edu/ai/minimax-ships-h3-weights-with-the-us-and-eu-excluded/); [MiniMax license FAQ](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md)
- **HiDream-O1-Image.** 8B, released about 2026-05-08, MIT licence for code and models, Dev-2604 variant included. The fal hosted endpoint costs $40 per 1k per AA. [HF HiDream-O1-Image-Dev-2604](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev-2604); [HF HiDream-O1-Image](https://huggingface.co/HiDream-ai/HiDream-O1-Image)
- **Z-Image family**
  - Z-Image Turbo: Apache 2.0, 6B, 8 steps ([invideo spec page, secondary](https://invideo.io/blog/z-image-ai-image-generator/)).
  - The non-distilled Z-Image (base) was released around 2026-01-27/28 and is described as ideal for fine-tuning. [GIGAZINE](https://gigazine.net/gsc_news/en/20260128-z-image-image-generation-ai/); [ComfyUI wiki](https://comfyui-wiki.com/en/news/2026-01-28-alibaba-z-image-base-release)
  - Z-Image-Edit and Z-Image-Omni-Base were originally planned for Q1 2026 and were still "to be released" as of Jun–Aug 2026. [GitHub issue #169](https://github.com/Tongyi-MAI/Z-Image/issues/169); [localaimaster](https://localaimaster.com/blog/z-image-base-edit-guide)
- **Low-quality sources.** Several "best open-source 2026" blogs are SEO content with inconsistent specs. [perplexityaimagazine](https://perplexityaimagazine.com/ai-tools/best-open-source-image-generation-models/); [sevenlabs](https://www.sevenlabs.site/blogs/open-source-image-generation-models-2026)

### Inferences
- **H3-Image** does not fit a 24 GB card at BF16: about 135 GB of weights including the text encoder. It would need H100/H200-class hardware or heavy quantization. Add the licence ambiguity, possibly excluding the US/EU, and it is not a fit for us today.
- **Qwen-Image-2.1 and FLUX.2 dev / klein-9B** are excluded on licence unless we pay for one.
- **FLUX 3 Image Klein** is the most interesting upgrade: about +100 AA Elo over Z-Image Turbo, a similar VRAM class, and LoRA tooling. It is commercially free only while the business earns under $5M a year, so record that as a policy and licence risk. Revenue is the owner's to confirm.
- **HiDream-O1-Image** (MIT) is the cleanest-licensed model that beats Z-Image Turbo (+43 Elo). Its speed on a 4090 is unknown.
- **FLUX.2 klein 4B** (Apache) scores below Z-Image Turbo (840 vs 936), so it is not an upgrade.

### Gaps
- Licence of Bria Fibo 2.0. Earlier Bria open weights were non-commercial and their search was not run, so it is unknown.
- Licences of HunyuanImage 3.0, Cosmos3, ERNIE Image and Ming-Image were not checked.
- No people-specific or photorealism-specific Elo split was found.
- 4090 and 3090 seconds per image for FLUX 3 Klein and HiDream-O1 are unmeasured.

## 2. Identity consistency: reference editing vs per-creator LoRA

### Takeaway
No rigorous large-sample comparison exists. The evidence available is vendor blogs and small pilots. They suggest:
- Reference editing (Seedream, Nano Banana) is good for short runs but drifts across many edits.
- A LoRA is still the standard for one identity reused across hundreds of scenes.
- For us that argues for keeping the LoRA path and benchmarking a reference-conditioned model as a second candidate.

### Cited Findings
- **img.ly pilot benchmark.** On 3-scene character consistency, Seedream 5.0 Pro scored 4.86/5 and Nano Banana Pro 3.89/5. Only 1 character and 9 runs per model. [img.ly consistency](https://img.ly/ai-benchmarks/findings/consistency/)
- **Hands-on test (secondary, via search summary).** Seedream kept identity better over multi-round edits, while Nano Banana showed "significant identity drift" in later iterations. [Higgsfield](https://higgsfield.ai/blog/Why-Does-Your-AI-Characters-Face-Keep-Changing)
- **Practitioner guides.** They recommend LoRA "for stable characters" and for strict identity reuse across hundreds of scenes, with references for one-off or fast-changing characters. A typical LoRA uses 15 to 30 reference images. [RunDiffusion](https://learn.rundiffusion.com/consistent-character-template); [invideo](https://invideo.io/faq/lora-fine-tuning-vs-reference-sheets-for-ai-character/)
- **Open models with multi-reference input.** FLUX 3 Image Klein takes up to 10 references, and also LoRA ([BFL](https://bfl.ai/blog/flux-3-image-klein)). Qwen-Image-2.1 takes up to 10, but its licence is non-commercial. H3-Image takes up to 16 ([HF](https://huggingface.co/MiniMaxAI/minimax-h3-image)).
- **Open editing leaders on AA.** H3-Image 1136, Fibo 2.0 1086, Qwen-Image-2.1 1082. [AA editing](https://artificialanalysis.ai/image/leaderboard/editing)
- **Closed editing leaders on AA.** MAI-Image-2.7 1215 ($34.82/1k), GPT Image 2.6 Sunburst 1211 ($210.72/1k), Kling Image 4.0 1183, MAI-Image-2.7-Flash 1179 ($10.24/1k), Seedream 5.5 1159 ($28/1k), Nano Banana 2.1 1152 ($33.60/1k). [AA editing](https://artificialanalysis.ai/image/leaderboard/editing)

### Inferences
- **The hybrid worth testing** is FLUX 3 Klein with a creator LoRA plus the anchor images as references. Both mechanisms run in one model on a consumer GPU.
- **Hosted reference editing at volume** would cost $10 to $35 per 1k before rejections, versus well under $1 per 1k self-hosted.
- **AA editing Elo measures edit quality, not identity retention.** The review gate's identity score is the real arbiter.

### Gaps
- No public benchmark measures face-embedding similarity (ArcFace or similar) across more than 50 scenes for LoRA vs reference-conditioning.
- No data on MAI-Image-2.7 or Seedream 5.5 identity retention, or on their commercial terms for synthetic-person accounts.

## 3. Hosted API prices vs self-hosting on a ~$0.3/h 4090

### Takeaway
Self-hosting Z-Image Turbo costs about **$0.0002–0.0004 per image** on a $0.3/h 4090. That is 10–25x cheaper than the cheapest hosted endpoint ($0.005) and 100x or more cheaper than frontier closed APIs ($0.03–0.21). FLUX 3 Klein self-hosted is probably about 2–3x the cost of Z-Image Turbo, still under $0.001.

### Cited Findings

AA prices are per 1,000 images at 1024².

**Cheapest hosted options**

| Model | Price | Source |
|---|---|---|
| Z-Image Turbo | $5 per 1k (AA); fal $0.005/MP | [AA open](https://artificialanalysis.ai/image/leaderboard/text-to-image/open-weights); [asset-lab listing citing fal, Oct 2026](https://assetlab.coastalcreativelab.com/models/) |
| FLUX.2 dev Flash | $5 per 1k | [AA open](https://artificialanalysis.ai/image/leaderboard/text-to-image/open-weights) |
| FLUX 3 Image Klein | $0.009 per image | [BFL](https://bfl.ai/blog/flux-3-image-klein) |
| MAI-Image-2.7-Flash | $10.24 per 1k | [AA overall](https://artificialanalysis.ai/image/leaderboard/text-to-image) |
| Meta Muse Image | $10 per 1k | [AA overall](https://artificialanalysis.ai/image/leaderboard/text-to-image) |

**Mid-tier and frontier closed APIs** ([AA overall](https://artificialanalysis.ai/image/leaderboard/text-to-image))

| Model | Price per 1k |
|---|---|
| Seedream 5.5 | $20 |
| Nano Banana 2.1 | $33.60 |
| MAI-Image-2.7 | $34.82 |
| Kling Image 4.0 | $63 |
| Nano Banana 2 | $67 |
| Seedream 5.0 Pro | $90 |
| Nano Banana Pro | $134 |
| GPT Image 2.x (max/high) | about $211 |

**fal-specific prices** (secondary aggregation, Oct–Nov 2026; fal's pricing page was not fetched directly):
- Nano Banana 2: $0.08/image at 1K.
- Nano Banana Pro: $0.15/image.
- Seedream 5.0 Lite / Flash / Pro: $0.035 / $0.027 / $0.0675.
- FLUX 3: $0.024/MP or $0.048 at 1K; the two sources conflict.

[teamday comparison](https://www.teamday.ai/blog/ai-image-video-api-providers-comparison-2026); [fal Nano Banana vs Seedream](https://fal.ai/learn/tools/nano-banana-vs-seedream)

**Self-hosted performance**
- **Z-Image Turbo:** about 2.3 s per 1024² image on a 4090 at 8 steps; others report 2–5 s on consumer cards. VRAM: 16 GB in BF16, 8 GB in FP8, 6 GB as GGUF; 24 GB leaves headroom for LoRA and batching. Recommended CFG is 1.5–2; higher CFG degrades output. [codesota](https://www.codesota.com/news/z-image-turbo-consumer-gpu); [runaihome](https://runaihome.com/blog/z-image-turbo-16gb-vram-home-lab-image-gen-2026/); [thundercompute](https://www.thundercompute.com/blog/z-image-turbo-comfyui)
- **FLUX 3 Klein distilled INT8:** about 4 s on a 5090 and under 9 s on a 4070. [BFL](https://bfl.ai/blog/flux-3-image-klein)
- **Not checked this session:** Together, Replicate and Runware per-image prices. The search did not surface primary pricing pages.

### Inferences
**Self-hosted cost arithmetic** at $0.30/h:
- At 2.3 s per image: about 1,565 images/h, roughly $0.0002 per image.
- At a realistic 4 s with overheads: about $0.0003 per image.
- On a 3090, perhaps 1.5–2x slower: still under $0.0006.

**FLUX 3 Klein on a 4090**, interpolated (unmeasured) between the 4070 and 5090 figures: about 5–6 s per 1MP, so roughly $0.0004–0.0005 per image.

**Cost per accepted output** = cost per image ÷ acceptance rate. A model 2x slower still wins if it moves acceptance from, say, 40% to over 80%. That ratio has to come from our own review gate, not from Elo.

**Hosted APIs** make sense only for low-volume, hard shots: complex edits, or rerouted rejects.

### Gaps
- No primary-source current price pages for fal, Together, Replicate, Runware, Google (Vertex/Gemini API) or BytePlus Seedream.
- No measured seconds per image for Klein or HiDream-O1 on a 3090 or 4090.

## 4. Step-distilled / turbo variants and their tradeoffs

### Takeaway
Distillation costs measurable Elo, typically 5 to 130 points depending on how aggressive it is, but cuts compute several-fold. For photos that pass a review gate, the mid-distilled models (8–16 steps) are the sweet spot.

### Cited Findings
- **FLUX.2 dev** (1000) → fal Turbo 996 → fal Flash 985, at $12 → $8 → $5 per 1k. Small loss. [AA open](https://artificialanalysis.ai/image/leaderboard/text-to-image/open-weights)
- **NVIDIA Cosmos3-Super** 988 → 4-step 973. [AA open](https://artificialanalysis.ai/image/leaderboard/text-to-image/open-weights)
- **HiDream-O1-Image** 979 → Dev 808. Large loss. [AA open](https://artificialanalysis.ai/image/leaderboard/text-to-image/open-weights)
- **ERNIE Image Turbo** (916) slightly above ERNIE Image (908). [AA open](https://artificialanalysis.ai/image/leaderboard/text-to-image/open-weights)
- **FLUX 3 Klein:** base model 50 steps, distilled model 16 steps. [BFL](https://bfl.ai/blog/flux-3-image-klein)
- **H3-Image:** ships only as a 6-step, CFG-free distilled checkpoint. [HF](https://huggingface.co/MiniMaxAI/minimax-h3-image)
- **Z-Image Turbo:** 8 steps, low CFG only. The non-distilled Z-Image base model is the recommended start point for LoRA training. [thundercompute](https://www.thundercompute.com/blog/z-image-turbo-comfyui); [GIGAZINE](https://gigazine.net/gsc_news/en/20260128-z-image-image-generation-ai/)
- **Diversity.** The base model is described as producing "a variety of faces and compositions", implying Turbo has less diversity. [GIGAZINE](https://gigazine.net/gsc_news/en/20260128-z-image-image-generation-ai/)

### Inferences
- **Train the LoRA on a base model** (Z-Image base, or Klein base), then run it on the distilled model. LoRA transfer quality to the distilled model should be verified.
- **Watch for diversity collapse.** Distilled models' lower diversity can make a feed look samey; that is a QA consideration for feed QA.

### Gaps
- No independent evaluation of how distillation affects skin, hands or realism of people specifically.

## 5. Leaderboards (late 2026)

### Takeaway
On AA (2026-11-21):
- **Text-to-image:** OpenAI GPT Image 2.6 Sunburst leads at 1263, then GPT Image 2.5 Sunburst 1230, GPT Image 2.6 Flare 1226 and Microsoft MAI-Image-2.7 1215.
- **Editing:** MAI-Image-2.7 leads at 1215.
- **Open weights:** led by H3-Image (1066). Z-Image Turbo is at 936.

### Cited Findings

**Overall text-to-image top 15** ([AA overall](https://artificialanalysis.ai/image/leaderboard/text-to-image))

| # | Model | Elo | $/1k |
|---|---|---|---|
| 1 | GPT Image 2.6 Sunburst (max) | 1263 | $210.72 |
| 2 | GPT Image 2.5 Sunburst | 1230 | — |
| 3 | GPT Image 2.6 Flare | 1226 | — |
| 4 | MAI-Image-2.7 | 1215 | $34.82 |
| 5 | GPT Image 2.5 Flare | 1209 | — |
| 6 | GPT Image 2 (high) | 1194 | — |
| 7 | Kling Image 4.0 | 1181 | — |
| 8 | Nano Banana 2.1 | 1175 | $33.60 |
| 9 | MAI-Image-2.6 | 1168 | — |
| 10 | Grok Imagine Image 2.0 | 1167 | — |
| 11 | MAI-Image-2.7-Flash | 1159 | $10.24 |
| 12 | Canva Craft | 1157 | — |
| 13 | Nano Banana 2 | 1144 | — |
| 14 | Seedream 5.5 | 1135 | $20 |
| 15 | Meta Muse Image | 1126 | $10 |

Further down: FLUX 3 Image (closed, full model) 1114 at $48; Qwen-Image-3.0-Pro 1094.

**Value standouts among closed models:** MAI-Image-2.7-Flash (1159 at $10.24/1k) and Meta Muse Image (1126 at $10/1k).

**LMArena:** not checked this session (gap).

### Inferences
- **If hosted spend is ever justified** (hero shots, repairs), MAI-Image-2.7-Flash or Muse Image offer near-frontier Elo at about $0.01 per image. Their terms for synthetic-person content need checking.

### Gaps
- No LMArena text-to-image or edit standings collected.
- No AA filter for portraits or photorealism found.

## 6. Provenance / C2PA for disclosure

### Takeaway
- **OpenAI and Google** attach C2PA Content Credentials (and Google SynthID) to their outputs.
- **Open-weight models** emit no provenance. None of the open model pages reviewed mention C2PA.
- We would need to sign our own C2PA manifests if we want machine-readable labels in addition to Instagram's `is_ai_generated` flag.

### Cited Findings
- **OpenAI:** a May 19, 2026 post pairs Content Credentials with Google SynthID and an early verification tool. An Oct 5, 2026 update extends this to text in the EU for AI Act compliance. [OpenAI](https://openai.com/index/advancing-content-provenance); [TNW](https://thenextweb.com/news/openai-c2pa-synthid-ai-image-detection-watermark)
- **Google Vertex AI:** supported Google image models, including Gemini 3 Pro Image, get Google-signed Content Credentials automatically. [Google Cloud docs](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/content-credentials)
- **Regulation:** EU AI Act Article 50 and California SB 942 reportedly became operative on 2026-08-02, pushing machine-readable labels. This is secondary and partly conflicting; the primary texts were not checked. [tech-insider](https://tech-insider.org/c2pa-labels-ai-images-2026/)
- **FLUX 3 Klein:** the release notes do not mention C2PA. [BFL](https://bfl.ai/blog/flux-3-image-klein)

### Inferences
- **Self-hosted pipeline:** add a C2PA signing step (for example the open-source c2pa tooling) at the "ready" stage. This is cheap and strengthens the disclosure posture under Article 50.
- **Platform stripping:** whether Instagram preserves or reads C2PA on upload is unknown, so the platform AI flag remains the primary label.

### Gaps
- Instagram/Meta handling of C2PA on upload.
- Seedream / BytePlus provenance.
- Primary text of EU AI Act Art. 50 obligations for deployers of synthetic-person content.

## Bottom line for the report writer

**Is Z-Image Turbo still right?**
- **On cost per image, yes.** It is the cheapest credible option, about $0.0002–0.0004 per image self-hosted, with an Apache 2.0 licence.
- **On quality, no longer.** It is about 100 AA Elo behind FLUX 3 Image Klein and about 300 behind the closed frontier.

**Candidate to benchmark: FLUX 3 Image Klein** (Oct 2026, 5.6B, fits 24 GB, LoRA and 10 references, about 5–6 s per image on a 4090, interpolated).
- Licence is free commercial use only while revenue is under $5M. The owner must accept that licence.
- Run it through the same review gate and compare cost per accepted output; Elo is not the test.

**Fallbacks**
- HiDream-O1-Image: MIT licence, +43 Elo, speed unknown.
- Hosted MAI-Image-2.7-Flash or Muse Image (about $0.01 per image) for repairs.

**Avoid on licence:** Qwen-Image-2.1, FLUX.2 dev, FLUX.2 klein 9B, and H3-Image (licence unclear, too large for consumer GPUs).
