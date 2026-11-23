# Lip-synced talking video models (audio + one image → talking person), November 2026

Research date: 2026-11-21. Budget-limited pass (~20 tool calls). Many hosted prices come from third-party aggregators. Each one is flagged as such. Check against the provider's own page before budgeting.

## Open-weight models: quality, speed, VRAM, length, license

### Takeaway
The open-weight field has moved past plain InfiniteTalk. Two kinds of model now beat or match it on lip sync at a small fraction of the compute:
- **Few-step distilled successors**: SoulX-FlashTalk (Apache-2.0) and LongCat-Video-Avatar 1.5/1.6 (MIT, 8-step distill, released 2026-11-15, claims more than 10 minutes of video).
- **Real-time streaming models**: LiveAvatar, SoulX-FlashHead.

InfiniteTalk at default settings (40 steps × 3 CFG passes = 120 network evaluations, "NFE") is the slowest option in every published comparison. The 4-step lightx2v LoRA is the documented fix, and it should be benchmarked before anything else. Wan2.2-S2V is clearly worse on sync in third-party tables.

### Cited Findings

**InfiniteTalk (MeiGen-AI)**
- License: Apache-2.0 ("The models in this repository are licensed under the Apache 2.0 License"). The README gives no separate terms for the HF weights. — [InfiniteTalk GitHub](https://github.com/MeiGen-AI/InfiniteTalk)
- Settings without a LoRA: text CFG 5, audio CFG 4, about 40 steps; audio CFG works best at 3–5. With a LoRA: text CFG 1, audio CFG 2. — [InfiniteTalk GitHub](https://github.com/MeiGen-AI/InfiniteTalk)
- lightx2v LoRA is listed as needing **4 steps**; FusionX LoRA needs 8 steps. The README warns FusionX "worsens color shift over about a minute and weakens identity preservation." — [InfiniteTalk GitHub](https://github.com/MeiGen-AI/InfiniteTalk)
- Quantization: only fp8 is documented, single-GPU only. int8 is still an unchecked to-do. Low-VRAM flag: `--num_persistent_param_in_dit 0`. TeaCache is available (`--use_teacache`). — [InfiniteTalk GitHub](https://github.com/MeiGen-AI/InfiniteTalk)
- Length:
  - `--mode streaming` handles long video; the default maximum is 40 s (1,000 frames, `--max_frame_num`).
  - Image-to-video is "good for up to about one minute"; color shift grows beyond that. Video-to-video is described as unlimited.
  - Resolutions: 480P and 720P.
  - InfiniteTalk itself has no 2026 updates; the repo's only 2026 news is LongCat-Video-Avatar 1.5.
  — [InfiniteTalk GitHub](https://github.com/MeiGen-AI/InfiniteTalk)
- Speed (community, **unverified**):
  - An RTX 4090 user reports "40 minutes for a 10 second video at 480p" with no quantization or LoRA, at 8 sample steps. That is about 240 s of compute per second of output. — [GitHub issue #210](https://github.com/MeiGen-AI/InfiniteTalk/issues/210)
  - A ComfyUI tutorial with the lightx2v LoRA (4 steps) reports "around 70 seconds on an RTX 4090." Clip length is not stated; the first window is 81 frames (about 3.24 s at 25 fps) and each extension adds about 2.8 s. — [Next Diffusion tutorial](https://www.nextdiffusion.ai/tutorials/create-lip-sync-videos-from-images-with-infinitetalk-in-comfyui) (via search snippet)
- LightX2V (ModelTC) lists InfiniteTalk as a supported model and claims it can run on 24 GB cards such as the 4090. Its benchmark tables are for the base Wan2.1-I2V at 40 steps (4090D single GPU: 20.26 s per step at 480P), not InfiniteTalk. — [LightX2V GitHub](https://github.com/modeltc/lightx2v)

**Comparison table from the AptAvatar paper (arXiv 2607.24013v2, 2026). Third party, but peer-style.** NFE is the number of network evaluations; ASE is an aesthetics score.

| Method | NFE | Short FID ↓ | Short Sync-C ↑ | Short Sync-D ↓ | Long (minute-level) Sync-C ↑ | Long Sync-D ↓ | Long ASE ↑ |
|---|---|---|---|---|---|---|---|
| InfiniteTalk | 120 | 24.81 | 7.17 | 8.01 | 8.18 | 7.87 | 3.149 |
| Wan2.2-S2V | 80 | 43.54 | 5.63 | 9.38 | 6.17 | 9.54 | 3.186 |
| LongCat-Video-Avatar 1.5 | 8 | 36.58 | 4.14 | 11.43 | 7.49 | 8.85 | 3.274 |
| LiveAvatar | 4 | 37.87 | 5.99 | 8.59 | 6.65 | 9.01 | 3.272 |
| SoulX-FlashTalk | 4 | 28.04 | 7.74 | 7.91 | 7.98 | 8.32 | 3.246 |
| AptAvatar | 2 | 24.76 | 7.87 | 7.74 | 8.40 | 7.78 | 3.382 |

  - Source: [AptAvatar HTML](https://arxiv.org/html/2607.24013v2)
  - User study (100 videos, 10 raters, InfiniteTalk not included): AptAvatar won 62%, SoulX-FlashTalk 20%, LongCat 1.5 18%, LiveAvatar 0%, Wan2.2-S2V 0%.
  - AptAvatar claims a 60× speedup over InfiniteTalk. It reports no wall-clock time or inference hardware.
  - Its code and weights release status is **not stated** in the paper.
  - Sync-C/Sync-D are cited to Chung & Zisserman 2016, i.e. the original SyncNet.

**LongCat-Video-Avatar (Meituan)**
- Repo and weights are MIT ("released under the MIT License").
  - 1.5 released 2026-05-21: Whisper-large-v3 audio encoder, 8-step distill (`--use_distill`).
  - 1.6 released 2026-11-15: DMD + OPSD distill LoRAs (`--mix_steps`), better mouth naturalness, less error build-up over long videos, "exceeding 10 minutes" via continuation segments.
  - 480P/720P; single-person and multi-person.
  - Base LongCat-Video is 13.6B dense.
  - VRAM and speed are **not stated**. FA3 is recommended on Hopper GPUs (H100/H200); FP8/INT8 options exist.
  — [LongCat-Video GitHub](https://github.com/meituan-longcat/LongCat-Video)
- fal hosts "LongCat Avatar" at $9.00/min (480p) and $18.00/min (720p), with about 51 s max. — [Stork pricing survey, Nov 2026](https://www.stork.ai/blog/talking-head-api-pricing-2026) (third party)

**SoulX-FlashTalk / SoulX-FlashHead (Soul AI Lab)**
- FlashTalk-14B: the HF model card metadata says Apache-2.0. Code and weights released 2026-01-08. Claims 0.87 s start-up latency and 32 FPS real-time streaming (hardware not captured; likely multi-GPU H-class, **unverified**). — [HF README](https://huggingface.co/Soul-AILab/SoulX-FlashTalk-14B/raw/main/README.md), [arXiv 2512.23379](https://arxiv.org/html/2512.23379v3)
- SoulX-FlashHead: "infinite real-time streaming talking heads". — [arXiv 2602.07449](https://arxiv.org/html/2602.07449)
- WaveSpeed hosts SoulX-FlashHead at $0.90/min (480p) and $1.80/min (720p), up to 30 min. — [Stork survey](https://www.stork.ai/blog/talking-head-api-pricing-2026) (third party)

**LiveAvatar (Alibaba Quark)**
- Real-time, streaming, infinite-length avatar framework on Wan. A third-party index says Apache-2.0 for most of the project and calls it a research preview. **Unverified**: check the LICENSE file. — [LiveAvatar GitHub](https://github.com/Alibaba-Quark/LiveAvatar), [SourceForge mirror](https://www.sourceforge.net/projects/liveavatar.mirror/)

**Wan2.2-S2V-14B**
- Apache-2.0; released 2025-08-26; 480P/720P; length follows the audio (`--num_clip`); pose-driven mode (`--pose_video`).
- The single-GPU command needs "at least 80GB VRAM" with offload. The model card does not mention lightx2v/distill.
- Source: [HF model card](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)

**LatentSync 1.6 (ByteDance, video-to-video lip dubbing)**
- Released 2025-06-11, trained at 512×512. Minimum inference VRAM is 18 GB (1.6) and 8 GB (1.5). The code links Apache-2.0. — [LatentSync GitHub](https://github.com/bytedance/LatentSync)
- The HF weights carry the **openrail++** tag. — [HF LatentSync-1.6](https://huggingface.co/ByteDance/LatentSync-1.6)

**MuseTalk 1.5 (Tencent Music Lyra Lab, video-to-video lip dubbing)**
- Code is MIT. The weights are "available for any purpose, even commercially."
- Third-party pieces (whisper-tiny, sd-vae-ft-mse, DWPose, face-parse-bisenet, S3FD) follow their own licenses.
- Face region is 256×256; "30fps+ on a V100"; runs in 4 GB VRAM.
- Released 2025-03-28. It uses LatentSync's SyncNet in training.
- Source: [MuseTalk GitHub](https://github.com/TMElyralab/MuseTalk)

**HunyuanVideo-Avatar, StableAvatar, EchoMimic v3, FantasyTalking**
- HunyuanVideo (base) is under the Tencent Hunyuan Community License. A secondary site says it excludes the EU, UK and South Korea (**unverified**). — [HF HunyuanVideo Notice](https://huggingface.co/tencent/HunyuanVideo/blob/main/Notice), [Apatero](https://apatero.com/blog/oss-avatar-generation-explosion-huanyuan-omniavatar-echomimic-2025)
- StableAvatar is built on Wan2.1-1.3B. — [HF StableAvatar](https://huggingface.co/FrancisRing/StableAvatar)
- I could not confirm the licenses for EchoMimic v3, FantasyTalking or StableAvatar.

**Other 2026 research** (open-source status not verified): TurboTalk (one-step distillation, [arXiv 2604.14580](https://arxiv.org/pdf/2604.14580)), Omni-LiveAvatar (minute-level real-time joint audio-video, [arXiv 2608.13602](https://arxiv.org/pdf/2608.13602)), ActAvatar (action control, [arXiv 2512.19546](https://arxiv.org/pdf/2512.19546)).

### Inferences
- **InfiniteTalk's timeout follows from the arithmetic.** 40 steps × 3 CFG passes is 120 NFE per window. The 4090 report of about 240 s of compute per output second (8 steps, no LoRA) puts default settings far above 50 min for a 6 s clip on an A40. With the lightx2v LoRA (4 steps, and CFG 1 should drop most of the extra guidance passes), NFE falls from 120 to roughly 4–8, a theoretical 15–30× cut. That would mean a few minutes per 6 s clip on a 4090/L40S. Not measured; benchmark it.
- **Priority for open-weight benchmarking**, by license × sync × speed:
  1. InfiniteTalk + lightx2v 4-step (cheapest change to the existing image; best long-video Sync-C among the old models).
  2. SoulX-FlashTalk (Apache-2.0, 4 NFE, best short-clip sync of the released models, real-time).
  3. LongCat-Video-Avatar 1.6 (MIT, built for 10+ minute takes, which suits the segment-joining design).
  4. Wan2.2-S2V: deprioritize (worst sync in the table, 80 GB VRAM).
- **Per-output-second cost formula** for self-hosting: cost = (GPU $/hr ÷ 3600) × (compute seconds per output second) ÷ accept rate. At 240 s/s on a roughly $0.70/hr card that is about $0.047 per raw second, before rejections. At a distilled 10–20 s/s it is about $0.002–0.004 per second. Hourly rates here are illustrative, not sourced; use the ledger's measured pool prices.
- The 94 GB weight download and >48 GB RAM need (multiple Wan2.1-14B variants plus fp32) is itself a cost per boot. A single fp8 14B checkpoint with LoRAs should shrink it.

### Gaps
- No verified wall-clock benchmark of InfiniteTalk + lightx2v, SoulX-FlashTalk or LongCat 1.6 on a single 4090, L40S or H100 at stated resolution and clip length.
- VRAM and RAM for LongCat-Video-Avatar 1.6 and SoulX-FlashTalk on a single GPU.
- Licenses for EchoMimic v3, FantasyTalking, StableAvatar, HunyuanVideo-Avatar (exact text), Hallo3 and Sonic: not checked in this pass. From memory, Hallo3 builds on CogVideoX and Sonic's weights were non-commercial; both **unverified**.
- Whether AptAvatar weights will be released.
- No quantitative data on body motion or expressiveness beyond ASE and the GSB user study.

## Two-stage alternative: silent image-to-video, then lip dubbing (LatentSync / MuseTalk)

### Takeaway
Two-stage is cheap and fully commercial-licensable with MuseTalk (MIT, weights for any purpose). LatentSync 1.6 is cheap too, with OpenRAIL++ weights. The risk is that both only repaint a cropped mouth/face region (256 px for MuseTalk, 512 px for LatentSync). Head and body motion are not driven by speech, so expressiveness depends on the silent clip. InfiniteTalk's own video-to-video mode, documented as "unlimited" length, is an end-to-end-quality dubbing alternative.

### Cited Findings
- MuseTalk: 256×256 face region, 30+ fps on a V100, 4 GB VRAM, MIT code, commercially usable weights. — [MuseTalk GitHub](https://github.com/TMElyralab/MuseTalk)
- LatentSync 1.6: 512×512 face crops, 18 GB VRAM; 20–50 steps trade quality for speed. — [LatentSync GitHub](https://github.com/bytedance/LatentSync)
- InfiniteTalk video-to-video is described as unlimited length; image-to-video is limited to about 1 min by color shift. Suggested workaround: turn the image into a pan/zoom video first, then run video-to-video. — [InfiniteTalk GitHub](https://github.com/MeiGen-AI/InfiniteTalk)
- Hosted dubbing: sync.so lipsync-2-pro costs $0.067–$0.083/s at 25 fps, billed per output frame ($0.00333/frame on Hobbyist), with up to 20% off on higher tiers. — [sync.so billing docs](https://sync.so/docs/billing), [sync.so pricing](https://sync.so/pricing). Resellers: WaveSpeed $0.08/s, Runware $0.0733/s, fal lipsync-2 $3/min and Pro $5/min. — [WaveSpeed](https://wavespeed.ai/docs-api/sync/sync-lipsync-2-pro), [Runware](https://runware.ai/models/lipsync-2-pro), [fal lipsync v2](https://fal.ai/models/fal-ai/sync-lipsync/v2)

### Inferences
- The project already has Wan 2.2 TI2V-5B silent clips (about 14 min per 5 s 720p on a 3090). A MuseTalk/LatentSync pass on top adds only seconds of GPU time. So two-stage costs about the same as the current story clip plus a little.
- The catch is gesture and speech mismatch. In "press conference" and "monologue" formats the body won't emphasize words.
- A good hybrid is a silent i2v clip as the "motion source", then InfiniteTalk video-to-video (lightx2v) for speech-driven face and head. Test it against end-to-end image-to-video.

### Gaps
- No published head-to-head (sync, identity, human preference) of two-stage versus end-to-end audio-driven models was found.

## Hosted APIs: prices and terms

### Takeaway
Hosted options cost roughly $0.015–$0.16 per output second (Nov 2026):
- **Cheapest:** WaveSpeed InfiniteTalk-Fast at $0.015/s, Replicate wan-2.2-s2v at $0.02/s, WaveSpeed SoulX-FlashHead at $0.015–$0.03/s.
- **Mid:** Hedra Character-3 at $0.025–$0.0625/s, Kling AI Avatar at about $0.056/s, BytePlus OmniHuman 2.0 at $0.07/s.
- **Premium:** OmniHuman 1.5 at $0.12–$0.16/s, Veo 3.1 with audio at $0.40/s.

Sora 2's API was shut down on 2026-09-24. fal's InfiniteTalk endpoint is deprecated. Hosted per-second prices are already close to what a well-tuned self-hosted distilled model would cost. So hosted APIs make sense as a fallback or a quality ceiling for the gate's calibration.

### Cited Findings
**Main price table.** Every price in it was read on providers' pages on 2026-11-13 and comes from the [Stork survey (published 2026-11-14, updated 2026-11-19)](https://www.stork.ai/blog/talking-head-api-pricing-2026). Stork sells a competing API.

| Provider / model | Price | Notes |
|---|---|---|
| WaveSpeed InfiniteTalk-Fast | $0.015/s ($0.90/min) | Up to 10 min |
| WaveSpeed SoulX-FlashHead | $0.90 / $1.80 per min | 480p / 720p; up to 30 min |
| Replicate wan-2.2-s2v | $0.02/s | |
| Replicate p-video-avatar | $0.025/s | 720p |
| Hedra Character-3 | $1.50 / $3.00 / $3.75 per min | 540p / 720p / 1080p; up to 10 min |
| HeyGen Avatar IV photo avatar | $2.31/min | |
| HeyGen Avatar III | $0.99/min | Plus $1.32 per photo avatar |
| Kling AI Avatar standard | $0.056/s | 720p; up to 300 s |
| Kling AI Avatar v2 Pro (fal) | $0.115/s | 1080p |
| BytePlus OmniHuman 2.0 | $0.07/s | 720p; 30–300 s |
| BytePlus OmniHuman 1.5 | $0.12/s | |
| fal OmniHuman 1.5 | $0.16/s | 60 s max at 720p, 30 s at 1080p |
| fal OmniHuman v2 | $0.105/s | |
| fal HeyGen Avatar 4 | $0.10/s | |
| fal VEED Fabric 1.0 | $4.80 / $9.00 per min | 480p / 720p |
| fal Creatify Aurora | $4.20 / $8.40 per min | 480p / 720p |
| Alibaba wan2.2-s2v | $4.30 / $7.74 per min | Beijing only; audio under 20 s |
| RunPod InfiniteTalk | $0.25–$0.50 per video flat | |

- Alibaba retired its LivePortrait and EMO APIs on 2026-10-10. — [Stork survey](https://www.stork.ai/blog/talking-head-api-pricing-2026)
- Stork's one-run SyncNet scores on a 20 s Kokoro clip: OmniHuman 1.5 6.87, Kling Avatar v2 Std 6.35, VEED Fabric 8.11, Stork 6.55–6.85. One run each, by a vendor, so treat as low confidence. — [Stork survey](https://www.stork.ai/blog/talking-head-api-pricing-2026)
- fal's own page confirms OmniHuman 1.5 at $0.16/s and the 720p/60 s and 1080p/30 s limits. — [fal OmniHuman v1.5](https://fal.ai/models/fal-ai/bytedance/omnihuman/v1.5), [fal guide](https://fal.ai/learn/devs/omnihuman-15-user-guide)
- fal InfiniteTalk (`fal-ai/infinitalk`) costs $0.20/s, doubled at 720p, and is marked deprecated ("This model is no longer supported"). It is tagged "Commercial use". — [fal Infinitalk](https://fal.ai/models/fal-ai/infinitalk)
- Hedra Character-3 rates from third parties: 2.5¢/s at 540p, 5¢/s at 720p, 6.25¢/s at 1080p. Pay-as-you-go wallet. Hedra also resells HeyGen Photo Avatar 4 at 10¢/s. — [adtestbench Hedra](https://adtestbench.com/tools/hedra/pricing/), [genburn](https://genburn.com/model/character-3)
- HeyGen API prices conflict: $0.05/s, $2.31/min, and about $4/min at 1080p are all cited, and no primary page was found. — [costperclip](https://costperclip.com/cheapest-ai-avatar-video-generator), [adtestbench compare](https://adtestbench.com/compare/hedra-vs-heygen/)
- Veo 3.1 (Gemini API, per [BenchLM 2026-09-11](https://benchlm.ai/media-pricing/veo)):
  - Standard: $0.40/s at 720p/1080p, $0.60/s at 4K.
  - Fast: $0.10/s at 720p, $0.12/s at 1080p.
  - Lite: $0.05/s at 720p, $0.08/s at 1080p.
  - OpenRouter: with audio from $0.40/s; without audio $0.20/s. — [OpenRouter](https://openrouter.ai/google/veo-3.1-20260320)
  - Third-party data; the Google price page was not confirmed.
- Sora 2: "models and Videos API were shut down on September 24, 2026". — [OpenAI Sora 2 docs](https://developers.openai.com/api/docs/models/sora-2)

### Inferences
- For 30 s clips, the hosted cost per raw clip is about:
  - $0.45 on WaveSpeed InfiniteTalk-Fast
  - $1.50 on Hedra 720p
  - $1.70 on Kling Avatar Std
  - $3.60–4.80 on OmniHuman 1.5
  - $12 on Veo 3.1 standard (8 s per clip, so it needs stitching; it doesn't use your own audio or identity image the same way)

  Divide by the gate's accept rate.
- Veo 3.1 generates its own speech and doesn't take an external voice track the way avatar APIs do. That makes it poorly suited to a fixed creator voice. This is an inference from the API design; confirm against the Veo docs.

### Gaps
- Commercial terms and AI-person policies (ToS clauses on synthetic people, disclosure, likeness, and whether outputs may be used on Instagram commercially) were **not verified** for any provider in this pass: BytePlus, Hedra, HeyGen, Kling, sync.so, Captions/Mirage, Tavus. This is a must-do before adoption.
- No sourced API prices for Captions/Mirage, Tavus (no single-photo + own-audio price published, per Stork), or Seedance with audio.
- No primary BytePlus or HeyGen API price page was retrieved.

## Automated lip-sync scoring with a commercial license

### Takeaway
No clean, permissively licensed drop-in for the classic SyncNet LSE-C/LSE-D was found:
- Original VGG SyncNet code is MIT, but the **weights are "for research purposes under Creative Commons Attribution"** and aren't covered by the code license.
- LatentSync's `stable_syncnet.pt` (94% accuracy on VoxCeleb2/HDTF) sits in an HF repo tagged **openrail++**. OpenRAIL++ generally allows commercial use subject to use-based restrictions, so it is the most practical licensable scorer. Read the actual license text.
- LSE-C/D correlates poorly with human preference, so pair it with a VLM judge on keyframes or short clips, plus mouth-open/audio-energy heuristics.

### Cited Findings
- syncnet-python (pip): code is MIT. The Oxford VGG page says "The model can be used for research purposes under Creative Commons Attribution License"; the MIT/Apache licenses "do not cover the weights". — [syncnet-python PyPI](https://pypi.org/project/syncnet-python/)
- LatentSync ships `eval/eval_sync_conf.sh` (sync confidence for a generated video) and `eval_syncnet_acc.sh`. Its SyncNet reaches 94% accuracy on VoxCeleb2 and HDTF. — [LatentSync GitHub](https://github.com/bytedance/LatentSync)
- `stable_syncnet.pt` is downloaded from `ByteDance/LatentSync-1.6`, whose model card is tagged openrail++. — [HF LatentSync-1.6](https://huggingface.co/ByteDance/LatentSync-1.6)
- A Gitee mirror lists the code as Apache-2.0. — [Gitee mirror](https://gitee.com/wslrj/LatentSync)
- Papers report that "the commonly used Syncnet confidence score (LSE-C) and distance (LSE-D) have been found to correlate poorly with human preferences" and that SyncNet is not shift-invariant. They propose AV-HuBERT-based metrics instead. — [arXiv 2405.04327](https://arxiv.org/html/2405.04327)
- THEval (CVPR 2026 Findings) proposes an evaluation framework for talking-head generation. The PDF returned 403, so its content was not reviewed. — [THEval PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Quignon_THEval._Evaluation_Framework_for_Talking_Head_Video_Generation_CVPRF_2026_paper.pdf)
- The AptAvatar paper and Stork's vendor test both still report SyncNet Sync-C/D, so it remains the de facto industry number. — [AptAvatar](https://arxiv.org/html/2607.24013v2), [Stork](https://www.stork.ai/blog/talking-head-api-pricing-2026)

### Inferences
- Recommended gate stack:
  1. LatentSync `stable_syncnet.pt` confidence and offset as a hard floor. Calibrate the threshold on accepted versus rejected clips; reference values: the hosted APIs score about 6.3–8.1 Sync-C, InfiniteTalk 7.2–8.2 in the AptAvatar tables.
  2. Audio-to-video offset check: reject if |offset| exceeds a few frames.
  3. The existing VLM keyframe judge for identity, artifacts and hands.
  4. A cheap heuristic correlating mouth-landmark openness (MediaPipe, Apache-2.0) with the audio RMS envelope. This is a license-clean fallback signal that needs no SyncNet weights.
- An option to remove all license doubt is to train a small SyncNet-style audio-visual sync classifier on the project's own accepted clips. LatentSync's training code is Apache-2.0. This only makes sense if the OpenRAIL++ terms prove unacceptable.

### Gaps
- The exact OpenRAIL++ text for LatentSync weights was not read. Confirm commercial use and use restrictions (e.g. no deceptive impersonation, which should be compatible with disclosed, original AI people).
- The AV-HuBERT license was not verified. From memory, Meta's AV-HuBERT carries a non-commercial license; **unverified**.
- The Wav2Lip expert discriminator license was not verified. From memory, it is non-commercial with commercial use via Sync Labs; **unverified**.
- THEval's metric definitions and licenses were not retrieved (403).
- No evidence was found on how well VLM-based lip-sync judging (e.g. Gemini/Claude on frame strips) correlates with human ratings.
