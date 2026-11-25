# Video model candidates vs our baselines (2026-11-24)

Web research only: nothing run, no weights downloaded. Baselines: Wan 2.2 I2V A14B + lightx2v, InfiniteTalk (I2V/V2V),
LongCat-Video-Avatar 1.6, Wan-Animate-2, on ComfyUI (Kijai WanVideoWrapper).
**V** = read on a primary source (repo README, HF card, licence file). **C** = the authors' own claim, or a third-party
page, not checked by anyone independent. **I** = our inference (for example size worked out from parameter count).

## Summary table

| Candidate | Exists | Commercial licence | Our voice in? | Fits 24/32 GB | Verdict |
|---|---|---|---|---|---|
| ckinpdx/longcat-avatar-fast | V (3 stars, 10 commits, 2026-11-16/17) | MIT code; LongCat MIT, LightVAE Apache-2.0 | yes (audio-driven) | C: 24.2 GB peak on a 5090 | **Benchmark first** |
| LTX-2.5 (Lightricks) | V | Allowed under US$10M annual revenue; AUP | yes (A2V pipeline) | C: int8 only at 24 GB; 32-48 GB is comfortable | **Benchmark second** (L40S/5090) |
| EchoMimicV3-Flash | V (HF 2026-01-22) | Apache-2.0; wav2vec MIT | yes | C: 12 GB | Cheap benchmark, low expectations |
| Kandinsky 6.0 Lite | V (2026-10-06) | MIT (repo + HF tag); audio VAE origin unclear | only "audio-file-driven video" (since 10-25) | V: timings on 4090/5090 | Maybe later (480x864 is landscape; 5 s) |
| Ovi 1.1 (Character.AI) | V | Apache-2.0 code/model, but **MMAudio VAE weights CC-BY-NC 4.0** | **no** (voice conditioning on the to-do list) | V: about 24 GB with fp8 + offload | **Skip** |

## 1. EchoMimicV3-Flash vs EchoMimicV3

- Repo `antgroup/echomimic_v3`; weights `BadToBest/EchoMimicV3` (Flash folder named `echomimicv3-flash-pro`, so the
  naming is inconsistent); base `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP`. V: https://github.com/antgroup/echomimic_v3
- Dates: code 2025-08-08; Flash on HF 2026-01-22; AAAI 2026. V (same page)
- Flash vs preview: Flash runs 8 steps (preview: 5 for heads, 15-25 for bodies), needs no face mask, takes
  `chinese-wav2vec2-base` instead of `wav2vec2-base-960h`, and goes up to 768x768. V (README). 12 GB VRAM: C.
  Tested GPUs: A100, 4090D, V100. V
- Licence: Apache-2.0 (`LICENSE.txt`). README: the models are Apache 2.0 licensed and the authors claim no rights
  over generated content. V. `TencentGameMate/chinese-wav2vec2-base`: MIT. V (https://huggingface.co/TencentGameMate/chinese-wav2vec2-base).
  Wan2.1-Fun base: Apache-2.0. I (Alibaba PAI's usual licence; not re-checked)
- Size: card says 2B params in BF16, about 4 GB for the transformer. I. You also need the Wan 1.3B base (UMT5-xxl
  text encoder, about 11 GB). I
- Length: no hard cap; Long Video CFG above 138 frames. V. Inputs: reference image, audio, prompt (and a mask on
  preview). V. Resolution: square up to 768; 9:16 is not documented (unverified).
- Quality: the paper claims the best Sync-C/Sync-D (C, https://arxiv.org/abs/2507.03905). It is a 1.3B model, so
  expect weaker identity, hands and body than our 14B baselines (I). No independent Flash comparison found.
- ComfyUI: `smthemex/ComfyUI_EchoMimic` supports V3 and Flash (Flash skips retinaface). C (https://comfy.icu/extension/smthemex__ComfyUI_EchoMimic)
- Requirements: Python 3.10/3.11, CUDA >= 12.1; torch version not stated. V
- Maturity: 1.1k stars, 24 open issues. V

## 2. ckinpdx/longcat-avatar-fast

Source: https://github.com/ckinpdx/longcat-avatar-fast (README and commits; all V as stated by the author, speed C).

- It is a launcher that patches Meituan's LongCat-Video-Avatar **1.6** demo from outside, leaving their repo
  unmodified. It works with the 1.6 weights (`meituan-longcat/LongCat-Video` + `meituan-longcat/LongCat-Video-Avatar-1.6`)
  and distributes no weights.
- Changes:
  - **Quantisation:** the DiT's 624 Linear layers run as W8A8 INT8 through comfy-kitchen, with a group-256 Hadamard
    rotation (ConvRot). Conversion writes a file of about 15 GB. Optional INT8 ConvRot UMT5; the author notes this
    has not been tried in a Wan video yet.
  - **Steps:** 6 by default (4 or 8 also possible), chosen from the 8 trained distillation sigmas (8 is Meituan's schedule).
  - **Attention:** SageAttention 2.2 for self-attention; cross-attention stays on FA2.
  - **Memory:** UMT5 encodes on the CPU (or loads from a ComfyUI file: 3.4 s vs 30.6 s); Whisper is loaded
    encoder-only (2.4 GB instead of 5.7 GB, bit-identical features); the cache is cleared after each step; 3 GB headroom.
  - **Decoder:** LightVAE-pro (`lightx2v/LightVAE`) in place of the Wan VAE, which can change quality.
  - **Other:** fused LoRA, `torch.compile`, batch and watch-folder modes (compile once); experimental `LONGCAT_VCOND`
    latent pass-through between segments; `--mask_frame_range=2` default (stops the pull back to the source pose
    every 3.2 s segment).
- Claimed speed (C, one setup): RTX 5090, 480p, one 3.2 s segment takes 30.0 s at 6 steps vs 115 s for the stock
  demo, so 3.8x (3x at 8 steps). About 9.4 s per output second. Peak 24.2 GB.
- Tested only on Windows 11, Python 3.11, torch 2.11 + CUDA 13.0. **Linux is untested**, and our workers are Linux.
- Licence: MIT (code); LongCat MIT; LightVAE, comfy-kitchen and SageAttention Apache-2.0. V (README). LongCat 1.6
  card: MIT, with no trademark or patent grant. V (https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6)
- Maturity: 3 stars, 1 fork, 10 commits (2026-11-16..17), no releases. One person's project; treat it as a recipe
  to port into our `talk.longcat` venv, not a dependency.

## 3. LTX-2.5 (Lightricks)

- Ids: `Lightricks/LTX-2.5` (gated: you must accept terms), `Lightricks/LTX-2.5-Diffusers`; code
  `Lightricks/LTX-2` (10.4k stars, 17 open issues). V (https://huggingface.co/Lightricks/LTX-2.5, https://github.com/Lightricks/LTX-2)
- Released 2026-08-11. C (RunPod blog, https://www.runpod.io/blog/ltx-2-5-the-open-weights-world-model-built-for-speed-and-how-to-run-it-on-runpod)
- 22B DiT. Checkpoints: distilled bf16 (8 steps, CFG 1), dev bf16, Comfy int8+ConvRot (distilled and dev), NVFP4
  distilled (Blackwell). Text encoder is Gemma 4 12B (bf16 or int8). Also a DiffVAE or lighter Conv VAE, an audio
  VAE + vocoder, x2 spatial and temporal upscalers, and a duration head. V (HF card)
- **Joint audio+video is native** (one model makes synchronised video and audio). V
- Pipelines (V, repo README): `DistilledPipeline` (8+4 steps), `DFRPipeline`, `TI2VidTwoStages(HQ)`,
  `ICLoraPipeline` (V2V/I2V), **`A2VidPipelineTwoStage` (audio-to-video from an input audio file)**,
  **`RetakePipeline` (regenerates a time region of an existing video)**, `DubItPipeline` (re-speak with speaker
  identity and lip matching), HDR.
- Retake (C, third-party walkthrough https://instagit.com/Lightricks/LTX-2/ltx-2-retake-pipeline-regenerate-time-regions/):
  encodes the source video and audio, masks [start, end] in time and denoises only that window; works with distilled
  or dev. Useful to repair a bad 1-2 s span of a talking clip instead of rerunning it all.
- Frames must satisfy `num_frames % 8 == 1`; width and height must be divisible by 32 (V). 9:16 at 544x960 or
  736x1280 fits that rule (I). Hosted A2V takes 2-20 s of audio (C, fal). No max length stated.
- VRAM (C, RunPod): 16-24 GB is int8 only with offload and low resolution; 32-48 GB (L40S) runs int8 well; bf16
  distilled is viable at 48 GB with offload; 80 GB is recommended. Distilled bf16 pack is about 66 GiB on disk (V,
  repo quick start). No measured 24 GB timings found.
- Requirements: Python >= 3.12, CUDA >= 12.7, torch ~2.7; managed by `uv`. V (HF card). ComfyUI: official templates;
  the int8 checkpoints are Comfy-only. V
- **Licence (LTX-2.x Community License, https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x, V):** free
  commercial and production use below US$10M annual revenue. Entities with annual revenues of at least $10,000,000
  need a paid licence; revenue counts all affiliates under common control. Outputs: "Licensor claims no rights in
  the Output". The AUP forbids content without disclaiming that it is machine generated, and forbids deepfake
  impersonation without consent. That matches our disclosure rule, but it makes disclosure a licence condition, not
  only a policy choice. Gemma 4's licence was not checked (flag). Code is Apache-2.0.
- Identity from one image, lip sync, hands: no independent evidence found for A2V with a reference face. Must be measured.

## 4. Kandinsky 6.0 Lite

- Ids: `kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers` (3B, I2V/T2V + audio, total repo 26.7 GB), VSR
  `kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers` (1B); code `kandinskylab/kandinsky-6` (331 stars,
  7 open issues). V (https://huggingface.co/kandinskylab, https://github.com/kandinskylab/kandinsky-6)
- Released 2026-10-06. Since 2026-10-25: first/last frame, **audio-file-driven video**, video-driven audio. V (repo)
- Output: 5 s, 121 frames at 24 fps, 44 kHz audio; base 480x864 shown as 16:9 landscape (864x480 in the card
  examples); VSR to 1920x1080. 9:16 is not documented. V
- Components (V, card): Qwen2.5-VL + CLIP text encoders, HunyuanVideo VAE class, `MMAudioVAE` + `MMAudioVocoder`
  (BigVGAN-style), PiFlow 10 steps. **Licence: MIT on repo and card (V).** But the audio VAE and vocoder classes are
  named after MMAudio, whose checkpoints are CC-BY-NC 4.0. Kandinsky Lab's KVAE report treats MMAudio as a baseline
  (C, https://huggingface.co/papers/2608.05798), so their weights are probably their own (I). Confirm provenance
  before commercial use.
- Speed (V, repo table, excluding load and encode) for Lite Distill SD 5 s: 4090 68 s, 5090 44 s, H100 31 s (about
  9-14 s per output second). Lite is resident on a 5090 and uses module offload on a 4090. Peak VRAM about 44 GiB
  fully on GPU. C (third-party citing the report)
- Lip sync is claimed, with no evaluation and no documented speech tag. V/C. With our own audio as input
  (audio-driven mode), quality is unknown. ComfyUI: official registry node `kandinsky6`; native core PR #16825. V
- Verdict: interesting as a cheap native-sound B-roll generator. For our talking creators it is landscape-first and
  5 s, so not a first benchmark.

## 5. Ovi (Character.AI)

- Ids: `character-ai/Ovi` (1.8k stars, 44 open issues), weights `chetwinlow1/Ovi` (12B BF16; three files: 720x720_5s,
  960x960_5s, 960x960_10s, each about 24 GB by I), fp8 `rkfg/Ovi-fp8_quantized`. V (https://github.com/character-ai/Ovi)
- Ovi 1.1 (2025-11-10): 10 s at 960x960, 24 fps; 9:16, 16:9 and 1:1 named; T2AV and I2AV; speech in `<S>...<E>`. V
- VRAM (V, README): about 80 GB without offload (83 s for 121 frames at 720x720, 50 steps); about 32 GB with offload
  (140 s); about 24 GB with fp8 + offload (quality loss; fp8 gives no speedup). So about 17-28 s per output second at
  50 steps. Requires torch 2.6.0 and flash_attn. ComfyUI via Kijai WanVideoWrapper (V, README).
- **Licence blocker:** Ovi is Apache-2.0 (V), but `download_weights.py` pulls `hkchengrex/MMAudio`
  `ext_weights/v1-16.pth` and `best_netG.pt` (V, https://raw.githubusercontent.com/character-ai/Ovi/main/download_weights.py).
  MMAudio README: "The checkpoints are released on Hugging Face under the CC-BY-NC 4.0 license", and the authors do
  not guarantee the models are suitable for commercial use. V (https://github.com/hkchengrex/MMAudio). Ovi's audio
  path therefore depends on non-commercial weights.
- Also: no reference-voice conditioning (on the to-do list, V), so the voice is invented each clip and a creator's
  designed voice cannot be kept. No distilled model yet.

## Ranked benchmark plan (24/32 GB where possible; each run needs the owner's yes)

1. **longcat-avatar-fast recipe on our LongCat 1.6** (5090 32 GB, or 4090 if 24.2 GB peak holds). Same weights and
   licence as our baseline. Port INT8 ConvRot + Sage + 6 steps + LightVAE into the `talk.longcat` venv on Linux. Run
   A/B against stock 1.6 on 3-4 lines of Maren: speed, identity, lip-sync (WER/sync), and LightVAE vs Wan VAE
   quality. Lowest risk, likely the biggest saving.
2. **LTX-2.5 A2V (int8, ComfyUI) + Retake** on L40S 48 GB (a 32 GB 5090 is a stretch; 24 GB not advised). Feed our
   VoxCPM voice + one reference frame at 544x960; compare with InfiniteTalk/LongCat on identity and sync. Separately
   test Retake to repair a bad span. Licence is fine below US$10M revenue; disclosure is required by the AUP (we do
   it anyway). Note the gated HF and the 66 GiB pack (volume space).
3. **EchoMimicV3-Flash** (4090 24 GB, ComfyUI smthemex): a cheap check (12 GB, 8 steps, Apache). Likely loses on
   identity and body, but worth one run as a low-cost talking-head tier.
4. **Kandinsky 6.0 Lite**: defer. Landscape 5 s base; audio-driven mode is new; audio VAE provenance unconfirmed.

**Skip Ovi**: MMAudio CC-BY-NC weights in its audio path (licence blocker for commercial use), no voice reference,
and 32 GB+ for usable quality.
