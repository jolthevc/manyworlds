# Talking avatars against the HeyGen bar, and a music critic that can hear (2026-11-23)

Web research only (nothing run). Quality claims are vendors' or blogs' unless marked; licences read from model cards
or LICENSE files where linked. Owner's bar for talking video: HeyGen-level mannerisms and naturalism.

## Talking avatars: ranked for a 90-second continuous take (self-hosted, commercial)

1. **LongCat-Video-Avatar 1.6** (Meituan, 13-15 Nov 2026). MIT code and weights (no trademark or patent grant).
   Autoregressive continuation anchored to a reference frame (`--ref_img_index`); card claims continuous videos over
   10 minutes with less error build-up; INT8/FP8 options. Vendor human study (33k votes) ranks it 2nd of 6, behind
   OmniHuman-2.0 (closed) and ahead of HeyGen. No Sync-C/D on the card. A third-party test of 1.5 peaked at 46.8 GB
   on an H200 for 3.6 s: tight on a 48 GB L40S (INT8 there, or H100). ComfyUI (WanVideoWrapper) supports 1.0/1.5;
   1.6 unverified. Known flaw: occasionally invents hand gestures and over-opens the mouth; the most expressive.
   https://github.com/meituan-longcat/LongCat-Video, https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6
2. **InfiniteTalk** (ours). Apache 2.0. Streaming capped at 1000 frames (40 s) by default; its README says
   image-to-video colour shifts grow beyond 1 minute; suggests a slowly moving source video (V2V) against it.
   LongCat's team attributes long-take drift to repeated VAE encode/decode: the same cause as our segment joins.
3. **Kandinsky S2V 1.0 Pro-distill** (Oct 2026). MIT code and all weights. Whole-figure, 25 fps, 5 s chunks with a
   three-scale visual memory. Vendor H100: 41.5 s per second of video at 46 GiB (~62 min for 90 s); Lite-distill
   8.7 s/s. ComfyUI setup in `dev/comfyui`. https://github.com/kandinskylab/kandinsky-s2v
4. **LiveAvatar** (Alibaba Quark). Apache 2.0, "research preview", a LoRA on Wan2.2-S2V-14B; claims >10,000 s
   continuous; 80 GB GPUs.
5. **EchoMimicV3-Flash** (Ant). Apache 2.0, 12 GB, ComfyUI; subtle motion, 768 max: a cheap fallback only.

Not recommended: HunyuanVideo-Avatar (licence excludes EU, UK, South Korea), Wan2.2-S2V (16 fps, conservative,
slow), MultiTalk/OmniAvatar (superseded), StableAvatar (older), Hallo3/4 (gated/unclear), FantasyTalking2 (no
confirmed weights), AptAvatar (licence and weights unverified).

**Realistic gap**: short clips may be near parity in lip sync and expressiveness if LongCat's study holds; the gaps
are native resolution (480-720p), hands, controllable eye contact, identity and colour over 60 s+, and video-to-video
consistency. Estimate (not measured): HeyGen-like for ~30 s, visible drift after until tested.

**Techniques regardless of model**: one continuous pass per take (cuts, if any, on b-roll or punch-ins); a moving V2V
source (a Wan I2V idle clip) for camera micro-motion and sway; per-segment action prompts from the script at stressed
words; chest-up framing at 480x832 then our upscale; re-anchor each chunk to the reference, colour-match chunks in
LAB, and check identity at 0/30/60/90 s, regenerating only a drifted chunk. No open model checked has blink or gaze
control; LivePortrait-style retargeting needs an InsightFace licence check.

**First test** (owner approved after this research, cap $3): one 90 s line in Maren's designed voice and one still:
LongCat 1.6 single take (INT8, L40S) against our InfiniteTalk segment chain; score Sync per 10 s window, identity at
four timestamps, colour drift (ΔE to the first frame), blink rate, wall time and $ per finished second; the owner's
blind look decides. Kandinsky Pro-distill (H100, ~1 h for 90 s) later if needed.

## A music critic that can hear

Off-the-shelf LLM judges do not reliably reproduce expert song ratings (MuseCritic, arXiv 2608.11755), and no model
predicts engagement (audio features alone explained R² 0.13 of popularity on 113k tracks, arXiv 2512.06259). So the
critic describes and measures, and its scores are calibrated against our own outcomes; it never trusts raw 1-10s.

**Commercial-safe parts**
- Listener: **Qwen3-Omni-30B-A3B-Instruct/Thinking** (Apache 2.0 on the card; HF tag says "other": read LICENSE
  before shipping; self-reported GTZAN 93.0; 79-145 GB BF16 with the talker, so hosted or an 80 GB card);
  **Qwen3-Omni Captioner** (Apache 2.0, audio-only captions); **Step-Audio-2-mini** (Apache 2.0, ~8B, MMAU music
  71.6, fits 24 GB); **Qwen2.5-Omni-7B** (Apache 2.0). Hosted Qwen3-Omni on Novita/DeepInfra/OpenRouter: unverified.
- Quality score: **Meta Audiobox Aesthetics** (CC-BY-4.0: production quality, production complexity, content
  enjoyment, content usefulness; small, CPU-able).
- Measurement: librosa (ISC), pyloudnorm, Demucs (licences from memory: verify).

**Excluded (non-commercial or unclear)**: Music Flamingo / Audio Flamingo 3 (NVIDIA non-commercial), SongEval and
CMI-RM predictors (CC BY-NC), MERT and MuQ weights (CC BY-NC), Essentia (AGPL / NC models), madmom models (NC),
all-in-one and SongFormer (depend on NC parts), MuseCritic and LAION-CLAP music checkpoints and MusicFM (no licence
stated), Kimi-Audio (no weights licence).

**Recommended stack**: measure (CPU) → Audiobox Aesthetics on mix and vocal stem → listener asked factual, checkable
questions (genre, mood, instruments, vocal timbre, hook entry, audible lyrics, mix problems) → Claude reasons over
description and metrics against the artist's targets, writes feedback for the role that can fix it, and logs a
prediction scored later against real engagement.

**Per-song metrics**: tempo and stability; key and mode confidence; LUFS, true peak, loudness range; energy curve per
section and the build into the chorus; time to first vocal and to first chorus (short-form: under ~15 s); repetition
(self-similarity, chorus repeats); section boundaries (librosa recurrence: the most repeated loud section as chorus);
vocal clarity (vocal stem vs instrumental, plus the lyric line coverage); spectral balance, crest factor, clipping;
stereo width and mono compatibility; onset density.
