# Natural mannerisms in talking video: gestures, head, blinks, gaze, hands (2026-11-24)

Web research only (nothing run, nothing spent). Vendor claims are marked as such; **[est]** marks our estimates.
Builds on `2026-11-23-avatar-and-music-critic.md` (model ranking for long takes) and `2026-11-22-video-quality-wave2-3.md`.
Trigger: LongCat 1.6, 90 s take: identity held, but the hand stayed frozen, no gestures, lip correlation 0.325.

## 1. Why the commercial avatars look natural (what is public)

- **HeyGen.** There are two products. *Hyper-realistic / Gesture Control* avatars are trained on the customer's own
  footage. Gestures are "real movements captured during the training process", and you place them on script words
  ([help](https://help.heygen.com/en/articles/11691624-how-to-use-gesture-control),
  [guide](https://www.heygen.com/blog/heygen-gesture-control-guide)). *Avatar IV* works from a photo, and
  [custom motion prompts](https://help.heygen.com/en/articles/12805098-how-to-fine-tune-avatar-gestures-and-movements-with-custom-motion-prompts)
  steer expression, gesture, posture and gaze in text, with the timing handled automatically. No technical paper.
- **Synthesia Express-2.** The voice is generated first. A DiT gesture model then produces body motion for that
  speech, and a renderer makes the frames. A CLIP-like "Express-Eval" scores how well the motion fits the audio. It
  renders 1080p30 with no length limit
  ([Synthesia](https://www.synthesia.io/post/express-2-is-synthesias-next-chapter-for-full-body-expressive-ai-avatars),
  [MIT TR](https://www.technologyreview.com/2025/09/04/1123054/synthesias-ai-clones-are-more-expressive-than-ever-soon-theyll-be-able-to-talk-back/amp/)).
- **Captions/Mirage.** An audio-to-video foundation model trained on talking "A-roll". The audio's pacing and
  dynamics drive the gestures ([arXiv 2506.08279](https://arxiv.org/abs/2506.08279v1)).
- **Arcads**: nothing technical is public.

Common pattern: (a) person-specific real footage, or a model trained on a lot of A-roll; (b) motion planned from
**audio and script**, separately from rendering; (c) an automatic scorer for how well motion matches speech;
(d) user cues on specific words. Our stack has none of (a)-(d). Our talking models are lip-sync models that add
whatever body motion their prior happens to give.

## 2. Open-weight options for motion (verified state, late 2026)

| Model | Licence | Body / hands | Length | VRAM | Evidence / verdict |
|---|---|---|---|---|---|
| **Wan2.2-S2V-14B** | Apache 2.0 ([repo](https://github.com/Wan-Video/Wan2.2)) | Half and full body; **accepts `--pose_video`**, so it lip-syncs audio while following a borrowed pose sequence | Follows the audio length | 80 GB single GPU (repo) | Earlier note found its own motion "conservative". **The pose input is the reason to retest it**: borrowed motion and lip sync in one pass |
| **LongCat-Video-Avatar 1.6** | MIT | "Coordinated facial and body motion"; the card has no gesture, pose or amplitude controls ([card](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6)) | 120 s+ | ~48 GB (INT8 on L40S) | Ours: no gestures in 90 s. Keep for long takes, not for mannerisms |
| **InfiniteTalk** (ours) | Apache 2.0 | Full body; V2V keeps camera and identity but **regenerates body motion from sparse keyframes**, not frame by frame ([paper](https://arxiv.org/html/2508.14033v1)) | Streaming | L40S ok | Gestures from a source clip can be smoothed away in V2V (see 3) |
| **EchoMimicV3 / -Flash** | Apache 2.0, code and weights ([repo](https://github.com/antgroup/echomimic_v3)) | "Talking body"; nothing specific on hands | 138+ frames with long-video CFG | 12-16 GB | Max 768x768, subtle motion. Cheap fallback only |
| **StableAvatar** | MIT code; weights licence unstated ([repo](https://github.com/Francis-Rings/StableAvatar)) | Full/half body claimed | "Infinite" | ~18 GB at 480x832 | 1.3B, 480p; NeurIPS 2026. Low priority |
| **OmniAvatar** | Apache 2.0 code ([repo](https://github.com/Omni-Avatar/OmniAvatar)) | "Adaptive body animation" | Token-bound | 8-36 GB | 480p only; superseded |
| **daVinci-MagiHuman** | Apache 2.0 (per [WaveSpeed](https://wavespeed.ai/blog/posts/davinci-magihuman-open-source-digital-human-lip-sync-2026/)) | Talking head; body not stated | 5 s clips tested | 1x H100 | 1080p, fast. Repo unverified |
| **Kandinsky S2V Pro-distill**, **LiveAvatar** | MIT / Apache | Whole figure | Long | 46-80 GB | See the 2026-11-23 note |
| HunyuanVideo-Avatar | Excludes EU/UK/KR | | | | Avoid |
| **TANGO** (gesture retrieval and reenactment) | **CC BY-NC 4.0** ([repo](https://github.com/CyberAgentAILab/TANGO)) | Body and hands, retrieved from a real clip graph | | | Not commercial, **but its idea is the one to copy** (3) |
| **EMAGE** (audio to SMPL-X face, body, hands) | Wrapper Apache-2.0 ([HF](https://huggingface.co/ZeyuLing/Motius-EMAGE-BEAT2)); BEAT data CC BY-NC ([PantoMatrix](https://github.com/PantoMatrix/PantoMatrix)); SMPL-X itself is non-commercial **[from memory, verify]** | Full | | small | A licence trap. Only usable as a research reference |
| MMGT, Cosh-DiT, EMO2, IMAE | Code partly public or none ([MMGT](https://arxiv.org/abs/2505.23120), [Cosh-DiT](https://link.springer.com/article/10.1007/s11263-026-02752-z)) | Upper body and hands | | | Not production-ready |

**Conclusion.** No open, commercially usable *audio-to-gesture* model good enough to ship was found. The open
models that render a person reliably are *pose-driven* (Wan-Animate, Wan-Animate-2, S2V's pose mode). So the
practical path is to **choose motion, then render it**: retrieve real motion (route 3), or pose prompts per beat.
The motion does not come from the lip-sync model's prior.

## 3. Borrowing real motion (motion only, never the likeness)

**Pipeline.** Script beat → pick a performance clip from the library (tags: style, energy, gesture type, framing,
duration) → **Wan-Animate (v1) animation mode** with *pose + face* videos from `preprocess_data.py`
([repo](https://github.com/Wan-Video/Wan2.2)), or Wan-Animate-2 (driven by raw video, no extractor;
[paper](https://www.alphaxiv.org/abs/2608.06009); Base weights "will release", licence unnamed) → lip sync onto the
new audio. Alternative in one pass: **S2V with `--pose_video` + audio**.

**Pitfalls.**
- *Identity leak.* Skeletons carry bone lengths and shoulder width, and face crops carry expression style. Wan-Animate-2
  gets its motion from raw video, so leakage is more likely (FixTalk shows motion features keep identity:
  [arXiv 2507.01390](https://arxiv.org/html/2507.01390v1)). Mitigations: use v1 with pose and face crops only;
  rescale the skeleton to the creator's proportions; performers of similar build; face-identity gate against the
  creator's references (we already have it); never store or use a performer's face as a reference image.
- *Lip conflict.* The face track carries the performer's mouth. Best fix: **the performer mimes our generated audio**
  while recording (mouth already roughly right; V2V only refines it). Otherwise neutralise the mouth in the face
  video (closed-mouth crop), or drive the face from S2V audio instead.
- *V2V wipes gestures.* InfiniteTalk V2V regenerates motion from keyframes, so borrowed gestures can be damped.
  Measure gesture energy before and after the lip sync (see 5). If it drops, use a **mouth-only inpainter**:
  LatentSync (weights licence: see our 2026-11-22 note) or MuseTalk (MIT per mirrors, but the official HF card says
  `creativeml-openrail-m`: [HF](https://huggingface.co/TMElyralab/musetalk)). Verify before use.
- *Hands.* Pose models still smear fingers in fast motion. Prefer library clips where hands move at mid speed and
  stay in frame; run the existing `detail.py` hands check.
- *Licence and consent.* Clips must come from performers who signed a **motion-only release** (paid, e.g. via
  freelance platforms) or from our own recordings. No scraped creator videos. Store the derived pose/face npz as the
  library asset; keep raw video only as long as the release allows.

**Library size [est]:** 5 content styles (story-time, rant/opinion, tutorial/explainer, GRWM chat, reaction) × ~15 clips
of 3-6 s = ~75 clips. Each clip is tagged by beat type (setup, build, punchline, aside, list item, question to camera),
energy (1-3), hands (none, one, both), posture shift, and gaze-away. A clip is chosen by matching tags to the
director's shot (`edit` and `performance` fields), never the same clip twice in a post. Small clips suit our 2-6 s
cuts: no long-take drift. This is TANGO's retrieval idea, done in our own pipeline with commercial-safe parts.

## 4. Cheap tricks that read as natural in edited social video

- **Cut before the model's stillness shows.** Most open models settle into a static pose after ~3-4 s **[est]**.
  2-4 s shots hide it. Cut on a breath or the end of a phrase, never mid-word.
- **Change framing every cut:** a 10-15% punch-in, one profile or 3/4 angle per post, one wide. Jump cuts on the same
  angle read as "creator edit" and hide pose resets.
- **Handheld camera, added in post:** low-frequency x/y/rotation noise (0.5-2 Hz, 0.3-1% of frame) on a 5% overscan
  crop, plus occasional micro-reframes. Cheaper and more controllable than prompting camera motion. One idle V2V
  source with real sway (already in the 2026-11-23 note) also works.
- **B-roll on the weakest seconds:** cover hands-heavy or stiff seconds with faceless shots from the aesthetics library
  while the voice carries on.
- **Speed ramps** of 1.05-1.15x on talking shots (pitch-preserved before lip sync, or the video only on B-roll).
  Faster delivery is native to the format and shortens stillness.
- **Blinks and gaze:** LivePortrait eye retargeting can add blinks and glances away on a finished clip; its code is MIT
  but its InsightFace detector is non-commercial, so use the MediaPipe detector
  ([LivePortraitKJ](https://github.com/kijai/ComfyUI-LivePortraitKJ)). Target ~26 blinks/min while talking (17 at rest;
  [Bentivoglio 1997](https://pubmed.ncbi.nlm.nih.gov/9399231/)), plus brief gaze-aways at thinking beats.
- **Sound** sells motion: room tone, clothing rustle on gestures, breath at the starts of phrases.

## 5. Measurable "lifeless" checks (automatic gate, report-only first)

All CPU or co-located, using MediaPipe Face/Pose/Hand Landmarker (Apache 2.0):
1. **Blink rate** from the eyeBlink blendshapes: flag below 8/min or above 45/min while talking **[est thresholds]**.
2. **Head-motion energy:** variance of yaw, pitch and roll and translation per second (as in THEval's head-motion
   dynamics, [arXiv 2511.04520](https://arxiv.org/html/2511.04520v1)). Flag near-zero variance over 2 s+, and also
   jitter (high-frequency energy).
3. **Gesture presence:** share of seconds with wrist speed above a threshold, and number of distinct hand poses.
   Flag a talking shot over 4 s with no gesture when the shot sheet asked for one. HKC/HKV (hand keypoint confidence
   and variance, used in [EchoMimicV2](https://arxiv.org/html/2411.10061v3)) for hand quality.
4. **Lip sync:** SyncNet Sync-C/Sync-D (LatentSync `stable_syncnet.pt`, licence caveats in our 2026-11-22 note) plus the
   existing MediaPipe mouth/RMS correlation. Our 0.325 should be calibrated against hosted reference clips. SyncNet
   scores correlate poorly with human preference ([arXiv 2511.04520](https://arxiv.org/html/2511.04520v1)), so use it
   as a floor, not a ranking.
5. **Gaze:** iris-to-eye-corner ratio. Flag 100% fixed camera stare over 10 s+ **[est]**.
6. **Gesture/speech alignment:** wrist-speed peaks within ±200 ms of stressed syllables **[est]**. Synthesia's
   Express-Eval does a learned version of this; ours is a heuristic.

Calibrate every threshold on ~20 real creator clips (consented or our own) and on HeyGen samples before enforcing.

## 6. Proposed A/B test (needs the owner's yes; not run)

One 20 s story-time script in Maren's designed voice, cut into 6 shots (2-5 s), same B-roll and finish for all arms:

- **A. Current:** InfiniteTalk per shot from stills, with cuts, punch-ins and handheld added in post.
- **B. Borrowed motion:** 3 library clips (one performer who mimed the TTS audio) → Wan-Animate v1 pose+face →
  InfiniteTalk V2V; plus B': mouth-only lip sync, if a licence-clean inpainter is ready.
- **C. One-pass pose+audio:** Wan2.2-S2V with `--pose_video` from the same clips (H100 80 GB). If that fails, LongCat 1.6 per shot.

**Compare:** the owner's blind 3-way ranking (main metric), plus Sync-C/D, mouth-RMS correlation, blink rate, head
variance, gesture-seconds share, face identity vs references, HKC/HKV, and gesture energy before and after V2V
(arm B), wall time and $ per finished second.

**Cost [est]:** A ~25 min on L40S (~$0.35); B ~45 min (~$0.65); C ~30-40 min on H100 (~$1.50). With one retry each
and boot/download time, **~$3-4 total**. Library clip: one 20 s recording of the owner or one paid performer (outside GPU spend).

Sources are linked inline.
