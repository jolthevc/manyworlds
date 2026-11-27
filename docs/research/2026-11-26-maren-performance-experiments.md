# Maren performance experiments (2026-11-26)

The three experiments approved by the owner (cap $3), run in batches `bench-exp1126-*`. They were sandbox runs only: nothing was published.

**Total spend: $1.00.** That includes one machine ($0.10) started for two motion jobs that were cancelled before they ran.

## 1. Voice: Qwen3-TTS vs IndexTTS-2.5

**Setup:**
- The same two Qwen3-designed Maren voices.
- Each engine spoke the five casting lines; the audio was kept for listening.
- IndexTTS spoke three Maren lines (excited, confiding, amused) at emotion strength 0.35, 0.75 and 1.0.
- Qwen3 spoke the same three lines as a baseline. It has no per-line emotion control.

**Results:**

| | Qwen3 | IndexTTS-2.5 |
|---|---|---|
| MOS (quality), cast lines | 4.38–4.54 | 2.56–4.36 |
| Cast lines passing the line check | 10/10 | 5/10 |
| Pitch movement (f0 std, semitones) | 1.7–4.8 | 2.0–4.6 |
| Emotion strength 0.35 → 0.75 → 1.0 | — | MOS fell (excited: 4.39 → 3.63 → 3.44). Pitch movement did not rise in order. |

**Reading:**
- Qwen3 is the better-sounding engine.
- IndexTTS's emotion control lowered quality without reliably adding expressiveness.
- Maren stays on Qwen3, pending the owner's listening verdict.

## 2. Photo performance controls (Z-Image Turbo, LoRA)

**Setup:**
- One bar shot and one seed.
- Six directions: sass 0 and 2, candidness 2 and 4, warmth 2 and 4. Each stays inside her natural range from the talent manager's performance profile.
- Four candidates per direction, 24 images in all.
- Every image passed every gate.

| Requested | Observed by the judge (mean) |
|---|---|
| sass 0 | 1.25 |
| sass 2 | 2.25 |
| candidness 2 | 3.0 |
| candidness 4 | 2.0 (inverted) |
| warmth 2 | 2.0 |
| warmth 4 | 2.0 (no change) |

**Reading:**
- Sass is controllable from words; the faces show it.
- Candidness and warmth are not, with a fixed composition.
- The seed fixes pose and framing. Performance phrases move only the face, so candidness and energy need different blocking and framing.
- The judge's appeal, charisma, expression and naturalness scores cluster between 6 and 8.

**Fixes found on the way** (all deployed):
- The performance section was trimmed out of every photo prompt.
- The set vanished when the prompt was trimmed.
- A dangling channel label ("camera.") was left behind by trimming.
- The beat's eyeline and the performance's gaze contradicted each other.

## 3. Talking video: photo start vs video start, energy 0 vs 2

**Setup:**
- One line (Qwen3) and one start photo.
- Image-to-video lip sync from the photo, compared with a directed Wan 2.2 motion clip lip-synced by InfiniteTalk video-to-video.

| | Lip sync (bar 0.3) | Gates | Observed energy | Naturalness |
|---|---|---|---|---|
| Photo start, energy 0 | 0.68 | all pass | 2 | 6 |
| Photo start, energy 2 | 0.68 | all pass | 2 | 5 |
| Video start, energy 0 | 0.06 | lip sync fails | — | — |
| Video start, energy 2 | 0.28 | lip sync fails | — | — |

**Reading:**
- Photo-start clips lip-sync well. They now open with a lead-in and end on a small reaction, not a blank stare.
- The energy requested in words barely changes the face: the voice drives it.
- Video-start clips look the most natural (subtle expression, real body movement, a wry ending), but the mouth barely moves.

**A defect found, and its real cause:** we were saving the source clip, not the render.
- Rerunning both clips at start step 1 produced byte-identical files with identical lip-sync scores.
- The upstream sampler does honour `start_step` (traced in WanVideoWrapper at our pinned commit), and our graph matches Kijai's V2V example.
- The cause was ours. ComfyUI v0.45's core `LoadVideo` lists its input file among a prompt's outputs, as a preview of type `input`. Our `comfy.run()` took the first `.mp4` it found: the source motion clip.
- It also returned as soon as that preview appeared, before the sampler finished. That is why video-start jobs took 88–120 GPU seconds, against 237 or more for photo start.
- So every walk-and-talk "render" and repair since the core-node swap measured the untouched motion clip with the voice muxed on.
- The same picker served multi-window graphs and motion transfer. The 2026-11-24 dance "pass" probably judged the driving clip.
- Fixed in talkcomfy v33: only files saved as `output` are taken. The two video-start clips were rerun to verify.

**After the fix (start step 2):**
- The clips are real renders now. Her mouth visibly speaks while the source clip's framing and body movement are kept, and the files differ from before.
- Both still fail the lip-sync check (corr 0.16 and −0.07, bar 0.3).
- The repair ladder (start step 1, then 0) now changes the render, so it can be measured.
- Untested: a timing offset in V2V, and whether half the frame is too little redraw for the mouth to follow the audio.
- V2V takes still have no lead-in or end hold (review finding, open).
- Spend for all three experiments: $1.09 of $3.

**Prompt fixes found on the way** (all deployed):
- Walk phrases were going into a standing shot.
- The performance's body phrase contradicted the beat's blocked hands.

## What this changes

- **Voice:** Qwen3 stays, subject to the owner's ear.
- **Photo direction:** expression controls work through the face. Pose, candidness and energy need changes to blocking and composition, not adjectives. The director should vary blocking per candidate when variety is wanted, not only the seed.
- **Talking video:** audio-first is confirmed as the main lever. Video start is the most natural-looking path, but it is blocked by the start-step defect.
- **Judges:** the observed levels were plausible for sass. Calibration rightly stays off until judge agreement has been measured against the owner's ratings.
