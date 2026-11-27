# Maren performance experiments (2026-11-26)

The three experiments approved by the owner (cap $3), run in batches `bench-exp1126-*`. They were sandbox runs only: nothing was published.

**Total spend: $1.89 of the $3 cap** (to 2026-11-27, the repair ladder included). That includes one machine ($0.10) started for two motion jobs that were cancelled before they ran, and ~$0.08 on a 4090 host where ComfyUI would not start (the job moved to an L40S).

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
- V2V takes had no lead-in or end hold (review finding; fixed in talkcomfy v35, §5).
- Spend for all three experiments at that point: $1.09 of $3.

**Prompt fixes found on the way** (all deployed):
- Walk phrases were going into a standing shot.
- The performance's body phrase contradicted the beat's blocked hands.

## 4. Audit: what the output-picker bug invalidated

**Scope.** Since 2026-11-22, every graph that loads a video with core `LoadVideo` returned that input as its
"render". There were 65 jobs: InfiniteTalk V2V (walk-and-talk) and Wan-Animate-2 (dance). Graphs that start from a
photo were unaffected: photo-start lip sync, story clips, LongCat and LTX.

| Claim | Status | Evidence |
|---|---|---|
| 2026-11-24 breadth: "dance passed" | **Void** | Both dance outputs are byte-for-byte the driving clips (sha256), and their identity reviews failed. |
| 2026-11-24/25: walk-and-talk drops were "V2V that never redrew the mouth" | **Void** | Each take was the silent source clip with the voice muxed on. |
| The "escalating V2V repairs" (lower start_step) | **Untested until now** | Repairs reran the same passthrough, so the scores were byte-identical. |
| perf-v3b: 5 walk-and-talk takes "passed lip sync" (corr 0.30–0.83) | **Void, and they expose a gate weakness** | They were untouched source clips. See "Lip-sync gate" below. |
| V2V and dance cost and render time (88–160 GPU s per take) | **Void** | The old picker returned as soon as the input preview appeared, before the sampler ran. A real V2V render took 146 s warm and 544 s cold. |
| Wan-Animate-2 cost and quality | **Unknown** | No real dance render has been measured. |
| "V2V looks the most natural" (§3, first version) | **Void** | That was the source motion clip itself. Only its motion quality is measured, not the lip-synced result. |
| Capability card, pattern cards | **Unaffected** | No card or capability row was ever computed from these jobs (both tables are empty). |

**Lip-sync gate.** The gate passes a clip when mouth opening correlates with the voice's loudness at r ≥ 0.3.
- Speech loudness moves slowly, so a mouth that opens now and then correlates with the voice almost as well when the voice is shifted 0.5–1.6 s.
- Against that null (the 95th percentile of correlations at 13–40 frames of shift):

| Takes | Real correlation | Null | Margin |
|---|---|---|---|
| Fake takes | up to 0.87 | 0.87 | ≤ 0.10 |
| Older photo-start takes | 0.18–0.79 | — | −0.13 to +0.30 |
| Today's photo-start takes (silent lead-in and hold) | 0.66–0.67 | 0.00–0.13 | +0.54 to +0.66 |

The margin is now recorded on every lip_sync row (`review.video.lip_sync.chance`). It is report-only until it is calibrated against clips judged by eye. The lead-in also makes the score far more informative.

**Timing checks before any new render** (§3's two real renders): the cheap, deterministic causes are ruled out.
- The mouth opens at frame 3 against speech at frame 1, and closes at frame 90 against 88.
- There is no global offset and no 16/25 fps speed mismatch.
- The motion clip (5.06 s) covers the line (3.69 s).

The weak articulation is in the render itself.

## 5. The V2V repair ladder (2026-11-27, talkcomfy v35, $0.72 of the cap)

**Setup:**
- The same two motion clips, line, seed and prompt as §3.
- Rendered at start steps 2, 1 and 0. Lower steps redraw more of the clip.
- Every take now has the 0.3 s lead-in and 0.6 s end hold, and the motion clip covers the padded line.
- Each render's provenance was checked: the sampler ran (not cached), with the start step requested.
- The judges ran with lip sync set to report-only, so every take was judged.

| Clip | Start step | Sync corr | Margin over chance | Body kept (SSIM vs source) | Identity / realism / motion | All gates | GPU s (warm) |
|---|---|---|---|---|---|---|---|
| energy 0, before (§3, no lead-in) | 2 | 0.16 | −0.17 | 0.85 | — | lip sync fails | 544 (cold) |
| energy 0 | 2 | 0.52 | +0.17 | 0.86 | 6 / 6 / 6 | **fails realism**: face warped mid-turn | 890 (cold, first job) |
| energy 0 | 1 | 0.64 | +0.45 | 0.79 | 8 / 7 / 7 | pass | 142 |
| energy 0 | 0 | 0.66 | +0.52 | 0.69 | 7 / 8 / 8 | pass | 179 |
| energy 2, before (§3, no lead-in) | 2 | −0.08 | −0.61 | 0.85 | — | lip sync fails | 146 |
| energy 2 | 2 | 0.63 | +0.49 | 0.87 | 8 / 7 / 8 | pass | 131 |
| energy 2 | 1 | 0.64 | +0.54 | 0.80 | 8 / 7 / 8 | pass | 155 |
| energy 2 | 0 | 0.68 | +0.60 | 0.69 | 8 / 7 / 8 | pass | 175 |

**Reading:**
- **The lead-in and hold were the main fix.** At the same start step, seed and clip, sync rose from 0.16 and −0.08 to 0.52 and 0.63, and both now beat chance.
  - Likely cause: InfiniteTalk's first frame is the clip's conditioning frame. A line that starts at frame 1 gives the mouth no room, and the silent hold settles the close.
  - This is a likely mechanism, not tested in isolation.
- **Lower start steps buy a little more sync and cost body movement.** Each step down adds about 0.02–0.12 corr and loses about 0.07–0.10 SSIM against the source motion.
  - At step 0 a third of the source clip's look is gone, so the directed motion is partly lost.
- **The default is now start step 1** (policy `generation.talk_motion.v2v`).
  - It keeps most of the motion, syncs on both clips and passed every gate on both.
  - The repair ladder (1, then 0) stays for takes that still fail.
- Only two clips and one line were tested. Treat this as a starting point, not a calibration.
- The mouth leads the voice by 1–4 frames (40–160 ms) at its best lag. That is inside the lag search; worth watching.
- **Cost:**
  - A warm V2V take takes 131–179 GPU s on an L40S, about $0.04–0.06 attributed.
  - The ladder's whole bill, across six takes, was $0.64 of GPU: about $0.11 a take all-in, with a cold boot and the failed 4090 host included.
  - Judging cost about $0.08 of Claude.
- Appeal and naturalness were not compared. Those scores are written in a separate review row, which the report-only pass does not return.

## What this changes

- **Voice:** Qwen3 stays, subject to the owner's ear.
- **Photo direction:** expression controls work through the face. Pose, candidness and energy need changes to blocking and composition, not adjectives. The director should vary blocking per candidate when variety is wanted, not only the seed.
- **Talking video:** audio-first is confirmed as the main lever. Video start (walk-and-talk V2V) now works: with the lead-in and hold, five of six takes passed every gate (§5). The default is start step 1.
- **Judges:** the observed levels were plausible for sass. Calibration rightly stays off until judge agreement has been measured against the owner's ratings.
