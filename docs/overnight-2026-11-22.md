# Overnight, 2026-11-22: video lock-down

Operations stayed paused. Nothing was published. No accounts, purchases or new paid services. The trend watcher
stayed off.

## The short version
- **Talking video works end to end on ComfyUI, and lip sync is now a hard gate.** At 6 steps, 1 of 3 clips was
  accepted. At 8 steps, the clips looked better but two of three lost lip sync. At **10 steps, 2 of 3 were
  accepted, all with lip sync 0.62–0.69. 10 steps is now the default**, at ~50 s of GPU per second of video
  (~$0.08 a 5 s clip on the L40S). The samples are small: re-measure on 8–10 clips before calling it locked.
- **Story clips (Wan 2.2 I2V) never rendered.** Each fix uncovered the next blocker on the shared weights volume.
  The last one (a full volume that couldn't even take its lock) is fixed in worker image v8, which builds this
  morning.
- **The GPU stopped at 08:09 UTC on the policy's own cap** (`gpu_runs.daily_gpu_usd: $10`, billed per UTC day).
  It worked as designed. I didn't raise it. That cap is yours, and it's below the $20 I allowed myself for the night.
- **The end-to-end hybrid post** (Maren, hook and sign-off lip-synced, cutaway photos in the middle) got through
  scene images and both cutaways. Its two talk segments were waiting for a GPU when the cap hit. It isn't finished.

## Spend (UTC 2026-11-22, to 10:30)
| What | $ |
|---|---|
| GPU billed (23 machines, all stopped; nothing live) | 10.04 |
| of which attributed to finished or failed jobs | 6.47 |
| LLM | 10.85 |
| Hosted API (Together video) | 0.50 |
| Fixed (weights volume) | 0.35 |

**LLM, $10.85.** Almost all of it came between 00:00 and 03:44 UTC:
- sandbox simulation: evaluator $3.7, showrunner $2.0, analyst $0.4
- model trials ("shadow"): $4.0, Kimi $1.5 of it

The trial cuts deployed at ~03:45. Since then the LLM spend is ~$0.35 an hour (the production showrunner and judge).
The sandbox runs were the M5 simulated loop from earlier. **Worth a decision:** give sandbox simulation its own
daily cap, or stop it until it's needed. I left it as is.

**GPU, $3.6 of $10 was idle or boot time.** Every new image version meant a cold boot plus weight checks. That
waste falls once the image stops changing.

## Talking video (ComfyUI InfiniteTalk, L40S with the weights volume, 3 clips each, 4.8–6.3 s)
Accepted means passing the visual gate and lip sync with correlation at least 0.3.

| Setting | Accepted | Lip-sync corr | Render (s) for 4.8 / 5.5 / 6.3 s of video | Notes |
|---|---|---|---|---|
| 6 steps (old default) | 1 of 3 | 0.61–0.70 | 168 / 165 / 319 | rejects: realism or motion 5 (hair clip doubles, face softens late) |
| 8 steps | 1 of 3 | **0.07**, 0.28, 0.60 | 226 / 222 / 421 | looks best, but lip sync collapses |
| **10 steps (new default)** | **2 of 3** | 0.62–0.69 | 273 / 273 / 400 | the reject scored realism and motion 4 |
| 8 steps, audio scale 1.5 | 1 of 3 | 0.54–0.69 | 219 / 216 / 333 | |
| 8 steps, audio cfg 2 | 1 of 1 | 0.70 | – / – / 607 | the extra pass doubles the time: dropped |
| 6 steps on a community 4090 ($0.34/h) | 0 of 3 | – | – | ComfyUI exited at start; v8 now reports why |

Speech WER was 0 throughout. The fp8-fast and torch.compile speed profiles are built and smoke-tested, but not yet
benchmarked.

## Story clips (ComfyUI Wan 2.2 I2V A14B)
Not yet rendered. Four blockers found and fixed in turn:
1. Two machines downloading to the shared volume at once broke huggingface_hub's records (UnboundLocalError).
   Fixed in v6: one downloader per volume, complete files skipped, broken records cleared and retried.
2. The 150 GB volume was full with the old research script's ~82 GB of weights ("Disk quota exceeded"). Fixed in
   v7: the worker prunes weights it no longer uses.
3. v7 was never reached: the full-volume failure was on a v6 machine. On a full volume even the lock file can't be
   written, so pruning inside the lock could never run, and the half-written lock would have blocked every machine
   for two hours. Fixed in v8: unused weights are pruned before the lock, and a lock that can't be written is
   removed.
4. When ComfyUI dies at start, the error now carries the end of its log (v8).

The hosted alternative is still Wan 2.7 on Together, at $0.29 per accepted clip.

## Built tonight
- **Model-trial spend cut.** Kimi is out of every trial. Trials sample 20% of calls (creator and judge were at
  100%). Trials are capped at $0.50/day. Judge trials run on vision models only.
- **Failed calls are ledgered.** An open-model call that times out is recorded at an estimated cost, since hosts
  bill for it. About $1.80 of Together's bill had been missing.
- **Hybrid talking video** (`generation.talk_hybrid`): the hook and sign-off are lip-synced, and enough of the
  middle is on camera to reach 40% of the words. The rest is cutaway photos under the same voice. Every segment is
  normalised to 720×1280, 30 fps, −14 LUFS and joined. Talking-video posts may plan one or two cutaway scenes.
- **The lip-sync gate is enforced** (`review.video.lip_sync.enforce: true`, min corr 0.3).
- **ComfyUI speed options to benchmark:** fp8 fast matmul (`fp8fast`) and torch.compile (`compiled`). The smoke
  test validates both graphs.
- **Worker images talkcomfy v4–v8:** C compiler, transformers pinned at 4.57.6 (v5 broke wav2vec), safe
  shared-volume downloads, pruning, and the v8 fixes.
- **Roadmap:** video cost levers, trend watcher next steps, sound effects (licence check first), new formats.

## Still queued (they run when the GPU cap resets at 00:00 UTC, within the $10 cap)
- 4 story clips (`bench-comfy.wan_i2v-1ac1a2`)
- 1 talk clip (`bench-talk.infinitetalk-4f3e84`)
- the e2e post's 2 talk segments

The e2e post only advances while its driver script runs. Restart it on the trends machine:
`nohup python /tmp/e2e.py > /tmp/e2e.log &`.

If you'd rather nothing runs before you look, mark those jobs dead.

## Needs you
- **Decide:** keep the $10/day GPU cap or raise it while video is being locked down. One full benchmark round
  costs ~$3–4.
- **Decide:** sandbox simulation's spend (above).
- ScrapeCreators API key (for the trend watcher).
- `NTFY_TOPIC`.
- `FLY_API_TOKEN` in GitHub. The deploy workflow passes but deploys nothing without it; I deployed by hand.
- The Meta setup in `docs/instagram-setup.md`.
- `talking_video` joins `production.formats` only once the e2e post passes. That isn't done yet, so there's no
  approval request.

## Evening build (2026-11-22, 18:50-22:00 UTC, owner away)
The owner asked for everything that can be built to be built, including professional captions, and for the plan
to keep finding and closing its own gaps. Plan: `docs/superpowers/plans/2026-11-22-video-quality-autonomous.md`,
with a running "Found gaps" list. Each wave: builders on separate files, then commit, CI green, push. No GPU spend
beyond the test queued for the 00:00 UTC cap reset, and no deploy, because a deploy would kill the test scripts.

### Built (all in CI and on main; deployed code is still the 18:40 version until the next deploy)
- **Direction foundation (afternoon):**
  - shot sheets
  - a director step: one Sonnet call per post, measured at $0.048
  - prompts compiled per model
  - per-creator designed voices (VoxCPM2)
  - one take per scene
  - LongCat as a second lip-sync model
  - a direction check in review
- **Walk-and-talk:** a moving shot is rendered first as a silent motion clip (Wan 2.2), chained up to the line's
  length, then lip-synced onto that clip with InfiniteTalk video-to-video.
- **Captions:** word-timed captions burned into the video.
  - Six styles: clean, karaoke, boxed, editorial, bold pop, minimal.
  - Fonts: open-licence (OFL) Poppins, Anton and DM Serif.
  - Kept inside Instagram's and TikTok's safe zones; words in `*asterisks*` are emphasised.
  - Timing comes from the voice's clause timings; word timestamps from transcription are built but off.
  - Rendered on the real control plane on Maren's accepted clip; samples are in `samples/captions/`.
- **Finish stage, for every video post:**
  - upscale to 720x1280, natural grade, subtle grain, two-pass loudness to -14 LUFS / -1 dBTP
  - captions, all in one ffmpeg pass
  - an automatic cover frame, used as the Reel cover
- **Natural speech:** synthesised clause by clause, with varied natural pauses, trimmed and faded joins; breaths
  and fillers are optional and off.
- **Natural eyes and idle movement** in talk prompts: irregular blinks, glances tied to the beats, and small
  movements matched to the scene's energy.
- **Sets and wardrobe:** each creator keeps a set and wardrobe bible, with a reference "plate" photo generated once
  per set and outfit. The references feed a consistency check. Reference-conditioned scene images
  (Qwen-Image-Edit-2511, Apache-2.0 confirmed) are built but off until benchmarked, on their own worker family.
- **Finer checks**, recorded but not enforced until calibrated:
  - hands and teeth, now with close-up crops from the worker
  - lips closing on b/p/m
  - background flicker
  - set and outfit consistency
- **Prompt budgets:** directed prompts fit each text encoder (Wan reads about 512 tokens) without losing camera,
  action or the exclusions.
- **720p close-up profile:** off until benchmarked.
- **Wave 3 adapters, all switched off:**
  - sound mix: ducking, room tone, mic and room character, and a library that refuses anything without a licence
  - effects and music generators
  - Wan-Animate-2 dance (our ComfyUI pin has the nodes)
  - Kling avatar comparison
  - a two-person design
- **Worker images:** talkcomfy v11 and image v26, built and smoke-tested; policy points at them.

### Waiting on the owner
- **LatentSync SyncNet licence** (a better lip-sync scorer): the code is Apache-2.0, the weights are "openrail++";
  see `docs/research/2026-11-22-latentsync-syncnet-licence.md`. It needs the owner's acceptance.
- **Stable Audio 3 Small-SFX for effects:** free only under $1M revenue, and its gate also binds the Gemma terms.
  Needs the owner's acceptance.
- **Hosted avatar comparison:** read the Kling API terms first (`docs/research/2026-11-22-hosted-avatar-terms.md`);
  it also needs a fal key. Hedra's acceptable-use policy bars "fake personas".
- **Dance:** needs reference clips we own.

### Unproven until a GPU run
- VoxCPM2 voices
- LongCat
- walk-and-talk V2V
- crops
- the Qwen edit model
- the 720p profile

The 00:00 UTC test covers the voice, InfiniteTalk vs LongCat, a 30 s single take and a walk-to-camera clip. That
test was queued with the 18:40 code, so it doesn't include tonight's captions, finish stage or prompt budgets.
Those reach production at the next deploy, which waits until the test finishes. The remaining gaps are listed in
the plan's "Found gaps".
