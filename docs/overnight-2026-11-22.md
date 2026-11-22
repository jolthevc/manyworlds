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
