# Overnight build, 2026-11-22 (proposal; runs only on the owner's go)

Rules that bind the whole night: no GPU machine, no benchmark, no breadth test; no new paid services; Claude spend
for review and analysis under $5 in total; every change through CI on `main`; no live switches (`operations.mode`
stays `paused`, launches 0, publishing paused). The breadth test waits for the owner in the morning.

## Where the money has gone (ledger, all time, read 2026-11-22)

| | $ |
|---|---|
| GPU billed by providers | 42.28 |
| GPU time that did a job (finished or failed) | 10.04 (24% of billed) |
| LLM | 34.31 (evaluator 17.45 incl. sandbox, shadow tests 5.32, sandbox showrunner 5.01) |
| Data (Apify) | 1.87 |
| Real posts made: 9 ready, 25 dropped, 1 producing | |

- 76% of GPU dollars bought boots, idle tails and failures. Idle-stop machines alone billed $15.91. Two Vast talk
  machines billed $10.27 and finished nothing.
- Failed attempts by cause:
  - 320 Z-Image CUDA out-of-memory failures (consumer cards with less VRAM than advertised)
  - 60 "CUDA unknown error" (bad hosts, retried anyway)
  - 24 package or signature errors that a smoke test should have caught
  - 49 InfiniteTalk failures, averaging 5 minutes of GPU time each
  - 8 dead `comfy.wan_i2v` jobs (disk quota, missing C compiler)
- 74% of real posts were dropped, nearly all as "scene N: no image passed review after 4 repairs". Each one paid
  for every repair before it was dropped.
- Judging costs more than making: the evaluator spent more than all useful GPU work combined. `llm.py` uses no
  prompt caching.

## Plan (in order; each block ends green on CI or is reported as not done)

**A. Review and fix the last two waves (≈2 h).**
- Three fresh reviewers in parallel, each on part of the diff since `65ea98d^` (production/finishing/media,
  direction/brain prompts/review, breadth/policy/autoscale/workers).
- Verify each finding, fix the confirmed ones with tests, and list the rejected ones and why.

**B. A dry rehearsal of the breadth test without GPUs (≈1 h).**
- Run all 21 items end to end on the `fake.*` handlers in CI, so every step after the GPU (join, finish, captions,
  overlays, sound, cover, final review, report) is exercised on real ffmpeg.
- What a GPU run can still break is then only the models themselves.

**C. GPU dollars that do work (≈2.5 h, the biggest money).**
1. Classify failures:
   - A host failure (CUDA unknown error, OOM on a card below the model's floor) marks the host and machine bad and
     reruns elsewhere at once.
   - A code failure (TypeError, missing package or compiler, the same error on two machines) stops retrying and
     tells the owner. No more 5-minute InfiniteTalk retries of a deterministic bug.
2. VRAM floors per job kind in policy, checked against measured free memory at worker start. A card that can't fit
   Z-Image never claims it.
3. Model-affinity claiming: a worker takes jobs for the model it already has loaded first. A merged machine runs all
   voice work, then all motion, then all lip sync, instead of swapping 30 GB models back and forth.
4. Fill the paid tail: when a warm machine's queue empties, it first runs queued lower-priority work its family can
   take, and extra candidates the production plan marked "optional" (best-of-N scene images, a second take), up to a
   policy cap. Idle time already paid for becomes quality.
5. A per-machine "rent efficiency" (job seconds ÷ billed seconds) in `mw machines` and the ledger report. Low
   efficiency on a pool lowers its score in `scoring.py`.
6. Preflight on the control plane before queueing (inputs exist, payload compiles, sizes fit the model), so bad jobs
   never cost a boot.

**D. LLM spend (≈1 h).**
- Prompt caching on the stable prefixes (rubrics, bibles, shot-sheet schema).
- `evaluator_check` off Opus to a cheaper model set in policy.
- Batch evaluator calls per post.
- The sandbox evaluator kept on its own budget.
- Target: at least halve the judge's cost per post, with no change to scores on the existing calibration set
  (compare offline from stored calls, with no new spend).

**E. Why images fail review (≈1 h, read-only analysis plus fixes).**
- Read the 25 drops' review rows: which gate, which reasons.
- Then fix what's fixable:
  - wording that produces the failure
  - gate thresholds miscalibrated against what passed
  - repairs that resend the same prompt
- Make repairs cheaper: first try a stronger scene variant, then stop sooner. Don't pay for four similar retries.

**F. Paperwork (≈0.5 h).**
- Update CLAUDE.md status and tech debt.
- A morning report covering what changed, what didn't, and anything that needs the owner.

Proposed but not started unless the owner says so: the archetype library (roadmap item 1). It changes what the
Brain plans, so it deserves its own go.
