# Technical debt

Known issues we chose not to fix yet, with when they must be fixed. "Before X" means the
capability X does not go live until the item is closed. Updated 2026-11-14 (Brain M1 merge).

## Noted 2026-11-23 (overnight build and the review of waves 1-2)

| Item | Why it matters | Where |
|---|---|---|
| `test_a_finding_becomes_an_experiment_whose_result_is_proven` is flaky: slot assignment seeds from the experiment's random UUID, and 10 posts per arm sometimes miss the planted effect (failed once on 2026-11-21 on a policy-only commit) | `tests/test_brain_learning_trace.py`, `manyworlds/brain/experiments.py` |
| The no-output guard counts a machine as failed when it tried work and finished none; a machine cut off by an interruptible host mid-first-job also counts | at worst it holds a family until the window passes or a new image tag ships, and tells the owner | `manyworlds/deploy/autoscale.py` |
| The talent manager has never run against a real model | its dossier sections and weekly briefs are tested with stubs only; the first real onboarding should be read by the owner before any brief steers production | `brain/agents/talent_manager.py`, `creator/dossier.py` |
| Market leads have no live test and no creator has a market yet (creators.market_id is never set at launch) | the lead activates only for markets with 3+ creators; set the market from the thesis at launch | `brain/launches.py`, `agents/market_lead.py` |
| Accepted cohort requests are recorded but do not create theses or launches | the CEO still launches from theses; a cohort should become theses for the launch flow | `brain/proposals.py`, `brain/agents/strategist.py` |
| Paid idle time is not yet filled with useful work (extra takes, best-of-N candidates) | a warm machine that drains its queue idles until the idle stop; filling it would turn already-paid minutes into quality | `worker/runner.py`, `deploy/autoscale.py` |
| Memory floors (`job_kinds.<kind>.min_vram_gb`) are set only for Wan-Animate-2 | Z-Image's OOMs on 24 GB consumer cards may need a floor or a free-memory check; set others once measured | policy `job_kinds` |
| A narrated edit's labels are timed as if photos split the video evenly | labels drift from their photo (narrated_edit is not in production.formats today) | `production.shot_spans` |
| Other test batches can share the merged breadth machine, or start a second L40S for the same kinds | run the breadth test with no other test batches open | `deploy/autoscale.py` scale_up |
| Set and outfit plates are made only by the producer tick, which is off while paused | breadth items that test set/outfit consistency run without plates | `creator/sets.py`, `breadth.py` |
| If the test overlay file goes missing, the autoscaler falls back to separate machines silently | three cold L40S boots instead of one | `policy.for_batches` |
| The video clear-pass and the image clear-pass still read identity on POV/cutaway clips | a faceless clip with a low identity score is confirmed (paid) rather than cleared | `review/video.py` clear_pass, `review/gates.py` |

## Before autonomous publishing (Brain M4)

Closed 2026-11-20: the brain worker reconnects after a dropped connection and rebuilds its Claude client when
the policy changes (`runtime.serve`); only an unknown agent is a permanent failure; a crashing job no longer
ends the tick for the others.

| Item | Why it matters | Where |
|---|---|---|

## Brain M6 (experiments), noted in review 2026-11-20

| Item | Why it matters | Where |
|---|---|---|
| A `no_difference` result is permanent (one experiment per creator, effect and mode, ever), even when the test had low power | a real effect on a noisy creator could be written off; the result records achieved `power`, so a later revision can reopen low-power nulls | `manyworlds/brain/experiments.py` |
| Nothing checks that an experiment idea actually followed its arm instruction | arms are compared as assigned (diluting, not biasing, the difference) | showrunner slot handling |
| Experiments are slow: ~36 posts per arm at 2 per arm per batch is ~54 days, one open at a time | learning pace; raise `slots_per_batch` or `max_open_per_creator` once creators post more | policy `brain.experiments` |

## Brain M7 (QA and repair), noted in review 2026-11-20

| Item | Why it matters | Where |
|---|---|---|
| A video rerouted to a photo keeps the caption and script written for the video | the caption may mention movement or speech the photo doesn't have; a caption rewrite (one cheap LLM call) would fix it | `manyworlds/production.py` reroute |
| Feed repetition is an exact match on free-text attributes ("cafe" vs "cozy cafe" don't match) | conservative: it misses some repeats, never blocks wrongly | `manyworlds/brain/feed_qa.py` |
| With `review.image.min_scene: 0` the scene gate never fails, so the scene repair can't trigger | only matters if scene becomes a hard gate again | `manyworlds/review/repair.py` |
| Flavors that need text on the image (tier list, recipe card, text-on-photo memes) aren't offered | needs a text-overlay step for photos (the narrated edit already burns captions into video) | policy `flavors` |

## Brain M8 / M4, noted in review 2026-11-20

| Item | Why it matters | Where |
|---|---|---|
| No differentiation check of a new creator against its siblings (spec 4.6a) | two launches could end up too alike; the strategist is told to diversify, nothing measures it | `manyworlds/brain/launches.py` bible step |
| An event over the daily rate limit is skipped, so it can fire late the next day; `qa_surge` payload has no `mode` label | minor noise for the owner | `manyworlds/brain/events.py` |
| Event detectors query every creator's last 60 posts each producer tick | fine at today's scale; batch it before dozens of creators | `manyworlds/brain/events.py` |
| JPEG copies for Instagram (`public/` in the bucket) are never cleaned up | small storage growth | `manyworlds/publish/instagram.py` |
| Presigned R2 URLs for Instagram are unverified against Meta's fetcher | first real publish may fail; it then holds the post for the owner | first connected account |
| Story `is_ai_generated` support is not stated in Meta's docs | if Meta rejects it on Stories, Story posts will be held | `manyworlds/publish/instagram.py` |

## Before talking video goes live (M10)

| Item | Why it matters | Where |
|---|---|---|
| The video gate judges keyframes, so it cannot see whether the mouth matches the words | an out-of-sync clip could pass; until a sync score (e.g. SyncNet confidence) is computed on the worker, the benchmark clips are checked by watching them | `manyworlds/review/video.py`, `talk_infinitetalk.py` |
| InfiniteTalk weights (~94 GB) download from Hugging Face on every new talk machine | each boot pays the download; mirror the base model and adapter into R2 once talk runs regularly (`models.mirror` with an allow-list, the adapter repo is 169 GB) | `manyworlds/worker/models.py` |

## Platform hardening, noted 2026-11-21

| Item | Why it matters | Where |
|---|---|---|
| GPU machines still get the control plane's R2 keys (`security.worker_storage: shared`) | a hostile host could read or delete any object in the bucket; needs a bucket-scoped R2 token in `WORKER_AWS_*`, or presigned URLs per job | `manyworlds/deploy/credentials.py` |
| The worker role can update any job and any attempt, not only its own | a hostile host could disrupt the queue (not read research, creators or the ledger); row-level security or a worker API on Fly would close it | migration 022 |
| `MW_HASH_SALT` is unset, so author handles hash with the code's default salt | the salt is public in the repo, so hashes of known handles can be recomputed; rotating it makes old and new hashes incomparable, so do it with a one-off rehash | `manyworlds/research/apify.py` |
| Long talking videos: each segment is voiced separately (Kokoro), so intonation resets at each cut, and the joined video isn't re-reviewed (each segment passed the gate) | small audible seams; `chain` mode can drift in identity over many segments (each segment is still judged against the references) | `manyworlds/production.py`, `talk_segments.py` |
| Shadow tests sample with `random.random()` and compare fields mechanically | fine for a first read; scoring roles should also be checked against owner labels (M2 calibration) before switching | `manyworlds/llm.py`, `shadow.py` |
| Alerts have one channel (ntfy) and the topic is the only secret | anyone with the topic reads alerts (ids and reasons only); add a second channel (email) if alerts become critical | `manyworlds/notify.py` |

## Before the Brain allocates real money (Brain M8/M9)

| Item | Why it matters | Where |
|---|---|---|
| `effective_policy` reads override rows and the version in two queries | under concurrent changes a decision could record a version that doesn't exactly match the settings it used | `manyworlds/brain/overrides.py` |
| Only Brain agents read the effective policy (`brain.authority.observed_sections: [brain]`); overrides of production, review or autoscaler settings are refused | the strategist can't tune production until those services read the effective policy; then add their sections to `observed_sections` | services' policy loading |

## Later

| Item | Where |
|---|---|
| `mw brain run <agent>` runs a whole tick and may run other queued jobs first | `manyworlds/cli.py` |
| `_waiting_kinds` scans the jobs table on each claim (fine at today's volume) | `manyworlds/brain/runtime.py` |

## Closed at the M1 merge

- Simulated data could in principle be read through raw queries: now structural (separate
  `simulated_*` tables, CHECK constraints on the real ones, guard test over the whole package).
- Approve-then-apply was not atomic; override versions could collide: one transaction, unique version.
- Brain LLM spend mixed into cost per output: now reported separately; cost per accepted output
  counts GPU and review only.
- Publications could be marked simulated on a real platform: CHECK constraint.

## Creative evaluation (Brain M2), deferred minors

| Item | Before | Where |
|---|---|---|
| A dimension scored by only one of three samples reports spread 0 (looks certain) | rubric recalibration by predictive validity (M6) | `brain/evaluation.py` `score` |
| A missing asset/idea row or a non-image asset fails retryably with a TypeError instead of a clear non-retryable reason; media type guessed from extension (png/jpeg only) | video keyframe scoring (M10) | `brain/agents/evaluator.py` |
| `checked()` depends on how an id is written (case, braces) | — | `brain/evaluation.py` |
| A crash between storing evaluation rows and the remembered decision can duplicate rows on rerun | M6 | `brain/agents/evaluator.py` |
| `mw brain calibrate`/`rubric-drift` don't check the evaluator's daily budget | — | `manyworlds/cli.py` |
| Rubric filename version not cross-checked with the file's `version:`; a stray `idea-vdraft.yaml` crashes; schema cache keyed without file hash; `register` race | — | `brain/rubrics.py`, `brain/evaluation.py` |
| Rubrics always load from the repo/`/app` path, not from `MW_POLICY`'s directory | — | `brain/rubrics.py` |
| Reference set and drift exist for the idea rubric only; execution and feed have none yet | M7 (feed QA) | `policy/rubrics/reference/` |

## Memory and showrunner (Brain M3), deferred minors

| Item | Before | Where |
|---|---|---|
| Canon enters memory when a post is scheduled; a post later dropped leaves its canon in memory | M4/M5 (move to publication time) | `brain/ideas.py` `schedule_ready` |
| Ideas dropped after max_wait leave their evaluator jobs queued (they still spend evaluator budget) | — | `brain/ideas.py` |
| Low-scoring ideas are scheduled anyway (by design: no dimension is a target); revisit once scores are shown to predict performance | M6 | — |
| Switching `brain.showrunner.enabled` off while a batch is proposed lets the planner plan the same period | — | `cli._produce` |
| A post linked to two ideas (from `from_post` and a scheduled idea) appears twice in the showrunner's recent posts | — | `brain/agents/showrunner.py` |
| Scheduled posts record the creator's profile version at scheduling time, not the version the batch was planned under | M6 | `brain/ideas.py` |
| Only `signature` (and keyed canon) close older facts; seeded places/people have no keys, so a moved home relies on the showrunner setting `canon_key` | — | `brain/memory.py` |

## Simulated loop and analyst (Brain M5), deferred minors

| Item | Before | Where |
|---|---|---|
| Trend bias during baseline warm-up could show as a dimension slope when scores drift over time; only the format-shift case is tested | M6 | `brain/analysis.py` `normalise` |
| Re-analysing every period gives repeated looks: transient tentative findings can appear in a null world; consider requiring persistence across two analyses | M6 (experiments) | `brain/analysis.py`, analyst |
| Effects are marginal (a dimension correlated with a format picks up the format's effect); say so in the analyst prompt or adjust | M6 | `brain/analysis.py` |
| Baseline is account-wide, not per format (spec 4.5 says per account and format); intentional so format effects share a baseline | — | `brain/analysis.py` |
| If `simulate_post` fails after a post is marked simulated, that post never gets outcomes; check-then-insert can duplicate under concurrent runs | — | `publish/scheduler.py`, `brain/simulation.py` |
| Stall recovery retries every `retry_after_hours` indefinitely (one event per period), bounded only by budget | — | `brain/showrunner_loop.py` |
| Small (the CLI's double `Claude` fixed 2026-11-21): a paused producer prints every loop; `sandbox.clone` crashes on an unknown handle; findings queried twice in the showrunner prompt; `effects()` writes `y` into the caller's rows | — | various |
| `human_approval` / `sampled` hold real posts but no approval flow exists yet | before switching from autonomous | `publish/scheduler.py` |
