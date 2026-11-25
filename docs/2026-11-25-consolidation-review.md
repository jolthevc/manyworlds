# Consolidation review, 2026-11-25

## How it was done

**Six independent reviewers.** Each started from a fresh context, read-only, with instructions to challenge the existing code rather than trust it. Each had one area:

1. orchestration and state;
2. budgets, GPU and weights;
3. agents, prompts and LLM cost;
4. safety, security and publishing;
5. QA and test quality;
6. research data and rights.

**Verification.** I checked their findings against the code, fixed the critical ones with regression tests that run through the production path, and verified the risky changes on the live control plane.

**Test suite.** 1,801 tests pass on CI; 1,770 passed at the start of this review.

## 1. Fixed this session (each with a regression test)

| # | Finding (severity) | Fix |
|---|---|---|
| 1 | **Rented GPU hosts could rewrite any job, forge ledger rows and push back their own deadline** (high). One malicious host could queue paid brain work while paused with an uncapped budget, hide spend from the daily cap, and outlive its deadline. | Migration 050, verified live (control plane still sees all 1,219 jobs): <br>• workers update only the lease and result columns of GPU jobs; row security hides brain jobs from them<br>• workers insert only non-negative `gpu` ledger rows<br>• a worker can only bring its deadline forward |
| 2 | **Media "pause" not enforced** (high). Metadata answers carry media URLs, so the monitor kept downloading third-party media. | `store_media` refuses unless a source is approved for transient media. No source is now: Apify, ScrapeCreators and Graph media are all off. |
| 3 | **Completion headroom unsafe** (high). It worked per job kind, so new posts rode on it, and its end killed in-flight renders. | Turned off (`completion_usd: 0`) until rebuilt per job. One-day owner grants remain. |
| 4 | **Candidate ranking summed report rows** (high). `distribution_risk` is "higher is worse", so the riskiest photo ranked first. | Ranking uses core gates only (identity, realism, scene, direction), in both selectors. |
| 5 | **One rejecting confirmation discarded all other screen-passed candidates** (high). With the shipped `confirm_top_per_job: 1`, this regenerated the scene and burned repairs. | Confirmation walks down to the next screen passes (up to 3). |
| 6 | **Talking-video route hand-off depended on score noise** (high). A route kept re-seeding until the shared cap dropped the post; the fallback was rarely reached. | Each route hands on after its own repairs, whatever the scores. |
| 7 | **The talking-video → photo reroute was unreachable** for the main failure path, which matched a phrase in the drop reason. | Structured `body.drop_kind`. |
| 8 | Hands cutaways showed a face ("phone held by someone else"; "…, no face" put first by repairs). | Point-of-view framing from her eyes; negations moved out of prompts. |
| 9 | Follower shots copied the lead photo (four hands; faces in cutaways). | An empty set plate is made first and followers are drawn into it; anatomy and framing checks now reject. |
| 10 | Talk route cap pre-empted the fallback. | Cap covers every route (see #6 for the fuller fix). |
| 11 | **Research statistics overconfident** (high). Many tests at p ≤ 0.1 with no correction; a single creator could make a "correlation"; 3 outcomes counted as "validated". | Multiple-testing-corrected `q`; ≥ 3 creators per cluster; ≥ 10 posts per side; ≥ 10 real outcomes to validate. |
| 12 | Research distill could overwrite reviewed and seed cards (medium-high, a poisoning route). | Research writes a sibling card; it never supersedes active or seed cards. |
| 13 | Third-party text (captions, bios, comments, web pages) was not framed as data in prompts (medium). | Research, distill, comment and desk prompts say it is data, never instructions. No agent that reads third-party text can change settings, spend outside caps or publish. Verified by the security reviewer. |
| 14 | Annotation schema too large for Claude's structured output (the analysis failed at once). | Schema-in-prompt mode with validation; proven with a live call. |
| 15 | Research desk stored sourceless "findings" when search failed. | Never stored; asked again next run. Basic web tools, because the filtering variant hit a code-execution limit live. |
| 16 | GDELT rate-limited from Fly's shared IP; ScrapeCreators out of credits; publisher RSS terms unclear. | GDELT and RSS replaced by the research desk. ScrapeCreators paused (owner: defer). |

## 2. Research architecture

**Research desk (built, live).**
- **Pieces:** Claude web search, at $10 per 1,000 searches with failed searches free, and web fetch, at tokens only.
  - Fetch honours `robots.txt` and only reads URLs that search found or that were given.
- **Questions** are quick (Haiku, ≤ 3 searches, ≈ $0.02–0.05) or deep (Sonnet, searches plus page reads, ≈ $0.3–0.5).
- **Reuse:** findings are reused across agents while fresh.
- **Claims** are sorted into verified fact, reported claim, speculation, opinion and uncertain. Code requires two independent domains before "verified fact".
- **Kept:** sources are kept as URL, title and the API's own snippet of up to 150 characters, never page content.
- **Watched topics:** seven topics run on a cadence within $0.50/day: fashion, beauty and grooming, interiors, travel, brand campaigns, creator economy, wellness.
- **Blocked:** Instagram, TikTok, Facebook and Threads are never searched or fetched.
- **First live run:** 4 of 7 topics returned 2–15 sources each, $0.11 in total.
- **Consumers:** the style director, casting and showrunner. The same desk can serve future research-driven formats (Tech Yap and similar).

**Verdict on GDELT and specialised news APIs:** not needed. The desk is simpler, covers far more subjects, and carries citations.

**What it does not replace:** precise social performance statistics (views, engagement, growth). Those still need an authorised source. See below.

**Data audit (independent).**
- **Instagram:** its terms forbid automated collection without permission, logged in or not, and `robots.txt` says the same.
- **TikTok:** its terms forbid scraping for any purpose.
- **Status:** both are **not authorised** by the platforms. Metadata collection through Apify continues only as the owner-accepted risk; media is now off.
- **Business Discovery (official API):** authorised for metadata. Media and a persistent registry need counsel (Platform Terms 3.a.v).

**Where third-party content still lingers:**
- Apify keeps its own copy of every run's raw items, including unsanitised bios.
- `decisions` rows of research runs (scout pools).
- `llm_journal` (captions, 14 days).
- Anthropic Batch API inputs (29 days; not zero-data-retention eligible).
- `jobs` payloads and results.
- An owner event stores the handle of a person who asked to be forgotten.
- Unrecorded windows for Neon point-in-time recovery, R2 and Fly logs.

`mw research forget` misses most of these.

**Recommendation for the 1,551 stored posts (needs your confirmation; deletion is permanent):**
- Delete all stored media now: 1,691 items, 448 MB.
- Delete all caption text now; keep `caption_features`.
- Keep the factual fields: URL, ids, dates, type, metrics, snapshots.
- Keep the 8 annotations (our own words).
- Delete the Apify run datasets.
- Purge research rows from `decisions`, `llm_journal` and `jobs`.

**Just-in-time design (next build, small):**
1. One job fetches media into a temporary directory, measures and reads it with direct (non-batch) calls, writes only our annotation, and deletes the files.
2. Nothing goes to storage.
3. If a post can't be read now, its URL is queued in `research_observations`, never the media.
4. A text-and-data-mining check before any fetch: `robots.txt`, the `tdm-reservation` header and `tdmrep.json`, with our own user agent (the trend watcher currently sends a spoofed one).

What is lost: re-reading old media later (re-fetch by URL when lawful) and human frame-grid review. Until a lawful media route exists, visual intelligence comes from our own posts and metadata-only signals.

**Needs counsel before scale:**
- Instagram and TikTok terms (*Meta v. Bright Data*).
- GDPR legitimate-interests assessment and Article 14 notice for the named creator registry; probably a data protection impact assessment.
- CCPA on derived scores.
- Text-and-data-mining opt-outs.
- Business Discovery "profile building".
- Retention of the 730-day registry.

## 3. GPU and persistent weights

**Root cause of the unused volume:**
- Recent test plans pinned the plain L40S pool on purpose, after the volume filled.
- Where the volume came first, RunPod's one data centre had no L40S, so launches fell through to the plain pool.
- Pool scoring marks the volume pool "exhausted" after its early failures, so production never tries it again.
- `volume_free_gb` was never recorded because the code that writes it shipped after the last volume boot.

**Plan:**
1. Run one approved verification job on the volume (about $0.50): it prunes the stray LongCat weights and records free space.
2. Put the volume first again for talk, comfy, voice and TTS.
3. Score the volume pool and its plain twin as one pool.
4. Replace the worker's "delete sibling families' weights" pruning with a keep-list (needs a worker image rebuild).
5. Either grow the volume to about 250 GB (about +$7/month) or keep only talk, comfy, voice and TTS on it (about 85 GB).

**Second volume: not yet.** Validate the first one, then decide based on measured capacity stalls.

**Other GPU findings, open:**
- A breadth run's dollar cap is enforced only by its driver process. The autoscaler should know run caps.
- No reservation at launch, so machines can be stopped while still booting.
- Workers claim jobs longer than their remaining deadline.
- Out-of-memory errors retire healthy machines.
- Idle detection counts paused work.

## 4. Economics (what we measure and what we don't)

**Cost per accepted output is understated about 4×.** The ledger attributes only job time. Boot, idle and downloads (76% of the bill) never reach `capability_card` or `mw status`, so casting and the showrunner decide on flattering numbers. Fix: spread each machine's billed cost over its jobs.

**LLM orchestration costs about 3–4× the GPU media cost on a photo post.** The director makes two Sonnet calls per post, even for stills, with an uncached creator block (about $0.08–0.12 a post). Savings that keep quality:
- skip the edit call for stills;
- cache the creator block;
- batch direction;
- cut the Opus shadow on the showrunner;
- run report-only reviews on a 20% sample.

**Model benchmarks must be run under production conditions.** LTX passed 3 of 4 takes in its benchmark but completed 1 of 9 lines in a full post. Benchmarks should be breadth plans that run the full post path and report:
- $ per accepted asset;
- $ per accepted video second;
- $ per finished post;
- first-pass rate;
- retries and fallbacks;
- setup and idle cost.

## 5. QA: creative flexibility vs functional failure

| | What it covers |
|---|---|
| **Rejects (functional)** | Wrong person or lost face; anatomy; unsafe (confirmed by the stronger model, not one cheap-screen flag); audio not saying the script; lip sync on an on-camera take; a face where none belongs; a person in a plate; a shot that fails its narrative function (planned: a narrow `serves_function` field, not fit scores); a finished file with no audio on a spoken format, or text over the face |
| **Ranks only (creative)** | Scene details, wardrobe and lighting deviations, direction beats, minor styling, distribution risk |

**Open QA issues:**
- There is no holistic judgement of the finished piece. The final review sees about 6 keyframes, with no audio and no script, and is report-only.
- Lip sync and final sync fail open: a measurement error counts as a pass, and a silent video passes.
- One cheap-screen safety flag can drop a post.
- One failed scene drops a whole carousel.
- An audio failure is "repaired" by re-rendering video.
- The test fixture runs a review configuration that doesn't ship.

**Editor adaptation:**
- Ship a carousel as N−1 photos.
- Put a failed take under a cutaway.
- Re-finish rather than drop on caption placement.
- Walk down candidates (done).

**Benchmark library:** a set of our own excellent and unacceptable assets, so QA is calibrated against the same standard across models. Not built yet.

## 6. Autonomy and safety: your checklist

| Requirement | Status |
|---|---|
| Concurrent jobs can't overspend the same budget | **Partial.** Autoscaler ticks are serialised by a lease, but there is no per-launch reservation, and a run cap is enforced only by its driver. |
| Resumed jobs don't repeat billable work | **Mostly.** Idempotency keys and leases hold. Brain journal replays are keyed by attempt, so a hard-killed brain job pays its LLM calls again. |
| Fallbacks respect spend and scope | **Paused-batch filter: yes. Run caps: no** (driver only). |
| Completion headroom can't be unlimited | **Now off**; rebuild per job. |
| External content can't trigger actions or expose secrets | **Yes.** No agent loops or tools on third-party text; framing added. Card poisoning is closed. |
| Master switch enforced everywhere | **Yes**, except the `trends` process, which checks its own `enabled` flag (currently off). Fix: gate on the switch. |
| No restart, deploy or queue event can publish | **Verified.** Publishing needs `operations: live` and a connected account. Test, breadth, audition and research posts are blocked by two independent checks. |

**Still open: rented machines hold the full storage key** (high). They can read and overwrite any object, including assets awaiting publication. The fix is a bucket-scoped R2 token, which you create, or presigned links (built, proven on one image job). Required before going live; recommended before the next GPU run.

## 7. Definition of done for this phase

| Capability | Proven | Implemented, unproven | Untested or broken |
|---|---|---|---|
| Research (desk) | Live: sourced findings, $0.11 | Reuse across agents, topic cadence | — |
| Research (creator intelligence) | Metadata monitoring live | Annotation (schema fixed, one live call), corrected statistics, pattern cards | No research card exists yet; media analysis paused by design; just-in-time design not built |
| Intelligence → decisions | Cards and desk findings injected into showrunner, director, style, casting | Card retrieval and weighting | Not selective (same cards for everyone); desk and casting inputs not recorded; "applied" counting inflates outcomes |
| Personality | Maren's bible, dossier, voice (cast automatically, excellent) | Talent manager, style director | — |
| Production: photos | Many accepted photos; plate-based shared sets work | Ranking and walk-down fixes | Carousel drops on one failed scene |
| Production: narrated edit, story clip, motion | Passed the 11-24 breadth test | — | Not re-run since many fixes |
| Production: talking video | Single segments pass; voice excellent | Per-route hand-off, reroute | **No fully edited talking video has passed end to end (4 attempts)** |
| Consistency (face, style, world) | Face identity gate; shared set plate | — | Outfit and world consistency only report-only |
| Recovery from failures | Leases, reclaim, retries, idempotency | Walk-down, route hand-off | Failure-injection tests mostly missing (see 8) |
| Budgets | Daily caps, grants, per-machine caps | — | Run caps not enforced by the autoscaler; no launch reservation |
| True cost | Ledger complete for billing | — | Cost per accepted output understated about 4× |
| Improvement loop | Predictions and scorecard code | Card outcome feedback | First-party loop dormant until launch, by design |

## 8. Prioritised backlog (next)

**P0, before more GPU spend:**
- Worker storage scoping (needs your bucket token, or a presigned rollout).
- Weights volume verification run.
- Autoscaler-enforced run caps.
- Stale "producing" post reaper.

**P1, for a production test that tells the truth about quality and cost:**
- Validation ladder runs.
- Spread machine overhead into cost per output.
- Shipped review configuration in tests.
- Failure-injection tests: misindexed or refused judge, interrupted worker, provider 429/5xx, budget exhausted mid-run, duplicate reviewers.
- Intelligibility → re-voice.
- Fail-closed sync.
- Carousel N−1.
- A single reviewer for test batches (the Fly reviewer and the driver both judge them today).

**P2, intelligence:**
- Just-in-time observation.
- Selective retrieval and evidence recording.
- Count only explicitly chosen cards as applied.
- Fix contradictory prompts: "phone photo" vs editorial; outfits recur vs don't; casting's implied-nude rule vs policy; and more listed in the agents review.
- One owner of visual identity (the style director).

**P3, cost:**
- Director call reduction and caching.
- Batching.
- Shadow sampling.

## 9. Production validation ladder (proposal)

Each rung is judged on:
- accepted / attempted, first pass;
- $ per accepted asset and per finished post, including setup and idle;
- wall-clock time;
- retries and fallbacks;
- a holistic score ("would you follow her?").

Then run it as complete workflows.

| Rung | Format | Plan |
|---|---|---|
| 1 | Single lifestyle photos | 12 across 4 sets, ≈ $1 |
| 2 | Carousels and photo dumps | 3 posts, ≈ $1.5 |
| 3 | Simple motion and ambient clips | 3 clips, ≈ $1.5 |
| 4 | Narrated edits (photos, voice, music, captions) | 2, ≈ $1 |
| 5 | Short speaking clips | 3 lines, ≈ $1.5 |
| 6 | Hybrid talking video | 1, ≈ $3 |

Rungs 1–4 are the commercial core and pass independently of rung 6.
