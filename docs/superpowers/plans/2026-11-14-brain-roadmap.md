# The Brain: build roadmap

**Spec:** `docs/superpowers/specs/2026-11-14-brain-design.md` (approved 2026-11-14, with the
owner's principles 1–19).

The Brain is several subsystems, so it is built as ten milestones. Each one ships working, tested
software on its own and gets its own task-level plan (`2026-11-14-brain-m1-foundations.md` is
first). Later plans are written when their milestone starts, so they build on what the earlier ones
taught us. This roadmap fixes the order, the dependencies and what "done" means for each.

## Order and dependencies

```
M1 Foundations ──► M2 Creative evaluation ──► M3 Memory + showrunner ──► M5 Simulated loop ──► M6 Analyst ──► M8 Strategist + events ──► M9 Portfolio layer
       │                                               │                        ▲                 │
       └──────────────► M4 Instagram (parallel) ───────┴────────────────────────┘                 │
                                         M7 QA & repair (after M3) ─────────────────────────────────┘
                                         M10 Video (after M2; parallel)
```

- **Phase A, the Maren loop:** M1–M7. Proven when the acceptance trace for M6 exists on real
  Instagram data (or, before enough real data, on the sandbox with every row marked SIMULATED).
- **Phase B, autonomous multi-creator operation:** M8. Proven when the strategist launches a
  second creator and then a concurrent cohort with no bespoke code.
- **Phase C, the portfolio layer:** M9. Proven when the Brain allocates the master budget across
  markets, clusters and creators within its authority, with traces.
- M4 (Instagram) runs in parallel from the end of M1 and is blocked only on the owner creating
  Maren's professional account and Meta app.

## Milestones

### M1 Foundations
Deliverables: migration `012_brain.sql` (every table in spec 4.10 plus `policy_overrides`, an
`agent` column on `cost_entries`, `creators.publish_mode/market_id/cluster_id`); the agent contract,
registry and brain runtime (`brain.*` jobs on pool `local`); decision recording; the evidence guard
(real vs simulated) as the only way to read findings and metrics; runtime policy overrides with
authority and approvals; per-agent LLM budgets; `mw brain` commands; `brain-worker` compose service.
Acceptance: a no-op agent enqueued as `brain.echo` runs through the runtime, records a decision and
its LLM cost under its agent name; a simulated finding is invisible to a real-mode read; an
override outside authority becomes a pending approval. All tests green.

### M2 Creative evaluation
Deliverables: rubric files `policy/rubrics/{idea,execution,feed}-v1.yaml` (dimensions, definitions,
anchors at 1/3/5/7/9); rubric loader and versioning (`rubric_versions`, file hash); evaluator agent
(samples, median, spread, rationale; independent prompt); strong-model double-scoring on a policy
sample; reference set and `mw brain rubric-drift`.
Acceptance: scoring the same idea `samples` times stores median and spread per dimension; changing
a rubric file creates a new version and re-scores the reference set; agreement with the strong model
is reported. Calibration check on 20 real Maren ideas: median spread ≤ policy threshold.

### M3 Memory and showrunner
Deliverables: persona & memory keeper (seed `creator_memory` from profile, canon and published
posts; update after publication; contradiction check); showrunner agent producing `ideas` for a
batch (free creative latitude, Stories format, exploration share, experiment arms); "no idea is
scheduled without an idea evaluation"; ideas → posts bridge into existing `production.py`; planner
retired for new content (old posts drain).
Acceptance: a batch for Maren of policy size where every idea has an evaluation written before its
post exists; ideas reference memory facts; the batch mixes formats per allocation; feed reads
naturally (manual review of one batch).

### M4 Instagram publishing and metrics (parallel)
Deliverables: `accounts` handling and token refresh; `publish/instagram.py` (photo, carousel, Reel,
Story; `is_ai_generated=true`; container → status → publish; idempotent); publish scheduler
(autonomy setting, kill switch, runtime publishing limit); metrics collector (observation ages,
Story polling before 24 h expiry, daily follower counts); daily digest (posts, spend, new video
formats/models).
Acceptance: against Meta's test account or Maren's real account, one post of each type publishes
with the AI label read back as true; metrics are captured at each observation age; a Story's
metrics are captured before expiry; kill switch stops the next publish.
External dependency: owner creates the professional account and Meta app and adds tokens to `.env`.

### M5 Simulated loop
Deliverables: `publish_mode` per creator; simulated publisher; simulated audience panel writing only
`simulated_metrics`; orchestrator creator cycle running end to end (memory → ideas → scoring →
production → QA → publish → metrics) on a schedule and on demand.
Acceptance: `mw brain cycle maren --sandbox` completes a full cycle with every simulated row and
report labelled SIMULATED; the strategist path refuses simulated findings (test).

### M6 Analyst and learning
Deliverables: baselines and log-lift normalisation; per-attribute and per-dimension lift with
bootstrap intervals (within creator and pooled); findings with scope and grade
(proven/tentative/speculative); experiments (arms, randomised assignment, minimum sample, stopping
rule); Thompson-sampled exploration consumed by the showrunner; rubric predictive-validity report;
"what changed and why" trace from findings to the next batch.
Acceptance: on seeded data with a planted effect, the analyst finds it with the right sign and an
interval excluding zero, and does not "find" effects in null data beyond the policy false-positive
rate; the next showrunner batch shifts toward the planted direction and the trace names the finding.

### M7 QA and repair
Deliverables: diagnosis + targeted repair actions with per-item budgets; caption voice/naturalness,
originality and policy checks; feed QA (last 9–12 posts + Stories); format reroute.
Acceptance: seeded failures (wrong scene, off-voice caption, repeated outfit) each get the intended
diagnosis and action; repair budgets stop loops; feed QA blocks a repetitive grid.

### M8 Strategist, conviction and events (Phase B)
Deliverables: stages (policy list) and creator conviction distributions; expected-value
continue/adjust/retire against alternatives; launches as cohorts through the creator lab
(differentiation check vs siblings); account requests to the owner and sandbox-until-account; event
detectors and routing with rate limits; agent commissioning and decline.
Acceptance: the strategist proposes and launches a second, distinct creator with no bespoke code;
then launches a cohort of N concurrently (N from policy for the test) that all run cycles in
parallel; a planted breakout event triggers a follow-up within one tick; an event storm is
rate-limited.

### M9 Portfolio layer (Phase C)
Deliverables: market dossiers and market conviction (scouts' market research, monetization recorded
separately); clusters with shared intelligence; master budget periods, hierarchical allocation,
pacing, reserve, exploration share; authority and approvals; spend enforcement reads allocations.
Acceptance: with a $750/month budget and three markets of differing evidence, allocation follows
conviction while holding the exploration share and reserve; an allocation beyond authority waits
for approval; spend never exceeds the allocation (test with the autoscaler and LLM budgets).

### M10 Video capability (parallel after M2)
Deliverables: talking-video capability; benchmark harness comparing candidate models through the
video gate on cost per accepted clip; routing by creator tier; digest entries for new models/formats.
Acceptance: a model enters production only after passing the gate on a benchmark set; talking video
for one creator passes the gate.

## Cross-cutting rules for every milestone

- Every number in policy; every LLM call and GPU second in the ledger with its agent; every decision
  recorded with policy and override versions.
- Simulated data never reaches a real-mode read (enforced in code, tested in M1, re-tested in M5/M8).
- Tests stub Claude, Apify and Instagram; no spend in tests.
- Live services rebuilt and restarted after each milestone; worker images only via
  `scripts/build-worker.sh`.

## Owner decisions, 2026-11-19 (toolbox and format direction)

- **Voice means lip-sync.** Anything with a creator speaking is lip-synced talking video; voice-over
  narration is off (`production.formats` no longer includes `narrated_edit`). M10 (talking video) moves up:
  benchmark lip-sync models on Vast through the video gate before any goes live.
- **Stagings.** A format is a production pipeline plus a staging: a template of setting, framing,
  wardrobe/props and script conventions (post-game press conference at the podium, locker room or court;
  field reporter; studio news desk; weather map; commentary booth; nature-documentary narrator; late-night
  monologue; reality-TV confessional; tour guide; cooking-show host; walk-and-talk; ASMR). A policy library
  the showrunner draws from; every post records its staging so the analyst learns which work per creator.
- **News.** Real everyday news is allowed; hot-button and sensitive topics are excluded (policy
  `scope.excluded_topics`). Fictional stories are fine. No real broadcasters, leagues or teams.
- **Fictional organisations persist.** A creator who reports for a fictional outlet always does so (memory
  fact with key `employer`); switching outlets is a storyline beat (`canon_key: employer`).
- **Styles.** Realistic creators first and mostly; animation, cartoon and other non-realistic styles are in
  the toolbox for the strategist to test later.
- **Animal creators.** Realistic animals (e.g. dogs) doing human-trend or comic scenes are a candidate
  market and creator kind (M8/M9 strategist + creator lab): identity consistency via a LoRA per animal,
  lip-sync for talking animals is a separate capability to benchmark, never depict harm to animals.
- **Music personas (later).** Disclosed AI music artists (a public example: River Cain, an AI country act) once
  a commercially licensed music-generation API is chosen; distribution must follow each platform's AI rules.
- **More flavors for the existing pipelines:** photo dump, expectation vs reality, tier list, recipe card,
  travel diary, outfit of the week, text-on-photo relatable memes. Reels need our own audio (the publishing
  API cannot attach licensed trending tracks).
