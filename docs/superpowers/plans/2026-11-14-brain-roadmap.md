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

## Progress, 2026-11-20

- **M6 done.** Experiments open from tentative findings (one per creator, effect and evidence mode), sized for
  80% power to detect `mde` (0.4 log-lift) given the creator's own outcome spread: about 36 posts per arm, so
  roughly 54 days per experiment at 2 slots per arm per 3-day batch. Slow but honest: 6 per arm could not confirm
  a planted +0.5 effect. Thompson-sampled exploration over measured effects and untried formats, stagings and
  flavors; every plan records its draws. Rubric validity report (Spearman, bootstrap interval) in each analysis.
  Acceptance shown in the sandbox (planted strong effect → tentative finding → experiment → proven, SIMULATED).
- **M7 done.** Targeted repair by diagnosed cause, video-to-photo reroute after review rejections, feed QA
  (repetition, originality, naturalness floor), all rule-based. Caption QA by LLM was not needed: the idea
  rubric's naturalness score is the voice check.
- **M10 in progress.** InfiniteTalk worker runs on RunPod (REST v2) A40 secure cloud; full-precision model
  needs >48 GB host RAM, so the fp8 build (with prebuilt quanto kernels, talk v6) is the benchmark candidate.
  No lip-sync metric is licensed for commercial use without review (SyncNet weights unlicensed; LatentSync's
  SyncNet under OpenRAIL++): sync is checked by eye until the owner decides.
- **M10 benchmark, 2026-11-20 (A40, RunPod secure, talk v6).** The worker now boots, downloads, loads and renders.
  At the README's settings (fp8, 40 steps, text + audio guidance = 3 model passes per step) a ~6 s clip did not
  finish in 40 min (GPU-resident) or 50 min (CPU-offloaded): far too slow to use. Next: the 8-step distill LoRA
  (one pass per step, ~15x less compute), which needs either full precision on a host with >48 GB RAM (e.g.
  RunPod secure L40S, 48 GB VRAM, ~125 GB RAM, $1.09/h) or a patched int8 path (its repo lacks t5_int8; fall back
  to the bf16 text encoder). Also: drop `--num_persistent_param_in_dit 0` automatically on cards with room.
- **M8 done (2026-11-20).** Events (breakout / drop / QA surge; once, routed, rate-limited); portfolio numbers in
  code (conviction from real evidence only, EV continue/stop, month budget); daily strategist whose proposals
  code validates; launches walk bible → references → anchors → LoRA → profile → ready (sandbox + account
  request) with no bespoke code. Launches wait for owner approval while policy says so. First live run proposed
  a cohort of two from the best-evidenced theses. Not yet: differentiation check vs siblings.
- **M4 built (2026-11-20), idle until an account is connected.** Publisher per Meta's v27 docs, crash-safe
  (never twice; uncertain → held for owner), AI label on every post; metrics per media type at the analysis
  ages; `mw account connect`.

## Owner direction, 2026-11-21

- Nothing runs on its own until the owner says so: master switch `operations.mode: paused` (default); launches off.
- Cloud-native platform, open-weight models where they do the job, long-form lip-sync, and a Brain chat in the
  web app (the owner discusses strategy; the Brain proposes and makes changes within its authority). Plan:
  `docs/superpowers/plans/2026-11-21-next-phase.md`.

## Owner decisions, 2026-11-21 (models)

From the model review (`docs/research/2026-11-21-model-landscape.md`):

- **Licences.** FLUX 3 Image Klein's licence (free under $5M revenue) is accepted. Video: use the best LTX model available
  (LTX-2.5 today; free under $10M revenue) as long as quality and cost hold at scale. We already disclose every post and
  never strip provenance, which its licence requires. MiniMax H3 is hosted-API only (its open licence excludes the US/EU/UK).
  Face identity checks use AuraFace (Apache-2.0), never InsightFace's non-commercial models.
- **The best model per job, at the lowest cost, shown by evidence.** Every Brain role keeps its Claude model until its
  shadow report shows a cheaper candidate holds up (`llm.shadow`, `mw llm shadow-report <role>`). Candidates are on
  Together: Kimi K3.1 Pro (text and vision, same price as Sonnet 5.5, half Opus), GLM-5.3, the Flash tier
  (GLM-5.3-Flash, DeepSeek V4.1 Flash, MiniMax M3.1 Flash, ~$0.15/$0.5 per Mtok) and Nemotron 3.5 VL (vision);
  Claude Haiku 5.5 competes on the cheap end. Qwen runs on Together through a third party (owner enabled data sharing), streamed only.
- **Next benchmarks, in order, each on the owner's go:** lip-sync score in the video gate; InfiniteTalk with the 4-step
  lightx2v LoRA; distilled Wan 2.2 and LTX-2.5 vs hosted Wan Turbo / MiniMax H3 Turbo / Kling 2.6 for story clips;
  designed voices (Qwen3-TTS VoiceDesign, VoxCPM2) cloned per creator and voiced in one pass; SoulX-FlashTalk and
  LongCat-Video-Avatar 1.6; AuraFace in the image gate, then FLUX 3 Klein and HiDream-O1 vs Z-Image Turbo.

## Owner direction, 2026-11-22 (video studio, intelligence, new formats)

Priority now: **lock down video** (talking video and story clips on the ComfyUI worker) before anything else in
this section.

### Video cost and speed (target: a 30 s talking reel at about $0.08-0.12, about $0.04 with the hybrid edit)
Measured 2026-11-22: ComfyUI InfiniteTalk on a secure L40S ($1.09/h), warm, about 30-35 s of compute per output
second, so about $0.05 per 5 s clip and about $0.30 per 30 s. 1 of 3 passed the gate at 6 steps (2 at realism or
motion 5); speech WER 0; lip sync 0.61-0.70.
1. Cheaper cards: RTX 4090 / 5090 community pools for the ComfyUI worker (fp8 fits 24-32 GB), measured per accepted
   output, weights from R2 or a volume.
2. Faster inference: SageAttention 2, fp8 fast matmul, torch.compile, step caching (TeaCache/MagCache); adopted
   only where the gate holds.
3. Hybrid talking edit: about a third lip-synced on camera, the rest photos with gentle motion under the same
   voice, cut on the control plane.
4. No idle GPU time: warm machines with full queues; RunPod Serverless once the owner approves an endpoint.
5. Quality levers under test: steps 8/10, 576p, audio and motion settings, prompts against face drift.

### Intelligence: the trend watcher (built 2026-11-22, off until the owner enables it; at most $3/day)
ScrapeCreators per niche, velocity snapshots, outliers vs controls described by a cheap vision model, a daily
brief and research findings read by the showrunner. Needs SCRAPECREATORS_API_KEY. Next: cross-platform lag
(TikTok trends reaching Reels), saturation, comment themes on top posts, peer AI-creator watch list, our own
results against niche benchmarks.

### Sound effects and ambience
Foley synced to the video (MMAudio, HunyuanVideo-Foley, ThinkSound) and general sound generation (Stable Audio
Open), plus CC0 libraries for ambience beds. First step: a licence check per model (MMAudio's weights are
non-commercial, for example); then a mix stage on the control plane (voice ducking, loudness to platform norms).

### New content formats (after video is locked; each as an experiment the trend watcher and our metrics judge)
1. Translated versions: the same post dubbed and re-lip-synced (Spanish, Portuguese, Hindi...): new audiences for
   the cost of the talk render alone.
2. Serialized micro-dramas: single-character POV episodes with cliffhangers first (memory and series exist), then
   two-person scenes (InfiniteTalk multi-speaker); full ReelShort-style drama last.
3. Stylized / animated characters (3D cartoon, anime): adults only, a style LoRA, natural disclosure.
4. Ambient and long-form (cozy scenes, rain, lo-fi; YouTube long-form): one image, looping motion, sound.
5. Blender / procedural 3D and "oddly satisfying" loops: CPU-rendered, very cheap, a channel type of its own.
6. Videos that answer comments (we read comments now).
7. YouTube Shorts as a third platform from the same assets.

## Owner direction, 2026-11-22 (direction and modular control)
"Where possible we want granular control into all aspects of how we create these media pieces... Modular."
Today a scene is one sentence and the talk prompt is generic: movement, camera and voice delivery are improvised
by the models. Build a **shot sheet** per scene, written by the showrunner, stored on the post, and checked by
the gates. Every field is optional with a default from the creator or format, and each module can be swapped
or re-rendered alone:
- **Set**: location, time of day, weather, lighting, set dressing and props, background action, depth of field.
  Recurring places get a set bible plus reference images (like character anchors), so Maren's kitchen is the
  same kitchen every time.
- **Framing and camera**: shot size, angle, lens look, camera move (static, handheld, push-in, pan, follow), and
  device feel (phone propped up vs held).
- **Blocking**: timed actions (looks away at 2 s, turns to camera, lifts mug), entrances and exits, walking,
  dancing. Motion transfer from a reference that is ours or licensed (Wan-Animate-2, licence to confirm).
- **Performance**: emotion arc and energy; voice delivery per line (tone, pace, pauses, emphasis) through a
  designed voice that takes instructions (Qwen3-TTS / VoxCPM2).
- **Sound**: room tone, effects, music bed (licence check first).
- **Edit**: one take per scene, cuts only at scene changes (owner rule), transitions, captions.
Pipeline mapping: set and framing go to the scene image; blocking and camera go to the motion model or
image-to-video; performance goes to the voice and lip sync; sound and edit go to the joiner. The gate adds
"was the direction followed". A library of social moves (phone-down reveal, walk to camera, outfit transition,
point at caption), fed by the trend watcher, offers ready-made shot sheets.

## Owner direction, 2026-11-23 (capabilities, archetypes, an AI-native talent agency)
Three ideas from the owner, to design before building:

**1. Capabilities, not formats.** "We shouldn't overfit to specific formats... have all the necessary weapons in our
arsenal... exact content will depend on the creator archetype."

The studio is a toolbox the Brain composes from:
- camera devices and moves
- actions: talk, walk, pose, dance, prop work, outfit change
- edits: cut on motion, labels, timestamps, hook text, voiceover over B-roll
- sound

Researched formats (GRWM, outfit check, day in the life...) are examples the director may draw on, never code paths.
The breadth test measures capability coverage.

**2. Archetypes drive the content mix.** "Some influencers don't need a ton of instructive content or really rich
video talking (e.g. lifestyle or fashion) and then others do -- so the focus and mission and ways in which we present
and communicate that archetype then matters."

Today `bible.archetype` is free text. Proposal: a policy library of archetypes (`archetypes.<name>`). Each one sets:
- mission
- how the creator communicates: talking share, voiceover vs on camera, text-label-led vs speech-led, silent
  aesthetic share
- capability mix and weights (e.g. fashion: mostly visual, posing, outfit changes, labels, little talking; educator:
  talking head, captions, B-roll; storyteller: long single takes)
- pacing and length ranges per platform
- caption and look defaults
- what "good" means for the evaluator (rubric weights)

The research and evaluation agents learn per archetype what works (real evidence only). Each creator's profile picks
an archetype and can override any dial.

**3. An AI-native talent agency.** "Each influencer gets a talent agent to manage their career, they report to the
brain - the CEO. Then we have talent scouts, brand strategists etc."

Mapping onto what exists:

| Agency role | Agent today | What it would add |
|---|---|---|
| CEO | strategist (portfolio, budgets, launches) | owns the agency's goals, allocates spend across talent, hires/retires talent |
| Talent scout | discover + creator_lab | scouts archetypes and niches with audience pull, pitches new talent to the CEO (owner approves launches) |
| Talent manager (one per creator) | new | owns one creator's career: goals, positioning, archetype dials, weekly review of their numbers, briefs the showrunner, escalates to the CEO |
| Producer / showrunner | showrunner + director | turns the manager's brief into posts and shot sheets |
| Creative QA | evaluator + review gates | unchanged |
| Analyst | analyst | per-creator and per-archetype learning, reported to managers and the CEO |
| Community manager | comment replies | replies in the creator's voice (draft by default) |
| Brand strategist / partnerships | new | brand fit, rate cards, sponsorship pitches, disclosure (never implies real first-hand use) |
| Trend desk | trend watcher | trend briefs for managers and showrunners |

Rules that still bind every role:
- No cross-account engagement between our creators: managers never coordinate likes, comments or collabs between our
  accounts.
- Simulated evidence never counts.
- Every decision is recorded with its cost, under per-agent budgets.
- Launches and money need the owner.

Build order to consider after the breadth test:
1. archetype library plus profile dials
2. the per-creator talent manager agent (weekly loop, reads real metrics, writes briefs)
3. the brand strategist, once an account has an audience

Each is an agent under `brain/agents/` with its own budget tag.
