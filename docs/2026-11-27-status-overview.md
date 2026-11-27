# Manyworlds: status overview (2026-11-27, ~06:00 UTC)

**Context.** Manyworlds is an autonomous engine for openly AI-generated lifestyle creators. A "Brain" (Claude agents) plans content, GPU workers render it, review gates judge it, and an intelligence engine researches what works. It is still being built:
- operations are paused, nothing is published, and no creator launches are scheduled;
- the control plane runs on Fly.io and GPU machines on rented providers;
- every cost goes to a ledger.

Sources:
- the full assessment and roadmap: `docs/2026-11-26-unified-intelligence.md`;
- the experiment write-ups: `docs/research/2026-11-26-maren-performance-experiments.md`.

## What got done

### 1. Production verification (the V2V investigation)
- **The output picker bug.**
  - Since 2026-11-22, the ComfyUI output picker returned a graph's *input* as its "render". Every walk-and-talk (video-to-video lip sync) take and every Wan-Animate dance clip was really its own source clip.
  - Fixed in worker v33.
  - Invalidated as a result: the dance "pass", the V2V results, and the V2V and dance cost figures (they were understated).
- **End-to-end verification across engines.**
  - Every ComfyUI run now waits for the prompt to complete.
  - It refuses an output that is identical to an input.
  - It records provenance: prompt id, graph hash, sampler and loader settings, and which nodes were served from the cache.
  - The worker refuses any media output identical to a downloaded input.
  - The image smoke test now runs a real load-and-save graph.
- **The lip-sync gate.**
  - It was passing fake clips: correlation 0.87 against a chance level of 0.87.
  - A null-margin measure is now recorded on every check. It stays report-only until it is calibrated.
- **Walk-and-talk lip sync now works.**
  - The lead-in and end hold were the real fix: sync went from 0.16/−0.08 to 0.52/0.63 at the same settings.
  - Five of six takes passed every gate.
  - The default start step is now 1.
  - The experiment cost $1.89 of a $3 cap.
- **Also fixed:**
  - a flaky test;
  - a policy-path gap in scene choice;
  - a voiced-span misalignment introduced by the lead-in.

### 2. Intelligence foundation (Phases 0–1, approved by the owner)
- **Research budget.** One $100/month ceiling over all research spend, with daily pacing, checked by every research spender.
- **The analytical spine** (database migration 056):
  - one shape for any content, source, format or level (creator, series, post, segment, shot, beat);
  - long-form attribute values;
  - outcomes: raw counts at each age horizon, plus normalised against the creator's own median, with the baseline kept;
  - typed relations between claims;
  - a trial ledger counting every analysis toward multiple-testing correction.
- **Registries.**
  - An attribute registry with tiers (universal, format, context, production) and classes (objective, subjective, inferred).
  - An open format and platform registry in policy, instead of database constraints.
- **Gold sets.**
  - Krippendorff's α by attribute and class.
  - Reference reads (Opus, Sonnet), a repeat of the production annotator, and owner spot checks.
  - Attributes that prove unreliable are dropped from analysis automatically.
  - Gold sets use third-party posts only, by owner decision: not our own posts while their quality is unknown.
- **Flexible research questions.**
  - Any outcome can be chosen (metric and horizon).
  - Exploratory methods: prevalence, outliers, trends with change points, groups, format fit and evolution.
  - Pre-registered tests run with a discovery and holdout split by creator, with false-discovery-rate control.
- **Loop wiring.**
  - Holdout-confirmed results become archetype-level hypotheses, which creators can test as experiments.
  - Code files knowledge gaps as questions.
  - The predictions scorecard compares against a base-rate baseline.
  - Research roles are shadow-tested against cheaper models.
  - A 15% random sampling slice keeps estimates representative.
  - Video measurement gained cheap loudness and silence measures.
- **Pegasus adapter.** A Twelve Labs adapter for native audiovisual reads of our own videos only. It is off until the owner opens an account and opts out of training.

### 3. Creator strategy
- **Audience propositions.** The owner's ten (aspirational icon, muse, best friend, entertainer, authority, tastemaker, protagonist, explorer, maverick, comfort figure) are kept separate from verticals and content playbooks.
  - Each one has its attention, follow, return and affinity jobs, observable signals and pitfalls.
  - A creator is a weighted mix of them.
  - Unknown reasons are kept as candidates for new propositions.
- **Dossier and casting.** Both now record proposition mixes, verticals, series jobs and arcs, and idea engines.
- **Creator-content fit.**
  - Every format is tiered per creator: core, complementary, experimental or deferred.
  - This replaced a portfolio-wide video quota.
  - Deferred formats leave the creator's plan, and her video count follows her own mix.
  - Casting caps feasibility at the weakest core format's reliability, priced from the capability card.

## Where things stand
- CI is green and everything is deployed. Operations are paused, nothing is published, and no GPU machine is running.
- **Research spend:** $21.20 of $100 this month, mostly an earlier scout run. Today's cycle has cost about $0.08 so far.
- **The spine is live:**
  - 1,596 third-party posts from 54 reference creators;
  - 25,846 outcomes, of which 4,946 are normalised.
- **Annotation of third-party posts is running.** 124 of about 1,556 are coded so far, through the half-price batch API: about 16 hours in total, roughly $1.60.

## What's left
1. Finish annotation and re-sync.
2. Run the third-party gold set (up to 60 posts, capped at $5) and read the agreement report.
3. Code the 54 creators' propositions: text only, about $0.05.
4. File the gap questions, answer them, and write up the first real research cycle honestly, including "insufficient data" results.
5. **Owner actions:**
   - an Instagram professional account and App Review (Business Discovery, hashtag search);
   - a Google Cloud project for the YouTube Data API, plus applications for YouTube derived metrics and the Google Trends API alpha;
   - optionally, a Twelve Labs account with the training opt-out, for Pegasus on our own videos;
   - optional spot checks of gold items;
   - still open from before: R2 credential rotation, infrastructure invoice figures, `NTFY_TOPIC`, `MW_HASH_SALT`.
6. **Not yet approved:**
   - Phase 2 (discovery engine and research bandit);
   - Phase 3 (pgvector embeddings);
   - lip-sync null-margin calibration;
   - re-testing dance properly.

## Red flags
1. **Silent failures are a pattern.** Four were found in one day:
   - the output picker;
   - the lip-sync gate passing chance;
   - own-content reads silently reading nothing (an image-size cap);
   - one off-vocabulary label killing an annotation job.

   All of them passed about 2,000 tests, because the tests stub exactly the boundaries that failed. Each is fixed, but more are likely. The next cheap safeguard is for every pipeline stage to report attempted against succeeded counts, and to alert on zeros.
2. **Most of today's research code is unproven against real models.** It passes CI, but only the spine sync has run cleanly live. The gold run and annotation each hit a bug on first contact.
3. **The casting schema is at the API limit.** A casting concept now has exactly 20 fields, the measured maximum for one structured-output object (21 failed before). Any further field will break casting.
4. **The evidence base is thin and skewed.**
   - 51 of the 54 reference creators are on TikTok and only 3 on Instagram, against an Instagram-first strategy.
   - The data comes from a scraping provider (Apify), which the data policy rates amber.
   - 54 creators is too few for proposition × vertical findings.
   - The owner's Instagram setup is the biggest unlock.
5. **No real outcomes yet.** Until something is published, the learning loop can't close, predictions can't resolve, and nothing about our own content is learned.
6. **Batch latency slows research.** The 16-hour annotation saves about $0.80. Fine overnight, but slow when a result is needed.
7. **Twelve Labs terms.** Training on uploads is on by default, and the terms bar content about identifiable individuals. So Pegasus is for our own content only, unless no-training enterprise terms are agreed.
8. **Process.** Two agent sessions shared one code checkout for a while. It worked with care, and the work is now centralised in one session.

## Open decision
Should the remaining annotation run overnight at the half price, or switch to direct calls for about $0.70 more, so the first research cycle finishes within the hour?
