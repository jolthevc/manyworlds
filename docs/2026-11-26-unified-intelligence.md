# Unified content intelligence: assessment and roadmap (2026-11-26)

This answers the owner's brief "Unified Content Intelligence, Quantitative Research and Recursive Learning".

What is in place now:
- The **$100/month research ceiling** is in code: policy `research_budget` and `manyworlds/research/budget.py`.
- Every research spender checks it, paced so that one day can take at most 3x the even share of what is left. It replaces the $1/day allowance.

Everything else here is a proposal for the owner to approve. Sources for the external claims are in the linked research notes at the end.

## 1. The short version

- **The architecture is further along than the evidence.**
  - Most of the vision has a place in code already: a shared annotation vocabulary for external and own content, creator-relative outliers, pre-registered recipe tests with discovery and holdout splits, hypotheses with replication and decay, predictions and an exploration temperature.
  - Almost none of it has been demonstrated:

    | Item | Count |
    |---|---|
    | Reference posts | 1,596 |
    | Coded annotations | 70 |
    | Pattern cards | 0 |
    | Findings from real data | 0 |
    | Answered research questions | 0 |
    | Own published outcomes | 0 |

  - The next gains come from trustworthy data flowing through what exists, not from new subsystems.
- **Measured reliability has to come before scale.** Today nobody knows how accurate the Haiku annotations are. The audit just finished showed what unmeasured instruments cost: a lip-sync gate passed fake clips, and an output picker made untested capabilities look proven. The first research spend should be a gold set and agreement measurements, so every later statistic rests on attributes whose accuracy is known.
- **The binding constraint is permitted data, not budget.**
  - Not available to a commercial company: Meta Content Library, the TikTok Research API, and TikTok Creative Center automation (it has no API and the terms forbid scripts).
  - The rights-clean backbone: Instagram Business Discovery and hashtag search (needs your professional account and App Review), the YouTube Data API (free; statistics can be kept 36 months under the derived-metrics programme), the EU ad libraries (free) and the web research desk.
  - At this volume, $100 a month is ample for models.
- **The one architectural change worth making now: a single analytical spine.**
  - Every content unit, ours or external, of any format, becomes one observation with open format and platform fields.
  - Each observation gets long-form attribute values from a tiered attribute registry, and one normalised outcome.
  - Claims (findings, hypotheses, cards) are linked by typed relations.
  - This is a few migrations and views in Postgres. It needs no warehouse, graph database or vector database.
- **Cross-format transfer is a hypothesis generator for years, not a learned fact.**
  - Real transfer evidence needs outcomes in several formats, and we have outcomes in none yet.
  - Build the shared vocabulary and the relations now, so the evidence accumulates in the right shape. Don't build transfer machinery ahead of data.

## 2. Where we are, honestly

| Capability | Implemented | Demonstrated | Gap |
|---|---|---|---|
| One vocabulary for external and own content (iia_v2, 33 coded attributes) | yes (`research/annotation.py`, `review/own_annotation.py`) | 70 external reads; own reads off | Flat namespace tuned to lifestyle; no tiers; `evidence.own_annotations` has no caller |
| Creator-relative outlier selection | yes (`monitor.select`, `pulls.baselines`, `over_own_median`) | small pilots | Not age-adjusted; baseline window not stored; the trend watcher uses the niche median |
| Content-first discovery | partly (hashtag pools, trend watcher, off) | pilot runs ($15.64 scout, 30 days) | Creator-first in practice; no random sample slice; no cross-category search |
| Research allocation by value per dollar | no (fixed caps; Thompson draws only for creative exploration) | — | No research bandit; no measure of "useful finding per $" |
| Temporal video understanding | partly (`video_measure`: cuts, shot lengths, words per second; 3 keyframes to Haiku) | a calibration set (2026-11-24) | No audio analysis (VAD, music, loudness contour), no per-shot frames, no native AV model |
| Embeddings and semantic discovery | no (no pgvector) | — | Everything |
| Evolving vocabulary | partly (per-field vocabulary hashes; misses and `new:` answers are counted) | — | Nothing turns misses into new attributes |
| Code-executed analysis from LLM questions | yes, narrow (`research_questions.translate` to a pre-registered spec, `recipes.py` with BH FDR and creator holdout) | 0 questions asked | No agent calls `ask`/`propose`; `recipes.card_tests` never reaches cards; fixed test types only |
| Hypotheses with provenance, replication, decay | yes (migration 055, `hypotheses.weight`) | simulated only | Levels lack asset, audience, format and cross-format |
| Predictions vs outcomes | yes (`predictions.py`, Brier, calibration) | 3 predictions, none resolved on real data | No naive baseline; no time or creator holdout |
| Champion-challenger | yes for production roles (`llm_shadow`, `selection.py`) | routing trial 2026-11-23 | Research roles (annotate, distill, desk) are not shadowed; no gold set |
| Closed loop to own outcomes | storage only | — | Nothing is published, so there are no outcomes |
| Format generality | partly (`applies_to` jsonb, platform columns) | — | CHECK constraints allow photo/carousel/video and four platforms only; vocabulary has no narrative, episode, music or ad structure |

## 3. Challenging the brief

1. **"Every new media business makes every other one more intelligent."** This is right as a design goal, and it is cheap to honour structurally: one spine, open format fields, universal-tier attributes. It is wrong as an expectation for the next year. Transfer is something we test, one claim at a time, with own experiments in the receiving format. The system should label every cross-format idea as `origin: transfer` and give it a low prior. External correlations from another format do not count as evidence in the receiving one.
2. **Autonomous quantitative research should not mean LLM-written code.** An agent that writes arbitrary SQL or Python over our data will find spurious patterns faster than we can check them. It also widens the security surface, since research text is untrusted.
   - The safer design keeps the current principle: the model writes a typed analysis spec, and vetted code runs it.
   - The fix is to widen the method library: outliers, change points, matched comparisons, interactions, partial pooling, simple predictive models.
   - Every analysis counts toward a trial ledger that sets the multiple-testing denominator.
   - A read-only DuckDB view of the spine gives the methods speed without new infrastructure.
   - Free-form code can come later, sandboxed and read-only, once the trial ledger has shown it can keep false discoveries down.
3. **Influencer correlations are weak evidence, however sophisticated the statistics.** Platform distribution, follower base, posting time and trend timing all confound them.
   - Creator-relative, age-adjusted outcomes remove the largest bias.
   - Matched comparisons and partial pooling remove more.
   - Only our own randomised experiments make causal claims.
   - The proprietary dataset you describe in §11 is the real moat. It starts on the first day we publish, and until then the loop cannot close.
4. **Twelve Labs is not a drop-in for third-party video.**
   - It trains on uploaded data by default (Terms §4(a); opt-out in account settings).
   - It prohibits content with information about identifiable individuals except as needed for the service (§13(c)(ix)).
   - It puts all rights warranties and indemnity on us (§12, §20), and it states no retention period.
   - Recommendation: Pegasus for our own content once training is opted out. For external clips, Pegasus only if a no-training enterprise term (their Bonterms clause) and a counsel view on §13(c)(ix) are in hand.
   - Until then: deterministic measures plus Claude over keyframes and transcripts (already rights-cleared), with Qwen3-Omni (Apache-2.0) as the self-hostable challenger.
5. **Embeddings of third-party media are amber, not green.** They are derived representations of someone else's work. Proposal:
   - Embed our own media freely.
   - Embed external content only as text: the annotation and its generalised description.
   - Keep any raw-media embedding of external content no longer than the media itself (24 h after read, 14 days at most), retaining only cluster ids and centroids of many items.
   - Never compute face or voice embeddings of real people (red).
   - This needs your decision.
6. **More volume is not more intelligence until annotation accuracy is known.** At 33 attributes, an annotator that is 70% reliable produces confident nonsense at scale. Gold sets and inter-rater agreement come first. Attributes below agreement α ≈ 0.67 get reworked or dropped, not analysed.

## 4. The architecture: one spine, four kinds of knowledge

**Observation.** Any unit of content: a post, a Reel, a YouTube video, an ad, a music video, a micro-drama episode, or one of our assets.
- Fields: `source` (external|own), `platform`, `format` (from a policy registry, not a CHECK), `medium`, `vertical`, `creator` (if any), `published_at`, `observed_at`, rights class.
- Today's `reference_posts` and our posts become two sources of one view first; a table later, if the view becomes slow.

**Attribute registry** (policy). Each attribute records:
- `tier`: universal | format | context | production;
- the formats it applies to;
- the vocabulary and its version;
- how it is measured: code | model | human;
- its measured reliability (α against gold).

Examples:
- Universal: hook type, emotional arc, pace, camera distance, gaze, energy, novelty.
- Format: carousel order, episode cliffhanger, ad product reveal.
- Context: audience, platform, archetype, commercial objective.
- Production: model, route, settings, cost, acceptance.

The 33 iia_v2 fields are mapped into the tiers; nothing is lost.

**Values: long-form.** One row per observation, attribute, method and version: (observation, attribute, value, confidence, method, vocab_version). An evolving vocabulary then needs no schema change, two annotators can be compared row by row, and code measures (cut rate, loudness) sit beside model reads. The existing `coded` JSON becomes a view over it.

**Outcome.** One normalised measure per observation and horizon:
- the log of reach (or views) over the creator's trailing median, at a fixed age (24 h, 7 d);
- the baseline window stored with it, and the follower count at posting;
- saves and shares per reach where available.

External and own content get the same definition, so they can be compared directly.

**Claims.** Findings, hypotheses and pattern cards are already close to one concept. Unify them behind a view and add a typed relation table: (claim, relation, target, evidence, confidence). Relations are `applies_to` a context, `derived_from`, `replicates`, `contradicts`, `transfers_to` a format or archetype, and `informed` a decision, post or experiment. Together with pgvector, that gives most of a knowledge graph's value with SQL.

**Semantic layer.** pgvector in Neon:
- `halfvec` embeddings of our own media (Qwen3-VL-Embedding-2B or SigLIP 2, both Apache-2.0, self-hosted on the image worker for fractions of a cent).
- Text embeddings of every annotation and claim.
- Uses: retrieval ("everything relevant to this brief"), novelty and saturation (distance to the k nearest), clusters of unexplained winners. Clusters are named by Haiku and proposed as candidate attributes, which must pass the gold set before they join the registry.

**Methods library.** Deterministic and versioned. Each run records method, data snapshot, trial count and result:
- Outlier detection;
- Change-point detection and trend velocity;
- Matched comparison;
- Interaction search with BH FDR and a creator holdout;
- Empirical-Bayes shrinkage (archetype to creator), then PyMC or bambi when two nested levels are needed;
- Simple predictive models compared against naive baselines.

The model proposes, and the library runs.

## 4a. MUVID: what to take from it

The paper is Bravin, Pouly, Clegg, Hofstetter and Berger, "How Closely Should You Follow a Trend? Atypicality and Engagement on Social Media" (*Journal of Marketing*, 2026). Its 65-page web appendix was read in full. The main article body was blocked, so the headline coefficients and the field experiment's result are unverified.

**What it did.**
- A self-supervised audio-visual encoder: SimCLR-style contrastive training, video and mel-spectrogram ResNet-18s, a 512-dimension embedding per modality, about 4 s clips averaged over the video. It was trained on about 864k scraped TikTok videos.
- **Atypicality**: a dance video's mean cosine distance to all earlier videos on the same sound (1,548 trends, about 86k videos).
- Engagement: likes per view in an aggregated logit.
- Trend, creator and time fixed effects; propensity matching; a Gaussian copula; quality controls; Prolific choice experiments.

**What it found.**
- More atypical videos get more likes per view, monotonically. There is no clear inverted-U: moderate and strong atypicality did not differ.
- The effect is stronger when the audience has seen more of the trend.
- The effect is small: about 0.02–0.04 logit per SD.
- An earlier conference version (EMAC 2023) argued the opposite direction. That is a sign of specification sensitivity.

**Limitations.**
- Dance and TikTok only, scraped through 2022.
- Survivorship through the recommender.
- Likes per view, not reach or growth.
- Weak quality controls, so atypical may stand in for skill.
- Visual-only atypicality, with the video's structure averaged away.
- videoanalyzer.app is a free research demo with no API or terms page.

**What we take.**
1. **The atypicality primitive:** the distance of an item to the prior entries of its cohort, where the cohort is a sound, hashtag, format or pattern card. It is a universal-tier attribute in the registry, and we compute it with off-the-shelf encoders rather than training our own.
2. **Saturation as a moderator:** deviate more on saturated trends, conform on fresh ones. This is a cheap, testable hypothesis for our own experiments.
3. **The analysis template:** cohort, creator and time fixed effects plus matching, with embeddings validated by downstream probes.

What we don't take:
- their effect sizes;
- their outcome;
- the assumption that audiences treat AI-generated content the same way;
- any use of their scraped corpus.

Related work worth a place in the desk's reading list:
- Askin and Mauskapf (2017), optimal differentiation in songs (an inverted-U);
- Adam and Yang (2026), where atypicality's effect depends on brand heritage;
- Yang, Zhang and Zhang (2025), TikTok video features linked to sales;
- Schwenzow et al. (2021), open video feature extraction.

## 5. Discovery and research allocation

- **Two stages, like YouTube's recommender.**
  - First, cheap code ranks candidates: creator-relative outliers (age-adjusted), velocity, novelty against the semantic layer, cross-category similarity, under-covered cells of the attribute grid, and contradictions of current claims.
  - Second, only the top candidates get paid annotation or deep reading.
  - Every candidate records why it was picked, so selection effects can be modelled later.
- **A research bandit over arms.**
  - Arms are (source × question type): Business Discovery outliers, YouTube most-popular, ad-library new creatives, desk topics, comparable underperformers, random sample.
  - Reward: useful observations, then replicated findings, per dollar.
  - Thompson sampling with costs; it learns which sources pay.
- **15% of research budget stays uniformly random.** That keeps estimates unbiased and leaves room for surprises outside the current worldview. The bandit never chooses the evaluation set.
- **Successive rounds.** A surprising observation becomes a research question. The question becomes a matched comparison sample. The sample becomes a claim with an interval. The claim becomes an experiment proposal. This is the existing `research_questions` flow, once agents actually file questions; today nothing calls `ask`.

## 6. Recursive improvement

What already exists and only needs extending:
- `llm_shadow` and `selection.py` champion-challenger;
- the vision bench;
- rubric versions;
- hypothesis weights.

What is added:
- **Gold sets per research role.**
  - 150–300 items each for annotation, distillation and desk claim classification.
  - Labelled once by Opus plus your spot checks.
  - Measured with Krippendorff's α, model against gold and model against itself.
- **Research roles join the shadow and selection loop.** A challenger prompt or model is promoted only when it beats the champion on held-out gold at equal or lower cost, with your approval, as today. GEPA (MIT) is the efficient optimiser for small gold sets. Optimise against agreement, never against engagement, which is too noisy.
- **Method and source scorecards.**
  - Per source: cost per usable observation, novelty rate, the share of its picks that became replicated findings.
  - Per method: replication rate out of sample.
  - Per prediction: the gain over naive baselines (persistence, creator median, format base rate).
  - These feed the bandit.
- **Every change is a versioned override.** Prompts, vocabularies, method parameters and source weights live in policy or versioned rows, change only through evidence-gated proposals within `brain.authority`, and revert with `mw policy revert`. Nothing self-modifies production policy.

## 7. Roadmap

Ordered by intelligence per dollar and by what unblocks what. No GPU production spend; nothing publishes.

**Phase 0: wire what exists (now, code only)**
1. Done: the $100/month ceiling with daily pacing.
2. Agents file research questions: the showrunner, style director, casting and the CEO call `research_questions.propose` for their open uncertainties.
3. `recipes.card_tests` results reach `pattern_cards`.
4. `evidence.own_annotations` gets its consumer: an own-vs-external comparison.
5. Add a naive baseline to the predictions scorecard.
6. Open the format and platform CHECKs (migration) and add an attribute-tier map for iia_v2.
7. Put the research roles in the shadow list.

**Phase 1: trustworthy instruments (weeks 1–3, ~$15)**
1. Build gold sets and compute α per attribute; drop or rework the weak ones.
2. Store age-adjusted creator-relative outcomes with their baseline.
3. Add deterministic AV measures to `video_measure`: Silero VAD, librosa tempo and onsets, loudness contour, Whisper word timing, PySceneDetect; all permissive and free on CPU.
4. Long-form attribute values.
5. The random sample slice.

**Phase 2: discovery engine (weeks 3–6, ~$30–50/month in use)**
1. Content-first outlier discovery on the permitted sources: Business Discovery once the account exists, the YouTube Data API, EU ad libraries, the desk, and ScrapeCreators metrics for TikTok (amber: metrics only).
2. The research bandit.
3. The trial ledger.
4. Attribute-frequency time series with change points (emerging, saturating, declining).

**Phase 3: semantic layer (weeks 6–10, ~$5/month)**
1. pgvector.
2. Embeddings of own media and annotation text.
3. Retrieval for agents ("what do we know relevant to this brief").
4. Novelty and saturation of our own feeds.
5. Cluster-named candidate attributes through the gold set.

**Phase 4: learning from outcomes (when publishing starts)**
1. Empirical-Bayes pooling, creator to archetype to portfolio.
2. Pre-publication predictions with time and creator holdouts against baselines.
3. Within-creator switchback experiments.
4. The research-ablation validation (design B in `docs/2026-11-26-validation-designs.md`).

**Later, when justified**
- Pegasus on own content (after opt-out), and on external content only with no-training terms and counsel.
- Hierarchical Bayesian models beyond two levels.
- Free-form sandboxed analysis code.
- Formats beyond the lifestyle set as new businesses launch.

## 8. Budget at $100/month (a ceiling, not a target)

| Use | Estimate |
|---|---|
| Data (ScrapeCreators/Apify metrics, SerpApi Trends while the Google alpha is pending, Social Blade history) | $15–30 |
| Broad annotation (Haiku, ~3,000 reads at ~$0.005) | $10–20 |
| Desk web research (quick and deep) | $15–25 |
| Strategic synthesis and hard interpretation (Sonnet/Opus) | $10–15 |
| Research-method evaluation (gold labels, challengers) | $5–10 (front-loaded) |
| AV analysis sample (own content, later) | $0–10 |
| Unspent headroom | the rest |

Free: Business Discovery, the YouTube Data API, the ad libraries, deterministic AV tools and own-media embeddings.

## 9. How we will know it works

- Cost per usable observation, falling.
- Attribute α, rising, with no analysis on attributes below the floor.
- The share of findings that replicate on the holdout and on later data.
- Predictions beating naive baselines out of time, and by how much.
- Downstream use:
  - which agent decisions cited which claims (the `informed` relation);
  - the share of findings that became experiments;
  - the outcomes of research-informed posts against the exploration control.
- The source and method scorecards, and the bandit's allocation shifting toward what pays.

## 10. What I need from you

1. **The Instagram professional account and App Review** for Business Discovery and hashtag search. This is the single biggest unlock.
2. **A Google Cloud project for the YouTube Data API.** Also: apply for the 36-month derived-metrics use case and for the Google Trends API alpha.
3. **The rule for embeddings of third-party media.** The proposal in §3.5 is text embeddings only, and raw-media embeddings no longer than the media.
4. **Twelve Labs:** whether to open an account (training opt-out first) for own-content tests only. Separately, whether to seek no-training terms and counsel for external clips.
5. **Approval of Phase 0–1.** It is code only, plus about $15 of gold-set work inside the ceiling.

## Sources

- Codebase and data volumes: audit on 2026-11-26.
- Data-source terms and prices: Meta Business Discovery, hashtag search and Content Library pages; TikTok Research API FAQ and Commercial Content API; YouTube quota, derived-metrics and developer policies; Google Trends API alpha; SerpApi, Apify, ScrapeCreators, Modash, Social Blade, Chartmetric, Spotify and Reddit pages. All checked late November 2026; unverified items are marked in the research notes.
- Twelve Labs: Terms of Use (2026-08-25) §4, §12, §13, §20; Privacy Policy; DPA; Pegasus 1.6 and Marengo 3.5 documentation. Prices are from a third-party capture, to be confirmed against a real usage report.
- Methods:
  - Covington et al. 2016 and Zhao et al. 2019 (YouTube);
  - PinSage, unified visual embeddings and ItemSage (Pinterest);
  - core-set and Thompson sampling;
  - Efron–Morris shrinkage;
  - Bailey and López de Prado on backtest overfitting;
  - Harvey, Liu and Zhu;
  - GEPA, DSPy and MIPROv2;
  - Krippendorff's α;
  - pgvector, BERTopic and TnT-LLM;
  - Johari et al. on always-valid inference;
  - switchback designs;
  - Kantar/WARC and System1/IPA creative studies (treated as light priors).
