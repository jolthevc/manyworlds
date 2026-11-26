# The intelligence engine: assessment against the owner's vision (2026-11-25)

The owner's view: research, multimodal intelligence, analysis and continuous learning may be the most defensible thing Manyworlds builds. Generation models will commoditise; knowing what works, and why, will not.

This document covers:
- what exists;
- what changed today;
- where the real gaps are;
- where I disagree.

The last section is a status ledger, with four levels for every capability:
1. implemented;
2. passes isolated tests;
3. verified on the production path;
4. proven end to end.

## 1. Strategy: commercially informed casting, growth-led execution

Changed today:
- **CEO prompt:** launched creators are run for reach, follower growth, shares and saves, and the CEO must "never trade growth for early monetisation".
- **Strategy thesis:** commercial intelligence decides who we develop; monetisation never constrains growth before the audience exists.
- **Economics principles:** the owner's six selection criteria are written in. They are surface area, desirability and viral potential, adjacencies, creative longevity, production economics, and monetisation paths.
- **Casting rubric:**
  - surface area now asks about growth potential, shareability and years of things to post;
  - its weight rises from 0.20 to 0.25;
  - the commercial weight falls from 0.20 to 0.15.
- **Brand bible:** the commercial thesis and signature associations stay.

## 2. Three layers, as they stand

| Layer | Built | Feeds | Gap |
|---|---|---|---|
| **A. General research** (Claude web search and fetch) | `research/desk.py`: quick and deep questions, findings reused while fresh, claims classified (a verified fact needs two independent domains), watched topics, $0.60/day | CEO, casting, style director, showrunner (`desk.brief`) | Topics are lifestyle and commerce only. Tech, business and AI "yap" topics are a policy list away, with no code needed. |
| **B. Structured social intelligence** | `research/providers.py` router (Business Discovery, Apify, ScrapeCreators paused); `monitor.py` (tiers, cadences, dedupe, deterministic selection); green data persists | Pattern statistics; growth benchmarks (`account_insights`); the portfolio map | Business Discovery is unverified until our professional account exists. Today's preparation: policy field lists for Graph v27, withheld fields recorded as missing rather than zero, and `mw research graph-check <handle>`, which settles every open question with one call. |
| **C. Multimodal creative intelligence** | Annotation is blind to performance; code attaches metrics; within-creator comparison; Bonferroni-corrected q; floors of 3 creators and 10 posts per side; `video_measure` (cuts, shot lengths, pace) | Pattern cards to showrunner, director, finish, casting, style director | **The schema could not answer the owner's questions.** Only 10 attributes were coded; the rest were free text that statistics cannot count. Schema v2 is being built now (§3). The visual input is also blocked until a permitted route exists (§5). |

**How the layers connect.**
- A pattern card's weight combines three things:
  - evidence basis (judgment, observed, correlation, validated);
  - production acceptance;
  - real outcomes.
- Agents receive the top cards for their archetype and shot vocabulary.
- Ideas record which cards they were offered and which they applied.
- Feedback recomputes each card from results.

This is the "agents get better because of the research" loop. It exists, but has no real outcome data yet.

## 3. Annotation: how sophisticated it really was

**v1 coded:** situation, shot type, device, gaze, video format, movement, hook, camera movement, structure, relationship.

**v1 had nothing countable for:**
- garments, fit or palette;
- styling category (swimwear, eveningwear…);
- hair, makeup or nails;
- accessories, jewelry, watches or bags;
- pose or body language as categories;
- composition or subject scale;
- photo camera angle or lighting kind;
- candid vs staged;
- environment as a category, or travel vs everyday;
- lifestyle narrative or emotional tone;
- carousel slide order;
- product integration level;
- face in the first second, setting changes, movement complexity, production complexity.

The wardrobe field said "silhouette, palette, formality" in prose, which a model writes differently every time.

**v2 (in progress):**
- Every attribute above is coded against controlled vocabularies kept in policy. An out-of-vocabulary answer becomes "unknown" rather than a failure.
- The statistics test every coded label. The number of tests counted for the Bonferroni correction stays honest, so adding 30 attributes raises the bar rather than producing false findings.
- Carousels get an ordered list of slide roles, so ordering patterns can be compared.

**The vision benchmark.**
- Haiku 5.5, Sonnet 5.5 and Opus 5.5 code the same assets. Our own assets come first, so rights are clean.
- Each model is scored on per-field agreement with the reference model, schema validity, cost and latency.
- "Sufficient" is set per field family from the first run. Coarse fields (environment, styling category, candid vs staged) should hold up on Haiku. Fine fields (jewelry, nails, composition) may need Sonnet or a two-stage read.
- Gemini Flash-Lite sits behind the same interface. It needs an API key, which only you can create.

## 4. Video intelligence: the economical design

A video read needs three layers. Only the third costs real money, and only a minority of posts get it.

1. **Deterministic (free, our CPU):** `video_measure` gives cut times, shot lengths, pace and frame samples. Add audio classification (speech, music, ambient) and silence; ffmpeg can do this, plus a small VAD.
2. **Selected frames to a cheap vision model (~$0.002–0.004 a post):** frames at 0, 1, 3 and 5 s, plus one per detected shot (capped at 8), read with the v2 video fields:
   - the face in the first second;
   - the opening hook;
   - environments, styling and movement;
   - candid vs cinematic vs selfie;
   - product integration;
   - production complexity.
3. **Native video understanding on a sample (Gemini Flash):** for posts whose frame reads disagree, or that sit in the top or bottom tail of the creator's distribution. It is better on temporal structure (what happens across the first three seconds, the narrative arc). It costs more per post, needs a key and a processor agreement under the data policy, and should be benchmarked against frames-plus-measure on our own videos first.

**Transcription:** only for spoken formats. Whisper on our own worker, or the cheapest hosted ASR. Transcripts of third-party videos are amber data under the data policy: features are kept, text discarded.

**Example:** "8–12 s lifestyle Reels with simple movement outperform talking videos in category X." This is answerable once layer 1 and layer 2's movement and production-complexity fields exist on enough posts. It needs no layer 3.

## 5. Data rights for visual observation

**Business Discovery returns `media_url` for most posts.** Meta's terms allow analysing Platform Data. They forbid:
- selling it;
- surveillance;
- building or augmenting user profiles without consent.

They also require deletion once the data is no longer needed. Sending media to Anthropic or Google requires a written service-provider agreement; our API terms are that agreement in substance, and counsel should confirm. Developer Policy 6.2 forbids importing or backing up content.

**The defensible pattern** is the one the data policy already encodes:
- fetch `media_url` just in time at annotation;
- analyse it;
- discard it;
- keep coded attributes, metrics and aggregated pattern cards.

**The real ambiguity** is whether a per-creator research record counts as a "profile". Aggregated cards are clearly safer than per-creator dossiers. I recommend counsel's read on two things before visual analysis is turned on:
- clause 3.a.v (profiles);
- transient analysis by an LLM processor.

Until then the media pause stays. When the account exists, metadata and metrics flow.

## 6. Cost accounting (fixed today)

**Before:** cost per output counted only the seconds a job ran.

**Now:**
- The capability card is all-in. Every machine's whole bill is spread over the jobs it ran: boot, model downloads, idle, retries and failed attempts. Machines that ran nothing are spread over their family's jobs, so no billed dollar disappears.
- Orchestration calls booked to no post (showrunner, evaluator, reviser, director) are spread over the window's posts.
- The card shows the old nominal figure beside the all-in one, a cost breakdown, and the share of the window that was development.
- `mw economics` splits the bill into:
  - production;
  - development (test and bench batches);
  - orchestration;
  - research;
  - agency;
  - idle machines.

  Development spend stays visible without inflating the economics of recurring production.

**Still missing from all-in:**
- fixed infrastructure (Fly, Neon, R2), until invoice figures are in policy;
- serverless execution, which is not booked per job;
- the finisher's Fly machine, if it isn't registered as a GPU group.

## 7. Where I disagree with your assumptions

1. **"The research engine is the moat."**
   - Mostly right, but the defensible part is first-party data, not third-party research. Every competitor can buy Apify and call Haiku.
   - What they can't buy:
     - our own posts' outcomes, joined to the exact brief, shot sheet, pattern cards and cost that produced them;
     - experiments that vary one attribute at a time.
   - External research tells us where to look. Only our own A/B evidence tells us what causes growth.
   - Invest in that loop: it is the only one where correlation can become causation.
2. **Comparing high and low performers finds correlates, not causes.** Even within one creator, a beach post may outperform because it was posted on holiday, when engagement rises anyway, not because of the beach.
   - The blind annotation and the corrections stop us fooling ourselves statistically.
   - They don't remove confounders: posting time, trend audio, the algorithm's mood, a collaboration.
   - Treat external findings as hypotheses, and promote a card to "validated" only on our own experiments. The card bases already encode this; it must stay strict.
3. **Public metrics are thin.**
   - Business Discovery gives likes (often hidden), comments, Reel views and reposts.
   - It does not give saves, shares, reach or follows gained, and those are what drive growth.
   - So external analysis can rank engagement but not growth per post. Growth per creator over time (follower snapshots) is the better external signal for which archetypes win. Per-post creative signal is weaker than it looks.
4. **Coverage bias.** Business Discovery reaches only Business and Creator accounts and omits licensed-audio Reels' media. Trend-audio Reels are exactly the viral ones, so the visual sample is skewed against them. Report coverage beside every finding.
5. **Scale before rigor.** 10,000 annotated posts cost about $3–5. The constraint is not cost:
   - it is the number of independent creators per cell;
   - and the correction for testing 30+ attributes at once.

   Expect few findings that survive, and treat that as a feature.

## 8. Status ledger

| Capability | Implemented | Isolated tests | Production path | Proven end to end |
|---|---|---|---|---|
| Growth-led CEO and casting prompts | yes | yes | — (no live CEO run since) | no: a sandbox portfolio exercise is still to run |
| All-in capability card, `mw economics` | yes | yes | deployed; no live numbers read yet | no |
| Run caps enforced by the autoscaler | yes | yes | deployed | no capped run yet |
| Stalled-post reaper | yes | yes | deployed, ran (nothing stuck) | — |
| Family-safe volume pruning, weight-reuse notes, boot inventory | yes (talkcomfy v27) | yes | yes | **yes (2026-11-26, $1.36, 6 machines):** talk reused 5/5 weight files twice with 0 GB downloaded; voice and talk weights both intact across each other's machines; ready in ~5 min |
| Presigned storage (no key on rented machines) | yes | yes | yes | **yes** for image, talk and voice jobs; the old R2 key can be rotated |
| Research desk (layer A) | yes | yes | yes ($0.11 live, 7 questions) | yes, for briefs; consumption by agents unmeasured |
| Business Discovery (layer B) | yes | yes (fixtures) | no account | no |
| Annotation v2 and vision benchmark (layer C) | in progress | — | — | — |
| Video frames + measure | measure yes; v2 frame reads in progress | measure yes | measure ran on 2026-11-24 calibration | no |
| Pattern feedback from real outcomes | yes | yes | no real outcomes exist | no |
| Maren talking video | yes | yes | 4 attempts, causes fixed | no |
