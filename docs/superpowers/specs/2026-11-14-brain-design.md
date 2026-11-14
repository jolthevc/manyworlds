# The Brain: design

Status: draft for approval, 2026-11-14. Builds on the existing factory (queue, workers, ledger,
storage, policy, production, review gates). Source brief: "Manyworlds / The Brain" one-pager plus
the owner's clarifications of 2026-11-14 (summarised in Intent and Principles below).

## 1. Intent

Manyworlds should be an autonomous, research-driven media intelligence system, not a content
factory. It runs a portfolio of openly AI-generated, fictional Instagram creators; continuously
discovers which personalities, aesthetics and posts audiences respond to; produces a lot of
good content very cheaply; learns from every post; and compounds that knowledge across a growing
portfolio. Routine operation needs no human content production or management.

**What success looks like first:** one complete, evidence-driven cycle for Maren (research →
ideation → standardized creative scoring → production → independent QA and repair → Instagram
publishing or clearly labelled simulated publishing → performance ingestion → analysis → updated
hypotheses → a next batch that is measurably different because of what was learned, with the trace
to show why), then a second, distinct creator launched by the Brain with no bespoke code.

### Owner's principles (decided)

1. **Objective: virality and audience growth.** Optimise for reach, breakout potential, follower
   growth and turning attention into sustained growth. Shares, saves, engagement and follower
   conversion are supporting signals; the Brain learns which predict sustained growth. Revenue,
   products and sponsorships are out of scope for now.
2. **Patience with creators.** Meaningful exploration windows, enough volume, several strategy
   adjustments before a creator is judged. Distinguish an unpromising creator from a promising one
   with weak execution or distribution. Cheap production is what makes long experiments affordable.
   Windows, evidence floors, improvement cycles and retirement criteria are policy, and the
   strategist weighs the value of learning against the cost of continuing.
3. **Natural feeds, no forced purpose.** Creators post what such a person would post (dinners,
   outfits, beach weekends, games, mirror selfies, opinions, small observations). Posts are not
   assigned a strategic category. Everything is still scored on standard characteristics, to learn,
   not to restrict.
4. **Appealing creators and feeds.** Photogenic, charismatic, stylish, distinctive people of varied
   ages, backgrounds and styles, not one beauty standard or generic model look. Attractive settings,
   good outfits, believable photography, strong taste.
5. **Video must be genuinely good.** Photos and cheap lifestyle content first, but video is built
   early and is not gated on traction. Formats are chosen per creator (talking video, narrated
   edits, lifestyle clips, …). No standard video style. If cheap video isn't convincing, improve the
   method or spend more selectively; never ship visibly inferior video.
6. **Disclosure: clear, unobtrusive.** Account-level disclosure (bio) and Instagram's AI label on
   every post (`is_ai_generated` on the media container). Captions stay in the creator's natural
   voice and don't announce AI. Fictional lifestyle content must not fabricate real endorsements or
   misleading claims.
7. **Two kinds of intelligence:** post-level (what content works) and creator-level (which
   identities, aesthetics, personality traits and format mixes grow audiences).
8. **Architecture:** existing Postgres and queue as the foundation, specialized agents with distinct
   responsibilities, durable shared memory, dynamic orchestration, recorded decisions; modular and
   configurable throughout. No agent framework for its own sake; agent count is not a goal.

Added by the owner on approval (2026-11-14, second round):

9. **Market intelligence.** Scouts research whole markets (fitness, lifestyle, beauty, fashion,
   sports, gaming, home, entertainment, emerging ones), not only our creators: audience size, growth,
   demographics and attentiveness; successful and emerging archetypes; resonant formats, aesthetics
   and personalities; competition, saturation and underserved segments; sustained-growth potential;
   monetization potential (informs long-term attractiveness only; early optimisation stays growth);
   feasibility and economics of convincing AI-native content. Both original invention and
   pattern extraction into differentiated fictional personalities.
10. **Lifecycle vs potential.** Creators move through conceptual stages (exploring, iterating,
    growing, breaking out, established, paused, retired; not a fixed implementation). Stage
    (observed results) is separate from **conviction** (evidence-based belief in future upside, with
    uncertainty). Continue/stop decisions maximise expected value: growth potential + information
    gained − incremental cost.
11. **Market clusters.** The Brain can decide a whole category deserves concentrated investment and
    grow a cluster ("node") of differentiated creators in it (from a few to many; no fixed numbers),
    with shared node intelligence, reuse of learnings, and awareness of overconcentration,
    repetition, audience overlap and correlated experiments. Deliberate expansion, never
    near-identical accounts; no artificial cross-account engagement.
12. **Master budget, allocated by the Brain.** One configurable operating budget; the Brain allocates
    across markets/clusters, creators, formats and experiments, models and quality tiers, and
    research vs new creators vs optimisation. It may fund high-conviction unproven bets, balances
    exploration, exploitation and diversification, keeps a reserve, and paces spend. Objective: best
    expected portfolio outcome per incremental dollar, counting growth and information. Authority
    and safeguards in versioned policy; reallocation within authority is autonomous and reasoned;
    beyond it needs approval.
13. **Three levels of intelligence** (post, creator, market) that inform each other, with findings
    graded proven / tentative / speculative, every allocation traceable to research, experiments,
    data and stated uncertainty.
14. **Event-driven, not only scheduled.** Daily and weekly cycles are defaults; the Brain also reacts
    to trends, breakout posts, research discoveries, QA problems, sudden performance changes and new
    market opportunities. Agents can commission work, challenge assumptions and request research.
15. **Video choice is the Brain's.** It picks production approaches per creator and format; the
    automated video gate decides what ships (video ships only when it meets the creative and
    technical bar).

Added on approval of the plan (2026-11-14, third round):

16. **Many creators at once.** Phases A/B/C are engineering validation, not a launch strategy. Once
    production and orchestration are reliable, the Brain launches and runs many creators
    concurrently, including large differentiated cohorts in attractive markets (e.g. 25-50 distinct
    fitness creators, illustrative). Audiences are not zero-sum; no arbitrary creator counts or
    conservative concentration rules. Real constraints: budget, infrastructure, production quality,
    genuine differentiation, platform rules.
17. **Monetization is optionality.** Record markets' commercial attractiveness (brand partnerships,
    sponsorships, advertising, products); let creators have natural tastes and interests; no forced
    placement or selling. Activate monetization after meaningful traction (partnerships first).
18. **Network effects later.** Accounts grow on their own content; no artificial cross-account
    engagement. The architecture should allow future authentic collaborations, shared partnerships
    and category-level initiatives without building cross-promotion now.
19. **Retirement is relative expected value**, not proof that a concept is bad. Initial budget
    settings (/month, 15% reserve, current safeguards) are starting values, not assumptions.

### Non-goals (for this design)

- Monetization, products, sponsorship matching (later).
- Platforms other than Instagram (adapters later; nothing Instagram-specific leaks into agents).
- Comment replies and DMs (policy `autonomy.comment_replies` stays disabled; a later design).
- A web console. Operator visibility is through `mw` commands and decision traces for now.

### Anti-goals (failure even if it technically works)

- A plot factory or engagement-bait formula: every post optimised into hooks.
- Learning that is really an LLM vaguely "told to improve", with no traceable evidence.
- Simulated audience numbers ever treated, displayed or used as evidence of real demand.
- Killing creators on early noise; or the opposite, funding a dead experiment forever.
- Endless blind regeneration; runaway spend or loops.
- Visibly AI-looking video shipped because "video is supported".
- Copying a real person's identity, face, voice or posts; coordinated behaviour between our accounts.

## 2. Constraints

Existing rules in CLAUDE.md stand: no magic numbers (policy), everything that costs money in the
ledger, decisions recorded with policy version, adapters for every vendor, adults only
(`creator_min_age`), original people, `scope.excluded_topics`, research privacy, no cross-account
engagement, secrets only in `.env`.

Platform facts (checked 2026-11-14; re-verify at build time):

- Instagram Content Publishing API (Instagram API with Instagram Login) publishes photos,
  carousels, Reels and Stories for professional accounts. Two steps: create container, publish.
  Containers expire after 24 h. `is_ai_generated=true` on the container applies the AI info label
  (carousel: on the parent only). Media must be at a public URL (R2 presigned URLs).
- The per-account publishing cap is documented inconsistently (25/50/100 per 24 h); read it at
  runtime from `content_publishing_limit`, never hardcode it.
- Media insights via `/{media-id}/insights`; saves, shares, total views also on the media object.
  **Story insights expire after 24 h**: they must be polled on a schedule inside that window.
  Webhooks would need a public HTTPS endpoint, which the laptop host doesn't have; polling is used.
- Accounts, the Meta developer app and accepting Meta's terms are the owner's steps; tokens go in
  `.env`. Long-lived tokens are refreshed automatically before expiry.

Real-world content: real places are fine (Venice, a beach, a stadium); real trademarks, logos and
team marks are not (existing `real brands without rights` rule), so "a college football game" is
shown without a real school's marks. Nothing implies endorsement by a real brand or person.

## 3. Approaches considered

| Approach | Trade-off | Verdict |
|---|---|---|
| **A. Agents as modules on the existing Postgres queue** (chosen) | Reuses durable jobs, retries, leases, idempotency, the ledger and decision log; one system of record. We write the orchestration logic ourselves. | Picked by the owner and builder: no second state store, nothing rebuilt. |
| B. An agent framework (LangGraph, CrewAI, …) | Ready-made graphs and handoffs, but its own state/checkpoint store duplicates the queue and the decision log; another dependency to keep in step. | Rejected: adds infrastructure for its own sake. |
| C. Hosted agents (Claude Managed Agents) for every role | Anthropic runs the loop and sandbox; less code. But state lives outside our DB and per-session costs are harder to attribute; overkill for structured calls. | Not for the core. May be used inside one agent (e.g. a scout that browses) behind an adapter. |

## 4. Design

### 4.1 Shape

```
                 portfolio cycle (daily)                     creator cycle (per creator, per batch)
   OBSERVE  ─ Scouts ─ Analyst ──┐                 Persona keeper ─ Showrunner ─ Creative evaluator (ideas)
   HYPOTHESIZE / ALLOCATE ─ Strategist ──────────►  ─ Production (existing) ─ QA & repair (assets + feed)
                                                   ─ Publisher (Instagram | simulated) ─ Metrics collector
   LEARN ◄──────────── Analyst/Learner ◄──────────── measurements
```

Two loops, both on the queue:

- **Portfolio cycle** (daily, policy): scouts refresh research; the analyst updates findings; the
  strategist reviews every creator's phase and evidence and allocates: launch / continue / adjust /
  scale / pause / retire, budgets, quality tiers, format mixes, and which experiments to run.
- **Creator cycle** (per production batch, policy cadence; weekly by default with a mid-week
  top-up): memory → ideas → idea scoring → production → QA/repair → publish schedule → metrics.

### 4.2 Agents

Each agent is a Python module `manyworlds/brain/agents/<name>.py` with: a typed input (read from
DB), a typed output (Pydantic), a `run(conn, policy, llm, task)` entry point, its LLM role from
`policy.llm.roles`, and an LLM budget from policy. Every run writes its outputs to tables and one
`decisions` row (inputs digest, output, model, policy version, rationale).

| Agent | Responsibility | Writes |
|---|---|---|
| **Scouts** | Continuous research on public, logged-out data: fast growers, archetypes, emerging formats, audience interests, unusual growth. Extract transferable patterns (never identities). Commissioned by the strategist with a question, or on schedule. | `research_runs`, `findings` (kind=research) |
| **Strategist** | Portfolio decisions (4.6): creator phases, launches from theses, budgets/tiers/format mixes, experiment portfolio, value-of-information vs cost. | `creator_states`, `experiments`, `allocations`, decisions |
| **Persona & memory keeper** | Owns each creator's identity: face/voice anchors (existing), tastes, style signatures, opinions, places, people, running canon, recent history. Updates memory from published posts; resolves contradictions. | `creator_memory`, profile versions (existing) |
| **Showrunner** | Plans the next batch of ideas as a natural feed: everyday moments, outfits, places, opinions, trends, occasional arcs, Stories. Uses memory, strategist allocation, open experiments and findings. Free creative latitude; no forced purpose. Replaces today's planner. | `ideas` (one per future post/story) |
| **Creative evaluator** | Independent of the showrunner (different prompt, no access to its reasoning). Scores every idea before production and every finished asset after, on the versioned rubric (4.4). | `evaluations` |
| **QA & repair** | Existing review gates (identity, realism, safety, scene; clip keyframes) plus caption voice/naturalness, originality, policy, and **feed QA** over the last 9–12 grid posts plus active Stories. Diagnoses failures and picks a targeted repair (4.7). | `reviews` (existing), `evaluations` (execution, feed), repair decisions |
| **Analyst / learner** | Joins pre-publication attributes to real outcomes; normalises; estimates effects with confidence; proposes and closes experiments; maintains creator-level and portfolio-level findings; recalibrates the rubric by predictive validity. | `findings`, `experiments`, rubric proposals |

Integrations (not agents): **Instagram publisher**, **metrics collector**, **simulated audience**,
existing production/workers.

### 4.3 Orchestration

- New job kinds `brain.<agent>` (e.g. `brain.showrunner`) run on pool `local` by a **brain worker**
  service on the control host (`docker compose --profile live`), using the existing runner,
  leases, retries and idempotency keys. Agents commission each other by enqueueing jobs with
  `parent_job_id` (e.g. strategist → `brain.scouts` with a question; QA → production with a repair
  instruction). Dependencies are explicit: a job is enqueued only when its inputs exist.
- An **orchestrator** module (deterministic code, not an LLM) advances each creator's cycle as a
  state machine over durable state (like `production.advance` does now), and runs the daily
  portfolio cycle. LLM judgement lives inside agents at decision points; control flow, limits and
  retries are code.
- Hard caps from policy: LLM spend per agent per day, jobs per cycle, repair attempts, research
  items, and the existing GPU caps. Anything over a cap stops and is recorded, never retried blindly.
- Recovery: every agent job is idempotent (keyed by creator, cycle and step); a crashed run reruns.
- **Events, not only schedules.** Detectors (code) watch durable state and raise `events`: a post
  outperforming its baseline by a policy multiple, a sudden drop, a QA failure pattern, a new
  research finding, a trend surfaced by scouts, budget pacing off-track. The orchestrator routes
  each event to the agent that owns it (e.g. breakout → analyst + showrunner for a follow-up;
  drop → analyst diagnosis; market finding → strategist), with per-event-type rate limits so an
  event storm can't trigger runaway work. Daily/weekly cycles remain as the floor.
- **Challenge and commission.** Any agent may enqueue a bounded request to another (research
  question, diagnosis, revision, experiment proposal) with a reason; the receiving agent may
  decline with a reason. Requests count against the requester's budget.

### 4.4 Creative scoring framework

**Rubrics live in versioned files** `policy/rubrics/idea-v1.yaml`, `execution-v1.yaml`,
`feed-v1.yaml`. Each dimension has a definition, what it is not, and anchors at 1, 3, 5, 7, 9
written as concrete descriptions plus one or two exemplar posts (described in text; later real
ones). Starting idea dimensions (refined during build):

| Dimension | Measures |
|---|---|
| Relatability | Would the target audience see their own life in it |
| Entertainment / humour | Fun, wit, charm |
| Novelty | Fresh angle vs. seen-it-before |
| Curiosity | Makes you want to look closer, swipe, or read |
| Emotional pull | Warmth, longing, nostalgia, awe |
| Shareability | Would someone send it to a friend or repost it |
| Aspiration / visual promise | A life, place, look or moment people want to see |
| Personality expression | Says something only this creator would say or do |
| Audience relevance | Fits what this creator's audience follows them for |
| Naturalness | Believable as a real person's post; not a formula or a hook |

Execution dimensions (on the finished asset): photographic believability, visual appeal and
composition, styling and outfit, identity consistency, setting quality, caption voice, technical
defects. Feed dimensions: grid coherence, variety, repetition, aesthetic consistency, how the
creator comes across.

**No dimension is a target.** A casual coffee Story scoring 2 on novelty is normal and fine. The
Brain learns weights from outcomes (4.5); nothing gates on a single dimension except the existing
safety/identity gates.

**Reproducibility protocol (policy-configured):**

- Each item is scored `samples` times (default 3) by the evaluator model; the median per dimension
  is stored with the spread. High spread flags an unclear rubric item.
- A fixed **reference set** of ideas and assets is re-scored whenever a rubric or evaluator model
  changes; scores are only compared across versions after measuring the shift on this set.
- The evaluator sees the creator's profile summary but not the showrunner's reasoning.
- Every evaluation row stores: item (idea or asset), rubric id+version, evaluator model, prompt
  version, per-dimension scores, spreads, rationale, creator profile version, format, timestamp.
- All idea evaluations are written before the post is published (enforced: an idea can't be
  scheduled without one).
- Cost control: a cheaper model scores at volume; a sample (policy rate) is double-scored by the
  stronger model to measure agreement.

**Idea vs execution:** ideas are scored from the plan text before anything renders; assets after
rendering. Analysis keeps both, so "great idea, weak render" is learned as an execution problem.

**Rubric recalibration:** the analyst periodically measures each dimension's predictive validity
(4.5). It may propose a new rubric version (reweight, redefine, split, drop, add); a proposal is
recorded with its evidence and adopted under policy limits. Predictive usefulness beats plausibility.

### 4.5 Measurement and learning

**Collection:** the metrics collector polls each published item at policy observation ages (default
posts: 1 h, 6 h, 24 h, 72 h, 7 d, 28 d; Stories: 1 h, 6 h, 20 h, before expiry) and stores reach,
views/plays, likes, comments, saves, shares, profile visits and follows where available, plus
account-level follower counts daily. Real data only in `metric_snapshots`.

**Simulated audience:** a separate module produces synthetic metrics from a panel of
LLM-simulated audience members, stored only in `simulated_metrics` with its model and version. Used
for: testing orchestration, data flow and whether the Brain adapts the next cycle. Code-level guard:
the analyst and strategist read real tables for any finding with `evidence=real`; findings derived
from simulated data are stored with `evidence=simulated` and the strategist's allocation logic
refuses them for promotion, retirement and budget. Reports label them "SIMULATED".

**Normalisation:** every outcome is compared to a baseline for that account and format at the
same post age (rolling median of recent comparable posts), as a log-lift, so a young account's
small numbers and a mature account's large ones are comparable. Covariates recorded per post:
account age, follower count at publish, format, publish hour/day, post age, whether trending audio
or a trend was used.

**Analysis, staged by data volume (thresholds in policy):**

1. Few posts: descriptive only; findings are hypotheses with low confidence.
2. Enough posts: per-attribute and per-dimension lift with bootstrap confidence intervals, within
   creator and pooled across creators.
3. Larger volume: regularised regression / hierarchical model with creator as a grouping factor,
   so creator-specific and portfolio-wide effects are separated.

**Questions it answers:** which dimensions and attributes predict reach and views (discovery) vs
follows, saves, shares and return engagement (attachment); which combinations outperform; how this
differs by format (photo, carousel, Reel, Story) and by creator; which early signals (e.g. shares
per reach at 6 h) predict sustained follower growth, which tells the strategist what to watch.

**Correlation vs causation:** findings carry confidence and an evidence type
(observational | experimental | simulated). The analyst turns promising observational findings
into **experiments**: a hypothesis, two or more arms (e.g. same creator, outfit-led vs place-led
photos), an outcome metric, a minimum sample from policy, randomised assignment across slots, and a
stopping rule. Results update confidence; only experimental findings with policy-level confidence
become "established" in the playbook.

**Exploration:** the showrunner gets a policy exploration share per batch: some ideas test open
experiments or untried directions; the rest exploit what is known. Allocation across directions
uses Thompson sampling over current findings, so uncertain-but-promising directions keep getting
tried.

**Three levels, one findings store.** Every finding has a scope (post-level within a creator,
creator-level, cluster, market, portfolio), an evidence grade (**proven**: replicated experimental
result at policy confidence; **tentative**: consistent observational pattern with a confidence
interval; **speculative**: hypothesis from research or a single observation) and links to its
sources. Levels feed each other explicitly: a strong post-level finding in one fitness creator
becomes a tentative cluster hypothesis queued for testing in siblings; several creators doing well
in a market raise market conviction; repeated failures open a diagnosis finding.

**Creator-level intelligence:** each creator is described by attributes (market, archetype,
aesthetic, personality traits, age band, setting, format mix, posting cadence, quality tier).
Growth curves are compared against a portfolio baseline curve for account age. With more creators
this becomes the dataset for "which identities grow", feeding the strategist, creator lab and scouts.

**Market-level intelligence:** market conviction combines scout research with our own creators'
outcomes in that market (4.6a), and its forecasts are scored against what then happens.

### 4.6 Strategist: lifecycle, conviction and continue/stop

**Stage** (observed results) and **conviction** (belief in future upside) are tracked separately.

Stages are labels the strategist assigns with a recorded reason; the default set (policy, not
code) is exploring, iterating, growing, breaking_out, established, paused, retired. Each stage
maps in policy to default behaviour (variety vs focus, cadence range, quality-tier range, minimum
evidence window before the next judgement). The strategist may move a creator between any stages;
policy sets minimum windows (posts, days, real reach) before a downward move and which moves need
approval.

**Conviction** is a distribution, not a score: the estimated long-run audience outcome for the
creator (e.g. followers at a policy horizon, relative to the portfolio's age-matched baseline) with
a credible interval. It combines, with weights the analyst re-estimates as outcomes arrive:

- market attractiveness (the creator's market and cluster, 4.6a),
- creator appeal (pre-publication creator-level evaluation: visual appeal, personality, distinctiveness),
- creative differentiation vs the market and vs our own creators,
- early performance signals that have been shown to predict growth (learned, 4.5),
- unexplored options remaining (formats, aesthetics, personality emphases not yet tried).

A creator can therefore be "exploring, weak results, high conviction" or "growing, strong results,
low conviction" and be managed accordingly.

**Continue/stop is an expected-value decision.** For each creator and the next evidence window:
expected growth value + value of information (how much the window could change the allocation,
from the width of the conviction interval) − incremental cost (ledger-based forecast), compared with
the best alternative use of the same money. Retirement happens when continuing is worth less than the
alternatives, never on a demand to prove the concept bad; policy sets minimum windows and a minimum
number of strategy adjustments before it is considered, and the diagnosis (concept vs execution vs
distribution) feeds the estimate. High-conviction creators that have not broken out can receive more
investment when there is a credible thesis and unexplored options remain. Every decision records the
inputs, the numbers and the reasoning.

Launches: the strategist decides when to launch a creator and what it should test (an
under-explored identity attribute, a market thesis, a cluster gap), and asks the scouts/creator
lab for a creator designed for that. Creator design follows principle 4 (appealing, distinctive,
varied). Launches are **cohorts**, not one at a time: the strategist sets cohort size, timing and
investment from conviction, opportunity, budget and capacity, and many creators in one market can run
at once (16).

Creators get natural, recognisable interests, tastes, wardrobe and aesthetics because real
influencers have them; these preserve later monetization optionality, but no trait, product
placement or selling is chosen for commercial reasons now (17).

### 4.6a Market intelligence and clusters

- **Markets** are first-class records: a category or sub-niche with a research dossier (the
  factors in principle 9), a **market conviction** distribution (attractiveness for sustained
  audience growth, with monetization potential as a separate long-term attribute, not in the
  growth objective), and evidence links. Scouts refresh them on schedule and on demand; findings
  from our own creators in that market update market conviction (several winners raise it,
  repeated failures prompt a diagnosis: strategy, production quality, or the thesis itself).
- **Clusters (nodes)** group our creators in a market. Each keeps shared intelligence: audience,
  formats, successful personalities, content styles, production strategies; showrunners and the
  creator lab read their cluster's findings, so learnings transfer while identities stay distinct.
- **Differentiation is measured, not hoped for:** each new creator in a cluster is checked against
  its siblings (identity, aesthetic, voice, content themes) with a policy similarity ceiling; feed
  QA also checks cross-creator repetition. The point of differentiation is breadth of experiments
  (aspirational, funny, casual, educational, athletic, fashionable, lifestyle …), not avoiding
  competition: audiences are not zero-sum, and there is **no default cap** on creators per market.
  Concentration (budget share, correlated outcomes) is reported and is one input to the allocator
  as uncertainty, never a hard limit unless the owner sets one in policy.
- **Platform integrity:** clusters are separate, disclosed accounts with distinct identities;
  no coordinated engagement, no cross-promotion by automation, account creation paced per policy.
  Meta treats coordinated networks of accounts as inauthentic behaviour, so account creation and
  launch pacing follow platform rules and every account must stand on its own. Accounts are created
  and authorised by the owner; the Brain queues account requests for each cohort.
- **Future network value (not built now):** collaborations, shared brand partnerships,
  category-level commercial initiatives and natural audience crossover once properties have their
  own audiences. The data model keeps creators, clusters and markets linked so these can be added
  later as authentic, platform-compliant features; nothing cross-promotes today.

### 4.6b Capital allocation

- One **master operating budget** (policy: amount per period, pacing curve, reserve share). The
  ledger already records every dollar; allocation turns it into budgets per bucket per period.
- **Hierarchical allocation**, re-run each portfolio cycle and on events: budget across
  research / new creators / existing creators / reserve → across markets and clusters → across
  creators → within a creator, across formats, experiments and quality tiers (the showrunner and
  production routing spend inside their creator's envelope).
- **Method:** each candidate use of money (continue creator X at tier T, launch in market M,
  expand cluster C, research question Q) is an option with expected growth value, information
  value and cost. The strategist proposes allocations; deterministic code checks them against
  policy (authority limits, minimum exploration share, reserve, pacing, and any optional
  concentration bounds the owner sets)
  and applies only what passes. A share is always held for exploration (new markets, high-conviction
  unproven creators), so winners never absorb everything.
- **Authority:** policy `allocation.authority` sets what the Brain may change alone (e.g. moves
  within a period's budget, tier changes up to a ceiling, launches within the budget and
  infrastructure capacity) and
  what queues for approval (budget increases, new markets beyond N, anything over `limits`).
  Approvals are a pending state the owner clears with `mw` or a private page.
- **Spend enforcement** stays in code: the autoscaler, per-day GPU caps and LLM budgets read the
  current allocation, so no agent can overspend its envelope.

### 4.7 QA, repair and feed review

- Existing gates stay (image: identity, realism, safety, scene; video: identity, motion, realism,
  safety, scene). Added: caption voice and naturalness, originality vs the creator's history, and
  policy (no real-brand implication).
- **Diagnosis before repair:** a failed item gets a diagnosis (category + evidence) and one targeted
  action from a policy-ordered list: re-prompt with the specific defect named; new seed; stronger or
  different model; different format (e.g. a clip that won't animate well becomes a carousel); edit
  caption; drop. Each item has a repair budget (attempts and dollars); exhausted means drop and
  record why. No blind regeneration loops.
- **Feed QA** runs before each publish window over the last 9–12 grid posts plus active Stories,
  with the next item in place: repetition, aesthetic consistency, variety, continuity with memory.
  Failures go back to the showrunner as a specific request, not a regeneration.

### 4.8 Production and video

Production stays as built (Step-based `production.py`, workers, review). Changes:

- Ideas, not planner posts, become production orders; format and quality tier come from the idea
  plus the strategist's allocation.
- **Stories** as a format (single photo or short clip, 9:16, lighter QA weight).
- **Routing by measured cost per accepted output and quality tier** (extends today's pool
  scoring to models): cheap tier first, better models where the creator's tier allows.
- **Video is built early per principle 5:** narrated edits and lifestyle clips exist; talking
  video (lip-synced speech from a photo and the creator's voice) is a new capability. Model choices
  (Wan 2.2 14B, LTX, a talking-head model) are benchmarked on real outputs and judged by the video
  gate before entering production; the cheapest that passes wins, and a creator's tier can buy a
  better one.

### 4.9 Publishing

- `manyworlds/publish/instagram.py` adapter (Instagram API with Instagram Login): photo, carousel,
  Reel, Story; `is_ai_generated=true` on every container; captions in the creator's voice; bio
  disclosure set at account setup; publish limit read at runtime; container → poll status → publish;
  stores the media id and permalink on the post; retries are idempotent (never double-post).
- A **publish scheduler** chooses times inside policy windows (learned later by the analyst).
- `manyworlds/publish/simulated.py` is the same interface for creators without a real account;
  every simulated publication is marked and lives only in simulated tables.
- **Autonomy:** policy `autonomy.publish` (human_approval | sampled | autonomous) already exists and
  gates the publisher. Setting it is the owner's call (see section 6).
- **Account safety:** one account per creator; no interaction between our accounts; no follows,
  likes or comments by automation; accounts warmed up per policy cadence ramp.

### 4.10 Data model (new migrations)

| Table | Purpose |
|---|---|
| `creator_memory` | versioned facts per creator: kind (taste, opinion, place, person, routine, canon), text, source post, valid_from/until |
| `creator_states` | phase, entered_at, window counters, current adjustment, quality tier, history |
| `ideas` | one future post/story: creator, batch, concept text, format, scenes/script/caption draft, attributes (setting, outfit, activity, mood, trend …), experiment arm, status, post_id once produced |
| `evaluations` | rubric scores per item (idea, asset, feed), as specified in 4.4 |
| `rubric_versions` | rubric id, version, file hash, adopted_at, reference-set results |
| `findings` | scope (creator, portfolio), statement, evidence type, effect size, interval, n, confidence, status (hypothesis, testing, established, rejected), sources |
| `experiments` (exists; extend) | hypothesis, arms, metric, min sample, assignment, status, result |
| `allocations` | strategist output per creator per period: budget, tier, format mix, cadence |
| `publications` | platform, account, external id, permalink, published_at, simulated flag, AI label set |
| `metric_snapshots` (exists) | real metrics only, with post age at capture |
| `simulated_metrics` | synthetic metrics, model and version |
| `accounts` | per creator per platform: handle, account id, token reference (name of the `.env` key, never the token), status |
| `markets` | category / sub-niche, dossier, market conviction (mean, interval), monetization notes, evidence links |
| `clusters` | market, strategy, shared-intelligence summary, member creators, concentration metrics |
| `convictions` | time series of creator / market conviction distributions with their inputs |
| `budgets` | master budget periods, allocations per bucket, pacing, reserve; actual spend joined from the ledger |
| `approvals` | anything beyond authority: what, why, amount, status, decided_by |
| `events` | detector output: type, subject, payload, routed_to, handled_at |

`posts` remains the unit of publication; an idea produces a post.

### 4.11 Models and cost

LLM roles from `policy.llm.roles`: strategist, showrunner, analyst on the strongest model; the
evaluator and QA screen on cheaper models with sampled strong-model checks; scouts mixed. Per-agent
daily LLM budgets in policy, all calls in the ledger. GPU work is unchanged (autoscaler, batching,
pool scoring). Cost per accepted output and cost per real follower gained are reported per creator.

## 5. Build sequence

1. **Foundations:** migrations (4.10), rubric files v1 with anchors, brain worker and orchestrator
   skeleton, agent module contract, `mw brain` commands to run and trace a cycle.
2. **Ideas and scoring:** persona keeper (memory from existing posts/canon), showrunner (replaces
   planner; ideas table), creative evaluator with the reproducibility protocol and reference set.
3. **Instagram path (parallel, as soon as an account exists):** adapter, token handling, publish
   scheduler, metrics collector with Story polling; first real Maren posts.
4. **Simulated path:** simulated publisher and audience; full creator cycle runs end to end.
5. **Learning:** analyst (normalisation, lift with intervals, findings), experiments, exploration
   share; showrunner consumes findings; a demonstrable "next batch changed because of X" trace.
6. **QA and repair:** diagnosis-driven repair, caption/originality checks, feed QA, Stories.
7. **Strategist v1:** stages, creator conviction, continue/stop by expected value, launches;
   portfolio cycle; events and commissioning.
8. **Second creator** proposed and launched by the strategist from research, with no bespoke code.
9. **Portfolio intelligence layer:** markets and market conviction (scouts' market research),
   clusters with shared intelligence and differentiation checks, master budget with hierarchical
   allocation, authority and approvals, pacing and reserve.
10. **Video:** talking video and model benchmarks through the video gate (can start in parallel
    once 1–2 are done, on the existing GPU infrastructure).

Phase A (steps 1–6, with 3 in parallel) is the Maren loop; Phase B (7–8) the second creator;
Phase C (9) the portfolio layer. Data structures for markets, clusters and budgets are created in
step 1 so Phase A writes into them from the start (Maren belongs to a market and a cluster of one).

Each step ships with tests (stubs for Claude/Instagram, as today), policy entries, and a demo
trace.

## 6. Owner's actions and calls

- Create the Instagram professional account for Maren and a Meta developer app; add the tokens to
  `.env` (I build everything after that).
- Set the master operating budget and the Brain's allocation authority in policy (defaults below).
- Decide the initial `autonomy.publish` setting for real posts (default below).

Proposed defaults (builder's call, owner may change in policy at any time):

- Master budget: $750 per month (today's $25/day overall cap), paced evenly with a 15% reserve;
  `limits.daily_usd_max` stays the hard ceiling. The Brain reallocates freely inside the month's
  budget; raising the budget or exceeding `limits` needs approval.
- Publishing: autonomous from the first real post, with a kill switch (`autonomy.publish:
  paused`) and a daily digest of what went out; Stories and 24 h containers make slow approval
  costly.
- Video: the automated video gate decides; the digest includes each new video format or model's
  first outputs so the owner can object.

## 6a. Builder's calls (made while writing; open to correction)

- **Simulated vs real creators.** Each creator publishes in one mode at a time: `simulated`
  (sandbox) or `instagram`. A sandbox creator's showrunner may use simulated findings, which is how
  the learning loop is proven before a real account exists; every such finding and trace is marked
  SIMULATED. A real creator's agents read real findings only (enforced in the findings query, not by
  convention), and the strategist never uses simulated findings for promotion, retirement or budget.
- **Launches and accounts.** A creator the strategist launches starts in sandbox mode with state
  `awaiting_account` and a recorded request to the owner. Its sandbox history does not count toward
  real exploration windows; the window starts at its first real post. Ready content can be
  published once the account is connected.
- **Autonomy over its own settings.** The Brain changes allocations, tiers, cadences, exploration
  shares and rubric versions itself, within the bounds in policy `limits`; those runtime values live
  in the DB as versioned overrides (each decision records the policy file version plus the override
  version). Anything beyond `limits` waits for a human. The policy file stays the human-edited base.
- **Existing planner posts** are left to drain through today's production path; new content comes
  from `ideas`. The planner is retired once the showrunner is live.
- **TikTok** stays dormant in policy; nothing is built for it in this design.

## 7. Left to the builder

Exact rubric wording and anchors (drafted, then tuned on the reference set); statistical method
details within the staged plan; prompt design; table columns beyond those listed; scheduling
defaults; which video models to benchmark first; the exact observation ages; module layout inside
`manyworlds/brain/`.
