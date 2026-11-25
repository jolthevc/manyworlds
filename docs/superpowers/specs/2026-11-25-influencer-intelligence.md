# Influencer Intelligence: audit and pilot design (2026-11-25)

Owner brief: "Influencer Intelligence Engine" (2026-11-24), section 11, first two steps: audit, then pilot design.
This spec stops before any paid run. Nothing here has run on Apify, an LLM or a GPU.

Owner, 2026-11-24: "this is really helpful for the brain to give us a better sense of data — learn what's working
with real creators, and learn from them. We are not taking any likeness or stealing."
Owner, 2026-11-25 (three decisions, folded in below):
- **The pattern library is multimodal.** Photos and videos are one system. Video is studied as video: formats,
  performance, cinematography, editing and production.
- **Research records may identify public creators.** We keep names, handles, links, annotations and provenance,
  and media as access-controlled research references. Third-party media is never a generation input unless its
  licence says so.
- **Emulate freely.** A named real creator may be the explicit reference in a brief ("X is successful; understand
  what makes her content work and develop a creator with comparable appeal"). One creator can be the whole
  reference, and nothing records why a creator is original. What stays is cheap checks:
  - no impersonation (name, handle or a near-identical face);
  - no near-duplicate copies of specific posts.

**In one paragraph.** Most of the pipe already exists:
- collection (Apify, ScrapeCreators);
- a cheap vision read of posts with outlier and control samples (`research/trends.py`);
- video measurement (the 2026-11-24 talking-video calibration);
- a 43-shot vocabulary, archetype playbooks, and the creator lab, style director, talent manager, showrunner and
  director;
- typed requests, predictions and the ledger.

Four things are missing:
1. a **research subject** that tracks named creators over time, not anonymous hashtag posts;
2. a **three-layer annotation** of photos and videos;
3. **pattern cards** with evidence and outcomes, which the director and showrunner retrieve;
4. a **synthesis and audition** step before casting.

The smallest fix is one new agent (`brain.intelligence`, the IIA), five tables, one policy section, one pattern
file, one CPU job kind (`finish.measure_video`) and one retrieval hook each in the director, the showrunner and talk
production.

---

## 1. Audit

| Brief need | What exists | State | Gap |
|---|---|---|---|
| Hypothesis (statement, audience, competing explanations, disconfirming evidence, commercial pathways, owner, budget, status, version) | `theses` (summary, evidence, status proposed/chosen/launched/rejected, body); `discover.Thesis` (archetype, audience, why_now, differentiation, risks, production_fit); market lead `cohort` requests turn into theses (`proposals._cohort_theses`) | partial | no competing explanations, disconfirming evidence, budget, owner role, version, or researching/auditioning/archived states |
| Logged-out public collection | `research/apify.py` (IG hashtag scraper, TikTok clockworks, $2.70 / $1.70 per 1k items); `research/scrapecreators.py` (TikTok search, trending, IG reels search, TikTok transcripts; $0.002 a credit); `research_runs`/`research_items` with handle-free post URLs and cover URLs | covered for hashtags and niches | **no per-creator pull** (a creator's last 20-40 posts) and no creator entity. Handles are salted hashes and raw items purge at 30 days, which is right for the trend watcher's anonymous statistics but can't track a named creator over months |
| Discovery and scouting | `brain/discover.py` + `agents/scouts.py`: research plan, hashtag pulls, theses | covered for markets | no reference-creator selection (leaders, risers, mid-sized, adjacent, low performers), no scoring, no coverage or confidence |
| Traction vs baseline | `trends.pick`: log10 views over the niche median, velocity from snapshots, outlier and control samples; `signals.summarize`: median engagement by format and tag | partial | measured against the niche only, never against **the creator's own median** or a follower band. No low-performer comparators chosen on purpose |
| Photo annotation | `trends.PostFeatures` (format, hook, opens on face, setting, tone, pacing, why it works) from the cover, caption and speech. Role `trend_features` (qwen3.8-flash, Haiku fallback) | partial | one layer only (production). Nothing on pose, hands and gaze, who "took" the photo, imperfections, wardrobe, light, social energy, the personality and relationship layer, or economics. Observed, mechanism and speculation are not kept apart |
| Video annotation | Calibration 2026-11-24 (`docs/research/2026-11-24-talking-video-calibration.md`): hard and soft cuts, shot lengths, first cut, words per second from Whisper word timestamps, face-to-camera and b-roll share, captions, hook, from a Haiku read of 8 frames. 84 videos for $1.17 Apify and about $0.10 Haiku. `song_measure.py` + `finish.measure_song` show how a CPU measurement runs as a finisher job | method proven, **code not in the repo** (`measure.py` and `analyze.py` ran by hand on the control plane) | port it as a job. Then add performance (gestures, expression, eye contact, mannerisms, cadence), cinematography (device, angle, movement, transitions) and editing (hook structure, B-roll, music, captions) |
| Comments as evidence | `agents/comment_themes.py` (our own posts' comments, scrubbed, through `evidence.py`) | ours only | none for reference creators, by design (third-party commenters; see section 2) |
| Shot vocabulary | `policy/aesthetics/aesthetics-v1.yaml`: 43 shots (category, face, subject, framing, use), grades, consistency and variety rules, photo-dump order, carousel templates | covered as a vocabulary | flat. It has no pose, expression, styling, social energy or movement dimensions, no evidence, no outcomes, and nothing for video |
| Video formats and recipes | `moves.library` (12 moves: storytime_to_camera, outfit_change_spin, pov_day_timestamps, mirror_talk, walking_vlog…); `generation.talk_edit` grammars (story_time, commercial, long_take; calibrated 2026-11-24); `capabilities`; `dance.references` (licensed motion clips); voice delivery bank; `docs/research/2026-11-24-mannerisms.md` | covered as recipes | no evidence, no per-archetype outcomes, and no performance parameters (pace, gesture cadence, eye contact) that a planner can retrieve |
| Archetypes | `policy/archetypes/` (9 playbooks: promise, communicates, mix, hooks, edit grammar, pitfalls) | covered | written from craft knowledge. Research should feed them, not replace them |
| Creator lab and casting | `creator/lab.py` (bible; "never base a creator on a real person"), `brain/launches.py` (bible → references → anchors → LoRA → profile → dossier → style) | covered for one thesis → one creator | **no 3-5 candidates and no audition** before the LoRA. Nothing records which evidence inspired which trait |
| Impersonation | `dossier.differentiation`: word overlap against siblings (`brain.talent_manager.max_overlap`); voice-lab ECAPA distance between our own creators' voices | siblings only | **no check against the real creators a brief names** (name, handle, face) and no near-duplicate check against the posts it cites |
| Style | `agents/style_director.py`: style, world, season from the aesthetics library | covered | doesn't read evidence. The variety rules ("a new outfit most posts", "no shot type twice") force rotation that the brief rejects for signature elements |
| Talent manager on stall | `talent_manager` review and diagnose (drop events route there); requests `research` → analyst | covered as a trigger | the `research` request reaches the analyst, which reads only our own posts. Nothing goes out to look at real creators |
| Showrunner and director retrieval | `director.creator_brief`: persona, dossier, playbooks, style, recent shot types, known sets, trend brief (`evidence.niche_trends`). Showrunner: trend briefs, comment themes, series | covered as a channel | there are no cards to retrieve |
| Typed requests | `brain/proposals.py` (kinds, routes, rights, allowances, cooldowns) | covered | needs a kind `intelligence` routed to `iia` |
| Predictions | `brain/predictions.py` (median_reach, reach_growth; Brier) | covered | used as-is for hypotheses after launch |
| Findings | `findings` (scope post…portfolio, evidence `research` = never proven), only through `evidence.py` | covered | cards' claims become `research` findings and go through `evidence.py` unchanged |
| Versioned memory | `creator_memory` (per creator, valid_from/until) | per creator only | the IIA's knowledge is the cards, the annotations and the creator records (below), not `creator_memory` |
| Jobs, budgets, ledger, batch | `queue`, `llm_batch` (Message Batches at half price), `budget.py` per agent, ledger `agent` tag, test overlay and `test-` batches | covered | reused as-is |

**Missing, in one list:** reference creators and posts as identifiable, durable research records; a per-creator
pull; baselines against the creator's own median; the three-layer photo and video annotation; video measurement
as code; pattern cards with evidence, acceptance and outcomes; the agent that curates, challenges, distils and
synthesises; auditions (3-5 candidates × a 9-post mini-feed); a cheap impersonation and near-duplicate check
against the named references; retrieval into the director, the showrunner and talk production.

### The smallest new role and data structures

**One role:** `brain/agents/intelligence.py`, agent `iia`, job kind `brain.intelligence`. Its modes are in
section 3.1. The rest is code: pulls, measurement, scoring and annotation batches.

**Data.** Reuse first:

| Brief structure | Where it lives | Why |
|---|---|---|
| Hypothesis | **`theses`, extended** (migration 041): `kind` (`market`, `creator_hypothesis`), `version`, `parent_id`, `owner_role`, `budget_usd`, a status set widened with `researching`, `auditioning`, `archived` (with `archive_reason`); the rest in `body` (statement, segment, audience, competing explanations, disconfirming evidence, commercial pathways, expected signals) | theses already feed launches and cohorts. A second table would split the same idea in two |
| Reference creator | **new `reference_creators`**: platform, handle, display name, profile URL, bio, follower count history (jsonb), why selected, role (leader, riser, mid, adjacent, low comparator), cluster, scores, coverage and confidence, selection-bias caveats, how found, first and last seen | identifiable by owner decision. Kept **apart from** `research_items`: the trend watcher stays anonymous, with its 30-day purge, and is unchanged. Forgetting is a delete plus a hashed blocklist (`research_forget`) |
| Reference post | **new `reference_posts`**: creator, platform, external id, post URL, posted_at, media type, caption (kept: public professional content), metrics at fetch plus a snapshot history, `media` (research copies: video, images, frames) and `media_purge_after`, `licence` (`{name, source, uses: [research, motion_reference, generation_input]}`; absent means research only) | one row per studied post. Media has its own retention. A licence may permit more than research |
| Post annotation | **new `reference_annotations`**: post, schema version, model, `measured` (code: cuts, pace, durations), `observed`, `mechanism`, `speculation` (three jsonb blocks, kept apart), confidence | apart from `research_features` (the trend watcher's anonymous features), so neither schema bends to the other |
| Pattern card | **new `pattern_cards`**: key, version, status (`illustrative`, `draft`, `active`, `retired`), medium (photo, carousel, video, sequence), dimensions, mechanism, audience role, best fit, recipe, constraints, failure modes, guardrails, `sources` (reference post ids and links; one creator is enough), evidence (n posts, n creators, normalised traction with interval), acceptance (our renders), outcomes (our posts), `signature_ok`; seeded from `policy/patterns/patterns-v0.yaml` | cards change at runtime (evidence, acceptance, outcomes), so they live in the database. The seed file keeps the illustrative ones reviewable in git |
| Synthesis and audition | **new `auditions`**: thesis, arm (A, B, C), candidate design (persona, audience promise, visual identity, content behaviour, commercial thesis), `source_cards`, `references` (the named creators the brief used), `checks` (impersonation and near-duplicate flags), `creator_id` (a sandbox creator), batch `test-iia-<run>`, estimated and actual cost, scores, decision (select, revise, reject) | ties the candidate, its nine posts (by `creator_id` and batch) and the decision together |

Unchanged and reused: `proposals` (new kind `intelligence`), `predictions`, `findings` through `evidence.py`
(card claims as `research`, `speculative` or `tentative`, never `proven`), `decisions` (every IIA call), `queue`,
`llm_batch`, the ledger (agent `iia:<stage>`), R2 (a separate research bucket, section 2), the test overlay.

**Reuse or change the trend pipeline's hashing?** We add separate identifiable tables and leave `research_items`
as it is:
- The trend watcher's purpose (anonymous niche statistics, at volume) differs from the IIA's (a small, durable,
  named research base).
- Different purposes deserve different retention and fields.
- Changing the hashing would make thousands of incidental authors identifiable for no gain.

A reference creator first found in `research_items` is re-fetched by handle through the per-creator pull; the hash
is never reversed.

---

## 2. Privacy and rights (revised to the owner's 2026-11-25 decisions)

The CLAUDE.md rule, as the owner set it (edited in this branch):
- Learn from and emulate successful creators freely; named references are allowed.
- Never impersonate a real person, or create a confusingly similar name, handle or face.
- Don't copy specific protected content (near-duplicates).
- Research media stays a research reference unless licensed for more.
- Logged-out public data, within law and provider terms.
- Deletion on request.

| Point | Design |
|---|---|
| **Purpose** (stated in `policy.intelligence.purpose` and in `mw research` help) | "Learn which creative and social patterns make public creators worth following, to direct original AI creators. Not profiling, contacting, or reproducing anyone." |
| **Who** | Public creator accounts with professional content, adults only. The IIA skips accounts that look like minors (vision check plus bio) and private accounts. Logged-out data only; no logins, no circumvention, and Apify or ScrapeCreators only within their terms |
| **Minimal fields** | Handle, display name, profile and post links, public counts, captions, transcripts of the first seconds, our annotations. **Never kept:** comments or commenters (third parties), tagged people, location tags past city level, anything about health, sexuality, religion, politics, family or children, contact details. The annotation schema has no field for them, and the vision prompt says not to describe them |
| **Annotations are about the content** | Coded attributes (enums and numbers) plus short notes on mechanisms. No private-life facts about the person. Named creators and creative dossiers (what makes their content work, how it evolved) are allowed. A pattern card may cite one creator |
| **Media** | Downloaded photos, videos and frames go to the **research store**: key prefix `research-refs/`, in a separate private bucket named by `RESEARCH_STORAGE_URL`. Its token is held only by the control plane and the Fly finisher, never by GPU workers. Retention `intelligence.retention.media_days` (180), deleted by the housekeeping sweep (`research.references.purge`) |
| **Hard separation from production** | Research URIs are refused as production inputs, in code, at three points: <br>• `queue.enqueue`: only `finish.measure_video` and `brain.intelligence` may carry them; every image, video, talk, voice, dance, train, finish and music job is refused; <br>• a database CHECK on `assets.uri` (no asset row points into `research-refs/`); <br>• `dance.reference` (never a motion input). <br>What we learn from videos is descriptions and parameters (cut rhythm, pace, gesture cadence), not footage. A post whose licence permits more (`licence.uses`: `motion_reference`, `generation_input`) becomes a production input only by a deliberate copy into production storage, recorded as a licensed asset (a later step). Never through a research URI |
| **Access** | The `mw_worker` database role has **no** grant on the new tables (a test checks it). Readers: the control plane, `mw research …`, and the IIA |
| **Retention per table** | <br>• **Media:** `media_days` (180) after fetch. <br>• **Creators**, with their posts and annotations: deleted `records_days` (730) after they were last seen, unless an open hypothesis's audition or an active or draft card cites them. <br>• **Pattern card `sources`:** links, kept while the card lives. <br>• **Trend watcher tables:** unchanged (30-day raw purge, anonymous) |
| **Delete on request** | `mw research forget <handle or url>` deletes the creator, their posts, annotations and media objects. It removes their links from card sources (lowering the card's evidence counts) and their id from auditions (marked `reference_forgotten`), and writes an `owner_event`. The handle goes on the `research_forget` blocklist as a salted hash, also for someone we hold nothing on yet, so they are never collected. The command is idempotent |
| **Impersonation and copying checks** (`brain/impersonation.py`, code only, no model call) | Against the real creators the brief names: <br>• (a) **name and handle** string similarity; <br>• (b) **face:** one appearance check at casting, using the face-identity embedding when the pipeline computes one. No new model, no extra LLM; none exists today, so until then this check reports "not run"; <br>• (c) **near-duplicate output:** a 64-bit difference hash of each audition image against the cited posts' research images. <br>Flags are warnings to the talent manager. An obvious impersonation (same or near-same name, handle or face, past `intelligence.checks.*_block`) blocks the launch. Voice: nothing beyond the existing distance between our own creators |

**Owner decisions still needed:**

| # | Decision | Recommendation |
|---|---|---|
| D1 | Retention numbers | Set as recommended: media 180 days, records 730 days unless cited (policy `intelligence.retention`). Shorter if you want less exposure |
| D2 | A separate R2 bucket and token for research media (create it in Cloudflare; Fly secret `RESEARCH_STORAGE_URL` plus its key) | Yes. Workers still lack a bucket-scoped token (tech debt), so research media must not share the production bucket. Until it exists, nothing downloads media |
| D4 | A legitimate-interest note for EU-resident creators (GDPR: purpose, necessity, balance, opt-out through `forget`) | Yes. I draft one page and you sign it before the first paid pull |
| D5 | Platform terms: Instagram and TikTok restrict automated collection; we rely on Apify and ScrapeCreators logged-out data, the same posture as the trend watcher and the calibration run | Confirm the existing posture covers per-creator pulls too, or keep the pilot to hashtag and search results only (weaker: no per-creator baseline) |
| D6 | Comments of reference creators | Not collected (third-party data, little extra signal over saves and shares) |
| D7 | The brief's "no artificial rotation" against `aesthetics.rules.variety` (a new outfit most posts, no shot type twice in a row) | Done in the first PR: the aesthetics rule `signature` lets a pose, set or outfit that her style names as signature, or a card marked `signature_ok`, recur. The variety rules stay for everything else |

---

## 3. Pilot design

### 3.1 The IIA role

`brain/agents/intelligence.py` (`iia`, kind `brain.intelligence`, its own daily budget `brain.llm_usd_per_day.iia`
and research cap `intelligence.daily_usd_max`). It persists everything to the tables above; each call is a
`decisions` row. It never carries a long conversation.

| Mode | Does | Model role (policy `llm.roles`) |
|---|---|---|
| `scout` | Code: candidate creators from hashtag and search pulls, then per-creator pulls, baselines and scores (3.2). One call ranks the shortlist against quotas | `iia_curate`: Opus, 1-2 calls |
| `challenge` | Argues against the hypothesis: crowding, competing explanations, what would disconfirm it, which comparators are missing. Can send `scout` back for counterexamples | `iia_curate` (Opus), 1 call |
| `annotate` | Code queues `finish.measure_video` and batched vision reads (3.3) | `iia_annotate`: Haiku 5.5 through Message Batches |
| `distill` | Clusters annotations (code: attribute co-occurrence over high vs low traction, relative to each creator's baseline), then writes or updates pattern cards from the numbers | `iia_distill`: Sonnet, a call per cluster; Opus for the final merge |
| `synthesize` | 3-5 candidates per hypothesis from cards and named references ("comparable appeal to X" is a fine brief); then the impersonation and near-duplicate checks (code) | `iia_synthesize`: Opus |
| `audition_review` | Reads the blind scores, acceptance and cost, decides select, revise or reject, files a `launch` approval for a winner | `iia_curate` (Opus) |
| `answer` | A question from the talent manager or market lead ("her outfit posts stalled: what do similar creators do with outfit content?"). Retrieves cards and annotations first, and pulls new data only if coverage is below `intelligence.refresh.min_coverage` | Sonnet; Opus only when it pulls new data |

**Triggers** (all through `proposals.file(kind="intelligence")`, routed to `iia`; `auto_accept` within each
requester's weekly allowance and `max_usd_per_commission`; above that it escalates to the owner. While
`operations.mode` is paused only test batches run, so the pilot runs as `test-iia-<run>` with the owner's go):
- **CEO or market lead:** a new creator hypothesis or market thesis → `scout`, `challenge`, `annotate`, `distill`.
- **Creator lab before casting:** a new launch step `casting` before `bible`
  (`brain/launches.NEXT`):
  - a thesis of kind `creator_hypothesis` with cards → `synthesize` and an audition;
  - otherwise the launch proceeds as today. Off until `intelligence.casting.enabled`.
- **Talent manager on stall:** its review may file `intelligence` with a question (new allowance
  `talent_manager.intelligence_per_week: 1`). Drop events can name it in the diagnosis.
- **Refresh on evidence change, not on a calendar.** A card or creator is refreshed only when:
  - a hypothesis gets a new version;
  - a prediction citing a card resolves against it;
  - the trend watcher shows a card's attribute share moving past `intelligence.refresh.share_shift`;
  - or a card in active use is older than `intelligence.refresh.max_age_days` (90).

### 3.2 Creator discovery and scoring

Pilot sample (assumptions, set in policy, not code): **40 creators × 30 posts**, by role:

| Role | Share |
|---|---|
| Category leaders | ≤ 20% |
| Fast risers (follower growth or post velocity) | 20% |
| Mid-sized (10k-150k) | 25% |
| Adjacent aesthetic or personality archetypes | 15% |
| Low-performing comparators (same niche, bottom quartile traction against their band) | ≥ 20% |

Each creator's 30 posts include ordinary posts and relative misses, and span at least 3 months.

**Sources** (fields verified on each source's page before the run):
0. **Preferred for Instagram, once our account is connected: the Instagram Graph API's Business Discovery.**
   - Through our own connected professional account and its system-user token (`publish/instagram.py`), it reads
     another business or creator account's public profile and media by username:
     `GET /{our-ig-user-id}?fields=business_discovery.username(<handle>){username,name,biography,followers_count,media_count,media{id,caption,media_type,media_product_type,permalink,timestamp,like_count,comments_count,media_url,thumbnail_url}}`.
     Media pages through `media.after(<cursor>)`.
   - It is official, within Meta's terms, and costs nothing per call (rate limits apply).
   - Its limits: business and creator accounts only, no views, reach or saves, and `media_url` may be withheld for
     some media.
   - It answers D5 for Instagram: Apify becomes the fallback (personal accounts, video files, views), pending your
     D5 decision.
1. Candidate pool. Configured today:
   - `apify~instagram-hashtag-scraper` (input `hashtags`, `resultsLimit`; $2.70 / 1k);
   - `clockworks~tiktok-scraper` (`hashtags`; $1.70 / 1k);
   - ScrapeCreators TikTok keyword search and IG reels search ($0.002 a page).

   The outputs carry `ownerUsername` and `authorMeta.name`. For IIA pulls these are kept in clear in
   `reference_creators`. The trend watcher's own rows stay hashed.
2. Per-creator posts: **not configured today.** On Instagram, use Business Discovery (0) when connected, else add
   `research.apify_actors.instagram_profile` (Apify's Instagram scraper with profile `directUrls`, posts mode). On
   TikTok, use `clockworks~tiktok-scraper` with `profiles`. Price and
   input to check on the actor pages; the estimate assumes ~$2.70 / 1k posts. Video files: IG items carry a
   `videoUrl`; TikTok needs `shouldDownloadVideos: true`. The calibration run paid $1.17 for 84 videos including
   search.
3. Snapshots: a second pull 7 days later gives each creator's velocity (`reference_posts` snapshot history).

**Scores** (code computes 0-1 features; Opus ranks the shortlist with them):

| Score | How |
|---|---|
| Relevance | Haiku reads 3 covers plus bio against the hypothesis's dimensions (fashion, evening, humour, conversation…) |
| Distinctness | Distance of the creator's attribute distribution from their cluster's centre: a distinct creator teaches more than a typical one |
| Traction vs baseline | Per post, log engagement (views for video, likes + comments + shares over followers otherwise) over **the creator's own median** and over the median of their follower band in the niche. The creator's score is the band-relative median, with an interval |
| Contrast | Greedy: how many attribute values this creator adds to the sample's coverage |
| Consistency | Spread of their own post-to-baseline ratios, and of their style attributes over time (stable identity, varied content) |
| Commercial fit | Share of paid-partnership or #ad posts, product categories seen (apparel, beauty, accessories), owned products (a shop link in the bio) |
| Coverage and confidence | Posts annotated, share with views, time span, metrics present. Coverage below `min_coverage` is reported, never filled in |

### 3.3 The annotation schema (photo, carousel, video: one system)

**Measured by code** (`finish.measure_video` on the Fly finisher, a port of the calibration's `measure.py` with
imageio-ffmpeg; photos get resolution and slide count only):
- duration;
- hard cuts (scene score > 0.3) and soft cuts (> 0.15, punch-ins);
- shot length median, p25 and p75;
- first hard cut;
- speech share, and words per second from Whisper word timestamps (Together, $0.0015 a minute);
- music presence (spectral flatness and loudness, from Demucs-free heuristics);
- 8 sampled frames (stored in the research bucket);
- optional on-screen text OCR, off in the pilot.

**Vision and transcript read** (Haiku, batched): 8 frames, or the slides, plus the caption plus the first 30 s of
transcript. Every field is tagged as `observed` (what is in the frames or words), `mechanism` (a plausible reason it
works, marked as such) or `speculation`. Unknown is a value; nothing is invented.

| Layer | Fields (enums unless noted) |
|---|---|
| **Visual and production** | situation (dinner, street, mirror, pool, bathroom, car, desk, event…); picture archetype → nearest aesthetics shot type, or `new:<words>`; pose; hands; gaze (lens, off, down, mixed); framing; device or "who took it" (self arm, mirror, friend, propped, pro); imperfections (flash, blur, crop, grain, none); lighting (source, quality, time); wardrobe (silhouette, palette, formality); expression; face share across slides or frames |
| **Video performance** | format (GRWM, storytime selfie, vlog, day in the life, outfit check, lifestyle montage, conversational selfie, tutorial, other); eye contact share; gestures (count and kind per 10 s); body movement (still, multitasking, walking, dancing); mannerisms (hair touch, laugh, aside to camera, pause; from the mannerisms list); cadence (words/s measured, pauses); humour type; emotional delivery |
| **Video cinematography and editing** | angle; camera movement (static, handheld, follow, push); transitions (cut, match, whip, jump); hook structure (text hook, line, visual, action) in the first 2 s; cut frequency (measured); B-roll share and kind (hands, product, place); burned captions and style; music versus room sound; story structure (setup → turn → payoff, list, routine) |
| **Personality and relationship** | tone of voice; humour; disclosure style (none, everyday, vulnerable); recurring ritual or series (generic description only); relationship with the audience (friend, big sister, expert, entertainer, muse); aspirational vs relatable (1-5); parasocial hooks (questions, callbacks, "you" address); why a viewer follows (one line, `mechanism`) |
| **Evidence and economics** | post type and age at fetch; views, likes, comments, shares and saves where public; ratio to the creator's median and to their band; sponsored (yes, no, unknown); product categories; **feasibility for us**: capabilities needed (names from policy `capabilities`), shot types and moves that exist, what we can't render yet; rights: `third_party_reference` |

**Pilot sample annotations.** The brief asks for 8-12. They come from the pilot's first annotated batch (real posts),
reviewed by the owner before `distill` runs. Two fictional records show the shape (fields abridged):

```yaml
# ILLUSTRATIVE (not a real post)
- post: {media: carousel, slides: 7, age_days: 12}
  observed: {situation: dinner, shot_types: [flash_night, ringed_hand_glass, table_flatlay_hands, after_the_meal],
             device: friend, imperfections: [flash, crop], face_share: 0.29, gaze: off, wardrobe: {formality: evening}}
  mechanism: ["the friend-taken flash reads as being there, not being shown", "detail slides make the night feel lived"]
  speculation: ["saves may come from the outfit slide"]
  evidence: {likes_over_own_median: 2.4, band_ratio: 1.8, sponsored: "no", confidence: 0.7}
- post: {media: video, duration_s: 31}
  measured: {hard_shot_median_s: 2.2, first_hard_cut_s: 6.8, words_per_s: 4.1, speech_share: 0.92}
  observed: {format: grwm, eye_contact: 0.68, gestures_per_10s: 3, mannerisms: [aside_to_camera, laugh],
             broll: {share: 0.22, kinds: [hands, product]}, hook: text_and_line, captions: none}
  mechanism: ["talking while doing something makes the confession feel unplanned"]
  evidence: {views_over_own_median: 3.1, band_ratio: 2.0, sponsored: "no", confidence: 0.6}
```

### 3.4 Pattern cards: one library for photo and video

Cards **extend** the aesthetics vocabulary; they don't replace it. A card's dimensions are:
- situation × shot type (aesthetics names; a `new:` type is proposed for review);
- pose × camera (`direction.FramingSpec` words) × expression × styling × lighting;
- social energy (alone, with a friend, with a group, addressed to the viewer) × movement.

Video cards add a **sequence** (shots in order), **performance** (pace band, eye contact, gesture cadence,
deliveries from the voice delivery bank, mannerisms), and an **edit** (a `generation.talk_edit` grammar name plus
bounded overrides, a `moves.library` move, licensed motion reference moves).

Fields:
- when and why it works;
- visual description;
- audience role (attraction, intimacy, humour, aspiration, utility);
- best-fit archetypes and personas;
- recipe (shot-sheet fragments; for video, the sequence and edit);
- model and hardware constraints;
- failure modes;
- guardrails (no near-duplicates of specific posts, no impersonation);
- `sources` (reference post links);
- evidence;
- acceptance (our renders that passed review);
- outcomes (our real posts, through `evidence.py`);
- `signature_ok` (may repeat, D7).

**Retrieval** (`brain/patterns.py`, like `aesthetics.py`; the loader checks every name against the aesthetics,
moves, talk_edit and capabilities libraries):
- **Director:** `creator_brief` gets up to `intelligence.retrieval.director_cards` (4) cards. They match her
  archetypes, the post's format and her style's shot families, ranked by evidence, then acceptance. Each card
  sends its recipe, failure modes and guardrails, never the source links. The director may combine cards or ignore
  them ("freely invent"). A shot names the card it used in a short `pattern` field on `ShotSheet`: the existing
  grammar-ceiling test decides whether it fits the strict shots schema; if not, it moves to the edit pass.
- **Showrunner:** sequence and format cards (titles, mechanism, audience role, best fit, one line each) next to the
  trend brief, so ideas pick a format with a reason. The idea's `edit_grammar` can come from the card.
- **Talk production (the performance layer):**
  - the card's performance band sets the per-creator dials, within limits:
    - pace (via the delivery bank, or a light time-stretch once that is built);
    - gesture and mannerism beats in `blocking`;
    - eye contact (`eyeline`);
  - the card's grammar overrides stay inside the `talk_edit` bounds.
- **Outcomes:** when a post that used card X is accepted or dropped in review, `acceptance` updates. Its real
  Instagram outcome later updates `outcomes`. A card whose outcomes beat its creator's baseline over
  `brain.experiments` sizing becomes an experiment arm, the only way a card's effect becomes `proven`.

**Example cards: ILLUSTRATIVE until research fills them** (status `illustrative`, written from general craft
knowledge, no sources):

| Card | Medium | Dimensions | Why it works (mechanism, unproven) | Recipe (our vocabulary) | Constraints and failure modes | Guardrails |
|---|---|---|---|---|---|---|
| `dinner_flash_friend` | carousel | upscale dinner × flash_night → ringed_hand_glass → table_flatlay_hands → after_the_meal; seated sideways, knowing smile, off-centre crop; group energy | "Being there": a friend took it, the flash is unflattering in a good way, the details make the night real | 5-7 slides in the photo-dump order; face on cover plus 1; `device: held` (a friend), grade `flash_digicam` | Z-Image hands on glasses (hands shots only), background faces (keep `background_people` ≤ 2, blurred), flash on skin looking plastic | No real restaurant names or branded bottles |
| `street_style_crossing` | photo, carousel | city street × outfit_full_body / blurred_walk / shoes_cobblestones; mid-stride, gaze down or off; low angle 26 mm; alone, in motion | Silhouette and movement show the outfit; looking away reads as unposed | Cover `outfit_full_body` (low angle), then `blurred_walk`, `shoes_cobblestones`, `architecture_detail`; grade `clean_bright` or `warm_film` | Legs and feet anatomy at full body; crowds; legible signs | A signature pose is allowed (`signature_ok`), never a known creator's crossing spot |
| `mirror_phone_cover` | photo, carousel | bedroom or lift mirror × mirror_outfit_phone / elevator_mirror; phone over the face, hip out; flash in the lift | Shows the whole outfit while hiding the face: intimacy without performance; the most native "I took this" frame | `device: mirror`, `subject: reflection` (faceless, no identity gate); 1 per carousel | Phone and hand deformations; mirror double images | Our own rooms only (anchor sets) |
| `poolside_edge` | carousel | pool or sea edge × **new: `pool_edge_legs`** (legs in the water, top-down) + beach_towel_objects + hair_motion_back + drink_against_view; relaxed; alone or two | Aspiration without a face; holiday "proof" in details | Propose `pool_edge_legs` {category: scenery, face: "no", subject: feet} to the aesthetics library; grade `warm_film` | Water refraction on skin; swimwear safety gate (keep modest, test first); hands holding glasses | No real hotel; modest swimwear (scope) |
| `night_out_dump` | sequence (carousel) | getting ready → taxi_exit → elevator_mirror → flash_night → mid_laugh_candid → street_blue_hour → after_the_meal | A night told as a story with a beginning and an end; the tail frame invites "what happened" comments | The photo-dump order; at least one imperfect frame; caption with a question | The variety rules hold, except a signature outfit may recur | Never another creator's event or series name |
| `grwm_confession` | video | bathroom_grwm talking + jewellery_closeup and vanity_tray cutaways; multitasking hands; eye contact about 0.65; 3.8-4.3 words/s; 20-40 s | Conversational intimacy: she is busy, so the story feels unplanned; small mess and asides make it personal; the payoff comes as she finishes | Move `mirror_talk` or `storytime_to_camera`; grammar `story_time` (text hook from frame 1, cutaways about 25%, punch-ins every 2-4 s); deliveries `dry`, `aside`; mannerisms: one laugh, one aside to the lens | Lip sync quality on long takes (keep takes ≤ 8 s); hands applying makeup (cutaway shots only, close) | Her own story from her bible; no real person's anecdote |
| `storytime_selfie_walk` | video | street or car × selfie arm, handheld; walking; eye contact about 0.8; a fast hook line | Proximity and motion: she grabbed the phone because the story couldn't wait | Move `walking_vlog`; walk-and-talk (`generation.talk_motion`, V2V); grammar `story_time` | V2V mouth redraw (escalating repairs); motion drift past 5 s (chain ≤ 3) | As above |
| `day_in_life_montage` | video | POV hands and places, timestamps, voiceover; 1-2.5 s shots; face to camera about a third | Routine as a story: many small aspirational frames, a single thread in the voiceover | Move `pov_day_timestamps`; grammar `commercial`; `narrated_edit` | Too many face shots raises cost and identity risk; keep to a third | A fictional routine and job |
| `outfit_check_spin` | video | the same spot × full body propped; spin or hair flip at each cut; numbered labels | One frame, many looks: easy to compare, easy to save | Move `outfit_change_spin`; licensed motion references (move `outfit_pose`) only; `caption_style: none` | Outfit-change consistency (detail review); spin blur on hands | Licensed motion clips only; never a real video's choreography |

### 3.5 Two competing creator hypotheses

| | **H1: Aspirational yet relatable night-out** (the brief's) | **H2: Calm expert hands, utility first** (the contrast) |
|---|---|---|
| Statement | An original, stylish woman (mid-20s) who mixes elevated evening fashion, slightly chaotic humour and intimate day-to-day conversation can attract young adult followers and later support apparel or beauty | An original woman (mid-30s) who restyles thrifted and mended clothes, calm and deadpan, mostly hands and garments on camera, can attract followers with saves and utility, and later support haberdashery, tools and owned patterns |
| Audience | Women 20-30 in cities, going out 1-2 times a week, budget-aware but style-led | Women 25-45 interested in sustainable fashion, makers and thrifters |
| Why follow | "She's the friend whose nights out look better than mine but who tells me when they went wrong" | "I learn one fix a week and her taste makes my clothes better" |
| Tension | Glamour against mess; aspiration against honesty | Calm against cleverness; frugal against beautiful |
| Formats | Carousels (night-out dumps, outfits), GRWM confessions, storytime selfies | Before/after carousels, process montages, short voiceover explainers; face 15-25% of frames |
| Expected signals | Shares and sends on dumps; comments on stories; follow rate from GRWM | Saves on carousels; long watch time on process videos; repeat viewers |
| Disconfirming evidence | Reference creators in this niche succeed mainly through celebrity, events or brand access (not reproducible); crowding (many similar accounts with flat traction); humour posts under-perform their baselines | Utility saves don't convert to follows (high saves, flat followers); the niche rewards face-led personalities over hands; production can't render fabric and hands well enough (acceptance below 60%) |
| Commercial paths | Apparel, accessories and beauty collaborations; owned-product feasibility (a capsule, a fragrance) evaluated separately | Haberdashery, sewing machines and tools, a pattern shop (owned, digital, high margin) |
| Production fit | Face-heavy: needs identity stability and flash realism; talking video | Faceless-heavy: cheap and identity-robust; hands are the risk |

Both are researched with the same sample rules. The `challenge` step must argue against each before synthesis.

### 3.6 Worked example: H1 → 3 audition briefs → shot sheets

**References.** In the pilot the brief may name real creators outright, for example "creator X's night-out carousels
and GRWM confessions work; develop someone with comparable appeal." One creator is enough. Here the illustrative
cards stand in for them.

**Audition briefs** (codenames only; the creator lab names the winner, and the impersonation check runs on the name,
handle and, once embeddings exist, the face):

| | **C1 "Closing Shift"** | **C2 "Site Visit"** | **C3 "Second Debut"** |
|---|---|---|---|
| Who | 26, a junior sommelier at a small wine bar; goes out after the late shift | 29, an architectural model maker who reviews every room she goes out in like a site visit | 38, newly single after a long relationship, relearning how to go out |
| Audience promise | Nights out from the side of the people who work them: what's worth ordering, what's worth wearing | Taste with a ruler: rooms, light and outfits judged with dry precision | Going out at 38 without pretending to be 25: outfits and confidence for her age |
| Visual identity | Black base plus one deep red; flash at night, warm film by day; signature: a silver cuff (`signature_ok`) | Monochrome tailoring plus one loud accessory; clean bright by day, flash at night; signature: a scale ruler in her bag | Soft colour and good coats; warm film; signature: a vintage evening bag |
| Content behaviour | After-shift dinner dumps, GRWM before a shift with service stories, wine-label-free "what I'd order" carousels | Room "reviews" as carousels, outfit checks against architecture, storytime walks | GRWM confessions, "first time doing X again" dumps, outfit checks |
| Commercial thesis | Apparel and accessories; later a hospitality-themed capsule | Accessories and eyewear; stationery and design objects | Apparel and beauty for 35+; an under-served segment |
| Risk | Alcohol content (scope: keep wine incidental, no promotion) | Too clever, too cool: may not create warmth | Age band: identity stability at 38, fewer references |

**C1's 9-post mini-feed** (one audition):
1. `night_out_dump` carousel
2. `grwm_confession` talking edit
3. `street_style_crossing` carousel
4. `dinner_flash_friend` carousel
5. `mirror_phone_cover` single photo
6. "what I'd order" text-slide carousel
7. `day_in_life_montage` narrated edit (her day off)
8. `outfit_check_spin`
9. a quiet morning `hand_coffee_table` single photo (the relatable half)

**An executable photo shot sheet** (post 4, slide 1; `direction.ShotSheet` fields; `subject` and `grade` are filled
by code from `shot_type` and her style):

```json
{"shot_type": "flash_night", "pattern": "dinner_flash_friend",
 "set": {"location": "a narrow wine bar's back table after closing, chairs half stacked", "set_id": "",
         "time_of_day": "late night", "light_direction": "on-camera flash from the front",
         "light_quality": "hard", "color_temp": "warm candles, neutral flash",
         "practical_lights": "two low candles, a red exit sign far behind",
         "set_dressing": "small marble table, three mismatched glasses, a bread basket",
         "props": "plain unlabelled bottle, a folded apron on the chair", "background_people": 1,
         "depth_of_field": "deep"},
 "wardrobe": {"outfit_id": "", "outfit": "black satin slip top, oversized charcoal blazer off one shoulder",
              "accessories": "her silver cuff", "state": "a little creased after a shift"},
 "look": {"hair_state": "pulled back, a few strands loose", "makeup": "a deep red lip", "skin_state": "a little flushed"},
 "framing": {"shot_size": "medium shot", "angle": "slightly high", "lens_mm": 26, "aspect": "4:5",
             "handheld": "held", "device": "held", "headroom": "tight"},
 "blocking": [{"at_s": 0, "action": "sits sideways on the chair, elbow on the table, mid-laugh at someone off frame",
               "eyeline": "off", "hands": "one hand loosely around a glass stem"}],
 "performance": {"emotion_arc": "tired to amused", "energy": "medium",
                 "body_language": "slouched sideways, relaxed after work", "lines": []},
 "sound": {}, "edit": {"cover": true},
 "continuity": {"carry_over": "the same blazer and cuff as slides 2-6", "story_time": "after closing"}}
```

**An executable cut-based talking edit** (post 2, GRWM before a shift; the idea's `edit_grammar: story_time`;
about 22 s at about 4 words a second; only shots with her face to the lens are lip-synced, cutaways run under the
voice):

```json
{"shots": [
  {"shot_type": "bathroom_grwm", "pattern": "grwm_confession",
   "set": {"location": "her small bathroom", "set_id": "bathroom_mirror", "time_of_day": "late afternoon",
           "light_direction": "vanity bulbs above the mirror", "light_quality": "soft", "color_temp": "warm"},
   "wardrobe": {"outfit": "black work shirt, sleeves not yet rolled"},
   "look": {"hair_state": "half up, damp ends", "makeup": "base done, no lip yet"},
   "framing": {"shot_size": "medium close-up", "angle": "eye level", "device": "propped", "handheld": "none",
               "camera_move": "static"},
   "blocking": [{"at_s": 0, "action": "leans into the mirror with a lip pencil", "eyeline": "lens",
                 "hands": "one hand holding the pencil near her mouth"},
                {"at_s": 3.5, "action": "stops and turns to the lens", "eyeline": "lens"}],
   "performance": {"energy": "medium", "emotion_arc": "deadpan to conspiratorial",
                   "lines": [{"text": "A man asked me last night which wine goes with a breakup.",
                              "delivery": "dry, matter-of-fact"},
                             {"text": "I had an answer ready. That's the worrying part.",
                              "delivery": "aside, half a smile", "pause_after_s": 0.4}]},
   "sound": {"mic": "phone", "room_tone": "small bathroom, extractor fan hum"},
   "edit": {"hook": "the breakup-wine line in the first second",
            "on_screen_text": [{"kind": "hook", "text": "the questions I get at work", "start_s": 0}],
            "caption_style": "clean", "cover": true}},
  {"shot_type": "jewellery_closeup", "cutaway": true,
   "set": {"location": "her small bathroom", "set_id": "bathroom_mirror"},
   "framing": {"shot_size": "close-up", "device": "propped"},
   "blocking": [{"at_s": 0, "action": "clips on her silver cuff", "eyeline": "prop", "hands": "fastening the cuff"}],
   "performance": {"lines": [{"text": "Red if you're the one who left. White if you're pretending you're fine.",
                              "delivery": "quick, listing"}]},
   "edit": {"transition_in": "cut"}},
  {"shot_type": "bathroom_grwm",
   "set": {"location": "her small bathroom", "set_id": "bathroom_mirror"},
   "framing": {"shot_size": "medium close-up", "device": "propped", "camera_move": "static"},
   "blocking": [{"at_s": 0, "action": "finishes the lip, presses her lips together", "eyeline": "lens"}],
   "performance": {"lines": [{"text": "He ordered the white. Then he ordered another white.",
                              "delivery": "flat, eyebrows up"}]},
   "edit": {"transition_in": "cut"}},
  {"shot_type": "mirror_outfit_phone", "cutaway": true,
   "set": {"location": "her hallway mirror"},
   "framing": {"shot_size": "full body", "device": "mirror"},
   "blocking": [{"at_s": 0, "action": "grabs her bag and checks the outfit once", "eyeline": "screen"}],
   "performance": {"lines": [{"text": "Anyway. Shift starts at six. Wish me luck.", "delivery": "brisk, warm"}]},
   "edit": {"transition_in": "whip"}}
]}
```

Every name above exists today: the shot types, `set_id`, `device`, `camera_move`, `eyeline`, `on_screen_text.kind`,
`caption_style`, `transition_in`, `cutaway`. The `story_time` grammar punches in between words. The two cutaways
are narrated over their photos (`lip_sync: on_camera`). `pattern` is the one new field.

### 3.7 The A/B/C comparison

The same creator test brief (H1), the same allowance per arm, and the same outputs: **3 candidates × 9 posts per
arm**, the post mix of section 3.6, and one talking edit per arm's finalist. Everything runs in `test-iia-<run>`
batches under the test overlay. Nothing is published.

| Arm | What changes |
|---|---|
| **A: current Brain** | Thesis → creator lab → style director → showrunner → director, as today |
| **B: expanded taxonomy** | A, plus the illustrative cards and dimensions in the director, showrunner and style briefs. No research. This isolates "more vocabulary" from "evidence" |
| **C: IIA-grounded** | Researched cards and named references, candidates synthesised from them, the impersonation and near-duplicate checks, retrieval in the director, showrunner and talk |

**Blind scoring.** Each feed is rendered and finished the same way, with arm labels removed, shown in a random order
as a 9-tile grid plus the posts. Three judges:
- **the owner** (the primary judge);
- **Opus**;
- **Qwen4-Max vision** (another model family, the existing `selection.py` blind-preference method).

| Criterion | Scale | Kind |
|---|---|---|
| Distinctiveness (vs the other feeds and existing Manyworlds accounts) | 1-10 | blind taste |
| Authenticity ("real phone life") | 1-10 | blind taste |
| Visual taste | 1-10 | blind taste |
| Follow-worthiness ("after these 9, would you follow?") | 1-10 plus yes/no | blind taste |
| Identity stability | review gates' identity scores (median, worst) | objective |
| Publishable acceptance | share of posts ready on the first attempt; drops by reason | objective |
| Cost per accepted post | ledger: research + LLM + GPU + retries + QA, per arm | objective |
| Impersonation and near-duplicates vs the named references | no block; warnings listed | gate |

**Go / no-go** (set in policy `intelligence.pilot.go`, recorded with the decision):
- **Go (scale IIA to a second market):**
  - C's blind taste composite beats A's by at least 1.0 point (of 10), with the owner's own scores agreeing in
    direction;
  - C wins at least 2/3 of pairwise feed preferences against A;
  - C's acceptance is no more than 5 points below A's;
  - C's cost per accepted post (research amortised over 6 months of the creator's posts) is at most 1.25× A's;
  - no impersonation block.
- **Vocabulary only:** B is within 0.5 of C and beats A by at least 1.0. Ship the cards and dimensions; run
  research only on triggers (stalls, new hypotheses), not before each launch.
- **No-go:** C beats A by less than 0.5, or acceptance drops by more than 10 points. Archive with the reasons
  (`theses.archive_reason`), and keep the measurement code.
- **Caveat.** 9 feeds is a directional test, not proof. Every result is recorded as a `speculative` finding.
  Real proof comes later, from launched creators' outcomes by format and maturity:
  - shares, saves and reach, follow conversion, sustained growth (`predictions`);
  - kept apart from external descriptive patterns.

### 3.8 Cost estimate and caps

Prices from policy (Apify $2.70 / $1.70 per 1k; ScrapeCreators $0.002 a credit; Haiku $0.10 / $0.50 per Mtok;
Sonnet $2 / $10; Opus $4 / $20; Whisper on Together $0.0015 a minute; Fly finisher about $0.09 an hour; vast 4090 up
to $0.40 an hour). Measured references:
- calibration: $1.17 Apify for 84 videos found and downloaded, about $0.0012 of Haiku per video;
- images: about 2.5-3 s each on a 4090, with about 49% of billed image-machine time busy;
- a talent manager and two-call director run: $0.53 of Claude;
- the breadth test: $4.17.

**E** marks an estimate.

| Stage | Line | Quantity | $ |
|---|---|---|---|
| **1 Research** (no GPU) | Candidate pool: hashtag and search pulls | 800 items + 20 search pages | 1.80 |
| | Per-creator pulls (actor price to check, **E**) | 40 × 30 = 1,200 posts, + 7-day snapshot pass | 3.90 |
| | Video downloads (**E**, from the calibration's rate) | 300 videos | 1.20 (within actor runs where bundled) |
| | `finish.measure_video` (CPU on the finisher, about 30 s each) | 300 | 0.25 |
| | Whisper (Together) | 300 × ~0.6 min | 0.30 |
| | Haiku video read (8 frames + transcript) | 300 | 0.40 |
| | Haiku photo and carousel read (up to 3 images) | 900 | 0.90 |
| | Haiku relevance screen of candidate creators | 120 | 0.15 |
| | Opus: curate ×2, challenge ×2, distil merge ×2, synthesise ×2 hypotheses, revise ×2, audition review ×2, plus margin (**E** ~$0.25 a call) | 16 calls | 4.00 |
| | Sonnet: distil per cluster, answers | 20 calls | 1.00 |
| | **Stage 1 subtotal / cap** | | **13.90 / cap 16** |
| **2 Auditions** (A/B/C) | Creator lab per candidate (bible, profile, dossier, style; **E**) | 9 × $0.50 | 4.50 |
| | Showrunner, evaluator, director per post (**E**) | 81 posts × $0.04 | 3.25 |
| | Review screens and confirms (Haiku per image, Sonnet on close calls; **E**) | ~1,700 images | 3.00 |
| | Impersonation and near-duplicate checks (code) | 9 candidates | 0.00 |
| | GPU images: reference packs 9 × 12 × 4 = 432, plus posts 81 × ~4 scenes × 4 candidates ≈ 1,300; ~90 busy minutes, ~3 h billed (**E**) | ~1,730 images | 2.00 |
| | LoRA for the 3 finalists (one per arm; **E** ~35 min each on a 4090) | 3 | 0.75 |
| | Talking edit per finalist (voice design and lip sync on an L40S, plus boot share; **E**) | 3 | 2.00 |
| | Blind judging: Opus and Qwen4-Max, feed level, two passes | 9 feeds | 3.00 |
| | **Stage 2 subtotal / cap** | | **18.50 / cap 25** (GPU part ≤ $5, inside the $10 a day GPU cap) |
| **Total** | | | **~$32, cap $41** (less if Business Discovery replaces the Instagram per-creator pulls: about −$2.50) |

**Lean option:** A vs C only, 2 candidates each, no talking edit. About $8 of research (20 creators) plus about $9
of auditions ≈ **$17, cap $22**. It loses the vocabulary-vs-evidence answer (B).

Each stage needs its own owner go, as for any paid run. Enforcement:
- `intelligence.pilot.caps` per stage;
- the IIA's ledger agent `iia:<stage>`;
- the existing `budget.py` per agent, and the autoscaler's caps on GPU machines;
- a stage stops at its cap and reports what it got.

---

## 4. Implementation plan (smallest first; every task has tests; stubs only, no paid calls)

| # | Task | Tests |
|---|---|---|
| 1 | **Migration 041**: theses extension (kind, version, parent_id, owner_role, budget_usd, statuses, archive_reason); `reference_creators`, `reference_posts` (+ snapshots), `reference_annotations`, `pattern_cards`, `auditions`, `research_forget`; no grant to `mw_worker` | migration applies; the worker role can't select from the new tables; theses' old statuses still valid |
| 2 | **Policy `intelligence:`** (`enabled: false`, purpose, quotas, caps, retention, refresh, retrieval, check thresholds, the pilot's go thresholds) + roles `iia_annotate` (Haiku), `iia_distill` (Sonnet), `iia_curate` and `iia_synthesize` (Opus), added to `llm.batch.roles` where nobody waits; the policy schema check | every path the code reads exists |
| 3 | **`policy/patterns/patterns-v0.yaml` + `brain/patterns.py`**: the 9 illustrative cards; loader and validator (shot types against aesthetics, moves, talk_edit grammars, capabilities, motion references `owner` ours or licensed); `new:` shot types reported, not accepted | unknown names refused; every illustrative card loads; `status: illustrative` never counts as evidence |
| 4 | **Research separation guard**: `research_uri()` and the separate research store; refusal in `queue.enqueue` for generation and training kinds, `assets`, motion references | a research URI in any generation, LoRA or voice payload raises; production still works with ours |
| 5 | **`mw research forget <handle or url>`** + media purge sweep | forget removes rows, media keys and card sources, blocklists the handle and is idempotent; the sweep deletes past `media_days` |
| 6 | **`finish.measure_video`** (`research/video_measure.py`): the calibration's measurement as a finisher job, writing `reference_annotations.measured`; frames to the research store | an offline synthetic video (ffmpeg testsrc with known cuts and a TTS clip) measures the right cuts and pace; no network |
| 7 | **Annotation schema and batcher** (`research/annotate.py`): three layers, observed, mechanism and speculation kept apart; no fields for sensitive data; batched through `llm_batch` | a stub LLM fills it; forbidden content is scrubbed; quotes over 12 words are cut; unknown stays unknown |
| 8 | **Reference pulls and scoring** (`research/references.py`): per-creator actor inputs (policy), baselines against the creator's median and band, quotas, coverage | fixtures for both platforms; low comparators are always sampled; coverage is reported, not imputed |
| 9 | **The agent** (`brain/agents/intelligence.py`): modes, `intelligence` request kind and route, talent manager allowance, launch step `casting` (off) | stub LLM end to end on a sandbox hypothesis; rights and caps refuse over-budget commissions |
| 10 | **Impersonation checks wired in** (`brain/impersonation.py`, built in the first PR): at casting (`auditions.checks`, a block stops the launch), near-duplicates over audition renders; face vectors once the pipeline computes a face-identity embedding | warnings reach the talent manager; a block stops the launch |
| 11 | **Retrieval**: cards in `director.creator_brief`, showrunner context and talk dials; the `pattern` field under the grammar-ceiling test | the brief carries at most N cards and no source links; the schema stays under the ceiling |
| 12 | **Audition runner and report**: `mw intel audition <thesis> --arms A,B,C --usd X [--dry-run]`, the blind packet, `mw intel report <run>` | a dry run prices the plan and makes no calls |
| 13 | **Outcome loop**: acceptance from the review gates, outcomes through `evidence.py`, `research` findings, refresh triggers | acceptance counts update; simulated outcomes never touch cards |

**The first PR (built on this branch):**
- tasks **1-6**;
- the annotation schema models from task 7 (`research/annotation.py`, models only, no calls);
- the impersonation and near-duplicate checks as functions (`brain/impersonation.py`; wiring is task 10);
- the aesthetics `signature` rule (D7);
- the CLAUDE.md rule and layout lines.

It adds no agent, makes no LLM, Apify or GPU calls, and changes nothing in the trend watcher. `finish.measure_video`
needs `RESEARCH_STORAGE_URL` on the finisher (D2) before it can run for real. After it merges, the first paid step is
stage 1, with the owner's go and decisions D2, D4 and D5.
