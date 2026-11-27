# The research operation today: jobs, dataset and data lifecycle (2026-11-27)

This is a factual inventory, taken from the code, the policy and the production database on 2026-11-27. "Running" means it runs on its own now (operations are paused; `operations.while_paused` lists what may run anyway). "Built" means the code exists and is tested but nothing schedules it yet. "Demonstrated" means it has produced real output.

## 1. The jobs, one by one

| Job | Trigger and frequency | What it answers | Sources and models | Produces, and where it goes | Status | Cost so far |
|---|---|---|---|---|---|---|
| **Discovery** (`brain.scouts`, `brain.discover`) | Every 7 days, live operations only | Who new is worth watching | Hashtag pools (Apify) | New reference creators | Built; **not running while paused** | — |
| **Scout / pilot** (`brain.intelligence` mode pilot, `mw research pilot`) | Manual | Build a creator sample for a hypothesis: leaders, risers, mid-sized, low comparators | Apify profile and post scrapers; Opus for the plan | `reference_creators`, `reference_posts`, research copies of media | Demonstrated once (2026-11-25): it built today's 54 creators | $15.64 |
| **Monitor** (mode monitor) | Daily; runs while paused | What each creator posted since we last looked, and which posts deserve reading | Apify or ScrapeCreators profile pages | New posts, metric snapshots, `selected` posts with reasons, research copies for selected posts only | **Running** (hot 5 creators checked daily, warm 32 weekly, cold 16 monthly, dormant 1) | $1.43 in 30 days |
| **Trend watcher** (`trends.py`) | Hourly when on | Anonymous niche trends | Apify hashtag items | `research_items` (360 rows, hashed handles, 30-day purge) | **Off** (`trends.enabled: false`) | — |
| **Visual annotation** (annotate / resume) | After a pilot, or by hand | Code each post in the 33-attribute vocabulary (iia_v2), with mechanism notes | Haiku on the post's research images: a photo, carousel slides, or a video's cover image or 8-frame grid; plus caption and measures | `reference_annotations`, mirrored to `attribute_values` | **Running now** (the first full pass): 204 posts coded | ~$0.25 |
| **Video measurement** (`finish.measure_video`) | Queued by annotation for videos whose file we hold | Cuts, shot lengths, pace, first word, loudness, silence (v2) | ffmpeg on the Fly finisher (CPU) | `reference_annotations` (schema measure_v1/v2), frames and a grid in the research store | Demonstrated on 40 videos | ~$0 |
| **Native audiovisual read** (Pegasus) | By hand | Beats, delivery, hook, naturalness, lip sync, defects | Twelve Labs Pegasus 1.6 | `beat` units under the clip in the spine | Built; **off** (needs your account and the training opt-out); our own videos only | — |
| **Creator propositions** (`mw research propositions`) | By hand | Why people follow each creator; their verticals | Haiku on a text digest: bio plus our own coded reads (no media sent) | `proposition` and `vertical` values on creator units | Built; **not yet run** | est. $0.05 for 54 creators |
| **Quick and deep web research** (research desk) | Watched topics, each every 3–7 days; runs while paused; also on demand | What is moving in fashion, beauty, interiors, travel, wellness, platforms, campaigns | Claude web search: Haiku quick, Sonnet deep with page reads | `research_findings` (claims, sources, fresh-until dates) | **Running**: 9 findings | $0.76 |
| **Quantitative research** (research questions) | Questions filed by code (gaps, after each spine sync) or by agents; answered once a day while paused (new) | Pre-registered tests, and exploratory looks | Code (`recipes`, `inquiry`); Haiku only to turn a question in words into a spec | Answers on `research_questions`; trial ledger in `analysis_runs`; confirmed effects become archetype hypotheses | Built; scheduled from today; **0 questions answered so far** | — |
| **Distillation and pattern cards** | After annotation, inside a pilot | Cluster coded posts into patterns production can render | Sonnet drafts cards; code validates against the shot and move libraries | `pattern_cards` | Built; **0 cards in the table** (earlier drafts never passed validation, or were never stored) | — |
| **Gold sets and agreement** (`mw research gold`) | By hand; then on every new vocabulary or model | How far each attribute can be trusted | A repeat read plus Opus and Sonnet reads of the same third-party posts | Agreement report (`analysis_runs`, method gold_agreement); unreliable attributes leave analysis | Built; **the first real run waits for annotation** | ≤ $5 approved |
| **Spine sync** | Every 6 h while paused (new) | Mirror research and production tables into one analytical shape | Code | `content_units`, `attribute_values`, `outcomes` (raw and normalised) | Demonstrated by hand today | $0 |
| **Live verification** (`mw verify`) | Daily while paused (new) | Did every stage really do its work | Code | Failing checks reach your inbox once a day | Built today | $0 |
| **Hypotheses and experiments** | Findings → archetype hypotheses → experiments for comparable creators | Is a pattern worth testing on our own creator | Code | `findings`, `experiments`, showrunner experiment slots | Demonstrated only in the sandbox simulation | — |

**How the intelligence is consumed today:**
- The desk's findings reach the style director, casting and the showrunner (`desk.brief`).
- Answered questions reach any agent that asked, or whose name is in the question's `uses` (`research_questions.brief`).
- Pattern cards reach the showrunner, the director, the finish pass and casting, but there are none yet.
- Hypotheses reach the showrunner's experiment slots.

**How each job learns to do better:**
- The spend checks (each job's cap and the monthly ceiling).
- Shadow tests of the annotation, distillation and question roles against cheaper models.
- The gold sets (attribute reliability).
- The trial ledger (multiple-testing honesty).
- What doesn't exist yet: a feedback loop that measures which sources and methods produce useful intelligence per dollar. That is the Phase 2 research bandit.

## 2. The dataset: what the 1,596 posts and 54 creators are

**How the creators were found.**
- Today's universe came from one scout pilot on 2026-11-25, run for a hypothesis about lifestyle creators.
- The pilot drew candidate pools from hashtags, then filled role quotas:

  | Role | Creators |
  |---|---|
  | Low comparators (bottom of engagement for their size, sampled on purpose) | 26 |
  | Leaders (most followers) | 15 |
  | Mid-sized | 11 |
  | Other | 2 |

- **Provenance gap:** `found_via` is empty for every creator, so we can't say which hashtag or query found whom. Fixing that is on the verification list.

**What they are.**

| Item | Value |
|---|---|
| Platforms | 51 TikTok, 3 Instagram |
| Followers | 600 to 30.3 million (median 323,000) |
| Bios held | 52 |
| Verticals and archetypes coded | none yet (the propositions read will add them) |

**What the posts are.**
- Each creator's most recent posts, about 30 each (3 to 35). So they are available recent posts, not chosen winners.
- Selection for deeper reading came later: the monitor's rules plus the new random slice.

| Platform and type | Posts | Date range |
|---|---|---|
| TikTok video | 1,477 | 2021-12 to 2026-11 |
| TikTok carousel | 71 | — |
| Instagram carousel | 27 | 2026-08 to 2026-11 |
| Instagram video | 17 | 2026-08 to 2026-11 |
| Instagram photo | 4 | 2026-08 to 2026-11 |

The TikTok date range is that long because slow posters' last 30 posts reach back years.

**What we hold per post.**

| Field | Coverage |
|---|---|
| Caption | 1,595 posts |
| Caption features (hashtags, mentions, partnership flags) | 1,596 posts |
| Public counts at fetch time | All posts; TikTok gives views, likes, comments, shares, saves; Instagram only likes and comments (and views on 7 Reels) |
| Metric snapshots over time | 1 snapshot: 766 posts; 2: 489; 3 or more: 341 |
| Research copies of media | 1,523 posts, 533 MB, until 2026-12-09 |

The research copies are:
- 1,576 images: photos, carousel slides, and **TikTok video covers**;
- 33 video files;
- 264 frames and 33 frame grids, for the measured videos.

**What has actually been analysed.**

| Analysis | Posts |
|---|---|
| Visual read (iia_v2) | 204, rising (plus 28 on the older iia_v1) |
| Video measured (cuts, pace, audio) | 40 |
| Native audiovisual | 0 |

**Important:** for nearly all of the 1,477 TikTok videos, the visual read sees **one cover image plus the caption**, not the video. That supports findings about covers, styling, settings, composition and captions. It says little about motion, performance, editing or sound. Real video analysis needs the video file, then measurement and frame grids (or Pegasus later). That costs about $0.0013 per video download plus Haiku.

**The outcomes.**
- **25,846 outcome rows** are raw public counts: views, likes, comments, shares, saves, plus their sum, engagement.
  - Each sits at the post's age when observed, banded into horizons: about 1, 3, 7 and 30 days, and mature.
  - All of them are directly observed counts.
- **4,946 rows also carry a normalised value**: the log of the count over the same creator's median at the same horizon, over at least 5 of their other posts in the 180 days before.
  - That value is calculated, not observed.
  - Most posts have only one snapshot, so a post's "horizon" is mostly the age at which we fetched it. The 1-day and 3-day horizons are thin (15 and 81 normalised rows).

**What can, and can't, be concluded.**
- **Can:** within-creator comparisons of covers and captions on TikTok lifestyle creators, as correlation, with creator holdouts.
- **Can't:**
  - anything about Instagram (3 creators);
  - anything about video craft (covers only);
  - causation;
  - anything about creators unlike these 54;
  - trends over time, since there is one fetch window.
- Every finding is stored with its context and sample size. Proposition × vertical findings need hundreds of creators, not 54.

## 3. Data lifecycle inventory

| What | How we get it | Where it lives | How long | Deletion |
|---|---|---|---|---|
| Creator profile facts (handle, bio, follower counts, history) | Apify, ScrapeCreators, logged-out public pages | Postgres (`reference_creators`) | 730 days if cited, else purged | `mw research forget <handle>` on request; `mw research purge` |
| Post metadata and public counts | Same | Postgres (`reference_posts`, `outcomes`) | Green: kept while useful | Same |
| Caption text | Same | Postgres | 30 days after its last fetch once read (`caption_features` stay) | Purge job |
| Third-party images and videos | Downloaded once, only for selected posts | Object storage under `research-refs/` (main bucket) | 24 h after it is read; 14 days at most if never read | Purge job deletes the objects and clears `media`; verifiable with a HEAD request (404) |
| Video files for measurement | Same | Copied to the Fly finisher's temp directory, deleted after the job | Minutes | Job cleanup |
| Media sent to model providers | The vision read sends images to Anthropic only (`data_policy.processors: [claude-]`) | Anthropic API | Per Anthropic's commercial API terms (we don't hold a zero-retention agreement) | Not verifiable by us |
| Scraper-side copies | Apify run datasets and key-value stores; ScrapeCreators responses | The provider's systems | **Apify keeps run datasets for its retention period (plan-dependent; we never delete them)** | **Not done today: see below** |
| Derived observations (coded attributes, measures, findings, cards) | Our analysis | Postgres | Kept (green: our own observations) | `forget` removes a creator's |
| Embeddings | None of third-party media | — | — | — |
| Trend items | Apify hashtag items, handles hashed | `research_items` | 30 days | Purge |
| Database backups | Neon point-in-time history | Neon | Neon's history window (deleted rows persist that long) | Ages out |

**Needs attention:**
1. **Apify datasets.** Our runs leave copies of scraped items, including caption text and media URLs, in Apify storage until their retention lapses. We should delete each run's dataset once ingested (a call to Apify's dataset API). This is a small change; I'll make it next.
2. **The research store sits in the main bucket** under `research-refs/`, because `RESEARCH_STORAGE_URL` isn't set. A separate bucket is cleaner, so research media can never sit beside production assets. This is optional, and needs an owner action (a new bucket and its credentials).
3. **Anthropic retention** for images we send. It's acceptable under the processor list, but worth knowing. Zero data retention would need an agreement with Anthropic.
4. **The 533 MB of held research media**: read copies purge automatically 24 hours after each read, and the rest on 2026-12-09.

**Source permissions:**
- **Established:** our own data; Claude web search and fetch (with robots.txt); public profile and post metadata through Apify and ScrapeCreators, which the data policy approves as amber (logged-out, metrics retained, media transient).
- **Uncertain or pending:** Instagram Business Discovery uses (waiting on counsel, `data_policy.sources.graph.pending_counsel`); TikTok Creative Center (no API; the terms forbid automation, so hand use only).
- **Restricted:** Meta Content Library and the TikTok Research API (non-commercial researchers only); any face or voice templates of real people; comment text; training on third-party media; raw embeddings of third-party media.

## 4. Gaps this inventory found
- `found_via` is missing for every creator: discovery provenance isn't recorded.
- There are no pattern cards, although distillation ran in the pilot.
- The chief of staff's research steps never ran while paused (fixed today: the spine sync, gaps, questions and checks now run while paused).
- Two `brain.music_sandbox` jobs show as "running". To check.
- Apify datasets are never deleted (next change).
