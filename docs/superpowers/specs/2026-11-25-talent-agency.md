# The talent agency: desirable lifestyle talent, cast and run as IP (2026-11-25)

Owner refinement, 2026-11-25: the proof of concept is a portfolio of highly desirable, aspirational, commercially
valuable virtual **lifestyle** personalities: an AI-native modeling agency, talent development company and lifestyle
media network that owns its talent IP. Desirability in many forms, across ages, demographics and cultures (a
sophisticated Italian gentleman, an ambitious metropolitan professional, a rugged outdoorsman, an effortlessly
elegant Parisian woman, a coastal free spirit, a distinguished older gentleman, a creative director, a stylish
entrepreneur: examples, not a taxonomy). Not educational, niche or news creators for now; the architecture stays
open beyond lifestyle. Fully autonomous (no human gates), few new agents, low model-call cost.

This spec records what exists, what was missing, what this branch builds and the lead's challenges. Nothing here has
run on a paid LLM, Apify or a GPU.

## 1. What exists and what was missing

| Need | Existed | Gap | Built here |
|---|---|---|---|
| A portfolio thesis and casting criteria | `scope.niche` (one line), strategist SYSTEM ("growth first") | no thesis, no rubric, nothing about desirability, commercial value or feasibility | policy `strategy` (thesis, focus, rubric A-E with weights, floors, feasibility hard gate, desirability, presentation registers, platform guidelines) |
| What production does reliably | `deploy/report.throughput` (per GPU family), `scoring.py` (per pool), pattern-card acceptance | nothing per *format* that a casting or a career can plan against | `brain/capability_card.py`: per format acceptance and $ per accepted post from posts and the ledger, policy defaults until `min_tried`; stored in the existing `capabilities` table |
| A brand bible | dossier: archetypes, positioning, audience insight, story, content system; style director: style, world, season | nothing on what she represents, why the value compounds, values and ethos, environments and hobbies as a brand, visual language, audience relationship, commercial categories, will/won't do, wear, promote | three new dossier sections, one checked call each: `brand_core`, `brand_world`, `brand_commerce`; `brand_core` is the stable core (changes only with real evidence, code-checked) |
| Casting | one thesis → one bible → one creator (`launches`: bible first) | no candidates, no comparison, no audition before the LoRA; the `auditions` table (migration 041) had no code | `brain/casting.py`: 3-5 concepts in one call, one judge call (rubric A-E), code floors and gates, top-k auditioned as a 9-photo mini-feed through normal production in a test batch, winner picked by rubric × acceptance × cost; a launch step `casting` before `bible` |
| Commercial intelligence | `research/annotation.py` reads `sponsored` on reference posts | no categories, no market or advertiser view | table `commercial_categories` seeded from policy (marked `estimate`), `brain/commerce.py`, an IIA mode `commerce` (sponsored share in category hashtags; off) |
| Portfolio view | `portfolio.snapshot` (numbers per creator) | no coverage, overlap or gap view across archetypes, audiences and categories | `brain/portfolio_map.py` (code), in the CEO's prompt and `mw brain portfolio-map` |
| Learning signals | `metric_snapshots` with reach, views, likes, comments, saves, shares, follows | no profile visits, no account-level follower growth or non-follower reach, no attribution by aesthetic, format, pattern card | `profile_visits` per post where Meta serves it; `account_insights` (daily follower count, follower/non-follower reach, profile views); `evidence.growth` and `evidence.attribution` |

## 2. Decisions

1. **Strategy lives in policy** (`strategy`), versioned with every decision like the rest. `strategy.focus` names the
   vertical casting favours (`lifestyle`); `strategy.verticals` keeps the others (education, news: deferred; music:
   its own department) so a later thesis can widen it without code.
2. **Casting rubric** (`strategy.rubric`): A desirability 0.30, B audience surface area 0.20, C commercial viability
   0.20, D production feasibility 0.15, E distinctiveness and longevity 0.15. Floors per criterion; D is also a
   **hard gate** whose score is the lower of the judge's and code's (the content mix weighted by the capability
   card's reliability). C is the mean of the judge's and the registry's index of the concept's categories. A concept
   whose positioning words overlap a sibling past `brain.talent_manager.max_overlap` has E capped.
3. **Desirability is an affirmative objective** (owner clarification 2026-11-25, policy `strategy.desirability`):
   casting, talent development and creative direction optimise attractiveness, magnetism, elegance, charm, style,
   charisma, confidence and romantic appeal, expressed per personality, to the standard of Vogue, GQ, premium fashion,
   luxury travel and swimwear campaigns. Judged in context, never by clothing alone: swimwear, shirtlessness,
   cleavage, revealing fashion, fitness photography and tasteful sensuality pass. Restricted, and only these: explicit
   sexual content, nudity or implied nudity, fetish content, sexual solicitation, simulated romantic or intimate
   relationships with followers (comment replies included), gratification-first content, anyone who could read as
   under age. Platform and partner rules are per-platform and per-partner lists (`strategy.platform_guidelines`,
   `partner_requirements`), not one blanket rule.
   - **Per-creator register**: `brand_core.presentation` (register from `strategy.presentation.registers`, how the
     attractiveness shows, wardrobe range, photographic style, never-list); showrunner, director and style director
     stay within it and live up to it (no sterile default).
   - **Anti-drift**: the learning loop never raises a register on engagement. A review's change to the register needs
     the brand core's evidence gate, a stated brand fit, and moves at most `brand.presentation.max_step` level
     (`talent_manager.presentation_gate`); pattern-card outcomes and attribution never touch it.
   - **Reviewed rules** (fixed): `creator/lab.py` SYSTEM ("never sexualize" → affirmative, never explicit); CLAUDE.md
     guardrails; `scope.excluded_topics` ("sexualized content" → the restrictions); the image and clip safety gate
     (`review/gates.py`, `review/video.py`: the `sexualized` flag now carries `review.common.SAFETY_RULE`, which keeps
     explicit and staged-nude content (superseded in section 6: implied nudity is judged on visible content only) and passes swimwear, shirtless and cleavage in context;
     the judge prompts say safety is contextual); `creator_lab.persona_guide`; `dossier` SYSTEM; casting prompts;
     `publish/comments.py` (never flirt back as a partner or role-play intimacy). No modesty-forcing negative prompts
     existed (`worker/comfy.py` NEGATIVE, `direction.negatives` checked); the aesthetics library and archetypes had no
     modesty rule. `review/final.py` and `review/detail.py` have no safety gate. Model calls stay flat: the definition
     rides in the existing `sexualized` field.
4. **Capability card** refreshed by code each chief-of-staff tick (no model call): measured acceptance once a format
   has `min_tried` judged posts in the window, else the policy default. Defaults: photo and carousel reliable; Story
   reliable; photo-dump Reel (narrated edit with music) reliable; simple motion (story clip; LTX-2.5 image-to-video)
   promising; talking video unproven.
5. **Brand bible**: three sections added to the dossier, built like the others (one structured call each, cleaned
   by code, versioned as profile changes). Showrunner, director, style director and the talent manager's review all
   read it. `brand_core` is the stable core: the talent manager may change it only when the creator has at least
   `brain.talent_manager.brand.core_change_min_real_posts` real published posts and its last change is older than
   `core_change_days`; otherwise it is refused (and recorded), never escalated to a human.
6. **Casting** is code around three model calls (concepts, judge, a bible per auditioned candidate) plus the anchor
   pick's vision calls; the renders go through normal production in `test-casting-<run>` batches (test batches run
   while operations are paused) and are excluded from the producer and publisher like breadth posts
   (`body.breadth_run`). Winner = rubric × acceptance^w × (reference cost / cost per accepted)^w, over a minimum
   acceptance. The winning audition creator *becomes* the launched creator (its bible, its brand core); the launch
   continues at `references` (full reference pack, LoRA, voice, profile, dossier, style). Losers are retired.
   Respects `brain.strategist.launches_per_week = 0` (no launch, no casting from the CEO) and runs as a sandbox test
   with `mw brain casting <thesis>` (`--dry-run`: the estimate, nothing created).
7. **Commercial intelligence is a capability, not an agent**: a table, a module, and a mode of the IIA that already
   exists. Read at casting, by the talent manager, by the CEO.
8. **Portfolio map** is code: archetypes × audience cells × commercial categories, overlaps and gaps ranked by the
   registry's commercial index × production fit, in the CEO's prompt.
9. **Metrics**: only what the Graph API serves for free; nothing simulated reaches a real read (`evidence.py`).

## 3. The lead's challenges

- **Photo-only vs Reels.** Instagram discovery is Reels-heavy; a still-only account reaches mostly its followers.
  The content guidance (`strategy.content`) keeps photography as the core (where production is strongest and
  cheapest) and adds a cheap, reliable **discovery layer**: photo-dump Reels with music (narrated edit, reliable)
  and short image-to-video moments (story clip, promising), a minority of posts. Talking video stays out of casting
  mixes until it is reliable.
- **Desirability vs drift.** Attractiveness is the product, so the risk is a learning loop that rewards ever more
  provocative posts. Bounded per creator by the presentation register and the anti-drift gate; the restrictions are
  narrow and contextual so the opposite failure (sterile, modest-by-default talent) is avoided too.
- **Broad vs bland.** Surface area rewards broad appeal; broad appeal drifts to generic. The rubric pairs B with E
  (distinctiveness and longevity), the judge is asked for the "generic risk" of each concept, and the persona guide
  keeps "never generic". A concept needs a memorable signature, not just a demographic.
- **Commercial precedent for virtual talent.** Virtual creators have signed fashion, beauty, tech and auto deals, but
  advertiser comfort varies by category. The registry keeps `virtual_precedent` per category as an estimate (none,
  some, strong) until research replaces it; commercial scores weigh it. No real brands appear until rights exist
  (scope), and no sponsored post ever claims first-hand experience.
- **Capabilities, not agents.** The owner asked not to make each suggestion an agent. Built: no new agent. Casting
  extends the creator lab; commerce is a table plus an IIA mode; the capability card, portfolio map and attribution
  are code.

## 4. Cost per cycle (estimates; nothing run)

| Cycle | Calls | Estimate |
|---|---|---|
| Casting (once per launch) | concepts 1 (Opus), judge 1 (Sonnet), audition bible 1 per auditioned candidate (Opus; top 2), anchor picks 6 per candidate (vision) | ~$0.70 LLM, plus ~$0.30-0.50 GPU for 2 × (12 reference images + 9 posts × 4 candidates) and their Haiku image screens |
| Talent manager onboard | +3 section calls once per creator (Sonnet) | ~$0.10 once |
| Talent manager review | no new call; the prompt grows by the brand bible, her categories, the capability card and attribution | ~+$0.005 a review |
| Strategist | no new call; the prompt grows by the portfolio map and the strategy | ~+$0.01 a day |
| Capability card, portfolio map, commerce seed | code | $0 |
| IIA `commerce` (off) | no LLM; one hashtag pull per category | ~$0.15 per category at 50 items (Apify) |

## 5. What remains

- The face-identity check in casting (no embedding is computed yet; `impersonation.faces` reports "not run").
- Real calibration of the rubric: the judge's scores against real outcomes (predictions on casting decisions).
- Commerce research beyond sponsored share (advertiser lists, CPMs) and exact-product integration (licensed product
  imagery, disclosure flags per post) once a brand deal exists.
- Non-follower reach per post: Meta serves follower/non-follower only at account level; per-post follows and profile
  visits only for feed posts and Stories.

## 6. Owner clarifications, second round (2026-11-25)

- **Implied nudity is judged on visible content**, context, composition and intent, never on speculation about what
  is under clothing (`review.common.SAFETY_RULE`): an oversized shirt, a hotel bathrobe, off-the-shoulder or backless
  clothes and a cardigan worn as a top pass when covered; only a deliberately staged nudity simulation or explicit
  content is rejected. The earlier "appears to wear nothing under the cardigan" reasoning no longer rejects a covered
  image. Policy `strategy.desirability.restrictions` and `scope.excluded_topics` say the same.
- **Social charm in replies** (`publish/comments.py`): warmly acknowledge compliments, joke and banter with personality
  and confidence, never cold or robotic; never a personal romantic relationship or sexual exchange with a follower.
- **Personal presentation is identity**: `brand_world.personal_style` (finish, grooming, hair and makeup, nails,
  jewelry and accessories, eyewear, footwear, silhouettes/materials/colours, possessions, seasonal and occasion dress,
  2-4 signature elements that recur subtly, what varies) and `brand_world.world_details` (objects that show taste).
  The aesthetics library gains rules `personal_style` and `details` (the style director reads them). The director's
  brief carries `dossier.styling_line` so shot sheets fill `wardrobe.accessories`, `look` and props from it; the image
  prompt budget is unchanged (`direction._fit` trims styling after face, set and wardrobe).
- **Commercial adjacency**: `strategy.commercial.adjacency` maps words in her style and world to registry categories
  (a manicure to beauty, a necklace to jewelry, a briefcase to leather goods, interiors to homeware;
  `commerce.adjacent`), shown to the brand-commerce section and the talent manager's review.
- **QA, three judgments in the existing call** (`review/gates.py`, `review/video.py`, `review/common.py`): content
  safety (hard, the `sexualized` flag); brand alignment against the creator's presentation register (0-10; rejects only
  under `review.image.min_brand_alignment`, a clear misfit); distribution risk with a label (a report row, always
  passes, for routing and boosting). Styling detail defects (malformed jewelry, unnatural nails, inconsistent
  accessories, grooming artifacts, distorted hands, implausible clothing) come back with a severity: `major` fails
  realism, `minor` is reported. Clips carry the same fields as reports. No new model call; the presentation adds one
  short text block to the image screen.
- **The range test**: `tests/test_talent_agency.py` checks, with stubs, that no prompt, policy text or negative prompt
  (casting, lab, dossier, persona guide, style rules, showrunner, director brief, ComfyUI and policy negatives, review,
  comments, excluded topics) carries a contradictory restriction, that twelve mainstream cases (resort swimwear,
  backless and low-cut evening wear, activewear, shirtless male beach and fitness, glamour editorial, magnetic poses,
  bathrobe mornings, a covered underwear campaign, fragrance and jewelry ads, manicure and grooming, styled jewelry, an
  oversized shirt) pass the review end to end, and that explicit, staged-nude, fetish, soliciting and minor-looking
  content is rejected. Breadth plan `range` (policy/test_overlay.yaml: 12 photos, Maren plus a male test creator
  `range.marco`, max $2) has a dry-run test; it has not run on a GPU.

Model calls per cycle are unchanged by this round: the brand and distribution judgments ride in the existing image
screen and confirm calls; personal style is part of the existing `brand_world` section call.
