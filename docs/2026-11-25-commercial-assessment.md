# Commercial assessment: is Manyworlds built as a talent, media and consumer business?

**Short answer:** the parts were mostly there; the framing undercut them. Today's changes fix the framing in policy, prompts and schemas, with no new agents. The biggest remaining gaps are evidence (no first-party data yet; no lawful route to visual creator analysis), not architecture.

## What already existed

- **The agency thesis** (`strategy.thesis`).
- **A casting rubric** with commercial weighted 20% and feasibility as a hard gate.
- **A capability card** giving $ per accepted post and reliability for each format. It reaches casting, the CEO and the talent manager.
- **A commercial category registry** (seed estimates).
- **A portfolio map** for the CEO.
- **Brand bible sections:** `brand_core`, `brand_world` (with personal style and signature elements) and `brand_commerce` (categories, will/won't).
- **Photo-first content guidance.**
- **Creator annotation is blind to performance.** The vision read sees the image, the caption and measured cuts. Code attaches metrics afterwards, so high and low performers are described the same way (verified in `intelligence._read_content`).

## What contradicted your thesis (fixed today)

| Where | Before | Now |
|---|---|---|
| CEO prompt | "Growth and virality come first; revenue later"; launch creators "distinct from the existing portfolio" | Reasons as a talent, media and consumer business: demand, audience value, adjacencies, cost, where value is captured; return per dollar; repeating proven archetypes is good strategy; each proposal carries its commercial case |
| Thesis | "Grow followers first" | Commercial value is the objective; audiences are the asset; influence, never advertising |
| Every strategy prompt | thesis plus desirability only | plus `strategy.economics`: the six-step value chain (talent → audience → monetisation → validated products → brands → portfolio distribution) and the principles (proven archetypes, optionality from inception, formats allocated by marginal return, photo-led by default) |
| Casting | "clearly different … from the portfolio"; distinctiveness = different | Distinct enough to own its own audience, sharing an archetype allowed. The commercial score now includes demonstrated demand, signature associations and a path to owned products. The judge's nudity rule now matches policy (it said "implied-nude") |
| Brand bible `brand_commerce` | categories, integration, will/won't | Adds a commercial thesis, 1–4 situational signature associations (e.g. vintage baseball hats: street and travel looks, never the pool or black tie), advertiser fit, monetisation path, up to 3 owned-product options each with the evidence needed first, and portfolio leverage. Old bibles still load |
| Content mix | from archetype playbooks | Allocated by expected return per dollar from the capability card; photo-led at launch; motion and talking need the archetype or evidence |
| Research | no business-model research | Weekly deep desk questions on how successful lifestyle creators monetise (sponsorship, affiliate, licensing, product lines, owned brands) and on archetype demand; the CEO now reads them |

## Answers to your questions

- **Commercially sophisticated enough today?**
  - Framing and structure: now yes.
  - Evidence: no. Every commercial number is a seed estimate until research and our own outcomes replace it.
- **Do agents reason explicitly about the economic upside?**
  - Casting and the CEO do now, with the reason written down.
  - Nothing yet computes an expected value per personality. That stays deliberate until there is data to fit it; a formula on seed estimates would be false precision.
- **Does the CEO connect audience potential, commercial opportunity and production economics?**
  - It sees all three (portfolio map, registry, capability card, desk findings) and is now told to connect them.
  - Weak link: the capability card's $ per output understates real cost about 4×, because machine overhead isn't spread over outputs (consolidation review §4). Fix that before trusting allocation.
- **Can we study successful creators' business models?**
  - Yes, through the research desk on public reporting: deals, product lines, owned brands.
  - This is factual business research and needs no third-party media.
- **Can research identify creative elements associated with stronger performance?**
  - The method is sound now: blind annotation, within-creator comparison, multiple-testing correction, at least 3 creators and 10 posts per side.
  - The input is blocked. Media analysis is paused until a lawful route exists, so annotations today come from captions only. **This is the biggest gap.**
- **Flexible about repeating proven archetypes?** Now yes. The positioning-overlap check still blocks near-clones.
- **Allocate production by expected return?**
  - The rule is set and the costs are known.
  - The returns are unknown until we publish. Experiment slots exist to measure follower gain per dollar by format once live.
- **Commercial optionality from inception?** Yes for new bibles: signature associations and owned-brand options. Maren's bible predates this and gets it at her next review.

## Creator deconstruction: the most economical technology

**Vision.**
- Claude Haiku 5.5 reads a post's 3 images, caption and cut data for about $0.0003–0.0005. The batch API halves that, but batch isn't zero-data-retention eligible.
- 10,000 posts cost about $3–5. Cost is not the constraint; data rights are.
- A Sonnet sample calibrates Haiku's reads.
- No open-weight model is needed, and they're blocked from research media until their terms are checked.

**Statistics.** Deterministic, in code:
- each post compared with its own creator's median;
- matched high/low groups;
- corrected q-values;
- creator-level floors.

Next steps when data justifies them: one comparison per creator with a sign test across creators, the same age window for every post, and per-metric-source baselines. No heavy machinery.

**Authorised sources for the visual input, ranked:**
1. **Our own posts.** The best evidence, available at launch.
2. **Creator-permissioned data APIs**, where creators connect their own accounts. This is the rights-cleanest route to both visuals and true performance data. Evaluate one: cost and coverage.
3. **Instagram Business Discovery** (official). Metadata is fine. Transient viewing of `media_url` needs counsel's view of Platform Terms 3.a.v; if cleared, it is the cheapest broad route.
4. **Brand campaign pages and creators' own media kits**, read by the desk with robots.txt honoured, for commercial and creative direction.
5. **Licensed analytics vendors** (audience and performance data, not media). Useful for archetype demand; check each vendor's terms.

Not used: scraping platform media (platform terms) or keeping archives (data policy).

## Challenges to your assumptions

1. **"20–50 personalities in one archetype."** Two risks:
   - *Platforms.* Many similar accounts run by one operator can read as a coordinated network, even with no cross-engagement. We already forbid cross-engagement.
   - *Cannibalisation.* Similar personalities compete for the same followers and sponsors.

   Scale an archetype in stages (1 → 3 → 8), measure audience overlap and per-account growth at each stage, and get counsel's read of Meta's inauthentic-behaviour policy for openly labelled AI talent.
2. **Photo-first vs video is the wrong split.** Instagram discovery is mostly Reels, so photo-only posts mostly reach existing followers.
   - The cheap middle is the photo Reel (a narrated edit or photo dump with music): photo cost, Reel distribution, already proven in production.
   - The real allocation question is photo Reel vs expensive talking video, and talking video must earn its place.
   - Launch mix proposal: photos and carousels for the brand, photo Reels for discovery, almost no talking video.
3. **AI talent can't give testimonials.** "Never claims first-hand experience" limits classic sponsorship. These fit better:
   - styling and placement;
   - licensing the persona;
   - merchandise and owned products where the persona is the brand.

   The monetisation path in each bible now says this. Commercial demand for virtual talent specifically is the key unknown; the desk's business-model topic should track it.
4. **Owned brands need validated demand.** The sequence in each bible requires evidence (affiliate clicks, saves on association posts, partner interest) before an owned product. That is right, and it means affiliate links and tracking are the first commerce infrastructure needed after launch, not a brand launch.

## Remaining gaps, by priority

1. **The lawful visual-input route:** counsel on Business Discovery transient viewing, or a creator-permissioned data API. Without it, creative pattern discovery stays caption-only.
2. **True cost per accepted output.** Spread machine overhead; review §4. Without it, allocation decisions run on numbers about 4× too low.
3. **First-party measurement design** for launch: follower gain per dollar by format, saves and shares on association posts, affiliate click-through. Built partly (experiments, metrics); needs the commerce signals.
4. **Commercial registry from research, not seeds:** the desk's commerce questions feed the registry (a small step).
5. **Maren's bible** regenerated with the new `brand_commerce` fields at her next review.
