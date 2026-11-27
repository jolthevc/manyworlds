# Refocusing on original creation and first-party learning: assessment (2026-11-27)

This is the assessment the owner asked for before any substantial new development. Third-party creator research has been discontinued and its data deleted (`docs/2026-11-27-third-party-research-discontinued.md`). It covers:
- what to keep;
- where we are rigid;
- the gap between what our agents can conceive and what production can execute;
- web research;
- first-party analytics;
- what we lose;
- revised priorities.

## 1. Retain, repurpose, simplify, discontinue

| Component | Decision | Why |
|---|---|---|
| Analytical spine (units, long-form attributes, outcomes, relations, trial ledger) | **Retain** | Source-agnostic. Own units already carry intent (the shot sheet), execution (reads), cost and, later, outcomes |
| Attribute registry and vocabularies | **Retain, extend** | Add the own-content attributes we control (shot type, performance controls, model and route) as registry entries, so intent and observation are comparable |
| Gold sets and agreement | **Repurpose** | Measure annotator reliability on our own finished content. Its purpose is the instrument's accuracy, not judging our creators, so it doesn't conflict with "our quality is unknown" |
| Recipes, inquiry, research questions | **Repurpose** | Point the loaders at own units and their first-party outcomes (today they read reference posts, now empty). The tests and methods are unchanged |
| Hypotheses, experiments, exploration temperature | **Retain** | Already first-party in design; they only wait for outcomes |
| Archetypes, propositions, format fit | **Retain** | Strategy concepts, independent of external data |
| Pattern cards | **Repurpose** | Today: illustrative seed cards only, none learned. Becomes the place where proven *own* techniques are kept (your "own reference universe") |
| Intelligence agent (scout, monitor, annotate, distill), providers, pulls, video_measure for references | **Dormant** | Sealed, kept and tested. Removing them would be work without benefit. `video_measure` is reusable on our own videos |
| Research desk (web) | **Retain, strengthen** | Now the main window on the world (§4) |
| Casting cards from patterns | **Simplify** | They drew on external-derived cards. Casting relies on desk findings, the proposition library and the capability card |

**Dependencies on external data that are now empty, and must not silently produce nothing:**
- recipes and gaps, which read reference posts;
- the showrunner's pattern-card offer, which falls back to the illustrative seed cards;
- casting cards.

`mw verify` should flag analyses that run on zero rows; that check comes next.

## 2. Creative search space: how free are we really?

**What exists.** A wide vocabulary of creative dimensions:
- archetypes and propositions;
- dossiers (story, world, idea engines, series, arcs);
- the style director (vibe, palette, grade, wardrobe rules, signature items, anchor sets, monthly seasons);
- the aesthetics library (44 shot types, many without the face);
- performance direction (14 controls at five levels, with context shifts and signatures);
- the showrunner's stagings, moves and flavours;
- the exploration temperature (refine, adjacent, novel);
- format fit.

The agents can conceive a great deal.

**Where it is rigid or generic** (from code and the 2026-11-26 experiments):
1. **Words don't move poses.**
   - In the photo experiment the seed fixed pose and framing; only facial expression responded to direction.
   - Candidness and warmth didn't change at all.
   - So variety and expressive range are capped by the image model's response to text. This is the largest single gap between what we conceive and what we execute.
2. **Prompt budgets flatten intent.** The photo prompt is trimmed to about 440 words in a fixed order. Rich direction (set, wardrobe, performance, camera) loses clauses. On 2026-11-26 the trimming silently dropped the performance section, then the set.
3. **Negative prompts do nothing on Z-Image** (guidance 0). "Not stiff" has to be phrased as a positive behaviour. The compiler does this, but a generic-AI look still comes from defaults we can't steer by negation.
4. **One identity route (LoRA) and one photo model.** Identity is strong, but the range of looks within it (lighting styles, lenses, film stocks, candid imperfection) depends on one model's priors.
5. **Video motion comes from source motion, not direction.** In walk-and-talk, gestures and body language come mostly from the Wan motion clip. Directing them means directing that clip: its blocking, then pose or trajectory conditioning.
6. **Defaults can win over intent.** Several controls default to "neutral, level 2". Only salient controls are compiled, so subtle intent can vanish. That is sensible for prompt length, but worth measuring.

**How much better prompts and orchestration can buy:**
- Material gains without new models:
  - varying blocking and composition per candidate, not only seeds;
  - giving each section a minimum budget so none is starved;
  - compiling the model's known-good phrasings (learned from first-party ablations);
  - letting the director choose framing and lens deliberately per shot.
- The step change in executing intent comes from **structural conditioning**: pose or skeleton references, depth or composition control, and first and last frames for video. Our own photography can provide all of it (point 6 of your brief).
- This is where the GPU development envelope should go first.

## 3. Our own creative reference universe

It mostly exists already, distributed:
- accepted assets with their full recipes;
- the review rows and performance observations of each asset;
- set and wardrobe plates;
- the voice casts;
- the capability card's measured acceptance.

What's missing is one thing. A **"proven" flag with a reason** on an asset or recipe:
- the director, style director or the owner's ratings mark a strong execution;
- the spine records why (its attributes);
- retrieval offers it as a reference for future productions.

That is a pattern card of our own. It needs no new library or agent: a card with `origin: own` and `sources` pointing at our assets. Retrieval by semantic similarity comes with Phase 3: text embeddings of the annotations, then image embeddings of our own media.

## 4. Web research: is it strong enough?

**Today:**
- Quick reads use Haiku with up to 3 searches; deep reads use Sonnet with searches and page reads.
- A fixed list of watched topics, each every 3–7 days.
- Claims are classified with sources.
- Findings are reused for 7 days.
- Briefs go to the style director, casting and the showrunner.

**Gaps:**
1. **No recursion.** Each question is one call. It doesn't follow leads, chase contradictions or ask a second, narrower question.
2. **Static topics.** The agents' open questions (gaps, the director's "outfit checks feel stiff") don't become web questions automatically. The web side of research questions exists but nobody files to it.
3. **Thin source evaluation.** Claims are classified, but there is no source-quality weighting and no cross-checking across sources.
4. **Craft research isn't a topic.** How exceptional photography, cinematography and editing are produced isn't watched, and nothing turns findings into production experiments.

**Fix, within the $100/month ceiling:**
- A bounded recursive desk loop: question → findings → at most two follow-ups chosen by the model, chasing gaps and contradictions → synthesis. Capped at about $0.50 per deep question.
- Agents file web questions from their open uncertainties.
- A craft topic set (photography, cinematography, editing, performance, AI production workflows), with findings routed to the director and production as experiment proposals.

## 5. First-party analytics: what each platform gives our own accounts

| Platform | Available for our own accounts (official APIs) | Notes |
|---|---|---|
| Instagram (Graph API, professional account) | Per post: reach, views, likes, comments, saves, shares, total interactions, follows and profile visits from the post. Reels: average watch time and total watch time. Account level: reach, follower demographics (aggregate), follower count over time | `publish/metrics.py` already pulls insights into `metric_snapshots`. Metric names change across API versions: confirm on connection |
| TikTok | Display API: our own videos' views, likes, comments, shares; account follower counts. Deeper analytics (watch time, completion, traffic sources) mostly in the app or business tools | Weakest official analytics; verify what the Content Posting and Display APIs return for our account type |
| YouTube | Analytics API: views, watch time, average view duration, audience retention curves, impressions and click-through rate, subscribers gained and lost, traffic sources | The richest retention data, for when YouTube is in scope |

**Measurement design for the cold start:**
- **Per-reach rates** (saves, shares and follows per reach) separate content quality from how much distribution a post got.
- **Retention** (watch time over length) is the cleanest quality signal for video.
- **Non-follower reach share** tells distribution problems apart, where the platform API gives it.
- **Comparisons** are against each creator's own baseline at the same post age. The spine normalises this way already.
- **Content tests run within a creator, in the same week**, so distribution drift is held roughly constant.

## 6. What we lose, and how we compensate

1. **No audience outcomes until we publish.** The research and experiment machinery can now learn only about production and the owner's taste. Compensation: owner ratings, production-quality learning, and a deliberate **controlled publishing pilot** when you're ready. Learning truly starts there, so I'd treat it as the next strategic milestone.
2. **No external exemplars.** Compensation: web research on craft, pretrained knowledge, and our own proven-assets universe.
3. **No external pre-tests of attribute ideas** (for example, whether candid outperforms editorial). These become first-party experiments, which is stronger evidence anyway.

## 7. Revised priorities

In order. Each item is small and demonstrable.

1. **Finish verification.**
   - Flag analyses that run on zero rows.
   - Point recipes and gaps at own units, so nothing silently analyses an empty table.
   - Record discovery provenance wherever we create units.
2. **Creative range audit**: one sandbox creator, about $3 of the GPU envelope.
   - Generate a deliberately wide set of concepts across the dimensions in §2.
   - Render the photos.
   - Measure diversity, identity and which intended attributes the images actually show (the gold-set annotator on own images).
   - The output: where conception exceeds execution, ranked.
3. **Structural conditioning for photography**: investigate pose and composition control for our photo model, and test the one with the most leverage. The ranked list from step 2 should choose it.
4. **Finishing ablation** (your point 5 from before): the same source clips with and without shot planning, grading, J/L cuts, pacing trims and ambience. It isolates how much quality finishing adds at no extra generation cost.
5. **Recursive web research** and craft topics (§4).
6. **First-party readiness:**
   - an end-to-end Instagram insights test once the account exists;
   - Pegasus on our own clips once the Twelve Labs account exists;
   - the "proven asset" flag (§3).
7. **Phase 3 semantic retrieval** over our own observations and findings, with a local embedding model first.

**What I recommend we don't do:** rebuild the creative stack, add agents, or build TikTok or YouTube analytics before publishing is in scope.
