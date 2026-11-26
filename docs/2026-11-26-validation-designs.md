# Two validation designs: the CEO's judgment, and whether research improves decisions (2026-11-26)

Neither design has been run. Both need the owner's go for their spend, both are sandbox only, and neither creates creators, accounts or posts.

## A. The CEO sandbox: a test of judgment, not of recall

**The risk.** A model that read our principles will repeat them ("growth first, photo-led, proven archetypes"). That proves nothing. The test must reward getting decisions right where repeating the principles gives the wrong answer.

**Method: scenarios with a known best answer.**
- Each scenario is a sandbox portfolio snapshot in the exact shape the CEO reads:
  - portfolio map;
  - capability card;
  - commercial registry;
  - desk findings;
  - creator states and evidence windows.
- Code builds each snapshot, and each hides a trap a careful reasoner catches.
- The CEO runs its real cycle (`agents/strategist.py`) against it. Nothing it proposes is applied.

| Scenario | The trap | A good decision |
|---|---|---|
| Saturated archetype | Highest demand, but the desk shows 40 near-identical creators and falling growth | Pick the second-largest archetype with growing demand |
| Expensive format | Talking video earns 1.6× the engagement but costs 25× per accepted post all-in | Keep it to a minority, or test it; don't shift the mix |
| Early noise | A creator's first 9 posts underperform, inside its evidence window | Patience, a strategy adjustment, no stage move |
| Commercial mirage | Category with the highest CPMs, but tiny audience surface area and poor shareability | Don't cast for it; note it as an adjacency |
| Hidden winner | A modest archetype whose recipes are cheap to produce (photo-only) and grow fast | Prefer it on return per dollar |
| Conflicting evidence | Desk says a trend is rising; first-party metrics say our posts on it fall | Weigh first-party evidence; file a research question |
| Budget squeeze | Requests exceed budget | Rank by expected return; reject with what would change the answer |

**Checks that separate judgment from recall.**
1. **Perturbation.** Each scenario is rerun with the decisive numbers changed (the format costs 2× instead of 25×; the archetype has 4 rivals instead of 40). A reasoner changes its decision; a principle-repeater doesn't. This check scores highest.
2. **Ablation.** The same scenarios run with the principles text removed. If decisions don't change, the principles weren't doing the work. If they get worse, the principles add judgment; if they get better, the principles are dogma.
3. **Explanation fidelity.** Each stated reason must cite a number in the snapshot. Code checks that the cited figures exist and are used correctly.
4. **Blind grading.** A judge of another model family grades decision quality against the scenario's rubric without knowing the expected answer. The owner reviews 3 scenarios by hand.

**Scoring per scenario:**
- correct decision (0/1);
- correct direction under perturbation (0/1);
- reasons grounded (share of citations correct);
- judge rubric score.

**Cost.** About 7 scenarios × 3 variants (base, perturbed, ablated) × one CEO cycle, plus judging. Roughly $3–5 of Claude.

## B. Does research improve what the Brain decides to create?

**The question.** Do casting concepts, brand bibles, content plans and photos made with the intelligence layer beat the same steps made without it?

**Arms.** Identical briefs, models and budgets. Only the intelligence differs.
- **With:** desk briefs, pattern cards, recipe findings and research answers in the prompts.
- **Without:** those blocks empty.

The code already assembles them in one place per agent, so the ablation is a switch, not a fork.

**Stage 1: decisions (cheap, LLM only).**
- Casting concepts: 5 theses × both arms.
- Showrunner idea sets: 3 creators × both arms.
- Blind pairwise judging, A/B order randomised, by:
  - a judge of another model family;
  - the casting rubric;
  - an owner sample.
- Metric: win rate of "with". It is significant only if it beats 50% beyond a binomial interval.
- Cost: about $2–4.

**Stage 2: made content (the GPU photo engine, after rung 1 of the validation ladder).**
- Each arm's ideas produce about 12 photo posts in sandbox batches.
- Measured:
  - acceptance rate;
  - all-in $ per accepted post;
  - blind desirability and aesthetic scores;
  - how often research-backed shot recipes rendered as planned.
- Cost: about $2–3 of GPU plus judging. It needs the owner's go for that run.

**Stage 3: real outcomes (after launch).**
- Alternate research-backed and baseline posts within the same creator, matched by format and slot.
- Compare follower gain, saves and shares per post through `evidence.py`.
- This is the only stage that can show research causes growth.
- It runs as experiment slots, which already exist.

**What would change the plan.**
- If "with" doesn't win stage 1, the intelligence isn't reaching decisions in a usable form. Fix consumption (what agents are given, and how) before collecting more data.
- If it wins stage 1 but loses stage 2, the research points at things production can't render. Feed production acceptance back into which recipes are offered; the pattern weight already includes acceptance.

**When to run it.** After the first real annotated corpus exists (v2 reads, plus recipes from enough creators). Today the intelligence is mostly illustrative seed cards and a few desk findings, so a null result now would say little.
