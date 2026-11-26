# The learning harness: from automated pipeline to self-improving creative lab (2026-11-26)

The owner asked for:
- a self-improving creative organisation;
- generous publishing ("the evaluator is not the audience");
- an exploration temperature;
- hypotheses rather than permanent beliefs;
- cross-portfolio learning;
- audience outcomes as the final arbiter;
- the simplest improvements that compound.

This document covers what exists (surveyed, with file references in the session notes), the gaps, and what we are building now. Nothing here adds an agent, a model or a rejection gate.

## 1. What already exists

| Capability | Where | State |
|---|---|---|
| **Defect gates after media exists** | review/gates.py, video.py | Identity, realism, safety, anatomy, framing, lip sync, intelligibility, motion. Scene and direction rank only; final, detail and sync checks report only. |
| **Selection among candidates** | production/steps.py | Score sum plus performance points plus a variety penalty. Never regenerates on taste. |
| **Repairs** | review/repair.py, talk_repair, line_check | Capped (4 per post for images, 2 for video). Triggered by gate failures. |
| **Reroute** | production/core.py | A failed video ships its accepted start photo instead. |
| **Experiments** | brain/experiments.py | Two-arm A/B per creator and effect. Power-sized samples, fixed horizon, no peeking. A significant result writes a `proven` finding. |
| **Exploration** | brain/exploration.py, brain.exploration_share 0.3 | Thompson draws over effects and untried directions. The share is only an instruction in the prompt. |
| **Findings** | migration 012 | Scope (post, creator, cluster, market, portfolio), evidence kind, grade (proven, tentative, speculative), effect with interval, sources. Superseded when replaced. |
| **Pattern cards** | brain/patterns.py | Evidence basis (judgement → validated), acceptance, real outcomes (clipped, at least 10 posts), versions. |
| **Recipes and research questions** | brain/recipes.py, research_questions.py | Pre-registered, discovery/holdout split by creator, BH FDR. |
| **Predictions and scorecard** | brain/predictions.py | Brier score per role. |
| **Outcome join** | evidence.attribution | Format, outfit, setting, shot type, card, archetype. Breakouts and drops become events. |
| **Policy changes** | brain/overrides.py | Versioned, bounded by `brain.authority`, with owner approvals beyond it. |
| **Performance calibration** | performance.py | Per model and control, with reliability floors. Off. |

## 2. Gaps, against the brief

1. **Generous publishing.** Two subjective gates still reject:
   - `brand` alignment, enforced at 3 or above;
   - the voice line check, which regenerates "flat" and "pacing" lines, not only unclear, robotic or wrong-voice ones.

   `set_match` (a carousel's photos in one place) is a functional continuity check, so it stays. Nothing compares "repair this asset" against "make another post".
2. **Exploration temperature.** The 0.3 share is an instruction in the prompt. Code neither enforces nor measures it, and there is no per-creator or portfolio setting, and no levels (refine, adjacent, novel).
3. **Hypotheses, not beliefs.** Findings and cards don't record:
   - where a finding came from, beyond sources;
   - the contexts it applies to;
   - whether it replicated across creators;
   - contradictory evidence;
   - when it was last tested, or when to retest.

   A `proven` finding stays proven forever.
4. **Cross-portfolio levels.** Experiments are per creator only. Nothing turns one creator's result into a hypothesis for comparable creators. Card outcomes pool all creators with no hierarchy.
5. **Outcome learning that resists noise.** Breakouts go straight into planning prompts. Costs, performance controls and most shot-sheet fields aren't joined to outcomes. A policy change has no rollback.
6. **Comparative evaluation.** Nothing makes blinded, randomised pairwise comparisons of our own outputs.
7. **Native video understanding.** Only keyframes and audio metrics. Gemini is removed; Twelve Labs Pegasus is assessed in §5.

## 3. Where I would push back

- **"Hypotheses, not beliefs" needs one firm rule:** our own controlled experiments outrank everything else. Contradicting a proven experiment should take another experiment, not an anecdote. Fluid, but not equal-weighted.
- **Exploration costs growth.** Each novel post is a likely underperformer. The temperature should be set per creator by stage:
  - high at launch, when the audience is unknown;
  - lower once a creator has a working formula;
  - raised again on a plateau or decline.
- **Pairwise judging is a proxy.** Use it for selection and to calibrate the judges, never as a growth signal. The audience decides growth.

## 4. What we build now (simplest first, all code and policy, no spend)

Status 2026-11-26: items 1-6 built (branch `learning-harness`; tests in `tests/test_learning_harness.py`; migration
055 applies on deploy). Nothing from "Later" was started.

1. **Defect-only rejection.** *Done:* `review.common.context_verdicts` (brand always passes, flagged `off_register`, ranked by `production.steps.RANKED_GATES`); `voice_eval.line_verdict` `regenerate` (policy `voice.line_check.regenerate_on`; needs a talkcomfy rebuild to reach the worker); policy `review.reject_only` with its gate-to-class map.
   - `brand` alignment becomes rank only: off-register images lose in selection and are never rejected.
   - The voice line check regenerates only unclear, robotic or wrong-voice lines. Flat and off-pace lines keep their best take and are recorded.
   - A policy principle names what may reject (owner's list): artifacts, anatomy, identity drift, lip-sync or audio defects, technical quality, safety and rights, functional failure.
2. **Exploration temperature, enforced and measured.** *Done:* `brain/exploration.py` (`shares`, `batch_targets`, `reconcile`), policy `brain.exploration_mix`, authority paths `brain.exploration_mix.creators.*.adjacent|novel`; the batch decision's `exploration_mix`, `creator_states.evidence_window.exploration`, attribution dimension `exploration`. Code trims and flags; it never relabels.
   - Each idea carries `exploration: refine | adjacent | novel`.
   - Policy sets the shares per stage (launch, growth, plateau), and the portfolio total.
   - A creator may override, within `brain.authority`.
   - The showrunner is given exact counts per batch, and code tops up or trims to the target.
   - Realised shares and their outcomes go in the evidence windows.
3. **Hypothesis records.** *Done:* migration 055, `brain/hypotheses.py` (weight, decay, view), policy `brain.hypotheses`. Findings and cards gain:
   - `origin`: post, creator, research, or experiment;
   - `applies_to`: archetype, vertical, audience, model, as context tags;
   - `level`: post, character, archetype, vertical, portfolio, production;
   - `replications`: creators where it held;
   - `contradictions`;
   - `last_tested` and `retest_after`.

   Evidence weight decays past `retest_after` until the finding is retested, and a stale proven finding shows as "due for retest". Planning prompts show each finding with its level, its replication count and its age.
4. **Cross-portfolio propagation.** *Done:* `hypotheses.on_experiment` (from `experiments.conclude`), `hypotheses_for` (experiment flow and showrunner), `casting._hypotheses_text`. A creator-level `proven` finding opens a speculative hypothesis at archetype level for comparable creators. It is offered to their showrunners as an experiment suggestion, never applied as a rule. Replication across creators raises its level and weight; a contradiction lowers it. Casting reads the archetype-level hypotheses with their replication counts.
5. **Noise guard.** *Done:* `events.replication` / `for_planning` (policy `brain.events.noise_guard`); `evidence.attribution` dimensions `control` and `brain.attribution.directed`, and `usd_per_post` / `reach_per_usd` per group. A breakout reaches planning as "one post, unreplicated" until the same attribute repeats across comparable posts. The outcome join adds performance controls, all-in cost and directed attributes.
6. **Rollback.** *Done:* `overrides.revert`, `mw policy overrides` and `mw policy revert <override-id> --reason R` (only the latest override of a path; the same authority rules). `mw policy revert <override>` restores the previous value as a new override, with the reason recorded.

**Later, each needing its own go-ahead:**
- blinded pairwise comparisons, used to calibrate the judges against owner picks;
- the Pegasus benchmark (§5);
- champion-challenger across creators.

## 5. Native video understanding: Twelve Labs Pegasus

**What it offers:**
- Pegasus 1.5 and 1.6 take a video, by URL or inline, with no indexing step, and answer in a JSON schema.
- About $0.01 per 10 s clip and $0.02 per 30 s clip, close to today's keyframe call. Cost doesn't decide this; quality does.

**Unproven where we need it most:**
- It reads speech, but nothing shows it judges tone of voice or prosody.
- It samples frames sparsely, so a lip-sync offset of 100–300 ms is probably below what it can see.
- It is marketed for long-video search, not short performance clips.

**Terms:**
- **Training.** The self-serve terms let Twelve Labs train on our uploads unless we opt out in the account settings before the first upload. Bedrock may avoid this; to be verified.
- **AUP.** It forbids using its outputs to train other models, and forbids inferring gender or age.

**Benchmark** (under $5, no GPU, needs an account you create):
- **Clips:** 20 of our finished clips, plus 20 copies deliberately broken with ffmpeg (audio shifted, a wrong-emotion voice, an ending cut mid-word, a freeze, a speed-up) as objective ground truth.
- **Arms:**
  - today's 4 keyframes to Claude;
  - Pegasus with the same rubric;
  - Claude with 12 frames and a word-timed transcript, to separate "better model" from "more frames".
- **Ground truth:** the broken copies' labels, and optionally 15 minutes of your blinded pairwise picks.
- **Decision rule:** adopt only as an added screen for voice-face coherence and endings, if it catches clearly more broken copies at equal or lower cost.
