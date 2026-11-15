# Brain M5: Simulated Closed Learning Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the closed loop end to end on a sandbox creator: posts are (simulated-)published, outcomes are ingested, an analyst finds which creative characteristics predict performance (by creator and format, with uncertainty), records evidence-graded hypotheses, and the showrunner's next batch traceably changes because of them, while insufficient evidence leads to more exploration, not a formula.

**Architecture:** A publisher interface (`publish/`) with a simulated implementation now and Instagram in M4; posts carry one `publications` row either way, and outcomes land in `simulated_metrics` (sandbox) or `metric_snapshots` (real) with the same metric keys. A deterministic, seeded scenario engine (`brain/simulation.py`, scenarios in `policy/scenarios/*.yaml`) plays the audience: hidden effects of rubric dimensions, formats and attributes on reach, plus breakouts, noise, account growth or decline. The `analyst` agent computes normalised outcomes and effects (pure Python: baseline log-lift, slopes and group differences with bootstrap intervals) in code, then asks the LLM only to word evidence-qualified hypotheses and next-batch guidance from those numbers. Findings go through `evidence.record_finding` (simulated for sandbox creators). The showrunner reads findings and the latest guidance, and records which findings shaped its batch; `mw brain learning <creator>` shows the trace.

**Tech Stack:** Python 3.11 standard library (`statistics`, `random`), psycopg 3, Pydantic 2. No numpy.

**Spec:** design §4.5 (measurement and learning), §4.9 (publishing), 6a (simulated vs real); owner's M5 brief (2026-11-14): simulated = engineering test only; actionable intelligence; creator- and market-level ready; autonomous recovery; consistent real/simulated interface.

## Global Constraints

- Simulated outcomes never reach real-mode reads or investment decisions (M1 wall; findings from sandbox creators are `evidence='simulated'`, labelled SIMULATED).
- Statistics are computed in code; the LLM only words and prioritises what the numbers show, and must carry the interval and n.
- Grades: `speculative` (an effect estimate with the interval crossing zero, or n below `min_n_tentative`), `tentative` (interval excludes zero and n ≥ `min_n_tentative`); `proven` is reserved for experiments (M6). Below `min_n_any` no effect is recorded at all: the analyst records "insufficient evidence" and asks for exploration.
- Every number (ages, minimum n, bootstrap draws, effect thresholds, scenario parameters) lives in policy or the scenario file.
- Findings carry scope `creator` now, with `sources` listing post ids, so cluster/market aggregation (M8/M9) can pool them later.

## Review Focus

1. A creator with a handful of posts: no tentative findings; the guidance says to keep exploring.
2. Null scenario (no hidden effects): over repeated seeds, tentative findings appear at most at the policy false-positive rate.
3. A declining or growing creator: the trend in raw reach doesn't produce spurious dimension effects (normalisation).
4. A single breakout post: it is reported, but alone it doesn't create a tentative finding.
5. Conflicting signals (an effect positive in photos, negative in story clips): per-format results show both, the pooled one is not graded above speculative.

---

### Task 1: Recovery for stalled planning periods
**Files:** `manyworlds/brain/showrunner_loop.py`, `policy/default.yaml` (`brain.showrunner.retry_after_hours: 12`), `tests/test_brain_schedule.py`.
- A period that used its attempts records one `events` row (`planning_stalled`, routed to the owner digest) and is retried after `retry_after_hours` (only attempts inside that window count). Periods whose start has passed are naturally replaced (next_start moves forward daily).
- Tests: `test_a_stalled_period_raises_one_event`, `test_a_stalled_period_is_retried_after_the_cooldown`.

### Task 2: Publisher interface and simulated publishing
**Files:** Create `manyworlds/publish/__init__.py`, `manyworlds/publish/simulated.py`, `manyworlds/publish/scheduler.py`; modify `policy/default.yaml` (`autonomy.publish: autonomous` per the approved default, kill switch value `paused`; `simulation.fast_forward: false`); Test `tests/test_publish.py`.
- `Publisher` protocol: `publish(conn, policy, post: dict) -> dict` returning `{external_id, permalink, ai_label, simulated}`.
- `scheduler.publish_due(conn, policy, *, now) -> dict`: posts `ready` with `scheduled_for <= now` (plus `planned` ones when `simulation.fast_forward` for sandbox creators only), publisher chosen by `creators.publish_mode`; writes `publications` and sets post `status` = `published` (real) or `simulated` (sandbox), `published_at`. `autonomy.publish: paused` publishes nothing. Instagram creators are skipped until M4 provides the adapter (counted as `waiting_for_adapter`).
- Tests: `test_sandbox_posts_are_simulated_not_published`, `test_the_kill_switch_stops_publishing`, `test_real_creators_wait_for_the_instagram_adapter`, `test_fast_forward_is_sandbox_only`.

### Task 3: Scenario engine (the simulated audience)
**Files:** Create `manyworlds/brain/simulation.py`, `policy/scenarios/{null,personality_wins,breakout,format_flop,conflicting,declining}.yaml`; Test `tests/test_simulation.py`.
- Scenario file: base reach and followers, daily growth/decline, noise sd, effects (`dimension: k, per_point: x` on log reach; `format: f, lift: x`; `attribute: key=value, lift: x`; `format_dimension` interactions), breakout probability and multiplier, metric ratios (likes/saves/shares/follows per reach).
- `simulate_post(conn, policy, post_id, *, scenario, seed) -> list[dict]` writes `simulated_metrics` at `simulation.ages_s` with the same keys real ingestion will use (`reach, views, likes, comments, saves, shares, follows`), deterministic for (post, seed). `run_scenario_hook`: the scheduler calls it for each simulated publication when the creator's `anchors.simulation.scenario` is set.
- Tests: `test_the_same_seed_gives_the_same_outcomes`, `test_a_planted_effect_moves_reach_in_its_direction`, `test_breakouts_happen_at_their_rate`, `test_metric_keys_match_real_ingestion`.

### Task 4: Analyst statistics (code, no LLM)
**Files:** Create `manyworlds/brain/analysis.py`; modify `policy/default.yaml` (`brain.analysis: {target_age_s: 259200, min_n_any: 8, min_n_tentative: 20, bootstrap: 400, ci: 0.9, baseline_window: 20, breakout_lift: 1.5}`); Test `tests/test_analysis.py`.
- `dataset(conn, creator_id, *, mode) -> list[Row]`: each published/simulated post with its idea scores (volume evaluator), format, attributes, outcome at the target age (latest available ≤ it) read through `evidence.metrics`.
- `normalise(rows)`: outcome = ln(reach / rolling median of the previous `baseline_window` posts' reach) (first posts use the running median).
- `effects(rows, policy, *, seed) -> dict` — per dimension: slope of outcome on score with bootstrap interval; per format and per attribute value: mean difference vs the rest with interval; per format × dimension where n allows; breakouts (lift ≥ `breakout_lift`). Each effect `{key, kind, effect, low, high, n, grade}` graded per Global Constraints; `insufficient: bool` when n < `min_n_any`.
- Tests: the five Review Focus cases as tests (`test_few_posts_are_insufficient`, `test_null_scenario_false_positive_rate`, `test_trend_does_not_create_dimension_effects`, `test_one_breakout_is_reported_not_graded`, `test_conflicting_signals_show_per_format`) plus `test_a_planted_effect_is_found_with_the_right_sign` (scenario `personality_wins`, 60 posts).

### Task 5: Analyst agent, findings and guidance; the showrunner uses them
**Files:** Create `manyworlds/brain/agents/analyst.py`; modify `agents/__init__.py`, `agents/showrunner.py`, `brain/showrunner_loop.py` (analyst before each new batch when new outcomes exist), `policy/default.yaml` (`job_kinds.brain.analyst`, `llm.roles` unchanged: role `brain`); Test `tests/test_brain_analyst.py`.
- Agent `analyst`, payload `{creator_id}`: `dataset` → `effects` (code) → LLM (role `brain`) writes `hypotheses` (statement + which effect keys support it + what would confirm it) and `guidance` (what the next batch should do more/less of and what to explore, honouring uncertainty), from the numbers only. Records findings via `evidence.record_finding` (scope creator, effect/interval/n, sources = post ids + table) and one decision (`kind='analysis'`) whose output holds the effects, hypotheses, guidance and finding ids. Insufficient data → no findings, guidance "keep exploring".
- Showrunner: prompt gets the latest analysis guidance and the findings (already); its decision `inputs` record `analysis_decision_id` and `finding_ids` used.
- `ensure_batches` enqueues `brain.analyst` (idempotent per creator and latest outcome) before asking for the next batch, and the showrunner job waits for it (commissioned dependency: showrunner payload carries `after_analysis`).
- Tests: `test_findings_from_a_sandbox_creator_are_simulated_and_labelled`, `test_insufficient_evidence_records_no_findings_and_asks_to_explore`, `test_the_next_batch_records_the_findings_and_guidance_it_used`, `test_the_llm_sees_numbers_not_raw_posts_it_could_overfit`.

### Task 6: Trace command and the live demonstration
**Files:** `manyworlds/cli.py` (`mw brain learning <creator>`; `mw brain simulate <creator> --scenario S --batches N` which fast-forwards: showrunner → evaluator → schedule → simulated publish → outcomes → analyst, N times, for a sandbox creator only), Test `tests/test_brain_learning_trace.py`.
- `learning` prints: the latest analysis (effects with intervals, grades, hypotheses, guidance), the next batch's decision inputs (finding ids, guidance used), and batch-composition change before vs after (format shares, mean score on the implicated dimensions, exploratory share).
- Live acceptance on a dedicated sandbox creator cloned from Maren (`maren-sim`, publish_mode simulated, scenario `personality_wins` with a story_clip flop and one breakout): run several batches; show (1) an outcome, (2) the analyst's evidence-qualified hypothesis with interval and n, (3) the next batch's decision citing it, (4) a measurable shift in that batch, and (5) early batches flagged insufficient → exploration. Images are not generated in fast-forward (cost and time); this is stated in the output.
