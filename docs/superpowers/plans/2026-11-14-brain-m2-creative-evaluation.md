# Brain M2: Creative Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every idea (before it is made) and every finished image (after) can be scored on a versioned, anchored rubric by an independent evaluator, reproducibly enough to compare over time, with the scores stored for later comparison with real Instagram results.

**Architecture:** Rubrics are YAML files in `policy/rubrics/` (dimensions, definitions, anchors at 1/3/5/7/9, a `version`), registered in `rubric_versions` by file hash. `brain/evaluation.py` scores content `samples` times with the evaluator model and keeps the median and spread per dimension; a deterministic share of items is also scored by the strong model to measure agreement. The `evaluator` agent wraps it as a `brain.evaluator` job. A reference set and a drift command show how scores move when a rubric or model changes.

**Tech Stack:** Python 3.11, Pydantic 2 (`create_model` for per-rubric schemas), psycopg 3, PyYAML; tests with scripted LLM stubs.

**Spec:** `docs/superpowers/specs/2026-11-14-brain-design.md` §4.4; roadmap M2 in `docs/superpowers/plans/2026-11-14-brain-roadmap.md`.

## Global Constraints

- No dimension is a target; nothing gates on a creative score (spec 4.4). Existing safety/identity review gates are unchanged.
- The evaluator sees the creator's profile summary but never the showrunner's reasoning (`ideas.body.reasoning` is excluded).
- Every evaluation row stores: item, rubric id + version, evaluator model, prompt version, per-dimension median, spread, rationale, samples, creator profile version, format.
- Scores are integers 1–10, clamped. Spread = max − min across samples.
- Models come from `policy.llm.roles` (`evaluator`, `evaluator_check`); sample counts and the check rate from `policy.brain.evaluation`.
- Evaluator LLM spend is Brain spend (agent `evaluator`), outside cost per accepted output.

## Review Focus

1. The LLM omits a dimension or returns a score outside 1–10: the dimension is clamped or recorded as missing for that sample, and the median uses the samples that have it; an evaluation with no score at all for a dimension stores null, never a fabricated number.
2. A rubric file is edited without bumping its `version`: registration refuses (scores under one version must mean one thing).
3. An idea is evaluated twice by a rerun of the same job: one evaluation row, not two (uses `agent.remember`).
4. An asset whose image can't be fetched: the job fails retryably with the reason; no partial evaluation row.
5. Double-check sampling is deterministic per item (the same idea is always or never checked at a given rate), so reruns don't re-roll it.

---

### Task 1: Rubrics as versioned files

**Files:** Create `policy/rubrics/idea-v1.yaml`, `policy/rubrics/execution-v1.yaml`, `policy/rubrics/feed-v1.yaml`, `manyworlds/brain/rubrics.py`; Test `tests/test_brain_rubrics.py`.

**Interfaces — Produces:**
- `class Dimension(BaseModel): key: str; name: str; definition: str; not_: str (alias "not"); anchors: dict[int, str]` (keys exactly 1, 3, 5, 7, 9)
- `class Rubric(BaseModel): rubric: str; version: int; applies_to: str; dimensions: list[Dimension]; file_hash: str`
- `rubrics.load(name: str, *, root: Path | None = None) -> Rubric` — reads `policy/rubrics/<name>-v*.yaml`, highest version.
- `rubrics.register(conn, r: Rubric) -> int` — inserts into `rubric_versions` if new; same version with a different hash raises `RubricChanged`.
- Content: idea dimensions = the ten in spec 4.4 (relatability, entertainment, novelty, curiosity, emotional_pull, shareability, aspiration, personality, audience_relevance, naturalness); execution = believability, visual_appeal, styling, identity, setting, caption_voice, defects (10 = none); feed = coherence, variety, repetition (10 = none), aesthetic_consistency, impression. Anchors written as concrete descriptions of a lifestyle post at that level.

- [ ] Step 1: tests — `test_every_rubric_has_complete_anchors` (each dimension has anchors 1,3,5,7,9 and non-empty definition), `test_editing_a_rubric_without_bumping_its_version_is_refused` (register, tamper body+hash in a temp copy, register → `RubricChanged`), `test_registering_twice_is_idempotent`.
- [ ] Step 2: run, expect FAIL. Step 3: implement. Step 4: suite PASS. Step 5: commit.

### Task 2: Scoring core

**Files:** Create `manyworlds/brain/evaluation.py`; Modify `policy/default.yaml` (`llm.roles.evaluator: claude-sonnet-5-5`, `llm.roles.evaluator_check: claude-opus-5-5`, `brain.evaluation: {samples: 3, check_rate: 0.1, prompt_version: "eval-v1"}`); Test `tests/test_brain_evaluation.py`.

**Interfaces — Produces:**
- `schema_for(r: Rubric) -> type[BaseModel]` — one field per dimension: `{score: int, why: str}`, plus `overall_note: str`.
- `@dataclass Scored: medians: dict[str, float | None]; spreads: dict[str, int | None]; rationale: dict[str, list[str]]; model: str; samples: int`
- `score(llm, policy, r: Rubric, content: list, *, role: str, samples: int, creator_id=None) -> Scored` — `samples` calls; clamps to 1–10; missing dimensions skipped per sample; median of present values (None if none).
- `checked(item_id, rate: float) -> bool` — deterministic: first 8 hex chars of sha256(str(item_id)) / 2^32 < rate.

- [ ] Step 1: tests with a `ScriptedLLM` returning given score dicts in order — `test_median_and_spread_across_samples` (scores 4,6,8 → median 6, spread 4), `test_out_of_range_scores_are_clamped`, `test_a_dimension_no_sample_scored_is_null_not_invented`, `test_double_check_sampling_is_deterministic` (same id same answer; rate 0 never, rate 1 always; ~rate share over 1000 ids within ±0.05).
- [ ] Step 2–5: FAIL, implement, suite PASS, commit.

### Task 3: Evaluator agent and stored evaluations

**Files:** Create `manyworlds/brain/agents/evaluator.py`; Modify `manyworlds/brain/agents/__init__.py`, `policy/default.yaml` (`job_kinds.brain.evaluator: {pools: [local], outputs_per_job: 0}`); Create `manyworlds/brain/ideas.py`; Test `tests/test_brain_evaluator.py`.

**Interfaces:**
- Consumes: Task 1 `load/register`, Task 2 `score/checked`, M1 `agent.remember`, `AgentContext`.
- Produces:
  - `ideas.from_post(conn, post_id) -> uuid` — makes an `ideas` row from a planned post (concept from scenes/script/caption, format, attributes pillar/tone/series), linked by `post_id`; idempotent per post. (M3's showrunner writes ideas directly; this bridges today's planned posts.)
  - `evaluator.evaluate_idea(ctx, idea_id) -> uuid` and `evaluator.evaluate_asset(ctx, asset_id) -> uuid` — build content (rubric with anchors as text; creator profile summary from `profile.current`; idea concept/body without `reasoning`, or the image via `image_block` plus the post's caption); `score` with role `evaluator`; insert `evaluations` (item_kind idea|asset); if `checked(item, check_rate)` also score with `evaluator_check` and insert a second row. Wrapped in `agent.remember` per item so reruns don't duplicate.
  - Agent `evaluator`: payload `{"idea_id": …}` or `{"asset_id": …}`; returns the evaluation ids.
  - `evaluation.latest(conn, *, idea_id=None, asset_id=None) -> dict | None` for later milestones.

- [ ] Step 1: tests — `test_an_idea_is_scored_and_stored_with_its_provenance` (rubric version, model, prompt_version, samples, profile_version, medians present), `test_the_evaluator_never_sees_the_showrunners_reasoning` (idea body has `reasoning: "SECRET"`; no LLM call content contains it), `test_a_rerun_does_not_store_a_second_evaluation`, `test_checked_items_get_a_strong_model_row_too` (policy check_rate 1.0 → two rows, models differ), `test_an_asset_that_cannot_be_fetched_fails_without_a_row`, `test_from_post_is_idempotent`.
- [ ] Step 2–5: FAIL, implement, suite PASS, commit.

### Task 4: Reference set, drift and calibration

**Files:** Create `policy/rubrics/reference/idea.yaml` (12 described lifestyle ideas spanning dull to striking, ordinary to novel; text only); Modify `manyworlds/brain/evaluation.py`, `manyworlds/cli.py`; Test `tests/test_brain_drift.py`.

**Interfaces — Produces:**
- `evaluation.score_reference(conn, policy, llm, rubric_name) -> dict[str, dict[str, float]]` — reference item id → dimension medians; stored in `rubric_versions.reference` for the current version.
- `evaluation.drift(before: dict, after: dict) -> dict[str, float]` — mean absolute shift per dimension over shared items.
- CLI: `mw brain rubric-drift <rubric>` (scores the reference set under the current version and model, stores it, prints drift vs the previous version's stored scores); `mw brain calibrate <creator> --posts N` (ideas from the creator's last N planned/produced posts, evaluates them, prints per-dimension median spread and the share of items within `brain.evaluation.max_spread`).

- [ ] Step 1: tests — `test_drift_is_mean_absolute_shift_on_shared_items`, `test_reference_scores_are_stored_on_the_rubric_version`.
- [ ] Step 2–5: FAIL, implement, suite PASS, commit.
- [ ] Step 6 (live acceptance): `mw brain calibrate maren.afterhours --posts 12` against the real database; record median spread per dimension; acceptance: median spread ≤ `brain.evaluation.max_spread` (start: 2) on at least 8 of 10 idea dimensions. If not, tighten the anchors of the failing dimensions (rubric v2) and re-run once; report either way.
