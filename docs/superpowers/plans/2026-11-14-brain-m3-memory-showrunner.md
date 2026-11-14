# Brain M3: Memory and Showrunner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Content comes from a showrunner agent that knows the creator (memory), sees how its recent ideas scored and what has been learned, explores on purpose, and writes ideas that are each scored by the evaluator before they become posts for the existing production pipeline, Stories included.

**Architecture:** `brain/memory.py` keeps versioned creator facts (seeded from the profile, plus canon from scheduled posts). The `showrunner` agent extends today's planner (same slotting, validation and voice rules, refactored into reusable functions) and writes `ideas` with attributes and hidden reasoning, then commissions one `brain.evaluator` job per idea. `ideas.schedule_ready` turns a batch into `posts` only once every idea in it has an evaluation. The producer loop asks for showrunner batches instead of planner plans when `brain.showrunner.enabled`; the planner stays for creators without it and drains.

**Tech Stack:** Python 3.11, psycopg 3, Pydantic 2; stubbed LLM in tests.

**Spec:** `docs/superpowers/specs/2026-11-14-brain-design.md` §4.2 (persona & memory keeper, showrunner), §4.5 (exploration), §4.8 (Stories); roadmap M3.

## Global Constraints

- Natural feed, no forced purpose: ideas carry attributes (setting, outfit, activity, mood, trend) for learning, never a strategic category (principle 3).
- No idea becomes a post without an evaluation by the volume evaluator (spec 4.4 "enforced").
- The showrunner's reasoning is stored on the idea (`body.reasoning`) and never shown to the evaluator (M2 guarantees this).
- Findings are read only through `brain/evidence.py` with the creator's mode (sandbox creators may see SIMULATED findings, labelled).
- Exploration share comes from the effective policy (`brain.exploration_share`, Brain-tunable).
- Existing voice rules apply unchanged (`planner.caption_issues`, excluded topics, no selling, disclosure flags on posts).

## Review Focus

1. The showrunner returns ideas in formats production can't make, for platforms out of scope, or with AI-sounding captions: they are dropped with reasons recorded, the rest kept (planner's `_clean` behaviour).
2. A batch where one idea's evaluation fails: the batch isn't scheduled until it has one, and the evaluator job's retries are bounded; after `brain.showrunner.max_wait_hours` unevaluated ideas are dropped and the rest scheduled.
3. Two producer ticks before the showrunner job runs: one showrunner job per creator and period (idempotency key), not two.
4. A creator with no profile or no reference images: no showrunner job (same filter as `planner.ensure_plans`).
5. Memory facts that contradict each other (two "lives in" facts): the newest wins in what the showrunner sees; the older is closed (`valid_until`).

---

### Task 1: Creator memory

**Files:** Create `manyworlds/brain/memory.py`; Test `tests/test_brain_memory.py`.

**Interfaces — Produces:**
- `memory.add(conn, creator_id, kind: str, text: str, *, source_post=None, key: str | None = None) -> uuid` — `key` names a single-valued fact (e.g. `home`, `partner`); adding a fact with an existing open `key` closes the old one.
- `memory.seed_from_profile(conn, creator_id) -> int` — facts from the current profile/bible (places, routines, interests, opinions and running jokes, people, signature item), idempotent (skips texts already open).
- `memory.facts(conn, creator_id, *, limit: int) -> list[dict]` — open facts, newest first, `{kind, text}`.
- Kinds: `place`, `routine`, `taste`, `opinion`, `person`, `canon`, `signature`.

- [ ] Tests: `test_seeding_from_the_profile_is_idempotent`, `test_a_newer_single_valued_fact_replaces_the_old_one`, `test_facts_are_open_ones_newest_first`. FAIL → implement → suite PASS → commit.

### Task 2: Showrunner agent

**Files:** Modify `manyworlds/brain/planner.py` (extract `validate(policy, item, *, platforms, days, prof, hashtags_max)` from `_clean`, and `slot_times(policy, start, days_of_items) -> list[datetime]` from `plan_period`'s scheduling); Create `manyworlds/brain/agents/showrunner.py`; Modify `manyworlds/brain/agents/__init__.py`, `policy/default.yaml` (`job_kinds.brain.showrunner`, `brain.showrunner: {enabled: true, days: 3, recent_posts: 30, memory_facts: 60, max_wait_hours: 6}`); Test `tests/test_brain_showrunner.py`.

**Interfaces:**
- Consumes: Task 1 `memory.facts`, M2 `evaluation.latest`, M1 `evidence.findings/mode_for`, `agent.remember/commission`, planner validation.
- Produces:
  - `class PlannedIdea(PlannedPost)`: adds `attributes: dict[str, str]` (setting, outfit, activity, mood, trend) and `reasoning: str` (why this post now; hidden from the evaluator), `exploratory: bool`.
  - Agent `showrunner`, payload `{creator_id, start (ISO date)}`: one remembered step that builds the prompt (profile, memory facts, recent posts with their idea scores, findings in the creator's mode, exploration share, formats incl. `story`, counts from dials), calls the `brain` role, validates, inserts `ideas` (status `proposed`, `batch = f"{creator}:{start}"`, body with scenes/script/caption/on_screen_text/hashtags/canon/pillar/tone/series/day/platform/reasoning), and commissions `brain.evaluator` per idea (key `eval:{idea_id}`). Returns `{batch, ideas, dropped}`.
- [ ] Tests: `test_a_batch_of_ideas_is_written_and_each_sent_for_evaluation`, `test_invalid_ideas_are_dropped_with_reasons`, `test_the_prompt_carries_memory_scores_findings_and_the_exploration_share` (sandbox creator: a SIMULATED finding appears labelled; exploration share from an override), `test_a_rerun_does_not_write_the_batch_twice`. FAIL → implement → PASS → commit.

### Task 3: From evaluated ideas to posts; the producer uses the showrunner

**Files:** Modify `manyworlds/brain/ideas.py` (`schedule_ready`), `manyworlds/cli.py` `_produce` (and/or `manyworlds/production.py`), `manyworlds/brain/planner.py` (`ensure_plans` skips creators the showrunner serves); Create `manyworlds/brain/showrunner_loop.py` with `ensure_batches(conn, policy, *, now) -> list[dict]`; Test `tests/test_brain_schedule.py`.

**Interfaces — Produces:**
- `ensure_batches(conn, policy, *, now)` — same creator filter and plan-ahead window as `planner.ensure_plans`; enqueues `brain.showrunner` with `idempotency_key=f"showrunner:{creator_id}:{start}"`.
- `ideas.schedule_ready(conn, policy, *, now) -> dict` — for each batch whose ideas all have a volume-evaluator evaluation (or whose oldest idea is older than `max_wait_hours`, dropping the unevaluated), insert `posts` (status `planned`, slots from `planner.slot_times`, `platform_ai_label: true`, body as the planner writes it plus `idea_id`), set `ideas.post_id`/status `scheduled`, add each idea's canon to memory (`kind=canon`, `source_post`). Records one `decisions` row per batch.
- The producer loop calls `ensure_batches` and `schedule_ready` when `brain.showrunner.enabled`, else `planner.ensure_plans`.
- [ ] Tests: `test_ideas_without_an_evaluation_are_not_scheduled`, `test_an_evaluated_batch_becomes_planned_posts_in_slots`, `test_after_max_wait_unevaluated_ideas_are_dropped_and_the_rest_scheduled`, `test_one_showrunner_job_per_creator_and_period`, `test_creators_without_references_get_no_batch`, `test_scheduled_canon_enters_memory`. FAIL → implement → PASS → commit.

### Task 4: Stories, and the live loop

**Files:** Modify `policy/default.yaml` (`planner.formats.story: [1, 1]`, `format_notes.story`, `production.formats += story`, `generation.image_by_format.story: {width: 720, height: 1280}`), `manyworlds/production.py` (`_scene_job` uses the per-format size); Test `tests/test_production.py` (`test_story_images_are_vertical`).
- [ ] Test, implement, suite PASS, commit.
- [ ] Live acceptance: seed Maren's memory; run the showrunner for her next period; let the brain worker evaluate; schedule; show the batch (ideas, scores, formats, which are exploratory) and read it as a feed. Acceptance: every scheduled post has an idea evaluation; the batch mixes formats per the dials (Stories included); ideas reference memory facts; nothing reads as a formula.
