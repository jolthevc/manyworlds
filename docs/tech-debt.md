# Technical debt

Known issues we chose not to fix yet, with when they must be fixed. "Before X" means the
capability X does not go live until the item is closed. Updated 2026-11-14 (Brain M1 merge).

## Before autonomous publishing (Brain M4)

| Item | Why it matters | Where |
|---|---|---|
| The brain worker dies on a dropped database connection (Docker restarts it; the job waits out its lease, up to 10 min) | publish and metrics-collection jobs must not stall silently, and Story metrics expire after 24 h | `manyworlds/cli.py` `_brain_ops` loop |
| A `KeyError` raised inside an agent's own code is treated as permanent (non-retryable) | a transient bug would drop a publish or collection job instead of retrying it | `manyworlds/brain/runtime.py` `run_job` |
| A retryably crashing job ends each runtime tick early until it runs out of attempts | other jobs (publishing, collection) wait a tick per failure | `manyworlds/brain/runtime.py` `tick` |

## Before the Brain allocates real money (Brain M8/M9)

| Item | Why it matters | Where |
|---|---|---|
| The brain worker's Claude client keeps the policy it started with (model roles, prices) until restarted | LLM cost entries could be priced or routed on stale settings | `manyworlds/cli.py` `_brain_ops` |
| `effective_policy` reads override rows and the version in two queries | under concurrent changes a decision could record a version that doesn't exactly match the settings it used | `manyworlds/brain/overrides.py` |
| Only Brain agents read the effective policy (`brain.authority.observed_sections: [brain]`); overrides of production, review or autoscaler settings are refused | the strategist can't tune production until those services read the effective policy; then add their sections to `observed_sections` | services' policy loading |

## Later

| Item | Where |
|---|---|
| `mw brain run <agent>` runs a whole tick and may run other queued jobs first | `manyworlds/cli.py` |
| `_waiting_kinds` scans the jobs table on each claim (fine at today's volume) | `manyworlds/brain/runtime.py` |

## Closed at the M1 merge

- Simulated data could in principle be read through raw queries: now structural (separate
  `simulated_*` tables, CHECK constraints on the real ones, guard test over the whole package).
- Approve-then-apply was not atomic; override versions could collide: one transaction, unique version.
- Brain LLM spend mixed into cost per output: now reported separately; cost per accepted output
  counts GPU and review only.
- Publications could be marked simulated on a real platform: CHECK constraint.

## Creative evaluation (Brain M2), deferred minors

| Item | Before | Where |
|---|---|---|
| A dimension scored by only one of three samples reports spread 0 (looks certain) | rubric recalibration by predictive validity (M6) | `brain/evaluation.py` `score` |
| A missing asset/idea row or a non-image asset fails retryably with a TypeError instead of a clear non-retryable reason; media type guessed from extension (png/jpeg only) | video keyframe scoring (M10) | `brain/agents/evaluator.py` |
| `checked()` depends on how an id is written (case, braces) | — | `brain/evaluation.py` |
| A crash between storing evaluation rows and the remembered decision can duplicate rows on rerun | M6 | `brain/agents/evaluator.py` |
| `mw brain calibrate`/`rubric-drift` don't check the evaluator's daily budget | — | `manyworlds/cli.py` |
| Rubric filename version not cross-checked with the file's `version:`; a stray `idea-vdraft.yaml` crashes; schema cache keyed without file hash; `register` race | — | `brain/rubrics.py`, `brain/evaluation.py` |
| Rubrics always load from the repo/`/app` path, not from `MW_POLICY`'s directory | — | `brain/rubrics.py` |
| Reference set and drift exist for the idea rubric only; execution and feed have none yet | M7 (feed QA) | `policy/rubrics/reference/` |

## Memory and showrunner (Brain M3), deferred minors

| Item | Before | Where |
|---|---|---|
| Canon enters memory when a post is scheduled; a post later dropped leaves its canon in memory | M4/M5 (move to publication time) | `brain/ideas.py` `schedule_ready` |
| Ideas dropped after max_wait leave their evaluator jobs queued (they still spend evaluator budget) | — | `brain/ideas.py` |
| Low-scoring ideas are scheduled anyway (by design: no dimension is a target); revisit once scores are shown to predict performance | M6 | — |
| Switching `brain.showrunner.enabled` off while a batch is proposed lets the planner plan the same period | — | `cli._produce` |
| A post linked to two ideas (from `from_post` and a scheduled idea) appears twice in the showrunner's recent posts | — | `brain/agents/showrunner.py` |
| Scheduled posts record the creator's profile version at scheduling time, not the version the batch was planned under | M6 | `brain/ideas.py` |
| Only `signature` (and keyed canon) close older facts; seeded places/people have no keys, so a moved home relies on the showrunner setting `canon_key` | — | `brain/memory.py` |
