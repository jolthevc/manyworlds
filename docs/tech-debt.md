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
