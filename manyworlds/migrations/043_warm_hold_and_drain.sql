-- The GPU factory's warm hold and daily-cap drain (docs/research/2026-11-25-gpu-factory-audit.md, items e and 6).
-- keep_warm_until: set by the autoscaler each tick from work soon due for the machine's kinds (queued jobs it could
-- serve, producing posts that will need its families, planned posts about to start), capped by its deadline and spend
-- cap; its worker's idle self-stop and the enforcer's idle stop wait until it has passed. NULL: no hold.
-- draining_since: the day's GPU budget was reached while the machine had a job in flight; its worker claims nothing
-- new and the enforcer stops it once that job ends, or at gpu_runs.daily_drain_grace_minutes / max overshoot.
-- The worker role reads both (it already reads gpu_groups); it writes neither.
ALTER TABLE gpu_groups ADD COLUMN keep_warm_until timestamptz;
ALTER TABLE gpu_groups ADD COLUMN draining_since timestamptz;
GRANT SELECT (keep_warm_until, draining_since) ON gpu_groups TO mw_worker;
