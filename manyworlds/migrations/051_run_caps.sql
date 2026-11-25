-- A test run's spend cap where the autoscaler can see it (review 2026-11-25): it was enforced only by the run's driver
-- process, so a dead or stalled driver let machines keep launching for the run until the daily cap. The autoscaler
-- now refuses to launch for a run at its cap and cancels its queued jobs (manyworlds.breadth.capped_runs).
CREATE TABLE run_caps (
    batch      text PRIMARY KEY,
    max_usd    numeric NOT NULL CHECK (max_usd >= 0),
    updated_at timestamptz NOT NULL DEFAULT now()
);
