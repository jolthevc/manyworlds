-- Owner grants (manyworlds.budget): a one-day exception that raises that UTC day's caps (budgets.daily_usd and
-- gpu_runs.daily_gpu_usd) by usd. Written only by `mw budget grant`; the policy numbers never change.
CREATE TABLE budget_grants (
    id         bigserial PRIMARY KEY,
    day        date NOT NULL,
    usd        numeric NOT NULL CHECK (usd > 0),
    reason     text NOT NULL,
    granted_by text NOT NULL DEFAULT 'owner',
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX budget_grants_day_idx ON budget_grants (day);
