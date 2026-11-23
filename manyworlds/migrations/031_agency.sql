-- The agency's connective tissue (docs/superpowers/plans/2026-11-23-agency.md).
--
-- proposals: typed requests between roles (a talent manager asks the CEO for a tier, a market lead proposes a
-- cohort, anyone escalates to the owner). Each has one owner who decides it once; code checks rights and money
-- (manyworlds.brain.proposals). A rejected request comes back only with new evidence or after a cooldown.
CREATE TABLE proposals (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kind         text NOT NULL,              -- tier_change | budget | research | experiment | cohort | production_upgrade | escalate
    from_role    text NOT NULL,              -- talent_manager | market_lead | strategist | analyst | chief_of_staff
    to_role      text NOT NULL,              -- strategist | owner | talent_manager | analyst
    creator_id   uuid REFERENCES creators(id),
    market_id    uuid REFERENCES markets(id),
    subject      jsonb NOT NULL DEFAULT '{}', -- what is asked: {tier: premium}, {usd: 3}, {question: ...}
    argument     text NOT NULL,
    evidence     jsonb NOT NULL DEFAULT '{}', -- finding ids, numbers, conviction, labelled real or simulated
    status       text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'accepted', 'rejected', 'expired', 'done')),
    resolution   jsonb NOT NULL DEFAULT '{}', -- who decided, why, what was applied or refused by code
    dedupe_key   text,
    job_id       uuid REFERENCES jobs(id),   -- the job that filed it
    due_at       timestamptz,
    decided_at   timestamptz,
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX proposals_dedupe_idx ON proposals (dedupe_key) WHERE status = 'open';   -- one open copy; refiled later is fine
CREATE INDEX proposals_open_idx ON proposals (to_role, status, created_at);

-- predictions: what an agent expected, scored later against real outcomes (manyworlds.brain.predictions). The
-- scorecard (calibration and hit rate per role per month) is how the organisation shows it decides better over
-- time, not just more. Sandbox predictions resolve against simulated outcomes and are labelled so.
CREATE TABLE predictions (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    role         text NOT NULL,              -- talent_manager | strategist | evaluator | market_lead
    creator_id   uuid REFERENCES creators(id),
    market_id    uuid REFERENCES markets(id),
    claim        text NOT NULL,
    metric       text NOT NULL,              -- reach_growth | median_reach | follower_growth | ...
    comparator   text NOT NULL CHECK (comparator IN ('above', 'below')),
    threshold    numeric NOT NULL,
    probability  numeric NOT NULL CHECK (probability >= 0 AND probability <= 1),
    horizon_days int NOT NULL,
    mode         text NOT NULL CHECK (mode IN ('real', 'sandbox')),
    decision_id  uuid REFERENCES decisions(id),
    resolve_at   timestamptz NOT NULL,
    resolved_at  timestamptz,
    observed     numeric,
    outcome      boolean,                     -- the claim held
    brier        numeric,                     -- (probability - outcome)^2: lower is better
    created_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX predictions_due_idx ON predictions (resolve_at) WHERE resolved_at IS NULL;
CREATE INDEX predictions_role_idx ON predictions (role, created_at);
