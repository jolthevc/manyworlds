-- Launches (Brain M8): one row per creator being brought up by the creator lab, from an approved launch to a
-- creator ready to plan. `jobs` holds the job ids of each step; `retries` counts failed attempts of the step.
CREATE TABLE launches (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    approval_id uuid UNIQUE REFERENCES approvals(id),
    thesis_id   uuid REFERENCES theses(id),
    creator_id  uuid REFERENCES creators(id),
    cohort      text,
    step        text NOT NULL DEFAULT 'bible',
    status      text NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'done', 'failed')),
    jobs        jsonb NOT NULL DEFAULT '{}',
    retries     int NOT NULL DEFAULT 0,
    error       text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX launches_status_idx ON launches (status);
