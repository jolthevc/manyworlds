-- Serverless GPU requests (manyworlds.deploy.serverless): one per job sent to a RunPod Serverless endpoint; settled
-- into the ledger from the provider's execution time when it finishes.
CREATE TABLE serverless_requests (
    id           text PRIMARY KEY,
    endpoint     text NOT NULL,
    pool         text NOT NULL,
    kinds        text[] NOT NULL,
    status       text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    finished_at  timestamptz,
    execution_ms bigint,
    usd          numeric
);
CREATE INDEX serverless_requests_open ON serverless_requests (pool) WHERE finished_at IS NULL;
