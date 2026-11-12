-- Billed GPU time is metered per container group, independent of whether jobs ran:
-- an idle group still costs money. Job-level 'gpu' cost entries stay for attributing
-- cost to outputs; 'gpu_billed' entries are what the provider actually charges.
CREATE TABLE gpu_groups (
    name           text PRIMARY KEY,           -- provider group name, e.g. mw-image-4090-r3
    provider       text NOT NULL,              -- salad
    pool           text NOT NULL,              -- policy pool, e.g. salad_batch_4090
    usd_per_hour   numeric NOT NULL,
    replicas       int NOT NULL DEFAULT 1,
    job_kinds      text[] NOT NULL DEFAULT '{}',
    deadline       timestamptz NOT NULL,       -- hard stop, whatever happens
    max_usd        numeric NOT NULL,           -- hard stop on this group's billed spend
    started_at     timestamptz NOT NULL DEFAULT now(),
    billed_until   timestamptz NOT NULL DEFAULT now(),
    billed_usd     numeric NOT NULL DEFAULT 0,
    last_busy_at   timestamptz NOT NULL DEFAULT now(),
    stopped_at     timestamptz,
    stop_reason    text
);

-- A rerun of a job must overwrite its outputs, not add duplicates.
DELETE FROM assets a USING assets b
WHERE a.job_id = b.job_id AND a.uri = b.uri AND a.created_at > b.created_at;
CREATE UNIQUE INDEX assets_job_uri_uq ON assets (job_id, uri);
