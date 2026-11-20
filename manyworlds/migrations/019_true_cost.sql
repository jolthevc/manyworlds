-- A machine's true cost (2026-11-20: Vast billed $40.40 where the ledger metered $4.27; $29.11 was download
-- bandwidth). est_extra_usd: the expected non-hourly charges (downloading the image and weights) known at launch,
-- counted against the spending cap at once. actual_usd: the provider's itemised total once reconciled.
-- cost_usd: what reports and pool scoring use.
ALTER TABLE gpu_groups ADD COLUMN est_extra_usd numeric NOT NULL DEFAULT 0;
ALTER TABLE gpu_groups ADD COLUMN actual_usd numeric;
ALTER TABLE gpu_groups ADD COLUMN reconciled_usd numeric NOT NULL DEFAULT 0;   -- ledger adjustments written so far
ALTER TABLE gpu_groups ADD COLUMN cost_usd numeric
    GENERATED ALWAYS AS (coalesce(actual_usd, billed_usd + est_extra_usd)) STORED;

-- The provider's own itemised charges, per instance, day and component (gpu, disk, bwd, bwu).
CREATE TABLE provider_charges (
    provider   text NOT NULL,
    ref        text NOT NULL,
    group_name text,
    day        date NOT NULL,
    component  text NOT NULL,
    usd        numeric NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (provider, ref, day, component)
);
CREATE INDEX provider_charges_group_idx ON provider_charges (group_name);
