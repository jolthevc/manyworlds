-- GPU machines are started and stopped by the autoscaler, on any provider.
-- provider_ref is the provider's own id for the machine (Vast contract id, RunPod pod id,
-- SaladCloud group name) so it can be stopped even if its name is ambiguous.
ALTER TABLE gpu_groups ADD COLUMN provider_ref text;
ALTER TABLE gpu_groups ADD COLUMN family text;

-- A job may be pinned to particular pools (benchmarks compare providers this way).
-- NULL means any pool that serves the kind.
ALTER TABLE jobs ADD COLUMN pools text[];
