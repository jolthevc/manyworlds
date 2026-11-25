-- The factory's clock (docs/research/2026-11-25-gpu-factory-audit.md): when a machine's worker came up, so a boot splits
-- into provisioning (provider start, image pull: started_at to worker_ready_at) and first-job warm-up (weights download
-- and model load: worker_ready_at to the first attempt). The worker role writes it on its own machine's row only, as it
-- does its deadline and the volume reading.
ALTER TABLE gpu_groups ADD COLUMN worker_ready_at timestamptz;
GRANT UPDATE (worker_ready_at) ON gpu_groups TO mw_worker;
