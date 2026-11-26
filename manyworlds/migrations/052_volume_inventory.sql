-- What a weights volume held when each machine booted (owner 2026-11-25: prove that weights persist across machine
-- lifecycles and that no family deletes another's): {path under the mount: GB}, written by the worker at boot.
ALTER TABLE gpu_groups ADD COLUMN IF NOT EXISTS volume_inventory jsonb;
GRANT UPDATE (volume_inventory) ON gpu_groups TO mw_worker;
