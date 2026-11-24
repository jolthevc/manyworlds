-- A weights volume's free space, measured by the worker at boot (manyworlds.worker.volume): the autoscaler stops
-- routing work to a volume's pools while its latest check is below the floor (gpu_pools.<pool>.network_volume
-- min_free_gb), and machines that gave up for that reason do not count toward the no-output guard or the relaunch
-- cooldown. The worker role writes these columns on its own machine's row only, as it does its deadline.
ALTER TABLE gpu_groups ADD COLUMN volume_free_gb numeric;
ALTER TABLE gpu_groups ADD COLUMN volume_low boolean;
ALTER TABLE gpu_groups ADD COLUMN volume_checked_at timestamptz;
GRANT UPDATE (volume_free_gb, volume_low, volume_checked_at) ON gpu_groups TO mw_worker;
