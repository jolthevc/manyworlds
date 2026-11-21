-- Rented GPU machines (consumer hosts we don't control) connect as a member of mw_worker: they can
-- claim, heartbeat and finish jobs, record attempts, assets and their GPU cost, and read their own
-- machine's deadline. Nothing else: no research, creators, decisions, accounts or ledger reads.
-- The login role itself (with its password) is made by `mw worker-credentials`, never in a migration.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mw_worker') THEN
        CREATE ROLE mw_worker NOLOGIN;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO mw_worker;
GRANT SELECT, UPDATE ON jobs TO mw_worker;
GRANT SELECT, INSERT, UPDATE ON attempts, assets TO mw_worker;
GRANT INSERT ON cost_entries TO mw_worker;
GRANT USAGE ON SEQUENCE cost_entries_id_seq TO mw_worker;
GRANT SELECT ON gpu_groups TO mw_worker;
GRANT UPDATE (deadline) ON gpu_groups TO mw_worker;
