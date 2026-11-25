-- Rented GPU hosts are untrusted (review 2026-11-25): the mw_worker grants of 022 let a host rewrite any job (payload,
-- kind, batch: queue brain work while paused with an uncapped budget), insert negative gpu_billed rows (defeat the
-- daily caps) and push its machine's deadline out. Now:
--   jobs          workers update only the columns claim/heartbeat/finish/release write, and see and change only GPU
--                 jobs (never brain.*);
--   cost_entries  workers insert only non-negative 'gpu' attribution rows;
--   gpu_groups    a worker may bring its deadline forward (self-stop), never push it out.
-- The control plane owns these tables, so row security does not apply to it; the permissive policy for everyone
-- else keeps any other non-owner role working as before.

REVOKE UPDATE ON jobs FROM mw_worker;
GRANT UPDATE (status, attempts, worker_id, lease_until, result, error, updated_at, run_after, worker_faults)
    ON jobs TO mw_worker;
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;
CREATE POLICY jobs_not_worker ON jobs FOR ALL TO PUBLIC
    USING (NOT pg_has_role(current_user, 'mw_worker', 'USAGE'))
    WITH CHECK (NOT pg_has_role(current_user, 'mw_worker', 'USAGE'));
CREATE POLICY jobs_worker_select ON jobs FOR SELECT TO mw_worker USING (kind NOT LIKE 'brain.%');
CREATE POLICY jobs_worker_update ON jobs FOR UPDATE TO mw_worker
    USING (kind NOT LIKE 'brain.%') WITH CHECK (kind NOT LIKE 'brain.%');

ALTER TABLE cost_entries ENABLE ROW LEVEL SECURITY;
CREATE POLICY cost_entries_not_worker ON cost_entries FOR ALL TO PUBLIC
    USING (NOT pg_has_role(current_user, 'mw_worker', 'USAGE'))
    WITH CHECK (NOT pg_has_role(current_user, 'mw_worker', 'USAGE'));
CREATE POLICY cost_entries_worker_insert ON cost_entries FOR INSERT TO mw_worker
    WITH CHECK (category = 'gpu' AND usd >= 0);

CREATE OR REPLACE FUNCTION gpu_groups_worker_deadline() RETURNS trigger AS $$
BEGIN
    IF pg_has_role(current_user, 'mw_worker', 'USAGE') AND NEW.deadline > OLD.deadline THEN
        RAISE EXCEPTION 'a worker may only bring its machine''s deadline forward';
    END IF;
    RETURN NEW;
END
$$ LANGUAGE plpgsql;
CREATE TRIGGER gpu_groups_worker_deadline BEFORE UPDATE OF deadline ON gpu_groups
    FOR EACH ROW EXECUTE FUNCTION gpu_groups_worker_deadline();
