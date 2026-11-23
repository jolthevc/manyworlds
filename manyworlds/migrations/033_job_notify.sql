-- Event-driven wakeups (manyworlds.wakeup): the producer and reviewer wait on these instead of only sleeping.
--
-- 'mw_jobs': a job is inserted, or its status changes to done, failed or dead. Payload: {id, kind, status, post_id}
-- (well under NOTIFY's 8000-byte limit). Workers update jobs as mw_worker (migration 022); the functions run as the
-- caller (SECURITY INVOKER, the default) and pg_notify needs no grant, so the worker role needs nothing new.
-- Notifications are sent at commit and dropped on rollback.
CREATE OR REPLACE FUNCTION mw_notify_job() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'INSERT'
       OR (NEW.status IS DISTINCT FROM OLD.status AND NEW.status IN ('done', 'failed', 'dead')) THEN
        PERFORM pg_notify('mw_jobs', json_build_object(
            'id', NEW.id, 'kind', NEW.kind, 'status', NEW.status, 'post_id', NEW.post_id)::text);
    END IF;
    RETURN NULL;
END
$$;

DROP TRIGGER IF EXISTS jobs_notify ON jobs;
CREATE TRIGGER jobs_notify AFTER INSERT OR UPDATE OF status ON jobs
    FOR EACH ROW EXECUTE FUNCTION mw_notify_job();

-- 'mw_reviews': reviews were written (one statement-level notification; identical payloads within a transaction are
-- folded into one by Postgres). The producer moves a post on once its outputs are reviewed.
CREATE OR REPLACE FUNCTION mw_notify_reviews() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM pg_notify('mw_reviews', '{}');
    RETURN NULL;
END
$$;

DROP TRIGGER IF EXISTS reviews_notify ON reviews;
CREATE TRIGGER reviews_notify AFTER INSERT ON reviews
    FOR EACH STATEMENT EXECUTE FUNCTION mw_notify_reviews();
