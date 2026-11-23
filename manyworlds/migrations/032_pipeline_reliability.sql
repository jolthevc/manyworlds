-- Pipeline reliability (2026-11-23 review).
--
-- run_after: a retried job waits attempts x queue.retry_backoff_seconds before it can be claimed again
-- (manyworlds.queue.fail), instead of bouncing straight back onto the next free worker.
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS run_after timestamptz;
-- review_errors / review_error_at: reviews of this job's output that raised (manyworlds.review.common.give_up). Past
-- review.<image|video>.max_review_errors its outputs count as rejected, so production repairs the post instead of
-- the reviewer paying a screen for it every loop forever.
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS review_errors int NOT NULL DEFAULT 0;
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS review_error_at timestamptz;

-- Assets are looked up by uri alone (production's joins, voice lines, the finished video); the unique
-- (job_id, uri) index can't serve that.
CREATE INDEX IF NOT EXISTS assets_uri_idx ON assets (uri);

-- The claim query only ever looks at open jobs; done/failed/dead rows pile up forever. A partial index keeps the
-- claim's scan to the open ones. jobs_claim_idx stays (nothing is dropped).
CREATE INDEX IF NOT EXISTS jobs_open_claim_idx ON jobs (kind, priority DESC, created_at)
    WHERE status IN ('queued', 'running');
