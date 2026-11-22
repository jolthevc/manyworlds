-- Indexes for the queries that run every tick or every post, before they become full scans at scale
-- (2026-11-22 scale audit): reviews by asset (production and review read them constantly), posts by creator (feeds,
-- repetition checks, event detectors), the ledger by day (daily caps sum today's entries every tick), clip and
-- evaluator jobs by the idea they serve, and decisions by kind and time (once-a-day jobs check their last run).
CREATE INDEX IF NOT EXISTS reviews_asset_idx ON reviews (asset_id);
CREATE INDEX IF NOT EXISTS posts_creator_idx ON posts (creator_id, scheduled_for);
CREATE INDEX IF NOT EXISTS cost_entries_created_idx ON cost_entries (created_at);
CREATE INDEX IF NOT EXISTS jobs_idea_idx ON jobs ((payload->>'idea_id')) WHERE kind = 'brain.evaluator';
CREATE INDEX IF NOT EXISTS jobs_creator_idx ON jobs (creator_id, status);
CREATE INDEX IF NOT EXISTS decisions_kind_idx ON decisions (kind, created_at DESC);
CREATE INDEX IF NOT EXISTS ideas_batch_idx ON ideas (batch, status);
