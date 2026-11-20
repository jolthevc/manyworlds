-- Experiments (Brain M6): which analysis effect each one tests (one experiment per creator and effect, ever),
-- the finding that prompted it (in findings or simulated_findings, so no foreign key), and when it ended.
ALTER TABLE experiments ADD COLUMN effect_key text;
ALTER TABLE experiments ADD COLUMN source_finding uuid;
ALTER TABLE experiments ADD COLUMN concluded_at timestamptz;
CREATE UNIQUE INDEX experiments_creator_effect ON experiments (creator_id, effect_key) WHERE effect_key IS NOT NULL;
