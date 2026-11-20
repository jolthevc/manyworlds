-- Experiments carry the evidence mode they ran in: a sandbox creator's experiments never continue, and never
-- block a retest, once the creator goes live (review of Brain M6, 2026-11-20).
ALTER TABLE experiments ADD COLUMN mode text NOT NULL DEFAULT 'real' CHECK (mode IN ('real', 'sandbox'));
DROP INDEX experiments_creator_effect;
CREATE UNIQUE INDEX experiments_creator_effect_mode ON experiments (creator_id, effect_key, mode) WHERE effect_key IS NOT NULL;
