-- The learning harness (docs/2026-11-26-learning-harness.md section 4.3-4.4): findings and pattern cards are
-- hypotheses, not permanent beliefs. Each records where it came from (origin: post, creator, research, experiment),
-- the contexts it applies to (applies_to: archetype, vertical, audience, model tags), the level it holds at (post,
-- character, archetype, vertical, portfolio, production), the creators where it held (replications) or failed
-- (contradictions), when it was last tested and when it is due for a retest (NULL: policy brain.hypotheses.retest_days
-- from last_tested). parent_id: the finding an archetype-level hypothesis was opened from (manyworlds.brain.hypotheses).
-- Columns are added to findings and simulated_findings in the same order, so `SELECT *` from both still lines up.
ALTER TABLE findings ADD COLUMN origin text CHECK (origin IN ('post', 'creator', 'research', 'experiment'));
ALTER TABLE findings ADD COLUMN applies_to jsonb NOT NULL DEFAULT '{}';
ALTER TABLE findings ADD COLUMN level text
    CHECK (level IN ('post', 'character', 'archetype', 'vertical', 'portfolio', 'production'));
ALTER TABLE findings ADD COLUMN replications jsonb NOT NULL DEFAULT '[]';
ALTER TABLE findings ADD COLUMN contradictions jsonb NOT NULL DEFAULT '[]';
ALTER TABLE findings ADD COLUMN last_tested timestamptz;
ALTER TABLE findings ADD COLUMN retest_after timestamptz;
ALTER TABLE findings ADD COLUMN parent_id uuid;

ALTER TABLE simulated_findings ADD COLUMN origin text CHECK (origin IN ('post', 'creator', 'research', 'experiment'));
ALTER TABLE simulated_findings ADD COLUMN applies_to jsonb NOT NULL DEFAULT '{}';
ALTER TABLE simulated_findings ADD COLUMN level text
    CHECK (level IN ('post', 'character', 'archetype', 'vertical', 'portfolio', 'production'));
ALTER TABLE simulated_findings ADD COLUMN replications jsonb NOT NULL DEFAULT '[]';
ALTER TABLE simulated_findings ADD COLUMN contradictions jsonb NOT NULL DEFAULT '[]';
ALTER TABLE simulated_findings ADD COLUMN last_tested timestamptz;
ALTER TABLE simulated_findings ADD COLUMN retest_after timestamptz;
ALTER TABLE simulated_findings ADD COLUMN parent_id uuid;

-- An archetype-level hypothesis has its own scope (no creator, cluster or market of its own).
DO $$
DECLARE c record;
BEGIN
    FOR c IN SELECT conrelid::regclass AS t, conname FROM pg_constraint
             WHERE conrelid IN ('findings'::regclass, 'simulated_findings'::regclass) AND contype = 'c'
               AND pg_get_constraintdef(oid) LIKE '%scope%'
    LOOP
        EXECUTE format('ALTER TABLE %s DROP CONSTRAINT %I', c.t, c.conname);
    END LOOP;
END $$;
ALTER TABLE findings ADD CONSTRAINT findings_scope_check
    CHECK (scope IN ('post', 'creator', 'cluster', 'market', 'portfolio', 'genre', 'archetype'));
ALTER TABLE simulated_findings ADD CONSTRAINT simulated_findings_scope_check
    CHECK (scope IN ('post', 'creator', 'cluster', 'market', 'portfolio', 'genre', 'archetype'));

-- Existing rows: origin and level from what they already say; last tested when they were last updated.
UPDATE findings SET
    origin = CASE WHEN evidence = 'experimental' THEN 'experiment' WHEN evidence = 'research' THEN 'research'
                  WHEN scope = 'post' THEN 'post' ELSE 'creator' END,
    level = CASE scope WHEN 'post' THEN 'post' WHEN 'creator' THEN 'character' WHEN 'portfolio' THEN 'portfolio'
                       ELSE 'vertical' END,
    last_tested = updated_at;
UPDATE simulated_findings SET
    origin = CASE WHEN sources::text LIKE '%experiment_id%'
                  THEN 'experiment' WHEN scope = 'post' THEN 'post' ELSE 'creator' END,
    level = CASE scope WHEN 'post' THEN 'post' WHEN 'creator' THEN 'character' WHEN 'portfolio' THEN 'portfolio'
                       ELSE 'vertical' END,
    last_tested = updated_at;
CREATE INDEX findings_level_idx ON findings (level) WHERE status = 'open';

-- Pattern cards carry the same hypothesis record; feedback (manyworlds.brain.patterns.feedback) fills replications,
-- contradictions and last_tested from our own posts' outcomes.
ALTER TABLE pattern_cards ADD COLUMN origin text CHECK (origin IN ('post', 'creator', 'research', 'experiment'));
ALTER TABLE pattern_cards ADD COLUMN applies_to jsonb NOT NULL DEFAULT '{}';
ALTER TABLE pattern_cards ADD COLUMN level text
    CHECK (level IN ('post', 'character', 'archetype', 'vertical', 'portfolio', 'production'));
ALTER TABLE pattern_cards ADD COLUMN replications jsonb NOT NULL DEFAULT '[]';
ALTER TABLE pattern_cards ADD COLUMN contradictions jsonb NOT NULL DEFAULT '[]';
ALTER TABLE pattern_cards ADD COLUMN last_tested timestamptz;
ALTER TABLE pattern_cards ADD COLUMN retest_after timestamptz;
UPDATE pattern_cards SET origin = 'research', level = 'archetype';
