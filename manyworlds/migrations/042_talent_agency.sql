-- The talent agency (docs/superpowers/specs/2026-11-25-talent-agency.md): casting with auditions, the commercial
-- category registry and account-level insights. None of these tables is granted to mw_worker.

-- Auditions (migration 041) carry the casting run, its launch (when a launch cast it) and a stage; arms go to E
-- (3-5 candidates a brief; the top ones are auditioned).
ALTER TABLE auditions DROP CONSTRAINT auditions_arm_check;
ALTER TABLE auditions ADD CONSTRAINT auditions_arm_check CHECK (arm IN ('A', 'B', 'C', 'D', 'E'));
ALTER TABLE auditions ADD COLUMN run text;
ALTER TABLE auditions ADD COLUMN launch_id uuid REFERENCES launches(id);
ALTER TABLE auditions ADD COLUMN stage text NOT NULL DEFAULT 'shortlisted'
    CHECK (stage IN ('shortlisted', 'not_shortlisted', 'bible', 'references', 'anchors', 'feed', 'judged', 'done'));
CREATE INDEX auditions_run_idx ON auditions (run);

-- Consumer categories a creator can be commercially valuable in (manyworlds.brain.commerce). Seeded from policy
-- (strategy.commercial.seed) as source 'estimate'; the intelligence agent's `commerce` mode replaces fields with
-- measured ones (source 'research') and records what it saw in evidence. Bands are words (policy band_values).
CREATE TABLE commercial_categories (
    key                  text PRIMARY KEY,
    name                 text NOT NULL,
    market_size          text,       -- small | medium | large | very_large
    advertiser_density   text,       -- low | medium | high
    influencer_activity  text,       -- low | medium | high
    virtual_precedent    text,       -- none | some | strong
    production_fit       text,       -- hard | maybe | good | excellent
    affiliate_fit        text,       -- low | medium | high
    licensing_fit        text,       -- low | medium | high
    hashtags             jsonb NOT NULL DEFAULT '[]',
    notes                text,
    source               text NOT NULL DEFAULT 'estimate' CHECK (source IN ('estimate', 'research')),
    evidence             jsonb NOT NULL DEFAULT '{}',
    refreshed_at         timestamptz NOT NULL DEFAULT now(),
    created_at           timestamptz NOT NULL DEFAULT now()
);

-- Account-level Instagram insights, one row per creator and day (publish/metrics.collect_account): follower count,
-- reach split by follower / non-follower, profile views, follows. Real accounts only; nothing simulated is written here.
CREATE TABLE account_insights (
    creator_id  uuid NOT NULL REFERENCES creators(id),
    day         date NOT NULL,
    metrics     jsonb NOT NULL DEFAULT '{}',
    captured_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (creator_id, day)
);
