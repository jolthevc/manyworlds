-- The analytical spine (docs/2026-11-26-unified-intelligence.md §4, owner approval 2026-11-26): one shape for any
-- content, ours or external, any format, at any level, so research, production and outcomes share one foundation.
--
-- content_units   a unit of content at a level: creator, series, post, segment, shot, beat. parent_id nests them
--                 (a shot inside a segment inside a post inside a series of a creator); seq and start_s/end_s order
--                 them in time. source external|own; platform, format, medium and vertical are open text, checked
--                 against the policy registry (intelligence.spine) in code, never a CHECK, so a new media business
--                 (YouTube, ads, music video, micro-drama) needs no migration. context: archetype, audience
--                 proposition, commercial objective... ref: the row it mirrors (reference_posts, posts, assets).
-- attribute_values  long form: one value per unit, attribute, method and annotator. method code|model|human|
--                 reference; purpose production|gold|repeat|spot_check. The attribute registry (policy
--                 intelligence.attributes) gives each attribute its tier (universal|format|context|production) and its
--                 class (objective|subjective|inferred). An evolving vocabulary adds rows, never columns.
-- outcomes        raw metrics kept as observed, at a horizon (age_s at observation, horizon bucket), with the
--                 normalised measure beside them (log of raw over the creator's baseline at that horizon) and the
--                 baseline it was computed from, so any outcome (reach, watch time, shares, saves, follows,
--                 retention, revenue, cost) can be analysed and the normalisation re-derived.
-- claim_relations typed links between claims (findings, hypotheses, cards, research questions, desk findings) and
--                 what they apply to, derive from, replicate, contradict, transfer to or informed.
-- analysis_runs   the trial ledger: every analysis an agent or code runs, its spec, method version, data snapshot,
--                 family size and result, so multiple testing counts every trial, dropped ones too.

CREATE TABLE content_units (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source      text NOT NULL CHECK (source IN ('external', 'own')),
    level       text NOT NULL CHECK (level IN ('creator', 'series', 'post', 'segment', 'shot', 'beat')),
    parent_id   uuid REFERENCES content_units(id) ON DELETE CASCADE,
    ref_table   text,
    ref_id      text,
    platform    text,
    format      text,
    medium      text,
    vertical    text,
    seq         int,
    start_s     numeric,
    end_s       numeric,
    published_at timestamptz,
    context     jsonb NOT NULL DEFAULT '{}',
    rights      text NOT NULL DEFAULT 'green' CHECK (rights IN ('green', 'amber')),
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX content_units_ref ON content_units (ref_table, ref_id, level, coalesce(seq, -1))
    WHERE ref_table IS NOT NULL;
CREATE INDEX content_units_parent ON content_units (parent_id);
CREATE INDEX content_units_kind ON content_units (source, level, format);

CREATE TABLE attribute_values (
    id            bigserial PRIMARY KEY,
    unit_id       uuid NOT NULL REFERENCES content_units(id) ON DELETE CASCADE,
    attribute     text NOT NULL,
    value         text,
    num           numeric,
    confidence    numeric,
    method        text NOT NULL CHECK (method IN ('code', 'model', 'human', 'reference')),
    annotator     text NOT NULL,
    purpose       text NOT NULL DEFAULT 'production' CHECK (purpose IN ('production', 'gold', 'repeat', 'spot_check')),
    vocab_version text,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX attribute_values_one ON attribute_values (unit_id, attribute, method, annotator, purpose,
                                                              coalesce(value, ''));
CREATE INDEX attribute_values_attr ON attribute_values (attribute, value);

CREATE TABLE outcomes (
    id              bigserial PRIMARY KEY,
    unit_id         uuid NOT NULL REFERENCES content_units(id) ON DELETE CASCADE,
    metric          text NOT NULL,
    horizon         text NOT NULL,
    age_s           numeric,
    raw             numeric,
    normalized      numeric,
    baseline        numeric,
    baseline_n      int,
    baseline_window text,
    followers_at    bigint,
    observed_at     timestamptz,
    source          text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (unit_id, metric, horizon)
);
CREATE INDEX outcomes_metric ON outcomes (metric, horizon);

CREATE TABLE claim_relations (
    id          bigserial PRIMARY KEY,
    claim_kind  text NOT NULL,
    claim_id    text NOT NULL,
    relation    text NOT NULL CHECK (relation IN ('applies_to', 'derived_from', 'replicates', 'contradicts',
                                                  'transfers_to', 'informed', 'tested_by')),
    target_kind text NOT NULL,
    target_id   text NOT NULL,
    evidence    jsonb NOT NULL DEFAULT '{}',
    confidence  numeric,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (claim_kind, claim_id, relation, target_kind, target_id)
);
CREATE INDEX claim_relations_target ON claim_relations (target_kind, target_id);

CREATE TABLE analysis_runs (
    id             bigserial PRIMARY KEY,
    requester      text NOT NULL,
    question_id    bigint,
    method         text NOT NULL,
    method_version text NOT NULL,
    spec           jsonb NOT NULL,
    family         text,
    family_size    int NOT NULL DEFAULT 1,
    data           jsonb NOT NULL DEFAULT '{}',
    result         jsonb NOT NULL DEFAULT '{}',
    policy_version text,
    created_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX analysis_runs_family ON analysis_runs (family, created_at);

-- Open the format and platform limits: checked against the policy registry in code (intelligence.spine).
ALTER TABLE reference_creators DROP CONSTRAINT IF EXISTS reference_creators_platform_check;
ALTER TABLE reference_posts DROP CONSTRAINT IF EXISTS reference_posts_media_type_check;
ALTER TABLE pattern_cards DROP CONSTRAINT IF EXISTS pattern_cards_medium_check;
-- Learning levels: asset, audience, format and cross-format beside the existing ones.
ALTER TABLE findings DROP CONSTRAINT IF EXISTS findings_level_check;
ALTER TABLE findings ADD CONSTRAINT findings_level_check CHECK (level IN
    ('asset', 'post', 'character', 'archetype', 'audience', 'format', 'cross_format', 'vertical', 'portfolio',
     'production'));
ALTER TABLE simulated_findings DROP CONSTRAINT IF EXISTS simulated_findings_level_check;
ALTER TABLE simulated_findings ADD CONSTRAINT simulated_findings_level_check CHECK (level IN
    ('asset', 'post', 'character', 'archetype', 'audience', 'format', 'cross_format', 'vertical', 'portfolio',
     'production'));
