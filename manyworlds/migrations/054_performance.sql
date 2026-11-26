-- Performance direction (manyworlds.performance; docs/2026-11-26-performance-direction.md).
--
-- performance_observations: per judged asset and review stage, what was requested of the performance (the shot's
-- control levels, the levels compiled after calibration, the intent and phrases used) and what the judge observed
-- (levels for the directed controls; appeal, charisma, expression, naturalness 1-10; a pose/expression descriptor for
-- variety), the closeness and the selection points. Rank only: nothing here rejects. `model` is the generation job's
-- kind (image.zimage, comfy.wan_i2v, talk.infinitetalk...), `judge` the review model. The job's all-in cost joins
-- through job_id (cost_entries). Written by the reviewer on the control plane; rented machines (mw_worker) never
-- read or write it, so nothing is granted to that role.
CREATE TABLE performance_observations (
    id              bigserial PRIMARY KEY,
    asset_id        uuid NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    post_id         uuid REFERENCES posts(id) ON DELETE SET NULL,
    creator_id      uuid REFERENCES creators(id) ON DELETE CASCADE,
    job_id          uuid REFERENCES jobs(id) ON DELETE SET NULL,
    model           text,
    judge           text,
    stage           text NOT NULL DEFAULT 'screen',
    target          text,
    intent          text NOT NULL DEFAULT '',
    requested       jsonb NOT NULL DEFAULT '{}',     -- {control: level} as the shot asked
    compiled        jsonb NOT NULL DEFAULT '{}',     -- {control: level} as the prompt said it (calibration applied)
    observed        jsonb NOT NULL DEFAULT '{}',     -- {control: level} as the judge saw it
    scores          jsonb NOT NULL DEFAULT '{}',     -- {appeal, charisma, expression, naturalness}
    descriptor      text NOT NULL DEFAULT '',
    phrases         jsonb NOT NULL DEFAULT '{}',     -- {channel: [phrase]}
    closeness       numeric,
    points          numeric,
    policy_version  text,
    library_version text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (asset_id, stage)
);
CREATE INDEX performance_observations_cell_idx ON performance_observations (model, target);
CREATE INDEX performance_observations_creator_idx ON performance_observations (creator_id, created_at);
CREATE INDEX performance_observations_post_idx ON performance_observations (post_id);

-- The calibration table: per (model, target, control, requested level) the level offset the compiler applies (a model
-- that undershoots is asked one level higher), changed only by manyworlds.performance.refresh_calibration from
-- reliable observations, each change a decision row (kind performance_calibration).
CREATE TABLE performance_calibration (
    model        text NOT NULL,
    target       text NOT NULL,
    control      text NOT NULL,
    level        int NOT NULL CHECK (level BETWEEN 0 AND 4),
    level_offset int NOT NULL DEFAULT 0,
    judge        text,
    stats        jsonb NOT NULL DEFAULT '{}',
    updated_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (model, target, control, level)
);

-- Our own generated content, annotated with the research engine's vocabulary (research.annotation, schema iia_v2) in
-- the same store as the reference posts, so research, direction, achieved characteristics, cost and outcomes join
-- (manyworlds.review.own_annotation; evidence.own_annotations). source 'own' rows carry our post and asset instead of
-- a reference post; every existing reader joins reference_posts on post_id, so it never sees them.
ALTER TABLE reference_annotations ADD COLUMN source text NOT NULL DEFAULT 'external'
    CHECK (source IN ('external', 'own'));
ALTER TABLE reference_annotations ADD COLUMN own_post_id uuid REFERENCES posts(id) ON DELETE CASCADE;
ALTER TABLE reference_annotations ADD COLUMN own_asset_id uuid REFERENCES assets(id) ON DELETE CASCADE;
ALTER TABLE reference_annotations ALTER COLUMN post_id DROP NOT NULL;
ALTER TABLE reference_annotations ADD CONSTRAINT reference_annotations_source_check2 CHECK (
    (source = 'external' AND post_id IS NOT NULL) OR (source = 'own' AND post_id IS NULL AND own_asset_id IS NOT NULL));
CREATE UNIQUE INDEX reference_annotations_own_idx ON reference_annotations (own_asset_id, schema) WHERE source = 'own';
